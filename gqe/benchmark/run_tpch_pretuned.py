#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe import Catalog
from gqe.benchmark.gqe_experiment import GqeExperimentConnection
from gqe.benchmark.run import (
    run_tpc,
    run_tpc_multiprocess,
    DataInfo,
    QueryInfo,
    Parameter,
    setup_db,
    parse_scale_factor,
    set_eager_module_loading,
    is_valid_identifier_type,
    fix_partial_filter_column_references,
    get_query_validator,
)
from gqe import lib


from database_benchmarking_tools.experiment import ExperimentDB
from database_benchmarking_tools.utility import generate_db_path

import argparse
import importlib
import os
import sqlite3


def get_best_parameters_file(sqlite_file: str):
    conn = sqlite3.connect(sqlite_file)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM gqe_best_parameters")
    best_parameters = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return best_parameters


def get_best_parameters_folder(df_folder: str):
    best_param_dict = {}
    for df_file in os.listdir(df_folder):
        db_path = os.path.join(df_folder, df_file)
        if not os.path.isfile(db_path):
            continue
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("SELECT * FROM gqe_best_parameters")
            rows = cursor.fetchall()
            for row in rows:
                e_name = row["e_name"]
                r_avg_duration_s = row["r_avg_duration_s"]
                # If this e_name is not seen yet, or this row is better, keep it
                if (
                    e_name not in best_param_dict
                    or r_avg_duration_s < best_param_dict[e_name]["r_avg_duration_s"]
                ):
                    best_param_dict[e_name] = dict(row)
        except Exception as e:
            print(f"Skipping {db_path} because of error getting best parameters: {e}")
            pass
        finally:
            conn.close()

    return sorted(best_param_dict.values(), key=lambda x: int(x["e_name"].lstrip("Q")))


def log_physical_plan(query_str: str, relation: lib.Relation, folder_path: str):
    file_path = os.path.join(folder_path, query_str + "_plan.json")
    lib.log_physical_plan(relation, file_path)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset", help="TPC-H dataset location")
    arg_parser.add_argument("plan", help="Substrait query plan location")
    arg_parser.add_argument("solution", help="Reference results location with pattern")
    arg_parser.add_argument(
        "--swept-sqlite-file",
        help="SQLite file that has the parameter sweep results",
        type=str,
        default=None,
    )
    arg_parser.add_argument(
        "--swept-sqlite-folder", help="Folder that has the parameter sweep results"
    )
    arg_parser.add_argument("--output", "-o", help="Output file path")
    arg_parser.add_argument(
        "--output-physical-plan", help="Output file folder of physical plans"
    )
    arg_parser.add_argument(
        "--queries",
        "-q",
        help="Which queries to run",
        nargs="+",
        action="extend",
        type=str,
    )
    arg_parser.add_argument(
        "--load-all-data",
        "-l",
        help="Whether to load all data or the data required for each query. Defaults to 0",
        type=int,
        choices=[0, 1],
        default=0,
    )
    arg_parser.add_argument(
        "--multiprocess", "-m", help="Run in multiprocess mode", action="store_true"
    )
    args = arg_parser.parse_args()
    # TODO: add --nsys-trace to collect nsys traces for the best parameters

    gqe_host = "localhost"
    scale_factor = parse_scale_factor(args.dataset)

    set_eager_module_loading()

    physical_plan_folder = (
        args.output_physical_plan if args.output_physical_plan else None
    )

    errors = []

    if args.swept_sqlite_file:
        best_parameters = get_best_parameters_file(args.swept_sqlite_file)
    elif args.swept_sqlite_folder:
        best_parameters = get_best_parameters_folder(args.swept_sqlite_folder)
    else:
        raise ValueError(
            "Either --swept_sqlite_file or --swept_sqlite_folder must be specified"
        )

    # TODO: Multiprocess mode needs to check if spawned ranks is equal to that in the best parameters
    # https://gitlab-master.nvidia.com/haog/gqe-python/-/issues/13
    if args.multiprocess:
        # Validate that all runs use parquet_file storage before initializing MPI
        for bp in best_parameters:
            if bp.get("d_storage_device_kind") != "parquet_file":
                raise ValueError(
                    "Multiprocess mode is only supported with parquet_file storage kind"
                )
        lib.mpi_init()

    is_root_rank = (not args.multiprocess) or (lib.mpi_rank() == 0)
    edb_file = None
    edb_config = None
    if is_root_rank:
        edb_file = (
            args.output if args.output else generate_db_path(f"gqe", "tpch", gqe_host)
        )
        edb_config = ExperimentDB(edb_file, gqe_host).set_connection_type(
            GqeExperimentConnection
        )
        print(f"Writing SQLite file to {edb_file}")

    if physical_plan_folder and is_root_rank:
        print(f"Writing Physical Plan to the folder {physical_plan_folder}")

    def run_all(edb, edb_info):
        errors_local = []
        for best_parameter in best_parameters:
            query_str = best_parameter["e_name"].lstrip("Q")
            query_idx = int(query_str.split("_")[0])
            if args.queries and query_str not in args.queries:
                print(
                    f"Skipping {best_parameter['e_name']} because it is not in the list of queries to run"
                )
                continue

            num_row_groups = best_parameter["de_num_row_groups"]
            char_type = best_parameter["d_char_type"]
            use_opt_type_for_single_char_col = char_type == "char"
            compression_format = best_parameter["de_compression_format"]
            compression_data_type = best_parameter["de_compression_data_type"]
            compression_chunk_size = best_parameter["de_compression_chunk_size"]
            zone_map_partition_size = best_parameter["de_zone_map_partition_size"]

            str_to_type = {
                "TypeId.int32": lib.TypeId.int32,
                "TypeId.int64": lib.TypeId.int64,
            }
            identifier_type = str_to_type[best_parameter["d_identifier_type"]]

            storage_kind = best_parameter["d_storage_device_kind"]
            query_source = best_parameter["e_query_source"]
            gqe_parameter = Parameter(
                best_parameter["p_num_workers"],
                best_parameter["p_num_partitions"],
                best_parameter["p_use_overlap_mtx"],
                best_parameter["p_join_use_hash_map_cache"],
                best_parameter["p_read_use_zero_copy"],
                best_parameter["p_join_use_unique_keys"],
                best_parameter["p_join_use_perfect_hash"],
                best_parameter["p_join_use_mark_join"],
                best_parameter["p_use_partition_pruning"],
                best_parameter["p_filter_use_like_shift_and"],
                best_parameter["p_aggregation_use_perfect_hash"],
            )

            if best_parameter["e_scale_factor"] != scale_factor:
                print(
                    f"Skipping {best_parameter['e_name']} because scale factor {best_parameter['e_scale_factor']} does not match input scale factor {scale_factor}"
                )
                continue

            data_info = DataInfo(
                storage_device_kind=storage_kind,
                format="internal",
                location=None,  # FIXME: set location as NUMA node, iff set in GQE
                not_null=False,
                identifier_type=str(identifier_type),
                char_type=char_type,
                decimal_type="float",
                num_row_groups=num_row_groups,
                compression_format=compression_format,
                compression_data_type=compression_data_type,
                compression_chunk_size=compression_chunk_size,
                zone_map_partition_size=zone_map_partition_size,
            )

            table_definitions = None
            catalog = Catalog()
            try:
                table_definitions = catalog.register_tpch(
                    args.dataset,
                    storage_kind,
                    num_row_groups,
                    0 if args.load_all_data else query_idx,
                    identifier_type,
                    use_opt_type_for_single_char_col,
                    compression_format,
                    compression_data_type,
                    compression_chunk_size,
                    zone_map_partition_size,
                )

            except Exception as e:
                print(f"Error registering in memory table for query {query_idx}: {e}")
                continue

            reference_file = args.solution.replace("%d", f"q{query_idx}")

            if query_source == "handcoded":
                query_identifier = "tpch_q" + query_str
                module = importlib.import_module(f"gqe.benchmark.{query_identifier}")
                query_object = getattr(module, query_identifier)(
                    scale_factor=scale_factor
                )
                root_relation = query_object.root_relation(table_definitions)
                query = QueryInfo(f"Q{query_str}", root_relation, reference_file)
                if validator := get_query_validator(query_object):
                    query.validator = validator
                if not args.load_all_data and (storage_kind != "parquet_file"):
                    fix_partial_filter_column_references(root_relation, query_idx)
                # make sure we call log_physical_plan after calling fix_partial_filter_column_references
                # so that root_relation is properly updated with column references
                if physical_plan_folder and is_root_rank:
                    root_relation.log_physical_plan(
                        f"Q{query_str}", physical_plan_folder
                    )

            elif query_source == "substrait":
                substrait_file = os.path.join(args.plan, f"df_q{query_idx}.bin")
                root_relation = catalog.load_substrait(substrait_file)
                if physical_plan_folder and is_root_rank:
                    log_physical_plan(
                        f"Q{query_idx}", root_relation, physical_plan_folder
                    )
                query = QueryInfo(f"Q{query_idx}", root_relation, reference_file)

            else:
                raise ValueError(f"Invalid query source: {query_source}")

            if args.multiprocess:
                run_tpc_multiprocess(
                    catalog,
                    data_info,
                    query,
                    scale_factor,
                    [gqe_parameter],
                    edb,
                    edb_info,
                    errors_local,
                    query_source,
                    is_root_rank,
                )
            else:
                run_tpc(
                    catalog,
                    data_info,
                    query,
                    scale_factor,
                    [gqe_parameter],
                    edb,
                    edb_info,
                    errors_local,
                    query_source,
                )

        return errors_local

    if is_root_rank:
        with edb_config as edb:
            edb_info = setup_db(edb)
            errors = run_all(edb, edb_info)

        print(f"Finished SQLite file at {edb_file}")
        if physical_plan_folder:
            print(f"Finished Physical Plan at the folder {physical_plan_folder}")
        
        if errors:
            print(
            "The following configurations run successfully but produce incorrect results"
        )
        print(errors)
    else:
        errors = run_all(None, None) 

    if args.multiprocess:
        lib.mpi_finalize()


if __name__ == "__main__":
    main()
