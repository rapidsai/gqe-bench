# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe import Catalog, execute
import argparse
import importlib


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("location", help="TPC-H dataset location")
    args = arg_parser.parse_args()

    catalog = Catalog()
    catalog.register_tpch(args.location, "memory")

    query_identifiers = ["tpch_q6", "tpch_q15", "tpch_q17", "tpch_q18", "tpch_q21"]
    for query_identifier in query_identifiers:
        print(f"Running {query_identifier}...")
        module = importlib.import_module(query_identifier)
        query = getattr(module, query_identifier)()
        execute(catalog, query.root_relation(), f"{query_identifier}_out.parquet")


if __name__ == "__main__":
    main()
