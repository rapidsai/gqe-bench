#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import argparse
import math
import os
import tempfile
import uuid

import duckdb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TPC-H data in Parquet format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--tmp", default="", type=str, help="Path for temporary DB on disk")
    parser.add_argument("-o", "--output", default="./data", type=str, help="Output path")
    parser.add_argument("-s", "--scale", default=1, type=float, help="Scale factor")
    parser.add_argument(
        "-c",
        "--chunksize",
        type=int,
        default=16_000_000,
        help="Max number of rows in each Parquet file (may be rounded up to multiple of 2048)",
    )
    parser.add_argument(
        "-r",
        "--sf_per_child",
        default=10,
        type=int,
        help="Ratio of scale factor to number of children; decrease to reduce memory footprint",
    )

    args = parser.parse_args()
    return args


def gen_parquet_data(tmp, opath, scale_factor, chunk_size, sf_per_child):
    print(f"Generating Parquet data with chunksize={chunk_size} and output={opath}.")
    if tmp:
        tempfile.tempdir = tmp
    tmp_db = tempfile.gettempdir() + "/tmp." + uuid.uuid1().hex
    print(f"Storing temporary DB at {tmp_db}")
    con = duckdb.connect(tmp_db)
    con.sql("SET allocator_background_threads = true;")
    # ratio of scale factor to number of children; decrease to reduce memory footprint
    children = math.ceil(scale_factor / sf_per_child)
    for step in range(children):
        con.sql(f"CALL dbgen(sf={scale_factor}, children={children}, step={step});")
    # ensure (approximately) correct chunk sizes by using only one thread for export
    con.sql("SET threads = 1;")
    con.sql(
        """EXPORT DATABASE '{0}' (
    FORMAT PARQUET,
    COMPRESSION SNAPPY,
    ROW_GROUP_SIZE {1},
    ROW_GROUPS_PER_FILE 1,
    PER_THREAD_OUTPUT true,
    OVERWRITE true
  );""".format(opath, chunk_size)
    )
    con.close()
    try:
        os.remove(tmp_db)
    except OSError:
        print(f"Error removing temporary DB at {tmp_db}!")
    else:
        print(f"Removing temporary DB at {tmp_db}")
    # remove .* suffix from subdir names
    for subdir in next(os.walk(opath))[1]:
        os.rename(
            os.path.join(opath, subdir),
            os.path.join(opath, subdir[: subdir.rfind(".")]),
        )


if __name__ == "__main__":
    # Parse cmd arguments
    args = parse_args()

    gen_parquet_data(args.tmp, args.output, args.scale, args.chunksize, args.sf_per_child)
