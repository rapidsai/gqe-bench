# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import gqe.lib
from typing import Dict
from gqe import table_definition


def get_row_counts() -> Dict[str, int]:
    # FIXME: Calculate based on scale factor. Hard-coding SF100 for now.
    return {
        "lineitem": 600121500,
        "orders": 150030000,
        "part": 20000000,
        "partsupp": 80000000,
        "customer": 15000000,
        "supplier": 1000000,
        "nation": 25,
        "region": 5
    }

def get_type_sizes() -> Dict[gqe.lib.TypeId, int]:
    return {
        gqe.lib.TypeId.int32: 4,
        gqe.lib.TypeId.int64: 8,
        gqe.lib.TypeId.float32: 4,
        gqe.lib.TypeId.float64: 8, 
        gqe.lib.TypeId.timestamp_days: 4, # date type
        gqe.lib.TypeId.string: 25,
        gqe.lib.TypeId.int8: 1 # char type
    }

def calculate_memory_requirements(definitions: dict[str, list[gqe.lib.ColumnTraits]]) -> int:
    type_sizes = get_type_sizes()
    row_counts = get_row_counts()
    
    total_memory = 0
    
    for table_name, columns in definitions.items():
        table_size = row_counts[table_name]
        row_size = 0
        
        for column_traits in columns:
            row_size += type_sizes[column_traits.data_type.type_id()]
            
        total_memory += table_size * row_size
        
    return total_memory

def estimate_memory_for_all_queries(identifier_type: gqe.lib.TypeId = gqe.lib.TypeId.int32, use_opt_char_type: bool = True):
    table_defs = table_definition.TPCHTableDefinitions(identifier_type, use_opt_char_type)
    
    for query_idx in range(23):  # 0-22 inclusive
        definitions = table_defs.query_table_definitions(query_idx)
        memory_needed = calculate_memory_requirements(definitions)
        
        if query_idx == 0:
            print("  Total", end='')
        else:
            print(f"Query {query_idx}", end='')
            
        print(f" memory needed: {memory_needed / (1024 * 1024 * 1024):.2f} GiB")

if __name__ == "__main__":
    estimate_memory_for_all_queries()
