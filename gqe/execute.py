# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gqe.relation import Relation
from gqe import Catalog
import gqe.lib


def execute(
        catalog: Catalog, relation: Relation, output_result: bool, log_time: bool = True) -> None:
    """
    Execute the logical plan.

    :param catalog: Catalog to execute the logical plan on.
    :param relation: Root relation for the logical plan.
    :param output_result: Whether to write the output of `relation` to a Parquet file
        `output.parquet`.
    :param log_time: Whether to log the execution time.
    """
    gqe.lib.execute(catalog._catalog, relation._to_cpp(), output_result, log_time)
