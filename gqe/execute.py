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
from typing import Optional  # Not needed with Python>=3.10


def execute(
        catalog: Catalog, relation: Relation,
        output_path: Optional[str], log_time: bool = True) -> None:
    """
    Execute the query plan.

    :param catalog: Catalog to execute the query plan on.
    :param relation: Root relation for the query plan.
    :param output_path: Path to write the output of `relation` to a Parquet file if this argument
        is valid `str`. If this argument is `None`, the output is not written. Note that the
        behavior is undefined if `output_path` is valid but `relation` does not produce an output.
    :param log_time: Whether to log the execution time.
    """
    gqe.lib.execute(catalog._catalog, relation._to_cpp(), output_path, log_time)
