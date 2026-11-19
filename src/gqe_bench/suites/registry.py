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

"""Suite class registry."""

from gqe_bench.suites.base import Suite
from gqe_bench.suites.tpch import TpchSuite

# Registered suite classes. Suites are stateless; the registry returns the
# class itself, and callers invoke its classmethods. Unknown names fall back
# to the generic ``Suite`` base class, which raises on suite-specialized
# methods (e.g. ``to_ddl``, ``query_sql``).
_REGISTRY: dict[str, type[Suite]] = {cls.NAME: cls for cls in [TpchSuite]}


def get_suite(suite_name: str) -> type[Suite]:
    """Return the Suite class registered under ``suite_name``, or ``Suite``."""
    return _REGISTRY.get(suite_name, Suite)
