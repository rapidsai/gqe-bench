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

import logging
import os
import sqlite3

import pytest

from database_benchmarking_tools.experiment import (
    ExperimentDB,
    ExperimentNuma,
    NumaInfo,
)
from database_benchmarking_tools.hardware_info import (
    NumaNode,
    NumaTopology,
    parse_cpu_list,
)


def write_topology(root, nodes):
    """Describe NUMA nodes under `root` as sysfs does: `node<N>/{meminfo,cpulist}`.

    `nodes` maps a node index to (MemTotal in kB, cpulist text).
    """
    for index, (memory_total_kb, cpulist) in nodes.items():
        node_dir = root / f"node{index}"
        node_dir.mkdir(parents=True)
        (node_dir / "meminfo").write_text(
            f"Node {index} MemTotal:       {memory_total_kb} kB\n"
            f"Node {index} MemFree:        1 kB\n"
        )
        (node_dir / "cpulist").write_text(cpulist + "\n")
    (root / "online").write_text("0\n")


def write_numa_maps(proc_root, first_mapping_policy):
    pid_dir = proc_root / "self"
    pid_dir.mkdir(parents=True)
    (pid_dir / "numa_maps").write_text(
        f"00400000 {first_mapping_policy} file=/usr/bin/python3 mapped=32\n"
        "7f0000000000 default anon=1 dirty=1\n"
    )


# A dual-socket host with one node per socket.
DUAL_SOCKET = {0: (1048576 * 1024, "0-191"), 1: (1048576 * 1024, "192-383")}


@pytest.fixture
def dual_socket(tmp_path):
    sysfs = tmp_path / "sys"
    write_topology(sysfs, DUAL_SOCKET)
    return NumaTopology(sysfs_root=str(sysfs), proc_root=str(tmp_path / "proc")), tmp_path


class TestParseCpuList:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("0", {0}),
            ("0-3", {0, 1, 2, 3}),
            ("0-1,8,10-11\n", {0, 1, 8, 10, 11}),
            ("", set()),
        ],
    )
    def test_parse(self, text, expected):
        assert parse_cpu_list(text) == expected


class TestNumaTopology:
    def test_nodes(self, dual_socket):
        topology, _ = dual_socket
        assert topology.nodes() == [
            NumaNode(0, 1048576, frozenset(range(0, 192))),
            NumaNode(1, 1048576, frozenset(range(192, 384))),
        ]

    def test_nodes_are_sorted_numerically(self, tmp_path):
        write_topology(tmp_path, {10: (1024, "0"), 2: (1024, "1")})
        assert [node.index for node in NumaTopology(sysfs_root=str(tmp_path)).nodes()] == [2, 10]

    def test_memory_only_node_has_no_cpus(self, tmp_path):
        write_topology(tmp_path, {0: (2048, "0-71"), 2: (1024, "")})
        nodes = NumaTopology(sysfs_root=str(tmp_path)).nodes()
        assert [(node.index, len(node.cpus)) for node in nodes] == [(0, 72), (2, 0)]

    def test_no_numa_sysfs(self, tmp_path):
        topology = NumaTopology(sysfs_root=str(tmp_path / "missing"))
        assert topology.nodes() == []
        assert topology.cpu_nodes([]) == []
        assert topology.memory_nodes([]) == []

    def test_node_with_unreadable_meminfo_is_skipped_with_a_warning(self, tmp_path, caplog):
        write_topology(tmp_path, {0: (2048, "0"), 1: (2048, "1")})
        (tmp_path / "node1" / "meminfo").write_text("garbage\n")
        with caplog.at_level(logging.WARNING):
            assert [node.index for node in NumaTopology(sysfs_root=str(tmp_path)).nodes()] == [0]
        assert "Skipping NUMA node 1" in caplog.text

    def test_node_without_cpulist_is_skipped_with_a_warning(self, tmp_path, caplog):
        write_topology(tmp_path, {0: (2048, "0"), 1: (2048, "1")})
        (tmp_path / "node1" / "cpulist").unlink()
        with caplog.at_level(logging.WARNING):
            assert [node.index for node in NumaTopology(sysfs_root=str(tmp_path)).nodes()] == [0]
        assert "Skipping NUMA node 1" in caplog.text

    def test_cpu_nodes_follow_affinity(self, dual_socket, monkeypatch):
        topology, _ = dual_socket
        nodes = topology.nodes()
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {3, 4})
        assert topology.cpu_nodes(nodes) == [0]
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {3, 200})
        assert topology.cpu_nodes(nodes) == [0, 1]

    @pytest.mark.parametrize(
        "policy, expected",
        [
            ("bind:1", [1]),
            ("bind:0-1", [0, 1]),
            ("default", [0, 1]),
            # Only `bind` is recognized. Other policies are reported as every node, the same as an
            # unbound process, even when they name a single node.
            ("interleave:1", [0, 1]),
            ("prefer:1", [0, 1]),
        ],
    )
    def test_memory_nodes(self, dual_socket, policy, expected):
        topology, root = dual_socket
        write_numa_maps(root / "proc", policy)
        assert topology.memory_nodes(topology.nodes()) == expected

    def test_memory_nodes_ignore_a_binding_of_a_later_mapping(self, dual_socket):
        topology, root = dual_socket
        write_numa_maps(root / "proc", "default")
        with open(root / "proc" / "self" / "numa_maps", "a") as numa_maps:
            numa_maps.write("7f1000000000 bind:1 anon=1\n")
        assert topology.memory_nodes(topology.nodes()) == [0, 1]

    def test_memory_nodes_without_procfs(self, dual_socket):
        topology, _ = dual_socket
        assert topology.memory_nodes(topology.nodes()) == [0, 1]


@pytest.fixture
def edb(tmp_path):
    db_path = str(tmp_path / "test.db")
    edb_config = ExperimentDB(db_path)
    edb_config.create_experiment_db()
    with edb_config as conn:
        conn._cursor.execute(
            "INSERT INTO hw_info (h_hostname, h_cpu_arch) VALUES ('host', 'x86_64')"
        )
        yield conn, db_path


def select(conn, sql):
    return conn._cursor.execute(sql).fetchall()


class TestNumaRecording:
    def test_topology_is_recorded_per_node(self, edb, dual_socket):
        conn, _ = edb
        topology, _ = dual_socket
        ids = conn.insert_numa_topology(1, topology.nodes())
        assert set(ids) == {0, 1}
        assert select(
            conn,
            "SELECT n_node_index, n_memory_total_mb, n_cpu_count FROM numa_info ORDER BY n_id",
        ) == [(0, 1048576, 192), (1, 1048576, 192)]

    def test_recording_twice_reuses_rows(self, edb, dual_socket):
        conn, _ = edb
        topology, _ = dual_socket
        assert conn.insert_numa_topology(1, topology.nodes()) == conn.insert_numa_topology(
            1, topology.nodes()
        )
        assert select(conn, "SELECT COUNT(*) FROM numa_info") == [(2,)]

    def test_repartitioning_creates_new_rows(self, edb, tmp_path):
        conn, _ = edb
        nps1 = tmp_path / "nps1"
        nps4 = tmp_path / "nps4"
        write_topology(nps1, {0: (4096, "0-7")})
        write_topology(
            nps4, {0: (1024, "0-1"), 1: (1024, "2-3"), 2: (1024, "4-5"), 3: (1024, "6-7")}
        )
        before = conn.insert_numa_topology(1, NumaTopology(sysfs_root=str(nps1)).nodes())
        after = conn.insert_numa_topology(1, NumaTopology(sysfs_root=str(nps4)).nodes())
        assert before[0] != after[0]
        assert select(conn, "SELECT COUNT(*) FROM numa_info") == [(5,)]

    def test_no_numa_records_nothing(self, edb, tmp_path):
        conn, _ = edb
        assert conn.insert_numa_topology(1, []) == {}

    def test_roles_are_independent(self, edb):
        conn, _ = edb
        node0 = conn.insert_numa_info(NumaInfo(1, 0, 2048, 8))
        node1 = conn.insert_numa_info(NumaInfo(1, 1, 4096, 8))
        # A remote-NUMA run: CPUs on node 0, memory from node 1.
        conn.insert_experiment_numa(ExperimentNuma(7, node0, "cpu"))
        conn.insert_experiment_numa(ExperimentNuma(7, node1, "memory"))
        assert select(
            conn,
            "SELECT role, node_count, nodes, memory_total_mb, cpu_count "
            "FROM experiment_numa_binding WHERE experiment_id = 7 ORDER BY role",
        ) == [("cpu", 1, "0", 2048, 8), ("memory", 1, "1", 4096, 8)]

    def test_binding_view_aggregates_nodes(self, edb, dual_socket):
        conn, _ = edb
        topology, _ = dual_socket
        for numa_info_id in conn.insert_numa_topology(1, topology.nodes()).values():
            conn.insert_experiment_numa(ExperimentNuma(7, numa_info_id, "memory"))
        assert select(
            conn,
            "SELECT node_count, memory_total_mb, cpu_count FROM experiment_numa_binding",
        ) == [(2, 2097152, 384)]

    def test_binding_view_lists_nodes_in_ascending_order(self, edb):
        conn, _ = edb
        # Insert the nodes in descending index order, so that `n_id` order and the order of the
        # junction's primary key both run against the node index.
        numa_info_ids = {
            index: conn.insert_numa_info(NumaInfo(1, index, 1024, 4)) for index in (3, 2, 1, 0)
        }
        for index in (2, 0, 3, 1):
            conn.insert_experiment_numa(ExperimentNuma(7, numa_info_ids[index], "memory"))
        assert select(conn, "SELECT nodes FROM experiment_numa_binding") == [("0,1,2,3",)]

    def test_unknown_role_is_rejected(self, edb):
        conn, _ = edb
        node = conn.insert_numa_info(NumaInfo(1, 0, 2048, 8))
        with pytest.raises(sqlite3.IntegrityError):
            conn.insert_experiment_numa(ExperimentNuma(7, node, "bogus"))
