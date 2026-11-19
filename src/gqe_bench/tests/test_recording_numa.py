# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that NUMA nodes and an experiment's NUMA binding reach the database.

Uses a database built from the real DDL and a topology described by a fake sysfs tree, so the tests
need no NUMA hardware.
"""

import importlib.resources
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from database_benchmarking_tools.hardware_info import NumaTopology
from gqe_bench import recording
from gqe_bench.recording import (
    _GQE_EXTENSION_DDL,
    NumaBinding,
    SweepContext,
    SystemIds,
    _GqeConnection,
    insert_data_dimensions,
    record_experiment,
)
from gqe_bench.tests.helpers import make_data_info, make_query, make_query_params

_HW_ID = 1


@pytest.fixture
def edb(tmp_path: Path) -> _GqeConnection:
    """An _GqeConnection on a database built from the real DDL, with one host."""
    base_ddl = (
        importlib.resources.files("database_benchmarking_tools.sql")
        .joinpath("create_experiment_db.sql")
        .read_text(encoding="utf-8")
    )
    conn = _GqeConnection(str(tmp_path / "test.db3"), "testhost")
    conn._cursor.executescript(base_ddl)
    conn._cursor.executescript(_GQE_EXTENSION_DDL)
    conn._cursor.execute("INSERT INTO hw_info (h_hostname, h_cpu_arch) VALUES ('host', 'x86_64')")
    conn.commit()
    return conn


@pytest.fixture
def two_node_topology(tmp_path: Path) -> NumaTopology:
    """Two nodes of 1 GiB with four CPUs each, and a process bound to the CPUs and memory of node 1."""
    sysfs = tmp_path / "sys"
    for index, cpulist in ((0, "0-3"), (1, "4-7")):
        node_dir = sysfs / f"node{index}"
        node_dir.mkdir(parents=True)
        (node_dir / "meminfo").write_text(f"Node {index} MemTotal: 1048576 kB\n")
        (node_dir / "cpulist").write_text(cpulist + "\n")
    numa_maps = tmp_path / "proc" / "self" / "numa_maps"
    numa_maps.parent.mkdir(parents=True)
    numa_maps.write_text("00400000 bind:1 file=/usr/bin/python3 mapped=32\n")
    return NumaTopology(sysfs_root=str(sysfs), proc_root=str(tmp_path / "proc"))


def _sweep(numa_bindings: tuple[NumaBinding, ...]) -> SweepContext:
    return SweepContext(
        db_mgr=MagicMock(),
        system_ids=SystemIds(1, _HW_ID, 1),
        gpu_info_ids=[],
        env={},
        suite_name="TPC-H",
        repeat=1,
        records_timing=True,
        numa_bindings=numa_bindings,
    )


def _experiment_numa(edb: _GqeConnection) -> list[tuple[int, str, int]]:
    return edb._cursor.execute(
        "SELECT en_numa_info_id, en_role, n_node_index FROM experiment_numa "
        "JOIN numa_info ON n_id = en_numa_info_id ORDER BY en_role, n_node_index"
    ).fetchall()


class TestReadNumaBindings:
    def test_binding_of_a_pinned_process(self, edb, two_node_topology, monkeypatch):
        monkeypatch.setattr(recording.hardware_info, "NumaTopology", lambda: two_node_topology)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {4, 5})

        bindings = recording._read_numa_bindings(edb, _HW_ID)

        node_ids = {
            node_index: numa_info_id
            for numa_info_id, node_index in edb._cursor.execute(
                "SELECT n_id, n_node_index FROM numa_info"
            )
        }
        assert set(bindings) == {
            NumaBinding(node_ids[1], "cpu"),
            NumaBinding(node_ids[1], "memory"),
        }

    def test_unbound_process_may_use_every_node(self, edb, two_node_topology, monkeypatch):
        (Path(two_node_topology._proc_root) / "self" / "numa_maps").write_text(
            "00400000 default file=/usr/bin/python3 mapped=32\n"
        )
        monkeypatch.setattr(recording.hardware_info, "NumaTopology", lambda: two_node_topology)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4, 5, 6, 7})

        bindings = recording._read_numa_bindings(edb, _HW_ID)

        assert sorted(binding.role for binding in bindings) == ["cpu", "cpu", "memory", "memory"]

    def test_bound_node_without_a_row_is_dropped_with_a_warning(
        self, edb, two_node_topology, monkeypatch, caplog
    ):
        # Node 2 is bound but absent from sysfs, so it gets no numa_info row.
        (Path(two_node_topology._proc_root) / "self" / "numa_maps").write_text(
            "00400000 bind:1-2 file=/usr/bin/python3 mapped=32\n"
        )
        monkeypatch.setattr(recording.hardware_info, "NumaTopology", lambda: two_node_topology)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {4, 5})

        with caplog.at_level(logging.WARNING):
            bindings = recording._read_numa_bindings(edb, _HW_ID)

        assert [binding.role for binding in bindings] == ["cpu", "memory"]
        assert "Not recording NUMA node 2 in the memory binding" in caplog.text

    def test_nodes_are_scanned_once(self, edb, two_node_topology, monkeypatch):
        scan_count = 0
        scan = two_node_topology.nodes

        def counting_scan():
            nonlocal scan_count
            scan_count += 1
            return scan()

        monkeypatch.setattr(two_node_topology, "nodes", counting_scan)
        monkeypatch.setattr(recording.hardware_info, "NumaTopology", lambda: two_node_topology)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {4, 5})

        recording._read_numa_bindings(edb, _HW_ID)

        assert scan_count == 1

    def test_host_without_numa(self, edb, tmp_path, monkeypatch):
        monkeypatch.setattr(
            recording.hardware_info,
            "NumaTopology",
            lambda: NumaTopology(sysfs_root=str(tmp_path / "missing")),
        )
        assert recording._read_numa_bindings(edb, _HW_ID) == ()


class TestRecordExperimentNuma:
    def _record(self, edb: _GqeConnection, sweep: SweepContext) -> int:
        dims = insert_data_dimensions(edb, make_data_info())
        return record_experiment(edb, dims, make_query(), make_query_params(), sweep)

    def test_experiment_is_tagged_with_its_binding(self, edb):
        node0 = edb.insert_numa_info(recording.exp.NumaInfo(_HW_ID, 0, 1024, 4))
        node1 = edb.insert_numa_info(recording.exp.NumaInfo(_HW_ID, 1, 1024, 4))
        sweep = _sweep((NumaBinding(node0, "cpu"), NumaBinding(node1, "memory")))

        experiment_id = self._record(edb, sweep)

        assert (
            edb._cursor.execute("SELECT en_experiment_id FROM experiment_numa").fetchall()
            == [(experiment_id,)] * 2
        )
        assert _experiment_numa(edb) == [(node0, "cpu", 0), (node1, "memory", 1)]

    def test_every_experiment_is_tagged(self, edb):
        node = edb.insert_numa_info(recording.exp.NumaInfo(_HW_ID, 0, 1024, 4))
        sweep = _sweep((NumaBinding(node, "cpu"),))
        dims = insert_data_dimensions(edb, make_data_info())

        first = record_experiment(edb, dims, make_query(), make_query_params(), sweep)
        second = record_experiment(edb, dims, make_query(name="Q2"), make_query_params(), sweep)

        assert edb._cursor.execute(
            "SELECT en_experiment_id FROM experiment_numa ORDER BY 1"
        ).fetchall() == [(first,), (second,)]

    def test_no_binding_records_nothing(self, edb):
        self._record(edb, _sweep(()))
        assert _experiment_numa(edb) == []
