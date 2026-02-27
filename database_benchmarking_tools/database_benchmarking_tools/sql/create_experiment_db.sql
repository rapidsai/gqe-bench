/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

-- Create an SQLite database to collect database measurements

BEGIN TRANSACTION;

-- System-under-test information
--
-- A dimension table that describes the system-under-test on which the
-- experiment is conducted.
--
-- To keep the system names consistent, we maintain a list of systems and
-- pre-insert them into this table. See the INSERT statements below.
CREATE TABLE sut_info(
  s_id INTEGER PRIMARY KEY,
  -- System name.
  s_name TEXT NOT NULL,
  UNIQUE (s_name)
);

-- Add a common entry for hand-coded benchmarks to keep their naming consistent:
INSERT INTO sut_info(s_id, s_name) VALUES (5993586517390597806, 'cuda');

-- Hardware information
--
-- A dimension table that describes the hardware and system environment in which
-- the experiment is conducted.
CREATE TABLE hw_info(
  h_id INTEGER PRIMARY KEY,
  -- Hostname.
  h_hostname TEXT NOT NULL,
  -- CPU architecture.
  h_cpu_arch TEXT NOT NULL,
  -- CPU model name (optional).
  h_cpu_model_name TEXT,
  -- CPU clock rate in MHz (optional).
  h_cpu_clock_mhz INTEGER,
  -- CPU physical cores (optional).
  h_cpu_physical_cores INTEGER,
  -- CPU logical cores (optional).
  h_cpu_logical_cores INTEGER,
  -- GPU product name (optional).
  h_gpu_product_name TEXT,
  -- System driver version (optional).
  h_nvidia_driver_version TEXT,
  -- CUDA driver version (optional).
  h_cuda_version TEXT,
  -- PCIe link generation (optional).
  h_gpu_pcie_link_generation INTEGER,
  -- Number of GPU CUDA cores (optional).
  h_gpu_cuda_cores INTEGER,
  -- SM maximum clock rate in MHz (optional).
  h_gpu_max_clock_sm_mhz INTEGER,
  -- Memory maximum clock rate in MHz (optional).
  h_gpu_max_clock_memory_mhz INTEGER,
  -- Total number of ECC memory errors detected by the GPU (optional).
  --
  -- Sanity check to detect faulty hardware.
  h_gpu_ecc_errors INTEGER,
  -- Each system (i.e., hardware + OS) should be unique within the experiment
  -- database.
  UNIQUE (
    h_hostname,
    h_cpu_arch,
    h_cpu_model_name,
    h_cpu_clock_mhz,
    h_cpu_physical_cores,
    h_cpu_logical_cores,
    h_gpu_product_name,
    h_nvidia_driver_version,
    h_cuda_version,
    h_gpu_pcie_link_generation,
    h_gpu_max_clock_sm_mhz,
    h_gpu_max_clock_memory_mhz,
    h_gpu_ecc_errors
  )
);

-- Build information
--
-- A dimension table that describes how the system-under-test was compiled. The
-- build information table is shared among all systems-under-test, and contains
-- no SUT-specific columns.
CREATE TABLE build_info(
  b_id INTEGER PRIMARY KEY,
  -- Version number (optional).
  b_version TEXT,
  -- Source code revision (optional).
  b_revision TEXT,
  -- Source code branch (optional).
  b_branch TEXT,
  -- Is source code working directory dirty (optional).
  b_is_dirty INT,
  -- Date and time of revision's commit as Unix timestamp (optional).
  b_commit_timestamp INTEGER,
  -- Date of revision's commit (virtual column).
  b_commit_date TEXT AS (date(b_commit_timestamp, 'unixepoch', '-08:00')),
  -- Time  of revision's commit (virtual column).
  b_commit_time TEXT AS (time(b_commit_timestamp, 'unixepoch', '-08:00')),
  -- Timezone of the data and time virtual columns (virtual column).
  b_commit_timezone TEXT AS ('PST'),
  -- Compiler optimization flags (optional).
  b_compiler_flags TEXT,
  -- Each build configuration should be unique within the experiment
  -- database.
  UNIQUE (b_version, b_revision, b_branch, b_is_dirty, b_commit_timestamp, b_compiler_flags)
);

-- Data information
--
-- A dimension table that describes the input data of the experiment.
CREATE TABLE data_info(
  d_id INTEGER PRIMARY KEY,
  -- Kind of storage device (optional).
  --
  -- Examples: 'disk', 'memory'.
  --
  -- Note that hardware information is gathered in `hw_info`.
  d_storage_device_kind TEXT,
  -- Data format (optional).
  --
  -- Examples: 'internal', 'csv', 'parquet'.
  d_format TEXT,
  -- Data storage location (optional).
  --
  -- This can be a (list of) filesystem path, a disk device identifier, or a
  -- NUMA memory node identifier.
  --
  -- Examples: '/storage/tpch/sf100', 'node0,node1'.
  d_location TEXT,
  -- SQL data types are annotated with `NOT NULL` whenever possible (optional).
  d_not_null INT,
  -- SQL data type used as row identifier (optional).
  --
  -- The row identifier is usually considered the primary key.
  --
  -- Examples: 'int', 'bigint'.
  d_identifier_type TEXT,
  -- SQL data type used for fixed-length string attributes (optional).
  --
  -- Examples: 'char', 'text'.
  d_char_type TEXT,
  -- SQL data type used for fixed-point decimal attributes (optional).
  --
  -- Examples: 'decimal', 'float', 'double'.
  d_decimal_type TEXT,
  -- TPC-{DS,H} scale factor.
  --
  -- Fractional scale factors are permitted for small data sizes (optional).
  d_scale_factor FLOAT,
  -- Each data specification should be unique within the experiment database.
  UNIQUE (
    d_storage_device_kind,
    d_format,
    d_location,
    d_identifier_type,
    d_char_type,
    d_decimal_type,
    d_scale_factor
  )
);

-- An experiment description
--
-- A fact table that describes how an experiment is set up. This references
-- dimension tables containing system-under-test parameters, hardware
-- information, build information, etc.
--
-- The `e_parameters_id` reference is optional to allow measuring new SUTs for
-- which a parameter table isn't yet defined in the experiment schema.
--
-- Note: The foreign key `e_parameters_id` must be manually maintained, due to
-- the "table-per-concrete" schema design. A `FOREIGN KEY` constraint cannot be
-- enforced.
CREATE TABLE experiment(
  e_id INTEGER PRIMARY KEY,
  -- Reference to the system-under-test.
  e_sut_info_id INTEGER NOT NULL,
  -- Reference to system-under-test parameters (optional).
  e_parameters_id INTEGER,
  -- Reference to hardware information (optional).
  e_hw_info_id INTEGER,
  -- Reference to build information (optional).
  e_build_info_id INTEGER,
  -- Reference to data information (optional).
  e_data_info_id INTEGER,
  -- Reference to query information (optional).
  e_query_info_id INTEGER,
  -- The # of runs expected as part of this experiment.
  e_sample_size INTEGER,
  -- Date and time when experiment was conducted.
  e_timestamp INTEGER NOT NULL DEFAULT (unixepoch('now')),
  -- Date when experiment was conducted (virtual column).
  e_date TEXT AS (date(e_timestamp, 'unixepoch', '-08:00')),
  -- Time when experiment was conducted (virtual column).
  e_time TEXT AS (time(e_timestamp, 'unixepoch', '-08:00')),
  -- Timezone of the data and time virtual columns (virtual column).
  e_timezone TEXT AS ('PST'),
  FOREIGN KEY (e_sut_info_id) REFERENCES sut_info(s_id),
  FOREIGN KEY (e_hw_info_id) REFERENCES hw_info(h_id),
  FOREIGN KEY (e_build_info_id) REFERENCES build_info(b_id),
  FOREIGN KEY (e_data_info_id) REFERENCES data_info(d_id),
  FOREIGN KEY (e_query_info_id) REFERENCES query_info(q_id)
);

-- An experiment run
--
-- A fact table that contains the measured result of an experiment run, along
-- with metadata directly associated with the run.
--
-- A run is directly associated with an experiment.
CREATE TABLE run(
  -- Experiment ID this run belongs to.
  r_experiment_id INTEGER NOT NULL,
  -- Run number of an experiment.
  --
  -- Run numbers should be unique within an experiment. Runs are usually numbered
  -- from 0 to N.
  r_number INTEGER NOT NULL,
  -- NVTX marker associated with this run.
  r_nvtx_marker TEXT,
  -- System execution time in seconds.
  r_duration_s REAL NOT NULL,
  PRIMARY KEY (r_experiment_id, r_number),
  FOREIGN KEY (r_experiment_id) REFERENCES experiment(e_id)
);

-- A failed experiment run
--
-- A fact table that contains an error indicating why the run failed.
-- A failed_run is directly associated with an experiment.
--
-- This is an EER specialization design; run and failed_run are specializations of
-- the same entity.
-- Invariant: The composite primary key should be unique across both tables.
CREATE TABLE failed_run(
  -- Experiment ID this run belongs to.
  fr_experiment_id INTEGER NOT NULL,
  -- Run number of an experiment.
  fr_number INTEGER NOT NULL,
  -- Error message (optional)
  fr_error_msg TEXT,
  PRIMARY KEY (fr_experiment_id, fr_number),
  FOREIGN KEY (fr_experiment_id) REFERENCES experiment(e_id)
);

CREATE TRIGGER prevent_repeat_run
BEFORE INSERT ON run
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'Constraint: run already exists as a failed_run')
    WHERE EXISTS (
        SELECT 1 From failed_run
        WHERE NEW.r_experiment_id = fr_experiment_id AND  NEW.r_number = fr_number);
END;

CREATE TRIGGER prevent_repeat_failed_run
BEFORE INSERT ON failed_run
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'Constraint: failed_run already exists as a run')
    WHERE EXISTS (
        SELECT 1 From run
        WHERE r_experiment_id = NEW.fr_experiment_id AND  r_number = NEW.fr_number);
END;

CREATE TABLE query_info(
  q_id INTEGER PRIMARY KEY,
  -- Query name.
  --
  -- E.g., 'Q1'.
  q_name TEXT NOT NULL,
  -- Query suite (optional).
  --
  -- E.g., 'TPC-H', 'TPC-DS'.
  q_suite TEXT,
  -- Query source (optional).
  --
  -- E.g., 'sql', 'handcoded', 'substrait'.
  q_source TEXT,
  UNIQUE (q_name, q_suite, q_source)
);

-- A join index for joining runs with their experiment description
CREATE INDEX index_run_experiment ON run(r_experiment_id);

COMMIT TRANSACTION;
