-- Create an SQLite database to collect GQE measuremnts

BEGIN TRANSACTION;

-- Parameters information
--
-- A dimension table that describes the GQE parameters passed to an experiment.
CREATE TABLE parameters(
  p_id INTEGER PRIMARY KEY,
  p_num_workers INTEGER NOT NULL,
  p_num_partitions INTEGER NOT NULL,
  p_join_use_hash_map_cache INTEGER NOT NULL,
  p_read_use_zero_copy INTEGER NOT NULL,
  -- Each parameter combination should be unique within the experiment database.
  UNIQUE (p_num_workers, p_num_partitions, p_join_use_hash_map_cache, p_read_use_zero_copy)
);

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
  -- GPU product name.
  h_gpu_product_name TEXT,
  -- System driver version.
  h_nvidia_driver_version TEXT,
  -- CUDA driver version.
  h_cuda_version TEXT,
  -- PCIe link generation.
  h_gpu_pcie_link_generation INTEGER,
  -- Number of GPU CUDA cores.
  h_gpu_cuda_cores INTEGER,
  -- SM maximum clock rate in MHz.
  h_gpu_max_clock_sm_mhz INTEGER,
  -- Memory maximum clock rate in MHz.
  h_gpu_max_clock_memory_mhz INTEGER,
  -- Total number of ECC memory errors detected by the GPU. Sanity check to
  -- detect faulty hardware.
  h_gpu_ecc_errors INTEGER,
  -- Each system (i.e., hardware + OS) should be unique within the experiment
  -- database.
  UNIQUE (
    h_hostname,
    h_cpu_arch,
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
-- A dimension table that describes how GQE was compiled.
CREATE TABLE build_info(
  b_id INTEGER PRIMARY KEY
  -- FIXME: b_git_revision TEXT,
  -- FIXME: b_cmake_build_type TEXT,
  -- FIXME: Each build configuration should be unique within the experiment
  -- database.
);

-- An experiment description
--
-- A fact table that describes how an experiment is set up. This references
-- dimension tables containing GQE parameters, hardware information, build
-- information, etc.
CREATE TABLE experiment(
  e_id INTEGER PRIMARY KEY,
  -- Reference to GQE parameters.
  e_parameters_id INTEGER NOT NULL,
  -- Reference to hardware information.
  e_hw_info_id INTEGER,
  -- Reference to build information.
  e_build_info_id INTEGER,
  -- Experiment name (e.g., 'Q1').
  e_name TEXT NOT NULL,
  -- Experiment suite (e.g., 'TPC-H').
  e_suite TEXT,
  -- Experiment scale factor
  e_scale_factor INTEGER,
  -- Date and time when experiment was conducted.
  e_timestamp INTEGER NOT NULL DEFAULT (unixepoch('now')),
  -- Date when experiment was conducted (virtual column).
  e_date TEXT AS (date(e_timestamp, 'unixepoch', '-08:00')),
  -- Time when experiment was conducted (virtual column).
  e_time TEXT AS (time(e_timestamp, 'unixepoch', '-08:00')),
  -- Timezone of the data and time virtual columns (virtual column).
  e_timezone TEXT AS ('PST'),
  FOREIGN KEY (e_parameters_id) REFERENCES parameters(p_id),
  FOREIGN KEY (e_hw_info_id) REFERENCES hw_info(h_id),
  FOREIGN KEY (e_build_info_id) REFERENCES build_info(b_id)
);

-- An experiment run
--
-- A fact table that contains the measured result of an experiment run, along
-- with metadata directly associated with the run.
--
-- A run is directly associated with an experiment.
CREATE TABLE run(
  r_id INTEGER PRIMARY KEY,
  -- Experiment ID this run belongs to.
  r_experiment_id INTEGER NOT NULL,
  -- Run number of an experiment.
  r_number INTEGER,
  -- NVTX marker associated with this run.
  r_nvtx_marker TEXT,
  -- GQE execution time in seconds.
  r_duration_s REAL NOT NULL,
  FOREIGN KEY (r_experiment_id) REFERENCES experiment(e_id),
  -- Run number should be unique within an experiment. Runs are usually numbered
  -- from 0 to N. `NULL` values are considered distinct from other values,
  -- including other `NULL`s.
  UNIQUE (r_experiment_id, r_number)
);

-- A join index for joining runs with their experiment description
CREATE INDEX index_run_experiment ON run(r_experiment_id);

-- A short description of a run.
--
-- A virtual view that joins runs with their experiment description and
-- parameters.
CREATE VIEW run_parameters AS
  SELECT e_suite, e_name, e_scale_factor, run.*, parameters.*
  FROM run
  JOIN experiment ON r_experiment_id = e_id
  JOIN parameters ON p_id = e_parameters_id
  ;

-- All information about a run.
CREATE VIEW run_all_info AS
  SELECT *
  FROM run
  JOIN experiment ON r_experiment_id = e_id
  JOIN parameters ON p_id = e_parameters_id
  JOIN hw_info ON h_id = e_hw_info_id
  -- FIXME: JOIN build_info ON b_id = e_build_info_id
  ;

COMMIT TRANSACTION;
