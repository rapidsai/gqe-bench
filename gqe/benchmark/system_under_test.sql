-- Add GQE as a system-under-test

BEGIN TRANSACTION;

-- GQE system-under-test information
--
-- Insert a SUT entry for GQE with a constant, globally unique primary key.
INSERT INTO sut_info(s_id, s_name) VALUES (2717457836325482278, 'gqe');

-- GQE parameters information
--
-- A dimension table that describes the GQE parameters passed to an experiment.
CREATE TABLE gqe_parameters(
  p_id INTEGER PRIMARY KEY,
  p_sut_info_id INTEGER NOT NULL,
  p_num_workers INTEGER NOT NULL,
  p_num_partitions INTEGER NOT NULL,
  p_use_overlap_mtx INTEGER NOT NULL,
  p_join_use_hash_map_cache INTEGER NOT NULL,
  p_read_use_zero_copy INTEGER NOT NULL,
  p_join_use_unique_keys INTEGER NOT NULL,
  p_join_use_perfect_hash INTEGER NOT NULL,
  p_join_use_mark_join INTEGER NOT NULL,
  p_use_partition_pruning INTEGER NOT NULL,
  p_filter_use_like_shift_and INTEGER NOT NULL,
  p_aggregation_use_perfect_hash INTEGER NOT NULL,
  -- Each parameter combination should be unique within the experiment database.
  UNIQUE (
    p_sut_info_id,
    p_num_workers,
    p_num_partitions,
    p_use_overlap_mtx,
    p_join_use_hash_map_cache,
    p_read_use_zero_copy,
    p_join_use_unique_keys,
    p_join_use_perfect_hash,
    p_join_use_mark_join,
    p_use_partition_pruning,
    p_filter_use_like_shift_and,
    p_aggregation_use_perfect_hash
  ),
  FOREIGN KEY (p_sut_info_id) REFERENCES sut_info(s_id)
);

-- GQE extended data information.
--
-- A dimension table that extends `data_info` with data parameters specific to
-- GQE.
CREATE TABLE gqe_data_info_ext(
  de_id INTEGER PRIMARY KEY,
  de_data_info_id INTEGER NOT NULL,
  de_num_row_groups INTEGER NOT NULL,
  de_compression_format TEXT NOT NULL,
  de_compression_chunk_size INTEGER NOT NULL,
  de_zone_map_partition_size INTEGER NOT NULL,
  de_compression_ratio_threshold REAL NOT NULL,
  de_secondary_compression_format TEXT NOT NULL,
  de_secondary_compression_ratio_threshold REAL NOT NULL,
  de_secondary_compression_multiplier_threshold REAL NOT NULL,
  de_use_cpu_compression INTEGER NOT NULL,
  de_compression_level INTEGER NOT NULL,
  UNIQUE (
    de_data_info_id,
    de_num_row_groups,
    de_compression_format,
    de_compression_chunk_size,
    de_zone_map_partition_size,
    de_compression_ratio_threshold,
    de_secondary_compression_format,
    de_secondary_compression_ratio_threshold,
    de_secondary_compression_multiplier_threshold,
    de_use_cpu_compression,
    de_compression_level
  ),
  FOREIGN KEY (de_data_info_id) REFERENCES data_info(d_id)
);

-- GQE Table statistics information.
--
-- A fact table that contains metadata about tables used during query execution. It depends on the gqe data parameters,
-- but not on the query run parameters. It references the gqe_data_info_ext table to associate the statistics with a specific data configuration.
--
-- Note that table statistics are query-specific (ts_query_name) because different queries access different subsets of the same table if load_all_data is not enabled.
-- A query may only read certain columns or certain row groups (row group pruning based on filters). This query-specific association allows us to understand the actual
-- data footprint and I/O costs for each query.
CREATE TABLE gqe_table_stats(
  ts_id INTEGER PRIMARY KEY,
  ts_data_info_ext_id INTEGER NOT NULL,
  ts_query_info_id INTEGER NOT NULL,
  ts_table_name TEXT NOT NULL,
  ts_columns INTEGER NOT NULL,
  ts_rows INTEGER NOT NULL,
  ts_row_groups INTEGER NOT NULL,
  UNIQUE (
    ts_data_info_ext_id,
    ts_query_info_id,
    ts_table_name
  ),
  FOREIGN KEY (ts_data_info_ext_id) REFERENCES gqe_data_info_ext(de_id),
  FOREIGN KEY (ts_query_info_id) REFERENCES query_info(q_id)
);

-- GQE Column statistics information.

-- A fact table that contains metadata about each columns used during query execution.
-- It references the gqe_table_stats table to associate the statistics with a specific table within a specific data configuration.
CREATE TABLE gqe_column_stats(
  cs_id INTEGER PRIMARY KEY,
  cs_gqe_table_stats_id INTEGER NOT NULL,
  cs_column_name TEXT NOT NULL,
  cs_compressed_size INTEGER NOT NULL,
  cs_uncompressed_size INTEGER NOT NULL,
  cs_compression_ratio REAL GENERATED ALWAYS AS (CAST(cs_uncompressed_size AS REAL) / cs_compressed_size) VIRTUAL,
  cs_slices INTEGER NOT NULL,
  cs_compressed_slices INTEGER NOT NULL,
  UNIQUE (
    cs_gqe_table_stats_id,
    cs_column_name
  ),
  FOREIGN KEY (cs_gqe_table_stats_id) REFERENCES gqe_table_stats(ts_id)
);

-- Add reference to gqe_data_info_ext in experiment table.
ALTER TABLE experiment ADD COLUMN e_data_info_ext_id INTEGER NOT NULL REFERENCES gqe_data_info_ext(de_id);

-- A unified view of general and GQE-extended data information.
CREATE VIEW gqe_data_info AS
  SELECT *
    FROM data_info
         JOIN gqe_data_info_ext ON d_id = de_data_info_id
                ;

-- GQE metric information.
--
-- A dimension table for describing profiler metrics, typically obtained from
-- CUPTI.
CREATE TABLE gqe_metric_info (
  m_id INTEGER PRIMARY KEY,
  m_name TEXT NOT NULL,
  UNIQUE (m_name)
);

-- GQE extended run.
--
-- A fact table that extends `run` with profiler metrics, typically obtained
-- from CUPTI.
CREATE TABLE gqe_run_ext(
  re_id INTEGER PRIMARY KEY,
  re_run_id INTEGER NOT NULL,
  re_metric_info_id INTEGER NOT NULL,
  re_metric_value REAL NOT NULL,
  FOREIGN KEY (re_run_id) REFERENCES run(r_id),
  FOREIGN KEY (re_metric_info_id) REFERENCES gqe_metric_info(m_id)
);

-- A short description of a GQE run.
--
-- A virtual view that joins runs with their experiment description and
-- parameters.
CREATE VIEW gqe_run_parameters AS
  SELECT q_suite, q_name, d_scale_factor, run.*, gqe_parameters.*, gqe_data_info_ext.*
    FROM run
         JOIN experiment ON r_experiment_id = e_id
         JOIN gqe_parameters ON gqe_parameters.p_id = e_parameters_id
             AND gqe_parameters.p_sut_info_id = e_sut_info_id
         JOIN gqe_data_info_ext ON e_data_info_ext_id = gqe_data_info_ext.de_id
         JOIN data_info ON d_id = gqe_data_info_ext.de_data_info_id
         JOIN query_info ON e_query_info_id = q_id
                ;

-- All information about a GQE run.
CREATE VIEW gqe_run_all_info AS
  SELECT *
    FROM run
         JOIN experiment ON r_experiment_id = e_id
         JOIN sut_info ON s_id = e_sut_info_id
         JOIN gqe_parameters ON gqe_parameters.p_id = e_parameters_id
             AND gqe_parameters.p_sut_info_id = e_sut_info_id
         JOIN hw_info ON h_id = e_hw_info_id
         JOIN data_info ON d_id = e_data_info_id
         JOIN gqe_data_info_ext ON de_id = e_data_info_ext_id
         JOIN build_info ON b_id = e_build_info_id
         JOIN query_info ON q_id = e_query_info_id
                ;

-- Best GQE optimization parameters per query.
--
-- The parameters that result in the lowest average query duration. These
-- parameters are gathered along with statistics over their query duration.
--
-- The first run of each experiment is filtered out as a warm-up run.
CREATE VIEW gqe_best_parameters AS
  WITH
  data AS (
    SELECT
      e_id,
      q_name,
      q_suite,
      q_source,
      p_num_workers,
      p_num_partitions,
      p_use_overlap_mtx,
      p_join_use_hash_map_cache,
      p_read_use_zero_copy,
      p_join_use_unique_keys,
      p_join_use_perfect_hash,
      p_join_use_mark_join,
      p_use_partition_pruning,
      p_filter_use_like_shift_and,
      p_aggregation_use_perfect_hash,
      d_storage_device_kind,
      d_format,
      d_location,
      d_not_null,
      d_identifier_type,
      d_char_type,
      d_decimal_type,
      d_scale_factor,
      de_data_info_id,
      de_num_row_groups,
      de_compression_format,
      de_compression_chunk_size,
      de_zone_map_partition_size,
      de_compression_ratio_threshold,
      de_secondary_compression_format,
      de_secondary_compression_ratio_threshold,
      de_secondary_compression_multiplier_threshold,
      de_use_cpu_compression,
      de_compression_level,
      avg(r_duration_s) AS r_avg_duration_s,
      min(r_duration_s) AS r_min_duration_s,
      max(r_duration_s) AS r_max_duration_s,
      count(r_duration_s) AS sample_size
      FROM
        experiment
        JOIN run ON e_id = r_experiment_id
        JOIN gqe_parameters ON e_parameters_id = p_id
        JOIN data_info ON d_id = e_data_info_id
        JOIN gqe_data_info_ext ON de_id = e_data_info_ext_id
        JOIN query_info ON e_query_info_id = q_id
     WHERE r_number > 0
     GROUP BY
      e_id,
      q_name,
      q_suite,
      p_num_workers,
      p_num_partitions,
      p_use_overlap_mtx,
      p_join_use_hash_map_cache,
      p_read_use_zero_copy,
      p_join_use_unique_keys,
      p_join_use_perfect_hash,
      p_join_use_mark_join,
      p_use_partition_pruning,
      p_filter_use_like_shift_and,
      p_aggregation_use_perfect_hash,
      d_scale_factor,
      d_storage_device_kind,
      d_format,
      d_location,
      d_not_null,
      d_identifier_type,
      d_char_type,
      d_decimal_type,
      de_data_info_id,
      de_num_row_groups,
      de_compression_format,
      de_compression_chunk_size,
      de_zone_map_partition_size,
      de_compression_ratio_threshold,
      de_secondary_compression_format,
      de_secondary_compression_ratio_threshold,
      de_secondary_compression_multiplier_threshold,
      de_use_cpu_compression,
      de_compression_level
  )
  SELECT
    data.*
    FROM
      data
      JOIN (
        SELECT
          q_name,
          min(r_avg_duration_s) AS min_duration
          FROM
            data
         GROUP BY
      q_name
      ) AS min_data ON data.q_name = min_data.q_name
          AND r_avg_duration_s = min_duration
   ORDER BY data.q_suite,
            CAST(rtrim(substr(data.q_name, 2, 2), '_') AS INTEGER),
            ltrim(substr(data.q_name, 4), '_'),
            data.d_scale_factor,
            data.e_id
            ;

-- Failed experiments along with their GQE parameters.
CREATE VIEW failed_experiments AS
  SELECT *
    FROM experiment
         JOIN gqe_parameters ON experiment.e_parameters_id = gqe_parameters.p_id
         JOIN data_info ON d_id = e_data_info_id
         JOIN gqe_data_info_ext ON de_id = e_data_info_ext_id
         JOIN query_info ON e_query_info_id = q_id
      WHERE experiment.e_id NOT IN
            (SELECT run.r_experiment_id
               FROM run)
            ;

-- GQE Compression statistics view.
--
-- A denormalized view that joins table-level and column-level statistics to provide
-- a complete picture of compression effectiveness for each column in each table used
-- by a query. This view combines metadata from gqe_table_stats (table context like
-- row count and table name) with detailed compression metrics from gqe_column_stats
-- (compressed/uncompressed sizes, compression ratios...).
--
-- This column-level granularity is useful for analyzing which specific columns benefit
-- most from compression and understanding the storage characteristics of the data
-- processed by each query.
CREATE VIEW gqe_compression_stats AS
  SELECT
    ts_data_info_ext_id,
    ts_table_name,
    ts_rows,
    cs_column_name,
    cs_compressed_size,
    cs_uncompressed_size,
    cs_compression_ratio,
    cs_slices,
    cs_compressed_slices,
    q_name,
    q_suite,
    q_source,
    d_scale_factor
    FROM gqe_table_stats ts
    JOIN gqe_column_stats cs ON ts.ts_id = cs.cs_gqe_table_stats_id
    JOIN query_info q ON ts.ts_query_info_id = q.q_id
    JOIN gqe_data_info_ext de ON ts.ts_data_info_ext_id = de.de_id
    JOIN data_info d ON de.de_data_info_id = d.d_id
                ;

-- GQE Compression statistics per table.
--
-- An aggregated view that rolls up column-level compression statistics to the table
-- level for each query. This view aggregates the stats across
-- all columns in each table to provide table-wide compression metrics.
--
-- Helpful for understanding the overall compression efficiency of each table used in a
-- query, identifying the compression effectiveness across different tables.
CREATE VIEW gqe_compression_stats_per_table AS
  SELECT
    ts_data_info_ext_id,
    ts_table_name,
    q_name,
    q_suite,
    q_source,
    d_scale_factor,
    SUM(cs_compressed_size) AS total_compressed_size,
    SUM(cs_uncompressed_size) AS total_uncompressed_size,
    CAST(SUM(cs_uncompressed_size) AS REAL) / SUM(cs_compressed_size) AS avg_compression_ratio,
    SUM(cs_slices) AS total_slices,
    SUM(cs_compressed_slices) AS total_compressed_slices
    FROM gqe_compression_stats
    GROUP BY ts_data_info_ext_id,
             ts_table_name,
             q_name,
             q_suite,
             q_source,
             d_scale_factor
                ;

-- GQE Compression statistics per data configuration.
--
-- A highly aggregated view that rolls up table-level compression statistics to the
-- data configuration level for each query. This view aggregates compression stats across
-- all tables used by a query to provide overall compression effectiveness for the
-- entire query's data footprint.
--
-- Helpful for comparing different data configurations (compression formats, chunk sizes,
-- etc.) and understanding the total compression efficiency of a
-- complete query execution across all its tables based on different data configurations.
CREATE VIEW gqe_compression_stats_per_data_info AS
  SELECT DISTINCT
    ts_data_info_ext_id,
    q_name,
    q_suite,
    d_scale_factor,
    SUM(cs_compressed_size) AS total_compressed_size,
    SUM(cs_uncompressed_size) AS total_uncompressed_size,
    CAST(SUM(cs_uncompressed_size) AS REAL) / SUM(cs_compressed_size) AS avg_compression_ratio,
    SUM(cs_slices) AS total_slices,
    SUM(cs_compressed_slices) AS total_compressed_slices
    FROM gqe_compression_stats
    GROUP BY ts_data_info_ext_id,
             q_name,
             q_suite,
             q_source,
             d_scale_factor
                ;

COMMIT TRANSACTION;
