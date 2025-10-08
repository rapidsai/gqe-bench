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
  de_compression_data_type TEXT NOT NULL,
  de_compression_chunk_size INTEGER NOT NULL,
  de_zone_map_partition_size INTEGER NOT NULL,
  UNIQUE (
    de_data_info_id,
    de_num_row_groups,
    de_compression_format,
    de_compression_data_type,
    de_compression_chunk_size,
    de_zone_map_partition_size
  ),
  FOREIGN KEY (de_data_info_id) REFERENCES data_info(d_id)
);

-- Add reference to gqe_data_info_ext in experiment table.
ALTER TABLE experiment ADD COLUMN e_data_info_ext_id INTEGER NOT NULL REFERENCES gqe_data_info_ext(de_id);

-- A unified view of general and GQE-extended data information.
CREATE VIEW gqe_data_info AS
  SELECT *
    FROM data_info
         JOIN gqe_data_info_ext ON d_id = de_data_info_id
                ;

-- A short description of a GQE run.
--
-- A virtual view that joins runs with their experiment description and
-- parameters.
CREATE VIEW gqe_run_parameters AS
  SELECT e_suite, e_name, e_scale_factor, run.*, gqe_parameters.*, gqe_data_info_ext.*
    FROM run
         JOIN experiment ON r_experiment_id = e_id
         JOIN gqe_parameters ON gqe_parameters.p_id = e_parameters_id
             AND gqe_parameters.p_sut_info_id = e_sut_info_id
         JOIN gqe_data_info_ext ON e_data_info_ext_id = gqe_data_info_ext.de_id
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
      e_name,
      e_suite,
      e_scale_factor,
      e_query_source,
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
      de_data_info_id,
      de_num_row_groups,
      de_compression_format,
      de_compression_data_type,
      de_compression_chunk_size,
      de_zone_map_partition_size,
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
     WHERE r_number > 0
     GROUP BY
      e_id,
      e_name,
      e_suite,
      e_scale_factor,
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
      de_data_info_id,
      de_num_row_groups,
      de_compression_format,
      de_compression_data_type,
      de_compression_chunk_size,
      de_zone_map_partition_size
  )
  SELECT
    data.*
    FROM
      data
      JOIN (
        SELECT
          e_name,
          min(r_avg_duration_s) AS min_duration
          FROM
            data
         GROUP BY
      e_name
      ) AS min_data ON data.e_name = min_data.e_name
          AND r_avg_duration_s = min_duration
   ORDER BY data.e_suite,
            CAST(rtrim(substr(data.e_name, 2, 2), '_') AS INTEGER),
            ltrim(substr(data.e_name, 4), '_'),
            data.e_scale_factor,
            data.e_id
            ;

-- Failed experiments along with their GQE parameters.
CREATE VIEW failed_experiments AS
  SELECT *
    FROM experiment
         JOIN gqe_parameters ON experiment.e_parameters_id = gqe_parameters.p_id
         JOIN data_info ON d_id = e_data_info_id
         JOIN gqe_data_info_ext ON de_id = e_data_info_ext_id
      WHERE experiment.e_id NOT IN
            (SELECT run.r_experiment_id
               FROM run)
            ;

COMMIT TRANSACTION;
