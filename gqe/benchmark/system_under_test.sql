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
  p_join_use_hash_map_cache INTEGER NOT NULL,
  p_read_use_zero_copy INTEGER NOT NULL,
  p_join_use_unique_keys INTEGER NOT NULL,
  -- Each parameter combination should be unique within the experiment database.
  UNIQUE (p_num_workers, p_num_partitions, p_join_use_hash_map_cache, p_read_use_zero_copy, p_join_use_unique_keys),
  FOREIGN KEY (p_sut_info_id) REFERENCES sut_info(s_id)
);

-- A short description of a GQE run.
--
-- A virtual view that joins runs with their experiment description and
-- parameters.
CREATE VIEW gqe_run_parameters AS
  SELECT e_suite, e_name, e_scale_factor, run.*, gqe_parameters.*
    FROM run
         JOIN experiment ON r_experiment_id = e_id
         JOIN gqe_parameters ON gqe_parameters.p_id = e_parameters_id
             AND gqe_parameters.p_sut_info_id = e_sut_info_id
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
  -- FIXME: JOIN build_info ON b_id = e_build_info_id
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
      p_num_workers,
      p_num_partitions,
      p_join_use_hash_map_cache,
      p_read_use_zero_copy,
      p_join_use_unique_keys,
      avg(r_duration_s) AS r_avg_duration_s,
      min(r_duration_s) AS r_min_duration_s,
      max(r_duration_s) AS r_max_duration_s,
      count(r_duration_s) AS sample_size
      FROM
        experiment
        JOIN run ON e_id = r_experiment_id
        JOIN gqe_parameters ON e_parameters_id = p_id
     WHERE r_number > 0
     GROUP BY
      e_id,
      e_name,
      e_suite,
      e_scale_factor,
      p_num_workers,
      p_num_partitions,
      p_join_use_hash_map_cache,
      p_read_use_zero_copy,
      p_join_use_unique_keys
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
            data.e_name,
            data.e_scale_factor,
            data.e_id
            ;

-- Failed experiments along with their GQE parameters.
CREATE VIEW failed_experiments AS
  SELECT *
    FROM experiment
         JOIN gqe_parameters ON experiment.e_parameters_id = gqe_parameters.p_id
      WHERE experiment.e_id NOT IN
            (SELECT run.r_experiment_id
               FROM run)
            ;

COMMIT TRANSACTION;
