/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <utility/config.hpp>

#include <cuco/types.cuh>

#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>

namespace gqe_python {
namespace utility {

/**
 * @brief The hash map key sentinel.
 *
 * Chosen to avoid overlap with the primary key domain of TPC-H. Very large tables could technically
 * reach this value for `int32_t`, but in those cases we typically switch to `int64_t` to avoid
 * integer overflow.
 *
 * @invariant The key domain is `[std::numeric_limits<T>::min(), std::numeric_limits<T>::max())`.
 * Note the exclusive upper bound.
 */
template <typename T>
constexpr cuco::empty_key<T> empty_key_sentinel(std::numeric_limits<T>::max());

/**
 * @brief The hash map payload sentinel.
 *
 * Chosen to trigger a clear error (e.g., segfault, invalid result, or similar) if the bucket is
 * used after initialization but without an insert. The operator's build function must take care to
 * insert an actual value.
 */
constexpr cuco::empty_value<cudf::size_type> empty_value_sentinel(-0xDEADBEEF);

/**
 * @brief A hash map wrapper to abstract the identifier type.
 */
struct abstract_hash_map {
  // `dynamic_cast` needs a vtable, thus add a virtual function.
  virtual ~abstract_hash_map() = default;
};

/**
 * @brief A hash map wrapper with a concrete identifier type.
 *
 * The container creates a CuCo hash map on construction.
 */
template <typename Identifier>
struct hash_map_instance : public abstract_hash_map {
  using groupjoin_map_type = utility::map_type<Identifier, cudf::size_type>;

  hash_map_instance(size_t cardinality_estimate, double load_factor)
    : _hash_map(cardinality_estimate,
                load_factor,
                empty_key_sentinel<Identifier>,
                empty_value_sentinel,
                {},
                {},
                {},
                {},
                utility::map_allocator_type<Identifier, cudf::size_type>{
                  utility::map_allocator_instance_type<Identifier, cudf::size_type>{},
                  rmm::cuda_stream_default})
  {
  }

  groupjoin_map_type _hash_map;
};

/**
 * @brief A hash map wrapper containing a singleton hash map instance.
 *
 * The task hash map is shared among the build, probe, and retrieve tasks. The first call to `get()`
 * atomically instantiates the hash map. Later calls to `get()` return this singleton hash map
 * instance.
 */
class task_hash_map {
 public:
  /**
   * @brief Creates the hash map wrapper with parameters used to instantiate the hash map.
   *
   * @arg cardinality_estimate The cardinality used to size the hash map.
   * @arg load_factor The load factor of the hash map.
   */
  explicit task_hash_map(size_t cardinality_estimate, double load_factor = 0.5)
    : _cardinality_estimate(cardinality_estimate),
      _load_factor(load_factor),
      _identifier_type(cudf::type_id::EMPTY)
  {
  }

  /**
   * @brief Return the cardinality estimate.
   */
  size_t cardinality_estimate() const noexcept { return _cardinality_estimate; }

  /**
   * @brief Return the cuDF type of the identifier.
   *
   * @pre Only valid after the first call to `get()`.
   */
  cudf::data_type identifier_type() const noexcept { return _identifier_type; }

  /**
   * @brief Return the singleton hash map instance.
   */
  template <typename Identifier>
  typename hash_map_instance<Identifier>::groupjoin_map_type& get()
  {
    auto& hash_map        = _hash_map;
    auto& identifier_type = _identifier_type;
    auto cardinality      = _cardinality_estimate;
    auto load_factor      = _load_factor;

    std::call_once(
      _is_hash_map_initialized, [&hash_map, &identifier_type, cardinality, load_factor]() {
        identifier_type = cudf::data_type(cudf::type_to_id<Identifier>());
        hash_map        = std::make_unique<hash_map_instance<Identifier>>(cardinality, load_factor);
      });

    return dynamic_cast<hash_map_instance<Identifier>*>(_hash_map.get())->_hash_map;
  }

 private:
  size_t _cardinality_estimate;                  /// The cardinality used to size the hash map.
  double _load_factor;                           /// The hash map load factor.
  cudf::data_type _identifier_type;              /// The identifier type used for the hash map key.
  std::once_flag _is_hash_map_initialized;       /// Used for initializing the hash map.
  std::unique_ptr<abstract_hash_map> _hash_map;  /// The internal hash map instance.
};

}  // namespace utility
}  // namespace gqe_python
