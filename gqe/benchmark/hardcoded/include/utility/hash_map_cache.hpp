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

#pragma once

#include <utility/config.hpp>
#include <utility/utility.hpp>

#include <cuco/types.cuh>

#include <cuco/bloom_filter_policies.cuh>
#include <cuco/bloom_filter_ref.cuh>
#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/allocator.hpp>
#include <cuco/utility/cuda_thread_scope.cuh>

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
 *
 * @note For integral types, uses -0xDEADBEEF. For floating point types, uses -1.0.
 */
template <typename T>
constexpr auto empty_value_sentinel()
{
  if constexpr (std::is_integral_v<T>) {
    return cuco::empty_value<T>(-0xDEADBEEF);
  } else if constexpr (std::is_floating_point_v<T>) {
    return cuco::empty_value<T>(-1.0);
  } else {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                  "empty_value_sentinel only supports integral and floating point types");
  }
}

/**
 * @brief A hash map wrapper to abstract the identifier type.
 */
struct abstract_hash_map {
  // `dynamic_cast` needs a vtable, thus add a virtual function.
  virtual ~abstract_hash_map() = default;
};

/**
 * @brief A bloom filter wrapper to abstract the identifier type.
 */
struct abstract_bloom_filter {
  // `dynamic_cast` needs a vtable, thus add a virtual function.
  virtual ~abstract_bloom_filter() = default;
};

/**
 * @brief A hash map wrapper with a concrete identifier type.
 *
 * The container creates a CuCo hash map and a CuCo bloom filter (if enabled) on construction.
 *
 * @tparam Identifier The key type of the hash map
 * @tparam Value The value type of the hash map
 * @tparam HashMapType The actual CuCo hash map type
 */
template <typename Identifier, typename Value, typename HashMapType>
struct hash_map_instance : public abstract_hash_map {
  using bloom_filter_type = utility::bloom_filter_type<Identifier>;

  hash_map_instance(size_t cardinality_estimate, double load_factor)
    : _hash_map(
        cardinality_estimate,
        load_factor,
        empty_key_sentinel<Identifier>,
        empty_value_sentinel<Value>(),
        {},
        {},
        {},
        {},
        utility::map_allocator_type<Identifier, Value>{
          utility::map_allocator_instance_type<Identifier, Value>{}, cudf::get_default_stream()})
  {
  }

  HashMapType _hash_map;
};

template <typename Identifier, typename Hash = cuco::xxhash_64<Identifier>>
struct bloom_filter_instance : public abstract_bloom_filter {
  bloom_filter_instance(size_t bloom_filter_num_blocks)
    : _bloom_filter(bloom_filter_num_blocks,
                    {},
                    {},
                    utility::bloom_filter_allocator_type{
                      utility::bloom_filter_allocator_instance_type{}, cudf::get_default_stream()})
  {
  }

  utility::bloom_filter_type<Identifier, Hash> _bloom_filter;
};

/**
 * @brief A hash map wrapper containing a singleton hash map instance.
 *
 * The task hash map is shared among different tasks among join.
 *
 * There are two ways to use this hash map cache:
 * 1) Create a task to build the hash map with call to `get()` and populate the hash map by
 * calling related `insert` functions. All the other probe, retrieve tasks after the build task can
 * call `get()` later to get the singleton hash map instance.
 *
 * 2) The hash map cache is initialized during hash map probe tasks.
 * In the build step, `create_map_and_insert()` would be called first to create the singleton hash
 * map and populate it at the same time to guarantee consistency. Then probe steps can call `get()`
 * to get the singleton hash map instance afterwards. It is undefined behavior to call `get_map()`
 * and insert into the hash map at the same time from different threads during the build step.
 */
class task_hash_map {
 public:
  /**
   * @brief Creates the hash map wrapper with parameters used to instantiate the hash map.
   *
   * @arg cardinality_estimate The cardinality used to size the hash map.
   * @arg load_factor The load factor of the hash map.
   */
  explicit task_hash_map(size_t cardinality_estimate,
                         double load_factor       = 0.5,
                         bool enable_bloom_filter = false,
                         // TODO: this was set because = 0 creates an illegal memory access; desires
                         // further investigation into why the default is being used
                         std::size_t num_filter_blocks = 1)
    : _cardinality_estimate(cardinality_estimate),
      _load_factor(load_factor),
      _identifier_type(cudf::type_id::EMPTY),
      _enable_bloom_filter(enable_bloom_filter),
      _num_filter_blocks(num_filter_blocks)
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
   * @brief Return the singleton hash map and bloom filter that are created without populating the
   * hash map and bloom filter.
   * @tparam Identifier The key type of the hash map
   * @tparam Value The value type of the hash map
   * @tparam HashMapType The actual CuCo hash map type
   * @param cardinality The cardinality used to size the hash map.
   * @param load_factor The load factor of the hash map.
   */
  template <typename Identifier, typename Value, typename HashMapType>
  HashMapType& get_map(size_t cardinality, double load_factor)
  {
    auto& hash_map        = _hash_map;
    auto& identifier_type = _identifier_type;

    std::call_once(_is_hash_map_initialized,
                   [&hash_map, &identifier_type, cardinality, load_factor]() {
                     identifier_type = cudf::data_type(cudf::type_to_id<Identifier>());
                     hash_map = std::make_unique<hash_map_instance<Identifier, Value, HashMapType>>(
                       cardinality, load_factor);
                   });

    return dynamic_cast<hash_map_instance<Identifier, Value, HashMapType>*>(_hash_map.get())
      ->_hash_map;
  }

  template <typename Identifier, typename Hash = cuco::xxhash_64<Identifier>>
  typename utility::bloom_filter_type<Identifier, Hash>& get_bloom_filter(
    std::size_t num_filter_blocks)
  {
    assert(_enable_bloom_filter && "bloom filter is not enabled");
    assert(num_filter_blocks > 0 && "number of filter blocks must be greater than zero");
    auto& bloom_filter = _bloom_filter;

    std::call_once(_is_bloom_filter_initialized, [&bloom_filter, num_filter_blocks]() {
      bloom_filter = std::make_unique<bloom_filter_instance<Identifier, Hash>>(num_filter_blocks);
    });

    return dynamic_cast<bloom_filter_instance<Identifier, Hash>*>(_bloom_filter.get())
      ->_bloom_filter;
  }

  /**
   * @brief Return the singleton hash map instance.
   * @tparam Identifier The key type of the hash map
   * @tparam Value The value type of the hash map
   * @tparam HashMapType The actual CuCo hash map type
   */
  template <typename Identifier, typename Value, typename HashMapType>
  HashMapType& get_map()
  {
    return get_map<Identifier, Value, HashMapType>(_cardinality_estimate, _load_factor);
  }

  template <typename Identifier, typename Hash = cuco::xxhash_64<Identifier>>
  typename utility::bloom_filter_type<Identifier, Hash>& get_bloom_filter()
  {
    return get_bloom_filter<Identifier, Hash>(_num_filter_blocks);
  }

  /**
   * @brief Create the singleton hash map and bloom filter and populate them.
   *
   * @tparam Identifier The key type of the hash map
   * @tparam Value The value type of the hash map
   * @tparam HashMapType The actual CuCo hash map type
   * @tparam InsertFunctor The functor type for inserting elements
   * @param cardinality The cardinality used to size the hash map.
   * @param load_factor The load factor of the hash map.
   * @param insert_functor The functor to insert elements into the hash map.
   */
  template <typename Identifier, typename Value, typename HashMapType, typename InsertFunctor>
  void create_map_and_insert(size_t cardinality, double load_factor, InsertFunctor insert_functor)
  {
    auto& hash_map        = _hash_map;
    auto& identifier_type = _identifier_type;

    std::call_once(
      _is_hash_map_initialized,
      [&hash_map, &identifier_type, cardinality, load_factor, insert_functor]() {
        identifier_type = cudf::data_type(cudf::type_to_id<Identifier>());
        hash_map = std::make_unique<hash_map_instance<Identifier, Value, HashMapType>>(cardinality,
                                                                                       load_factor);
        insert_functor(
          dynamic_cast<hash_map_instance<Identifier, Value, HashMapType>*>(hash_map.get())
            ->_hash_map);
      });
  }

  /**
   * @brief Create a map along with a bloom filter and populate them at one call.
   *
   * @tparam Identifier The key type of the hash map
   * @tparam Value The value type of the hash map
   * @tparam HashMapType The actual CuCo hash map type
   * @tparam InsertFunctor The functor type for inserting elements
   * @param cardinality The cardinality used to size the hash map.
   * @param load_factor The load factor of the hash map.
   * @param num_filter_blocks The number of filter blocks for the bloom filter.
   * @param insert_functor The functor to insert elements into the bloom filter.
   */
  template <typename Identifier,
            typename Value,
            typename HashMapType,
            typename InsertFunctor,
            typename BFHash = cuco::xxhash_64<Identifier>>
  void create_map_bf_and_insert(size_t cardinality,
                                double load_factor,
                                std::size_t num_filter_blocks,
                                InsertFunctor insert_functor)
  {
    auto& hash_map                    = _hash_map;
    auto& bloom_filter                = _bloom_filter;
    auto& identifier_type             = _identifier_type;
    auto& is_hash_map_initialized     = _is_hash_map_initialized;
    auto& is_bloom_filter_initialized = _is_bloom_filter_initialized;

    std::call_once(
      _is_both_initialized,
      [&is_hash_map_initialized,
       &is_bloom_filter_initialized,
       &hash_map,
       &bloom_filter,
       &identifier_type,
       cardinality,
       load_factor,
       num_filter_blocks,
       insert_functor]() {
        identifier_type = cudf::data_type(cudf::type_to_id<Identifier>());
        hash_map = std::make_unique<hash_map_instance<Identifier, Value, HashMapType>>(cardinality,
                                                                                       load_factor);
        bloom_filter =
          std::make_unique<bloom_filter_instance<Identifier, BFHash>>(num_filter_blocks);
        insert_functor(
          dynamic_cast<hash_map_instance<Identifier, Value, HashMapType>*>(hash_map.get())
            ->_hash_map,
          dynamic_cast<bloom_filter_instance<Identifier, BFHash>*>(bloom_filter.get())
            ->_bloom_filter);
        std::call_once(is_hash_map_initialized, [] {});
        std::call_once(is_bloom_filter_initialized, [] {});
      });
  }

 private:
  size_t _cardinality_estimate;             ///< The cardinality used to size the hash map.
  double _load_factor;                      ///< The hash map load factor.
  bool _enable_bloom_filter = false;        ///< Whether to enable the bloom filter.
  std::size_t _num_filter_blocks;           ///< The number of filter blocks for the bloom filter.
  cudf::data_type _identifier_type;         ///< The identifier type used for the hash map key.
  std::once_flag _is_hash_map_initialized;  ///< Used for initializing the hash map.
  std::once_flag _is_bloom_filter_initialized;  ///< Used for initializing the bloom filter.
  std::once_flag _is_both_initialized;  ///< Used for initializing both the hash map and bloom
                                        ///< filter. This is used because sometimes hash map insert
                                        ///< and bloom filter insert are finished inside the same
                                        ///< kernels, and thus must be initialized together.
  std::unique_ptr<abstract_hash_map> _hash_map;  ///< The internal hash map instance.
  std::unique_ptr<abstract_bloom_filter>
    _bloom_filter;  ///< The internal bloom filter instance (if enabled).
};

}  // namespace utility
}  // namespace gqe_python
