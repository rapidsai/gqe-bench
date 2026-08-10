/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/**
 * @file nvtx_domain_filter.cpp
 * @brief NVTX injection entry point that forwards only the engine's domain to CUPTI.
 *
 * This file is a workaround for a defect in CUPTI's NVTX attribute interning, described
 * below. It exists only to keep that cost off the plugin's path and is expected to be
 * deleted once CUPTI is fixed.
 *
 * TODO: remove this file when CUPTI's NVTX attribute interning no longer scans its whole
 * intern table per event. To revert: delete this file, drop it from
 * `nvtx_plugin/CMakeLists.txt`, point `configure_nvtx_injection()` back at libcupti (see
 * the TODO there), and drop `k_gqe_nvtx_domain` from `stages.hpp.in` if nothing else has
 * come to use it.
 *
 * `CUPTI_CB_DOMAIN_NVTX` is process-wide and all-or-nothing: with it enabled every NVTX
 * event in the process traps into CUPTI, which interns the event attributes by scanning
 * its whole intern table under a process-global mutex. Producers that attach a varying
 * payload to hot ranges — kvikio tags its reads with the transfer byte count — make
 * every event a distinct entry, so the table grows and the cost becomes quadratic in
 * events.
 *
 * The plugin only observes ranges in the `k_gqe_nvtx_domain` domain. This file installs
 * itself as the process's NVTX injection, chains to CUPTI so CUPTI still sees those
 * ranges (preserving its callbacks and its `CUPTI_ACTIVITY_KIND_MARKER` records), and
 * drops every other domain's events before CUPTI observes them.
 *
 * NVTX reads `NVTX_INJECTION64_PATH`, `dlopen`s it, and calls
 * `InitializeInjectionNvtx2(getExportTable)` on it. An injection installs itself by
 * writing function pointers into slots obtained from the callbacks export table. So the
 * sequence here is: call CUPTI's initializer with the same export table so CUPTI
 * populates the slots exactly as it normally would, read those slots back to capture
 * CUPTI's implementations, then overwrite the attribute-carrying slots with wrappers.
 *
 * NVTX state is per client module: `nvtxGlobals` is hidden-weak, so every executable and
 * shared object that emits NVTX holds its own function table and initialises the
 * injection separately, on its first NVTX call. This entry point therefore runs several
 * times per process, at unrelated moments, including while ranges are open.
 *
 * Only the domain-scoped (CORE2) entry points are wrapped. The non-domain (CORE) entry
 * points operate on the default domain and are left pointing at CUPTI's implementations.
 *
 * This is built on `nvtx3/nvtxDetail` internals, which are not a stable public ABI. The
 * table size is checked before any slot is written, and a slot that is absent leaves
 * CUPTI's implementation in place.
 */

#include "log.hpp"
#include <nvtx_plugin/stages.hpp>

#include <cupti_callbacks.h>
#include <nvtx3/nvToolsExt.h>

#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

namespace gqe_bench {

namespace {

/**
 * @brief Range id returned in place of a filtered `RangeStartEx`.
 *
 * `RangeStartEx` / `RangeEnd` are not stack-paired the way push/pop are, so a filtered
 * start must return an id the matching end can be recognised by. CUPTI issues ids
 * sequentially from a low base, so an id this large cannot be one of its own. Every
 * filtered range gets this same value: they are all dropped, so nothing distinguishes
 * them.
 */
constexpr nvtxRangeId_t k_filtered_range_id = std::uint64_t{1} << 63;

/**
 * @brief CUPTI's implementations, captured from the function table after chaining.
 *
 * Written once during injection init, before any NVTX event can be emitted through the
 * wrappers, and only read afterwards. A null entry means the slot was absent, in which
 * case the corresponding wrapper degrades to dropping the event rather than calling
 * through a null pointer.
 */
struct cupti_impls {
  nvtxDomainCreateA_impl_fntype domain_create_a     = nullptr;
  nvtxDomainCreateW_impl_fntype domain_create_w     = nullptr;
  nvtxDomainDestroy_impl_fntype domain_destroy      = nullptr;
  nvtxDomainMarkEx_impl_fntype mark_ex              = nullptr;
  nvtxDomainRangeStartEx_impl_fntype range_start_ex = nullptr;
  nvtxDomainRangeEnd_impl_fntype range_end          = nullptr;
  nvtxDomainRangePushEx_impl_fntype range_push_ex   = nullptr;
  nvtxDomainRangePop_impl_fntype range_pop          = nullptr;
};

cupti_impls g_cupti;

/// Set when the filter reports itself, so the report is emitted once per process rather
/// than once per NVTX client module.
std::atomic<bool> g_announced{false};

/**
 * @brief Handle of the forwarded domain, learned when it is created.
 *
 * Null until then, which is correct: no event in a domain can exist before the domain.
 */
std::atomic<nvtxDomainHandle_t> g_forwarded_domain{nullptr};

/**
 * @brief Per-thread record of whether each open push was forwarded, so the matching pop
 * is forwarded or dropped identically.
 *
 * NVTX push/pop are per-thread by specification, so this needs no synchronisation.
 */
thread_local std::vector<bool> t_push_forwarded;

bool is_forwarded_domain(nvtxDomainHandle_t domain)
{
  return domain != nullptr && domain == g_forwarded_domain.load(std::memory_order_acquire);
}

// The wrappers below replace CUPTI's implementations in the CORE2 function table. Each
// calls through to the captured CUPTI implementation for the forwarded domain and
// otherwise returns without doing so, leaving CUPTI unaware the event occurred. Domain
// creation and destruction always call through; they are wrapped only to track which
// handle identifies the forwarded domain.

nvtxDomainHandle_t NVTX_API filtered_domain_create_a(char const* name)
{
  nvtxDomainHandle_t const handle =
    g_cupti.domain_create_a ? g_cupti.domain_create_a(name) : nullptr;
  if (name && std::strcmp(name, k_gqe_nvtx_domain) == 0) {
    g_forwarded_domain.store(handle, std::memory_order_release);
    GQE_BENCH_LOG_INFO("nvtx filter: forwarding domain '{}'", name);
  }
  return handle;
}

nvtxDomainHandle_t NVTX_API filtered_domain_create_w(wchar_t const* name)
{
  // Wide-character domain names are forwarded but never matched: the engine registers
  // its domain through the narrow API, so a wide name cannot be the forwarded domain.
  return g_cupti.domain_create_w ? g_cupti.domain_create_w(name) : nullptr;
}

void NVTX_API filtered_domain_destroy(nvtxDomainHandle_t domain)
{
  if (is_forwarded_domain(domain)) { g_forwarded_domain.store(nullptr, std::memory_order_release); }
  if (g_cupti.domain_destroy) { g_cupti.domain_destroy(domain); }
}

void NVTX_API filtered_mark_ex(nvtxDomainHandle_t domain, nvtxEventAttributes_t const* attr)
{
  if (is_forwarded_domain(domain) && g_cupti.mark_ex) { g_cupti.mark_ex(domain, attr); }
}

nvtxRangeId_t NVTX_API filtered_range_start_ex(nvtxDomainHandle_t domain,
                                               nvtxEventAttributes_t const* attr)
{
  if (is_forwarded_domain(domain)) {
    return g_cupti.range_start_ex ? g_cupti.range_start_ex(domain, attr) : 0;
  }
  // CUPTI never saw this range, so it must not see the matching end either.
  return k_filtered_range_id;
}

void NVTX_API filtered_range_end(nvtxDomainHandle_t domain, nvtxRangeId_t id)
{
  if (id == k_filtered_range_id) { return; }
  if (g_cupti.range_end) { g_cupti.range_end(domain, id); }
}

int NVTX_API filtered_range_push_ex(nvtxDomainHandle_t domain, nvtxEventAttributes_t const* attr)
{
  bool const forward = is_forwarded_domain(domain);
  t_push_forwarded.push_back(forward);
  if (forward && g_cupti.range_push_ex) { return g_cupti.range_push_ex(domain, attr); }
  // Report this thread's nesting depth so a caller inspecting the return value still
  // observes a monotonic, self-consistent value.
  return static_cast<int>(t_push_forwarded.size()) - 1;
}

int NVTX_API filtered_range_pop(nvtxDomainHandle_t domain)
{
  if (t_push_forwarded.empty()) {
    // An unbalanced pop, or a push that predates this injection. Not ours to interpret.
    return g_cupti.range_pop ? g_cupti.range_pop(domain) : 0;
  }
  bool const was_forwarded = t_push_forwarded.back();
  t_push_forwarded.pop_back();
  if (was_forwarded && g_cupti.range_pop) { return g_cupti.range_pop(domain); }
  return static_cast<int>(t_push_forwarded.size());
}

/**
 * @brief Resolve `InitializeInjectionNvtx2` in the libcupti this library is linked
 * against, or null if it cannot be resolved.
 *
 * `dladdr` on a libcupti symbol names the copy the loader actually bound, so the
 * `dlopen` below re-opens that exact file rather than re-resolving a soname that could
 * select a different one.
 */
NvtxInitializeInjectionNvtxFunc_t resolve_cupti_injection()
{
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void const*>(&cuptiSubscribe), &info) || !info.dli_fname) {
    GQE_BENCH_LOG_ERROR("nvtx filter: dladdr(cuptiSubscribe) failed; cannot chain to CUPTI");
    return nullptr;
  }
  void* const handle = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
  if (!handle) {
    // RTLD_NOLOAD reports "not already loaded" by returning null without
    // necessarily setting an error, so dlerror() can legitimately be null here.
    char const* const err = dlerror();
    GQE_BENCH_LOG_ERROR(
      "nvtx filter: dlopen({}) failed: {}", info.dli_fname, err ? err : "no error reported");
    return nullptr;
  }
  auto* const init =
    reinterpret_cast<NvtxInitializeInjectionNvtxFunc_t>(dlsym(handle, "InitializeInjectionNvtx2"));
  if (!init) {
    GQE_BENCH_LOG_ERROR("nvtx filter: {} has no InitializeInjectionNvtx2", info.dli_fname);
    return nullptr;
  }
  GQE_BENCH_LOG_INFO("nvtx filter: chaining to {}", info.dli_fname);
  return init;
}

/**
 * @brief Capture CUPTI's implementation from slot `cbid` and install `wrapper` over it.
 *
 * Runs once per NVTX client module, and the first module's implementation is the one
 * retained: the wrappers are single addresses shared by every module's table, so they
 * cannot select a per-module implementation. A module whose slot disagrees with the
 * retained one is reported, since forwarding a range into one module's implementation and
 * ending it through another's would corrupt the timings silently.
 *
 * @return False if the slot is absent, leaving CUPTI's implementation in place.
 */
template <typename FnType>
bool install(NvtxFunctionTable table, int cbid, FnType* saved, void* wrapper)
{
  NvtxFunctionPointer* const slot = table[cbid];
  if (!slot) { return false; }
  // This table already carries the wrapper, so the slot is not CUPTI's implementation.
  if (*slot == reinterpret_cast<NvtxFunctionPointer>(wrapper)) { return true; }
  auto const current = reinterpret_cast<FnType>(*slot);
  if (*saved == nullptr) {
    *saved = current;
  } else if (*saved != current) {
    GQE_BENCH_LOG_WARN(
      "nvtx filter: cbid {} differs across NVTX client modules: saved={:#x} new={:#x}",
      cbid,
      reinterpret_cast<std::uintptr_t>(*saved),
      reinterpret_cast<std::uintptr_t>(current));
  }
  *slot = reinterpret_cast<NvtxFunctionPointer>(wrapper);
  return true;
}

}  // namespace

}  // namespace gqe_bench

/**
 * @brief NVTX injection entry point, resolved by NVTX via `dlsym` on the library named
 * by `NVTX_INJECTION64_PATH`.
 *
 * Chains to CUPTI first so CUPTI installs its handlers as it normally would, then wraps
 * the domain-scoped slots to forward only `k_gqe_nvtx_domain`.
 *
 * If CUPTI cannot be chained to, returns 0 to report that no injection was installed. If
 * CUPTI initialises but the wrappers cannot be installed, returns 1 so CUPTI is left in
 * place unfiltered: that is slower but correct, whereas reporting failure at that point
 * would leave NVTX with no tool attached and silently empty the plugin's stage timings.
 *
 * @param get_export_table NVTX's export-table accessor.
 * @return 1 if an injection is installed, 0 otherwise.
 */
extern "C" int NVTX_API InitializeInjectionNvtx2(NvtxGetExportTableFunc_t get_export_table)
{
  using namespace gqe_bench;

  if (!get_export_table) { return 0; }

  auto* const cupti_init = resolve_cupti_injection();
  if (!cupti_init) { return 0; }
  if (!cupti_init(get_export_table)) {
    GQE_BENCH_LOG_ERROR("nvtx filter: CUPTI InitializeInjectionNvtx2 reported failure");
    return 0;
  }

  auto const* const callbacks =
    static_cast<NvtxExportTableCallbacks const*>(get_export_table(NVTX_ETID_CALLBACKS));
  if (!callbacks || !callbacks->GetModuleFunctionTable) {
    GQE_BENCH_LOG_WARN("nvtx filter: callbacks export table unavailable; leaving CUPTI unfiltered");
    return 1;
  }

  NvtxFunctionTable table = nullptr;
  unsigned int size       = 0;
  if (!callbacks->GetModuleFunctionTable(NVTX_CB_MODULE_CORE2, &table, &size) || !table) {
    GQE_BENCH_LOG_WARN("nvtx filter: CORE2 function table unavailable; leaving CUPTI unfiltered");
    return 1;
  }
  if (size <= static_cast<unsigned int>(NVTX_CBID_CORE2_DomainDestroy)) {
    GQE_BENCH_LOG_WARN(
      "nvtx filter: CORE2 table smaller than expected (size={}); leaving CUPTI unfiltered", size);
    return 1;
  }

  // Push/pop carry the range volume that makes this worth doing, and domain creation is
  // how the forwarded domain is identified at all, so a missing slot among these three
  // means filtering would not work.
  bool const essential = install(table,
                                 NVTX_CBID_CORE2_DomainCreateA,
                                 &g_cupti.domain_create_a,
                                 reinterpret_cast<void*>(&filtered_domain_create_a)) &&
                         install(table,
                                 NVTX_CBID_CORE2_DomainRangePushEx,
                                 &g_cupti.range_push_ex,
                                 reinterpret_cast<void*>(&filtered_range_push_ex)) &&
                         install(table,
                                 NVTX_CBID_CORE2_DomainRangePop,
                                 &g_cupti.range_pop,
                                 reinterpret_cast<void*>(&filtered_range_pop));
  if (!essential) {
    GQE_BENCH_LOG_WARN("nvtx filter: a required CORE2 slot was absent; leaving CUPTI unfiltered");
    return 1;
  }

  install(table,
          NVTX_CBID_CORE2_DomainCreateW,
          &g_cupti.domain_create_w,
          reinterpret_cast<void*>(&filtered_domain_create_w));
  install(table,
          NVTX_CBID_CORE2_DomainDestroy,
          &g_cupti.domain_destroy,
          reinterpret_cast<void*>(&filtered_domain_destroy));
  install(table,
          NVTX_CBID_CORE2_DomainMarkEx,
          &g_cupti.mark_ex,
          reinterpret_cast<void*>(&filtered_mark_ex));
  install(table,
          NVTX_CBID_CORE2_DomainRangeStartEx,
          &g_cupti.range_start_ex,
          reinterpret_cast<void*>(&filtered_range_start_ex));
  install(table,
          NVTX_CBID_CORE2_DomainRangeEnd,
          &g_cupti.range_end,
          reinterpret_cast<void*>(&filtered_range_end));

  if (!g_announced.exchange(true, std::memory_order_relaxed)) {
    GQE_BENCH_LOG_WARN(
      "nvtx domain filter ACTIVE (workaround, see nvtx_domain_filter.cpp): NVTX events "
      "outside domain '{}' are dropped before CUPTI sees them pid={}",
      k_gqe_nvtx_domain,
      getpid());
  }
  return 1;
}
