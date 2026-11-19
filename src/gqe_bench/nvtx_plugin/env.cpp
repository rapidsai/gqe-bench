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

#include <nvtx_plugin/env.hpp>

#include <cstdlib>
#include <sstream>
#include <string>
#include <strings.h>
#include <vector>

namespace gqe_bench {

bool env_flag(char const* name)
{
  char const* v = std::getenv(name);
  if (!v || !*v) return false;
  return strcasecmp(v, "1") == 0 || strcasecmp(v, "true") == 0 || strcasecmp(v, "yes") == 0;
}

std::vector<std::string> env_str_list(char const* name)
{
  std::vector<std::string> out;
  char const* v = std::getenv(name);
  if (!v || !*v) return out;

  std::stringstream ss(v);
  std::string token;
  while (std::getline(ss, token, ',')) {
    size_t start = token.find_first_not_of(" \t");
    size_t end   = token.find_last_not_of(" \t");
    if (start == std::string::npos) continue;
    out.emplace_back(token.substr(start, end - start + 1));
  }
  return out;
}

}  // namespace gqe_bench
