#include "table_definitions.hpp"
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

std::unordered_map<std::string, size_t> getRowCounts()
{
  return {{"lineitem", 6001215000},
          {"orders", 1500300000},
          {"part", 200000000},
          {"partsupp", 800000000},
          {"customer", 150000000},
          {"supplier", 10000000},
          {"nation", 25},
          {"region", 5}};
}

std::unordered_map<std::string, size_t> getTypeSizes()
{
  return {{"identifier_type", 4},
          {"integer_type", 4},
          {"decimal_type", 8},
          {"date_type", 4},
          {"string_type", 25}};
}

size_t calculateMemoryRequirements(
  const std::unordered_map<std::string, std::vector<tpch::column_definition_type>>& definitions)
{
  auto typeSizes = getTypeSizes();
  auto rowCounts = getRowCounts();

  size_t totalMemory = 0;

  for (const auto& table : definitions) {
    const auto& tableName = table.first;
    const auto& columns   = table.second;
    size_t tableSize      = rowCounts[tableName];
    size_t rowSize        = 0;

    for (const auto& column : columns) {
      const auto& colType = column.second;
      rowSize += typeSizes[colType];
    }

    totalMemory += tableSize * rowSize;
  }

  return totalMemory;
}

std::unordered_map<std::string, std::vector<tpch::column_definition_type>> query_table_definitions(
  int query_idx);

void estimateMemoryForAllQueries()
{
  for (int query_idx = 1; query_idx <= 22; ++query_idx) {
    auto definitions    = query_table_definitions(query_idx);
    size_t memoryNeeded = calculateMemoryRequirements(definitions);
    std::cout << "Query " << query_idx
              << " Memory needed: " << static_cast<double>(memoryNeeded) / (1024 * 1024 * 1024)
              << " GB" << std::endl;
  }
}

int main()
{
  estimateMemoryForAllQueries();
  return 0;
}
