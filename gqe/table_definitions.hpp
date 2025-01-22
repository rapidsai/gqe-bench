/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/utility/tpch.hpp>

namespace tpch = gqe::utility::tpch;

std::unordered_map<std::string, std::vector<tpch::column_definition_type>> query_table_definitions(
  int query_idx)
{
  std::unordered_map<std::string, std::vector<tpch::column_definition_type>> definitions;
  switch (query_idx) {
    case 0: definitions = tpch::table_definitions(); break;
    case 1:
      definitions = {{"lineitem",
                      {{"l_returnflag", tpch::string_type},
                       {"l_linestatus", tpch::string_type},
                       {"l_quantity", tpch::decimal_type},
                       {"l_extendedprice", tpch::decimal_type},
                       {"l_discount", tpch::decimal_type},
                       {"l_tax", tpch::decimal_type},
                       {"l_shipdate", tpch::date_type}}}};
      break;
    case 2:
      definitions = {
        {"part",
         {{"p_partkey", tpch::identifier_type},
          {"p_size", tpch::integer_type},
          {"p_type", tpch::string_type},
          {"p_mfgr", tpch::string_type}}},
        {"supplier",
         {{"s_suppkey", tpch::identifier_type},
          {"s_nationkey", tpch::identifier_type},
          {"s_acctbal", tpch::decimal_type},
          {"s_name", tpch::string_type},
          {"s_address", tpch::string_type},
          {"s_phone", tpch::string_type},
          {"s_comment", tpch::string_type}}},
        {"partsupp",
         {{"ps_suppkey", tpch::identifier_type},
          {"ps_partkey", tpch::identifier_type},
          {"ps_supplycost", tpch::decimal_type}}},
        {"nation",
         {{"n_name", tpch::string_type},
          {"n_nationkey", tpch::identifier_type},
          {"n_regionkey", tpch::identifier_type}}},
        {"region", {{"r_name", tpch::string_type}, {"r_regionkey", tpch::identifier_type}}}};
      break;
    case 3:
      definitions = {
        {"customer", {{"c_custkey", tpch::identifier_type}, {"c_mktsegment", tpch::string_type}}},
        {"orders",
         {{"o_orderkey", tpch::identifier_type},
          {"o_orderdate", tpch::date_type},
          {"o_shippriority", tpch::integer_type},
          {"o_custkey", tpch::identifier_type}}},
        {"lineitem",
         {{"l_orderkey", tpch::identifier_type},
          {"l_extendedprice", tpch::decimal_type},
          {"l_discount", tpch::decimal_type},
          {"l_shipdate", tpch::date_type}}}};
      break;
    case 4:
      definitions = {{"orders",
                      {{"o_orderkey", tpch::identifier_type},
                       {"o_orderpriority", tpch::string_type},
                       {"o_orderdate", tpch::date_type}}},
                     {"lineitem",
                      {{"l_orderkey", tpch::identifier_type},
                       {"l_commitdate", tpch::date_type},
                       {"l_receiptdate", tpch::date_type}}}};
      break;
    case 5:
      definitions = {
        {"customer",
         {{"c_custkey", tpch::identifier_type}, {"c_nationkey", tpch::identifier_type}}},
        {"orders",
         {{"o_orderkey", tpch::identifier_type},
          {"o_custkey", tpch::identifier_type},
          {"o_orderdate", tpch::date_type}}},
        {"lineitem",
         {{"l_orderkey", tpch::identifier_type},
          {"l_extendedprice", tpch::decimal_type},
          {"l_discount", tpch::decimal_type},
          {"l_suppkey", tpch::identifier_type}}},
        {"supplier",
         {{"s_suppkey", tpch::identifier_type}, {"s_nationkey", tpch::identifier_type}}},
        {"nation",
         {{"n_name", tpch::string_type},
          {"n_nationkey", tpch::identifier_type},
          {"n_regionkey", tpch::identifier_type}}},
        {"region", {{"r_regionkey", tpch::identifier_type}, {"r_name", tpch::string_type}}}};
      break;
    case 6:
      definitions = {{"lineitem",
                      {{"l_extendedprice", tpch::decimal_type},
                       {"l_discount", tpch::decimal_type},
                       {"l_shipdate", tpch::date_type},
                       {"l_quantity", tpch::decimal_type}}}};
      break;
    case 7:
      definitions = {
        {"supplier",
         {{"s_suppkey", tpch::identifier_type}, {"s_nationkey", tpch::identifier_type}}},
        {"lineitem",
         {{"l_suppkey", tpch::identifier_type},
          {"l_orderkey", tpch::identifier_type},
          {"l_extendedprice", tpch::decimal_type},
          {"l_discount", tpch::decimal_type},
          {"l_shipdate", tpch::date_type}}},
        {"orders", {{"o_orderkey", tpch::identifier_type}, {"o_custkey", tpch::identifier_type}}},
        {"customer",
         {{"c_custkey", tpch::identifier_type}, {"c_nationkey", tpch::identifier_type}}},
        {"nation",
         {{"n_name", tpch::string_type},
          {"n_nationkey", tpch::identifier_type},
          {"n_regionkey", tpch::identifier_type}}}};
      break;
    case 8:
      definitions = {
        {"part", {{"p_partkey", tpch::identifier_type}, {"p_type", tpch::string_type}}},
        {"supplier",
         {{"s_suppkey", tpch::identifier_type}, {"s_nationkey", tpch::identifier_type}}},
        {"lineitem",
         {{"l_partkey", tpch::identifier_type},
          {"l_suppkey", tpch::identifier_type},
          {"l_orderkey", tpch::identifier_type},
          {"l_extendedprice", tpch::decimal_type},
          {"l_discount", tpch::decimal_type}}},
        {"orders",
         {{"o_orderkey", tpch::identifier_type},
          {"o_custkey", tpch::identifier_type},
          {"o_orderdate", tpch::date_type}}},
        {"customer",
         {{"c_custkey", tpch::identifier_type}, {"c_nationkey", tpch::identifier_type}}},
        {"nation",
         {{"n_name", tpch::string_type},
          {"n_nationkey", tpch::identifier_type},
          {"n_regionkey", tpch::identifier_type}}},
        {"region", {{"r_name", tpch::string_type}, {"r_regionkey", tpch::identifier_type}}}};
      break;
    case 9:
      definitions = {
        {"part", {{"p_name", tpch::string_type}, {"p_partkey", tpch::identifier_type}}},
        {"lineitem",
         {{"l_partkey", tpch::identifier_type},
          {"l_suppkey", tpch::identifier_type},
          {"l_orderkey", tpch::identifier_type},
          {"l_extendedprice", tpch::decimal_type},
          {"l_discount", tpch::decimal_type},
          {"l_quantity", tpch::decimal_type},
          {"l_shipdate", tpch::date_type}}},
        {"orders", {{"o_orderkey", tpch::identifier_type}, {"o_orderdate", tpch::date_type}}},
        {"supplier",
         {{"s_suppkey", tpch::identifier_type}, {"s_nationkey", tpch::identifier_type}}},
        {"nation", {{"n_name", tpch::string_type}, {"n_nationkey", tpch::identifier_type}}},
        {"partsupp",
         {{"ps_suppkey", tpch::identifier_type},
          {"ps_partkey", tpch::identifier_type},
          {"ps_supplycost", tpch::decimal_type}}}};
      break;
    case 10:
      definitions = {
        {"customer",
         {{"c_custkey", tpch::identifier_type},
          {"c_name", tpch::string_type},
          {"c_acctbal", tpch::decimal_type},
          {"c_phone", tpch::string_type},
          {"c_address", tpch::string_type},
          {"c_comment", tpch::string_type},
          {"c_nationkey", tpch::identifier_type}}},
        {"orders",
         {{"o_orderkey", tpch::identifier_type},
          {"o_custkey", tpch::identifier_type},
          {"o_orderdate", tpch::date_type},
          {"o_totalprice", tpch::decimal_type}}},
        {"lineitem",
         {{"l_orderkey", tpch::identifier_type},
          {"l_returnflag", tpch::string_type},
          {"l_extendedprice", tpch::decimal_type},
          {"l_discount", tpch::decimal_type}}},
        {"nation", {{"n_nationkey", tpch::identifier_type}, {"n_name", tpch::string_type}}}};
      break;
    case 11:
      definitions = {
        {"partsupp",
         {{"ps_partkey", tpch::identifier_type},
          {"ps_suppkey", tpch::identifier_type},
          {"ps_supplycost", tpch::decimal_type},
          {"ps_availqty", tpch::integer_type}}},
        {"supplier",
         {{"s_suppkey", tpch::identifier_type}, {"s_nationkey", tpch::identifier_type}}},
        {"nation", {{"n_name", tpch::string_type}, {"n_nationkey", tpch::identifier_type}}}};
      break;
    case 12:
      definitions = {{"orders",
                      {{"o_orderkey", tpch::identifier_type},
                       {"o_orderpriority", tpch::string_type},
                       {"o_orderdate", tpch::date_type}}},
                     {"lineitem",
                      {{"l_orderkey", tpch::identifier_type},
                       {"l_commitdate", tpch::date_type},
                       {"l_receiptdate", tpch::date_type},
                       {"l_shipdate", tpch::date_type},
                       {"l_shipmode", tpch::string_type}}}};
      break;
    case 13:
      definitions = {{"customer", {{"c_custkey", tpch::identifier_type}}},
                     {"orders",
                      {{"o_orderkey", tpch::identifier_type},
                       {"o_custkey", tpch::identifier_type},
                       {"o_comment", tpch::string_type}}}};
      break;
    case 14:
      definitions = {
        {"lineitem",
         {{"l_partkey", tpch::identifier_type},
          {"l_extendedprice", tpch::decimal_type},
          {"l_discount", tpch::decimal_type},
          {"l_shipdate", tpch::date_type}}},
        {"part", {{"p_partkey", tpch::identifier_type}, {"p_type", tpch::string_type}}}};
      break;
    case 15:
      definitions = {{"lineitem",
                      {{"l_suppkey", tpch::identifier_type},
                       {"l_extendedprice", tpch::decimal_type},
                       {"l_discount", tpch::decimal_type},
                       {"l_shipdate", tpch::date_type}}},
                     {"supplier",
                      {
                        {"s_suppkey", tpch::identifier_type},
                        {"s_name", tpch::string_type},
                        {"s_address", tpch::string_type},
                        {"s_phone", tpch::string_type},
                      }}};
      break;
    case 16:
      definitions = {
        {"part",
         {{"p_brand", tpch::string_type},
          {"p_type", tpch::string_type},
          {"p_size", tpch::integer_type},
          {"p_partkey", tpch::identifier_type}}},
        {"partsupp",
         {{"ps_partkey", tpch::identifier_type}, {"ps_suppkey", tpch::identifier_type}}},
        {"supplier", {{"s_suppkey", tpch::identifier_type}, {"s_comment", tpch::string_type}}}};
      break;
    case 17:
      definitions = {{"lineitem",
                      {{"l_partkey", tpch::identifier_type},
                       {"l_extendedprice", tpch::decimal_type},
                       {"l_quantity", tpch::decimal_type}}},
                     {"part",
                      {{"p_partkey", tpch::identifier_type},
                       {"p_brand", tpch::string_type},
                       {"p_container", tpch::string_type}}}};
      break;
    case 18:
      definitions = {
        {"customer", {{"c_custkey", tpch::identifier_type}, {"c_name", tpch::string_type}}},
        {"orders",
         {{"o_orderkey", tpch::identifier_type},
          {"o_custkey", tpch::identifier_type},
          {"o_orderdate", tpch::date_type},
          {"o_totalprice", tpch::decimal_type}}},
        {"lineitem", {{"l_orderkey", tpch::identifier_type}, {"l_quantity", tpch::decimal_type}}}};
      break;
    case 19:
      definitions = {{"lineitem",
                      {{"l_partkey", tpch::identifier_type},
                       {"l_extendedprice", tpch::decimal_type},
                       {"l_discount", tpch::decimal_type},
                       {"l_quantity", tpch::decimal_type},
                       {"l_shipmode", tpch::string_type},
                       {"l_shipinstruct", tpch::string_type}}},
                     {"part",
                      {{"p_partkey", tpch::identifier_type},
                       {"p_brand", tpch::string_type},
                       {"p_container", tpch::string_type},
                       {"p_size", tpch::integer_type}}}};
      break;
    case 20:
      definitions = {
        {"part", {{"p_partkey", tpch::identifier_type}, {"p_name", tpch::string_type}}},
        {"supplier",
         {{"s_suppkey", tpch::identifier_type},
          {"s_nationkey", tpch::identifier_type},
          {"s_name", tpch::string_type},
          {"s_address", tpch::string_type}}},
        {"partsupp",
         {{"ps_partkey", tpch::identifier_type},
          {"ps_suppkey", tpch::identifier_type},
          {"ps_availqty", tpch::integer_type}}},
        {"nation", {{"n_name", tpch::string_type}, {"n_nationkey", tpch::identifier_type}}},
        {"lineitem",
         {{"l_partkey", tpch::identifier_type},
          {"l_suppkey", tpch::identifier_type},
          {"l_quantity", tpch::decimal_type},
          {"l_shipdate", tpch::date_type}}}};
      break;
    case 21:
      definitions = {
        {"supplier",
         {{"s_suppkey", tpch::identifier_type},
          {"s_nationkey", tpch::identifier_type},
          {"s_name", tpch::string_type}}},
        {"lineitem",
         {{"l_suppkey", tpch::identifier_type},
          {"l_orderkey", tpch::identifier_type},
          {"l_receiptdate", tpch::date_type},
          {"l_commitdate", tpch::date_type}}},
        {"orders", {{"o_orderkey", tpch::identifier_type}, {"o_orderstatus", tpch::string_type}}},
        {"nation", {{"n_nationkey", tpch::identifier_type}, {"n_name", tpch::string_type}}}};
      break;
    case 22:
      definitions = {{"customer",
                      {{"c_custkey", tpch::identifier_type},
                       {"c_acctbal", tpch::decimal_type},
                       {"c_phone", tpch::string_type},
                       {"c_nationkey", tpch::identifier_type}}},
                     {"orders", {{"o_custkey", tpch::identifier_type}}},
                     {"nation", {{"n_nationkey", tpch::identifier_type}}}};
      break;
    default: throw std::logic_error("No table definition for this query");
  }
  return definitions;
}
