// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_page_id = get_arg_val<uint32_t>(2);

    // Compile-time args
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t input_y = get_compile_time_arg_val(2);
    constexpr uint32_t input_x = get_compile_time_arg_val(3);
    constexpr uint32_t output_y = get_compile_time_arg_val(4);
    constexpr uint32_t output_x = get_compile_time_arg_val(5);
    constexpr uint32_t packed_pad_value = get_compile_time_arg_val(6);

    constexpr uint32_t tile_height = 32;
    constexpr uint32_t tile_width = 32;

#ifdef SHARDED
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(7),    // Memory layout
        get_compile_time_arg_val(8),    // The number of sharding cores
        get_compile_time_arg_val(9),    // The page size we offset each read from
        get_compile_time_arg_val(10),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(11),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(12),   // pages_per_shard_x
        get_compile_time_arg_val(13)>;  // pages_per_shard_y

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(3));
    experimental::ShardedAddrGen<tensor_shard_info> s = {.bank_base_address = src_addr, .shard_array = mapping_table};
#endif

    // Calculate dimensions
    uint32_t num_tiles_width = output_x / tile_width;
    uint32_t num_tiles_height = output_y / tile_height;
    uint32_t input_row_size_bytes = input_x * 2;  // bfloat16 = 2 bytes
    uint32_t tile_row_size_bytes = tile_width * 2;

#ifndef SHARDED
    // Setup interleaved address generator for non-sharded case
    const DataFormat data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGen<false> addr_gen = {.bank_base_address = src_addr, .page_size = input_row_size_bytes};
#endif

    // Process tiles row by row
    for (uint32_t tile_row = 0; tile_row < num_tiles_height; ++tile_row) {
        for (uint32_t tile_col = 0; tile_col < num_tiles_width; ++tile_col) {
            cb_reserve_back(cb_id_in0, 1);
            uint32_t write_addr = get_write_ptr(cb_id_in0);

            // Process 32 rows of this tile
            for (uint32_t row_in_tile = 0; row_in_tile < tile_height; ++row_in_tile) {
                uint32_t global_row = tile_row * tile_height + row_in_tile;
                uint32_t tile_col_offset_bytes = tile_col * tile_width * 2;

                if (global_row < input_y && tile_col_offset_bytes < input_row_size_bytes) {
                    // Read actual data from input
                    uint32_t read_size = (tile_col_offset_bytes + tile_row_size_bytes <= input_row_size_bytes)
                                             ? tile_row_size_bytes
                                             : (input_row_size_bytes - tile_col_offset_bytes);
#ifdef SHARDED
                    // For sharded: page_id is the row number, offset is column offset within that row
                    uint32_t page_id = global_row;
                    uint32_t page_offset = tile_col_offset_bytes;
                    uint64_t src_noc_addr = get_noc_addr(page_id, s, page_offset);
#else
                    // For interleaved: use standard address generation
                    uint64_t src_noc_addr = get_noc_addr(global_row, addr_gen, tile_col_offset_bytes);
#endif
                    noc_async_read(src_noc_addr, write_addr, read_size);
                    noc_async_read_barrier();

                    // Pad remaining width if needed
                    if (read_size < tile_row_size_bytes) {
                        uint32_t* pad_ptr = reinterpret_cast<uint32_t*>(write_addr + read_size);
                        for (uint32_t i = 0; i < (tile_row_size_bytes - read_size) / 4; ++i) {
                            pad_ptr[i] = packed_pad_value;
                        }
                    }
                } else {
                    // Pad entire row (beyond input height or width)
                    uint32_t* pad_ptr = reinterpret_cast<uint32_t*>(write_addr);
                    for (uint32_t i = 0; i < tile_row_size_bytes / 4; ++i) {
                        pad_ptr[i] = packed_pad_value;
                    }
                }
                write_addr += tile_row_size_bytes;
            }

            cb_push_back(cb_id_in0, 1);
        }
    }
}
