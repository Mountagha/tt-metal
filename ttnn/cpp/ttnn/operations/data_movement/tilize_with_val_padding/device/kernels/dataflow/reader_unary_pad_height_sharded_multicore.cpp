// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

template <uint32_t val_size>
FORCE_INLINE void fill_with_val(uint32_t start_addr, uint32_t n_bytes, uint32_t val) {
    static_assert(val_size == 2 || val_size == 4, "Unsupported val_size");
    using IntType = std::conditional_t<(val_size == 2), uint16_t, uint32_t>;

    uint32_t end_addr = start_addr + n_bytes;
    uint32_t start_addr_4B = (start_addr + 0x3) & 0xFFFFFFFC;
    uint32_t end_addr_4B = end_addr & 0xFFFFFFFC;

    // 4B writes
    auto* start_ptr_4B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_addr_4B);
    auto* end_ptr_4B =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*> for (auto* p = start_ptr_4B; p < end_addr_4B; ++p) {
        *p = val;
    }

    if constexpr (val_size < 4) {
        auto* start_ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(start_addr);
        auto* end_ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(end_addr);
        auto* start_ptr_a = reinterpret_cast<volatile tt_l1_ptr IntType*>(start_addr_4B);
        auto* end_ptr_a = reinterpret_cast<volatile tt_l1_ptr IntType*>(end_addr_4B);
        IntType val_ = static_cast<IntType>(val);

        for (auto* p = start_ptr; p < start_ptr_a; ++p) {
            *p = val_;
        }
        for (auto* p = end_ptr_a; p < end_ptr; ++p) {
            *p = val_;
        }
    }
}

void kernel_main() {
    // CT
    constexpr uint32_t cb_id_0 = get_compile_time_arg_val(0);
    constexpr uint32_t element_size = get_compile_time_arg_val(1);
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);
    constexpr uint32_t tile_w = get_compile_time_arg_val(3);

    // RT
    uint32_t rt = 0;
    const uint32_t src_base_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t logical_width = get_arg_val<uint32_t>(rt++);  // this core's logical width (in elements)
    const uint32_t padded_width = get_arg_val<uint32_t>(rt++);
    const uint32_t logical_height_core = get_arg_val<uint32_t>(rt++);  // this core's real height (rows)
}
