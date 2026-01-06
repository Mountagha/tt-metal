// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_single_core_sharded_program_factory.hpp"

#include <cmath>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::tilize_with_val_padding::program {

TilizeWithValPaddingSingleCoreShardedFactory::cached_program_t TilizeWithValPaddingSingleCoreShardedFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const Tensor& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;

    // Single core at (0, 0)
    CoreRange core({0, 0}, {0, 0});

    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32;

    int32_t num_tiles = output.physical_volume() / TILE_HW;

    auto true_input_shape = a.padded_shape();
    auto true_output_shape = output.padded_shape();

    auto input_y = true_input_shape.rank() >= 2 ? true_input_shape[-2] : 1;
    auto input_x = true_input_shape[-1];

    auto output_y = true_output_shape.rank() >= 2 ? true_output_shape[-2] : 1;
    auto output_x = true_output_shape[-1];

    uint32_t num_tiles_in_row = output_x / TILE_WIDTH;
    uint32_t num_tiles_per_block = num_tiles_in_row;  // Process full rows

    // Create circular buffers
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    tt::tt_metal::CircularBufferConfig src0_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * input_single_tile_size, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, input_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, src0_cb_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = num_tiles_per_block;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    uint32_t packed_pad_value = detail::get_packed_value(a, operation_attributes.pad_value);

    // Determine if input/output is sharded
    bool input_is_sharded = a.is_sharded();
    bool output_is_sharded = output.is_sharded();

    // Conditional defines for kernels
    std::map<std::string, std::string> reader_defines;
    if (input_is_sharded) {
        reader_defines["SHARDED"] = "1";
    }

    std::map<std::string, std::string> writer_defines;
    if (output_is_sharded) {
        writer_defines["SHARDED"] = "1";
    }

    // Reader compile-time args
    uint32_t tile_bytes = output_single_tile_size;
    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index,
        tile_bytes,
        input_y,
        input_x,
        output_y,
        output_x,
        packed_pad_value,
    };
    if (input_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(a, reader_compile_time_args);
    }

    // Reader runtime args
    std::vector<uint32_t> reader_kernel_args = {src0_buffer->address(), (uint32_t)num_tiles, 0};
    if (input_is_sharded) {
        shard_builder::extend_sharding_run_time_args(a, reader_kernel_args);
    }

    // Tilized reader for sharded input
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
        "reader_unary_pad_dims_sharded.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    // Writer compile-time args
    std::vector<uint32_t> writer_compile_time_args = {output_cb_index, tile_bytes};
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output, writer_compile_time_args);
    }

    // Writer runtime args
    std::vector<uint32_t> writer_kernel_args = {dst_buffer->address(), (uint32_t)num_tiles, 0};
    if (output_is_sharded) {
        shard_builder::extend_sharding_run_time_args(output, writer_kernel_args);
    }

    // Tilized writer for sharded output
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
        "writer_unary_sharded.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    // Compute kernel
    std::vector<uint32_t> compute_kernel_args = {
        uint32_t(num_tiles / num_tiles_per_block), uint32_t(num_tiles_per_block)};

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
        core,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_llk_acc,
            .compile_args = compute_kernel_args,
        });

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);
    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);

    return cached_program_t(
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = unary_reader_kernel_id, .writer_kernel_id = unary_writer_kernel_id, .core = core});
}

void TilizeWithValPaddingSingleCoreShardedFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    const auto& core = shared_variables.core;

    auto* src_buffer = tensor_args.input_tensor.buffer();
    auto* dst_buffer = output.buffer();

    CoreCoord core_0 = corerange_to_cores(core).at(0);

    {
        auto& runtime_args = GetRuntimeArgs(program, shared_variables.reader_kernel_id, core_0);
        runtime_args[0] = src_buffer->address();
    }

    {
        auto& runtime_args = GetRuntimeArgs(program, shared_variables.writer_kernel_id, core_0);
        runtime_args[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::data_movement::tilize_with_val_padding::program
