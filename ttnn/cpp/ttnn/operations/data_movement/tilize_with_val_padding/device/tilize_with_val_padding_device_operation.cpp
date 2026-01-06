// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_device_operation.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::data_movement {

TilizeWithValPaddingDeviceOperation::program_factory_t TilizeWithValPaddingDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    if (input_tensor.memory_config().is_sharded()) {
        TT_FATAL(
            !operation_attributes.sub_core_grids.has_value(),
            "Sharded tilize does not support sub core grid specification");

        // Route based on shard type following maintainer guidance:
        // - WIDTH_SHARDED: use existing multi-core sharded path
        // - HEIGHT_SHARDED/BLOCK_SHARDED: use single-core with ShardedAddrGen
        auto memory_layout = input_tensor.memory_config().memory_layout();
        if (memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            return tilize_with_val_padding::program::TilizeWithValPaddingMultiCoreShardedFactory{};
        } else {
            // HEIGHT_SHARDED or BLOCK_SHARDED: use ShardedAddrGen approach
            return tilize_with_val_padding::program::TilizeWithValPaddingSingleCoreShardedFactory{};
        }
    }
    if (!operation_attributes.enough_space_height) {
        return tilize_with_val_padding::program::TilizeWithValPaddingMultiCoreBlockInterleavedFactory{};
    }
    if (!operation_attributes.use_multicore) {
        return tilize_with_val_padding::program::TilizeWithValPaddingSingleCoreFactory{};
    }

    return tilize_with_val_padding::program::TilizeWithValPaddingMultiCoreInterleavedFactory{};
}

void TilizeWithValPaddingDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void TilizeWithValPaddingDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.padded_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "Can only tilize row major data");
    TT_FATAL(
        input_tensor.dtype() == DataType::BFLOAT16 or input_tensor.dtype() == DataType::INT32 or
            input_tensor.dtype() == DataType::UINT32 or input_tensor.dtype() == DataType::FLOAT32 or
            input_tensor.dtype() == DataType::UINT16,
        "Can only tilize bfloat16/float32 or int32/uint32/uint16 tensors");

    TT_FATAL(input_shape.rank() >= 1, "Input tensor must be of rank >= 1, but its shape is {}", input_shape);

    if (input_shape.rank() == 1) {
        // Special case: if input tensor is 1D row-major, output tiled tensor will have 1D logical shape
        // but 2D padded shape
        TT_FATAL(
            input_shape[0] <= operation_attributes.output_padded_shape[-1],
            "Output tensor shape {} must be greater than or equal to input shape {} in each dimension, but is "
            "smaller in dimension {}",
            operation_attributes.output_padded_shape,
            input_shape,
            0);
    } else {
        for (auto i = 0; i < input_shape.rank(); i++) {
            TT_FATAL(
                input_shape[i] <= operation_attributes.output_padded_shape[i],
                "Output tensor shape {} must be greater than or equal to input shape {} in each dimension, but is "
                "smaller in dimension {}",
                operation_attributes.output_padded_shape,
                input_shape,
                i);
        }
    }

    uint32_t num_rows = operation_attributes.output_padded_shape[-1];
    uint32_t inner_dim = operation_attributes.output_padded_shape[-2];
    TT_FATAL(
        inner_dim % TILE_WIDTH == 0 && num_rows % TILE_HEIGHT == 0,
        "To be tilizable output tensor shape {} must be divisible by tile size ({}, {})",
        operation_attributes.output_padded_shape,
        TILE_WIDTH,
        TILE_HEIGHT);

    auto layout = input_tensor.memory_config().memory_layout();
    if (input_tensor.memory_config().is_sharded()) {
        TT_FATAL(
            layout == TensorMemoryLayout::WIDTH_SHARDED || layout == TensorMemoryLayout::HEIGHT_SHARDED,
            "Input tensor must be width sharded");
        TT_FATAL(
            operation_attributes.output_mem_config.memory_layout() == input_tensor.memory_config().memory_layout(),
            "Output tensor must have the same memory layout as input tensor");
        for (uint32_t i = 0; i < input_tensor.padded_shape().rank(); i++) {
            if (i != input_shape.rank() - 2) {
                TT_FATAL(
                    input_shape[i] == operation_attributes.output_padded_shape[i],
                    "Input shape[{}] ({}) must equal output padded shape[{}] ({})",
                    i,
                    input_shape[i],
                    i,
                    operation_attributes.output_padded_shape[i]);
            }
        }
    }
}

TilizeWithValPaddingDeviceOperation::spec_return_value_t TilizeWithValPaddingDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    // Output logical shape should match the requested output shape (with padding)
    auto output_logical_shape = operation_attributes.output_padded_shape;

    if (input_tensor.memory_config().is_sharded()) {
        auto shard_spec = input_tensor.shard_spec().value();
        // For tiled output, shard dimensions must be multiples of tile size (32x32)
        auto output_height = operation_attributes.output_padded_shape[-2];
        auto output_width = operation_attributes.output_padded_shape[-1];

        // Calculate number of cores from input shard grid
        uint32_t num_cores = 0;
        const auto& core_ranges = shard_spec.grid.ranges();
        for (const auto& range : core_ranges) {
            uint32_t cores_in_range =
                (range.end_coord.x - range.start_coord.x + 1) * (range.end_coord.y - range.start_coord.y + 1);
            num_cores += cores_in_range;
        }

        // For TILE layout output, each shard must contain complete tiles (32x32)
        // Update shard shape for output
        if (operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            // Each core gets output_height/num_cores rows, rounded to tile boundaries
            uint32_t rows_per_core = (output_height + num_cores - 1) / num_cores;  // Round up
            rows_per_core = ((rows_per_core + 31) / 32) * 32;                      // Round to tile size (32)
            shard_spec.shape[0] = rows_per_core;
            shard_spec.shape[1] = output_width;
        } else if (operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            uint32_t cols_per_core = (output_width + num_cores - 1) / num_cores;  // Round up
            cols_per_core = ((cols_per_core + 31) / 32) * 32;                     // Round to tile size (32)
            shard_spec.shape[0] = output_height;
            shard_spec.shape[1] = cols_per_core;
        } else {  // BLOCK_SHARDED
            uint32_t cores_per_dim = std::sqrt(num_cores);
            uint32_t rows_per_core = ((output_height + cores_per_dim - 1) / cores_per_dim + 31) / 32 * 32;
            uint32_t cols_per_core = ((output_width + cores_per_dim - 1) / cores_per_dim + 31) / 32 * 32;
            shard_spec.shape[0] = rows_per_core;
            shard_spec.shape[1] = cols_per_core;
        }

        auto mem_config = operation_attributes.output_mem_config.with_shard_spec(shard_spec);
        return TensorSpec(
            output_logical_shape,
            TensorLayout::fromPaddedShape(
                operation_attributes.output_dtype,
                PageConfig(Layout::TILE),
                mem_config,
                output_logical_shape,
                operation_attributes.output_padded_shape));
    }

    return TensorSpec(
        output_logical_shape,
        TensorLayout::fromPaddedShape(
            operation_attributes.output_dtype,
            PageConfig(Layout::TILE),
            operation_attributes.output_mem_config,
            output_logical_shape,
            operation_attributes.output_padded_shape));
}

TilizeWithValPaddingDeviceOperation::tensor_return_value_t TilizeWithValPaddingDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::TilizeWithValPaddingDeviceOperation::tensor_return_value_t tilize_with_val_padding(
    const Tensor& input_tensor,
    const ttnn::Shape& output_padded_shape,
    const PadValue& pad_value,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<DataType>& output_dtype,
    bool use_multicore,
    bool enough_space_width,
    bool enough_space_height,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::operations::data_movement::TilizeWithValPaddingDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_padded_shape = output_padded_shape,
            .pad_value = pad_value,
            .output_mem_config = output_mem_config.value_or(input_tensor.memory_config()),
            .output_dtype = output_dtype.value_or(input_tensor.dtype()),
            .use_multicore = use_multicore,
            .enough_space_width = enough_space_width,
            .enough_space_height = enough_space_height,
            .sub_core_grids = sub_core_grids,
        },
        OperationType::tensor_args_t{.input_tensor = input_tensor});
}
}  // namespace ttnn::prim
