#!/usr/bin/env python3
import torch
import ttnn

# Replicate test setup
device = ttnn.open_device(device_id=0)

# Input: [1, 1, 48, 64] HEIGHT_SHARDED with 4 cores, 16 rows per core
input_shape = [1, 1, 48, 64]
input_torch = torch.randn(input_shape, dtype=torch.bfloat16)

# Create HEIGHT_SHARDED input
shard_spec = ttnn.ShardSpec(
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
    (16, 64),  # 16 rows per core, 64 elements per row
    ttnn.ShardOrientation.ROW_MAJOR,
)

input_mem_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    shard_spec,
)

# Create tensor on device with sharding
input_ttnn = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
input_ttnn = ttnn.to_device(input_ttnn, device, memory_config=input_mem_config)

print(f"Input tensor shape: {input_ttnn.shape}")
print(f"Input tensor padded shape: {input_ttnn.padded_shape}")
print(f"Input is_sharded: {input_ttnn.is_sharded()}")
print(f"Input memory_config: {input_ttnn.memory_config()}")

# Try tilize with padding
output_shape = [1, 1, 64, 64]
output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

try:
    output = ttnn.tilize_with_val_padding(
        input_ttnn,
        output_shape,
        pad_value=10.0,
        memory_config=output_mem_config,
    )
    print(f"\nOutput shape: {output.shape}")
    print(f"Output padded shape: {output.padded_shape}")

    # Convert back to torch
    output_torch = ttnn.to_torch(output)

    # Check output shape
    print(f"\nActual torch output shape: {output_torch.shape}")
    print(f"Expected output shape: [1, 1, 64, 64]")

    # Check a few values
    print(f"\nSample output values:")
    print(f"  output[0,0,0,:5] = {output_torch[0,0,0,:5]}")
    print(f"  input[0,0,0,:5] = {input_torch[0,0,0,:5]}")

    # Check if padding worked (row 48-63 should be pad value)
    if output_torch.shape[2] > 48:
        print(f"\n  Padding check - output[0,0,48,:5] (should be pad=10) = {output_torch[0,0,48,:5]}")
    else:
        print(f"\n  WARNING: Output has wrong shape - no padding rows visible!")
        print(f"  This suggests compute_output_specs may not be creating correct output shape")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()

ttnn.close_device(device)
