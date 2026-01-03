# Guide: Adding Sharded Address Generation to TilizeWithValPadding

## 🎯 Quick Start: What You Actually Need for the Bounty

**TL;DR:** Your `tilize_with_val_padding` **already has multi-core sharded support**! The best approach is to support **BOTH**:

**Phase 1 (Easy, Required):** Enable HEIGHT_SHARDED in existing multi-core factory
1. Remove validation that blocks HEIGHT_SHARDED
2. Verify shard spec computation is correct
3. Test it

**Phase 2 (More Work, Complete Solution):** Add ShardedAddrGen to single-core factory
1. Add ShardedAddrGen support for when `use_multicore=false` but output is sharded
2. Supports edge cases and provides full flexibility

**Why both?** They serve different use cases and the operation will automatically select the right factory based on input/output configuration.

Jump to:
- [Option A: Multi-Core (Start Here)](#option-a-enable-height_sharded-in-existing-multi-core-factory-recommended)
- [Option B: Single-Core](#option-b-add-single-core-sharded-support-uses-shardedaddrgen)

---

## Overview

The sharded address generator (`ShardedAddrGen`) allows operations to support HEIGHT_SHARDED, WIDTH_SHARDED, and BLOCK_SHARDED memory layouts without needing separate kernel implementations for each sharding type. It replaces the interleaved address generator when the input tensor is sharded.

## Key Components

### 1. Helper Functions (`sharding_addrgen_helper.hpp/cpp`)

Located in: `ttnn/cpp/ttnn/operations/ccl/sharding_addrgen_helper.{hpp,cpp}`

**Key functions:**
- `extend_sharding_compile_time_args(const Tensor& t, std::vector<uint32_t>& args)` - Adds 7 compile-time args
- `extend_sharding_run_time_args(const Tensor& t, std::vector<uint32_t>& args)` - Adds shard core mapping

**Compile-time args added (7 total):**
1. Memory layout (HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED)
2. Number of sharding cores
3. Aligned page size
4. Number of pages per tensor row (excluding padding)
5. Contiguity type (padding characteristics)
6. Pages per shard X
7. Pages per shard Y

### 2. Kernel-side Address Generator

Located in: `ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp` and `ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp`

**Usage in kernel:**
```cpp
#ifdef SHARDED
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(N),     // Memory layout
        get_compile_time_arg_val(N+1),   // Number of sharding cores
        get_compile_time_arg_val(N+2),   // Aligned page size
        get_compile_time_arg_val(N+3),   // Pages per row
        get_compile_time_arg_val(N+4),   // Contiguity type
        get_compile_time_arg_val(N+5),   // pages_per_shard_x
        get_compile_time_arg_val(N+6)>;  // pages_per_shard_y

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(rt_arg_ind));
    experimental::ShardedAddrGen<tensor_shard_info> s0 = {
        .bank_base_address = dst_addr,
        .shard_array = mapping_table
    };
#else
    // Use interleaved address generator
    constexpr auto dst_args = TensorAccessorArgs<N>();
    const auto s0 = TensorAccessor(dst_args, dst_addr, page_size);
#endif
```

## Multi-Core vs Single-Core: Do You Need ShardedAddrGen?

**Short answer: It depends on your existing architecture.**

### TilizeWithValPadding Current Architecture

Your `tilize_with_val_padding` operation already has:
1. `TilizeWithValPaddingSingleCoreFactory` - for single core execution
2. `TilizeWithValPaddingMultiCoreInterleavedFactory` - for multi-core interleaved
3. `TilizeWithValPaddingMultiCoreShardedFactory` - **ALREADY EXISTS for sharded!**

### Key Insight: Multi-Core Sharded is DIFFERENT from ShardedAddrGen

Looking at your existing `TilizeWithValPaddingMultiCoreShardedFactory`:
- It launches **one kernel per shard core** (uses `all_cores` from `output_shard_spec.grid`)
- Each core processes **its own local shard** (in L1 via circular buffers)
- Uses `reader_unary_pad_height_width_sharded.cpp` which reads from shard CB directly
- **Does NOT need ShardedAddrGen** because data is already distributed via sharding

```cpp
// From tilize_with_val_padding_multi_core_sharded_program_factory.cpp
auto all_cores = output_shard_spec.grid;  // Each core gets its own shard

// Circular buffers bound to shard buffers
auto [src0_cb_index, cb_src0] = create_cb(
    tt::CBIndex::c_1, program, all_cores, input_shard_width_bytes, num_input_rows,
    input_cb_data_format,
    src_sharded ? a.buffer() : nullptr);  // ← Directly bound to shard buffer!

// Kernel runs on all cores
unary_reader_kernel_id = tt::tt_metal::CreateKernel(
    program, "reader_unary_pad_height_width_sharded.cpp",
    all_cores,  // ← One kernel instance per shard core
    tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
```

### When DO You Need ShardedAddrGen?

You need `ShardedAddrGen` when:
- ✅ **Single core (or fewer cores than shards)** needs to access **sharded output in DRAM/L1**
- ✅ Writing to **sharded interleaved format** from non-sharded compute
- ✅ One core needs to read/write multiple shards

You DON'T need `ShardedAddrGen` when:
- ❌ Each core processes only its own shard via circular buffers
- ❌ Sharded buffers are directly bound to CBs (like your current multi-core sharded factory)

### Your Current Status

**What you already have:**
- ✅ WIDTH_SHARDED support in `TilizeWithValPaddingMultiCoreShardedFactory`
- ✅ Each core processes its own shard locally

**What's missing for HEIGHT_SHARDED:**
The validation currently blocks HEIGHT_SHARDED. You need to:
1. **Update validation** to allow HEIGHT_SHARDED memory layout
2. **Verify shard calculation logic** handles height sharding correctly
3. **No kernel changes needed** - the existing sharded reader/writer should work!

The existing multi-core sharded implementation should work for HEIGHT_SHARDED if:
- Shard specs are computed correctly for height sharding
- Validation doesn't block it
- The reader kernel correctly handles the shard shape

### Comparison: fill_pad vs tilize_with_val_padding

| Aspect | fill_pad (PR #17692) | tilize_with_val_padding |
|--------|---------------------|-------------------------|
| **Architecture** | Single factory with multi-core work split | Separate factories for different strategies |
| **Multi-core approach** | Splits work across cores, each accesses full tensor | Each core owns a shard |
| **ShardedAddrGen use** | YES - cores need to access sharded DRAM/L1 | NO (for multi-core sharded) - CBs bound to shards |
| **When needed** | Single kernel writes to sharded output | Only if adding single-core sharded support |

## Step-by-Step Implementation for TilizeWithValPadding

### Option A: Enable HEIGHT_SHARDED in Existing Multi-Core Factory (Recommended)

This is likely what you need for the bounty!

#### Step A1: Update Validation

**File:** `tilize_with_val_padding_single_core_program_factory.cpp` (or the multi-core variant)

```cpp
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

// In the create() function:

// Detect if sharded
bool sharded = output.memory_config().is_sharded();

// Build compile-time args
std::vector<uint32_t> writer_compile_time_args = {
    // ... your existing compile-time args ...
};

// Add defines and sharding args
std::map<std::string, std::string> writer_defines;
if (sharded) {
    shard_builder::extend_sharding_compile_time_args(output, writer_compile_time_args);
    writer_defines["SHARDED"] = "1";
} else {
    // For interleaved, add TensorAccessorArgs
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);
}

// Create kernel with defines
KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
    program,
    "path/to/your/writer_kernel.cpp",
    all_cores,
    tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

// Setup runtime args
std::vector<uint32_t> writer_runtime_args = {
    output.buffer()->address(),
    // ... other runtime args ...
};

if (sharded) {
    shard_builder::extend_sharding_run_time_args(output, writer_runtime_args);
}

#### Step A2: Verify Shard Spec Computation

**File:** Check `compute_output_specs` or similar

```cpp
// Ensure output shard spec is computed correctly for HEIGHT_SHARDED
if (input.memory_config().is_sharded()) {
    auto input_shard_spec = input.shard_spec().value();

    // For height sharding, shard_shape should be:
    // [rows_per_core, padded_width]

    // Compute output shard shape based on padding
    auto output_shard_shape = std::array<uint32_t, 2>{
        input_shard_spec.shape[0] + padding_rows_per_core,  // Height per core
        padded_width  // Full width per shard
    };

    // Create output shard spec with same grid but adjusted shape
    auto output_shard_spec = ShardSpec(
        input_shard_spec.grid,
        output_shard_shape,
        input_shard_spec.orientation
    );
}
```

#### Step A3: Test It!

The existing `TilizeWithValPaddingMultiCoreShardedFactory` should work because:
- It already uses `all_cores` from shard grid
- Reader kernel processes local shard data
- No global memory access needed

**That's it for multi-core!** No ShardedAddrGen needed.

---

### Option B: Add Single-Core Sharded Support (Uses ShardedAddrGen)

If you want **single-core** to support sharded output, you need ShardedAddrGen:

#### Step B1: Modify Program Factory (Host Side)

**File:** `tilize_with_val_padding_single_core_program_factory.cpp`

```cpp
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

// In the create() function:

// Detect if sharded
bool sharded = output.memory_config().is_sharded();

// Build compile-time args
std::vector<uint32_t> writer_compile_time_args = {
    // ... your existing compile-time args ...
};

// Add defines and sharding args
std::map<std::string, std::string> writer_defines;
if (sharded) {
    shard_builder::extend_sharding_compile_time_args(output, writer_compile_time_args);
    writer_defines["SHARDED"] = "1";
} else {
    // For interleaved, add TensorAccessorArgs
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);
}

// Create kernel with defines
KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
    program,
    "path/to/your/writer_kernel.cpp",
    all_cores,
    tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

// Setup runtime args
std::vector<uint32_t> writer_runtime_args = {
    output.buffer()->address(),
    // ... other runtime args ...
};

if (sharded) {
    shard_builder::extend_sharding_run_time_args(output, writer_runtime_args);
}

SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
```

#### Step B2: Modify Writer Kernel (Device Side)

**File:** Your writer kernel (e.g., `writer_unary_stick_layout_split_rows.cpp`)

```cpp
#include "api/dataflow/dataflow_api.h"
#ifdef SHARDED
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#endif

void kernel_main() {
    // Get your normal compile-time args
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    // ... more args ...
    constexpr uint32_t last_interleaved_arg_index = X; // Track where interleaved args end

    // Get runtime args
    uint32_t rt_arg_ind = 0;
    uint32_t dst_addr = get_arg_val<uint32_t>(rt_arg_ind++);
    // ... more runtime args ...

#ifdef SHARDED
    // Define sharding info using the 7 args added by extend_sharding_compile_time_args
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(last_interleaved_arg_index + 1),  // Memory layout
        get_compile_time_arg_val(last_interleaved_arg_index + 2),  // Num cores
        get_compile_time_arg_val(last_interleaved_arg_index + 3),  // Page size
        get_compile_time_arg_val(last_interleaved_arg_index + 4),  // Pages per row
        get_compile_time_arg_val(last_interleaved_arg_index + 5),  // Contiguity
        get_compile_time_arg_val(last_interleaved_arg_index + 6),  // pages_per_shard_x
        get_compile_time_arg_val(last_interleaved_arg_index + 7)>; // pages_per_shard_y

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(rt_arg_ind));
    experimental::ShardedAddrGen<tensor_shard_info> s0 = {
        .bank_base_address = dst_addr,
        .shard_array = mapping_table
    };
#else
    // Interleaved path
    constexpr auto dst_args = TensorAccessorArgs<last_interleaved_arg_index + 1>();
    const auto s0 = TensorAccessor(dst_args, dst_addr, tile_size);
#endif

    // Use s0 the same way for both sharded and interleaved
    for (uint32_t tile_id = 0; tile_id < num_tiles; tile_id++) {
        uint64_t tile_noc_addr = get_noc_addr(tile_id, s0);
        noc_async_write(src_addr, tile_noc_addr, tile_bytes);
    }
}
```

---

## Which Option Should You Choose?

### For the Bounty (#5965):

**Implement BOTH for a complete solution:**

**Start with Option A** (Quick Win):
- ✅ Simpler - likely no kernel code changes
- ✅ Faster execution - parallel processing
- ✅ Matches the primary use case: "convs are often height sharded"
- ✅ Can be done in ~30 minutes if validation is the only blocker

**Then add Option B** (Complete Coverage):
- ✅ Handles edge cases (small tensors, single-core preference)
- ✅ Provides flexibility for different workload sizes
- ✅ Makes the op truly "sharding-complete"
- ✅ Shows thoroughness (good for bounty submission!)

### When Each Gets Used:

The operation's `select_program_factory` will automatically choose:

```cpp
// From tilize_with_val_padding_device_operation.cpp
if (input_tensor.memory_config().is_sharded()) {
    return TilizeWithValPaddingMultiCoreShardedFactory{};  // ← Option A
}
if (!operation_attributes.use_multicore) {
    return TilizeWithValPaddingSingleCoreFactory{};  // ← Option B (with ShardedAddrGen)
}
```

Both can coexist! The factory selection logic determines which path to take based on:
- Input/output sharding configuration
- `use_multicore` flag
- Tensor size and core availability

### Benefits of Supporting Both:

| Scenario | Factory Used | Why |
|----------|--------------|-----|
| Large tensor, multi-core available | Multi-core sharded | Best performance |
| Small tensor, overhead concerns | Single-core sharded | Less kernel launch overhead |
| Debugging/testing | Single-core sharded | Easier to debug |
| Input already sharded | Multi-core sharded | Data already distributed |
| Converting interleaved→sharded | Single-core sharded | One core does conversion |

**Bottom line:** Option A gets you 90% of use cases quickly. Option B makes your implementation bulletproof.

## Original Step-by-Step (Option B - Single Core with ShardedAddrGen)

**Note:** The following steps are for Option B (single-core with ShardedAddrGen). Most likely you need Option A instead!

### Step 1: Update Validation

### Step 1: Update Validation

**File:** `tilize_with_val_padding_device_operation.cpp`

```cpp
void TilizeWithValPaddingDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {

    const auto& input_tensor = tensor_args.input_tensor;

    // Allow sharded inputs
    if (input_tensor.memory_config().is_sharded()) {
        TT_FATAL(
            operation_attributes.output_mem_config.is_sharded(),
            "If input is sharded, output must also be sharded");
    }
}
```

### Step 2: Update compute_output_specs

Ensure output inherits sharding from input when appropriate:

```cpp
std::vector<TensorSpec> TilizeWithValPaddingDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {

    const auto& input = input_tensors[0];
    auto output_shape = /* compute padded shape */;

    // Inherit sharding if input is sharded
    auto mem_config = this->output_mem_config;
    if (input.memory_config().is_sharded() && !mem_config.is_sharded()) {
        mem_config = input.memory_config(); // or create appropriate shard spec
    }

    return {TensorSpec(output_shape, TensorLayout(dtype, PageConfig(TILE), mem_config))};
}
```

---

    // TT_FATAL(
    //     input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
    //     "TilizeWithValPadding does not currently support sharding");

    // Allow HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED
    if (input_tensor.memory_config().is_sharded()) {
        TT_FATAL(
            operation_attributes.output_mem_config.is_sharded(),
            "If input is sharded, output must also be sharded");
    }
}
```

### Step 4: Update compute_output_specs

Ensure output inherits sharding from input when appropriate:

```cpp
std::vector<TensorSpec> TilizeWithValPaddingDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {

    const auto& input = input_tensors[0];
    auto output_shape = /* compute padded shape */;

    // Inherit sharding if input is sharded
    auto mem_config = this->output_mem_config;
    if (input.memory_config().is_sharded() && !mem_config.is_sharded()) {
        mem_config = input.memory_config(); // or create appropriate shard spec
    }

    return {TensorSpec(output_shape, TensorLayout(dtype, PageConfig(TILE), mem_config))};
}
```

---

## Summary: Why Support Both?

### The Power of Dual Implementation:

**Multi-Core Sharded (Option A):**
- Optimal for production workloads (convs, large tensors)
- Each core processes its own shard in parallel
- Maximum hardware utilization

**Single-Core with ShardedAddrGen (Option B):**
- Handles edge cases (small tensors, single-core only hardware)
- Supports interleaved→sharded conversions
- Debugging and testing friendly

### They're Complementary, Not Competing:

```cpp
// The factory selection logic uses BOTH:
ProgramFactory select_program_factory(...) {
    if (input.is_sharded()) {
        return MultiCoreSharded{};      // ← Option A handles this
    }
    if (!use_multicore) {
        return SingleCore{};             // ← Option B handles sharded output here
    }
    return MultiCoreInterleaved{};
}
```

### Real-World Example:

**Large Conv (typical):**
```
Input: 1x32x256x256, HEIGHT_SHARDED across 8 cores
→ Uses MultiCoreSharded factory (Option A)
→ Each core: processes 32 rows
→ Fast parallel execution
```

**Small preprocessing:**
```
Input: 1x1x64x64, INTERLEAVED
Output: INTERLEAVED, but want sharded-capable op
→ Uses SingleCore factory (Option B with ShardedAddrGen)
→ One core: converts to sharded if needed
→ Low overhead for small tensor
```

**Both scenarios now work!**

| Scenario | Need ShardedAddrGen? | Why? |
|----------|---------------------|------|
| **Multi-core accessing sharded tensor** | ✅ YES (Option A) | Each core reads its own shard via CB |
| **Single-core accessing sharded tensor** | ✅ YES (Option B) | One core needs to access multiple shards |
| **Multi-core interleaved** | ❌ NO | Use TensorAccessor for interleaved |
| **Single-core interleaved** | ❌ NO | Use TensorAccessor for interleaved |
| **Converting interleaved → sharded** | ✅ MAYBE | If single-core does conversion (Option B) |

**For the bounty:** Implement **BOTH OPTIONS** for a complete, production-ready solution!

---

## Important Notes (For Option B - ShardedAddrGen)

### Address Generation API

Both `ShardedAddrGen` and `TensorAccessor` (interleaved) expose the same interface:
```cpp
uint64_t noc_addr = get_noc_addr(tile_id, address_generator);
```

This allows the same kernel logic to work for both sharded and interleaved tensors.

### Compile-Time Args Order

The sharding compile-time args MUST be the last 7 args, immediately after your operation-specific args:
```
[op_arg_0, op_arg_1, ..., op_arg_N,
 shard_layout, shard_cores, shard_page_size, shard_pages_per_row,
 shard_contiguity, shard_x, shard_y]
```

For interleaved, you replace those 7 with TensorAccessorArgs (variable count based on tensor properties).

### Runtime Args Order

Runtime args structure:
```
[op_runtime_args..., shard_core_map...]
```

The shard core map is variable-length (one entry per 2 cores, packed as x,y pairs).

### Testing

After implementation, test with:
```python
@pytest.mark.parametrize("shard_scheme", [
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
])
def test_tilize_with_val_padding_sharded(device, shard_scheme):
    # Create sharded input
    # Run operation
    # Verify output matches expected result
```

## Example: fill_pad Implementation

See PR #17692 for a complete working example:
- **Program factory:** `ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/fill_pad_program_factory.cpp`
- **Kernel:** `ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_writer.cpp`
- **Tests:** `tests/ttnn/unit_tests/operations/test_fill_pad.py`

## Common Pitfalls

### For Option A (Multi-Core Sharded - Recommended):
1. **Not removing HEIGHT_SHARDED validation blocks**: Check for any TT_FATAL that prevents HEIGHT_SHARDED
2. **Incorrect shard spec computation**: Ensure output shard shape accounts for padding correctly
3. **Assuming you need ShardedAddrGen**: You don't! The existing factory should work.

### For Option B (Single-Core with ShardedAddrGen):
1. **Wrong compile-time arg index**: Ensure sharding args start at the correct index
2. **Missing includes**: Don't forget the sharding headers in kernel code
3. **Forgetting SHARDED define**: Must add to writer_defines map
4. **Not extending runtime args**: Must call `extend_sharding_run_time_args`
5. **Validation blocking sharded**: Remove old validation that forbids sharding

## Benefits

### Option A (Multi-Core Sharded):
- ✅ Supports HEIGHT_SHARDED (required for convs)
- ✅ Supports WIDTH_SHARDED (existing behavior preserved)
- ✅ Supports BLOCK_SHARDED (if shard spec is correct)
- ✅ No kernel changes needed
- ✅ Maximum parallelism (one core per shard)
- ✅ Eliminates expensive interleaved conversions

### Option B (Single-Core with ShardedAddrGen):
- ✅ Supports HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED
- ✅ Single kernel handles all layouts
- ✅ No performance regression (compile-time branching)
- ✅ Useful for interleaved-to-sharded conversions
- ✅ Handles small tensors efficiently
- ❌ Less parallel than multi-core (single core does all work)

**Together, they provide complete sharding support across all use cases!**

---

## Final Recommendation for Bounty #5965

**Implement Both Options for Maximum Points:**

### Phase 1: Multi-Core HEIGHT_SHARDED (Priority 1)
```
Time estimate: 30 minutes - 2 hours
Difficulty: Easy
Impact: Solves 90% of conv use cases
```

**Steps:**
1. ✅ Check if `TilizeWithValPaddingMultiCoreShardedFactory` already supports HEIGHT_SHARDED
2. ✅ Remove any validation blocking HEIGHT_SHARDED
3. ✅ Verify shard spec computation for height sharding
4. ✅ Add tests for HEIGHT_SHARDED inputs (multi-core)

### Phase 2: Single-Core HEIGHT_SHARDED (Priority 2)
```
Time estimate: 2-4 hours
Difficulty: Medium
Impact: Handles edge cases, completes the feature
```

**Steps:**
1. ✅ Add ShardedAddrGen to `TilizeWithValPaddingSingleCoreFactory`
2. ✅ Modify writer kernel with `#ifdef SHARDED` path
3. ✅ Add tests for HEIGHT_SHARDED with single-core
4. ✅ Test interleaved→sharded conversion

### Why Both?

**From the bounty description:**
> "Currently TilizeWithValPadding only supports width sharding, convs are often height sharded so they're unable to use TilizeWithValPadding without going to interleaved which is expensive."

**Your implementation should:**
- ✅ Support HEIGHT_SHARDED in multi-core (primary requirement)
- ✅ Support HEIGHT_SHARDED in single-core (completeness)
- ✅ Maintain existing WIDTH_SHARDED support
- ✅ Get BLOCK_SHARDED for free (both factories)

**Deliverable checklist:**
- [ ] HEIGHT_SHARDED works in multi-core path (convs use this)
- [ ] HEIGHT_SHARDED works in single-core path (edge cases)
- [ ] Tests cover HEIGHT_SHARDED + WIDTH_SHARDED + BLOCK_SHARDED
- [ ] No performance regression on existing workloads
- [ ] Bounty requirement met: "convs are often height sharded" ✅

### Submission Strategy:

**Option 1 (Conservative):** Submit Phase 1 only
- Gets you the core requirement
- Faster to implement and test
- Still worthy of bounty

**Option 2 (Thorough):** Submit Both Phases
- Shows complete understanding
- Handles all edge cases
- Demonstrates engineering excellence
- Likely to impress reviewers more

**I recommend Option 2** - the extra 2-4 hours for Phase 2 makes your submission much more robust and complete. Plus, the ShardedAddrGen pattern is reusable knowledge for future work!

### Implementation Order (Both Phases):

```
Day 1: Multi-Core Support (Phase 1)
├─ Morning: Review existing multi-core sharded factory
├─ Identify validation blocks for HEIGHT_SHARDED
├─ Verify shard spec computation logic
├─ Remove validation blocks
└─ Afternoon: Write tests for HEIGHT_SHARDED multi-core
   └─ Run tests, fix any issues

Day 2: Single-Core Support (Phase 2)
├─ Morning: Add ShardedAddrGen to single-core factory
├─ Modify writer kernel with SHARDED path
├─ Add compile-time and runtime arg handling
└─ Afternoon: Write tests for HEIGHT_SHARDED single-core
   ├─ Test small tensors
   ├─ Test interleaved→sharded conversion
   └─ Compare performance vs multi-core

Day 3: Polish & Submit
├─ Run full test suite
├─ Add BLOCK_SHARDED tests (should work automatically)
├─ Document changes
└─ Submit PR with both phases
```

### Testing Strategy (Cover All Paths):

```python
# Test matrix for complete coverage
test_configs = [
    # (memory_layout, use_multicore, factory_expected)
    (HEIGHT_SHARDED, True,  "MultiCoreSharded"),     # ← Phase 1
    (HEIGHT_SHARDED, False, "SingleCore+AddrGen"),   # ← Phase 2
    (WIDTH_SHARDED,  True,  "MultiCoreSharded"),     # Existing
    (WIDTH_SHARDED,  False, "SingleCore+AddrGen"),   # Phase 2 bonus
    (BLOCK_SHARDED,  True,  "MultiCoreSharded"),     # Should work
    (BLOCK_SHARDED,  False, "SingleCore+AddrGen"),   # Phase 2 bonus
]
```

Good luck with the bounty! 🚀

---

## Additional Resources

- **PR #17692**: fill_pad sharded implementation (uses ShardedAddrGen - Option B approach)
- **Sharding helper**: `ttnn/cpp/ttnn/operations/ccl/sharding_addrgen_helper.{hpp,cpp}`
- **Your existing sharded factory**: `tilize_with_val_padding_multi_core_sharded_program_factory.cpp`
- **Existing sharded kernel**: `reader_unary_pad_height_width_sharded.cpp`
