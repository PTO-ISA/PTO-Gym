# PTO micro Instruction Spec — Draft (A5)

- v0.6: Refresh the micro Instruction reference and add Special Scalar operations
- v0.5: Add CUBE instruction docs; Rename MTE instruction and address space
- v0.4: Update DMA instruction docs and add PTO Tile Instruction SPEC
- v0.3: Add runtime block query and vector-interval legality notes; Normalize load/store distribution families; Update get_buf/rls_buf details
- v0.2: Update micro Instruction latency and throughput
- v0.1: Doc Init

[toc]

---

## Part I: Architecture Overview

### Overview

This document defines the PTO micro Instruction, a compiler-internal and externally facing specification designed to represent vector compute kernels within the PTO architecture. Much like NVVM provides a robust IR for GPU architectures, the PTO micro Instruction serves as the direct bridge between high-level programming models and the underlying hardware ISA, providing a precise, architecture-aware representation of vector workloads explicitly designed for the Ascend 950 architecture.

#### Position in the Stack and Layer Modeled

The PTO micro Instruction operates as an explicit intermediate representation within the PTO compiler stack. It is designed to accurately express the user-visible architectural information needed for Ascend 950 kernels, including vector lane organization, memory space hierarchy, synchronization, and hardware-specific fusion semantics.

#### PTO Instruction Modes and Compilation Flows

Within the end-to-end PTO software stack, PTO instructions may appear in three closely related authoring or lowering modes:

- **PTO Tile Instruction**: tile-oriented PTO code that serves as a nano-kernel encapsulation of tile instructions, primarily expressing computation and data movement in terms of tile buffers, tile shapes, and tile-local layout.
- **PTO micro Instruction**: vector-execution-oriented PTO code that makes DMA setup, vector registers, masks, synchronization, and `__VEC_SCOPE__` boundaries explicit. This document is centered on this mode.
- **PTO Tile+micro Instruction**: a hybrid PTO form that keeps tile-level orchestration while embedding explicit micro-instruction regions where direct vector-pipeline control is required.

From these PTO instruction forms, the stack can proceed along two main compilation flows:

- **CCE generation flow**: PTO ISA is lowered into a CCE-oriented representation, which is then compiled by the BiSheng toolchain into Ascend device binaries.
- **VPTO flow**: PTO ISA is lowered through the VPTO backend for A5 device code generation. PTOAS organizes the device components and invokes the BiSheng compiler internally to produce the final device artifact.

```text
        High-level frameworks / DSLs / library kernels
                             |
                             v
            +----------------------------------+
            |          PTO ISA layer           |
            |                                  |
            |  (1) PTO Tile Instruction        |
            |  (2) PTO micro Instruction       |
            |  (3) PTO Tile+micro Instruction  |
            +----------------+-----------------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
 +-------------------------+   +-------------------------+
 | Path A: generate CCE    |   | Path B: generate        |
 | (CCE-oriented form)     |   | bytecode                |
 +------------+------------+   +------------+------------+
              |                             |
              v                             v
 +-------------------------+   +-------------------------+
 | BiSheng compiler        |   | BiSheng compiler        |
 | invoked explicitly      |   | invoked inside PTOAS    |
 +------------+------------+   +------------+------------+
              |                             |
              +--------------+--------------+
                             |
                             v
              +-----------------------------+
              |   Ascend device binaries    |
              +-----------------------------+
```

#### Why External Developers Read or Author PTO micro Instruction

While the majority of users will interact with the PTO architecture via higher-level frameworks, external developers may need to read or author PTO micro Instruction directly for several key reasons:

- Custom Toolchain Development: build custom compiler frontends or domain-specific languages (DSLs) that target the Ascend 950 architecture with maximum hardware utilization.
- Performance Engineering: inspect the output of high-level compiler passes, verify fine-grained optimization behaviors, and pinpoint performance bottlenecks at the architectural level.
- Micro-Optimization: hand-author highly optimized, critical mathematical kernels using a stable, precise IR when higher-level abstractions cannot achieve the theoretical peak performance of the hardware.

#### Relationship to CCE

The PTO micro Instruction is designed to express the full semantic capabilities of the Compute Cube Engine (CCE), but with significant structural and pipeline advantages for compiler development.

- Bypassing the C/Clang Pipeline: while CCE heavily relies on C/C++ extensions parsed by Clang, the PTO micro Instruction operates entirely independently of the C language frontend. By bypassing Clang AST generation and frontend processing, utilizing the PTO micro Instruction significantly reduces overall compilation time and memory overhead.
- Enhanced IR Verification: because the PTO micro Instruction is a strongly typed, SSA-based (Static Single Assignment) compiler IR rather than a C-wrapper API, it provides a much more rigorous and detailed IR verification process. Structural inconsistencies, invalid memory access patterns, and operand type mismatches are caught immediately with precise, explicit diagnostic feedback, providing developers with much higher visibility into kernel correctness than traditional CCE error reporting.

#### Intended Audience

This document is written for compiler engineers, library writers, and advanced performance architects. We expect the reader to have a working understanding of modern compiler infrastructure, specifically MLIR, the principles of Static Single Assignment (SSA) form, and a deep understanding of the vector-processing capabilities of the Ascend 950 architecture.

### Getting Started

The PTO micro Instruction is architected as a performance-critical layer within the compiler stack, specifically designed to exploit the **Decoupled Access-Execute** (DAE) nature of the Ascend 950 hardware.

#### Authoring VPTO `.pto` Files

A VPTO source file must make the target architecture, launched device function,
and cube/vector placement explicit. The recommended authoring form is a single
outer module with one or more `pto.kernel` functions whose bodies are split by
`pto.section.vector` and `pto.section.cube`. The Vector section describes the
Vector-unit program, and the Cube section describes the Cube-unit program.
Synchronization and communication between the two units are written as normal
operations in the relevant section bodies.

**Common module attributes:**

| Attribute | Attachment site | Required | Meaning |
|-----------|-----------------|----------|---------|
| `pto.target_arch = "a5"` | outer `module` | Recommended in source files | Selects the A5 PTO parser and verifier contract. A command-line `--pto-arch` value overrides the module attribute. |
| `pto.kernel` | `func.func` | Required for externally launched device kernels | Marks the function as a device kernel entry. Helper functions inside the same module do not need this attribute unless they are launched directly. |
| `pto.section.vector` | region inside a `pto.kernel` function | Required for vector-core code in the recommended source form | Contains the Vector program. |
| `pto.section.cube` | region inside a `pto.kernel` function | Required for cube-core code in the recommended source form | Contains the Cube program. |
| `pto.kernel_kind = #pto.kernel_kind<vector>` | normalized kernel `module` | Advanced/frontend-emitted form only | Marks a normalized submodule as vector-core code. |
| `pto.kernel_kind = #pto.kernel_kind<cube>` | normalized kernel `module` | Advanced/frontend-emitted form only | Marks a normalized submodule as cube-core code. |

In this source form, every `pto.kernel` function must contain one or both
sections. A function may contain at most one `pto.section.vector` and at most
one `pto.section.cube`; nested sections are invalid. Values defined outside the
sections may be used by both sections, but values defined inside one section
are local to that section.

For the recommended source form, keep `pto.target_arch` on the outer module,
mark the launched function with `pto.kernel`, and place core-specific code
inside `pto.section.vector` and/or `pto.section.cube`:

```mlir
module attributes {pto.target_arch = "a5"} {
  func.func @mixed_kernel(%a: !pto.ptr<f16, gm>,
                          %b: !pto.ptr<f16, gm>,
                          %out: !pto.ptr<f32, gm>) attributes {pto.kernel} {
    %c0_i64 = arith.constant 0 : i64
    %l1 = pto.castptr %c0_i64 : i64 -> !pto.ptr<f16, l1>
    %ub = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>

    pto.section.cube {
      // Cube program body.
    }

    pto.section.vector {
      // Vector program body.
    }

    return
  }
}
```

Vector-only and cube-only kernels use the same structure with only the section
they need:

```mlir
module attributes {pto.target_arch = "a5"} {
  func.func @vadd_kernel(%lhs: !pto.ptr<f32, gm>,
                         %rhs: !pto.ptr<f32, gm>,
                         %out: !pto.ptr<f32, gm>) attributes {pto.kernel} {
    %c0_i64 = arith.constant 0 : i64
    %ub = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>

    pto.section.vector {
      // Vector-core program body.
    }

    return
  }
}
```

Advanced frontends may emit a normalized container directly. This is not the
preferred hand-authored source shape, but it is a valid compiler-facing form:

```mlir
module attributes {pto.target_arch = "a5"} {
  module attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    func.func @kernel(%in: !pto.ptr<f32, gm>,
                      %out: !pto.ptr<f32, gm>) attributes {pto.kernel} {
      return
    }
  }
  module attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    func.func @kernel(%in: !pto.ptr<f32, gm>,
                      %out: !pto.ptr<f32, gm>) attributes {pto.kernel} {
      return
    }
  }
}
```

At the container top level, only kernel submodules are valid. Each kernel
submodule must carry exactly one `pto.kernel_kind`. Put `pto.target_arch` on
the outer module so all submodules share the same target contract.

**Compilation:**

```bash
ptoas --pto-arch=a5 --pto-backend=vpto kernel.pto -o kernel.o
```

This command emits the final device artifact.

#### Hardware Pipeline Modeling

The IR is structured to mirror the three primary hardware pipelines of the Ascend 950 architecture. Correct PTO micro Instruction authoring requires managing the interaction between these asynchronous units:

**MTE2** (Memory Transfer Engine - Inbound): Responsible for moving data from Global Memory (GM) to the Unified Buffer (UB).

**Vector Core** (Computation): The primary engine for executing SIMD operations on data stored in UB.

**MTE3** (Memory Transfer Engine - Outbound): Responsible for moving processed data from UB back to GM.

#### Architecture Detail: Vector Lane (VLane)

The vector register is organized as **8 VLanes** of 32 bytes each. A VLane is the atomic unit for group reduction operations.

```
vreg (256 bytes total):
┌─────────┬─────────┬─────────┬─────┬─────────┬─────────┐
│ VLane 0 │ VLane 1 │ VLane 2 │ ... │ VLane 6 │ VLane 7 │
│   32B   │   32B   │   32B   │     │   32B   │   32B   │
└─────────┴─────────┴─────────┴─────┴─────────┴─────────┘
```

Elements per VLane by data type:

| Data Type | Elements/VLane | Total Elements/vreg |
|-----------|---------------|-------------------|
| i8/si8/ui8 | 32 | 256 |
| i16/si16/ui16/f16/bf16 | 16 | 128 |
| i32/si32/ui32/f32 | 8 | 64 |
| i64/si64/ui64 | 4 | 32 |

#### Memory and Synchronization Model

The PTO micro Instruction enforces a strict memory hierarchy. The Unified Buffer (UB) is the only valid operand source for vector compute instructions. Consequently, the architecture of a PTO micro Instruction program is defined by the explicit management of data movement:

**Address Space Isolation**: The IR uses `!pto.ptr<element-type, space>` to distinguish between GM (`!pto.ptr<T, gm>`) and UB (`!pto.ptr<T, ub>`). The verifier ensures that vector compute operations do not access GM directly; data must first be moved into UB.

**UB Capacity**: The Unified Buffer provides 256KB of on-chip SRAM (also referred to as "vecTile").

**Data Flow**:

```
┌─────────────────────────────────────────────┐
│                 Global Memory (GM)           │
│              (Off-chip HBM/DDR)              │
└─────────────────────┬───────────────────────┘
                      │ DMA (MTE2 inbound / MTE3 outbound)
┌─────────────────────▼───────────────────────┐
│              Unified Buffer (UB)             │
│            (On-chip SRAM, 256KB)             │
└─────────────────────┬───────────────────────┘
                      │ Vector Load/Store (PIPE_V)
┌─────────────────────▼───────────────────────┐
│           Vector Register File (VRF)         │
│     vreg (256B each) + mask (256-bit each)   │
└─────────────────────────────────────────────┘
```

1. **GM → UB**: DMA transfer via MTE2 (`pto.mte_gm_ub`)
2. **UB → vreg**: Vector Load instructions (`pto.vlds`, `pto.vldsx2`, etc.)
3. **vreg → vreg**: Compute instructions (`pto.vadd`, `pto.vmul`, etc.)
4. **vreg → UB**: Vector Store instructions (`pto.vsts`, `pto.vstsx2`, etc.)
5. **UB → GM**: DMA transfer via MTE3 (`pto.mte_ub_gm`)

The grouped DMA surface in this specification covers `pto.mte_gm_ub`
(GM→UB), `pto.mte_ub_gm` (UB→GM), and `pto.mte_ub_ub` / `pto.mte_ub_l1`
(UB→UB or UB→CBUF).

**Load/Store Access Patterns**:

For UB↔vreg data movement, besides contiguous load/store, the architecture provides rich access pattern support including strided access, pack/unpack, interleave/deinterleave, broadcast, upsample/downsample, channel split/merge, gather/scatter, and squeeze/expand operations. For detailed instruction syntax and distribution modes, refer to the [Vector Load/Store](#micro-03-vector-load-store) group in the ISA specification.

#### Synchronization Model

The Ascend 950 architecture employs a cluster-based design with a 1:2 ratio of Cube cores to Vector cores. The PTO micro Instruction provides multiple levels of synchronization to manage concurrent execution across pipelines and cores:

**Inter-Core Synchronization (within a cluster):**

Synchronization between cores within the same cluster is achieved via the core sync mechanism using `pto.set_intra_core` and `pto.wait_intra_core` operations. This enables coordination between Cube and Vector cores sharing the same cluster resources.

**Vector Core Pipeline Synchronization:**

Within a single core, multiple pipelines operate asynchronously:

- **MTE2 (PIPE_MTE2)**: DMA copy-in from GM to UB
- **MTE3 (PIPE_MTE3)**: DMA copy-out from UB to GM
- **Vector Compute (PIPE_V)**: Vector ALU operations
- **Scalar (PIPE_S)**: Scalar unit running the kernel program

Pipeline synchronization can be achieved through two mechanisms:

1. **Flag/Event mechanism**: `pto.set_flag` and `pto.wait_flag` operations resolve Read-After-Write (RAW) and Write-After-Read (WAR) hazards between pipelines.

2. **Buffer-ID mechanism**: `pto.get_buf` and `pto.rls_buf` provide finer-grained synchronization through buffer acquisition and release semantics for producer-consumer coordination.

**Intra-Pipeline Memory Barriers (within `__VEC_SCOPE__`):**

Within the vector execution scope, the hardware does not track UB address aliasing between reg↔UB accesses. When UB addresses overlap or alias between vector load/store operations, explicit memory barriers are required:

```c
pto.mem_bar "VV_ALL"      // All prior vector ops complete before subsequent
pto.mem_bar "VST_VLD"     // All prior vector stores visible before subsequent loads
pto.mem_bar "VLD_VST"     // All prior vector loads complete before subsequent stores
pto.dcci %gm "ENTIRE_DATA_CACHE", "CACHELINE_OUT" : !pto.ptr<i8, gm>
pto.dsb "ALL"
```

Without proper barriers, loads may see stale data or stores may be reordered incorrectly.

#### Execution Scopes (__VEC_SCOPE__)

`__VEC_SCOPE__` is the IR-level representation of a Vector Function (VF) launch. In the PTO architecture, it defines the hardware interface between the Scalar Unit and the Vector Thread.

In PTO micro Instruction source IR, vector execution scopes are modeled as dedicated region ops. The default form is `pto.vecscope`; when the scope body must reject implicit capture and require explicit region arguments, use `pto.strict_vecscope`.

**Scalar-Vector Interface:**

The execution model follows non-blocking fork semantics:

- Scalar invocation: the scalar processor invokes a vector thread by calling a VF. Once the launch command is issued, the scalar unit does not stall and continues executing subsequent instructions in the pipeline.
- Vector execution: after invocation, the vector thread independently fetches and executes the instructions defined within the VF scope.
- Parallelism: this decoupled execution allows the scalar and vector units to run in parallel, so the scalar unit can prepare addresses or manage control flow while the vector unit performs heavy SIMD computation.

**Launch Mechanism And Constraints:**

- Parameter buffering: all arguments required by the VF must be staged in hardware-specific buffers.
- Launch overhead: launching a VF incurs a latency of a few cycles. Very small VFs should account for this overhead because launch cost can rival useful computation time.

**MLIR Representation:**

```mlir
pto.vecscope {
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %v = pto.vlds %ub[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
}
```

**Strict MLIR Representation:**

```mlir
pto.strict_vecscope(%ub, %ub_out, %lane) {
^bb0(%in: !pto.ptr<f32, ub>, %out: !pto.ptr<f32, ub>, %iv: index):
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %v = pto.vlds %in[%iv] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %out[%iv], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
} : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, index) -> ()
```

`pto.strict_vecscope` is the strict form of `pto.vecscope`.

- `pto.vecscope` allows the body to use surrounding SSA values directly.
- `pto.strict_vecscope` requires every external value used by the body to be passed through the op operand list and received as a body block argument.
- `pto.strict_vecscope` rejects implicit capture from the surrounding scope.
- both ops still represent one explicit VPTO vector interval.
- regardless of whether the source form uses `pto.vecscope`,
  `pto.strict_vecscope`, or a lowered carrier loop with
  `llvm.loop.aivector_scope`, every op that produces or consumes `!pto.vreg`,
  `!pto.mask<...>`, or `!pto.align` must be enclosed by exactly one vector
  interval
- nested vector intervals are not part of the legal VPTO surface; ordinary
  nested `scf.for` structure is fine, but one vector interval may not contain
  another vector interval

### Example: VecScope

```mlir
pto.mte_gm_ub %7, %2, %c0_i64, %c128_i64
  nburst(%c32_i64, %c128_i64, %c128_i64)
  : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64

pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

pto.vecscope {
  scf.for %lane = %c0 to %9 step %c64 {
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
    %v = pto.vlds %2[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %abs, %8[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
  }
}

pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.mte_ub_gm %8, %14, %c128_i64
  nburst(%c32_i64, %c128_i64, %c128_i64) l2_cache_ctl(%c0_i64)
  : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64
```

### Example: Strict VecScope

```mlir
pto.strict_vecscope(%ub_in, %ub_out, %lane, %remaining) {
^bb0(%in: !pto.ptr<f32, ub>, %out: !pto.ptr<f32, ub>, %iv: index, %rem: i32):
  %mask, %next_remaining = pto.plt_b32 %rem : i32 -> !pto.mask<b32>, i32
  %v = pto.vlds %in[%iv] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %out[%iv], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
} : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, index, i32) -> ()
```

Use `pto.strict_vecscope` when the source form should make all vector-scope inputs explicit in the region signature instead of relying on surrounding SSA visibility. The scope op itself only defines the vector-interval boundary and region argument contract.

### Cluster Programming Model

#### Overview

An A5 cluster contains one **Cube block** (AIC) and two **Vector blocks** (AIV0, AIV1). Each
block runs an **independent program** under its own Scalar Unit (SU), with its own issue queues:

| Block | Issue Queues |
|---|---|
| Cube (AIC) | MTE2, MTE1, CUBE, FIXP |
| Vector (AIV) | MTE2, VEC, MTE3 |

There is no implicit synchronization between blocks. All coordination between the Cube and Vector
programs is **explicit**, via the primitives described below.

```
┌─────────────────────────────────────── A5 CLUSTER ───────────────────────────────────────┐
│                                                                                           │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐           │
│  │   CUBE CORE (AIC)   │    │  VECTOR 0 (AIV0)    │    │  VECTOR 1 (AIV1)    │           │
│  │                     │    │   subblock_id = 0   │    │   subblock_id = 1   │           │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │    │  ┌───────────────┐  │           │
│  │  │  Scalar Unit  │  │    │  │  Scalar Unit  │  │    │  │  Scalar Unit  │  │           │
│  │  │  (SU)         │  │    │  │  (SU)         │  │    │  │  (SU)         │  │           │
│  │  │  runs cube    │  │    │  │  runs vec     │  │    │  │  runs vec     │  │           │
│  │  │  program      │  │    │  │  program      │  │    │  │  program      │  │           │
│  │  └───────────────┘  │    │  └───────────────┘  │    │  └───────────────┘  │           │
│  │   ── Issue Queues ─ │    │   ── Issue Queues ─ │    │   ── Issue Queues ─ │           │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │    │  ┌───────────────┐  │           │
│  │  │     MTE2      │  │    │  │     MTE2      │  │    │  │     MTE2      │  │           │
│  │  │    GM → L1    │  │    │  │    GM → UB    │  │    │  │    GM → UB    │  │           │
│  │  ├───────────────┤  │    │  ├───────────────┤  │    │  ├───────────────┤  │           │
│  │  │     MTE1      │  │    │  │      VEC      │  │    │  │      VEC      │  │           │
│  │  │   L1 → L0A/B  │  │    │  │  SIMD compute │  │    │  │  SIMD compute │  │           │
│  │  ├───────────────┤  │    │  ├───────────────┤  │    │  ├───────────────┤  │           │
│  │  │     CUBE      │  │    │  │     MTE3      │  │    │  │     MTE3      │  │           │
│  │  │  MMAD (L0C)   │  │    │  │    UB → GM    │  │    │  │    UB → GM    │  │           │
│  │  ├───────────────┤  │    │  └───────────────┘  │    │  └───────────────┘  │           │
│  │  │     FIXP      │  │    │                     │    │                     │           │
│  │  │  L0C → UB     │  │    │                     │    │                     │           │
│  │  │  (fixpipe)    │  │    │                     │    │                     │           │
│  │  └───────────────┘  │    │                     │    │                     │           │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘           │
│                                                                                           │
│  ┌────────────────────── SC (System Controller) ──────────────────────────────────────┐  │
│  │                                                                                     │  │
│  │   32 semaphores · 4-bit counter each · shared for C→V and V→C directions           │  │
│  │                                                                                     │  │
│  │   ┌──────────────────────────────────────────────────────────────────────────────┐ │  │
│  │   │  sema_id 0 –15  │ [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7][ 8][ 9][10][11][12][13][14][15] │ │  │
│  │   │                 │                    ↕  C→V / V→C  ↕                         │ │  │
│  │   │                 │              communicate with AIV0 (subblock_id=0)          │ │  │
│  │   ├──────────────────────────────────────────────────────────────────────────────┤ │  │
│  │   │  sema_id 16–31  │ [16][17][18][19][20][21][22][23][24][25][26][27][28][29][30][31] │ │  │
│  │   │                 │                    ↕  C→V / V→C  ↕                         │ │  │
│  │   │                 │              communicate with AIV1 (subblock_id=1)          │ │  │
│  │   └──────────────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                                     │  │
│  │   → 16 sema_id pairs (0–15) available for 1:2 C:V sync per slot                   │  │
│  │                                                                                     │  │
│  │   set_intra_block(trigger_pipe, sema_id)  ──►  increments semaphore                │  │
│  │   wait_intra_core(wait_pipe,    sema_id)  ──►  stalls pipe until semaphore > 0     │  │
│  │                                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Intra-Cluster Synchronization

Within a cluster, the PTO micro ISA provides two levels of synchronization:

**Intra-core pipeline sync** (`pto.set_flag` / `pto.wait_flag`): coordinates the asynchronous
pipelines *within a single block* — for example, ensuring MTE2 completes a GM→UB load before
the VEC pipeline begins computation. This does not cross block boundaries.

**Inter-block sync** (`pto.set_intra_block` / `pto.wait_intra_core`): coordinates between the
Cube block and a Vector block within the same cluster. The sender specifies which **local
pipeline** commits the signal, ensuring the preceding operation on that pipeline has completed
before the signal is issued. The receiver specifies which **local pipeline** should stall until
the signal arrives. This is the fundamental IPC primitive for Cube–Vector cooperation on A5.

For the public `pto.sync.set` / `pto.sync.wait` surface on A5, event IDs are physical semaphore
IDs in the `0-31` range. AIV subblock 0 uses event IDs `0-15`, and AIV subblock 1 uses event
IDs `16-31`. Code that signals or waits for both AIV subblocks should explicitly emit both
the base ID and `base_id + 16`.

> **Note:** `pto.set_cross_core` / `pto.wait_cross_core` operate at **multi-cluster** scope and
> are not used for intra-cluster communication.

#### Intra-Cluster Data Paths

A5 provides dedicated on-chip data paths between the Cube and Vector blocks, bypassing Global
Memory entirely. These are the **recommended high-performance paths** for intra-cluster tile
exchange.

##### C→V: Cube L0C → Vector UB (fixpipe)

The **fixpipe** instruction transfers data directly from Cube's L0C buffer to a Vector block's UB.
Because Cube natively produces results in **NZ fractal layout** and Vector operates on **ND
(row-major) layout**, fixpipe performs the layout conversion in hardware:

```
Cube L0C  (NZ layout)  ──[fixpipe, NZ2ND]──▶  Vector UB  (ND layout)
```

Fixpipe supports a **dual-destination mode**: a single transfer can write to *both* AIV0's UB and
AIV1's UB simultaneously, with the tile split in hardware along either the row axis
(`DualModeSplitM`) or the column axis (`DualModeSplitN`):

| Split | AIV0 receives | AIV1 receives |
|---|---|---|
| Split-M (rows) | Upper `[M/2, N]` in ND | Lower `[M/2, N]` in ND |
| Split-N (cols) | Left `[M, N/2]` in ND | Right `[M, N/2]` in ND |

This 1→2 broadcast with in-hardware tile split is the architectural basis for 1:2
Cube-to-Vector tile distribution.

##### V→C: Vector UB → Cube L1

The reverse path transfers data from a Vector block's UB into Cube's L1 buffer.
A key architectural constraint: Cube's L1 stores tiles in **NZ fractal layout** (e.g.
`K1M1M0K0` — for fp16: `K0=16`, `M0=16`) so they can be loaded into L0A/L0B for MMAD
computation. Since Vector produces tiles in **ND layout**, the layout conversion from ND to NZ
must be applied as part of the V→C transfer:

```
Vector UB  (ND layout)  ──[ND→NZ movement]──▶  Cube L1  (NZ K1M1M0K0)
```

For 1:2 mode, both AIV0 and AIV1 each transfer a sub-tile into Cube's L1. The two sub-tiles are
assembled into a single contiguous NZ Mat tile in L1, ready for use as a LeftTile or RightTile
input to MMAD:

| Split | AIV0 writes to L1 | AIV1 writes to L1 | Assembled in L1 |
|---|---|---|---|
| Split-M (rows) | `[K/2, N]` NZ at base | `[K/2, N]` NZ at offset | Full `[K, N]` NZ Mat tile |
| Split-N (cols) | `[K, N/2]` NZ at base | `[K, N/2]` NZ at offset | Full `[K, N]` NZ Mat tile |

##### Fallback: GM-Staged Transfer

When the local data path is not applicable, data can be exchanged via a **Global Memory staging
buffer**: the producer DMAs data to GM, and the consumer DMAs from GM. This path incurs off-chip
bandwidth cost and higher latency, but serves as a general fallback.

#### Cube Internal Buffer Layout: NZ Fractal Format

All cube unit internal buffers (L1/cbuf, L0A, L0B, L0C) use a **fractal NZ layout** rather than
row-major ND. Understanding this layout is essential when authoring cube data-movement ops.

##### Definition

Given hardware constant `C0 = 32 bytes`, for element type with byte width `E = sizeof(T)`:

- Inner tile width: `K0 = N0 = C0 / E` (e.g. `K0 = 16` for fp16/bf16)
- Inner tile height: `M0 = 16`

NZ re-indexing for a logical `[M, K]` tensor:

```
NZ index: (k1, m1, m0, k0)
  where  k1 = k / K0,  k0 = k % K0
         m1 = m / M0,  m0 = m % M0
Physical layout: K1 x M1 x M0 x K0  (last dimension contiguous)
```

##### Per-buffer NZ Layouts

| Buffer | Logical shape | Physical NZ layout | Notes |
|--------|--------------|-------------------|-------|
| L1 (cbuf) - Tensor A | `[M, K]` | `K1 M1 M0 K0` | Row-major A staged into NZ layout |
| L1 (cbuf) - Tensor B | `[K, N]` | `K1 N1 K0 N0` | Row-major B staged into NZ layout |
| L0A (left operand)   | -        | `K1 M1 M0 K0` | FRACTAL_NZ (A5) / FRACTAL_ZZ (A3): same NZ order as L1 cbuf |
| L0B (right operand)  | -        | `K1 N1 N0 K0` | FRACTAL_ZN: row-major outer, col-major inner (K0 innermost) |
| L0C (accumulator)    | `[M, N]` | `N1 M1 M0 N0` | output of MMAD (FRACTAL_NZ: col-major outer, row-major inner) |

##### Data Flow: GM -> L1 -> L0A/B -> L0C

```
+------------------------------------------------------------------------------+
|              GEMM Data Layout: GM -> L1 (NZ) -> L0A/B -> L0C               |
+------------------------------------------------------------------------------+

STEP 1 - Global Memory (ND, row-major)
--------------------------------------
 Tensor A [M, K]                     Tensor B [K, N]
 (K is the contiguous axis)          (N is the contiguous axis)

  col->  k0  k1  ...  kK-1             col->  n0  n1  ...  nN-1
row|   +--------------------+         row|   +--------------------+
  m0  | a00 a01 ...        |           k0  | b00 b01 ...        |
  m1  | a10 a11 ...        |           k1  | b10 b11 ...        |
  ... |                    |           ... |                    |
  mM-1|                    |          kK-1 |                    |
      +--------------------+               +--------------------+
  Physical: A[m*K + k]                 Physical: B[k*N + n]

STEP 2 - GM -> L1 (cbuf): NDtoNZ fractal repack
-------------------------------------------------
 Use the structured cube load surface to stage row-major A and B into L1 NZ layout.

 A in L1: K1 x M1 x M0 x K0          B in L1: K1 x N1 x K0 x N0
 For each outer block (k1, m1):       For each outer block (k1, n1):
 +----------------------------+       +----------------------------+
 |  M0 rows x K0 cols         |       |  K0 rows x N0 cols         |
 |  (16x16 elems contiguous)  |       |  (16x16 elems contiguous)  |
 |  m0|  k0-> [0 .. K0-1]     |       |  k0|  n0-> [0 .. N0-1]     |
 |   0   [a a a a ...]        |       |   0   [b b b b ...]        |
 |   1   [a a a a ...]        |       |   1   [b b b b ...]        |
 |  ...                       |       |  ...                       |
 |  M0-1 [a a a a ...]        |       |  K0-1 [b b b b ...]        |
 +----------------------------+       +----------------------------+
 Physical: A_nz[k1][m1][m0][k0]       Physical: B_nz[k1][n1][k0][n0]

 NOTE: For GEMM with row-major A/B, stage both operands from GM to L1 as
   logical ND-to-NZ movement. If the source is already in a transposed logical
   layout, express that at the structured load level instead of relying on a
   later interpretation of the same bytes.

STEP 3 - L1 -> L0A / L0B
--------------------------
 L0A: cbuf K1 M1 M0 K0 --mte_l1_l0a-->  L0A K1 M1 M0 K0  (FRACTAL_NZ on A5)
 L0B: cbuf K1 N1 K0 N0 --mte_l1_l0b--> L0B K1 N1 N0 K0  (FRACTAL_ZN, K0 innermost)

 Why transpose at L1->L0B and not at GM->L1?
 --------------------------------------------
 The cube reduction axis is K. L0B requires K innermost (N1 K1 K0 N0)
 so the cube hardware reads all K0 elements per cycle without striding.
 The inner-box transpose is performed as part of the structured right-load
 movement itself; no separate user-visible pass is required.
 Each 512B fractal z-block is permuted as it moves from L1 to L0B.

  L0A tile (cube LEFT port):           L0B tile (cube RIGHT port):
  +---------------------+              +---------------------+
  |  shape: [M0, K0]    |       x      |  shape: [K0, N0]    |
  |  M0 rows, K0 cols   |              |  K0 rows, N0 cols   |
  |  K innermost (fast) |              |  K innermost (fast) |
  +---------------------+              +---------------------+
          |                                      |
          +-----------------+--------------------+
                            |  pto.mad (MMAD)
                            v

STEP 4 - L0C output layout: N1 M1 M0 N0
-----------------------------------------
  For each outer block (n1, m1):
  +------------------------------+
  |  M0 rows x N0 cols           |
  |  = result sub-tile of C[M,N] |
  |  n0->  [0 .. N0-1]           |
  |  m0|  [c c c c ...]          |
  |       [c c c c ...]          |
  +------------------------------+
  Physical: C_nz[n1][m1][m0][n0]  ->  C_nd[m1*M0+m0][n1*N0+n0]

  Writeback: FIXPIPE MTE ops convert the L0C NZ result to the requested
             destination layout and memory space.

Full pipeline summary
----------------------
  GM (ND)          L1/cbuf (NZ)            L0A/B (NZ)          L0C (NZ)    GM (ND)

  A[M,K] --mte_gm_l1_frac/mte_gm_l1--> K1 M1 M0 K0 --mte_l1_l0a-->  K1 M1 M0 K0 -+
                                                               +-MAD-> N1 M1 M0 N0 --> C[M,N]
  B[K,N] --mte_gm_l1_frac/mte_gm_l1--> K1 N1 K0 N0 --mte_l1_l0b--> K1 N1 N0 K0 -+
                                 ^
                      transpose as part of mte_l1_l0b when requested
                      NOT at GM->L1
```

#### Programming Model

The common pattern for Cube–Vector co-programming is a **software pipeline**: the Cube and Vector
programs run a coordinated loop where each iteration the Cube produces a tile and the Vector
consumes it (or vice versa), with explicit `pto.set_intra_block` / `pto.wait_intra_core`
handshakes at each step to maintain correct data ordering.

The PTO micro ISA exposes all the hardware primitives above directly. Higher-level constructs
that simplify this pattern (such as in-order FIFO abstractions) can be implemented as software
libraries on top of these primitives; they are not part of the ISA itself.

### Scope

This document is the interface specification centered on the `mlir::pto` dialect and the shared MLIR surface used alongside it in PTO micro Instruction programs.

It only describes:

- operation names
- operand and result lists
- operand and result types
- important attributes
- C-style semantics for each operation

It does not describe lowering strategy.

PTO micro Instruction source programs are not restricted to `pto` operations alone. In practice they also use shared MLIR dialect ops, most notably the full scalar operation surface of `arith` together with structured control-flow ops from `scf`, to express scalar constants, scalar arithmetic, type conversion, comparisons, and structured control flow around PTO vector or tile regions. These shared-dialect ops are part of the supported PTO micro Instruction source surface and should be regarded as part of PTO-ISA alongside `pto` dialect operations.

### Shared MLIR Dialects

- `arith`: the full scalar `arith` surface is supported in PTO micro Instruction programs, covering scalar integer, floating-point, boolean, and `index` operations. In current samples the most common uses are still constants, offset/bounds arithmetic, casts, compares, and selects.
- `scf`: structured control flow used to model counted loops, conditional regions, loop-carried state, and break-like control around PTO compute and data-movement ops.
- Shared dialect ops remain in standard MLIR form so that PTO analyses and backend passes can reason about control flow and scalar state without re-encoding them as PTO-specific instructions.

### BlockDim Query Operations

These ops expose the current kernel instance's execution coordinates to scalar code. They are the PTO-level equivalent of runtime queries such as `GetBlockIdx()` and `GetBlockNum()` in kernel programming models.

Use them when the same kernel body is launched across multiple blocks or subblocks and each execution instance must figure out which slice of the global workload it owns.

A common pattern is:

- split the full input/output tensor into `block_num` disjoint block-sized regions
- let each block compute its own starting offset from `block_idx`
- within one block, further tile the local region and drive the tile loop with ordinary scalar `arith` / `scf` ops

For example, if a tensor is split evenly across 8 blocks and each block handles `block_length = 2048` elements, then block `b` owns the global range `[b * block_length, (b + 1) * block_length)`. The per-block GM base pointer can be formed by adding `block_idx * block_length` elements to the original base pointer.

At the PTO micro Instruction level, these runtime-query ops are pure scalar producers. They do not perform data movement, do not allocate memory, and do not by themselves create tiling or double buffering. Instead, they provide the scalar values used by surrounding address computation and structured control flow.

#### Example: block-level data partitioning

```mlir
%block = pto.get_block_idx
%block_num = pto.get_block_num
%block_len = arith.constant 2048 : index
%base = arith.index_cast %block : i64 to index
%offset = arith.muli %base, %block_len : index
%block_in = pto.addptr %gm_in, %offset : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
%block_out = pto.addptr %gm_out, %offset : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
```

In this pattern, all blocks execute the same kernel body, but each block sees a different `%block` value and therefore computes a different GM window.

The complete syntax, result types, constraints, semantics, pseudocode, and
partitioning example for `pto.get_block_idx`, `pto.get_subblock_idx`,
`pto.get_block_num`, and `pto.get_subblock_num` are documented in
[Special Scalar Operations](#kernel-execution-query-operations).

#### `pto.store_vfsimt_info`

- **syntax:** `pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32`
- **operands:** `i32, i32, i32`
- **semantics:** Configure the SIMT VF launch descriptor consumed by a subsequent SIMT entry invocation. The three operands are the launch dimensions in `z, y, x` order.
- **placement:** This op must appear in the outer non-SIMT caller. It must not appear inside a function marked with `pto.simt_entry`.

```c
store_vfsimt_info(dim_z, dim_y, dim_x);
```

#### `pto.simt_launch`

- **syntax:** `pto.simt_launch @body<<<%dim_x, %dim_y, %dim_z>>>(%arg0, ...) : (arg_types...) -> ()`
- **operands:** `%dim_x`, `%dim_y`, and `%dim_z` are `i32` workitem counts in `x, y, z` launch order. The remaining operands are passed to `@body`.
- **semantics:** Invoke the SIMT entry `@body` for the workitem space described by `%dim_x * %dim_y * %dim_z`. Workitems in `@body` observe thread coordinates through the SIMT query ops.
- **placement:** This op must appear in the outer non-SIMT caller. The callee must be marked with `pto.simt_entry` and must return no values.

```mlir
pto.simt_launch @simt_write<<<%dim_x, %dim_y, %dim_z>>>(%ub_out)
  : (!pto.ptr<i32, ub>) -> ()
```

#### `pto.get_tid_x`

- **syntax:** `%tx = pto.get_tid_x : i32`
- **result:** `i32`
- **semantics:** Return the current SIMT lane X coordinate inside the active VF launch.

```c
tx = get_tid_x();
```

#### `pto.get_tid_y`

- **syntax:** `%ty = pto.get_tid_y : i32`
- **result:** `i32`
- **semantics:** Return the current SIMT lane Y coordinate inside the active VF launch.

```c
ty = get_tid_y();
```

#### `pto.get_tid_z`

- **syntax:** `%tz = pto.get_tid_z : i32`
- **result:** `i32`
- **semantics:** Return the current SIMT lane Z coordinate inside the active VF launch.

```c
tz = get_tid_z();
```

Example:

```mlir
module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @simt_store_tid_kernel(%out: !pto.ptr<i32, gm>) attributes {pto.kernel} {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %dim_z = arith.constant 1 : i32
    %dim_y = arith.constant 32 : i32
    %dim_x = arith.constant 32 : i32

    %ub_out = pto.castptr %c0_i64 : i64 -> !pto.ptr<i32, ub>
    pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32
    func.call @simt_write(%ub_out) : (!pto.ptr<i32, ub>) -> ()

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.dma_store %ub_out, %out, %c128_i64
      nburst(%c32_i64, %c128_i64, %c128_i64)
      : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i64
    return
  }

  func.func @simt_write(%dst: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {
    %tx = pto.get_tid_x : i32
    %ty = pto.get_tid_y : i32
    %tz = pto.get_tid_z : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %ty_shift = arith.shli %ty, %c8_i32 : i32
    %tz_shift = arith.shli %tz, %c16_i32 : i32
    %xy = arith.ori %tx, %ty_shift : i32
    %xyz = arith.ori %xy, %tz_shift : i32
    %lane_base = arith.muli %ty, %c32_i32 : i32
    %tid = arith.addi %lane_base, %tx : i32
    %tid_idx = arith.index_castui %tid : i32 to index
    pto.store %xyz, %dst[%tid_idx] : !pto.ptr<i32, ub>, i32
    return
  }
}
```

Typical usage:

```mlir
%block = pto.get_block_idx
%subblock = pto.get_subblock_idx
%block_num = pto.get_block_num
%subblock_num = pto.get_subblock_num
```

### VMS4 Status Query

#### `pto.get_vms4_sr`

- **syntax:** `%list0, %list1, %list2, %list3 = pto.get_vms4_sr : i16, i16, i16, i16`
- **results:** four `i16` values
- **semantics:** Read `VMS4_SR` and return the finished element counts for
  source lists 0, 1, 2, and 3. After an exhausted `pto.vmrgsort4`, these are
  the per-source-list executed counts.

| Bits | Meaning |
|------|---------|
| `[15:0]` | finished count for source list 0 |
| `[31:16]` | finished count for source list 1 |
| `[47:32]` | finished count for source list 2 |
| `[63:48]` | finished count for source list 3 |

```c
status = VMS4_SR;
list0 = (uint16_t)(status & 0xffff);
list1 = (uint16_t)((status >> 16) & 0xffff);
list2 = (uint16_t)((status >> 32) & 0xffff);
list3 = (uint16_t)((status >> 48) & 0xffff);
```

### Core Types

### Element Types
`vreg<T>`: `!pto.vreg<NxT>` Fixed-width PTO micro Instruction vector type with total width exactly 256 bytes (2048 bits). `N` is the lane count, `T` is the element type, and `N * bitwidth(T) = 2048`.

| Type | Bits | Description |
|------|------|-------------|
| `i8` / `si8` / `ui8` | 8 | Signless/signed/unsigned 8-bit integer |
| `i16` / `si16` / `ui16` | 16 | Signless/signed/unsigned 16-bit integer |
| `i32` / `si32` / `ui32` | 32 | Signless/signed/unsigned 32-bit integer |
| `i64` / `si64` / `ui64` | 64 | Signless/signed/unsigned 64-bit integer |
| `f16` | 16 | IEEE 754 half precision |
| `bf16` | 16 | Brain floating point |
| `f32` | 32 | IEEE 754 single precision |

### Mask Types

`mask<G>`: `!pto.mask<G>` Typed predicate-register view. `G` is one of `b8`, `b16`, `b32` and records the byte-granularity interpretation used by VPTO ops and verifiers.

Typed masks are also the primary legality contract for predicated VPTO code:

- vector ops over `f32`, `i32`, `si32`, and `ui32` consume `!pto.mask<b32>`
- vector ops over `f16`, `bf16`, `i16`, `si16`, and `ui16` consume
  `!pto.mask<b16>`
- vector ops over 8-bit element families consume `!pto.mask<b8>`
- compare families keep seed-mask and result-mask granularity aligned with the
  compared vector family
- carry families keep carry-in, carry-out, and execution-mask granularity
  aligned with the data-vector family
- mask-only ops that do not explicitly change granularity preserve the same `G`

### Address Space Conventions

PTO micro Instruction memory operands use `!pto.ptr<element-type, space>`. This specification models the following memory-space attributes:

| Space | Interpretation |
|-------|----------------|
| `gm` | Global Memory (GM), off-chip HBM/DDR storage |
| `ub` | Unified Buffer (UB), on-chip vector buffer |

Typical pointer construction and pointer arithmetic follow the same `!pto.ptr<..., space>` form:

```mlir
%0 = pto.castptr %c0 : i64 -> !pto.ptr<f32, ub>
%1 = pto.addptr %0, %c1024 : !pto.ptr<f32, ub> -> !pto.ptr<f32, ub>
```

### `!pto.ptr<T, space>`

`!pto.ptr<T, space>` is the typed pointer form used for explicit memory operands in PTO micro Instruction.

- `T` is the element type associated with the pointed-to storage.
- `space` is the memory domain, typically `gm` or `ub` in this specification.
- A `pto.ptr` value carries an address plus its element-type / memory-space interpretation, but it does not carry tensor shape or stride metadata by itself.
- Tensor semantics are introduced separately through view-building operations such as `pto.make_tensor_view`.
- Pointer arithmetic is element-based rather than byte-based.

Typical examples:

- `!pto.ptr<f32, gm>`
- `!pto.ptr<f32, ub>`
- `!pto.ptr<bf16, gm>`

### Pointer Operations

The complete contracts for `pto.castptr`, `pto.addptr`, `pto.load_scalar`, and
`pto.store_scalar` are documented in
[Special Scalar Operations](#typed-pointer-and-address-operations).

#### `pto.load`

- **syntax:** `%value = pto.load %ptr[%offset] : !pto.ptr<T, space> -> T`
- **semantics:** Load one scalar element from a VPTO pointer-like operand.

```c
value = ptr[offset];
```

- **inputs:**
  `%ptr` is a typed PTO pointer `!pto.ptr<T, space>` or a memref operand that
  will be normalized to a PTO pointer before LLVM emission. `%offset` is an
  `index` displacement counted in elements.
- **outputs:**
  `%value` is the loaded scalar element.
- **constraints and limitations:**
  The result type MUST match the element type of `%ptr`. This is the preferred
  scalar memory op for VPTO/SIMT authoring.

#### `pto.store`

- **syntax:** `pto.store %value, %ptr[%offset] : !pto.ptr<T, space>, T`
- **semantics:** Store one scalar element to a VPTO pointer-like operand.

```c
ptr[offset] = value;
```

- **inputs:**
  `%value` is the scalar value to store. `%ptr` is a typed PTO pointer
  `!pto.ptr<T, space>` or a memref operand that will be normalized to a PTO
  pointer before LLVM emission. `%offset` is an `index` displacement counted in
  elements.
- **constraints and limitations:**
  The stored value type MUST match the element type of `%ptr`. This is the
  preferred scalar memory op for VPTO/SIMT authoring.

The complete syntax, type restrictions, execution-scope rules, cache behavior,
target availability, and examples for `pto.ld_dev` and `pto.st_dev` are
documented in
[Special Scalar Operations](#aicore-scalar-gm-l1-bypass-operations).

#### Pointer-Based Vector Access Example

The following lowered-style fragment shows how typed PTO pointers flow through
pointer construction, pointer arithmetic, structured control flow, and PTO
memory ops. Scalar memory access is expressed on `!pto.ptr<T, space>` in
general, but the common VPTO pattern here is UB-local scalar access alongside
UB vector loads/stores:

```mlir
%0 = pto.castptr %c0 : i64 -> !pto.ptr<f32, ub>
%1 = pto.addptr %0, %c1024 : !pto.ptr<f32, ub> -> !pto.ptr<f32, ub>
pto.vecscope {
  %16 = scf.for %arg3 = %c0 to %11 step %c64 iter_args(%arg4 = %12) -> (i32) {
    %mask, %scalar_out = pto.plt_b32 %arg4 : i32 -> !pto.mask<b32>, i32
    %s = pto.load %1[%c4] : !pto.ptr<f32, ub> -> f32
    pto.store %s, %1[%c8] : !pto.ptr<f32, ub>, f32
    %17 = pto.vlds %1[%arg3] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %18 = pto.vabs %17, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %18, %10[%arg3], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
    scf.yield %scalar_out : i32
  }
}
```

In this pattern, `pto.castptr` materializes a typed UB pointer, `pto.addptr` shifts the base by 1024 `f32` elements, and the subsequent `[%arg3]` indexing on `pto.vlds` / `pto.vsts` applies an additional element offset relative to that base.

### Special Types

#### `!pto.mask<G>`

`!pto.mask<G>` models an A5 predicate register (256-bit) under a typed granularity view, not an integer vector.

`G` is part of the type and MUST be one of:

- `b32`
- `b16`
- `b8`

All three forms describe the same physical 256-bit predicate-register class. The type parameter does not encode how many lanes are currently active. Instead, it records how VPTO interprets the register when matching mask-producing ops, mask-consuming ops, and verifier legality rules.

In the ISA chapters below, this document uses `!pto.mask<G>` as shorthand when a
family is generic over granularity. For op families whose names already encode
the granularity, such as `pset_b32`, `pge_b16`, `plt_b8`,
`pdintlv_b8`, and `pintlv_b16`, examples use the corresponding concrete typed
mask.

**Mask Granularity:**

The predicate register is 256 bits in length, where each bit controls 1 byte of data. `G` therefore describes how many bytes form one logical element slot:

| Mask Type | Bytes / Element Slot | Typical Element Family | Derived Logical Lanes |
|-----------|----------------------|------------------------|-----------------------|
| `!pto.mask<b32>` | 4 | `f32` / `i32` | 64 |
| `!pto.mask<b16>` | 2 | `f16` / `bf16` / `i16` | 128 |
| `!pto.mask<b8>` | 1 | 8-bit element family | 256 |

This is intentionally different from a lane-vector model such as `mask<64xi1>`:

- `!pto.mask<b32>` still denotes a 256-bit predicate register;
- `64` is only the derived logical lane count for the `b32` view;
- value-level patterns such as `PAT_VL32` describe which lanes are active, not a different type.
- `pto.vaddc`, `pto.vsubc`, `pto.vaddcs`, and `pto.vsubcs` use `!pto.mask<G>`
  to carry their per-lane carry results, interpreted with this same
  granularity.

**Predication Behavior (Zero-Merge):**

The native hardware predication mode is **ZEROING** — inactive lanes produce zero:

```c
dst[i] = mask[i] ? op(src0[i], src1[i]) : 0    // ZEROING mode
```

```mlir
// Predicated add: inactive lanes produce zero
%mask = pto.pset_b32 "PAT_VL32" : !pto.mask<b32>   // first 32 logical b32 lanes active
%result = pto.vcmp %a, %b, %mask, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>
```

```mlir
// Compare and select: generate mask from comparison, use for conditional select
%mask = pto.vcmp %lhs, %rhs, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>
%out = pto.vsel %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

#### `!pto.align`

`!pto.align` models the A5 vector-align carrier state. It is not payload data.

```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align_out = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align

%store_align = pto.init_align : !pto.align
%next_align = pto.vstus %store_align, %offset, %vec, %ub
    : !pto.align, i32, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align
```

## Part II: Notation Convention

This section defines the MLIR syntax patterns and C-style semantic notation used throughout the ISA reference (Part III).

### MLIR Op Syntax Patterns

All PTO micro Instruction operations follow standard MLIR syntax. The common patterns are:

**Unary (one vector in, one vector out):**

```mlir
%result = pto.<op> %input : !pto.vreg<NxT> -> !pto.vreg<NxT>
```

**Binary (two vectors in, one vector out):**

```mlir
%result = pto.<op> %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

**Vec-Scalar (one vector + one scalar in, one vector out):**

```mlir
%result = pto.<op> %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>
```

**Load (memory to register):**

```mlir
%result = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>
%result, %updated_base = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>, !pto.ptr<T, ub>
```

**Store (register to memory):**

```mlir
pto.vsts %value, %destination[%offset] {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>
%updated_base = pto.vsts %value, %destination[%offset] {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.ptr<T, ub>
```

**Dual Load (one load, two results — deinterleave):**

```mlir
%low, %high = pto.vldsx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

**Dual Store (two inputs, one interleaved store):**

```mlir
pto.vstsx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask<G>
```

**Compare (two vectors + seed mask in, mask out):**

```mlir
%mask = pto.vcmp %src0, %src1, %seed, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.mask<G>
```

**Conversion (one vector in, different-typed vector out):**

```mlir
%result = pto.vcvt %input, %mask {rnd = "R", sat = "SAT", part = "EVEN"} : !pto.vreg<NxT0>, !pto.mask<G> -> !pto.vreg<MxT1>
```

**Predicate construction:**

```mlir
%mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
%tail = pto.pge_b32 "PAT_VL16" : !pto.mask<b32>
```

**Sync operations:**

```mlir
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

**Pointer construction and arithmetic:**

```mlir
%ptr = pto.castptr %addr : i64 -> !pto.ptr<T, SPACE>
%ptr2 = pto.addptr %ptr, %offset : !pto.ptr<T, SPACE> -> !pto.ptr<T, SPACE>
```

### Shared Dialect Syntax Patterns

PTO micro Instruction programs may interleave PTO ops with standard MLIR `arith` and `scf` ops.
The examples below emphasize common index-heavy patterns, but `arith` support is not limited to index arithmetic.

**Scalar / index constant:**

```mlir
%c0 = arith.constant 0 : index
%zero = arith.constant 0.0 : f32
```

**Scalar arithmetic (integer / float / boolean-style bitwise):**

```mlir
%sum_i = arith.addi %lhs_i, %rhs_i : i32
%sum_f = arith.addf %lhs_f, %rhs_f : f32
%bits = arith.andi %flags0, %flags1 : i32
```

**Scalar compare and select:**

```mlir
%cond = arith.cmpi eq, %lhs, %rhs : index
%bound = arith.select %cond, %a, %b : index
```

**Counted loop with loop-carried values:**

```mlir
%result = scf.for %iv = %lb to %ub step %step
    iter_args(%acc = %init) -> (index) {
  %next = arith.addi %acc, %iv : index
  scf.yield %next : index
}
```

**Structured conditional region:**

```mlir
%selected = scf.if %cond -> (index) {
  scf.yield %then_value : index
} else {
  scf.yield %else_value : index
}
```

**Structured while loop:**

```mlir
%state:2 = scf.while (%iv = %c0, %alive = %true) : (index, i1) -> (index, i1) {
  %keep_going = arith.cmpi slt, %iv, %limit : index
  scf.condition(%keep_going) %iv, %alive : index, i1
} do {
^bb0(%iv_in: index, %alive_in: i1):
  %iv_next = arith.addi %iv_in, %c1 : index
  scf.yield %iv_next, %alive_in : index, i1
}
```

### C-Style Semantics Convention

For each ISA operation in Part III, semantics are expressed as C code. The convention:

```c
// Vector register contents as arrays:
T dst[N];       // destination
T src0[N];      // first source
T src1[N];      // second source (binary ops)
T scalar;       // scalar operand (vec-scalar ops)
int mask[N];    // per-lane predicate (0 or 1)

// N = lane count determined by type:
//   N = 256 for i8/si8/ui8
//   N = 128 for i16/si16/ui16/f16/bf16
//   N = 64  for i32/si32/ui32/f32
//   N = 32  for i64/si64/ui64
```

**Example — pto.vadd semantics:**

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] + src1[i];
```

**Example — pto.vcgadd (group reduction per VLane) semantics:**

```c
int groups = 8;
int K = 32 / sizeof(T);  // elements per 32-byte VLane
for (int g = 0; g < 8; g++) {
    T sum = 0;
    for (int i = 0; i < K; i++)
        if (mask[g*K + i])
            sum += src[g*K + i];
    dst[g] = sum;
}
for (int i = groups; i < N; i++)
    dst[i] = 0;
```

For A5 reduction result types:

- `pto.vcadd` widens `i8 -> i16`, `u8 -> u16`, `i16 -> i32`, and `u16 -> u32`,
  with the lane count halved in each widening case.
- `pto.vcadd` keeps the same result type for `f16`, `f32`, `i32`, and `u32`.

### Template Placeholder Conventions

| Placeholder | Meaning |
|-------------|---------|
| `"SRC_PIPE"`, `"DST_PIPE"` | Pipeline identifiers: `"PIPE_MTE2"`, `"PIPE_V"`, `"PIPE_MTE3"` |
| `"EVENT_ID"` | Event identifier: `"EVENT_ID0"` etc. |
| `"DIST"` | Distribution mode string (see the relevant load/store ISA group in Part III) |
| `"CMP_MODE"` | Compare predicate: `eq \| ne \| lt \| le \| gt \| ge` |
| `"RND"` | Rounding mode: `R \| A \| F \| C \| Z \| O` |
| `"SAT"` | Saturation: `SAT \| NOSAT` |
| `"PART"` | Half selector: `EVEN \| ODD` |
| `"PAT_*"` | Predicate pattern literal |
| `T` | Element type (f32, f16, bf16, i32, i16, i8, etc.) |
| `N` | Lane count (`N * bitwidth(T) = 2048`) |

## Part III: ISA Instruction Reference — Summary

This section provides a categorized overview of all PTO micro Instruction operations plus the shared MLIR `arith` and `scf` ops that may appear in PTO micro Instruction programs. Detailed documentation for each group is available in the linked files.

## Instruction Groups

| # | Group | Description | Count | Details |
|---|-------|-------------|-------|---------|
| 1 | [Pipeline Sync](#micro-01-pipeline-sync) | Intra-core pipeline synchronization | 5 | `pto.set_flag`, `pto.wait_flag`, `pto.pipe_barrier`, `pto.get_buf`, `pto.rls_buf` |
| 2 | [DMA Copy Programming](#micro-02-dma-copy) | Public DMA transfer interface between GM↔UB, UB→UB, and UB→L1 | 4 | `pto.mte_gm_ub`, `pto.mte_ub_gm`, `pto.mte_ub_ub`, `pto.mte_ub_l1` |
| 3 | [Vector Load/Store](#micro-03-vector-load-store) | UB↔vreg data movement with various access patterns | ~23 | `pto.vlds`, `pto.vldsx2`, `pto.vgather2`, `pto.vsts`, `pto.vstsx2`, `pto.vscatter`, `pto.sprclr`, `pto.sprsti`, `pto.sprsts`, etc. |
| 4 | [Predicate Load/Store](#micro-04-predicate-load-store) | UB↔mask register movement | 5 | `pto.plds`, `pto.pldi`, `pto.psts`, `pto.psti`, `pto.pstu` |
| 5 | [Materialization & Predicate Ops](#micro-05-materialization-predicate) | Scalar broadcast, predicate generation and manipulation | ~20 | `pto.vbr`, `pto.vdup`, `pto.pset_b*`, `pto.pge_b*`, `pto.plt_b*`, `pto.pltm_b*`, `pto.ppack`, `pto.punpack`, `pto.pnot`, `pto.psel`, etc. |
| 6 | [Unary Vector Ops](#micro-06-unary-vector-ops) | Single-input element-wise operations | 7 | `pto.vabs`, `pto.vneg`, `pto.vexp`, `pto.vln`, `pto.vsqrt`, `pto.vrelu`, `pto.vnot` |
| 7 | [Binary Vector Ops](#micro-07-binary-vector-ops) | Two-input element-wise operations | 14 | `pto.vadd`, `pto.vsub`, `pto.vmul`, `pto.vdiv`, `pto.vmax`, `pto.vmin`, `pto.vmadd`, `pto.vand`, `pto.vor`, `pto.vxor`, `pto.vshl`, `pto.vshr`, `pto.vaddc`, `pto.vsubc` |
| 8 | [Vec-Scalar Ops](#micro-08-vec-scalar-ops) | Vector-scalar operations | 9 | `pto.vadds`, `pto.vmuls`, `pto.vmaxs`, `pto.vmins`, `pto.vlrelu`, `pto.vshls`, `pto.vshrs`, `pto.vaddcs`, `pto.vsubcs` |
| 9 | [Conversion Ops](#micro-09-conversion-ops) | Type conversion with rounding/saturation control | 4 | `pto.vcvt`, `pto.vtrc`, `pto.vbitcast`, `pto.pbitcast` |
| 10 | [Reduction Ops](#micro-10-reduction-ops) | Vector reductions | 11 | `pto.vcadd`, `pto.vcmax`, `pto.vcmin`, `pto.vcbmax`, `pto.vcbmin`, `pto.vcgadd`, `pto.vcgmax`, `pto.vcgmin`, `pto.vcpadd`, `pto.chistv2`, `pto.dhistv2` |
| 11 | [Compare & Select](#micro-11-compare-select) | Comparison and conditional selection | 4 (+1 not A5) | `pto.vcmp`, `pto.vcmps`, `pto.vsel`, `pto.vselr` (`pto.vselrv2` removed: not A5) |
| 12 | [Data Rearrangement](#micro-12-data-rearrangement) | In-register data movement and permutation | 2 (+2 not A5) | `pto.vintlv`, `pto.vdintlv` (`pto.vintlvv2`, `pto.vdintlvv2` removed: not A5) |
| 13 | [DSA/SFU Ops](#micro-13-dsa-sfu-ops) | Specialized ops, index generation, and sorting helpers | 11 | `pto.vlrelu`, `pto.vprelu`, `pto.vexpdif`, `pto.vaxpy`, `pto.vmulscvt`, `pto.vmull`, `pto.vmula`, `pto.vci`, `pto.vbitsort`, `pto.vmrgsort4`, `pto.get_vms4_sr` |
| 14 | [Arith (Shared MLIR Dialect)](#micro-14-shared-arith) | Full scalar `arith` surface used around PTO ops; the companion page lists categories and representative examples | all scalar ops | `arith.constant`, `arith.addi`, `arith.addf`, `arith.cmpi`, `arith.cmpf`, `arith.select`, `arith.index_cast`, `arith.extsi`, `arith.trunci`, `arith.andi`, `arith.shli`, etc. |
| 15 | [SCF (Shared MLIR Dialect)](#micro-15-shared-scf) | Structured loops, branches, and loop-carried state around PTO regions | 5 | `scf.for`, `scf.if`, `scf.while`, `scf.condition`, `scf.yield` |
| 16 | [Cube Matrix Multiply](#micro-16-cube-matmul) | GM↔L1 (`l1`/cbuf) staging, L1 (`l1`)↔UB/BT/FB side moves, L1→L0A/L0B loads, L0C (`l0c`) matmul, and FIXPIPE MTE writeback | 19 | `pto.mte_gm_l1`, `pto.mte_l1_ub`, `pto.mte_gm_l1_frac`, `pto.mte_l1_bt`, `pto.mte_l1_fb`, `pto.mte_l1_l0a`, `pto.mte_l1_l0b`, `pto.mte_l1_l0a_mx`, `pto.mte_l1_l0b_mx`, `pto.mad`, `pto.mad_acc`, `pto.mad_bias`, `pto.mad_mx`, `pto.mad_mx_acc`, `pto.mad_mx_bias`, `pto.mte_l0c_l1`, `pto.mte_l0c_gm`, `pto.mte_l0c_ub` |
| 17 | [SIMT Ops](#micro-17-simt) | SIMT launch, thread/lane queries, vote/shuffle/redux, scalar memory, atomics, scalar math, conversion, entry synchronization, and state preservation | ~65 | `pto.store_vfsimt_info`, `pto.simt_launch`, `pto.get_tid_x`, `pto.get_laneid`, `pto.vote_*`, `pto.shuffle_*`, `pto.redux_*`, `pto.load`, `pto.store`, `pto.atomic_*`, `pto.convert`, `pto.syncthreads`, `pto.keep`, `pto.resume`, etc. |
| 18 | [Special Scalar Operations](#micro-18-special-scalar) | PTO scalar kernel queries, typed pointer/address calculation, scalar-pipeline memory, and ordinary AICore GM L1-bypass access | 10 | `pto.get_block_idx`, `pto.get_subblock_idx`, `pto.get_block_num`, `pto.get_subblock_num`, `pto.castptr`, `pto.addptr`, `pto.load_scalar`, `pto.store_scalar`, `pto.ld_dev`, `pto.st_dev` |

## Detailed ISA Group Reference

This section inlines the 18 ISA group documents so the architectural overview, notation, summary table, and per-group semantics can be read in a single file.

<a id="micro-01-pipeline-sync"></a>

### 1. Pipeline Synchronization

> **Category:** Synchronization primitives for coordinating pipeline execution
> **Pipelines:** MTE2 (GM→UB), PIPE_V (Vector), MTE3 (UB→GM)

The PTO micro Instruction model operates on the Ascend 950's **Decoupled Access-Execute** architecture. The MTE and Vector pipelines run asynchronously, requiring explicit synchronization to prevent data hazards.

---

#### Intra-Core Pipeline Sync

These ops coordinate data flow between pipelines within a single vector core.

##### `pto.set_flag`

- **syntax:** `pto.set_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- **semantics:** Signal event from source pipe to destination pipe.

```c
set_flag(src_pipe, dst_pipe, event_id);
```

**Example:** After MTE2 completes GM→UB transfer, signal Vector pipe:
```mlir
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

---

##### `pto.wait_flag`

- **syntax:** `pto.wait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- **semantics:** Block destination pipe until source pipe signals event.

```c
wait_flag(src_pipe, dst_pipe, event_id);
```

**Example:** Vector pipe waits for MTE2 data to arrive:
```mlir
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

---

##### `pto.pipe_barrier`

- **syntax:** `pto.pipe_barrier "PIPE_*"`
- **semantics:** Drain all pending ops in the specified pipe. All previously issued operations on that pipe complete before any subsequent operation begins.

```c
pipe_barrier(pipe);
```

**Pipe identifiers:** `PIPE_MTE2`, `PIPE_V`, `PIPE_MTE3`

**Example:** Two back-to-back `pto.mte_ub_gm` calls writing to the same GM address. Without a barrier, MTE3 may reorder them and the final GM value is non-deterministic:

```mlir
// Both stores target the same GM address — order matters!
pto.mte_ub_gm %ub_partial_0, %gm_result, %len_burst ...
// Without pipe_barrier, MTE3 could execute the second copy before the first
// completes, producing a non-deterministic result at %gm_result.
pto.pipe_barrier "PIPE_MTE3"
// After barrier: first copy is guaranteed complete. Second copy overwrites deterministically.
pto.mte_ub_gm %ub_partial_1, %gm_result, %len_burst ...
```

---

##### `pto.get_buf`

- **syntax:** `pto.get_buf "PIPE_*", %buf_id, %mode : i64, i64`
- **semantics:** Acquire buffer slot for inter-pipeline double-buffering coordination.

```c
get_buf(pipe, buf_id, mode);
```

---

##### `pto.rls_buf`

- **syntax:** `pto.rls_buf "PIPE_*", %buf_id, %mode : i64, i64`
- **semantics:** Release buffer slot to allow other pipeline to proceed.

```c
rls_buf(pipe, buf_id, mode);
```

---

##### Mode Parameter for `get_buf` / `rls_buf`

The `mode` parameter controls how `get_buf` and `rls_buf` interact with pipeline execution and dependency tracking:

| Mode | `get_buf` Behavior | `rls_buf` Behavior | Use Case |
|------|-------------------|-------------------|----------|
| **0** (default) | **Blocking acquire**: waits for all previous `rls_buf` with same `buf_id` from all pipelines (in program order) before the specified pipe can proceed | **Immediate release**: signals completion for only the instructions related to the specified pipe | **Automatic ping/pong dependency** — recommended for double/multi-buffering |
| **1** | **Non-blocking acquire**: does not wait; pipe execution proceeds immediately | **Deferred release**: waits for all instructions across all pipelines with same `buf_id` to retire before signaling | **Backward compatibility** with `set_flag`/`wait_flag` semantics |

**Mode 0 (Default — Recommended):**
- `get_buf`: The specified pipeline blocks until all previous `rls_buf` operations for the same buffer ID (from any pipeline) have completed, respecting program order.
- `rls_buf`: Immediately signals that the specified pipeline has finished using the buffer — only waits for that pipe's related instructions.
- This mode provides **automatic RAW/WAR/WAW dependency resolution** based on buffer ID and program order, making it ideal for ping/pong and N-buffer patterns.

**Mode 1 (Legacy Compatibility):**
- `get_buf`: Does not block — the pipeline proceeds immediately without waiting.
- `rls_buf`: Waits for **all** previous instructions across **all** pipelines with the same buffer ID to retire before signaling release.
- This mode emulates `set_flag`/`wait_flag` behavior and is provided for backward compatibility with existing code patterns.

> **Note:** A5 supports both `set_flag`/`wait_flag` and `get_buf`/`rls_buf` mechanisms. Mode 1 is rarely needed since mode 0 provides a more programmer-friendly approach for buffer-based synchronization.

---

##### `pto.get_buf_dyn` / `pto.rls_buf_dyn`

Dynamic variants of `get_buf`/`rls_buf` where `buf_id` is provided as an SSA value instead of a static integer attribute. This enables runtime-computed buf_id patterns such as SIMT ping-pong buffering.

- **syntax:**
  - String shorthand: `pto.get_buf_dyn "PIPE_MTE2", %buf_id, 0`
  - Bracket form: `pto.get_buf_dyn [TLOAD, %buf_id, 0]`
- **semantics:** Same as `get_buf`/`rls_buf`, but the buffer-id is an `index`-typed SSA value resolved at runtime.
- **inputs:**
  - `op_type`: same pipe-like attribute as the static form
  - `buf_id`: an SSA value of `index` type (e.g. `iter & 1` for ping-pong)
  - `mode`: same mode parameter (default `0`)
- **constraints and limitations:** The BufidSync auto-insertion pass only uses the static form (`get_buf`/`rls_buf`). Use the dynamic form (`get_buf_dyn`/`rls_buf_dyn`) when buf_id must be computed at runtime.

Example (SIMT double-buffering with `iter & 1`):

```mlir
  %c1 = arith.constant 1 : index
  %buf_id = arith.andi %iter, %c1 : index
  pto.get_buf_dyn [TLOAD, %buf_id, 0]
  // ... tload to ubuf slot %buf_id ...
  pto.rls_buf_dyn [TLOAD, %buf_id, 0]
```

---

##### `pto.mem_bar`

- **syntax:** `pto.mem_bar "BARRIER_TYPE"`
- **semantics:** Shared-memory (UB address space) memory fence within `__VEC_SCOPE__`. Required when UB addresses alias between memory operations. The barrier type selects which classes of prior instructions must complete before which classes of subsequent instructions may proceed.

```c
mem_bar(barrier_type);
```

**Barrier types** are organized into three families by the scope of prior vs. subsequent instructions:

| Family | Barrier type | Prior instructions | Subsequent instructions |
|--------|-------------|-------------------|------------------------|
| **VV** (vector→vector) | `VV_ALL` | All vector load/store | All vector load/store |
| | `VST_VLD` | All vector store | All vector load |
| | `VLD_VST` | All vector load | All vector store |
| | `VST_VST` | All vector store | All vector store |
| **VS** (vector→scalar) | `VS_ALL` | All vector load/store | All scalar load/store |
| | `VST_LD` | All vector store | All scalar load |
| | `VLD_ST` | All vector load | All scalar store |
| | `VST_ST` | All vector store | All scalar store |
| **SV** (scalar→vector) | `SV_ALL` | All scalar load/store | All vector load/store |
| | `ST_VLD` | All scalar store | All vector load |
| | `LD_VST` | All scalar load | All vector store |
| | `ST_VST` | All scalar store | All vector store |

**Example:** Ensure vector stores are visible before subsequent vector loads to the same UB region:
```mlir
pto.vsts %v0, %ub[%c0] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>
pto.mem_bar "VST_VLD"
%v1 = pto.vlds %ub[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

---

##### `pto.dsb`

- **syntax:** `pto.dsb "MEM_DOMAIN"`
- **semantics:** Issues a data synchronization barrier for the selected memory domain. All prior memory effects covered by the domain must become complete before subsequent memory effects proceed.

**Memory domains:**

| Domain | Meaning |
|--------|---------|
| `ALL` | Wait for all memory-access classes covered by the target |
| `DDR` | Wait for DDR/GM memory-access effects |
| `UB` | Wait for UB memory-access effects |
| `SEQ` | Wait for sequencer-visible memory-access effects |

**Example:** Ensure prior GM stores are complete before publishing a scalar signal:

```mlir
pto.dsb "DDR"
```

---

##### `pto.dcci`

- **syntax:** `pto.dcci %ptr "CACHE_SCOPE" : !pto.ptr<T, gm|ub>`
- **syntax:** `pto.dcci %ptr "CACHE_SCOPE", "CACHE_DST" : !pto.ptr<T, gm|ub>`
- **semantics:** Performs data-cache clean/invalidate maintenance for the selected cache scope. The pointer address space selects whether the operation applies to GM or UB-backed cache state. The optional destination domain further restricts which cache-line class is affected.

**Cache scopes:**

| Scope | Meaning |
|-------|---------|
| `SINGLE_CACHE_LINE` | Apply maintenance to the cache line containing `%ptr` |
| `ENTIRE_DATA_CACHE` | Apply maintenance to the entire data cache; `%ptr` is still required by the IR form |

**Destination domains:**

| Destination | Meaning |
|-------------|---------|
| `CACHELINE_ALL` | All supported cache-line domains |
| `CACHELINE_UB` | UB cache-line domain |
| `CACHELINE_OUT` | Output/GM-visible cache-line domain |
| `CACHELINE_ATOMIC` | Atomic cache-line domain |

**Constraints:**
- `%ptr` must be a PTO pointer or buffer-like value in GM or UB address space.
- Omitting `CACHE_DST` uses the target's default destination-domain form.

**Example:** Flush GM-visible cache state after scalar GM stores:

```mlir
pto.dcci %gm "ENTIRE_DATA_CACHE", "CACHELINE_OUT" : !pto.ptr<i8, gm>
pto.dsb "ALL"
```

---

#### Why `get_buf` / `rls_buf` is More Programmer-Friendly

The buffer-based synchronization (`get_buf`/`rls_buf`) provides the **same functional capability** as `set_flag`/`wait_flag` for maintaining correct ordering of RAW/WAR/WAW dependencies across pipelines, but with significant usability advantages:

##### 1. No Manual Priming or Draining

With `set_flag`/`wait_flag`, ping/pong loops require:
- **Pre-loop priming**: 4× `set_flag` to initialize reverse-dependency signals (otherwise first iteration deadlocks)
- **Post-loop draining**: 4× `wait_flag` to consume leftover signals from final iterations

With `get_buf`/`rls_buf`:
- **First iteration**: Buffer is initially free, so `get_buf` proceeds immediately — no priming needed
- **Final iteration**: Last `rls_buf` simply completes — no draining required

##### 2. No Loop Peeling for Complex Dependencies

For non-1:1 producer-consumer ratios (e.g., 1 MTE2 load : N Vector compute slices), `set_flag`/`wait_flag` requires **peeling the set_flag outside the loop**:

```mlir
// set_flag/wait_flag: 1 MTE2 load, 8 Vector computes on slices
// MTE2 loads large tile once
pto.mte_gm_ub %gm_ptr, %ub_tile, ...
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVT_TILE_READY"]  // ◀ MUST be outside loop

// Vector consumes in 8 slices — but wait_flag can only fire ONCE
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVT_TILE_READY"] // ◀ MUST peel before loop
scf.for %slice = %c0 to %c8 step %c1 {
  // compute on %ub_tile[%slice]
  // Cannot put wait_flag here — would deadlock on iteration 1+
}
```

With `get_buf`/`rls_buf`, acquire/release can be **inside the loop** — no peeling needed:

```mlir
// get_buf/rls_buf: same 1:8 pattern, acquire/release inside loop works fine
// MTE2 loads large tile
pto.get_buf "PIPE_MTE2", %bufid_tile, %c0 : i64, i64
pto.mte_gm_ub %gm_ptr, %ub_tile, ...
pto.rls_buf "PIPE_MTE2", %bufid_tile, %c0 : i64, i64

// Vector acquires/releases per slice — all 8 iterations work correctly
scf.for %slice = %c0 to %c8 step %c1 {
  pto.get_buf "PIPE_V", %bufid_tile, %c0 : i64, i64  // iteration 0: blocks until MTE2 done
                                                      // iteration 1-7: proceeds immediately (already acquired)
  // compute on %ub_tile[%slice]
  pto.rls_buf "PIPE_V", %bufid_tile, %c0 : i64, i64
}
// No peeling required — get_buf handles the MTE2→V dependency automatically
```

##### 3. Simpler Mental Model

| Aspect | `set_flag`/`wait_flag` | `get_buf`/`rls_buf` |
|--------|------------------------|---------------------|
| **Dependency tracking** | Manual: track event IDs, signal directions, pair every set with wait | Automatic: buffer ID + program order |
| **Event ID management** | **8 IDs per pipe-pair direction** (HW limit); each buffer occupies 1 ID per direction | **1 buffer ID per shared resource** (HW limit: 32 global); same ID used across all pipelines |
| **Error-prone areas** | Forgetting prime/drain, mismatched IDs, wrong direction | Forgetting release (but compile-time checkable) |

##### Quick Example Comparison

**Problem:** MTE2 loads into `buf[i%2]`, Vector processes, MTE3 stores — standard ping/pong.

**set_flag/wait_flag approach:**
```mlir
// BEFORE loop: prime 4 reverse-dep signals
pto.set_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_0"]
pto.set_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_1"]
pto.set_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_0"]
pto.set_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_1"]

scf.for %i = ... {
  // 4 set_flag + 4 wait_flag inside loop
  // Must track 4 IDs: 2 pipe-pair directions × 2 ping/pong buffers
}

// AFTER loop: drain 4 signals
pto.wait_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_0"]
pto.wait_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_1"]
pto.wait_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_0"]
pto.wait_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_1"]
```

**get_buf/rls_buf approach:**
```mlir
scf.for %i = ... {
  pto.get_buf %bufid_in[%pp], "PIPE_MTE2"
  // ... MTE2 work ...
  pto.rls_buf %bufid_in[%pp], "PIPE_MTE2"

  pto.get_buf %bufid_in[%pp], "PIPE_V"
  pto.get_buf %bufid_out[%pp], "PIPE_V"
  // ... Vector work ...
  pto.rls_buf %bufid_in[%pp], "PIPE_V"
  pto.rls_buf %bufid_out[%pp], "PIPE_V"

  pto.get_buf %bufid_out[%pp], "PIPE_MTE3"
  // ... MTE3 work ...
  pto.rls_buf %bufid_out[%pp], "PIPE_MTE3"
}
// Done. No prime. No drain. Dependencies resolved by buffer ID + program order.
```

---

#### Intra-Core Sync Patterns & Examples

##### Example 1: `set_flag` / `wait_flag` (Explicit Events)

Each cross-pipeline data dependency requires an explicit signal/wait pair. The programmer must manually insert `set_flag` after the producer and `wait_flag` before the consumer.

```mlir
// ─── Stage 1: MTE2 loads data from GM into UB ───
pto.mte_gm_ub %gm_ptr, %ub_ptr, ...

// MTE2 signals: "UB data is ready for Vector pipe"
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

// ─── Stage 2: Vector pipe consumes UB data ───
// Vector waits until MTE2's signal arrives
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

pto.vecscope {
  %v   = pto.vlds %ub_ptr[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
}

// Vector signals: "UB output is ready for MTE3"
pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

// ─── Stage 3: MTE3 stores result from UB back to GM ───
// MTE3 waits until Vector's signal arrives
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

pto.mte_ub_gm %ub_out, %gm_out, %len_burst ...
```

**Key property:** Every cross-pipeline edge is an explicit `(set_flag, wait_flag)` pair. Simple for straight-line code, but gets verbose in loops (see Example 3).

---

##### Example 2: `get_buf` / `rls_buf` (Resource-Based)

Instead of naming events, each pipeline declares when it **acquires** (`get_buf`) and **releases** (`rls_buf`) a shared UB buffer. Cross-pipeline RAW/WAR dependencies are resolved implicitly by program order — if MTE2 releases `buf_A` and Vector later acquires `buf_A`, the hardware ensures the acquire cannot proceed until the release completes.

```mlir
// ─── Stage 1: MTE2 loads data into UB ───
// MTE2 acquires ub_ptr — blocks if Vector hasn't released it from a prior iteration
pto.get_buf "PIPE_MTE2", %bufid_ub_ptr, %c0 : i64, i64   // mode=0 (default)
pto.mte_gm_ub %gm_ptr, %ub_ptr, ...
// MTE2 done writing ub_ptr — release it so Vector can consume
pto.rls_buf "PIPE_MTE2", %bufid_ub_ptr, %c0 : i64, i64

// ─── Stage 2: Vector computation ───
// Vector acquires ub_ptr (input) — blocks until MTE2 releases it (RAW: MTE2 write → V read)
pto.get_buf "PIPE_V", %bufid_ub_ptr, %c0 : i64, i64
// Vector acquires ub_out (output) — blocks until MTE3 releases it from a prior iteration (WAR: MTE3 read → V write)
pto.get_buf "PIPE_V", %bufid_ub_out, %c0 : i64, i64

pto.vecscope {
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %v   = pto.vlds %ub_ptr[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
}

// Vector done reading ub_ptr — release so MTE2 can reuse it in next iteration
pto.rls_buf "PIPE_V", %bufid_ub_ptr, %c0 : i64, i64
// Vector done writing ub_out — release so MTE3 can consume
pto.rls_buf "PIPE_V", %bufid_ub_out, %c0 : i64, i64

// ─── Stage 3: MTE3 stores result to GM ───
// MTE3 acquires ub_out — blocks until Vector releases it (RAW: V write → MTE3 read)
pto.get_buf "PIPE_MTE3", %bufid_ub_out, %c0 : i64, i64
pto.mte_ub_gm %ub_out, %gm_out, %len_burst ...
// MTE3 done reading ub_out — release so Vector can reuse it in next iteration
pto.rls_buf "PIPE_MTE3", %bufid_ub_out, %c0 : i64, i64
```

**Key property:** No event IDs needed. Dependencies are implicit from program order of `get_buf`/`rls_buf` on the same buffer ID. This becomes much more convenient in multi-iteration loops (see Example 3).

---

##### Example 3: Ping/Pong Double-Buffering Loop

Double-buffering overlaps DMA and compute by using two UB buffers alternately. All three stages (MTE2, Vector, MTE3) appear in the **same iteration** — the hardware pipelines them across iterations because different iterations operate on different buffers (`buf[i%2]`).

###### Event ID scheme (`set_flag` / `wait_flag`)

With 2 ping/pong buffers and 2 pipeline pairs (MTE2↔V, V↔MTE3), `set_flag`/`wait_flag` needs **8 event IDs** = 2 pipe-pairs × 2 buffers × (forward + reverse):

**MTE2 ↔ Vector (input buffers):**

| Event ID | Direction | Purpose |
|----------|-----------|---------|
| `EVT_IN_FWD_0` | MTE2 → V | RAW: buf_in[0] data ready |
| `EVT_IN_FWD_1` | MTE2 → V | RAW: buf_in[1] data ready |
| `EVT_IN_REV_0` | V → MTE2 | WAR: Vector done reading buf_in[0] |
| `EVT_IN_REV_1` | V → MTE2 | WAR: Vector done reading buf_in[1] |

**Vector ↔ MTE3 (output buffers):**

| Event ID | Direction | Purpose |
|----------|-----------|---------|
| `EVT_OUT_FWD_0` | V → MTE3 | RAW: buf_out[0] result ready |
| `EVT_OUT_FWD_1` | V → MTE3 | RAW: buf_out[1] result ready |
| `EVT_OUT_REV_0` | MTE3 → V | WAR: MTE3 done reading buf_out[0] |
| `EVT_OUT_REV_1` | MTE3 → V | WAR: MTE3 done reading buf_out[1] |

###### 3a. `set_flag` / `wait_flag` version

```mlir
// ═══ Pre-loop: prime ALL reverse-dependency signals ═══
// Both input and output buffers start unused. We must pre-send
// reverse-dep signals so the first iteration's wait_flags don't deadlock.
pto.set_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_0"]   // ◀ PRIME: buf_in[0] "free"
pto.set_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_1"]   // ◀ PRIME: buf_in[1] "free"
pto.set_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_0"]  // ◀ PRIME: buf_out[0] "free"
pto.set_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_1"]  // ◀ PRIME: buf_out[1] "free"

scf.for %i = %c0 to %N step %c1 {
  // ── All 3 stages in same iteration, indexed by i%2 ──
  // %pp = i % 2  (ping/pong selector for buffer & event IDs)

  // ── MTE2: load tile[i] into buf_in[i%2] ──
  // WAR: wait until Vector has released buf_in[i%2] from iteration i-2
  pto.wait_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_{pp}"]
  pto.mte_gm_ub %gm_ptr[%i], %ub_in[%pp], ...
  // RAW: signal Vector that buf_in[i%2] data is ready
  pto.set_flag["PIPE_MTE2", "PIPE_V", "EVT_IN_FWD_{pp}"]

  // ── Vector: compute buf_in[i%2] → buf_out[i%2] ──
  // RAW: wait for MTE2 to finish loading buf_in[i%2]
  pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVT_IN_FWD_{pp}"]
  // WAR: wait for MTE3 to finish reading buf_out[i%2] from iteration i-2
  pto.wait_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_{pp}"]
  scf.for %dummy = %c0 to %c1 step %c1 {
    %v   = pto.vlds %ub_in[%pp][%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %abs, %ub_out[%pp][%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
  } {llvm.loop.aivector_scope}
  // WAR: tell MTE2 "done reading buf_in[i%2]"
  pto.set_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_{pp}"]
  // RAW: tell MTE3 "buf_out[i%2] result ready"
  pto.set_flag["PIPE_V", "PIPE_MTE3", "EVT_OUT_FWD_{pp}"]

  // ── MTE3: store result from buf_out[i%2] to GM ──
  // RAW: wait for Vector to finish writing buf_out[i%2]
  pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVT_OUT_FWD_{pp}"]
  pto.mte_ub_gm %ub_out[%pp], %gm_out[%i], %len_burst ...
  // WAR: tell Vector "done reading buf_out[i%2]"
  pto.set_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_{pp}"]
}

// ═══ Post-loop: drain — match every pre-loop prime with a wait ═══
// Each priming set_flag must be paired. The last loop iteration's
// set_flags are consumed by wait_flags that will never fire inside the
// loop (there is no iteration i+2). Drain them here.
pto.wait_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_{(N-1)%2}"]  // ◀ DRAIN
pto.wait_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_{(N-2)%2}"]  // ◀ DRAIN
pto.wait_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_{(N-1)%2}"] // ◀ DRAIN
pto.wait_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_{(N-2)%2}"] // ◀ DRAIN
```

**What `set_flag`/`wait_flag` requires outside the loop:**
- **Before the loop (4 × `set_flag`):** Prime every reverse-dependency event ID — one per buffer per pipe-pair. Without this, the first iteration's `wait_flag` for reverse deps would deadlock (no signal was ever sent).
- **After the loop (4 × `wait_flag`):** Drain the matching reverse-dep signals from the last iterations. Every `set_flag` must be paired with a `wait_flag` — the last loop iterations produce signals that no subsequent iteration consumes, so they must be drained explicitly.

###### 3b. `get_buf` / `rls_buf` version

Same ping/pong double-buffering, but **no pre-loop priming or post-loop draining needed.** Buffer acquire/release semantics handle everything.

```mlir
scf.for %i = %c0 to %N step %c1 {
  // %pp = i % 2  (ping/pong selector)

  // ── MTE2: load tile[i] into buf[i%2] ──
  // Acquires buf[i%2] — on first iteration, buffer is free so proceeds immediately.
  // On later iterations, blocks until Vector releases buf[i%2] (WAR: automatic).
  pto.get_buf "PIPE_MTE2", %bufid_buf[%pp], %c0 : i64, i64   // mode=0
  pto.mte_gm_ub %gm_ptr[%i], %ub_buf[%pp], ...
  pto.rls_buf "PIPE_MTE2", %bufid_buf[%pp], %c0 : i64, i64

  // ── Vector: compute on buf[i%2] ──
  // Acquires buf[i%2] — blocks until MTE2 releases it (RAW: automatic)
  pto.get_buf "PIPE_V", %bufid_buf[%pp], %c0 : i64, i64
  pto.get_buf "PIPE_V", %bufid_out[%pp], %c0 : i64, i64
  scf.for %dummy = %c0 to %c1 step %c1 {
    %v   = pto.vlds %ub_buf[%pp][%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %abs, %ub_out[%pp][%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
  } {llvm.loop.aivector_scope}
  // Release buf[i%2] — MTE2 can reuse in iteration i+2 (WAR resolved)
  pto.rls_buf "PIPE_V", %bufid_buf[%pp], %c0 : i64, i64
  pto.rls_buf "PIPE_V", %bufid_out[%pp], %c0 : i64, i64

  // ── MTE3: store result ──
  // Acquires out[i%2] — blocks until Vector releases it (RAW: automatic)
  pto.get_buf "PIPE_MTE3", %bufid_out[%pp], %c0 : i64, i64
  pto.mte_ub_gm %ub_out[%pp], %gm_out[%i], %len_burst ...
  pto.rls_buf "PIPE_MTE3", %bufid_out[%pp], %c0 : i64, i64
}
// No post-loop drain needed — last rls_buf completes the pipeline.
```

**No priming, no draining, no event IDs.** The acquire/release protocol on buffer IDs indexed by `i%2` implicitly resolves all cross-pipeline dependencies:
- **RAW** (MTE2→V): Vector's `get_buf` blocks until MTE2's `rls_buf` on `buf[i%2]`
- **WAR** (V→MTE2): MTE2's `get_buf` in iteration `i+2` blocks until Vector's `rls_buf` in iteration `i` (same buffer)
- **First iteration:** Buffer is initially free, so `get_buf` proceeds without blocking — no priming needed

---

#### Comparison Summary

| Aspect | `set_flag` / `wait_flag` | `get_buf` / `rls_buf` |
|--------|--------------------------|------------------------|
| Dependency model | Explicit event signals | Implicit via buffer acquire/release |
| IDs per pipe-pair | 2 IDs per buffer: 1 for forward (e.g., MTE2→V) + 1 for reverse (V→MTE2) | **1 ID per buffer** (handles both directions automatically) |
| Total HW IDs | **8 per pipe-pair** (hardware limit) | **32 global** across all pipes |
| Reverse (WAR) deps | Extra `set_flag`/`wait_flag` pair per buffer | Handled automatically |
| Pre-loop setup | `set_flag` to prime each reverse dep | **None** |
| Post-loop teardown | `wait_flag` to drain all primed signals | **None** |
| Loop peeling for complex deps | Required for non-1:1 or nested loops | **Not required** |
| Straight-line code | Simple, clear | Slightly more verbose (bracket each stage) |
| Ping/pong loops | 8 event IDs + 4 prime + 4 drain | Same pattern, **no overhead** |
| Best used for | Simple pipelines, fine-grained control | Double/multi-buffering, complex loops |

---

#### Inter-Core Sync

> **Note:** Inter-core sync is only needed for **mixed Cube+Vector tasks** where Cube produces data that Vector consumes (or vice versa). **Vec-only tasks can ignore this section entirely.**

These ops coordinate execution across the Cube block and Vector subblocks within a cluster. Each core cluster consists of **1 Cube block : 2 Vector subblocks**, each with its own **SU (Sequencer Unit)** running independent instruction streams.

```
Core Cluster (1:2 ratio)
┌─────────────────────────────────────────────┐
│  ┌──────────────┐    ┌──────────────┐       │
│  │  AIC (Cube)  │    │  AIV0 (Vec)  │       │
│  │  ┌────────┐  │    │  ┌────────┐  │       │
│  │  │   SU   │──┼────┼──│   SU   │  │       │
│  │  └────────┘  │    │  └────────┘  │       │
│  │  CUBE pipe   │    │  MTE2/V/MTE3 │       │
│  │  L0C buffer  │    │  UB (256KB)  │       │
│  └──────────────┘    └──────────────┘       │
│                      ┌──────────────┐       │
│                      │  AIV1 (Vec)  │       │
│                      │  ┌────────┐  │       │
│                      │  │   SU   │  │       │
│                      │  └────────┘  │       │
│                      │  MTE2/V/MTE3 │       │
│                      │  UB (256KB)  │       │
│                      └──────────────┘       │
└─────────────────────────────────────────────┘
```

##### Platform Comparison

| Aspect | A2A3 (Ascend 910) | A5 (Ascend 950) |
|--------|-------------------|-----------------|
| **Signal op** | `set_cross_core` (mode2) | `set_intra_block` |
| **Wait op** | `wait_flag_dev` | `wait_intra_core` |
| **Wait behavior** | SU-level blocking (entire core stalls) | Per-pipeline (only named pipe stalls) |
| **Semaphore pool** | 16 IDs per cluster, 4-bit counter | 16 IDs, but 32-ID address space (see below) |
| **C→V** | **Broadcast**: one `set` reaches both AIV0+AIV1 | **1:1**: separate `set` per subblock required |
| **V→C** | **Reduce**: Cube waits for both subblocks in one `wait` | **1:1**: Cube needs separate `wait` per subblock |

##### A2A3: `set_cross_core` / `wait_flag_dev`

```c
// mode2 broadcast/reduce semantics for 1:2 cluster
set_cross_core(pipe, semaphore_id);   // pipe: VEC/MTE2/CUBE/FIX
wait_flag_dev(semaphore_id);          // SU-level blocking
```

```
C→V Broadcast (one set reaches both):
    AIC ──set_cross_core──┬──> AIV0 sema++
                          └──> AIV1 sema++

V→C Reduce (one wait for both):
    AIV0 ──set_cross_core──┐
                           ├──> AIC wait_flag_dev (blocks until BOTH)
    AIV1 ──set_cross_core──┘
```

##### `pto.set_cross_core`

- **syntax:** `pto.set_cross_core %core_id, %event_id : i64, i64`
- **semantics:** Signal event to another core. Uses **mode2** for 1:2 cluster on A2A3.

##### `pto.wait_flag_dev`

- **syntax:** `pto.wait_flag_dev %core_id, %event_id : i64, i64`
- **semantics:** Wait for event from another core. **SU-level blocking** — entire core stalls.

##### A5: `set_intra_block` / `wait_intra_core`

```c
set_intra_block(trigger_pipe, semaphore_id);
wait_intra_core(wait_pipe, semaphore_id);   // only named pipe stalls
```

**A5 semaphore address space:** The hardware has **16 physical semaphore IDs** but exposes a **32-ID address space** to support 1:1 signaling to each subblock:

| ID Range | Target |
|----------|--------|
| 0–15 | AIV0 (subblock 0) |
| 16–31 (+15 offset) | AIV1 (subblock 1) |

This means C→V requires **separate `set_intra_block` calls** per subblock (no broadcast), and V→C requires **separate `wait_intra_core` calls** per subblock (no hardware reduce).

```
C→V on A5 (1:1, no broadcast — need two sets):
    AIC ──set_intra_block(pipe, sema_id)────> AIV0
    AIC ──set_intra_block(pipe, sema_id+15)──> AIV1

V→C on A5 (1:1, no reduce — need two waits):
    AIV0 ──set_intra_block──> AIC wait_intra_core(pipe, sema_id)
    AIV1 ──set_intra_block──> AIC wait_intra_core(pipe, sema_id+15)  // extra wait
```

##### `pto.set_intra_block`

- **syntax:** `pto.set_intra_block %block_id, %event_id : i64, i64`
- **semantics:** Signal event within a block (A5). Specifies **trigger pipe**. 1:1 per subblock.

##### `pto.wait_intra_core`

- **syntax:** `pto.wait_intra_core %block_id, %event_id : i64, i64`
- **semantics:** Wait for event within block (A5). Specifies **which pipeline should wait** — only that pipe stalls, SU and other pipes continue.

##### Wait Granularity Comparison

```
A2A3 wait_flag_dev (SU-level stall):
    SU ──┬── PIPE_MTE2 ───╳ ALL STALLED
         ├── PIPE_V    ───╳ ALL STALLED
         └── PIPE_MTE3 ───╳ ALL STALLED

A5 wait_intra_core "PIPE_MTE2" (per-pipe stall):
    SU ──┬── PIPE_MTE2 ───╳ STALLED (waiting for Cube)
         ├── PIPE_V    ─── ✓ RUNNING
         └── PIPE_MTE3 ─── ✓ RUNNING
```

<a id="micro-02-dma-copy"></a>

### 2. DMA Copy Programming

> **Category:** DMA transfer configuration and execution
> **Pipelines:** MTE2 (GM→UB), MTE3 (UB→GM)

DMA transfers move data between Global Memory (GM) and Unified Buffer (UB). The MTE engines operate asynchronously from the Vector core, requiring explicit sync (see [Pipeline Sync](#micro-01-pipeline-sync)).

This document describes the public grouped DMA interfaces:

- `pto.mte_gm_ub`
- `pto.mte_ub_gm`
- `pto.mte_ub_ub`
- `pto.mte_ub_l1`

---

#### DMA Transfer Execution

##### `pto.mte_gm_ub`

- **syntax:**
```mlir
pto.mte_gm_ub %gm_src, %ub_dst, %l2_cache_ctl, %len_burst
  nburst(%n_burst, %src_stride, %dst_stride)
  [loop(%loop_count, %loop_src_stride, %loop_dst_stride)]*
  [pad(%pad_value[, %left_padding_count, %right_padding_count])]
  : !pto.ptr<T, gm>, !pto.ptr<T, ub>, i64, i64, i64,
    [loop i64, i64, i64,]*
    [pad T[, i64, i64]]
```
- **semantics:** Grouped GM→UB DMA transfer. `nburst(...)` defines the innermost repeated burst transfer, optional `loop(...)` groups add outer repetition levels, and `pad(...)` controls UB row padding.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%gm_src` | ptr | GM source pointer (`!pto.ptr<T, gm>`) |
| `%ub_dst` | ptr | UB destination pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%l2_cache_ctl` | 2 bits | L2 cache allocate control |
| `%len_burst` | 16 bits | Contiguous bytes transferred per burst row |
| `nburst(%n_burst, %src_stride, %dst_stride)` | 16 bits / 40 bits / 21 bits | Required innermost burst group: count, GM source stride, UB destination stride |
| `loop(%loop_count, %loop_src_stride, %loop_dst_stride)` | 21 bits / 40 bits / 21 bits | Optional outer repetition group: count, GM source stride, UB destination stride |
| `pad(%pad_value[, %left_padding_count, %right_padding_count])` | scalar / 8 bits / 8 bits | Optional padding: fill value, optional left padding count, optional right padding count |

**Constraints:**

- `nburst(...)` is always required.
- Each `loop(...)` group must be provided as a complete triple when present.
- `nburst(...)` is the innermost group.
- `loop(...)` groups are ordered from inner to outer.
- The first `loop(...)` group wraps `nburst(...)`.
- Each additional `loop(...)` group wraps all earlier groups.
- `pad(...)` may contain only `%pad_value`; omitted left and right padding counts default to 0.
- If either left or right padding count is provided, both counts must be provided.
- `pad(...)` is independent of the optional `loop(...)` groups.
- A DMA load may use `nburst(...) pad(...)` without any `loop(...)` group.

**Example:**

```mlir
pto.mte_gm_ub %gm_in, %ub_out, %cache, %len_burst
  nburst(%rows, %gm_row_stride, %ub_row_stride)
  loop(%tiles, %gm_tile_stride, %ub_tile_stride)
  pad(%pad)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64,
    loop i64, i64, i64, pad f16
```

---

##### `pto.mte_ub_gm`

- **syntax:**
```mlir
pto.mte_ub_gm %ub_src, %gm_dst, %len_burst
  nburst(%n_burst, %src_stride, %dst_stride) l2_cache_ctl(%l2_cache_ctl)
  [loop(%loop_count, %loop_src_stride, %loop_dst_stride)]*
  : !pto.ptr<T, ub>, !pto.ptr<T, gm>, i64, i64, i64, i64, i64,
    [loop i64, i64, i64,]*
```
- **semantics:** Grouped UB→GM DMA transfer. `nburst(...)` defines the innermost repeated burst transfer, and optional `loop(...)` groups add outer repetition levels.
  The `l2_cache_ctl(...)` group is optional in textual VPTO IR; when omitted, lowering uses `0`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%ub_src` | ptr | UB source pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%gm_dst` | ptr | GM destination pointer (`!pto.ptr<T, gm>`) |
| `%len_burst` | 16 bits | Contiguous bytes transferred per burst row |
| `nburst(%n_burst, %src_stride, %dst_stride)` | 16 bits / 21 bits / 40 bits | Required innermost burst group: count, UB source stride, GM destination stride |
| `l2_cache_ctl(%l2_cache_ctl)` | 4 bits | Optional GM store-side L2 cache control; omitted means `0` |
| `loop(%loop_count, %loop_src_stride, %loop_dst_stride)` | 21 bits / 21 bits / 40 bits | Optional outer repetition group: count, UB source stride, GM destination stride |

**Constraints:**

- `nburst(...)` is always required.
- Each `loop(...)` group must be provided as a complete triple when present.
- `nburst(...)` is the innermost group.
- `loop(...)` groups are ordered from inner to outer.
- The first `loop(...)` group wraps `nburst(...)`.
- Each additional `loop(...)` group wraps all earlier groups.

**Example:**

```mlir
pto.mte_ub_gm %ub_in, %gm_out, %len_burst
  nburst(%rows, %ub_row_stride, %gm_row_stride) l2_cache_ctl(%l2_cache_ctl)
  loop(%tiles, %ub_tile_stride, %gm_tile_stride)
  loop(%batches, %ub_batch_stride, %gm_batch_stride)
  : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64,
    loop i64, i64, i64, loop i64, i64, i64
```

---

##### `pto.mte_ub_ub`

- **syntax:**
```mlir
pto.mte_ub_ub %ub_src, %ub_dst, %len_burst
  nburst(%n_burst, %src_gap, %dst_gap)
  : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64, i64, i64, i64
```
- **semantics:** Grouped UB→UB copy.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%ub_src` | ptr | UB source pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%ub_dst` | ptr | UB destination pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%len_burst` | 16 bits | Burst length in units of 32 bytes |
| `nburst(%n_burst, %src_gap, %dst_gap)` | 16 bits / 16 bits / 16 bits | Required copy burst group: count, source gap, destination gap |

**Constraints:**

- UB source and destination addresses must be 32B-aligned.
- `%len_burst`, `%src_gap`, and `%dst_gap` are encoded in units of 32 bytes.

**Example:**

```mlir
pto.mte_ub_ub %ub_src, %ub_dst, %len32b
  nburst(%rows, %src_gap, %dst_gap)
  : !pto.ptr<i16, ub>, !pto.ptr<i16, ub>, i64, i64, i64, i64
```

---

##### `pto.mte_ub_l1`

- **syntax:**
```mlir
pto.mte_ub_l1 %ub_src, %l1_dst, %len_burst
  nburst(%n_burst, %src_gap, %dst_gap)
  : !pto.ptr<T, ub>, !pto.ptr<T, l1>, i64, i64, i64, i64
```
- **semantics:** Grouped UB→L1/CBUF copy.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%ub_src` | ptr | UB source pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%l1_dst` | ptr | L1 destination pointer (`!pto.ptr<T, l1>`, 32B-aligned) |
| `%len_burst` | 16 bits | Burst length in units of 32 bytes |
| `nburst(%n_burst, %src_gap, %dst_gap)` | 16 bits / 16 bits / 16 bits | Required copy burst group: count, source gap, destination gap |

**Constraints:**

- UB source and L1 destination addresses must be 32B-aligned.
- `%len_burst`, `%src_gap`, and `%dst_gap` are encoded in units of 32 bytes.

**Example:**

```mlir
pto.mte_ub_l1 %ub_src, %l1_dst, %len32b
  nburst(%rows, %src_gap, %dst_gap)
  : !pto.ptr<i16, ub>, !pto.ptr<i16, l1>, i64, i64, i64, i64
```

---

#### Grouped DMA Burst / Stride / Pad Model

This section describes the grouped DMA interfaces in this document:
`pto.mte_gm_ub` and `pto.mte_ub_gm`.

For these grouped DMA ops, the innermost `nburst(...)` group is
**stride-based**: the source and destination stride operands are the
start-to-start byte distance from one burst row to the next row.

##### Key Terms

```
burst    = lenBurst contiguous bytes transferred per row
stride   = distance (bytes) from start of row[r] to start of row[r+1]
pad      = ub_stride - lenBurst, padded to the 32B alignment boundary
```

##### Alignment Constraints

- **UB addresses** (both source and destination) must be **32-byte aligned**.
- **GM→UB padding**: When `pad(...)` is present on `pto.mte_gm_ub`, each UB row is padded from `lenBurst` up to the **32B-aligned boundary** of `ub_stride` with `pad_val`. This ensures every UB row starts at a 32B-aligned offset.
- **UB→GM de-padding**: MTE3 reads `lenBurst` bytes from each 32B-aligned UB row (skipping any padding that was added during load), writing only valid data to GM. This effectively strips padding on store.

---

#### UB Copy Burst / Step Model

This section describes the grouped UB-copy interface in this document:
`pto.mte_ub_ub` and `pto.mte_ub_l1`.

For `pto.mte_ub_ub` and `pto.mte_ub_l1`, each burst copies `len_burst * 32` bytes.

The next burst starts at:

```text
src_next = src_curr + (len_burst + src_gap) * 32 bytes
dst_next = dst_curr + (len_burst + dst_gap) * 32 bytes
```

So `src_gap` and `dst_gap` are gap fields that advance to the next burst
after the copied 32B blocks.

##### 2D Diagram: GM→UB (`pto.mte_gm_ub`)

```
GM (source, `!pto.ptr<T, gm>`):

          |<--- src_stride (start-to-start) --->|
          |<- len_burst ->|                     |
Row 0:    [##DATA########]......................|
Row 1:    [##DATA########]......................|
Row 2:    [##DATA########]......................|
          ...
Row N-1:  [##DATA########]

UB (destination, `!pto.ptr<T, ub>`, 32B-aligned):

          |<---------- dst_stride (32B-aligned) ---------->|
          |<- len_burst ->|<- pad (to 32B boundary) ->|    |
Row 0:    [##DATA########][000000 PAD 000000000000000]
Row 1:    [##DATA########][000000 PAD 000000000000000]
Row 2:    [##DATA########][000000 PAD 000000000000000]
          ...
Row N-1:  [##DATA########][000000 PAD 000000000000000]

N = n_burst
stride = start of row[r] to start of row[r+1]
pad    = filled with pad_val to 32B boundary (`pad(...)` present)
[DATA] = valid data transferred by DMA
[PAD]  = pad_val fill (from `pad(...)`)
```

##### 2D Diagram: UB→GM (`pto.mte_ub_gm` with GM destination)

```
UB (source, `!pto.ptr<T, ub>`, 32B-aligned start addr):

          |<---------- src_stride (32B-aligned) --------->|
          |<- len_burst ->|<-- pad (ignored on read) -->| |
Row 0:    [##DATA########][000 pad 000000000000000000]
Row 1:    [##DATA########][000 pad 000000000000000000]
Row 2:    [##DATA########][000 pad 000000000000000000]
          ...
Row N-1:  [##DATA########][000 pad 000000000000000000]

GM (destination, `!pto.ptr<T, gm>`):

          |<--- dst_stride (start-to-start) --->|
          |<- len_burst ->|                     |
Row 0:    [##DATA########]......................|
Row 1:    [##DATA########]......................|
Row 2:    [##DATA########]......................|
          ...
Row N-1:  [##DATA########]

N = n_burst
MTE3 reads only len_burst bytes from each UB row (de-padding).
Only len_burst bytes are written to each GM row.
```

---

#### Multi-Level Loop Semantics

The full DMA transfer is a nested loop. `nburst(...)` is the innermost group.
If one or more `loop(...)` groups are present, they wrap `nburst(...)` in the
same order they appear in the op: the first `loop(...)` is the innermost outer
group, the second `loop(...)` wraps the first one, and so on.

##### GM→UB Full Loop

For a form

```mlir
pto.mte_gm_ub %gm_src, %ub_dst, %l2_cache_ctl, %len_burst
  nburst(%n_burst, %src_stride, %dst_stride)
  loop(%c0, %s0, %d0)
  loop(%c1, %s1, %d1)
  ...
  loop(%cN, %sN, %dN)
  [pad(%pad_value[, %left_padding_count, %right_padding_count])]
```

the transfer is equivalent to:

```c
for (int lN = 0; lN < cN; ++lN) {
  ...
  for (int l1 = 0; l1 < c1; ++l1) {
    for (int l0 = 0; l0 < c0; ++l0) {
      uint8_t *gm_base = gm_src + l0 * s0 + l1 * s1 + ... + lN * sN;
      uint8_t *ub_base = ub_dst + l0 * d0 + l1 * d1 + ... + lN * dN;
      for (int r = 0; r < n_burst; ++r) {
        memcpy(ub_base + r * dst_stride,
               gm_base + r * src_stride,
               len_burst);
        if (pad_enabled)
          memset(ub_base + r * dst_stride + len_burst,
                 pad_val,
                 dst_stride - len_burst);
      }
    }
  }
}
```

If no `loop(...)` group is present, only the innermost `nburst(...)` loop
remains.

##### UB→Destination Full Loop

For a form

```mlir
pto.mte_ub_gm %ub_src, %dst, %len_burst
  nburst(%n_burst, %src_stride, %dst_stride)
  l2_cache_ctl(%l2_cache_ctl)
  loop(%c0, %s0, %d0)
  loop(%c1, %s1, %d1)
  ...
  loop(%cN, %sN, %dN)
```

the transfer is equivalent to:

```c
for (int lN = 0; lN < cN; ++lN) {
  ...
  for (int l1 = 0; l1 < c1; ++l1) {
    for (int l0 = 0; l0 < c0; ++l0) {
      uint8_t *ub_base = ub_src + l0 * s0 + l1 * s1 + ... + lN * sN;
      uint8_t *dst_base = dst + l0 * d0 + l1 * d1 + ... + lN * dN;
      for (int r = 0; r < n_burst; ++r) {
        memcpy(dst_base + r * dst_stride,
               ub_base + r * src_stride,
               len_burst);
      }
    }
  }
}
```

If no `loop(...)` group is present, only the innermost `nburst(...)` loop
remains.

---

#### Example 1: GM→UB — Load a 32×32 f32 Tile (Simple Case)

Load a 32×32 f32 tile from GM into UB. This matches the `abs_kernel_2d` test case.

```
GM layout (32 × 32 f32, contiguous):

    |<- len_burst = 128B (32 × 4) ->|
    |<- src_stride = 128B --------->|
    +--[#######TILE#######]--+  row 0
    +--[#######TILE#######]--+  row 1
    ...
    +--[#######TILE#######]--+  row 31

UB layout (32 × 32 f32, 32B-aligned, contiguous):

    |<- dst_stride = 128B (32B-aligned) ->|
    +--[#######TILE#######]--+  row 0
    +--[#######TILE#######]--+  row 1
    ...
    +--[#######TILE#######]--+  row 31

    len_burst   = 32 × 4 = 128 bytes
    src_stride  = 128 bytes (contiguous rows)
    dst_stride  = 128 bytes (already 32B-aligned, no padding)
```

```mlir
// Simple 2D load — only nburst(...) is needed
pto.mte_gm_ub %arg0, %ub_in, %c0_i64, %c128_i64
  nburst(%c32_i64, %c128_i64, %c128_i64)
  : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64
```

---

#### Example 2: GM→UB — Load a 2D Tile from a Larger Matrix

Load a 64×128 tile (f16) from a 1024×512 matrix in GM into UB.

```
GM layout (1024 × 512 f16):

    col 0          col 128               col 512
    |              |                     |
    +--[###TILE###]+.....................+  row R
    +--[###TILE###]+.....................+  row R+1
    ...
    +--[###TILE###]+.....................+  row R+63

    |<--------- src_stride = 1024B ----------->|
    |<-len_burst=256B->|

    len_burst   = 128 × 2 = 256 bytes (128 f16 elements)
    src_stride  = 512 × 2 = 1024 bytes (start-to-start, full GM row)

UB layout (64 × 128 f16, 32B-aligned, contiguous):

    +--[###TILE###]--+  row 0  (256 bytes, 32B-aligned, no pad)
    +--[###TILE###]--+  row 1
    ...
    +--[###TILE###]--+  row 63

    dst_stride = 256 bytes (= len_burst, already 32B-aligned, no padding)
```

```mlir
pto.mte_gm_ub %gm_ptr, %ub_ptr, %c0_i64, %c256_i64
  nburst(%c64_i64, %c1024_i64, %c256_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64
```

---

#### Example 3: GM→UB — Load with Padding

Load 100 valid columns from GM into a 128-wide UB tile (f16). The remaining 28 columns are zero-padded.

```
GM (100 cols valid, contiguous):

    |<-len_burst=200B->|
    |<- src_stride=200B (start-to-start) ->|
    +--[####DATA####]-+  row 0
    +--[####DATA####]-+  row 1
    ...
    +--[####DATA####]-+  row 63

UB (128 cols wide, 32B-aligned, padded):

    |<--------- dst_stride = 256B (32B-aligned) --------->|
    |<-len_burst=200B->|<---- pad = 56B to 32B boundary ->|
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 0
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 1
    ...
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 63

    len_burst   = 100 × 2 = 200 bytes
    src_stride  = 200 bytes (start-to-start, contiguous in GM)
    dst_stride  = 128 × 2 = 256 bytes (32B-aligned tile width in UB)
    pad         = 256 - 200 = 56 bytes (padded to 32B boundary with pad_val)
```

```mlir
%pad = arith.constant 0 : i16
pto.mte_gm_ub %gm_ptr, %ub_ptr, %c0_i64, %c200_i64
  nburst(%c64_i64, %c200_i64, %c256_i64)
  pad(%pad, %c0_i64, %c0_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, pad i16, i64, i64
```

---

#### Example 4: UB→GM — Store a 32×32 f32 Tile (Simple Case)

Store a 32×32 f32 tile from UB back to GM. This matches the `abs_kernel_2d` test case.

```
UB (source, 32B-aligned, 32 × 32 f32):

    |<- src_stride = 128B (32B-aligned) ->|
    |<- len_burst = 128B ->|
    +--[#######TILE#######]---+  row 0
    +--[#######TILE#######]---+  row 1
    ...
    +--[#######TILE#######]---+  row 31

    (no padding here — len_burst == src_stride)

GM (dest, 32 × 32 f32):

    |<- dst_stride = 128B ->|
    |<- len_burst = 128B -->|
    +--[#######TILE#######]---+  row 0
    +--[#######TILE#######]---+  row 1
    ...
    +--[#######TILE#######]---+  row 31
```

```mlir
pto.mte_ub_gm %ub_out, %arg1, %c128_i64
  nburst(%c32_i64, %c128_i64, %c128_i64) l2_cache_ctl(%c0_i64)
  : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64
```

---

#### Example 5: UB→GM — Store a 2D Tile Back to a Larger Matrix

Store a 64×128 tile (f16) from UB back to a 1024×512 GM matrix at an offset.

```
UB (source, 32B-aligned, 64 × 128 f16):

    |<- src_stride = 256B (32B-aligned) ->|
    |<- len_burst = 256B ->|
    +--[#####TILE#####]---+  row 0
    +--[#####TILE#####]---+  row 1
    ...
    +--[#####TILE#####]---+  row 63

    (no padding here — len_burst == src_stride)

GM (dest, into 1024 × 512 matrix):

    |<----------- dst_stride = 1024B (start-to-start) --------->|
    |<- len_burst = 256B ->|                                    |
    col 0          col 128                              col 512
    +--[#####TILE#####]---+.............................+  row R
    +--[#####TILE#####]---+.............................+  row R+1
    ...
    +--[#####TILE#####]---+.............................+  row R+63

    MTE3 reads len_burst bytes from each 32B-aligned UB row,
    writes only len_burst bytes per GM row (stride controls row spacing).
```

```mlir
pto.mte_ub_gm %ub_ptr, %gm_ptr, %c256_i64
  nburst(%c64_i64, %c256_i64, %c1024_i64) l2_cache_ctl(%c0_i64)
  : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64
```

---

#### Example 6: GM→UB with Multi-Level Loop (Batch of Tiles)

Load 4 batches of 8×128 tiles from a [4, 8, 128] f16 tensor using one outer
`loop(...)` group.

```
GM [4, 8, 128] f16 (contiguous):        UB (4 tiles laid out sequentially):

    batch 0: 8 rows × 256 bytes          [batch 0: 8×128][batch 1: 8×128]
    batch 1: 8 rows × 256 bytes          [batch 2: 8×128][batch 3: 8×128]
    batch 2: 8 rows × 256 bytes
    batch 3: 8 rows × 256 bytes          outer loop src_stride = 2048 bytes (8 × 256)
                                          outer loop dst_stride = 2048 bytes (8 × 256)
    Each batch = 8 × 256 = 2048 bytes     outer loop count = 4 (iterate over batches)
```

```mlir
// One outer loop group over 4 batches
pto.mte_gm_ub %gm_ptr, %ub_ptr, %c0_i64, %c256_i64
  nburst(%c8_i64, %c256_i64, %c256_i64)
  loop(%c4_i64, %c2048_i64, %c2048_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, loop i64, i64, i64
```

Execution trace:

```
loop iter 0: gm_ptr + 0×2048 → ub_ptr + 0×2048, DMA 8 rows × 256B
loop iter 1: gm_ptr + 1×2048 → ub_ptr + 1×2048, DMA 8 rows × 256B
loop iter 2: gm_ptr + 2×2048 → ub_ptr + 2×2048, DMA 8 rows × 256B
loop iter 3: gm_ptr + 3×2048 → ub_ptr + 3×2048, DMA 8 rows × 256B
```

<a id="micro-03-vector-load-store"></a>

### 3. Vector Load/Store

> **Category:** UB ↔ Vector Register data movement
> **Pipeline:** PIPE_V (Vector Core)

Vector loads move data from Unified Buffer (UB) to vector registers (`vreg`). Vector stores move data from `vreg` back to UB. All vector compute operates only on `vreg` — UB is the staging area between DMA and compute.

#### Common Operand Model

- `%source` / `%dest` is the base address operand in SSA form. The base pointer
  MUST address the Vector tile buffer / UB space.
- `%offset` is the displacement operand in SSA form. The exact encoding is
  instruction-specific, but the effective address and any post-update behavior
  MUST match the selected instruction form.
- `%mask` is the predicate operand for predicated memory families. For memory
  families,
  inactive lanes or inactive blocks MUST NOT issue memory requests unless the
  instruction explicitly documents a different behavior.
- `%result` is the destination vector register value in SSA form.
- `!pto.align` is the SSA carrier for alignment-buffer state used by unaligned
  load/store families. The PTO micro Instruction representation makes that state explicit rather than implicit.

---

#### Latency and throughput (A5)

**Cycle-accurate simulator (CA model)** issue→retire timings for vector-side instructions behind this chapter. Values are **simulator** results, **not** guaranteed for silicon.

**SOC:** Tables below are from **Ascend910_9599** CA sim (the pto-isa ST default when **Ascend950PR_9599** is not selected).

**Log `dist:` tokens:** PTO load/store modes lower to **`RV_VLD` / `RV_VLDI` / `RV_VST` / `RV_VSTI`** with a **`dist:`** field on the vector pipes (`RVECLD` / `RVECST`). Some simulator logs typo contiguous load as `dist:NORAML`; treat as **`NORMAL`**.

##### Reference op latencies (A5 mnemonics)

| A5 mnemonic | Mode / note | Typical issue→retire (cycles) |
|-------------|-------------|------------------------------|
| `RV_VLD` | `dist:NORMAL` / `NORAML` | **9** |
| `RV_VLDI` | `dist:DINTLV` (dual vreg) | **9** |
| `RV_VST` / `RV_VSTI` | `dist:NORM_B8` / `NORM_B16` / `NORM_B32` | **9** |
| `RV_VGATHER2` | `Dtype: B32` | **27–28** |
| `RV_VGATHERB` | indexed byte gather | **~21** |
| `RV_VSCATTER` | `Dtype: B16` | **~17** |
| `RV_VADD` | F32 between UB-backed ops | **7** |

##### `dist:` tokens (issue→retire)

Most **`dist:`** tokens are **9** issue→retire cycles. **`INTLV_B8` / `INTLV_B16` / `INTLV_B32`** on **`RV_VSTI`** are **12** cycles.

| `dist:` (as in log) | RV op | issue→retire (cycles) |
|---------------------|-------|----------------------|
| `DINTLV_B8` / `DINTLV_B16` / `DINTLV_B32` | `RV_VLDI` | **9** |
| `BRC_B8` / `BRC_B16` / `BRC_B32` | `RV_VLD` | **9** |
| `BRC_BLK` | `RV_VLD` | **9** |
| `INTLV_B8` / `INTLV_B16` / `INTLV_B32` | `RV_VSTI` | **12** |
| `UNPK_B8` / `UNPK_B16` / `UNPK_B32` | `RV_VLD` | **9** |
| `NORM_B8` / `NORM_B16` / `NORM_B32` | `RV_VSTI` | **9** |
| `PK_B16` / `PK_B32` / `PK_B64` / `PK4_B32` | `RV_VSTI` | **9** |
| `NORMAL` / `NORAML` | `RV_VLD` | **9** |

**Note:** PTO intrinsic **`BRC_BLK`** matches the **`BRC_BLK`** `dist:` string on **`RV_VLD`** in simulator logs (block-replicate path; not a plain contiguous copy in the usual tiling use).

**Issue (vector load/store):** `pto.vlds` (**`RV_VLD`**) is **dual-issue capable**: two independent `pto.vlds` can issue **in the same cycle**. **Alternatively**, the hardware can issue **one** `pto.vlds` **and** **one** `pto.vsts` together (**1+1**) in the same cycle. Each cycle is **either** dual **`vlds`** **or** **`vlds` + `vsts` (1+1)**—those two issue modes are mutually exclusive. Sustained throughput still depends on RAW hazards and loop structure.

**Throughput (simulator, pattern-dependent):**

- **`RV_VLD` / `pto.vlds`:** Dual-issue **or** half of a **1+1** with `vsts`, per the rule above.
- **`RV_VST` / `pto.vsts`:** In a **1+1** cycle, pairs with one `vlds`; otherwise typically **one** store per cycle in tight loops.
- **`RV_VGATHER2`:** Much lower than contiguous `RV_VLD` (on the order of **~0.1** ops/cycle in steady-state alongside 27–28-cycle latency).

##### PTO `dist` summary (loads)

| PTO `dist` (load) | Latency |
|-------------------|-------------------|
| `NORM` | **9** cycles |
| `UNPK_B8` / `UNPK_B16` / `UNPK_B32` | **9** cycles |
| `DINTLV_B8` / `DINTLV_B16` / `DINTLV_B32` | **9** cycles (`RV_VLDI`) |
| `BRC_B8` / `BRC_B16` / `BRC_B32` | **9** cycles (`RV_VLD`) |
| `BRC_BLK` | **9** cycles as **`dist:BRC_BLK`** on `RV_VLD` |
| `BDINTLV` | **9** cycles |
| `US_B8` / `US_B16`, `DS_B8` / `DS_B16`, `SPLT4CHN`, `SPLT2CHN_B8` / `SPLT2CHN_B16` | **9** cycles |

##### PTO `dist` summary (stores)

| PTO `dist` (store) | Latency |
|--------------------|-------------------|
| `NORM_B8` / `NORM_B16` / `NORM_B32` | **9** cycles (`RV_VSTI`) |
| `PK_B16` / `PK_B32` / `PK_B64` / `PK4_B32` | **9** cycles |
| `INTLV_B8` / `INTLV_B16` / `INTLV_B32` (`pto.vstsx2`) | **12** cycles |
| `MRG4CHN_B8`, `MRG2CHN_B8`, `MRG2CHN_B16` | **9** cycles (surface retained; current A5 hardware still reports them unsupported at validation time) |

##### Gather, scatter, and special addressing

| PTO op | A5-level | Latency |
|--------|----------|-------------------|
| `pto.vgather2` | `RV_VGATHER2` | **27–28** cycles (pattern-dependent) |
| `pto.vgatherb` | `RV_VGATHERB` | **~21** cycles issue→retire |
| `pto.vgather2_bc` | (broadcast gather) | **27–28** cycles (same as **`pto.vgather2`**) |
| `pto.vscatter` | `RV_VSCATTER` | **~17** cycles for **`Dtype: B16`** |

##### Strided loads/stores, unaligned ops, alignment state

Ops such as **`pto.vldas`**, **`pto.vldus`**, **`pto.vsld`**, **`pto.vsldb`**, **`pto.vsst`**, **`pto.vsstb`**, **`pto.vsta`**, **`pto.vstas`**, **`pto.vstar`**, **`pto.vstu`**, **`pto.vstus`**, **`pto.vstur`**: **9** cycles (same vector load/store pipe family as contiguous `RV_VLD` / `RV_VST` unless listed otherwise above).

##### Dual-issue vs DMA

DMA **`TLOAD` / `TSTORE`** (global memory ↔ UB) use **MTE** pipes, not `RV_VLD`/`RV_VST`. **MTE2** `MOV_*` latency is not the same as vector `RV_VLD` latency (see `02-dma-copy.md` for GM↔UB movement).

---

#### Contiguous Loads

##### `pto.vlds`

- **syntax:** `%result = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>`

  Post-update form:

  `%result, %updated_base = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>, !pto.ptr<T, ub>`
- **semantics:** Vector load with distribution mode.
- **inputs:**
  `%source` is the UB base address, `%offset` is the load displacement, and
  `DIST` selects the distribution mode.
- **outputs:**
  `%result` is the loaded vector register value. In the post-update form,
  `%updated_base` is the base pointer advanced according to `%offset`.
- **constraints and limitations:**
  The effective address MUST satisfy the alignment rule of the selected
  distribution mode. `NORM` reads one full vector footprint. Broadcast,
  upsample, downsample, unpack, split-channel, and deinterleave modes change
  how memory bytes are mapped into destination lanes, but they do not change the
  fact that the source is UB memory. PTO surface exposes load `dist` as family
  tokens, and each family only supports the element widths listed below.
  The optional `%updated_base` result must have the same pointer type as the
  base address operand.

**Distribution families:**

| Family | Allowed element widths | C semantics | Latency |
|------|-------------|-------------|-------------|
| `NORM` | width-agnostic | `dst[i] = UB[base + i * sizeof(T)]` | **9** cycles |
| `BRC_B8` / `BRC_B16` / `BRC_B32` | `b8`, `b16`, `b32` | `dst[i] = UB[base]` for all `i` | **9** cycles |
| `US_B8` / `US_B16` | `b8`, `b16` | `dst[2*i] = dst[2*i+1] = UB[base + i]` | **9** cycles |
| `DS_B8` / `DS_B16` | `b8`, `b16` | `dst[i] = UB[base + 2*i]` | **9** cycles |
| `UNPK_B8` / `UNPK_B16` / `UNPK_B32` | `b8`, `b16`, `b32` | Expand packed source data into wider lanes | **9** cycles |
| `BRC_BLK` | width-agnostic | Block-replicate load path; simulator logs may print `dist:BRC_BLK` | **9** cycles |
| `E2B_B16` / `E2B_B32` | `b16`, `b32` | Load element groups and expand them into byte-oriented lane layout | **9** cycles |
| `UNPK4` | `b8` | Unpack 4-way packed `b8` source groups into destination lanes | **9** cycles |
| `SPLT4CHN` | `b8` | Split 4-channel interleaved source into one channel plane | **9** cycles |
| `SPLT2CHN_B8` / `SPLT2CHN_B16` | `b8`, `b16` | Split 2-channel interleaved source into one channel plane | **9** cycles |

`pto.vlds` currently covers only single-result load families. Dual-result
deinterleave forms are modeled separately in PTO surface as
[`pto.vldsx2`](#ptovldsx2): `BDINTLV` is the block-deinterleave family, while
`DINTLV_B8` / `DINTLV_B16` / `DINTLV_B32` are the element-width-sensitive
deinterleave forms.

**Example — Contiguous load:**
```mlir
%v = pto.vlds %ub[%offset] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

**Example — Broadcast scalar to all lanes:**
```mlir
%v = pto.vlds %ub[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

---

##### `pto.vldas`

- **syntax:** `%result = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align`
- **semantics:** Prime alignment buffer for subsequent unaligned load.
- **inputs:**
  `%source` is the UB address whose surrounding aligned block seeds the load
  alignment state.
- **outputs:**
  `%result` is the initialized load-alignment state.
- **constraints and limitations:**
  This op is the required leading operation for a `pto.vldus` stream using the
  same alignment state. The source address itself need not be 32-byte aligned;
  hardware truncates it to the aligned block boundary for the priming load.
- **Latency:** **9** cycles.

---

##### `pto.vldus`

- **syntax:** `%result, %align_out = pto.vldus %source, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align`
- **semantics:** Unaligned load using primed align state.
- **inputs:**
  `%source` is the current UB address and `%align` is the incoming load
  alignment state primed by `pto.vldas` or a prior `pto.vldus`.
- **outputs:**
  `%result` is the assembled vector value and `%align_out` is the updated
  alignment state.
- **constraints and limitations:**
  A matching `pto.vldas` MUST appear before the first dependent `pto.vldus`
  stream in the same vector loop. The installed no-post A5 interface keeps a
  struct-shaped internal return for lowering convenience, but its no-post
  `base` field is not meaningful user-visible state. VPTO therefore hides that
  value and only exposes the updated align carrier. Reusing the original
  `%source` starts a new explicit access point; if the caller wants another
  no-post access, it should compute the next source pointer explicitly and pair
  it with the required align setup.
- **Latency:** **9** cycles.

**Unaligned load pattern:**
```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align2 = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align
```

---

##### `pto.init_align`

- **syntax:** `%result = pto.init_align : !pto.align`
- **semantics:** Initialize store-side align carrier state.
- **outputs:**
  `%result` is a fresh zero-initialized align carrier for store-side unaligned
  streams such as `pto.vstus`, `pto.vstur`, `pto.vstar`, `pto.vstas`, and
  `pto.pstu`.
- **constraints and limitations:**
  This op is for store-family initialization only. Unaligned load streams still
  start from `pto.vldas`.

---

#### Dual Loads (Deinterleave)

##### `pto.vldsx2`

- **syntax:** `%low, %high = pto.vldsx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Dual load with deinterleave (AoS → SoA conversion).
- **inputs:**
  `%source` is the UB base pointer, `%offset` is the displacement, and `DIST`
  selects a dual-load/deinterleave layout.
- **outputs:**
  `%low` and `%high` are the two destination vectors.
- **constraints and limitations:**
  This family is only legal for interleave/deinterleave style distributions.
  The two outputs form an ordered pair, and that pairing MUST be preserved.
  PTO surface accepts deinterleave families. `BDINTLV` is element-width
  agnostic, while `DINTLV_B8` / `DINTLV_B16` / `DINTLV_B32` support only the
  element widths listed in the
  table.
- **latency:** `BDINTLV` / `DINTLV_B8` / `DINTLV_B16` / `DINTLV_B32` are all
  **9** cycles.

**Distribution families:**

| Family | Allowed element widths | C semantics | Latency |
|------|-------------|-------------|-------------|
| `BDINTLV` | width-agnostic | Block deinterleave into two destination vectors | **9** cycles |
| `DINTLV_B8` / `DINTLV_B16` / `DINTLV_B32` | `b8`, `b16`, `b32` | Deinterleave alternating elements into `%low` / `%high` | **9** cycles |

```c
// DINTLV_B32 family on 32-bit elements: deinterleave 32-bit elements
for (int i = 0; i < 64; i++) {
    low[i]  = UB[base + 8*i];       // even elements
    high[i] = UB[base + 8*i + 4];   // odd elements
}
```

**Example — Load interleaved XY pairs into separate X/Y vectors:**
```mlir
%x, %y = pto.vldsx2 %ub[%offset], "DINTLV_B32" : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
```

##### `pto.vsldb`

- **syntax:** `%result = pto.vsldb %source, %block_stride, %repeat_stride, %mask : !pto.ptr<T, ub>, i16, i16, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Block-strided load for 2D tile access.
- **inputs:**
  `%source` is the UB base pointer. `%block_stride` and `%repeat_stride` are
  the two 16-bit fields of the hardware control word, and `%mask` controls
  which blocks participate.
- **outputs:**
  `%result` is the loaded vector.
- **constraints and limitations:**
  PTO surface does not expose the packed control word directly. If a block is
  masked off, the corresponding destination block is zeroed and MUST NOT raise
  an address overflow exception for that block.
- **Latency:** **9** cycles.

```c
// Block-strided load on 32-bit elements: one 32B block = 8 lanes.
for (int blk = 0; blk < 8; ++blk) {
    if (pg_b32[blk])
        dst_block[blk] = UB_block[base + repeat_stride + blk * block_stride];
    else
        dst_block[blk] = 0;
}
```

---

#### Gather (Indexed) Loads

##### `pto.vgather2`

- **syntax:** `%result = pto.vgather2 %source, %offsets, %mask : !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Indexed gather from UB.
- **inputs:**
  `%source` is the UB base pointer, `%offsets` provides one unsigned element
  index for each logical gather lane, and `%mask` selects those logical
  index/result lanes.
- **outputs:**
  `%result` is the gathered vector.
- **addressing:**
  Each active lane computes its UB byte address from the source element width:

  ```text
  addr[i] = byte_address(%source) + unsigned(%offsets[i]) * sizeof(source element)
  ```

  For `i8/ui8` sources, `sizeof(source element) == 1`; for 16-bit sources it is
  `2`; for 32-bit sources it is `4`. The computed address must be aligned for
  the source element type. For `i8/ui8` sources, each loaded 8-bit payload is
  zero-extended into a 16-bit result lane.
- **semantic pseudocode:**

  ```text
  for i in lanes:
    if mask[i]:
      value = UB_load(source_element_type, addr[i])
      if source_element_type is i8 or ui8:
        result[i] = zero_extend_to_16_bits(value)
      else:
        result[i] = value
    else:
      result[i] = 0
  ```
- **constraints and limitations:**
  Only masked-on indices participate. The index element width
  and interpretation MUST match the selected gather form, and each effective
  address must satisfy that form's alignment rules.
  Supported forms are:
  `i8/ui8 -> i16/ui16` with `!pto.vreg<128xui16>` offsets and
  `!pto.mask<b16>`; `i16/ui16/f16/bf16 -> same type` with
  `!pto.vreg<128xui16>` offsets and `!pto.mask<b16>`; and
  `i32/ui32/f32 -> same type` with `!pto.vreg<64xui32>` offsets and
  `!pto.mask<b32>`. Signless integer offsets are accepted as storage-compatible
  aliases for the unsigned offset register payload.
- **Latency:** **27–28** cycles per `RV_VGATHER2`; throughput much lower than contiguous `RV_VLD` (see **Latency and throughput (A5)** at the start of this chapter).

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = UB[base + offsets[i] * sizeof(T)];
```

---

##### `pto.vgatherb`

- **syntax:** `%result = pto.vgatherb %source, %offsets, %mask : !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask<b32> -> !pto.vreg<NxT>`
- **semantics:** Block gather load from UB.
- **inputs:**
  `%source` is the UB base pointer, `%offsets` is a `ui32` offset vector, and
  `%mask` is a `b32` predicate over the block-index lanes.
- **outputs:**
  `%result` is the gathered vector.
- **constraints and limitations:**
  This is a 32-byte block gather, not an element gather. `%source` MUST be
  32-byte aligned. Each participating `offsets[i]` is interpreted as a byte
  offset and MUST itself be 32-byte aligned. Only the low `VL/8` bytes of the
  offset vector are semantically valid; the effective block address is
  `block_addr[i] = offsets_u32[i] + base`. If a `b32` predicate position is
  false, the corresponding block does not participate in address coalescing,
  does not raise overflow on that block address, and the destination block is
  zero-filled.
- **Latency:** **~21** cycles issue→retire.

```c
for (int blk = 0; blk < VL / 32; ++blk) {
    if (pg_b32[blk])
        dst_block[blk] = UB_block[base + offsets_u32[blk]];
    else
        dst_block[blk] = 0;
}
```

---

##### `pto.vgather2_bc`

- **syntax:** `%result = pto.vgather2_bc %source, %offsets, %mask : !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Gather with broadcast, conditioned by mask.
- **inputs:**
  `%source` is the UB base pointer, `%offsets` contains gather indices, and
  `%mask` gates which lanes participate.
- **outputs:**
  `%result` is the gathered vector.
- **constraints and limitations:**
  This is a backward-compatible family. Masked-off lanes do not participate in
  address coalescing and do not trigger address overflow exceptions; their
  destination lanes are zero-filled. On the current PTO surface, `%offsets`
  uses 32-bit integer elements.
- **Latency:** **27–28** cycles (same as **`pto.vgather2`**).

---

#### Contiguous Stores

##### `pto.vsts`

- **syntax:** `pto.vsts %value, %dest[%offset], %mask {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.mask<G>`
- **post-update syntax:** `%updated_dest = pto.vsts %value, %dest[%offset], %mask {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.mask<G> -> !pto.ptr<T, ub>`
- **semantics:** Vector store with distribution mode.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer, `%offset` is
  the displacement, `%mask` is the predicate operand, and
  `DIST` selects the store distribution.
- **outputs:**
  The normal form has no SSA result and writes to UB memory. In the post-update
  form, `%updated_dest` is the destination pointer advanced according to
  `%offset`.
- **constraints and limitations:**
  The effective destination address MUST satisfy the alignment rule of the
  selected store mode. The single-input `pto.vsts` family covers contiguous
  store, first-element-only store, packed store, and channel-merge store.
  Dual-input interleave store remains in `pto.vstsx2`. PTO surface exposes
  store `dist` as family tokens, and each family only supports the element
  widths listed below.

**Distribution families:**

| Family | Allowed element widths | C semantics | Latency |
|------|-------------|-------------|-------------|
| `NORM_B8` / `NORM_B16` / `NORM_B32` | `b8`, `b16`, `b32` | `UB[base + i] = src[i]` | **9** cycles |
| `1PT_B8` / `1PT_B16` / `1PT_B32` | `b8`, `b16`, `b32` | Only element 0 is written to the destination footprint; the predicate register is ignored. | **9** cycles |
| `PK_B16` | `b16` | Pack the source vector, extract the lower half bits of all elements, and only store the active elements. The predicate is interpreted for 16-bit data. | **9** cycles |
| `PK_B32` | `b32` | Pack the source vector, extract the lower half bits of all elements, and only store the active elements. The predicate is interpreted for 32-bit data. | **9** cycles |
| `PK_B64` | `b64` | Pack the source vector, extract the lower half bits of all elements, and only store the active elements. The predicate is interpreted for 64-bit data. | **9** cycles |
| `PK4_B32` | `b32` | Pack the source vector, extract the lower 8 bits of all elements, and only store the active elements. The predicate is interpreted for 32-bit data. | **9** cycles |
| `MRG4CHN_B8` | `b8` | Merge 4 interleaved 8-bit channels within each 32B block; the predicate is interpreted for 32-bit data and applies after channel merge. | **9** cycles |
| `MRG2CHN_B8` / `MRG2CHN_B16` | `b8`, `b16` | Merge 2 interleaved channels within each 32B block; for `MRG2CHN_B8` the predicate is interpreted for 16-bit data, and for `MRG2CHN_B16` it is interpreted for 32-bit data; in both cases it applies after channel merge. | **9** cycles |

**Example — Contiguous store:**
```mlir
pto.vsts %v, %ub[%offset], %mask {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<G>
```

---

#### Dual Stores (Interleave)

##### `pto.vstsx2`

- **syntax:** `pto.vstsx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask<G>`
- **semantics:** Dual interleaved store (SoA → AoS conversion).
- **inputs:**
  `%low` and `%high` are the two source vectors, `%dest` is the UB base pointer,
  `%offset` is the displacement, `DIST` selects the interleave layout, and
  `%mask` is the predicate operand.
- **outputs:**
  This op has no SSA result; it writes an interleaved stream to UB.
- **constraints and limitations:**
  This family is only legal for interleave distributions. The two source
  vectors form an ordered pair, and the interleave semantics of that pair MUST
  be preserved. PTO surface accepts the `INTLV` family, which only supports the
  element widths listed below. For all `INTLV_*` distributions, the predicate
  register is ignored.
- **latency:** `INTLV` is **12** cycles。

**Distribution families:**

| Family | Allowed element widths | C semantics | Latency |
|------|-------------|-------------|-------------|
| `INTLV` | `b8`, `b16`, `b32` | Interleave `%low` / `%high` into one destination stream | **12** cycles |

```c
// INTLV family on 32-bit elements:
for (int i = 0; i < 64; i++) {
    UB[base + 8*i]     = low[i];
    UB[base + 8*i + 4] = high[i];
}
```

##### `pto.vsstb`

- **syntax:** `pto.vsstb %value, %dest, %block_stride, %repeat_stride, %mask : !pto.vreg<NxT>, !pto.ptr<T, ub>, i16, i16, !pto.mask<G>`
- **post-update syntax:** `%updated_dest = pto.vsstb %value, %dest, %block_stride, %repeat_stride, %mask : !pto.vreg<NxT>, !pto.ptr<T, ub>, i16, i16, !pto.mask<G> -> !pto.ptr<T, ub>`
- **semantics:** Block-strided store for 2D tile access.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer,
  `%block_stride` and `%repeat_stride` are the two 16-bit fields of the
  hardware control word, and `%mask` controls block participation.
- **outputs:**
  The normal form writes UB memory and returns no SSA value. In the post-update
  form, `%updated_dest` is the destination pointer advanced according to the
  packed stride control word.
- **constraints and limitations:**
  PTO surface does not expose the packed control word directly. Masked-off
  blocks MUST NOT issue memory writes.
- **Latency:** **9** cycles.

```c
// Block-strided store on 32-bit elements: one 32B block = 8 lanes.
for (int blk = 0; blk < 8; ++blk) {
    if (pg_b32[blk])
        UB_block[base + repeat_stride + blk * block_stride] = src_block[blk];
}
```

---

#### Scatter (Indexed) Stores

##### `pto.vscatter`

- **syntax:** `pto.vscatter %value, %dest, %offsets, %mask : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask<G>`
- **semantics:** Indexed scatter to UB.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer, `%offsets`
  provides unsigned element indices relative to `%dest`, and `%mask` selects
  the active requests. The legal type combinations are:

  | Value and destination type | Offset type | Mask type | Requests |
  | --- | --- | --- | --- |
  | `b8` (`!pto.vreg<256xT>`) | `!pto.vreg<128xui16>` or `!pto.vreg<128xi16>` | `!pto.mask<b16>` | 128 |
  | `b16` (`!pto.vreg<128xT>`) | `!pto.vreg<128xui16>` or `!pto.vreg<128xi16>` | `!pto.mask<b16>` | 128 |
  | `b32` (`!pto.vreg<64xT>`) | `!pto.vreg<64xui32>` or `!pto.vreg<64xi32>` | `!pto.mask<b32>` | 64 |

  Signless offset element types have the same unsigned interpretation as the
  corresponding `ui16` or `ui32` type.
- **outputs:**
  This op writes UB memory and returns no SSA value.
- **constraints and limitations:**
  Only `b8`, `b16`, and `b32` element sizes are supported, and `%dest` must have
  the same element type as `%value`. Each destination address is
  `%dest + %offsets[i]` in elements, equivalently
  `byte_address(%dest) + unsigned(%offsets[i]) * sizeof(T)`, and MUST be
  element-aligned. For `b8`, request `i` stores `%value[2*i]`; odd-numbered
  source bytes are ignored. If two or more active indices are equal, only one
  write is guaranteed and the winning request is implementation-defined.
- **Latency:** **~17** cycles for **`Dtype: B16`**.

```c
for (int i = 0; i < num_requests; i++)
    if (mask[i])
        dest[unsigned(offsets[i])] =
            sizeof(T) == 1 ? value[2 * i] : value[i];
```

---

#### SPR State Ops

##### `pto.sprclr`

- **syntax:** `pto.sprclr "AR"`
- **semantics:** Clear the hardware SPR AR state used by AR-driven unaligned
  store forms.
- **constraints and limitations:**
  Only `"AR"` is supported.

---

##### `pto.sprsti`

- **syntax:** `pto.sprsti "AR", %dest[%offset] : !pto.ptr<ui32, ub>, i32`
- **semantics:** Store SPR AR to UB using a signed 8-bit immediate offset.
- **inputs:**
  `%dest` is the UB base pointer and `%offset` is the immediate offset in units
  of the SPR data width.
- **constraints and limitations:**
  Only `"AR"` is supported. `%dest` must be a `ui32` or signless `i32` UB
  pointer. `%offset` must be a constant signed 8-bit `i32`. The current VPTO
  surface models only the no-post-update form, so no updated base pointer is
  returned.

---

##### `pto.sprsts`

- **syntax:** `pto.sprsts "AR", %dest[%offset] : !pto.ptr<ui32, ub>, i32`
- **semantics:** Store SPR AR to UB using a scalar-register offset.
- **inputs:**
  `%dest` is the UB base pointer and `%offset` is the scalar offset in bytes.
- **constraints and limitations:**
  Only `"AR"` is supported. `%dest` must be a `ui32` or signless `i32` UB
  pointer. The current VPTO surface models only the no-post-update form, so no
  updated base pointer is returned.

---

#### Alignment State Stores

##### `pto.vstas`
- **syntax:** `pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32`
- **semantics:** Scalar-register-offset form of alignment-state flush.
- **inputs:**
  `%value` is the pending store-alignment state, `%dest` is the UB base
  pointer, and `%offset` is the scalar-register style displacement.
- **outputs:**
  This op writes buffered tail bytes to UB and returns no SSA value.
- **constraints and limitations:**
  This family flushes pending store-alignment state using an explicit scalar
  offset and keeps the scalar-offset form explicit. The incoming `%value`
  should come from `pto.init_align` or from a prior state-producing unaligned
  store op in the same stream. `%dest` and `%offset` together must identify the
  same logical flush point produced by the immediately preceding stateful
  unaligned-store step on that stream; using an unrelated base/offset pair is
  invalid even if `%value` itself came from the same stream.
- **Latency:** **9** cycles.

---

##### `pto.vstar`
- **syntax:** `pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>`
- **semantics:** Flush alignment state using the register-update form.
- **inputs:**
  `%value` is the pending store-alignment state and `%dest` is the UB base
  pointer.
- **outputs:**
  This op writes buffered tail bytes to UB and returns no SSA value.
- **constraints and limitations:**
  The implicit update state consumed by this flush MUST correspond to the same
  store stream that produced `%value`. The first store-side state in a stream
  should be created by `pto.init_align`.
- **Latency:** **9** cycles.

---

##### `pto.vstar`

- **syntax:** `pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>`
- **semantics:** Flush remaining alignment state.
- **inputs:**
  `%value` is the pending alignment/buffer state that still needs to be emitted,
  and `%dest` is the UB destination base pointer.
- **outputs:**
  No SSA result. The effect is a memory-side flush that writes the remaining
  buffered bytes to memory.
- **constraints and limitations:**
  This op terminates an unaligned-store sequence. It MUST be paired with a
  compatible prior state-producing store sequence so that the pending tail state
  is well-defined.
- **Latency:** **9** cycles.

---

#### Stateful Store Ops

These ops make reference-updated state explicit as SSA results.

##### `pto.vstus`

- **syntax:** `%align_out = pto.vstus %align_in, %offset, %value, %base : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`
- **semantics:** No-post unaligned store with scalar offset.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%offset` is the scalar
  displacement, `%value` is the vector being stored, and `%base` is the UB base
  pointer.
- **outputs:**
  `%align_out` is the updated buffered-tail state.
- **constraints and limitations:**
  This is the scalar-offset stateful form of the unaligned store family. The
  scalar offset width MUST match the selected form, and a later flush op is
  still required. The first `%align_in` in the stream should come from
  `pto.init_align`. This op does not mean "store a full vector starting at
  `%base + %offset`". Instead, `%offset` describes how far the store stream
  advances at this step, and `%align_out` carries any residual tail that could
  not be committed yet. The no-post surface does not expose an updated base
  pointer. A later flush op must therefore use an explicit destination/offset
  pair that identifies the same logical flush point as this `pto.vstus`.
- **Latency:** **9** cycles.

---

##### `pto.vstur`

- **syntax:** `%align_out = pto.vstur %align_in, %value, %base, "MODE" : !pto.align, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`
- **semantics:** Unaligned store with residual flush and SPR-AR-driven state update.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%value` is the vector to
  store, `%base` is the UB base pointer, and `MODE` selects whether the
  hardware updates `SPR AR` after the store.
- **outputs:**
  `%align_out` is the updated residual state after the current partial store.
- **constraints and limitations:**
  The effective address is `base + AR`, where `AR` is the hardware SPR state
  carried outside SSA. `POST_UPDATE` means hardware may advance `SPR AR`
  according to the fixed `SPR SQZN` configuration; `NO_POST_UPDATE` preserves
  the current `SPR AR` value. This form exposes only the evolving residual
  align-state in SSA; it does not by itself guarantee that all buffered bytes
  have reached memory. A compatible final flush is still required unless the
  surrounding sequence is known to be complete. Independent sequences typically
  begin from `AR = 0`; if the surrounding program does not already guarantee
  that, the hardware sequence should clear `SPR AR` before the first dependent
  `pto.vstur`. The first `%align_in` in the stream should come from
  `pto.init_align`. `pto.vstur` also consumes the fixed `SPR SQZN` state, so a
  preceding squeeze producer such as `pto.vsqz` / `pto.vusqz` MUST establish
  the byte count before the store. `MODE` MUST be one of `POST_UPDATE` or
  `NO_POST_UPDATE`.
- **Latency:** **9** cycles.

<a id="micro-04-predicate-load-store"></a>

### 4. Predicate Load/Store

> **Category:** UB ↔ Predicate Register data movement
> **Pipeline:** PIPE_V (Vector Core)

Predicate registers (`!pto.mask<G>`) are 256-bit registers that enable per-lane conditional execution. These ops move predicate values between UB and predicate registers.

In concrete examples, `G` should be chosen to match the consumer family. The
examples below use `b32` when the loaded/stored mask is used with `f32`
vector compares or selects.

The predicate load/store ops documented on this page always use explicit
`base[offset]` addressing. The immediate forms (`pldi`, `psti`) and dynamic
forms (`plds`, `psts`) differ only in how `%offset` is supplied.

---

#### Predicate Loads

##### `pto.plds`

- **syntax:** `%result = pto.plds %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.mask<G>`
- **semantics:** Load predicate register with runtime offset. This is the
  dynamic-offset form of `pto.pldi`: the predicate payload interpretation is
  the same, but `%offset` is supplied as an SSA `index` instead of a constant
  `index` immediate.
- **DIST:** mandatory string token, one of `NORM`, `US`, `DS`.
  - `NORM`: load a normal packed predicate payload of size `VL/8`.
  - `US`: load a packed predicate payload of size `VL/16`, then duplicate each
    loaded bit once.
  - `DS`: load a packed predicate payload of size `2 * VL/8`, then keep one
    bit out of every two bits.

The loaded payload is a packed predicate image in UB. Consumer ops interpret
the resulting `!pto.mask<G>` according to the mask granularity `G`.
`pto.plds` only
models the explicit `base[offset]` form.

**Example:**
```mlir
%mask = pto.plds %ub[%c0], "NORM" : !pto.ptr<T, ub>, index -> !pto.mask<G>
```

---

##### `pto.pldi`

- **syntax:** `%result = pto.pldi %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.mask<G>`
- **offset:** must be a constant `index` immediate in PTO surface form.
- **semantics:** Load predicate register with immediate offset.
- **DIST:** mandatory string token, one of `NORM`, `US`, `DS`.
  - `NORM`: load a normal packed predicate payload of size `VL/8`.
  - `US`: load a packed predicate payload of size `VL/16`, then duplicate each
    loaded bit once.
  - `DS`: load a packed predicate payload of size `2 * VL/8`, then keep one
    bit out of every two bits.

Like `pto.plds`, this op reads a packed predicate payload from UB and
materializes it as `!pto.mask<G>`.

---

#### Predicate Stores

##### `pto.psts`

- **syntax:** `pto.psts %value, %dest[%offset], "DIST" : !pto.mask<G>, !pto.ptr<T, ub>, index`
- **semantics:** Store predicate register with runtime offset. This is the
  dynamic-offset form of `pto.psti`: the predicate payload interpretation is
  the same, but `%offset` is supplied as an SSA `index` instead of a constant
  `index` immediate.
- **DIST:** mandatory string token, one of `NORM`, `PK`.
  - `NORM`: store the packed predicate payload into a normal destination space
    of size `VL/8`.
  - `PK`: store the packed predicate payload into a destination space of size
    `VL/16`, keeping one bit out of every two bits.

`pto.psts` stores the packed predicate payload represented by `!pto.mask<G>`.
It only models the explicit `base[offset]` form.

**Example:**
```mlir
pto.psts %mask, %ub[%c0], "NORM" : !pto.mask<G>, !pto.ptr<T, ub>, index
```

---

##### `pto.psti`

- **syntax:** `pto.psti %value, %dest[%offset], "DIST" : !pto.mask<G>, !pto.ptr<T, ub>, index`
- **offset:** must be a constant `index` immediate in PTO surface form.
- **semantics:** Store predicate register with immediate offset.
- **DIST:** mandatory string token, one of `NORM`, `PK`.
  - `NORM`: store the packed predicate payload into a normal destination space
    of size `VL/8`.
  - `PK`: store the packed predicate payload into a destination space of size
    `VL/16`, keeping one bit out of every two bits.

`pto.psti` and `pto.psts` store the packed predicate payload represented by
`!pto.mask<G>`. The surface distinction is only immediate-offset versus
dynamic-offset.

---

##### `pto.pstu`

- **syntax:** `%align_out, %base_out = pto.pstu %align_in, %value, %base : !pto.align, !pto.mask<b16>, !pto.ptr<ui16, ub> -> !pto.align, !pto.ptr<ui16, ub>`
- **syntax:** `%align_out, %base_out = pto.pstu %align_in, %value, %base : !pto.align, !pto.mask<b32>, !pto.ptr<ui32, ub> -> !pto.align, !pto.ptr<ui32, ub>`
- **semantics:** Predicate unaligned store with align/base state update. The base type is fixed by mask granularity: `b16 <-> ui16`, `b32 <-> ui32`.
- **outputs:**
  `%align_out` and `%base_out` are the updated unaligned-store state and are
  intended to be used by a later `pto.pstu` call.
- **constraints and limitations:**
  The first `%align_in` in a predicate unaligned-store stream should come from
  `pto.init_align`.

---

#### Typical Usage Pattern

```mlir
// Generate comparison mask
%mask = pto.vcmp %v0, %v1, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>

// Store mask to UB for later use
pto.psts %mask, %ub_mask[%c0], "NORM" : !pto.mask<b32>, !pto.ptr<T, ub>, index

// ... later in another kernel ...

// Load mask from UB
%saved_mask = pto.plds %ub_mask[%c0], "NORM" : !pto.ptr<T, ub>, index -> !pto.mask<b32>

// Use for predicated select
%result = pto.vsel %v_true, %v_false, %saved_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

<a id="micro-05-materialization-predicate"></a>

### 5. Materialization & Predicate Ops

> **Category:** Scalar broadcast, predicate generation and manipulation
> **Pipeline:** PIPE_V (Vector Core)

These ops create vectors from scalar values and manipulate predicate registers.

#### Common Operand Model

- `%value` is the scalar source value in SSA form.
- `%input` is either a source scalar or a source vector depending on the op.
- `%result` is the destination vector register value.
- For 32-bit scalar inputs, the scalar source MUST satisfy the backend's legal
  scalar-source constraints for this family.

---

#### Scalar Materialization

##### `pto.vbr`

- **syntax:** `%result = pto.vbr %value : T -> !pto.vreg<NxT>`
- **semantics:** Broadcast scalar to all vector lanes.
- **inputs:**
  `%value` is the scalar source.
- **outputs:**
  `%result` is a vector whose active lanes all carry `%value`.
- **constraints and limitations:**
  Supported forms are `b8`, `b16`, and `b32`. For `b8`, only the low 8 bits of
  the scalar source are consumed.

```c
for (int i = 0; i < N; i++)
    dst[i] = value;
```

**Example:**
```mlir
%one = pto.vbr %c1_f32 : f32 -> !pto.vreg<64xf32>
```

---

##### `pto.vdup`

- **syntax:** `%result = pto.vdup %input, %mask {position = "LOWEST|HIGHEST"} : T|!pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Duplicate scalar or vector element to all lanes.
- **inputs:**
  `%input` supplies the scalar or source-lane value selected by `position`,
  and `%mask` controls the active lanes.
- **outputs:**
  `%result` is the duplicated vector.
- **constraints and limitations:**
  `position` selects which source vector element is duplicated and is only valid
  for vector input. `position` defaults to `LOWEST`.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? input_scalar_or_element : 0;
```

---

#### Predicate Generation

##### `pto.pset_b8` / `pto.pset_b16` / `pto.pset_b32`

- **syntax:** `%result = pto.pset_b8 "PATTERN" : !pto.mask<b8>`
- **syntax:** `%result = pto.pset_b16 "PATTERN" : !pto.mask<b16>`
- **syntax:** `%result = pto.pset_b32 "PATTERN" : !pto.mask<b32>`
- **semantics:** Materialize a predicate register from a named pattern token.

**Supported pattern tokens:**

| Pattern | Description |
|---------|-------------|
| `PAT_ALL` | All lanes active |
| `PAT_ALLF` | All lanes inactive |
| `PAT_H` | High half active |
| `PAT_Q` | Upper quarter active |
| `PAT_VL1`...`PAT_VL128` | First N logical lanes active |
| `PAT_M3`, `PAT_M4` | Modular patterns |

`PAT_ALL` is the PTO spelling of the VISA-style all-true predicate pattern.
The other tokens listed above are also concrete installed-toolchain pattern
objects, not PTO-only aliases.

**Example — All 64 f32 lanes active:**
```mlir
%all_active = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
```

**Example — First 16 lanes active:**
```mlir
%first_16 = pto.pset_b32 "PAT_VL16" : !pto.mask<b32>
```

---

##### `pto.pge_b8` / `pto.pge_b16` / `pto.pge_b32`

- **syntax:** `%result = pto.pge_b8 "PATTERN" : !pto.mask<b8>`
- **syntax:** `%result = pto.pge_b16 "PATTERN" : !pto.mask<b16>`
- **syntax:** `%result = pto.pge_b32 "PATTERN" : !pto.mask<b32>`
- **semantics:** Generate a predicate from a lane-count pattern token. In the
  common tail-mask form, `PAT_VL<N>` marks the first `N` logical lanes active.
- **supported pattern tokens:** `PAT_ALL`, `PAT_ALLF`, `PAT_H`, `PAT_Q`,
  `PAT_VL1`, `PAT_VL2`, `PAT_VL3`, `PAT_VL4`, `PAT_VL8`, `PAT_VL16`,
  `PAT_VL32`, `PAT_VL64`, `PAT_VL128`, `PAT_M3`, `PAT_M4`

```c
for (int i = 0; i < TOTAL_LANES; i++)
    mask[i] = (i < len);
```

**Example — Tail mask for remainder loop:**
```mlir
%tail_mask = pto.pge_b32 "PAT_VL8" : !pto.mask<b32>
```

---

##### `pto.plt_b8` / `pto.plt_b16` / `pto.plt_b32`

- **syntax:** `%mask, %scalar_out = pto.plt_b8 %scalar : i32 -> !pto.mask<b8>, i32`
- **syntax:** `%mask, %scalar_out = pto.plt_b16 %scalar : i32 -> !pto.mask<b16>, i32`
- **syntax:** `%mask, %scalar_out = pto.plt_b32 %scalar : i32 -> !pto.mask<b32>, i32`
- **semantics:** Generate a tail-style predicate from an SSA lane-count value.
  On A5/V300-style toolchains, this family is exposed as a post-update wrapper:
  the predicate result becomes `%mask`, and the wrapper's carry-out scalar state
  is surfaced as `%scalar_out`.
- **inputs:**
  `%scalar` is the incoming lane-count / remaining-count state.
- **outputs:**
  `%mask` is the generated predicate.
  `%scalar_out` is the post-update scalar carry-out from the same `plt` call
  and can be threaded into a subsequent `pto.plt_b*` call in the same chain.

```c
for (int i = 0; i < VL_t; ++i)
    mask[i] = (i < scalar_in);

scalar_out = (scalar_in < VL_t) ? 0 : (scalar_in - VL_t);
```

Where `VL_t` is the logical lane count of the concrete op variant:

- `pto.plt_b8`: `VL_t = 256`
- `pto.plt_b16`: `VL_t = 128`
- `pto.plt_b32`: `VL_t = 64`

---

##### `pto.pltm_b8` / `pto.pltm_b16` / `pto.pltm_b32`

- **syntax:** `%mask = pto.pltm_b8 %loop, %bound : i16, i32 -> !pto.mask<b8>`
- **syntax:** `%mask = pto.pltm_b16 %loop, %bound : i16, i32 -> !pto.mask<b16>`
- **syntax:** `%mask = pto.pltm_b32 %loop, %bound : i16, i32 -> !pto.mask<b32>`
- **semantics:** Generate a loop-indexed tail predicate without updating the
  bound scalar.
- **inputs:**
  `%loop` is the current logical loop index multiplier and `%bound` is the
  total element bound.
- **outputs:**
  `%mask` is true for lanes whose logical index is still below `%bound`.

```c
for (int i = 0; i < VL_t; ++i)
    mask[i] = (i + loop * VL_t) < bound;
```

Where `VL_t` is the logical lane count of the concrete op variant:

- `pto.pltm_b8`: `VL_t = 256`
- `pto.pltm_b16`: `VL_t = 128`
- `pto.pltm_b32`: `VL_t = 64`

Unlike `pto.plt_b*`, `pto.pltm_b*` does not return a post-update scalar. It is
used when the loop index is already tracked separately in scalar SSA.

---

#### Predicate Pack/Unpack

##### `pto.ppack`

- **syntax:** `%result = pto.ppack %input, "PART" : !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Narrowing pack of predicate register.
- **part tokens:**
  - `LOWER`: pack into the lower half of `%result`; the upper half is zeroed.
  - `HIGHER`: pack into the higher half of `%result`; the lower half is zeroed.

Conceptually, `pto.ppack` keeps one bit out of each adjacent 2-bit group from
`%input`, packs those kept bits into the selected half of `%result`, and fills
the other half with zeros.

```c
// Let VL be the logical lane count of the destination predicate.
// LOWER
for (int i = 0; i < VL / 2; ++i)
    result[i] = input[2 * i];
for (int i = VL / 2; i < VL; ++i)
    result[i] = 0;

// HIGHER
for (int i = 0; i < VL / 2; ++i)
    result[VL / 2 + i] = input[2 * i];
for (int i = 0; i < VL / 2; ++i)
    result[i] = 0;
```

---

##### `pto.punpack`

- **syntax:** `%result = pto.punpack %input, "PART" : !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Widening unpack of predicate register.
- **part tokens:**
  - `LOWER`: unpack from the lower half of `%input`.
  - `HIGHER`: unpack from the higher half of `%input`.

Conceptually, `pto.punpack` reads the selected half of `%input`, zero-extends
each 1-bit predicate element into a 2-bit group in `%result`, and leaves the
expanded image in the full destination predicate register.

```c
// Let VL be the logical lane count of the destination predicate.
// LOWER
for (int i = 0; i < VL / 2; ++i) {
    result[2 * i] = input[i];
    result[2 * i + 1] = 0;
}

// HIGHER
for (int i = 0; i < VL / 2; ++i) {
    result[2 * i] = input[VL / 2 + i];
    result[2 * i + 1] = 0;
}
```

---

#### Predicate Logical Ops

##### `pto.pand`

- **syntax:** `%result = pto.pand %src0, %src1, %mask : !pto.mask<G>, !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate bitwise AND gated by a governing predicate.

Inactive lanes selected out by `%mask` are zeroed.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? (src0[i] & src1[i]) : 0;
```

---

##### `pto.por`

- **syntax:** `%result = pto.por %src0, %src1, %mask : !pto.mask<G>, !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate bitwise OR gated by a governing predicate.

Inactive lanes selected out by `%mask` are zeroed.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? (src0[i] | src1[i]) : 0;
```

---

##### `pto.pxor`

- **syntax:** `%result = pto.pxor %src0, %src1, %mask : !pto.mask<G>, !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate bitwise XOR gated by a governing predicate.

Inactive lanes selected by `%mask` are zeroed.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? (src0[i] ^ src1[i]) : 0;
```

---

##### `pto.pnot`

- **syntax:** `%result = pto.pnot %input, %mask : !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate bitwise NOT gated by a governing predicate.

Inactive lanes selected by `%mask` are zeroed.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? (~src[i]) : 0;
```

---

##### `pto.psel`

- **syntax:** `%result = pto.psel %src0, %src1, %sel : !pto.mask<G>, !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate select (mux). `%sel` is the governing predicate that
  chooses lanes from `%src0` or `%src1`.

```c
for (int i = 0; i < N; i++)
    dst[i] = sel[i] ? src0[i] : src1[i];
```

---

##### `pto.pdintlv_b8` / `pto.pdintlv_b16` / `pto.pdintlv_b32`

- **syntax:** `%low, %high = pto.pdintlv_b8 %src0, %src1 : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>`
- **syntax:** `%low, %high = pto.pdintlv_b16 %src0, %src1 : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>`
- **syntax:** `%low, %high = pto.pdintlv_b32 %src0, %src1 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>`
- **semantics:** De-interleave two predicate sources and return the two
  de-interleaved predicate images in the same predicate element family.

---

##### `pto.pintlv_b8` / `pto.pintlv_b16` / `pto.pintlv_b32`

- **syntax:** `%low, %high = pto.pintlv_b8 %src0, %src1 : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>`
- **syntax:** `%low, %high = pto.pintlv_b16 %src0, %src1 : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>`
- **syntax:** `%low, %high = pto.pintlv_b32 %src0, %src1 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>`
- **semantics:** Interleave two predicate sources and return the two
  resulting predicate images in the same predicate element family.

---

#### Typical Usage

```mlir
// Generate all-active mask for f32 (64 lanes)
%all = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>

// Generate tail mask for remainder (last 12 elements)
%tail = pto.pge_b32 "PAT_VL12" : !pto.mask<b32>

// Compare and generate mask
%cmp_mask = pto.vcmp %a, %b, %all, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>

// Combine masks: only process tail elements that passed comparison
%combined = pto.pand %cmp_mask, %tail, %all : !pto.mask<b32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>

// Use for predicated operation
%result = pto.vsel %true_vals, %false_vals, %combined : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

<a id="micro-06-unary-vector-ops"></a>

### 6. Unary Vector Ops

> **Category:** Single-input vector operations
> **Pipeline:** PIPE_V (Vector Core)

Element-wise operations that take one vector input and produce one vector output.

#### Common Operand Model

- `%input` is the source vector register value.
- `%mask` is the predicate operand. For this family, inactive lanes follow the
  predication behavior of the selected instruction form: zeroing forms
  zero-fill inactive lanes, while merging forms preserve the destination value.
- `%result` is the destination vector register value. Unless stated otherwise,
  `%result` has the same lane count and element type as `%input`.

#### CA latency (A5, Ascend910_9599 CA)

Cycle-accurate simulator **popped→retire** latency (cycles). **fp16** values use **aclFloat16** in traces where measured. **bf16:** no simple-tile ST coverage on this surface; treat as **—**.

| PTO op | RV (CA) | fp32 | fp16 | bf16 |
|--------|---------|------|------|------|
| `pto.vabs` | `RV_VABS_FP` | **5** | **5** | — |
| `pto.vneg` | `RV_VMULS` | **8** | **8** | — |
| `pto.vexp` | `RV_VEXP` | **16** | **21** | — |
| `pto.vln` | `RV_VLN` | **18** | **23** | — |
| `pto.vsqrt` | `RV_VSQRT` | **17** | **22** | — |
| `pto.vrelu` | `RV_VRELU` | **5** | **5** | — |
| `pto.vnot` | `RV_VNOT` | — | int-only paths | — |
| `pto.vmov` | `RV_VLD` proxy | **9** | **9** | — |

---

#### Arithmetic

##### `pto.vabs`

- **syntax:** `%result = pto.vabs %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < 0) ? -src[i] : src[i];
```

- **inputs:** `%input` supplies the source lanes and `%mask` selects which lanes
  participate.
- **outputs:** `%result` receives the lane-wise absolute values.
- **constraints and limitations:** Source and result types MUST match. On A5,
  integer overflow follows the ISA default truncation behavior for this family;
  `pto.vabs` is not an explicit saturating op.

---

##### `pto.vneg`

- **syntax:** `%result = pto.vneg %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = -src[i];
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise arithmetic negation.
- **constraints and limitations:** Source and result types MUST match.

---

#### Transcendental

##### `pto.vexp`

- **syntax:** `%result = pto.vexp %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds `exp(input[i])` per active lane.
- **constraints and limitations:** Only floating-point element types are legal.

---

##### `pto.vln`

- **syntax:** `%result = pto.vln %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = logf(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the natural logarithm per active lane.
- **constraints and limitations:** Only floating-point element types are legal.
  For real-number semantics, active inputs SHOULD be strictly positive; non-
  positive inputs follow the target's exception/NaN rules.

---

##### `pto.vsqrt`

- **syntax:** `%result = pto.vsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = sqrtf(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the square root per active lane.
- **constraints and limitations:** Only floating-point element types are legal.
  Negative active inputs follow the target's exception/NaN rules.

---

#### Activation

##### `pto.vrelu`

- **syntax:** `%result = pto.vrelu %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** si32, i32, f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > 0) ? src[i] : 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds `max(input[i], 0)` per active lane.
- **constraints and limitations:** Signed or signless 32-bit integer and
  floating-point element types are legal on the current A5 surface described
  here.

---

#### Bitwise

##### `pto.vnot`

- **syntax:** `%result = pto.vnot %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = ~src[i];
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the lane-wise bitwise inversion.
- **constraints and limitations:** Integer element types only.

---

#### Movement

#### Typical Usage

```mlir
// Softmax numerator: exp(x - max)
%sub = pto.vsub %x, %max_broadcast, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
%exp = pto.vexp %sub, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>

// ReLU activation
%activated = pto.vrelu %linear_out, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
```

<a id="micro-07-binary-vector-ops"></a>

### 7. Binary Vector Ops

> **Category:** Two-input vector operations
> **Pipeline:** PIPE_V (Vector Core)

Element-wise operations that take two vector inputs and produce one vector output.

#### Common Operand Model

- `%lhs` and `%rhs` are the two source vector register values.
- `%mask` is the predicate operand `Pg` that gates which lanes participate.
- `%result` is the destination vector register value. Unless explicitly noted,
  it has the same lane count and element type as the inputs.
- Unless explicitly documented otherwise, `%lhs`, `%rhs`, and `%result` MUST
  have matching vector shapes and element types.

#### CA latency (A5, Ascend910_9599 CA)

Cycle-accurate simulator **popped→retire** latency (cycles). **fp16** uses **aclFloat16** in measured traces. **bf16:** — (no dedicated vec tile ST on this surface).

| PTO op | RV (CA) | fp32 | fp16 | bf16 |
|--------|---------|------|------|------|
| `pto.vadd` | `RV_VADD` | **7** | **7** | — |
| `pto.vsub` | `RV_VSUB` | **7** | **7** | — |
| `pto.vmul` | `RV_VMUL` | **8** | **8** | — |
| `pto.vdiv` | `RV_VDIV` | **17** | **22** | — |
| `pto.vmadd` | `RV_VMADD` | — | — | — |

---

#### Arithmetic

##### `pto.vadd`

- **syntax:** `%result = pto.vadd %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i8-i64, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] + src1[i];
```

- **inputs:** `%lhs` and `%rhs` are added lane-wise; `%mask` selects active
  lanes.
- **outputs:** `%result` is the lane-wise sum.
- **constraints and limitations:** Input and result types MUST match.

---

##### `pto.vsub`

- **syntax:** `%result = pto.vsub %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i8-i64, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] - src1[i];
```

- **inputs:** `%lhs` is the minuend, `%rhs` is the subtrahend, and `%mask`
  selects active lanes.
- **outputs:** `%result` is the lane-wise difference.
- **constraints and limitations:** Input and result types MUST match.

---

##### `pto.vmul`

- **syntax:** `%result = pto.vmul %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, bf16, f32 (**NOT** i8/ui8)

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] * src1[i];
```

- **inputs:** `%lhs` and `%rhs` are multiplied lane-wise; `%mask` selects
  active lanes.
- **outputs:** `%result` is the lane-wise product.
- **constraints and limitations:** The current A5 profile excludes `i8/ui8`
  forms from this surface.

---

##### `pto.vdiv`

- **syntax:** `%result = pto.vdiv %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32 only (no integer division)

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] / src1[i];
```

- **inputs:** `%lhs` is the numerator, `%rhs` is the denominator, and `%mask`
  selects active lanes.
- **outputs:** `%result` is the lane-wise quotient.
- **constraints and limitations:** Floating-point element types only. Active
  denominators containing `+0` or `-0` follow the target's exceptional
  behavior.

---

##### `pto.vmax`

- **syntax:** `%result = pto.vmax %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src0[i] > src1[i]) ? src0[i] : src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` holds the lane-wise maximum.
- **constraints and limitations:** Input and result types MUST match.

---

##### `pto.vmin`

- **syntax:** `%result = pto.vmin %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src0[i] < src1[i]) ? src0[i] : src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` holds the lane-wise minimum.
- **constraints and limitations:** Input and result types MUST match.

---

##### `pto.vmadd`

- **syntax:** `%result = pto.vmadd %acc, %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] * acc[i] + src1[i];
```

- **inputs:** `%acc` is the destination-as-source multiplicand, `%lhs` is the
  other multiplicand, `%rhs` is the addend, and `%mask` selects active lanes.
- **outputs:** `%result` holds the multiply-add result.
- **constraints and limitations:** `%acc`, `%lhs`, `%rhs`, and `%result` MUST
  have matching vector shapes and element types. This is a direct fused
  multiply-add semantic and should not be assumed equivalent to separate
  multiply and add operations for floating-point results.

---

#### Bitwise

##### `pto.vand`

- **syntax:** `%result = pto.vand %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] & src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise AND.
- **constraints and limitations:** Integer element types only.

---

##### `pto.vor`

- **syntax:** `%result = pto.vor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] | src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise OR.
- **constraints and limitations:** Integer element types only.

---

##### `pto.vxor`

- **syntax:** `%result = pto.vxor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] ^ src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise XOR.
- **constraints and limitations:** Integer element types only.

---

#### Shift

##### `pto.vshl`

- **syntax:** `%result = pto.vshl %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] << src1[i];
```

- **inputs:** `%lhs` supplies the shifted value, `%rhs` supplies the per-lane
  shift amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only. Shift counts
  SHOULD stay within `[0, bitwidth(T) - 1]`; out-of-range behavior is target-
  defined unless the verifier narrows it further.

---

##### `pto.vshr`

- **syntax:** `%result = pto.vshr %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] >> src1[i];  // arithmetic for signed, logical for unsigned
```

- **inputs:** `%lhs` supplies the shifted value, `%rhs` supplies the per-lane
  shift amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only. Signedness of the
  element type determines arithmetic vs logical behavior.

---

#### Carry Operations

##### `pto.vaddc`

- **syntax:** `%result, %carry = pto.vaddc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
- **semantics:** Add with carry output.

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i];
    dst[i] = (T)r;
    carry[i] = (r >> bitwidth);
}
```

- **inputs:** `%lhs` and `%rhs` are added lane-wise and `%mask` selects active
  lanes.
- **outputs:** `%result` is the truncated arithmetic result and `%carry` is the
  carry/overflow predicate per lane.
- **A5 types:** `i32`, `si32`, `ui32`
- **constraints and limitations:** This is a carry-chain integer add family. On
  the current A5 surface, only 32-bit integer element types are supported.
  `%mask` and `%carry` therefore use the same typed-mask granularity as the
  data vector family, which on the current documented A5 surface means
  `!pto.mask<b32>`.

---

##### `pto.vsubc`

- **syntax:** `%result, %carry = pto.vsubc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
- **semantics:** Subtract with per-lane carry output.

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i];
    carry[i] = (src0[i] >= src1[i]);
}
```

- **inputs:** `%lhs` and `%rhs` are subtracted lane-wise and `%mask` selects
  active lanes.
- **outputs:** `%result` is the arithmetic difference and `%carry` is the
  per-lane carry predicate. For this subtraction family, active lanes set
  `%carry[i] = 1` when the subtraction completes without borrow, and
  `%carry[i] = 0` when a borrow occurs.
- **A5 types:** `i32`, `si32`, `ui32`
- **constraints and limitations:** This operation is currently restricted to
  the 32-bit integer carry/borrow-chain family. `%mask` and `%carry`
  therefore use the same typed-mask granularity as the data vector family,
  which on the current documented A5 surface means `!pto.mask<b32>`.

---

#### Typical Usage

```mlir
// Vector addition
%sum = pto.vadd %a, %b, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>

// Element-wise multiply
%prod = pto.vmul %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>

// Clamp to range [min, max]
%clamped_low = pto.vmax %input, %min_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
%clamped = pto.vmin %clamped_low, %max_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>

// Bit manipulation
%masked = pto.vand %data, %bitmask, %mask : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<G> -> !pto.vreg<64xi32>
```

<a id="micro-08-vec-scalar-ops"></a>

### 8. Vec-Scalar Ops

> **Category:** Vector-scalar operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that combine a vector with a scalar value, applying the scalar to every lane.

#### Common Operand Model

- `%input` is the source vector register value.
- `%scalar` is the scalar operand in SSA form.
- `%mask` is the predicate operand.
- `%result` is the destination vector register value.
- For 32-bit scalar forms, the scalar source MUST satisfy the backend's legal
  scalar-source constraints for this family.
- For elementwise vec-scalar families whose scalar conceptually matches the
  vector element type (`pto.vadds`, `pto.vmuls`, `pto.vmaxs`,
  `pto.vmins`, `pto.vlrelu`):
  - signed integer vectors accept signed integer scalars with the same width,
    and also accept signless `i<width>`
  - unsigned integer vectors accept unsigned integer scalars with the same
    width, and also accept signless `i<width>`
  - signless integer vectors accept signless `i<width>`
- `pto.vshls` and `pto.vshrs` are not part of that rule; their scalar operand
  is the shift amount and remains fixed to `i16`.

#### CA latency (A5, Ascend910_9599 CA)

Cycle-accurate simulator **popped→retire** latency (cycles). **fp16** uses **aclFloat16** in measured traces. **bf16:** —.

| PTO op | RV (CA) | fp32 | fp16 | bf16 |
|--------|---------|------|------|------|
| `pto.vadds` | `RV_VADDS` | **7** | **7** | — |
| `pto.vmuls` | `RV_VMULS` | **8** | **8** | — |

---

#### Arithmetic

##### `pto.vadds`

- **syntax:** `%result = pto.vadds %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** `si8`, `si16`, `si32`, `ui8`, `ui16`, `ui32`, `f16`, `bf16`, `f32`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] + scalar;
```

- **inputs:** `%input` is the source vector, `%scalar` is broadcast logically to
  each lane, and `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise sum.
- **constraints and limitations:** Input vector element type, scalar type, and
  result vector element type MUST match. For integer vector forms, `%scalar`
  may also use matching-signedness integer or signless `i<width>` with the same
  bit width as the vector element type, so it can be fed directly from `arith`
  constants.

---

##### `pto.vmuls`

- **syntax:** `%result = pto.vmuls %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] * scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise product.
- **constraints and limitations:** Supported element types are hardware-family
  specific; the current PTO micro Instruction documentation covers the common
  numeric cases. For integer vector forms, `%scalar` may use matching-signedness
  integer or signless `i<width>` with the same bit width as the vector element
  type.

---

##### `pto.vmaxs`

- **syntax:** `%result = pto.vmaxs %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > scalar) ? src[i] : scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise maximum.
- **constraints and limitations:** Input and result types MUST match. For
  integer vector forms, `%scalar` may use matching-signedness integer or
  signless `i<width>` with the same bit width as the vector element type.

---

##### `pto.vmins`

- **syntax:** `%result = pto.vmins %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < scalar) ? src[i] : scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise minimum.
- **constraints and limitations:** Input and result types MUST match. For
  integer vector forms, `%scalar` may use matching-signedness integer or
  signless `i<width>` with the same bit width as the vector element type.

---

#### Shift

##### `pto.vshls`

- **syntax:** `%result = pto.vshls %input, %scalar, %mask : !pto.vreg<NxT>, i16, !pto.mask<G> -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] << scalar;
```

- **inputs:** `%input` is the value vector, `%scalar` is the uniform `i16` shift
  amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only. The shift amount
  SHOULD stay within the source element width.

---

##### `pto.vshrs`

- **syntax:** `%result = pto.vshrs %input, %scalar, %mask : !pto.vreg<NxT>, i16, !pto.mask<G> -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] >> scalar;
```

- **inputs:** `%input` is the value vector, `%scalar` is the uniform `i16` shift
  amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only.

---

##### `pto.vlrelu`

- **syntax:** `%result = pto.vlrelu %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : scalar * src[i];
```

- **inputs:** `%input` is the activation vector, `%scalar` is the leaky slope,
  and `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise leaky-ReLU result.
- **constraints and limitations:** Only `f16` and `f32` forms are currently
  documented for `pto.vlrelu`.

---

#### Carry Operations

##### `pto.vaddcs`

- **syntax:** `%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
- **semantics:** Add with carry-in and carry-out.

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i] + carry_in[i];
    dst[i] = (T)r;
    carry_out[i] = (r >> bitwidth);
}
```

- **inputs:** `%lhs` and `%rhs` are the value vectors, `%carry_in` is the
  incoming carry predicate, and `%mask` selects active lanes.
- **outputs:** `%result` is the arithmetic result and `%carry` is the carry-out
  predicate.
- **A5 types:** `i32`, `si32`, `ui32`
- **constraints and limitations:** This is the scalar-extended carry-chain
  family. On the current A5 surface, only 32-bit integer element types are
  supported. `%carry_in`, `%mask`, and `%carry` therefore all use the same
  typed-mask granularity as the data vector family, which on the current
  documented A5 surface means `!pto.mask<b32>`.

---

##### `pto.vsubcs`

- **syntax:** `%result, %carry = pto.vsubcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
- **semantics:** Subtract with carry input and output.

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i] - (1 - carry_in[i]);
    carry_out[i] = (src0[i] >= src1[i] + (1 - carry_in[i]));
}
```

- **inputs:** `%lhs` and `%rhs` are the value vectors, `%carry_in` is the
  incoming carry predicate, and `%mask` selects active lanes.
- **outputs:** `%result` is the arithmetic result and `%carry` is the
  carry predicate after the lane-wise subtraction. For this subtraction family,
  active lanes set `%carry[i] = 1` when the subtraction completes without
  borrow, and `%carry[i] = 0` when a borrow occurs.
- **A5 types:** `i32`, `si32`, `ui32`
- **constraints and limitations:** This is the scalar-extended borrow-chain
  family and is currently restricted to 32-bit integer element types.
  `%carry_in`, `%mask`, and `%carry` therefore all use the same typed-mask
  granularity as the data vector family, which on the current documented A5
  surface means `!pto.mask<b32>`.

---

#### Typical Usage

```mlir
// Add bias to all elements
%biased = pto.vadds %activation, %bias_scalar, %mask : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.vreg<64xf32>

// Scale by constant
%scaled = pto.vmuls %input, %scale, %mask : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.vreg<64xf32>

// Clamp to [0, 255] for uint8 quantization
%clamped_low = pto.vmaxs %input, %c0, %mask : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.vreg<64xf32>
%clamped = pto.vmins %clamped_low, %c255, %mask : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.vreg<64xf32>

// Shift right by fixed amount
%shifted = pto.vshrs %data, %c4, %mask : !pto.vreg<64xi32>, i16, !pto.mask<G> -> !pto.vreg<64xi32>
```

<a id="micro-09-conversion-ops"></a>

### 9. Conversion Ops

> **Category:** Type conversion operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that convert between data types (float/int, narrowing/widening).

#### Common Operand Model

- `%input` is the source vector register value.
- `%mask` is the predicate mask that selects active conversion lanes.
- `%result` is the destination vector register value.
- `rnd`, `sat`, and `part` are optional attributes that refine
  conversion behavior when the selected source/destination type pair needs
  rounding, saturation, or lane placement control.
- The single `pto.vcvt` surface covers float-int, float-float, int-float, and
  int-int conversion families.

#### CA latency (A5, Ascend910_9599 CA)

Cycle-accurate simulator **popped→retire** latency (cycles). Only representative traces below; other `pto.vcvt` conversion pairs depend on the RV lowering in the trace.

| PTO op | RV (CA) | Note | Latency |
|--------|---------|------|---------|
| `pto.vcvt` | `RV_VCVT_F2F` | f32→f16 | **7** |
| `pto.vci` | — | no vector `RV_*` in sampled `veccore0` trace | — |

---

#### `pto.vci`

- **syntax:** `%result = pto.vci %index {order = "ASC|DESC"} : T -> !pto.vreg<NxT>`
- **semantics:** Generate a lane-index vector from a scalar base value.
- **inputs:**
  `%index` is the scalar base value. Supported scalar types are `i8/i16/i32`,
  `f16`, and `f32`.
- **outputs:**
  `%result` is the generated index vector.
- **constraints and limitations:**
  This is an index-generation family, not a numeric conversion. `order` and
  the result element type together determine whether lanes are generated as
  `base + lane_id` or `base - lane_id`. Supported result types are
  `!pto.vreg<256xsi8>`, `!pto.vreg<128xsi16>`, `!pto.vreg<64xsi32>`,
  `!pto.vreg<128xf16>`, and `!pto.vreg<64xf32>`. `%index` must use the
  matching scalar type for `f16`/`f32`; for integer results, `%index` must use
  the same bit width and may be signless or signed.

---

#### `pto.vcvt`

- **syntax:** `%result = pto.vcvt %input, %mask {rnd = "RND", sat = "SAT", part = "PART"} : !pto.vreg<NxT0>, !pto.mask<G> -> !pto.vreg<MxT1>`
- **semantics:** Type conversion between float/int types with rounding control.

```c
for (int i = 0; i < min(N, M); i++)
    if (mask[i])
        dst[i] = convert(src[i], T0, T1, rnd);
```

- **inputs:**
  `%input` is the source vector, `%mask` selects active lanes, and attributes
  select rounding, saturation, and output placement when the conversion changes
  width or packs into sub-lane positions.
- **outputs:**
  `%result` is the converted vector.
- **constraints and limitations:**
  Only documented source/destination type pairs are legal. All three
  attributes are optional at the surface level, but only the subset meaningful
  to the selected conversion kind should be provided. The execution mask must
  use the typed-mask granularity that matches the source vector family on the
  current surface; there is no `!pto.mask<b64>` form in VPTO.

---

##### Rounding Modes

| Mode | Description |
|------|-------------|
| `R` | Round to nearest, ties to even (default) |
| `A` | Round away from zero |
| `F` | Round toward negative infinity (floor) |
| `C` | Round toward positive infinity (ceil) |
| `Z` | Round toward zero (truncate) |
| `O` | Round to odd |

---

##### Saturation Modes

| Mode | Description |
|------|-------------|
| `SAT` | Saturate on overflow |
| `NOSAT` | No saturation (wrap/undefined on overflow) |

---

##### Part Modes

Use `part` when a width-changing conversion writes only one half of each wider
destination lane group.

- `Part` (`PART_EVEN`, `PART_ODD`)
  - Used by ordinary width-changing conversions.
  - Typical cases include `32 -> 16`, `16 -> 32`, and other even/odd packing
    or unpacking forms.
- `Part_T` (`PART_P0`, `PART_P1`, `PART_P2`, `PART_P3`)
  - Used by lower-level packed placement forms.
  - Typical cases include `32 -> 8`, packed fp8/fp4 conversion paths, and
    other flows where the result is written into one of four sub-parts before a
    later merge or compact step.

| Mode | Description |
|------|-------------|
| `EVEN` | Output to even-indexed lanes |
| `ODD` | Output to odd-indexed lanes |
| `P0` | Output to sub-part 0 in 4-way packed placement forms |
| `P1` | Output to sub-part 1 in 4-way packed placement forms |
| `P2` | Output to sub-part 2 in 4-way packed placement forms |
| `P3` | Output to sub-part 3 in 4-way packed placement forms |

---

##### Attribute Guidance

- `rnd`
  - Use when the conversion needs an explicit rounding rule, especially for
    float-to-int, float-to-float narrowing, or integer-to-float forms that do
    not map exactly.
- `mask`
  - Use to select which source lanes participate in the conversion. In
    width-changing conversions, `mask` works together with `part` / `pp` to
    determine which logical lane positions are produced.
- `sat`
  - Use when the conversion may overflow the destination range and hardware
    exposes a saturating form.
- `part`
  - Use for width-changing conversions that select the even or odd half of the
    destination packing layout.

###### Float To Int

- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<32xsi64>`
- `%dst = pto.vcvt %src, %mask {rnd, sat} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xsi32>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {rnd, part} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xsi32>`
- `%dst = pto.vcvt %src, %mask {rnd, sat} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<256xsi8>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<64xsi32>`

###### Float To Float

- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xbf16>`
- `%dst = pto.vcvt %src, %mask {rnd, sat} : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<128xf16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<64xf32>`

###### Int To Float

- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xui8>, !pto.mask<b8> -> !pto.vreg<128xf16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xsi8>, !pto.mask<b8> -> !pto.vreg<128xf16>`
- `%dst = pto.vcvt %src, %mask {rnd} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<128xf16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<64xf32>`
- `%dst = pto.vcvt %src, %mask {rnd} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<64xf32>`

###### Int To Int

- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xui8>, !pto.mask<b8> -> !pto.vreg<128xui16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xui8>, !pto.mask<b8> -> !pto.vreg<64xui32>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xsi8>, !pto.mask<b8> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xsi8>, !pto.mask<b8> -> !pto.vreg<64xsi32>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<128xui16>, !pto.mask<b16> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xui16>, !pto.mask<b16> -> !pto.vreg<64xui32>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<64xui32>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<64xsi32>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xui32>, !pto.mask<b32> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xui32>, !pto.mask<b32> -> !pto.vreg<128xui16>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xui32>, !pto.mask<b32> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<128xui16>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<32xsi64>`

##### A5 Supported Type Matrix

The table below is only a summary. For exact attribute combinations, use the
per-form entries above as the source of truth.

| `src \ dst` | `ui8` | `si8` | `ui16` | `si16` | `ui32` | `si32` | `si64` | `f16` | `f32` | `bf16` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ui8` |  |  | Y |  | Y |  |  | Y |  |  |
| `si8` |  |  |  | Y |  | Y |  | Y |  |  |
| `ui16` | Y |  |  |  | Y |  |  |  |  |  |
| `si16` | Y |  |  |  | Y | Y |  | Y | Y |  |
| `ui32` | Y |  | Y | Y |  |  |  |  |  |  |
| `si32` | Y |  | Y | Y |  |  | Y |  | Y |  |
| `si64` |  |  |  |  |  |  |  |  |  |  |
| `f16` | Y | Y |  | Y |  | Y |  |  | Y |  |
| `f32` |  |  |  | Y |  | Y | Y | Y |  | Y |
| `bf16` |  |  |  |  |  | Y |  | Y | Y |  |

---

##### Width-Changing Conversion Pattern

For conversions that change width (e.g., f32→f16), use even/odd parts and combine:

```mlir
// Convert two f32 vectors to one f16 vector
%even = pto.vcvt %in0, %mask {rnd = "R", sat = "SAT", part = "EVEN"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%odd  = pto.vcvt %in1, %mask {rnd = "R", sat = "SAT", part = "ODD"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%result = pto.vor %even, %odd, %mask : !pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<128xf16>
```

---

#### `pto.vtrc`

- **syntax:** `%result = pto.vtrc %input, %mask, "RND" : !pto.vreg<NxT>, !pto.mask<BW> -> !pto.vreg<NxT>`
- **semantics:** Truncate/round float to integer-valued float (stays in float type).

```c
for (int i = 0; i < N; i++)
    dst[i] = round_to_int_valued_float(src[i], rnd);
```

- **inputs:**
  `%input` is the floating-point source vector, `%mask` selects active lanes,
  and `RND` selects the truncation/rounding rule.
- **outputs:**
  `%result` is still a floating-point vector, but each active lane now carries
  an integer-valued floating-point result.
- **constraints and limitations:**
  This op does not change the element type. `T` must be `f16`, `f32`, or
  `bf16`. `RND` must be one of `R`, `A`, `F`, `C`, or `Z`. `BW` must match the
  element width: `b16` for `f16`/`bf16`, `b32` for `f32`.

**Example:**
```mlir
// Round to nearest integer, keep as float
%rounded = pto.vtrc %input, %mask, "R" : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
// input:  [1.4, 2.6, -1.5, 3.0]
// output: [1.0, 3.0, -2.0, 3.0]
```

---

#### Typical Usage

```mlir
// Quantization: f32 → i8 with saturation
%scaled = pto.vmuls %input, %scale, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%quantized = pto.vcvt %scaled, %mask {rnd = "R", sat = "SAT"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xi32>
// Then narrow i32 → i8 via pack ops

// Mixed precision: bf16 → f32 for accumulation
%f32_vec = pto.vcvt %bf16_input, %mask {part = "EVEN"}
    : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<64xf32>

// Floor for integer division
%floored = pto.vtrc %ratio, %mask, "F" : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%int_div = pto.vcvt %floored, %mask {rnd = "Z"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xi32>
```

---

#### `pto.vbitcast`

- **syntax:** `%result = pto.vbitcast %input : !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- **semantics:** Bitwise reinterpretation of a vreg vector without changing the underlying bit pattern. This operation performs a pure type cast that preserves the exact bits of each element, changing only their interpretation (e.g., from floating-point to integer).

- **inputs:**
  `%input` is the source vector register value.
- **outputs:**
  `%result` is the reinterpreted vector register value.
- **constraints and limitations:**
  1. Both source and result must be `!pto.vreg<...>` types.
  2. Source and result vectors must have the same total bit width (currently 2048 bits).
  3. Only integer and floating-point element types are supported.

**Element bit-width equality examples:**
- `f32<64>` → `i32<64>`  (both 32-bit elements, total 2048 bits)
- `f16<128>` → `i16<128>` (both 16-bit elements, total 2048 bits)
- `bf16<128>` → `ui16<128>` (both 16-bit elements, total 2048 bits)
- `si32<64>` → `ui32<64>` (both 32-bit elements, total 2048 bits)
- `f32<64>` → `i16<128>` (32-bit/16-bit elements, total 2048 bits)

**Verification:** The operation verifies that:
1. Both input and result are `!pto.vreg<...>` types.
2. Total bit width equals 2048 (the fixed vreg size).

**Comparison with `pto.vcvt`:**
- `pto.vcvt` performs value conversion with rounding, saturation, and lane placement control.
- `pto.vbitcast` performs bitwise reinterpretation without changing the underlying bit pattern.

**Example: Reinterpreting float as integer for bit manipulation**
```mlir
// Prepare a vector of float values
%fvec = pto.vlds %ub[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// Reinterpret as integer for bitwise operations
%ivec = pto.vbitcast %fvec : !pto.vreg<64xf32> -> !pto.vreg<64xi32>

// Extract sign bit (bit 31)
%sign_bits = pto.vand %ivec, %sign_mask, %mask : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>

// Reinterpret back to float
%fvec_without_sign = pto.vbitcast %sign_bits : !pto.vreg<64xi32> -> !pto.vreg<64xf32>
```

**Example: Type punning between signed and unsigned integer**
```mlir
// Convert signed to unsigned without changing bits
%signed = pto.vlds %ub[%lane] : !pto.ptr<si32, ub> -> !pto.vreg<64xsi32>
%unsigned = pto.vbitcast %signed : !pto.vreg<64xsi32> -> !pto.vreg<64xui32>
// Bits are identical; interpretation changes from signed to unsigned
```

#### `pto.pbitcast`

- **syntax:** `%result = pto.pbitcast %input : !pto.mask<G0> -> !pto.mask<G1>`
- **semantics:** Bitwise reinterpretation of a predicate register without
  changing the underlying predicate-register image. This op makes mask-family
  reinterpretation explicit in VPTO IR when a producer and consumer expect
  different `!pto.mask<...>` views of the same hardware predicate state.

- **inputs:**
  `%input` is the source predicate register value.
- **outputs:**
  `%result` is the reinterpreted predicate register value.
- **constraints and limitations:**
  1. Both source and result must be `!pto.mask<...>` types.
  2. `pto.pbitcast` does not materialize or normalize predicate contents; it
     only changes which mask granularity the surrounding VPTO IR uses to
     interpret the same predicate bits.

**Example: Reinterpret a b16 predicate as b32 before a consumer**
```mlir
%m16 = pto.pintlv_b16 %lhs, %rhs : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>
%m32 = pto.pbitcast %m16#0 : !pto.mask<b16> -> !pto.mask<b32>
%result = pto.vsel %a, %b, %m32 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

<a id="micro-10-reduction-ops"></a>

### 10. Reduction Ops

> **Category:** Vector reduction operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that reduce a vector to a scalar or per-group result.

#### Common Operand Model

- `%input` is the source vector register value.
- `%mask` is the predicate operand `Pg`; inactive lanes do not participate.
- `%result` is the destination vector register value.
- Reduction results are written into the low-significance portion of the
  destination vector and the remaining destination bits are zero-filled.

---

#### Full Vector Reductions

##### `pto.vcadd`

- **syntax:** `%result = pto.vcadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<MxU>`
- **A5 types:** i8-i32, f16, f32
- **semantics:** Sum all elements. Result in lane 0, others zeroed.

```c
T sum = 0;
for (int i = 0; i < N; i++)
    sum += src[i];
dst[0] = sum;
for (int i = 1; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains the reduction result in its low element(s).
- **constraints and limitations:** On A5, `i8/u8` inputs produce widened
  `i16/u16` results with half as many lanes (`M = N / 2`), and `i16/u16` inputs
  produce widened `i32/u32` results with half as many lanes. For
  `i32/u32/f16/f32` inputs, `U = T` and `M = N`. If all predicate bits are
  zero, the result is zero.

---

##### `pto.vcmax`

- **syntax:** `%result = pto.vcmax %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, f32
- **semantics:** Find max element with argmax. The lowest destination element
  stores the maximum value, the second-lowest destination element stores the
  index of the first maximum, and all remaining elements are zero-filled.

```c
T mx = -INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] > mx) { mx = src[i]; idx = i; }
dst[0] = mx;
dst[1] = idx;
for (int i = 2; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result[0]` holds the extremum value and `%result[1]` holds the
  index. Other destination elements are zero-filled.
- **constraints and limitations:** If there are multiple maxima, the minimum
  index is written. For floating-point types, inactive lanes are treated as
  `-INF`; if all lanes are inactive, `%result[0]` becomes `-INF`. For integer
  types, inactive lanes are treated as the literal minimum value; if all lanes
  are inactive, `%result[0]` becomes that literal minimum value. The index is
  written into the second destination element slot of the same destination
  vector register.

---

##### `pto.vcmin`

- **syntax:** `%result = pto.vcmin %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, f32
- **semantics:** Find min element with argmin. The lowest destination element
  stores the minimum value, the second-lowest destination element stores the
  index of the first minimum, and all remaining elements are zero-filled.

```c
T mn = INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] < mn) { mn = src[i]; idx = i; }
dst[0] = mn;
dst[1] = idx;
for (int i = 2; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result[0]` holds the extremum value and `%result[1]` holds the
  index. Other destination elements are zero-filled.
- **constraints and limitations:** If there are multiple minima, the minimum
  index is written. For floating-point types, inactive lanes are treated as
  `+INF`; if all lanes are inactive, `%result[0]` becomes `+INF`. For integer
  types, inactive lanes are treated as the literal maximum value; if all lanes
  are inactive, `%result[0]` becomes that literal maximum value. The index is
  written into the second destination element slot of the same destination
  vector register.

---

##### `pto.vcbmax`

- **syntax:** `%value, %predicate = pto.vcbmax %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
- **A5 types:** i8-i32, f16, f32
- **semantics:** Find the maximum value and produce a predicate marking every
  participating lane whose value matches that maximum.

```c
T mx = max_active(src, mask);
for (int i = 0; i < N; i++) {
    value[i] = (i == 0) ? mx : 0;
    predicate[i] = mask[i] && matches_max(src[i], mx);
}
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%value[0]` holds the maximum and remaining value elements are
  zero-filled. `%predicate` marks all active lanes matching the maximum.
- **constraints and limitations:** If all lanes are inactive, `%predicate` is
  all zero. Floating-point inactive lanes are treated as `-INF`; integer
  inactive lanes are treated as the literal minimum value. For floating-point
  `+0/-0`, the value result follows the target maximum rule while predicate
  matching marks both zero signs as matching locations.

---

##### `pto.vcbmin`

- **syntax:** `%value, %predicate = pto.vcbmin %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
- **A5 types:** i8-i32, f16, f32
- **semantics:** Find the minimum value and produce a predicate marking every
  participating lane whose value matches that minimum.

```c
T mn = min_active(src, mask);
for (int i = 0; i < N; i++) {
    value[i] = (i == 0) ? mn : 0;
    predicate[i] = mask[i] && matches_min(src[i], mn);
}
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%value[0]` holds the minimum and remaining value elements are
  zero-filled. `%predicate` marks all active lanes matching the minimum.
- **constraints and limitations:** If all lanes are inactive, `%predicate` is
  all zero. Floating-point inactive lanes are treated as `+INF`; integer
  inactive lanes are treated as the literal maximum value. Floating-point NaN
  handling follows the target instruction semantics.

---

#### Histogram Reductions

##### `pto.chistv2`

- **syntax:** `%result = pto.chistv2 %acc, %source, %mask, %bin : !pto.vreg<128xui16>, !pto.vreg<256xui8>, !pto.mask<b8>, i32 -> !pto.vreg<128xui16>`
- **semantics:** Cumulative histogram update over unsigned 8-bit source lanes.
  `%acc` provides the incoming 16-bit bin accumulators and `%result` contains
  the updated accumulators.
- **inputs:** `%source` provides 256 unsigned 8-bit samples, `%mask` selects
  active source lanes, and `%bin` is the target bin/control operand passed to
  the A5 histogram instruction.
- **constraints and limitations:** `%acc` and `%result` are fixed to
  `!pto.vreg<128xui16>`, `%source` is fixed to `!pto.vreg<256xui8>`, and the
  mask granularity is fixed to `b8`.

---

##### `pto.dhistv2`

- **syntax:** `%result = pto.dhistv2 %acc, %source, %mask, %bin : !pto.vreg<128xui16>, !pto.vreg<256xui8>, !pto.mask<b8>, i32 -> !pto.vreg<128xui16>`
- **semantics:** Distribution histogram update over unsigned 8-bit source
  lanes. `%acc` provides the incoming 16-bit bin accumulators and `%result`
  contains the updated accumulators.
- **inputs:** `%source` provides 256 unsigned 8-bit samples, `%mask` selects
  active source lanes, and `%bin` is the target bin/control operand passed to
  the A5 histogram instruction.
- **constraints and limitations:** `%acc` and `%result` are fixed to
  `!pto.vreg<128xui16>`, `%source` is fixed to `!pto.vreg<256xui8>`, and the
  mask granularity is fixed to `b8`.

---

#### Per-VLane (Group) Reductions

The vector register is organized as **8 VLanes** of 32 bytes each. Group
reductions operate within each VLane independently and produce one result per
VLane. The 8 VLane results are written contiguously to the low elements of the
destination vector; all remaining destination elements are zero.

```
vreg layout (f32 example, 64 elements total):
VLane 0: [0..7]   VLane 1: [8..15]  VLane 2: [16..23] VLane 3: [24..31]
VLane 4: [32..39] VLane 5: [40..47] VLane 6: [48..55] VLane 7: [56..63]
```

##### `pto.vcgadd`

- **syntax:** `%result = pto.vcgadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, f32
- **semantics:** Sum active elements within each 32-byte VLane. The 8 VLane
  sums are written to result elements `0..7`; all other result elements are
  zero.

```c
int groups = 8;
int K = 32 / sizeof(T);  // elements per 32-byte VLane
for (int g = 0; g < 8; g++) {
    T sum = 0;
    for (int i = 0; i < K; i++)
        if (mask[g*K + i])
            sum += src[g*K + i];
    dst[g] = sum;
}
for (int i = groups; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one sum per 32-byte VLane group, written
  contiguously to the low elements of the result vector.
- **constraints and limitations:** This is a per-32-byte VLane-group reduction.
  Inactive lanes are treated as zero. If all lanes in a VLane are inactive, the
  corresponding result element is `0` (`+0` for floating-point types).

---

##### `pto.vcgmax`

- **syntax:** `%result = pto.vcgmax %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, f32
- **semantics:** Find the maximum active element within each 32-byte VLane. The
  8 VLane maxima are written to result elements `0..7`; all other result
  elements are zero.

```c
int groups = 8;
int K = 32 / sizeof(T);
for (int g = 0; g < 8; g++) {
    T mx = max_identity_for_T;  // -INF for float, minimum value for integer
    for (int i = 0; i < K; i++)
        if (mask[g*K + i])
            mx = max(mx, src[g*K + i]);
    dst[g] = mx;
}
for (int i = groups; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one maximum per 32-byte VLane group, written
  contiguously to the low elements of the result vector.
- **constraints and limitations:** Grouping is by hardware 32-byte VLane, not by
  arbitrary software subvector. Inactive floating-point lanes are treated as
  `-INF`; inactive integer lanes are treated as the element type's minimum
  value. If all lanes in a VLane are inactive, that neutral value is written for
  the corresponding VLane result. For floating-point values, `max(+0, -0)`
  returns `+0`.

---

##### `pto.vcgmin`

- **syntax:** `%result = pto.vcgmin %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, f32
- **semantics:** Find the minimum active element within each 32-byte VLane. The
  8 VLane minima are written to result elements `0..7`; all other result
  elements are zero.

```c
int groups = 8;
int K = 32 / sizeof(T);
for (int g = 0; g < 8; g++) {
    T mn = min_identity_for_T;  // +INF for float, maximum value for integer
    for (int i = 0; i < K; i++)
        if (mask[g*K + i])
            mn = min(mn, src[g*K + i]);
    dst[g] = mn;
}
for (int i = groups; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one minimum per 32-byte VLane group, written
  contiguously to the low elements of the result vector.
- **constraints and limitations:** Grouping is by hardware 32-byte VLane, not by
  arbitrary software subvector. Inactive floating-point lanes are treated as
  `+INF`; inactive integer lanes are treated as the element type's maximum
  value. If all lanes in a VLane are inactive, that neutral value is written for
  the corresponding VLane result. For floating-point values, `min(-0, +0)`
  returns `-0`.

---

#### Prefix Operations

##### `pto.vcpadd`

- **syntax:** `%result = pto.vcpadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Inclusive prefix sum (scan).

```c
dst[0] = src[0];
for (int i = 1; i < N; i++)
    dst[i] = dst[i-1] + src[i];
```

**Example:**
```c
// input:  [1, 2, 3, 4, 5, ...]
// output: [1, 3, 6, 10, 15, ...]
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` is the inclusive prefix-sum vector.
- **constraints and limitations:** Only floating-point element types are
  documented on the current A5 surface here.

---

#### Typical Usage

```mlir
// Softmax: find max for numerical stability
%max_vec = pto.vcmax %logits, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
// max is in lane 0, broadcast it
%max_broadcast = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// Per-VLane sums using vcgadd
%row_sums = pto.vcgadd %tile, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
// Results at indices 0..7; remaining elements are zero

// Full vector sum for normalization
%total = pto.vcadd %values, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
// total[0] contains the sum

// Prefix sum for cumulative distribution
%cdf = pto.vcpadd %pdf, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
```

<a id="micro-11-compare-select"></a>

### 11. Compare & Select

> **Category:** Comparison and conditional selection operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that compare vectors and conditionally select elements.

#### Common Operand Model

- `%src0` and `%src1` are source vector operands.
- `%scalar` is the scalar operand for scalar-comparison families.
- `%seed` is the incoming predicate that limits which lanes participate in the
  compare.
- `%result` is either a predicate mask (`vcmp`, `vcmps`) or a vector register
  (`vsel`, `vselr`, `vselrv2`).

---

#### Comparison Operations

##### `pto.vcmp`

- **syntax:** `%result = pto.vcmp %src0, %src1, %seed, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Element-wise comparison, output predicate mask.

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src0[i] CMP src1[i]) ? 1 : 0;
```

**Compare modes:**

| Mode | Operation |
|------|-----------|
| `eq` | Equal (==) |
| `ne` | Not equal (!=) |
| `lt` | Less than (<) |
| `le` | Less than or equal (<=) |
| `gt` | Greater than (>) |
| `ge` | Greater than or equal (>=) |

**Example:**
```mlir
%all_active = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
%lt_mask = pto.vcmp %a, %b, %all_active, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>
// lt_mask[i] = 1 if a[i] < b[i]
```

- **inputs:** `%src0`, `%src1`, and `%seed`; `CMP_MODE` selects the comparison
  predicate.
- **outputs:** `%result` is the generated predicate mask.
- **constraints and limitations:** Only lanes enabled by `%seed` participate.
  Integer and floating-point comparisons follow their own element-type-specific
  comparison rules. `%seed` and `%result` keep the typed-mask granularity that
  matches `%src0` / `%src1`.

---

##### `pto.vcmps`

- **syntax:** `%result = pto.vcmps %src, %scalar, %seed, "CMP_MODE" : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Compare vector against scalar.

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src[i] CMP scalar) ? 1 : 0;
```

**Example:**
```mlir
%positive_mask = pto.vcmps %values, %c0_f32, %all_active, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>
// positive_mask[i] = 1 if values[i] > 0
```

- **inputs:** `%src` is the vector source, `%scalar` is the scalar comparison
  value, and `%seed` is the incoming predicate.
- **outputs:** `%result` is the generated predicate mask.
- **constraints and limitations:** For 32-bit scalar forms, the scalar source
  MUST satisfy the backend's legal scalar-source constraints for this family.
  `%seed` and `%result` keep the typed-mask granularity that matches `%src`.

---

#### Selection Operations

##### `pto.vsel`

- **syntax:** `%result = pto.vsel %src0, %src1, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Per-lane select based on mask.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src0[i] : src1[i];
```

**Example — Conditional assignment:**
```mlir
// dst = mask ? true_vals : false_vals
%result = pto.vsel %true_vals, %false_vals, %condition
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

- **inputs:** `%src0` is the true-path vector, `%src1` is the false-path vector,
  and `%mask` selects between them.
- **outputs:** `%result` is the selected vector.
- **constraints and limitations:** Source vectors and result MUST have matching
  vector shapes and element types. `%mask` keeps the typed-mask granularity
  that matches the selected vector family.

---

##### `pto.vselr`

- **syntax:** `%result = pto.vselr %src, %idx : !pto.vreg<NxT>, !pto.vreg<Nxi<width>> -> !pto.vreg<NxT>`
- **semantics:** Lane-select by index vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = src[idx[i]];
```

- **inputs:** `%src` is the source vector. `%idx` is the lane-index vector.
- **outputs:** `%result` is the reordered vector.
- **constraints and limitations:** `%idx` must use integer elements. `%idx`
  must have the same lane count as `%src`, and its integer element width must
  match the bit width of `%src` element type.

---

##### `pto.vselrv2`

- **syntax:** `%result = pto.vselrv2 %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **semantics:** Variant select form with the same current two-vector operand shape.
- **inputs:** `%src0` and `%src1` are the source vectors.
- **outputs:** `%result` is the selected vector.
- **constraints and limitations:** This page records the surface shape only.
  Lowering MUST preserve the exact A5 variant semantics selected for this form.

---

#### Typical Usage

```mlir
// Clamp negative values to zero (manual ReLU)
%all = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
%zero = pto.vbr %c0_f32 : f32 -> !pto.vreg<64xf32>
%neg_mask = pto.vcmps %input, %c0_f32, %all, "lt" : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>
%clamped = pto.vsel %zero, %input, %neg_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Element-wise max via compare+select
%gt_mask = pto.vcmp %a, %b, %all, "gt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>
%max_ab = pto.vsel %a, %b, %gt_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Threshold filter
%above_thresh = pto.vcmps %scores, %threshold, %all, "ge" : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>
%filtered = pto.vsel %scores, %zero, %above_thresh : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

---

#### Compare + Select Pattern

```mlir
// Softmax safe exp: exp(x - max) where x < max returns exp of negative
// but we want to clamp to avoid underflow

%all = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>

// 1. Compare against threshold
%too_small = pto.vcmps %x_minus_max, %min_exp_arg, %all, "lt"
    : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>

// 2. Clamp values below threshold
%clamped = pto.vsel %min_exp_arg_vec, %x_minus_max, %too_small
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// 3. Safe exp
%exp_result = pto.vexp %clamped, %all : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

<a id="micro-12-data-rearrangement"></a>

### 12. Data Rearrangement

> **Category:** In-register data movement and permutation
> **Pipeline:** PIPE_V (Vector Core)

Operations that rearrange data within or between vector registers without memory access.

#### Common Operand Model

- `%lhs` / `%rhs` are source vector register values.
- `%src` is a single source vector register value.
- `%result` is the destination vector register value unless an op explicitly
  returns multiple vectors.
- These families do not access UB directly; they only rearrange register
  contents.

---

#### Interleave / Deinterleave

##### `pto.vintlv`

- **syntax:** `%low, %high = pto.vintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Interleave elements from two sources.

```c
// Interleave: merge even/odd elements from two sources
// low  = {src0[0], src1[0], src0[1], src1[1], ...}
// high = {src0[N/2], src1[N/2], src0[N/2+1], src1[N/2+1], ...}
```

- **inputs:** `%lhs` and `%rhs` are the two source vectors.
- **outputs:** `%low` and `%high` are the two destination vectors.
- **constraints and limitations:** The two outputs form a paired interleave
  result. The PTO micro Instruction representation exposes that pair as two SSA results, and the pair ordering MUST
  be preserved.

---

##### `pto.vdintlv`

- **syntax:** `%low, %high = pto.vdintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Deinterleave elements into even/odd.

```c
// Deinterleave: separate even/odd elements
// low  = {src0[0], src0[2], src0[4], ...}  // even
// high = {src0[1], src0[3], src0[5], ...}  // odd
```

- **inputs:** `%lhs` and `%rhs` represent the interleaved source stream in the
  current PTO micro Instruction representation.
- **outputs:** `%low` and `%high` are the separated destination vectors.
- **constraints and limitations:** The two outputs form the even/odd
  deinterleave result pair, and their ordering MUST be preserved.

---

#### Compress / Expand

##### `pto.vsqz`

- **syntax:** `%result = pto.vsqz %src, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Compress — pack active lanes to front.

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[j++] = src[i];
while (j < N) dst[j++] = 0;
```

**Use case:** Sparse data compaction, filtering.

- **inputs:** `%src` is the source vector and `%mask` selects which elements are
  kept.
- **outputs:** `%result` is the compacted vector.
- **constraints and limitations:** This is a reduction-style compaction family.
  Preserved element order MUST match source lane order.

---

##### `pto.vusqz`

- **syntax:** `%result = pto.vusqz %src, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Generate per-lane prefix counts from the governing predicate.

```c
dst[0] = 0;
for (int i = 1; i < N; i++)
    dst[i] = mask[i - 1] ? (dst[i - 1] + 1) : dst[i - 1];
```

- **inputs:** `%mask` is the governing predicate. The current PTO surface keeps
  `%src` in the operand list for interface compatibility, but the observable
  result semantics are determined by `%mask`.
- **outputs:** `%result[i]` equals the number of active lanes in `%mask[0:i)`,
  with `%result[0] = 0`.
- **constraints and limitations:** `T` is currently limited to `si8`, `si16`,
  or `si32`. This operation is a predicate-derived counting/rearrangement
  primitive rather than a value-placement primitive. The final predicate lane
  does not contribute to a later output lane because there is no `dst[N]`.

---

---

##### `pto.vselr`

- **syntax:** `%result = pto.vselr %src, %idx : !pto.vreg<NxT>, !pto.vreg<Nxi<width>> -> !pto.vreg<NxT>`
- **semantics:** Register lane-select with an explicit index vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = src[idx[i]];
```

- **inputs:** `%src` is the source vector. `%idx` is the lane-index vector.
- **outputs:** `%result` is the reordered vector.
- **constraints and limitations:** This page records the rearrangement use of
  the family; the compare/select page documents the same name from the predicate
  selection perspective.

---

#### Pack / Unpack

##### `pto.vpack`

- **syntax:** `%result = pto.vpack %src, "PART" : !pto.vreg<NxT_wide> -> !pto.vreg<2NxT_narrow>`
- **semantics:** Narrow one wide vector and place the narrowed payload into the
  selected half of the result. The other half is filled with zero.

```c
// e.g., vreg<64xi32> → vreg<128xui16>
for (int i = 0; i < N; i++)
    dst[i] = 0;

if (part == LOWER) {
    for (int i = 0; i < N; i++)
        dst[i] = truncate(src[i]);
} else { // HIGHER
    for (int i = 0; i < N; i++)
        dst[N + i] = truncate(src[i]);
}
```

- **inputs:** `%src` is the wide source vector. `"LOWER"` and `"HIGHER"`
  select whether the narrowed payload lands in the lower or upper half.
- **outputs:** `%result` is the packed narrow vector.
- **constraints and limitations:** Packing is a narrowing conversion with
  truncation semantics. Current VPTO surface supports `i32/ui32 -> ui16` and
  `i16/ui16 -> ui8`.

---

##### `pto.vsunpack`

- **syntax:** `%result = pto.vsunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>`
- **semantics:** Sign-extending unpack — narrow to wide (half).

```c
// e.g., vreg<128xi16> → vreg<64xi32> (one half)
for (int i = 0; i < N/2; i++)
    dst[i] = sign_extend(src[part_offset + i]);
```

- **inputs:** `%src` is the packed narrow vector and `%part` selects which half
  is unpacked.
- **outputs:** `%result` is the widened vector.
- **constraints and limitations:** This is the sign-extending unpack family.

---

##### `pto.vzunpack`

- **syntax:** `%result = pto.vzunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>`
- **semantics:** Zero-extending unpack — narrow to wide (half).

```c
for (int i = 0; i < N/2; i++)
    dst[i] = zero_extend(src[part_offset + i]);
```

- **inputs:** `%src` is the packed narrow vector and `%part` selects which half
  is unpacked.
- **outputs:** `%result` is the widened vector.
- **constraints and limitations:** This is the zero-extending unpack family.

---

#### Typical Usage

```mlir
// AoS → SoA conversion using deinterleave
%even, %odd = pto.vdintlv %interleaved0, %interleaved1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

// Filter: keep only elements passing condition
%pass_mask = pto.vcmps %values, %threshold, %all, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.mask<G>
%compacted = pto.vsqz %values, %pass_mask
    : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>

// Type narrowing via pack
%packed_i16 = pto.vpack %wide_i32, "LOWER"
  : !pto.vreg<64xi32> -> !pto.vreg<128xui16>
```

---

#### V2 Interleave Forms

##### `pto.vintlvv2`

- **syntax:** `%result = pto.vintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **inputs:** `%lhs` and `%rhs` are source vectors and `PART` selects the
  returned half of the V2 interleave result.
- **outputs:** `%result` is the selected interleave half.
- **constraints and limitations:** This op exposes only one half of the V2
  result in SSA form.

##### `pto.vdintlvv2`

- **syntax:** `%result = pto.vdintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **inputs:** `%lhs` and `%rhs` are source vectors and `PART` selects the
  returned half of the V2 deinterleave result.
- **outputs:** `%result` is the selected deinterleave half.
- **constraints and limitations:** This op exposes only one half of the V2
  result in SSA form.

<a id="micro-13-dsa-sfu-ops"></a>

### 13. DSA/SFU Ops

> **Category:** Domain-specific accelerator and special function unit operations
> **Pipeline:** PIPE_V (Vector Core) / SFU

Fused operations, special functions, and UB-to-UB operations that leverage hardware acceleration.

#### Common Operand Model

- `%input`, `%lhs`, `%rhs`, `%acc`, and `%alpha` are source SSA values whose
  roles are called out per instruction.
- `%mask` is the predicate operand `Pg` when present.
- `%result` is the destination SSA value.
- This page mixes three different backend shapes: pure `vreg -> vreg` ops,
  conversion/fusion ops, and UB-to-UB helpers. Each instruction section calls
  out which storage model it uses.

---

#### Fused Activation Ops (vreg→vreg)

##### `pto.vlrelu`

- **syntax:** `%result = pto.vlrelu %input, %alpha, %mask : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Leaky ReLU with scalar alpha.

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha * src[i];
```

- **inputs:** `%input` is the activation vector, `%alpha` is the scalar slope,
  and `%mask` selects active lanes.
- **outputs:** `%result` is the leaky-ReLU vector.
- **constraints and limitations:** Only `f16` and `f32` forms are currently
  documented for `pto.vlrelu`.

---

##### `pto.vprelu`

- **syntax:** `%result = pto.vprelu %input, %alpha, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<bW> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Parametric ReLU with per-element alpha vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha[i] * src[i];
```

- **inputs:** `%input` is the activation vector, `%alpha` is the per-element
  slope vector, and `%mask` selects active lanes.
- **outputs:** `%result` is the parametric-ReLU vector.
- **constraints and limitations:** Floating-point element types only on the
  current A5 surface.

---

##### `pto.vexpdif`

- **syntax:** `%result = pto.vexpdif %input, %max, %mask, "EVEN|ODD" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<bW> -> !pto.vreg<Mxf32>`
- **A5 types:** input `f16` or `f32`, output `f32`
- **semantics:** Fused exp(x - max) for numerically stable softmax.

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i] - max[i]);
```

**Use case:** Softmax numerator computation with numerical stability.

- **inputs:** `%input` is the source vector, `%max` is the broadcasted
  subtraction term, `%mask` selects active source lanes, and `%part` selects
  `EVEN` or `ODD` for the underlying hardware contract.
- **outputs:** `%result` is the fused `exp(input - max)` vector with `f32`
  elements.
- **constraints and limitations:** Source vectors must be `f16` or `f32`, the
  result vector must be `f32`, the mask granularity must match the input
  vector element width, and source/result storage width must match.

---

#### Fused Compute+Convert Ops

##### `pto.vaxpy`

- **syntax:** `%result = pto.vaxpy %src0, %src1, %alpha, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** AXPY — scalar-vector multiply-add.

```c
for (int i = 0; i < N; i++)
    dst[i] = alpha * src0[i] + src1[i];
```

- **inputs:** `%src0` is the scaled vector, `%src1` is the addend vector,
  `%alpha` is the scalar multiplier, and `%mask` selects active lanes.
- **outputs:** `%result` is the fused AXPY result.
- **constraints and limitations:** Floating-point element types only on the
  current documented surface.

---

##### `pto.vmulscvt`

- **syntax:** `%result = pto.vmulscvt %input, %scalar, %mask, %rnd, %part : !pto.vreg<NxT0>, T0, !pto.mask<G> -> !pto.vreg<MxT1>`
- **A5 types:** input `f32`, output `f16` (primary documented pair)
- **semantics:** Fused multiply-by-scalar and type conversion. Each active lane
  is multiplied by `%scalar` and then converted from the source element type to
  the destination element type in a single hardware step using the authored
  round mode.

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = convert_type<T1>(src[i] * scalar, rnd);
```

**Use case:** Softmax scale-and-downcast: apply the reciprocal scale factor and
narrow from `f32` (accumulator precision) to `f16` (storage precision) before a
block-strided store.

- **inputs:** `%input` is the source vector (wider type), `%scalar` is the
  uniform scale factor, `%mask` selects active lanes, `%rnd` selects the cast
  round mode, and `%part` selects `EVEN` or `ODD` for half-width output
  placement.
- **outputs:** `%result` is the scaled and converted vector with the narrower
  destination element type.
- **constraints and limitations:** The source/destination type pair must be a
  legal hardware narrowing conversion (e.g., `f32 -> f16`). Illegal pairs are
  rejected. `%scalar` must match the source element type. The mask granularity
  must match the source vector element width.

**Example** — softmax scale and downcast:
```mlir
// Apply scale=1.0 and narrow f32 -> f16, writing into even half of the
// destination packing layout
%f16_even = pto.vmulscvt %f32_exp, %one, %mask, "A", "EVEN"
    : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<128xf16>
%f16_odd  = pto.vmulscvt %f32_exp2, %one, %mask, "A", "ODD"
    : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<128xf16>
```

---

#### Extended Arithmetic

##### `pto.vmull`

- **syntax:** `%low, %high = pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **A5 types:** i32/ui32 (native 32×32→64 widening multiply)
- **semantics:** Widening multiply with high/low results.

```c
for (int i = 0; i < 64; i++) {
    int64_t r = (int64_t)src0_i32[i] * (int64_t)src1_i32[i];
    dst_lo[i] = (int32_t)(r & 0xFFFFFFFF);
    dst_hi[i] = (int32_t)(r >> 32);
}
```

- **inputs:** `%lhs` and `%rhs` are the source vectors and `%mask` selects
  active lanes.
- **outputs:** `%low` and `%high` expose the widened-product low/high parts.
- **constraints and limitations:** The current documented A5 form is the native
  widening 32x32->64 integer multiply family.

---

##### `pto.vmula`

- **syntax:** `%result = pto.vmula %acc, %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Multiply-accumulate.

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = acc[i] + lhs[i] * rhs[i];
```

- **inputs:** `%acc` is the accumulator input, `%lhs` and `%rhs` are the
  multiplicands, and `%mask` selects active lanes.
- **outputs:** `%result` is the multiply-accumulate result.
- **constraints and limitations:** `pto.vmula` is a fused multiply-accumulate
  operation and is not always interchangeable with separate `vmul` plus `vadd`.

---

#### Index Generation

##### `pto.vci`

- **syntax:** `%result = pto.vci %index {order = "ASC|DESC"} : T -> !pto.vreg<NxT>`
- **semantics:** Generate a lane index vector from a scalar base value.

```c
for (int i = 0; i < N; i++)
    dst[i] = (order == ASC) ? (base_index + i) : (base_index - i);
```

**Use case:** Generate indices for gather/scatter, argsort, etc.

- **inputs:** `%index` is the scalar base value. Supported scalar types are
  `i8/i16/i32`, `f16`, and `f32`.
- **outputs:** `%result` is the generated index vector.
- **constraints and limitations:** `%result` element type determines both the
  generated element type and the lane count. Supported result types are
  `!pto.vreg<256xsi8>`, `!pto.vreg<128xsi16>`, `!pto.vreg<64xsi32>`,
  `!pto.vreg<128xf16>`, and `!pto.vreg<64xf32>`. `%index` must use the
  matching scalar type for `f16`/`f32`; for integer results, `%index` must use
  the same bit width and may be signless or signed.

---

#### Sorting Operations

##### `pto.vbitsort`

- **syntax:** `pto.vbitsort %dest, %src, %indices, %repeat_times : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, index`
- **semantics:** Sort 32 region proposals by score and materialize sorted
  proposal records into `%dest`.
- **inputs:** `%dest` is the UB destination buffer. `%src` is the UB score
  buffer. `%indices` is the UB index buffer. `%repeat_times` is the repeat
  count; each repeat processes the next adjacent group of 32 scores and 32
  indices.
- **outputs:** This op writes UB memory and returns no SSA value. Each output
  record occupies 8 bytes: the upper 4 bytes hold the index and the lower
  4 bytes hold the score. For `f16` score forms, the score uses the lower
  2 bytes of that 4-byte score field and the upper 2 bytes are reserved.
- **constraints and limitations:** `%dest`, `%src`, and `%indices` MUST be
  UB-backed pointers and SHOULD satisfy the backend alignment contract expected
  by the A5 `VBS32` instruction. Scores are sorted in descending order, so the
  highest score is written to the lowest destination address. Equal-score ties
  preserve the earlier input proposal first. This is a UB helper, not a pure
  `vreg -> vreg` op.

---

##### `pto.vmrgsort4`

- **syntax:** `pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64, i64`
- **semantics:** Merge-sort 4 pre-sorted input vectors.
- **inputs:** `%dest` is the UB destination, `%src0..%src3` are the four
  pre-sorted UB inputs, `%count` is the number of valid elements, and `%config`
  is the operation control word.
- **outputs:** This op writes UB memory and returns no SSA value.
- **constraints and limitations:** Inputs MUST already be sorted according to
  the sort order encoded by `%config`.

---

##### `pto.get_vms4_sr`

- **syntax:** `%list0, %list1, %list2, %list3 = pto.get_vms4_sr : i16, i16, i16, i16`
- **semantics:** Read `VMS4_SR` and return the finished counts for source
  lists 0, 1, 2, and 3. After exhausted `pto.vmrgsort4`, the four results map
  to `VMS4_SR[15:0]`, `VMS4_SR[31:16]`, `VMS4_SR[47:32]`, and
  `VMS4_SR[63:48]`.

---

#### Current Implementation Surface Summary

- `pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- `pto.vmula %acc, %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- `pto.vmulscvt %input, %scalar, %mask, %rnd, %part : !pto.vreg<NxT0>, T0, !pto.mask<G> -> !pto.vreg<MxT1>`
- `pto.vci %index {order = "ASC|DESC"} : T -> !pto.vreg<NxT>`
- `pto.vbitsort %dest, %src, %indices, %repeat_times : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, index`
- `pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, i64, i64`
- `pto.get_vms4_sr : i16, i16, i16, i16`

---

#### Typical Usage

```mlir
// Softmax with fused expdiff
%max_broadcast = pto.vlds %ub_max[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%exp_stable = pto.vexpdif %logits, %max_broadcast, %mask, "ODD" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Leaky ReLU activation
%activated = pto.vlrelu %linear_out, %alpha_scalar, %mask : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.vreg<64xf32>

// Generate ascending si32 indices for argsort
%indices = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xsi32>
```

<a id="micro-14-shared-arith"></a>

### 14. Arith (Shared MLIR Dialect)

> **Category:** Shared full scalar `arith` surface used around PTO ops
> **Dialect:** `arith`
> **Upstream Reference:** https://mlir.llvm.org/docs/Dialects/ArithOps/

The upstream MLIR `arith` dialect defines primitive arithmetic, comparison, select, and cast operations over signless integer, index, floating-point, and boolean-compatible scalar values. Within PTO micro Instruction code, the full scalar operation surface of `arith` is supported. These ops are used around PTO instructions to build constants, compute offsets and loop bounds, perform general scalar math, derive valid-shape metadata, and form predicates for `scf` control flow.

These ops are part of the documented PTO micro Instruction surface, but they are not PTO ISA instructions.

---

#### Role in PTO micro Instruction Code

- materialize scalar constants used by PTO scalar operands and loop bounds
- compute scalar/index offsets for tensor views, partitioning, and dynamic shapes
- perform general scalar integer and floating-point math outside PTO vector/tile payload operations
- derive scalar predicates that guard `scf.if` or `scf.while`
- apply scalar casts, width changes, bitwise ops, and selects without introducing PTO-specific control ops

Prefer PTO ops for vector or tile payload math. Use `arith` for scalar computation and bookkeeping that surrounds PTO regions.

---

#### Supported Surface

The documented PTO micro Instruction surface supports the full scalar operation surface of upstream `arith`. The upstream `arith` dialect reference remains authoritative for the exhaustive op-by-op syntax and semantics. The categories below summarize how that support is used in PTO micro Instruction code.

| Category | Representative Ops | Typical Use in PTO micro Instruction Code |
|----------|--------------------|------------------|
| Constants | `arith.constant` | integer, floating-point, boolean, and `index` constants |
| Integer / Index Arithmetic | `arith.addi`, `arith.subi`, `arith.muli`, `arith.divsi`, `arith.divui`, `arith.ceildivsi`, `arith.ceildivui`, `arith.floordivsi`, `arith.remsi`, `arith.remui` | offsets, bounds, chunk sizes, scalar math |
| Floating-Point Arithmetic | `arith.addf`, `arith.subf`, `arith.mulf`, `arith.divf`, `arith.negf`, `arith.maximumf`, `arith.minimumf`, `arith.maxnumf`, `arith.minnumf` | scalar math around PTO regions |
| Bitwise / Shift Ops | `arith.andi`, `arith.ori`, `arith.xori`, `arith.shli`, `arith.shrsi`, `arith.shrui` | flags, masks, packed scalar fields |
| Comparisons / Select | `arith.cmpi`, `arith.cmpf`, `arith.select`, `arith.maxsi`, `arith.minui` | predicates, clamps, scalar muxes |
| Casts / Width Changes | `arith.index_cast`, `arith.index_castui`, `arith.extsi`, `arith.extui`, `arith.trunci`, `arith.sitofp`, `arith.uitofp`, `arith.fptosi`, `arith.fptoui`, `arith.extf`, `arith.truncf`, `arith.bitcast` | ABI glue, dynamic-shape plumbing, scalar type adaptation |

---

#### Current PTOAS Coverage

- the current repository examples are still dominated by constants, casts, integer/index arithmetic, compares, and selects because those are the most common surrounding-scalar patterns in existing kernels
- backend-specific tests such as the PTO shared-dialect fixture visibly exercise only a representative subset of `arith` ops in a single path
- the documented PTO micro Instruction source-level contract is nevertheless the full scalar `arith` surface, not just the index-heavy subset that appears most often in current samples

This section therefore uses representative categories and examples instead of pretending that the supported `arith` surface is limited to the currently most common sample patterns.

---

#### Typical Patterns

##### Scalar Setup

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%scale = arith.constant 2.0 : f32
```

##### Dynamic Offset Computation

```mlir
%vrow = arith.index_cast %valid_row : i32 to index
%chunk = arith.muli %row, %c32 : index
%tail = arith.subi %limit, %chunk : index
```

##### General Scalar Arithmetic

```mlir
%sum_i = arith.addi %lhs_i, %rhs_i : i32
%sum_f = arith.addf %lhs_f, %rhs_f : f32
%prod_f = arith.mulf %sum_f, %scale : f32
```

##### Scalar Predicate and Selection

```mlir
%is_first = arith.cmpi eq, %i, %c0 : index
%active = arith.select %is_first, %first_count, %steady_count : index
```

##### Bitwise / Width Adaptation

```mlir
%flags = arith.andi %flags0, %flags1 : i32
%wide = arith.extui %flags : i32 to i64
%shrunk = arith.trunci %wide : i64 to i16
```

---

#### Authoring Guidance

- treat upstream `arith` scalar semantics as the source of truth for supported scalar ops
- keep `arith` values scalar or `index` typed; do not use `arith` as a substitute for PTO vector/tile compute
- use `arith` for general scalar math, scalar comparisons, bitwise operations, and casts around PTO regions, not just for `index` arithmetic
- use `arith.cmpi` / `arith.cmpf` plus `scf.if` / `scf.while` for control flow, not ad hoc control intrinsics
- prefer `arith.index_cast` / `arith.index_castui` at ABI or shape boundaries where `index` is required, but do not read that as a restriction on the rest of scalar `arith`

<a id="micro-15-shared-scf"></a>

### 15. SCF (Shared MLIR Dialect)

> **Category:** Shared structured control flow around PTO regions
> **Dialect:** `scf`
> **Upstream Reference:** https://mlir.llvm.org/docs/Dialects/SCFDialect/

The upstream MLIR `scf` dialect defines structured control flow operations with regions, including counted loops, conditional regions, and while-style loops. In PTO micro Instruction code, `scf` is the control shell around PTO ops: it sequences DMA, vector, and tile operations; carries scalar or tile state across iterations; and preserves analyzable control flow for PTO-specific analyses and lowerings.

These ops are part of the documented PTO micro Instruction surface, but they are shared MLIR control-flow constructs rather than PTO ISA instructions.

---

#### Supported Ops

| Op | Role in PTO micro Instruction Code | Notes |
|----|------------------------|-------|
| `scf.for` | counted loops and loop-carried values | common structured counted loop form |
| `scf.if` | structured conditional execution | may yield values or act as side-effect-only branch |
| `scf.yield` | region terminator for `for` / `if` / `while` bodies | carries loop or branch results |
| `scf.while` | break-like or stateful loops | useful for source-level structured control |
| `scf.condition` | loop-continue / loop-exit decision for `scf.while` | placed in the "before" region |

Ops such as `scf.execute_region`, `scf.forall`, or `scf.index_switch` are not part of the documented shared-dialect portion of the PTO micro Instruction surface here.

---

#### Current PTOAS Coverage

- `scf.for`, `scf.if`, and `scf.yield` are directly exercised in the shared-dialect PTO fixture and appear widely across PTO samples
- PTO synchronization and memory analyses explicitly reason about `scf.for`, `scf.if`, `scf.yield`, and `scf.while`
- `scf.while` and `scf.condition` appear in control-flow samples and are handled in PTO-to-EmitC control-flow lowering, but they are less broadly exercised than `for` / `if` on all backend paths

---

#### Typical Patterns

##### Counted Loop

```mlir
scf.for %i = %c0 to %c4 step %c1 {
  %offset = arith.muli %i, %c32 : index
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %v = pto.vlds %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%offset], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
}
```

##### Counted Loop with Loop-Carried State

```mlir
%final_alive = scf.for %i = %c0 to %c4 step %c1
    iter_args(%alive = %true) -> (i1) {
  %break_now = arith.cmpi eq, %i, %c2 : index
  %next_alive = scf.if %break_now -> (i1) {
    scf.yield %false : i1
  } else {
    scf.yield %alive : i1
  }
  scf.yield %next_alive : i1
}
```

##### Structured Conditional Region

```mlir
%is_mode_a = arith.cmpi eq, %mode, %c0_i32 : i32
scf.if %is_mode_a {
  pto.tmuls ins(%data, %scale_a : !pto.tile_buf<...>, f32) outs(%data : !pto.tile_buf<...>)
} else {
  pto.tadds ins(%data, %bias_b : !pto.tile_buf<...>, f32) outs(%data : !pto.tile_buf<...>)
}
```

##### While-Style Break Loop

```mlir
%final:2 = scf.while (%i = %c0, %alive = %true) : (index, i1) -> (index, i1) {
  %lt = arith.cmpi slt, %i, %c4 : index
  %go = arith.andi %lt, %alive : i1
  scf.condition(%go) %i, %alive : index, i1
} do {
^bb0(%i2: index, %alive2: i1):
  %next_i = arith.addi %i2, %c1 : index
  scf.yield %next_i, %alive2 : index, i1
}
```

---

#### Authoring Guidance

- use `scf.for` for regular counted loops and loop-carried scalar/tile state
- use `scf.if` for structured branching around PTO regions instead of inventing PTO-specific branch ops
- keep region results explicit with `scf.yield`; this is important for PTO analyses that track carried buffers and aliasing
- use `scf.while` only when a counted loop cannot express the control cleanly; `scf.for` remains the more common and better-exercised form in the current repository
- build branch predicates and loop conditions with `arith` ops, not PTO vector masks, unless the control decision truly comes from a scalarized value

<a id="micro-16-cube-matmul"></a>

### 16. Cube Matrix Multiply

> **Category:** Cube unit ops — staged load/store, matrix multiply, and
> FIXPIPE MTE writeback

This chapter documents the high-level Cube VPTO surface. It describes logical
data objects, operand units, layout contracts, numeric behavior, and writeback
effects from the user's point of view.

---

#### Common Cube Operand Model

Cube ops use typed PTO pointers to name logical storage domains. The canonical
`!pto.ptr` address-space names are the hardware-domain names below. The legacy
names are accepted only as parser aliases and are printed back as canonical
names.

| Canonical address space | Legacy alias | Logical role |
|-------------------------|--------------|--------------|
| `gm` | - | Global memory |
| `l1` | `mat` | L1 matrix staging buffer |
| `l0a` | `left` | Left matrix operand tile for Cube compute |
| `l0b` | `right` | Right matrix operand tile for Cube compute |
| `l0c` | `acc` | Accumulator/result tile produced by Cube compute |
| `bt` | `bias` | Bias vector payload consumed by bias matmul forms |
| `fb` | `scaling` | FIXPIPE parameter payloads consumed by vector quant/ReLU clauses |
| `ub` | `vec` | Unified Buffer destination/source for vector-side use |

Unless an op says otherwise:

- Shape operands such as `%m`, `%n`, `%k`, `shape(%n, %d)` are logical element
  counts, not byte counts.
- Length operands named `%len_burst` in byte-copy surfaces are byte counts
  unless the op explicitly states a different unit.
- Strides named `src_stride` or `dst_stride` are start-to-start distances in
  the unit stated by the op. Do not infer byte units from the name alone.
- Pointer operands select the base address of the logical object. Sub-tile
  selection is expressed by computing a different base pointer before calling
  the op, unless the op exposes an explicit start or group operand.
- Cache/session hint operands may affect the memory path but do not change the
  mathematical value written or read.

---

#### Cube Compute Ops

The `pto.mad*` family computes logical matrix multiplication over tiles already
prepared in `l0a` and `l0b`:

```text
lhs: M x K
rhs: K x N
dst: M x N
```

The matrix element types are inferred from `%lhs`, `%rhs`, and `%dst` pointer
element types. There is no separate type selector. Unsupported type
combinations are invalid programs.

The current VPTO surface enforces the Cube storage roles through pointer
address spaces: `%lhs` is `l0a`, `%rhs` is `l0b`, and `%dst` is `l0c`.
Bias forms additionally require `%bias` in the `bt` address space with the
same element type as `%dst`. MX forms require MX element types on both `%lhs`
and `%rhs`; the current target-profile MX data type is `f8E4M3FN`.

##### MAD Common Clauses

| Clause | Values | Effect |
|--------|--------|--------|
| `unit_flag(...)` | `check_only`, `check_and_set` | Participates in producer-side tile synchronization. `check_only` checks that the producer slot can be used. `check_and_set` also publishes the produced `%dst` tile for later consumers. Omit the clause when the schedule does not use unit flags for this tile. |
| `disable_gemv` | flag | Applies only when `%m = 1`. Omitted means GEMV A-vector consumption: `%lhs` must contain the logical `1 x K` row in the target GEMV left-tile organization. Present means normal matmul left-tile organization. The mathematical result is still `lhs @ rhs`; only the required `%lhs` organization changes. For `%m != 1`, normal matmul organization is used. |
| `sat` / `nosat` | flags | Floating exceptional-value mode for floating and MX MAD forms. With `sat`, exceptional multiply inputs are normalized before arithmetic (`+/-inf` to finite type extrema, `nan` to 0) and finite overflow saturates to the finite type range. With `nosat`, exceptional inputs are preserved and overflow may produce exceptional outputs. Omit both to use the execution mode selected outside this op. Integer MAD forms do not accept these flags. |
| `tf32_mode(...)` | `round_even`, `round_away` | Valid only for non-MX `f32 x f32 -> f32`. FP32 inputs are rounded to TF32 precision before multiplication; accumulation and output remain FP32. |
| `n_dir` | flag | Requests N-direction result production order for schedules that combine compute with unit flags and later layout movement. It does not change `dst[m, n]`. |

Reference semantics for non-MX forms:

```text
product[m, n] = sum k in 0 .. K-1:
                  numeric_lhs(lhs[m, k]) * numeric_rhs(rhs[k, n])

pto.mad:      dst[m, n] = product[m, n]
pto.mad_acc:  dst[m, n] = dst[m, n] + product[m, n]
pto.mad_bias: dst[m, n] = product[m, n] + bias[n]
```

For integer forms, the op multiplies the typed values already present in
`l0a` and `l0b`. Per-input offset correction for quantized integer
algorithms is not an operand of `pto.mad*`; apply such correction before
loading the Cube operands when the algorithm needs it.

##### MX Matmul Model

`pto.mad_mx*` additionally applies microscaling. The scale payloads are loaded
with `pto.mte_l1_l0a_mx` / `pto.mte_l1_l0b_mx` and are associated with the
selected `%lhs` / `%rhs` tiles; they are not direct operands of `pto.mad_mx*`.

The K dimension is partitioned into 32-element groups:

```text
k_group = floor(k / 32)

mx_product[m, n] =
  sum k in 0 .. K-1:
    (lhs[m, k] * lhs_scale[m, k_group]) *
    (rhs[k, n] * rhs_scale[k_group, n])
```

Current target-profile MX data tiles use `f8E4M3FN`. `%k` must be compatible
with MX grouping. On the current target profile, MX matmul consumes K in
64-element multiples, which contain two 32-element scale groups.

##### `pto.mad`

- **syntax:**
```mlir
pto.mad %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```
- **semantics:** Zero-init matrix multiply, `dst[m, n] = sum_k(lhs[m, k] * rhs[k, n])`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | Left operand tile in `l0a`, interpreted as logical `M x K` |
| `%rhs` | ptr | Right operand tile in `l0b`, interpreted as logical `K x N` |
| `%dst` | ptr | Accumulator destination tile in `l0c`, interpreted as logical `M x N` |
| `%m` | i64 | Logical M element count |
| `%n` | i64 | Logical N element count |
| `%k` | i64 | Logical K element count |
| optional clauses | - | See [MAD Common Clauses](#mad-common-clauses) |

**Constraints:**

- `%lhs`, `%rhs`, and `%dst` must be in `l0a`, `l0b`, and `l0c`.
- `%m`, `%n`, and `%k` must be positive and satisfy the target shape limits
  for the selected element-type combination.
- `tf32_mode(...)` requires `f32` lhs, rhs, and dst element types.
- `sat` / `nosat` requires a floating element-type combination.
- Packed 4-bit integer data requires `%k` to select an even number of K
  elements.

**Example:**

```mlir
pto.mad %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c32_i64
  : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

---

##### `pto.mad_acc`

- **syntax:**
```mlir
pto.mad_acc %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```
- **semantics:** Accumulating matrix multiply,
  `dst[m, n] = dst[m, n] + sum_k(lhs[m, k] * rhs[k, n])`.

**Parameter Table:** same as `pto.mad`.

**Constraints:** same as `pto.mad`.

**Example:**

```mlir
pto.mad_acc %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c32_i64 unit_flag(check_only)
  : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

---

##### `pto.mad_bias`

- **syntax:**
```mlir
pto.mad_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, !pto.ptr<C, bt>, i64, i64, i64
```
- **semantics:** Bias-init matrix multiply,
  `dst[m, n] = sum_k(lhs[m, k] * rhs[k, n]) + bias[n]`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs`, `%rhs`, `%dst`, `%m`, `%n`, `%k` | - | Same as `pto.mad` |
| `%bias` | ptr | Bias vector in `bt`, interpreted as `N` values and broadcast across M |
| optional clauses | - | See [MAD Common Clauses](#mad-common-clauses) |

**Constraints:**

- `%bias` must be in `bt` address space.
- `%bias` element type must match `%dst` element type.
- Only `N` bias values are consumed; `%bias` is not an `M x N` matrix.
- Other constraints match `pto.mad`.

**Example:**

```mlir
pto.mad_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c32_i64
  : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, !pto.ptr<f32, bt>, i64, i64, i64
```

---

##### `pto.mad_mx`

- **syntax:**
```mlir
pto.mad_mx %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```
- **semantics:** Zero-init MX matrix multiply, `dst[m, n] = mx_product[m, n]`.

**Parameter Table:** same as `pto.mad`; `%lhs` and `%rhs` must have matching
MX scale payloads prepared by the MX load ops.

**Constraints:**

- Operands must use a target-supported MX dtype combination.
- Matching left and right MX scale payloads must be loaded before this op.
- `%k` must satisfy the MX grouping rule described in [MX Matmul Model](#mx-matmul-model).
- `tf32_mode(...)` is not a clause of MX MAD.

**Example:**

```mlir
pto.mad_mx %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, l0a>, !pto.ptr<f8E4M3FN, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

---

##### `pto.mad_mx_acc`

- **syntax:**
```mlir
pto.mad_mx_acc %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```
- **semantics:** Accumulating MX matrix multiply,
  `dst[m, n] = dst[m, n] + mx_product[m, n]`.

**Parameter Table:** same as `pto.mad_mx`.

**Constraints:** same as `pto.mad_mx`.

**Example:**

```mlir
pto.mad_mx_acc %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, l0a>, !pto.ptr<f8E4M3FN, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

---

##### `pto.mad_mx_bias`

- **syntax:**
```mlir
pto.mad_mx_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, !pto.ptr<C, bt>, i64, i64, i64
```
- **semantics:** Bias-init MX matrix multiply,
  `dst[m, n] = mx_product[m, n] + bias[n]`.

**Parameter Table:** same as `pto.mad_bias`, with MX `%lhs` / `%rhs` scale
payload requirements from `pto.mad_mx`.

**Constraints:** same as `pto.mad_mx` plus `pto.mad_bias` bias constraints.

**Example:**

```mlir
pto.mad_mx_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, l0a>, !pto.ptr<f8E4M3FN, l0b>, !pto.ptr<f32, l0c>, !pto.ptr<f32, bt>, i64, i64, i64
```

---

#### Cube Data Movement Ops

##### Cube Burst / Loop Addressing Model

`pto.mte_gm_l1` and `pto.mte_l1_ub` use the same grouped transfer model:

```text
burst(row) = len_burst contiguous bytes
nburst     = innermost repeated burst group
loop       = optional outer repetition group
```

For each `nburst` row, the source and destination start addresses advance by
`src_stride` and `dst_stride` after a burst row. Optional `loop(...)` groups
wrap the full inner transfer pattern and advance by their own source and
destination strides between repetitions. All lengths and strides in this model
are bytes.

##### `pto.mte_gm_l1`

- **syntax:**
```mlir
pto.mte_gm_l1 %src, %dst, %len_burst
  nburst(%count, %src_stride, %dst_stride)
  [loop(%count_i, %src_stride_i, %dst_stride_i)]*
  : !pto.ptr<T, gm>, !pto.ptr<T, l1>, i64, i64, i64, i64
```
- **semantics:** Structured GM-to-L1 copy. The op copies grouped byte ranges
  from `%src` in `gm` to `%dst` in `l1`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | GM source base pointer |
| `%dst` | ptr | L1 matrix destination base pointer in `l1` |
| `%len_burst` | i64 | Bytes copied per burst row |
| `nburst(%count, %src_stride, %dst_stride)` | i64 triple | Innermost burst count and byte strides between row starts |
| `loop(%count_i, %src_stride_i, %dst_stride_i)` | i64 triple | Optional outer repetition; strides are byte advances between enclosed patterns |

**Constraints:**

- `nburst(...)` is required.
- Each `loop(...)` group must provide all three operands.
- For a contiguous 16-element f16 vector, use `%len_burst = 32`.

**Example:**

```mlir
pto.mte_gm_l1 %bias_gm, %l1_bias, %c32_i64
  nburst(%c4_i64, %c64_i64, %c32_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, i64, i64, i64, i64
```

---

##### `pto.mte_l1_ub`

- **syntax:**
```mlir
pto.mte_l1_ub %src, %dst, %len_burst
  nburst(%count, %src_stride, %dst_stride)
  [loop(%count_i, %src_stride_i, %dst_stride_i)]*
  : !pto.ptr<T, l1>, !pto.ptr<T, ub>, i64, i64, i64, i64
```
- **semantics:** Structured L1-to-UB copy. The grouped byte ranges are read
  from `%src` in `l1` and written to `%dst` in `ub`.

**Parameter Table:** same grouped byte model as `pto.mte_gm_l1`, with source
and destination address spaces reversed to `l1 -> ub`.

**Constraints:**

- `%src` must be in `l1`, `%dst` must be in `ub`.
- `nburst(...)` is required.
- Each `loop(...)` group must provide all three operands.

**Example:**

```mlir
pto.mte_l1_ub %l1_src, %ub_dst, %c64_i64
  nburst(%c2_i64, %c128_i64, %c64_i64)
  : !pto.ptr<f16, l1>, !pto.ptr<f16, ub>, i64, i64, i64, i64
```

---

##### `pto.mte_gm_l1_frac`

- **syntax:**
```mlir
pto.mte_gm_l1_frac %src, %dst, nd2nz|dn2nz,
  shape(%n_value, %d_value),
  src_layout(%src_inner_stride[, %src_outer_stride]),
  dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride, %dst_loop4_stride),
  ctrl(%l2_cache_ctrl, %smallc0_en)
  : !pto.ptr<T, gm>, !pto.ptr<T, l1>, ...
```
- **semantics:** Load a logical 2-D GM region and write one or more L1 NZ
  matrix groups. `nd2nz` reads a logical `src[n, d]` matrix. `dn2nz` reads a
  logical `src[d, n]` matrix and writes the same logical `N x D` result into
  NZ layout.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | GM source base pointer |
| `%dst` | ptr | L1 NZ destination base pointer in `l1` |
| `nd2nz` / `dn2nz` | keyword | Source logical layout mode |
| `shape(%n_value, %d_value)` | i64 pair | Logical output shape before NZ packing |
| `src_layout(%src_inner_stride[, %src_outer_stride])` | i64 / optional i64 | Source row/matrix byte strides |
| `dst_group(...)` | i64 tuple | Destination group count and placement strides in C0-size units |
| `ctrl(%l2_cache_ctrl, %smallc0_en)` | i64, i1 | Cache hint and small-C0 packing enable |

`src_layout(%src_inner_stride)` describes one logical source matrix. For
`nd2nz`, `%src_inner_stride` is the byte distance from `src[n, 0]` to
`src[n + 1, 0]`. For `dn2nz`, it is the byte distance from `src[d, 0]` to
`src[d + 1, 0]`. When `%src_outer_stride` is present, it is the byte distance
between adjacent source matrices. When omitted, the outer source stride is 0.

`dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride,
%dst_loop4_stride)` writes `%group_count` logical matrices. Destination strides
are measured in C0-size units; one C0-size unit is 32 bytes. These strides
place generated NZ blocks relative to `%dst`. They do not select a separate
memory block.

Reference addressing:

```text
for g in 0 .. group_count-1:
  src_g = src + g * src_outer_stride
  dst_g = dst + g * dst_loop4_stride * 32

  for n in 0 .. n_value-1:
    for d in 0 .. d_value-1:
      if mode == nd2nz:
        value = load(src_g + n * src_inner_stride + d * sizeof(T))
      else:
        value = load(src_g + d * src_inner_stride + n * sizeof(T))
      store value into NZ position for logical [n, d] under dst_g

  invalid lanes in the final C0 group are written as zero
```

**Constraints:**

- Source strides are bytes. For row-major `16 x 16` f16 input,
  `src_layout(32)` describes consecutive rows.
- Destination strides are C0-size units, not bytes and not elements.
- `smallc0_en = true` is valid only for target-supported small-C0 cases. The
  current contract rejects `d_value > 4` in small-C0 mode.
- In normal C0 mode, each destination C0 burst is padded to 32 bytes. In
  small-C0 mode, each destination burst is padded to 4 logical channels, and
  the generated inner-N and C0 destination placement is fixed by that
  small-C0 packing rule. `%dst_loop4_stride` still places adjacent matrix
  groups.
- In small-C0 mode, missing logical `N` rows and invalid `D` lanes are written
  as zero, and the tail of a generated NZ matrix is padded to the 32-byte C0
  boundary.
- Destination regions selected by `%dst` and `dst_group(...)` must not overlap.
  If two generated writes target the same bytes, the final value is not a
  stable program result.

**Example:**

```mlir
pto.mte_gm_l1_frac %src, %dst, nd2nz,
  shape(%c32_i64, %c16_i64),
  src_layout(%c32_i64, %c1024_i64),
  dst_group(%c2_i64, %c1_i64, %c16_i64, %c64_i64),
  ctrl(%c0_i64, %false)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, nd2nz, shape i64, i64,
    src_layout(i64, i64), dst_group i64, i64, i64, i64, ctrl i64, i1
```

---

##### `pto.mte_l1_bt`

- **syntax:**
```mlir
pto.mte_l1_bt %src, %dst, %len_burst
  nburst(%count, %src_gap, %dst_gap)
  : !pto.ptr<T, l1>, !pto.ptr<U, bt>, i64, i64, i64, i64
```
- **semantics:** Load an L1 bias payload into the `bt` address space for
  later `pto.mad_bias` / `pto.mad_mx_bias` consumption. The consumer interprets
  the result as an `N`-element bias vector `bias[n]`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer in `l1` |
| `%dst` | ptr | Bias destination pointer in `bt` |
| `%len_burst` | i64 | Number of bias-load units per burst |
| `%count` | i64 | Burst count |
| `%src_gap` | i64 | Source gap between bursts, in bias-load units |
| `%dst_gap` | i64 | Destination gap between bursts, in bias-load units |

One burst loads `%len_burst` units from `%src` and writes the corresponding
bias values to `%dst`. After each burst except the last, source and destination
advance by the burst length plus the corresponding gap.

**Constraints:**

- Supported type pairs: `f32->f32`, `i32->i32`, `f16->f32`, `bf16->f32`.
- For `bf16->f32`, compact bf16 source values are always widened to f32 bias
  values. For `f16->f32`, compact f16 source values are widened when the load
  is used as an f32 bias payload; otherwise the f16 payload is stored in the
  32-bit bias slot with unused high bits.
- Load exactly the channel bias values needed by the consumer tile; the bias
  payload is not result-shaped.

**Example:**

```mlir
pto.mte_l1_bt %l1_bias, %bt, %c1_i64 nburst(%c4_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f16, l1>, !pto.ptr<f32, bt>, i64, i64, i64, i64
```

---

##### `pto.mte_l1_fb`

- **syntax:**
```mlir
pto.mte_l1_fb %src, %dst, %len_burst
  nburst(%count, %src_gap, %dst_gap)
  : !pto.ptr<T, l1>, !pto.ptr<U, fb>, i64, i64, i64, i64
```
- **semantics:** Load FIXPIPE parameter payloads from L1 into `fb`.
  Vector `pre_quant(...)` and `pre_relu(...)` clauses in `pto.mte_l0c_l1*`
  later consume these payloads through `fb` pointers.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer in `l1` |
| `%dst` | ptr | Scaling destination pointer in `fb` |
| `%len_burst` | i64 | Number of parameter-load units per burst |
| `%count` | i64 | Burst count |
| `%src_gap` | i64 | Source gap between bursts, in parameter-load units |
| `%dst_gap` | i64 | Destination gap between bursts, in parameter-load units |

The copy unit of `pto.mte_l1_fb` is the parameter-load unit of this op. It is
separate from the row size consumed by `mte_l0c_*` vector payloads.
`%len_burst` and the `nburst(...)` gaps are counted in these load units, not
in bytes and not in destination elements. After `pto.mte_l1_fb` materializes the
payload in `fb`, vector pre-ReLU consumers read it as 64B parameter rows
and vector pre-quant consumers read it as 128B parameter rows. The payload
pointer passed to `mte_l0c_*` must point at the first row for the logical
output tile, and rows must follow the same channel/NZ order consumed by that
store.

**Constraints:**

- `%src` must be in `l1`, `%dst` must be in `fb`.
- Vector `pre_quant` and `pre_relu` consumers require parameter data prepared
  in the row order documented by [FIXPIPE MTE Ops](#fixpipe-mte-ops).

**Example:**

```mlir
pto.mte_l1_fb %l1_fp, %fb_fp, %c2_i64 nburst(%c4_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f32, l1>, !pto.ptr<f32, fb>, i64, i64, i64, i64
```

---

##### Left / Right Tile Load Model

`pto.mte_l1_l0a` and `pto.mte_l1_l0b` move L1 cube-fractal tiles into the
compute operand domains. `%src` must already point to an L1 cube-fractal tile;
these ops do not convert arbitrary row-major matrices. Use
`pto.mte_gm_l1_frac` first when the original data is plain ND/DN layout.

If `transpose = true`, the selected logical source tile is transposed before it
is placed in the destination operand domain. Omitting the attribute means
`transpose = false`.

The `%start_row` and `%start_col` operands select the row and column offset of
the source tile extraction start position. Frontends that expose these as
optional user arguments must materialize `0` for both operands when omitted.

##### `pto.mte_l1_l0a`

- **syntax:**
```mlir
pto.mte_l1_l0a %src, %dst, %m, %k, %start_row, %start_col
  : !pto.ptr<T, l1>, !pto.ptr<T, l0a>, i64, i64, i64, i64
```
- **semantics:** Load a logical `%m x %k` left tile from L1 `l1` into `l0a`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 cube-fractal source tile in `l1` |
| `%dst` | ptr | Left operand destination in `l0a` |
| `%m` | i64 | Logical M extent |
| `%k` | i64 | Logical K extent |
| `%start_row` | i64 | Source row offset |
| `%start_col` | i64 | Source column offset |
| `transpose` | attr | Optional boolean source-tile transpose before destination placement |

**Constraints:**

- `%src` must be in `l1`, `%dst` must be in `l0a`.
- `%src` and `%dst` must satisfy the target alignment for Cube tile loads.
- `transpose = true` requires a tile shape supported by the element-type
  transpose granularity.

**Example:**

```mlir
pto.mte_l1_l0a %l1_a, %l0a, %c16_i64, %c32_i64, %c0_i64, %c0_i64
  : !pto.ptr<f16, l1>, !pto.ptr<f16, l0a>, i64, i64, i64, i64
```

---

##### `pto.mte_l1_l0b`

- **syntax:**
```mlir
pto.mte_l1_l0b %src, %dst, %k, %n, %start_row, %start_col
  : !pto.ptr<T, l1>, !pto.ptr<T, l0b>, i64, i64, i64, i64
```
- **semantics:** Load a logical `%k x %n` right tile from L1 `l1` into
  `l0b`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 cube-fractal source tile in `l1` |
| `%dst` | ptr | Right operand destination in `l0b` |
| `%k` | i64 | Logical K extent |
| `%n` | i64 | Logical N extent |
| `%start_row` | i64 | Source row offset |
| `%start_col` | i64 | Source column offset |
| `transpose` | attr | Optional boolean source-tile transpose before destination placement |

**Constraints:**

- `%src` must be in `l1`, `%dst` must be in `l0b`.
- `%src` and `%dst` must satisfy the target alignment for Cube tile loads.
- `transpose = true` requires a tile shape supported by the element-type
  transpose granularity.

**Example:**

```mlir
pto.mte_l1_l0b %l1_b, %l0b, %c32_i64, %c16_i64, %c0_i64, %c0_i64
  : !pto.ptr<f16, l1>, !pto.ptr<f16, l0b>, i64, i64, i64, i64
```

---

##### MX Scale Load Model

MX scale loads prepare the scale payloads consumed by `pto.mad_mx*`. Each scale
entry applies to one 32-element K group.

- Left scale logical shape: `[M, ceil(K / 32)]`.
- Right scale logical shape: `[ceil(K / 32), N]`.
- L1 source data is organized as 32B scale fragments in the same logical order
  as the associated data tile.

##### `pto.mte_l1_l0a_mx`

- **syntax:**
```mlir
pto.mte_l1_l0a_mx %src, %dst, %m, %k, %start_row, %start_col
  : !pto.ptr<T, l1>, !pto.ptr<T, l0a>, i64, i64, i64, i64
```
- **semantics:** Load left-side MX scale fragments for a logical `%m x %k`
  left data tile.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 MX scale source in `l1` |
| `%dst` | ptr | Left-side MX payload destination associated with `l0a` |
| `%m` | i64 | M extent of the associated left data tile |
| `%k` | i64 | K extent; scale grouping is by 32 K elements |
| `%start_row` | i64 | Source MX-fractal row offset in the packed L1 big matrix |
| `%start_col` | i64 | Source MX-fractal column offset in the packed L1 big matrix |

**Constraints:**

- `%src` must be in `l1`, `%dst` must be in `l0a`.
- `%src` and `%dst` must satisfy 32B MX scale-fragment alignment.

**Example:**

```mlir
pto.mte_l1_l0a_mx %l1_a_scale, %l0a_scale, %c16_i64, %c64_i64, %c0_i64, %c0_i64
  : !pto.ptr<f8E4M3FN, l1>, !pto.ptr<f8E4M3FN, l0a>, i64, i64, i64, i64
```

---

##### `pto.mte_l1_l0b_mx`

- **syntax:**
```mlir
pto.mte_l1_l0b_mx %src, %dst, %k, %n, %start_row, %start_col
  : !pto.ptr<T, l1>, !pto.ptr<T, l0b>, i64, i64, i64, i64
```
- **semantics:** Load right-side MX scale fragments for a logical `%k x %n`
  right data tile.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 MX scale source in `l1` |
| `%dst` | ptr | Right-side MX payload destination associated with `l0b` |
| `%k` | i64 | K extent; scale grouping is by 32 K elements |
| `%n` | i64 | N extent of the associated right data tile |
| `%start_row` | i64 | Source MX-fractal row offset in the packed L1 big matrix |
| `%start_col` | i64 | Source MX-fractal column offset in the packed L1 big matrix |

**Constraints:**

- `%src` must be in `l1`, `%dst` must be in `l0b`.
- `%src` and `%dst` must satisfy 32B MX scale-fragment alignment.

**Example:**

```mlir
pto.mte_l1_l0b_mx %l1_b_scale, %l0b_scale, %c64_i64, %c16_i64, %c0_i64, %c0_i64
  : !pto.ptr<f8E4M3FN, l1>, !pto.ptr<f8E4M3FN, l0b>, i64, i64, i64, i64
```

---

#### FIXPIPE MTE Ops

`pto.mte_l0c_l1*` writes logical accumulator results from `l0c` to `l1`, `gm`,
or `ub`. The family shares this pipeline order:

```text
1. Read logical acc[m, n] from %src using the selected layout mode.
2. Optionally participate in consumer-side unit-flag synchronization.
3. Optionally apply pre_quant(payload, mode).
4. Optionally apply pre_relu(payload, mode), then optional clip.
5. Convert to the destination element type using sat/nosat behavior.
6. Write to the selected destination layout and address space.
7. Apply store-target effects such as GM atomic or UB dual destination.
```

Only the clauses documented here affect `pto.mte_l0c_l1*`. Other transforms
must be represented by separate PTO ops before producing `l0c` or after the
writeback destination is materialized.

##### FIXPIPE Common Clauses

| Clause | Values | Effect |
|--------|--------|--------|
| `unit_flag(...)` | `check_only`, `check_and_clear` | Checks that the accumulator tile is ready for consumption. `check_and_clear` also clears the consumed tile state for later reuse. Omit when the schedule does not use unit flags. |
| `pre_quant(%payload, mode = ...)` | see below | Applies the selected pre-quantization or conversion before ReLU/clip and final store. |
| `pre_relu([%payload, ]mode = ...[, clip = %clip])` | `no_relu`, `normal_relu`, `scalar_relu`, `vector_relu` | Applies ReLU-family activation before final destination conversion. `clip` is part of this clause and applies after the selected ReLU mode. |
| `nz2nd` / `nz2dn(...)` / `nz2nz(...)` | layout modes | Selects how logical `acc[m, n]` is written to the destination layout. |
| `loop3(%count, %src_stride3, %dst_stride3)` | i64 triple | Repeats the whole selected `m x n` writeback pattern. |
| `sat` / `sat(preserve_nan)` / `nosat` | flags | Selects final conversion behavior for floating exceptional values and finite overflow where the destination type is affected. |

`pre_quant` legal modes:

```text
f32_f16,
qf322hif8_pre_vec, qf322hif8_pre_scalar,
qf322hif8_pre_hybrid_vec, qf322hif8_pre_hybrid_scalar,
deqs32_int_vec, deqs32_int_scalar,
req8_vec, req8_scalar,
deqf16_vec, deqf16_scalar,
qf322fp8_pre_vec, qf322fp8_pre_scalar,
qf322f32_pre_vec, qf322f32_pre_scalar,
f32_bf16,
qf162b8_pre_vec, qf162b8_pre_scalar,
qf162s4_pre_vec, qf162s4_pre_scalar,
req4_vec, req4_scalar,
qf322b8_pre_vec, qf322b8_pre_scalar,
qf322s4_pre_vec, qf322s4_pre_scalar,
deqs16_vec, deqs16_scalar,
qf162s16_pre_vec, qf162s16_pre_scalar,
qf322f16_pre_vec, qf322f16_pre_scalar,
qf322bf16_pre_vec, qf322bf16_pre_scalar,
qs322bf16_pre_vec, qs322bf16_pre_scalar
```

`_scalar` modes take one floating scalar payload (`f16`, `bf16`, or `f32`)
broadcast to the whole logical output tile. `f16` and `bf16` scalar payloads
are first interpreted as numeric values and widened to `f32`; `f32` payloads
are used directly. `_vec` modes take a `!pto.ptr<f16|bf16|f32, fb>`
payload pointer. The pointer element type is the logical parameter element
type, not a packed transport carrier. The pointer names the first parameter
row for this store; later rows
advance in the same channel/NZ order as the logical accumulator elements
consumed by the selected layout mode. Each vector pre-quant row is a 128B
parameter row prepared by `pto.mte_l1_fb`; each row supplies the per-channel
scale and any mode-specific offset/sign controls used by the selected
quantization family. Vector pre-ReLU rows are 64B parameter rows and supply
the per-channel alpha values consumed by `vector_relu`.

`pre_quant` mode families:

| Family | Acc source | Result meaning | Payload |
|--------|------------|----------------|---------|
| `f32_f16`, `f32_bf16` | `f32` | Convert f32 accumulator values to f16 or bf16; rounding is nearest, ties to even | Scalar payload is required by syntax but does not select per-channel scaling |
| `qf322hif8_pre_*`, `qf322fp8_pre_*` | `f32` | Scale and quantize f32 to hif8/fp8-style destination payloads | Scalar scale or vector scale rows; hybrid modes use the target hybrid rule |
| `qf322f32_pre_*` | `f32` | Apply quant scaling while keeping f32 destination values | Scalar scale or vector scale rows |
| `qf322f16_pre_*`, `qf322bf16_pre_*` | `f32` | Scale f32, then convert to f16 or bf16 destination values | Scalar scale or vector scale rows |
| `qf322b8_pre_*`, `qf322s4_pre_*` | `f32` | Scale, offset, round, and narrow f32 to 8-bit or signed 4-bit integer payloads | Scalar or vector scale/offset parameter set |
| `qf162b8_pre_*`, `qf162s4_pre_*` | `f32` | Convert through an f16-domain pre-stage, then scale/narrow to integer payloads | Scalar or vector scale/offset parameter set |
| `qf162s16_pre_*` | `i32` | Convert through an f16-domain pre-stage, then scale/narrow to signed 16-bit payloads | Scalar or vector scale/offset parameter set |
| `deqs32_int_*`, `deqs16_*` | `i32` | Rescale integer accumulator values in an integer destination family | Scalar or vector multiplier/offset parameter set |
| `req8_*`, `req4_*` | `i32` | Requantize i32 accumulator values to 8-bit or 4-bit integer payloads | Scalar or vector multiplier/offset/sign parameter set |
| `deqf16_*` | `i32` | Dequantize i32 accumulator values to f16 destination values | Scalar or vector multiplier/offset parameter set |
| `qs322bf16_pre_*` | `i32` | Scale i32 accumulator values and convert to bf16 destination values | Scalar or vector multiplier/offset parameter set |

The mode name determines the accepted accumulator source family. `f32_f16`,
`f32_bf16`, `qf322hif8_pre_*`, `qf322fp8_pre_*`, `qf322f32_pre_*`,
`qf322f16_pre_*`, `qf322bf16_pre_*`, `qf322b8_pre_*`,
`qf322s4_pre_*`, `qf162b8_pre_*`, and `qf162s4_pre_*` consume `f32`
accumulator values. `deqs32_int_*`, `deqs16_*`, `req8_*`, `req4_*`,
`deqf16_*`, `qf162s16_pre_*`, and `qs322bf16_pre_*` consume `i32`
accumulator values. The final destination element type must match the result
family implied by the mode name; for example, `qf322f16_pre_*` writes an
f16-family result, while `req8_*` writes an 8-bit integer-family result.

Integer quantization families with `b8` in the name can produce either signed
8-bit or unsigned 8-bit results according to the sign control carried by the
scalar or vector parameter set. Families with `s4` or `s16` produce signed
4-bit or signed 16-bit results. Offset fields are added after scaling and
before the final narrow/saturate step. When a family has no offset/sign in its
payload, the payload scale alone controls the conversion.

`pre_relu` semantics:

```text
no_relu:      y = x
normal_relu:  y = max(x, 0)
scalar_relu:  y = x >= 0 ? x : alpha * x
vector_relu:  y = x >= 0 ? x : alpha[channel] * x
```

`scalar_relu` takes a floating scalar payload (`f16`, `bf16`, or `f32`) and
broadcasts it to all negative values in the logical tile. `vector_relu` takes
a `!pto.ptr<f16|bf16|f32, fb>` pointer whose elements are per-channel
alpha values and whose 64B rows follow the same channel/NZ order as the store.
`no_relu` and `normal_relu` do not take a payload. If
`clip = %clip` is present:

```text
y = min(y, clip)
```

`sat`, `sat(preserve_nan)`, and `nosat` control final conversion to destination
element types affected by FIXPIPE saturation:

- `sat`: finite overflow clamps to the destination finite range; `+/-inf`
  clamps to finite extrema; `nan` writes as 0.
- `sat(preserve_nan)`: same finite overflow and infinity handling as `sat`,
  but NaN writes as NaN when the destination format can represent NaN. This is
  intended for fp8 and hif8 destination families; for formats without a NaN
  encoding it is equivalent to `sat`.
- `nosat`: finite overflow may produce destination exceptional values;
  exceptional input values are preserved where the destination format supports
  them.
- For fp8 and hif8 destination families, `nosat` preserves NaN; overflow
  becomes the destination exceptional value when the destination encoding
  supports it.
- For integer destination families, `sat`/`nosat` is not the integer overflow
  policy; integer narrowing and clipping are determined by the selected
  pre-quant mode, its payload, and any `clip` clause.
- For `f32` destinations, floating exceptional values are preserved; `sat`
  does not force f32 `inf`/`nan` into finite values.

##### FIXPIPE Layout Model

`%src` points to the base accumulator tile. `%m` and `%n` select the logical
result rectangle to write. If the physical accumulator tile contains dummy rows
or lanes outside that rectangle, they are not written to the destination.

Layout modes:

| Mode | Destination layout | Extra operand |
|------|--------------------|---------------|
| omitted | Normal target-profile writeback layout | none |
| `nz2nd` | Logical ND order | none |
| `nz2dn(%loop0_src_stride)` | Logical D/N-swapped order | `%loop0_src_stride` in C0-size units |
| `nz2nz(%split)` | NZ-style destination | `%split`, destination split point |

`%src_stride` is measured in C0-size units and advances the accumulator source
between adjacent source groups selected by the layout mode. `%dst_stride` is
measured in destination elements and advances the destination row/group
selected by the layout mode. In `loop3`, `%src_stride3` is in C0-size units and
`%dst_stride3` is in destination elements.

Reference semantics:

```text
repeat_count = loop3.count if loop3 is present else 1

for r in 0 .. repeat_count-1:
  src_r = src + r * loop3.src_stride * 32
  dst_r = dst + r * loop3.dst_stride * sizeof(dst_element)

  for m in 0 .. M-1:
    for n in 0 .. N-1:
      x = read_acc_logical(src_r, m, n, src_stride, layout_mode)

      if pre_quant:
        x = apply_pre_quant(x, payload, mode)

      if pre_relu:
        x = apply_pre_relu(x, payload, mode)
        if clip:
          x = min(x, clip)

      y = convert_to_destination_type(x, sat_or_nosat)
      write_destination(dst_r, y, m, n, dst_stride, layout_mode)
```

When no layout clause is present, the store uses the target-profile normal
writeback layout for the destination address space. This mode performs no
explicit ND/DN/NZ layout transform; `%dst_stride` is still the destination
start-to-start stride in destination elements for the normal writeback rows or
groups.

For `nz2nd`, `write_destination` stores logical `y[m, n]` in ND order. For
`nz2dn`, it stores the same logical result with the D/N dimensions swapped; the
extra `%loop0_src_stride` selects how the swapped source walk advances through
the accumulator tile. For `nz2nz`, it preserves NZ-style destination packing
and uses `%split` as the destination split point.

##### `pto.mte_l0c_l1`

- **syntax:**
```mlir
pto.mte_l0c_l1 %src, %dst, %m, %n, %src_stride, %dst_stride
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu([%payload, ]mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat | sat(preserve_nan) | nosat]?
  : ...
```
- **semantics:** FIXPIPE writeback from `l0c` to L1 `l1`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | buffer-like | Accumulator source in `l0c` |
| `%dst` | buffer-like | L1 destination in `l1` |
| `%m` | i64 | Logical M element count |
| `%n` | i64 | Logical N element count |
| `%src_stride` | i64 | Source stride in C0-size units |
| `%dst_stride` | i64 | Destination stride in destination elements |
| optional clauses | - | See [FIXPIPE Common Clauses](#fixpipe-common-clauses) and [FIXPIPE Layout Model](#fixpipe-layout-model) |

**Constraints:**

- Clauses must appear in canonical order:
  `unit_flag` -> `pre_quant` -> `pre_relu` -> layout -> `loop3` -> `sat`/`nosat`.
- `pre_quant` requires payload and mode together.
- Vector `pre_quant` modes require a `fb` pointer with `f16`, `bf16`, or
  `f32` element type.
- Scalar `pre_quant` modes require an `f16`, `bf16`, or `f32` scalar payload.
- `pre_quant` source element type must be `f32` or `i32`, and the selected
  mode must be compatible with the source and destination element types.
- `no_relu` and `normal_relu` do not accept a payload.
- `scalar_relu` requires an `f16`, `bf16`, or `f32` scalar payload.
- `vector_relu` requires a `fb` pointer with `f16`, `bf16`, or `f32`
  element type.
- `clip` can appear only inside `pre_relu(...)`.
- `clip` is supported for destination `f16`, `ui8`, and signed/signless
  4/8/16-bit integer destinations. The clip payload must match the destination
  family: `f16` for f16, 16-bit unsigned/signless payload for `ui8`, and
  signed/signless `i4/i8/i16` for signed integer destinations.
- `nz2dn` requires `%loop0_src_stride`; `nz2nd` and `nz2nz` do not accept it.
- `unit_flag` must be omitted when `nz2dn(%loop0_src_stride)` uses a value
  other than 1.
- `nz2nz` requires `f32` destination element type and does not accept `loop3`.
- `sat`, `sat(preserve_nan)`, and `nosat` are mutually exclusive.

**Example:**

```mlir
pto.mte_l0c_l1 %l0c, %l1_out, %c16_i64, %c32_i64, %c16_i64, %c32_i64,
  pre_quant(%c1_f32, mode = qf322f16_pre_scalar),
  pre_relu(%c025_f32, mode = scalar_relu),
  nz2nd,
  sat
  : !pto.ptr<f32, l0c>, !pto.ptr<f16, l1>, i64, i64, i64, i64, f32, f32
```

---

##### `pto.mte_l0c_gm`

- **syntax:**
```mlir
pto.mte_l0c_gm %src, %dst, %m, %n, %src_stride, %dst_stride, %sid, %l2_cache_ctrl
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu([%payload, ]mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat | sat(preserve_nan) | nosat]?
    [, atomic(type = <atomic_type>, op = <atomic_op>)]?
  : ...
```
- **semantics:** FIXPIPE writeback from `l0c` to GM. The data transform clauses
  match `pto.mte_l0c_l1`; GM-specific operands select the GM write path and
  optional atomic update behavior.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src`, `%m`, `%n`, `%src_stride` | - | Same as `pto.mte_l0c_l1` |
| `%dst` | buffer-like | GM destination |
| `%dst_stride` | i64 | GM destination stride in destination elements |
| `%sid` | i64 | GM stream/session hint for the OUT/GM path; does not change written values |
| `%l2_cache_ctrl` | i64 | GM store cache hint; does not change written values |
| `atomic(type = ..., op = ...)` | clause | Optional GM read-modify-write |
| other optional clauses | - | Same as `pto.mte_l0c_l1` |

`%sid` and `%l2_cache_ctrl` affect the memory path only. They do not change
the logical result, destination layout, numeric conversion, or atomic
operation. For target-profile GM writeback, constant `%sid` values must be in
`[0, 3]`; use `0` unless the surrounding memory system deliberately assigns a
different stream/session hint. Constant `%l2_cache_ctrl` values must fit in the
target cache-control hint range `[0, 15]`.

`atomic(type = T, op = add|max|min)` performs an atomic read-modify-write at
each GM destination element. `add` accumulates the converted value into the
existing GM value. `max` and `min` compare using `T` and write the selected
value. Supported atomic types are `f32`, `f16`, `bf16`, `s32`, `s16`, and `s8`.

**Constraints:**

- `atomic(...)` is valid only on `pto.mte_l0c_gm`.
- `atomic` requires both `type` and `op`.
- Atomic op values are `add`, `max`, and `min`.
- If `%sid` or `%l2_cache_ctrl` is a constant, it must be in the target range
  described above.
- Other constraints match `pto.mte_l0c_l1`.

**Example:**

```mlir
pto.mte_l0c_gm %l0c, %out, %c16_i64, %c32_i64, %c16_i64, %c32_i64,
  %c0_i64, %c0_i64,
  pre_quant(%c1_f32, mode = qf322f16_pre_scalar),
  nz2nd,
  atomic(type = f16, op = add)
  : !pto.ptr<f32, l0c>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64, i64, f32
```

---

##### `pto.mte_l0c_ub`

- **syntax:**
```mlir
pto.mte_l0c_ub %src, %dst, %m, %n, %src_stride, %dst_stride,
    dst_mode(%sub_blockid | split_m | split_n)
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu([%payload, ]mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat | sat(preserve_nan) | nosat]?
  : ...
```
- **semantics:** FIXPIPE writeback from `l0c` to UB. The data transform clauses
  match `pto.mte_l0c_l1`; UB-specific operands select single or dual destination
  behavior.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src`, `%m`, `%n`, `%src_stride` | - | Same as `pto.mte_l0c_l1` |
| `%dst` | buffer-like | UB destination |
| `%dst_stride` | i64 | UB destination stride in destination elements |
| `dst_mode(%sub_blockid)` | i64 operand | Single-destination mode. `%sub_blockid` selects UB sub-block `0` or `1`; the value may be dynamic. |
| `dst_mode(split_m)` | keyword | Dual-destination mode that splits the logical tile along M. |
| `dst_mode(split_n)` | keyword | Dual-destination mode that splits the logical tile along N. |
| optional clauses | - | Same as `pto.mte_l0c_l1`; `atomic(...)` is not supported |

In `dst_mode(%sub_blockid)`, the whole logical result tile is written to the
selected UB sub-block using the selected layout mode and `%dst` as that
sub-block's base destination pointer.

In `dst_mode(split_m)`, the logical tile is split into two M ranges:
`[0, m/2)` and `[m/2, m)`. The first range is written to UB sub-block 0 and the
second range is written to UB sub-block 1. Each sub-block sees its own
destination origin at `%dst`; within each sub-block, the written logical tile
has shape `(m / 2) x n`.

In `dst_mode(split_n)`, the logical tile is split into two N ranges:
`[0, n/2)` and `[n/2, n)`. The first range is written to UB sub-block 0 and the
second range is written to UB sub-block 1. Each sub-block sees its own
destination origin at `%dst`; within each sub-block, the written logical tile
has shape `m x (n / 2)`.

**Constraints:**

- `atomic(...)` is not supported.
- `dst_mode(%sub_blockid)` writes the whole logical tile to one UB sub-block.
  Runtime `%sub_blockid` values must be `0` or `1`; constant values are checked
  statically when available.
- `dst_mode(split_m)` splits the logical tile along M into two equal-height
  sub-block regions. `%m` must be even; each sub-block receives an
  `(m / 2) x n` tile.
- `dst_mode(split_n)` splits the logical tile along N into two equal-width
  sub-block regions. `%n` must be a multiple of 32; each sub-block receives an
  `m x (n / 2)` tile.
- Dual-destination split modes are valid only for target-supported normal or
  `nz2nd` writeback cases with pre-quant, pre-ReLU/clip, and other transform
  clauses omitted.
- Other constraints match `pto.mte_l0c_l1`.

**Example:**

```mlir
pto.mte_l0c_ub %l0c, %ub_out, %c16_i64, %c32_i64, %c16_i64, %c32_i64,
  dst_mode(%c1_i64),
  nz2nd
  : !pto.ptr<f32, l0c>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64
```

---

#### Typical Usage / Patterns

A common Cube matmul flow is:

```text
GM row/column-major data
  -> pto.mte_gm_l1_frac or pto.mte_gm_l1 into L1 `l1`
  -> pto.mte_l1_l0a / pto.mte_l1_l0b into `l0a`/`l0b` tiles
  -> pto.mad* produces `l0c` tile
  -> pto.mte_l0c_l1* writes L1, GM, or UB with optional FIXPIPE transforms
```

For MX matmul, load the data tiles and the matching MX scale payloads before
calling `pto.mad_mx*`:

```text
left data tile + left scale payload
right data tile + right scale payload
  -> pto.mad_mx*
```

For bias matmul, prepare the bias vector in `bt` with `pto.mte_l1_bt` before the
`pto.mad_bias` / `pto.mad_mx_bias` consumer.

<a id="micro-17-simt"></a>

### 17. SIMT Ops

> **Category:** SIMT scalar execution, lane collectives, scalar memory, and
> memory-reduction operations
> **Pipeline:** Vector-side SIMT execution

SIMT ops are scalar operations executed by a group of workitems. A VPTO SIMT
program has an outer `pto.aicore` kernel that configures a VF subtask launch and
calls a SIMT body function marked with `pto.simt_entry`. The body is executed by
the logical workitems in the configured `dim_x * dim_y * dim_z` launch space.

---

#### Common SIMT Execution Model

- The outer non-SIMT kernel configures launch dimensions with
  `pto.store_vfsimt_info`.
- The SIMT body is a normal `func.func` with the `pto.simt_entry` attribute.
- Each active workitem executes the same SIMT body with its own scalar SSA
  values, thread coordinates, lane id, and lane-mask state.
- SIMT scalar memory offsets are element offsets, not byte offsets.
- Vector-register ops such as `pto.vlds`, `pto.vadd`, and `pto.vsts` belong to
  normal vector code, not to the SIMT body.

Example SIMT body:

```mlir
func.func @body(%dst: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {
  %tx = pto.get_tid_x : i32
  %idx = arith.index_castui %tx : i32 to index
  pto.store %tx, %dst[%idx] : !pto.ptr<i32, ub>, i32
  return
}
```

##### Supported PTO SIMT Operation Surface

The current PTO SIMT surface supports these operation families:

| Family | Ops |
|--------|-----|
| Launch configuration | `pto.store_vfsimt_info`, `pto.simt_launch` |
| Thread and lane queries | `pto.get_tid_x`, `pto.get_tid_y`, `pto.get_tid_z`, `pto.get_block_dim_x`, `pto.get_block_dim_y`, `pto.get_block_dim_z`, `pto.get_grid_dim_x`, `pto.get_grid_dim_y`, `pto.get_grid_dim_z`, `pto.get_block_idx_x`, `pto.get_block_idx_y`, `pto.get_block_idx_z`, `pto.get_veccoreid`, `pto.get_clock32`, `pto.get_clock64`, `pto.get_laneid`, `pto.get_lanemask_eq`, `pto.get_lanemask_le`, `pto.get_lanemask_lt`, `pto.get_lanemask_ge`, `pto.get_lanemask_gt` |
| Lane collectives | `pto.vote_all`, `pto.vote_any`, `pto.vote_uni`, `pto.vote_ballot`, `pto.shuffle_idx`, `pto.shuffle_up`, `pto.shuffle_down`, `pto.shuffle_bfly`, `pto.redux_add`, `pto.redux_max`, `pto.redux_min` |
| Scalar memory | `pto.load`, `pto.store`, `pto.ldg`, `pto.stg` |
| Atomic memory | `pto.atomic_exch`, `pto.atomic_add`, `pto.atomic_sub`, `pto.atomic_min`, `pto.atomic_max`, `pto.atomic_and`, `pto.atomic_or`, `pto.atomic_xor`, `pto.atomic_cas` |
| Scalar math | `pto.prmt`, `pto.mulhi`, `pto.mul_i32toi64`, `pto.absf`, `pto.sqrt`, `pto.exp`, `pto.log`, `pto.pow`, `pto.ceil`, `pto.floor`, `pto.rint`, `pto.round`, `pto.fmin`, `pto.fmax`, `pto.fma` |
| Conversion | `pto.convert` |
| Entry synchronization and state | `pto.syncthreads`, `pto.threadfence`, `pto.threadfence_block`, `pto.keep`, `pto.resume` |

One optional function attribute may be attached to a `pto.simt_entry`
function:

| Function attribute | Type | Default | Meaning |
|--------------------|------|---------|---------|
| `pto.simt_max_threads` | signless `i32` integer attribute | `1024` | Compile-time launch envelope. It should cover the largest `dim_x * dim_y * dim_z` launch count used for this entry. |

`pto.simt_max_threads` may only appear on functions that also carry
`pto.simt_entry`. It must be a positive `i32` value no greater than 2048. The
thread envelope determines the emitted scalar register budget:

| `pto.simt_max_threads` | Emitted `simt-max-registers` |
|------------------------|------------------------------|
| `1` to `256` | `128` |
| `257` to `512` | `64` |
| `513` to `1024` | `32` |
| `1025` to `2048` | `16` |

The register budget is derived automatically and is not independently
configurable. `pto.simt_max_threads` does not launch work by itself; the actual
workitem count comes from `pto.store_vfsimt_info` or `pto.simt_launch`.

```mlir
func.func @body(%dst: !pto.ptr<i32, ub>)
    attributes {pto.simt_entry,
                pto.simt_max_threads = 256 : i32} {
  return
}
```

---

#### Launch Configuration

##### `pto.store_vfsimt_info`

- **syntax:** `pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32`
- **semantics:** Configure the launch descriptor consumed by a subsequent SIMT
  entry call sequence in the current outer vector-side kernel.

```text
configured_dim_z = dim_z
configured_dim_y = dim_y
configured_dim_x = dim_x
logical_workitems = dim_x * dim_y * dim_z
call one or more simt_entry_body(...) functions
```

- **inputs:** `%dim_z`, `%dim_y`, and `%dim_x` are `i32` workitem counts in
  `z, y, x` order.
- **outputs:** None.
- **constraints and limitations:** This op belongs in the outer non-SIMT
  caller and must not appear inside a function marked with `pto.simt_entry`.
  SIMT entry calls that use the descriptor must be dominated by the matching
  launch configuration. On the current SIMT VF model, the launch count is
  bounded by 2048.
  If `pto.simt_max_threads` is present on the callee, it should be at least the
  largest launch count used for that callee.

Typical outer-kernel pattern:

```mlir
%dim_z = arith.constant 1 : i32
%dim_y = arith.constant 1 : i32
%dim_x = arith.constant 32 : i32
pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32
func.call @body(%ub_out) : (!pto.ptr<i32, ub>) -> ()
```

##### `pto.simt_launch`

- **syntax:** `pto.simt_launch @body<<<%dim_x, %dim_y, %dim_z>>>(%arg0, ...) : (arg_types...) -> ()`
- **semantics:** Launch the SIMT body `@body` using the workitem dimensions
  `%dim_x`, `%dim_y`, and `%dim_z`. The dimension order follows the launch-site
  order `x, y, z`; each active workitem in the body observes coordinates in the
  ranges `tid_x in [0, dim_x)`, `tid_y in [0, dim_y)`, and
  `tid_z in [0, dim_z)`.
- **inputs:** `%dim_x`, `%dim_y`, and `%dim_z` are `i32` workitem counts. The
  remaining operands are passed to the SIMT body and must match the callee
  function signature.
- **outputs:** None. The SIMT body must return no values.
- **constraints and limitations:** The callee must be a `func.func` marked with
  `pto.simt_entry`. The launch op belongs in the outer non-SIMT caller and must
  not appear inside a function marked with `pto.simt_entry`. The launch count is
  `dim_x * dim_y * dim_z` and is bounded by the same limits as
  `pto.store_vfsimt_info`.

Example launch-site pattern:

```mlir
%dim_x = arith.constant 32 : i32
%dim_y = arith.constant 1 : i32
%dim_z = arith.constant 1 : i32
pto.simt_launch @body<<<%dim_x, %dim_y, %dim_z>>>(%ub_out)
  : (!pto.ptr<i32, ub>) -> ()
```

---

#### Thread and Lane Query Ops

Thread and lane query ops are nullary pure scalar ops. They return the value
visible to the current workitem.

##### `pto.get_tid_x` / `pto.get_tid_y` / `pto.get_tid_z`

- **syntax:** `%tx = pto.get_tid_x : i32`
- **semantics:** Return the current workitem coordinate in the selected launch
  dimension.

```text
0 <= tid_x < dim_x
0 <= tid_y < dim_y
0 <= tid_z < dim_z
linear_tid = tid_x + dim_x * (tid_y + dim_y * tid_z)
```

- **inputs:** None.
- **outputs:** One `i32` coordinate.
- **constraints and limitations:** Use these coordinates for logical indexing.
  They are launch coordinates, not necessarily the same value as the physical
  lane id.

##### `pto.get_block_dim_x` / `pto.get_block_dim_y` / `pto.get_block_dim_z`

- **syntax:** `%v = pto.get_block_dim_x : i32`
- **semantics:** Return the block dimension visible to the current workitem in
  the selected dimension.
- **inputs:** None.
- **outputs:** One `i32` block dimension.
- **constraints and limitations:** For single-block VF launches, block
  dimensions match the configured launch dimensions.

##### `pto.get_grid_dim_x` / `pto.get_grid_dim_y` / `pto.get_grid_dim_z`

- **syntax:** `%v = pto.get_grid_dim_x : i32`
- **semantics:** Return the grid dimension visible to the current workitem in
  the selected dimension.
- **inputs:** None.
- **outputs:** One `i32` grid dimension.
- **constraints and limitations:** Use grid dimensions with block dimensions and
  block indices when deriving global workitem coordinates.

##### `pto.get_block_idx_x` / `pto.get_block_idx_y` / `pto.get_block_idx_z`

- **syntax:** `%v = pto.get_block_idx_x : i32`
- **semantics:** Return the current block index in the selected dimension.
- **inputs:** None.
- **outputs:** One `i32` block index.
- **constraints and limitations:** For single-block VF launches, block indices
  are normally zero.

##### `pto.get_veccoreid`

- **syntax:** `%core = pto.get_veccoreid : i32`
- **semantics:** Return the vector-core id visible to the current workitem.
- **inputs:** None.
- **outputs:** One `i32` vector-core id.
- **constraints and limitations:** The value is target scoped; use it only when
  the algorithm intentionally depends on the executing vector core.

##### `pto.get_clock32` / `pto.get_clock64`

- **syntax:** `%c32 = pto.get_clock32 : i32`, `%c64 = pto.get_clock64 : i64`
- **semantics:** Sample the target clock counter visible to the current
  workitem.
- **inputs:** None.
- **outputs:** `pto.get_clock32` returns `i32`; `pto.get_clock64` returns `i64`.
- **constraints and limitations:** Use `get_clock64` when 32-bit wraparound
  could make elapsed-time comparisons ambiguous.

##### `pto.get_laneid`

- **syntax:** `%lane = pto.get_laneid : i32`
- **semantics:** Return the physical SIMT lane id for the current workitem.
- **inputs:** None.
- **outputs:** One `i32` lane id.
- **constraints and limitations:** Use lane id for lane-mask, vote, shuffle,
  and reduction logic. Use `get_tid_x/y/z` for logical tensor indexing.

##### `pto.get_lanemask_eq` / `pto.get_lanemask_le` / `pto.get_lanemask_lt` / `pto.get_lanemask_ge` / `pto.get_lanemask_gt`

- **syntax:** `%mask = pto.get_lanemask_lt : i32`
- **semantics:** Return a 32-bit mask derived from the current lane id.

```text
get_lanemask_eq = 1 << laneid
get_lanemask_lt = bits for lanes 0 .. laneid-1
get_lanemask_le = bits for lanes 0 .. laneid
get_lanemask_gt = bits for lanes laneid+1 .. subgroup_width-1
get_lanemask_ge = bits for lanes laneid .. subgroup_width-1
```

- **inputs:** None.
- **outputs:** One `i32` mask value.
- **constraints and limitations:** The mask is indexed by physical lane id.

---

#### Vote Ops

Vote ops consume one `i1` predicate from each participating active lane and
return a collective result to each participating active lane.

##### `pto.vote_all` / `pto.vote_any` / `pto.vote_uni` / `pto.vote_ballot`

- **syntax:**
```mlir
%all = pto.vote_all %pred : i1 -> i1
%any = pto.vote_any %pred : i1 -> i1
%uni = pto.vote_uni %pred : i1 -> i1
%bits = pto.vote_ballot %pred : i1 -> i32
```
- **semantics:**
```text
active = participating active lanes
vote_all    = forall lane in active: pred[lane]
vote_any    = exists lane in active: pred[lane]
vote_uni    = all pred[lane] values in active are equal
vote_ballot = bitset of lanes in active where pred[lane] is true
```
- **inputs:** `%pred` is the current lane's `i1` predicate.
- **outputs:** `vote_all`, `vote_any`, and `vote_uni` return `i1`.
  `vote_ballot` returns an `i32` lane bit mask.
- **constraints and limitations:** Inactive lanes do not contribute predicate
  values to the vote.

Example:

```mlir
%lane = pto.get_laneid : i32
%one = arith.constant 1 : i32
%low = arith.andi %lane, %one : i32
%is_odd = arith.cmpi eq, %low, %one : i32
%odd_mask = pto.vote_ballot %is_odd : i1 -> i32
```

---

#### Shuffle Ops

Shuffle ops exchange values between participating lanes. The source value and
result have the same type.

##### `pto.shuffle_idx`

- **syntax:** `%r = pto.shuffle_idx %value, %index {width = 16 : i32} : T, i32 -> T`
- **semantics:** Read `%value` from absolute `%index` inside the current
  subgroup.
- **inputs:** `%value` is the current lane's payload. `%index` is the
  source lane index inside the subgroup.
- **outputs:** `%r` is the selected source lane's value.
- **constraints and limitations:** `T` is `i32`, `i64`, `f16`, `f32`, or
  `vector<2xf16>`. `%index` is `i32`. `width` is `16` or `32` and defaults to
  `32`.

##### `pto.shuffle_up` / `pto.shuffle_down` / `pto.shuffle_bfly`

- **syntax:**
```mlir
%u = pto.shuffle_up %value, %offset : T, i32 -> T
%d = pto.shuffle_down %value, %offset : T, i32 -> T
%b = pto.shuffle_bfly %value, %mask : T, i32 -> T
```
- **semantics:**
```text
group_base = floor(lane / width) * width
local_lane = lane - group_base
shuffle_up:   source = group_base + local_lane - offset
shuffle_down: source = group_base + local_lane + offset
shuffle_bfly: source = group_base + (local_lane xor mask)
result        = value[source] when source is a valid participating lane
```
- **inputs:** `%value` is the current lane's payload. `%offset` is the
  relative lane distance. `%mask` is the XOR mask for butterfly selection.
- **outputs:** The selected source lane's value.
- **constraints and limitations:** `T` is `i32`, `i64`, `f16`, `f32`, or
  `vector<2xf16>`. Control operands are `i32`. The optional `width` attribute is
  `16` or `32` and defaults to `32`. Out-of-range source-lane behavior is
  target-scoped and should not be used for portable algorithms.

---

#### Lane Redux Ops

Redux ops reduce one scalar value from each participating active lane and
return the reduction result to each participating active lane.

##### `pto.redux_add` / `pto.redux_max` / `pto.redux_min`

- **syntax:**
```mlir
%sum_i = pto.redux_add %v signed : i32 -> i32
%max_u = pto.redux_max %v unsigned : i32 -> i32
%sum_f = pto.redux_add %f : f32 -> f32
```
- **semantics:**
```text
redux_add = sum(value[lane] for lane in active)
redux_max = max(value[lane] for lane in active)
redux_min = min(value[lane] for lane in active)
result    = the selected reduction value in every participating active lane
```
- **inputs:** `%value` is `i32`, `f16`, or `f32`.
- **outputs:** The result type matches `%value`.
- **constraints and limitations:** Floating-point forms do not accept
  signedness. For `i32`, `pto.redux_max` and `pto.redux_min` require explicit
  `signed` or `unsigned`. `pto.redux_add` accepts signedness for consistency
  with integer authoring, but addition has the same two's-complement bit result
  for signed and unsigned inputs.

---

#### Common SIMT Memory Attributes

**L1 cache.**

`l1cache(cache)` and `l1cache(uncache)` are accepted on GM scalar `pto.ldg` /
`pto.stg` forms.

| Attribute | Meaning |
|-----------|---------|
| `l1cache(cache)` | Request cacheable GM scalar access |
| `l1cache(uncache)` | Request uncacheable GM scalar access |

The L1 cache clause selects the GM access path. It does not change the scalar value
being loaded or stored.

**L2 cache.**

L2 cache clauses select the memory hierarchy behavior attached to GM `pto.ldg`,
GM `pto.stg`, and atomic ops. They do not select the
mathematical operation; the op mnemonic still determines the load, store,
or atomic update.

Load `l2cache(...)` uses these tokens:

| Token | Meaning |
|-------|---------|
| `nmfv` | Normal allocation, first-victim replacement priority |
| `nmlv` | Normal allocation, last-victim replacement priority |
| `nmprs` | Normal allocation, persistent cache residency hint |
| `nmpref` | Normal allocation, prefetch-oriented hint |
| `nakeep` | Not-allocate, keep existing cache line state |
| `naclean` | Not-allocate, clean cache line state |
| `nadrop` | Not-allocate, drop cache line state |
| `idsfv` | Inter-domain-share, first-victim replacement priority |
| `idslv` | Inter-domain-share, last-victim replacement priority |
| `idsprs` | Inter-domain-share, persistent cache residency hint |
| `idspref` | Inter-domain-share, prefetch-oriented hint |
| `exfv` | Exclusive, first-victim replacement priority |
| `exlv` | Exclusive, last-victim replacement priority |
| `exprs` | Exclusive, persistent cache residency hint |
| `expref` | Exclusive, prefetch-oriented hint |

Store and atomic `l2cache(...)` uses these tokens:

| Token | Meaning |
|-------|---------|
| `nmfv` | Normal allocation, first-victim replacement priority |
| `nmlv` | Normal allocation, last-victim replacement priority |
| `nmprs` | Normal allocation, persistent cache residency hint |
| `nmred` | Normal allocation, reduce-oriented update hint |
| `naci` | Not-allocate, clean-invalid cache line state |
| `napw` | Not-allocate, clean pre-writeback cache line state |
| `napi` | Not-allocate, pre-invalid cache line state |
| `nared` | Not-allocate, reduce-oriented update hint |
| `wbhfv` | Write-back-home, first-victim replacement priority |
| `wbhlv` | Write-back-home, last-victim replacement priority |
| `wbhprs` | Write-back-home, persistent cache residency hint |
| `wbhred` | Write-back-home, reduce-oriented update hint |
| `wtsfv` | Write-through-share, first-victim replacement priority |
| `wtslv` | Write-through-share, last-victim replacement priority |
| `wtsprs` | Write-through-share, persistent cache residency hint |
| `wtsred` | Write-through-share, reduce-oriented update hint |

For scalar GM `pto.ldg` / `pto.stg` and atomic syntax, write
`l2cache(...)`. Omitted `l2cache(...)` means `l2cache(nmfv)`. On `pto.ldg` /
`pto.stg`, omitted `l1cache(...)` means `l1cache(cache)`.

In syntax summaries, `<ld-l2cache>` means one token from the load L2 cache
table, `<st-l2cache>` means one token from the store/atomic L2 cache table,
`?` marks an optional clause, and `signedness?` means either
`signed`, `unsigned`, or no signedness clause.

---

#### SIMT Scalar Memory Ops

##### `pto.load`

- **syntax:** `%value = pto.load %ptr[%offset] : !pto.ptr<T, space> -> T`
- **accepted forms:**

```mlir
// Plain scalar load. Uses the ordinary scalar memory path.
%value = pto.load %ptr[%offset] : !pto.ptr<T, space> -> T
```

- **semantics:** Load one scalar element from `%ptr + %offset`.

```text
effective_element = ptr + offset
result = memory[effective_element]
```

- **inputs:** `%ptr` is a `!pto.ptr<T, space>` or memref. `%offset` is an
  `index` element offset, not a byte offset.
- **outputs:** One scalar value of type `T`.
- **constraints and limitations:** The result type must match the pointer
  element type. This op does not accept cache-control clauses; use `pto.ldg`
  for GM scalar loads that need `l1cache(...)` or `l2cache(...)`.

##### `pto.store`

- **syntax:** `pto.store %value, %ptr[%offset] : !pto.ptr<T, space>, T`
- **accepted forms:**

```mlir
// Plain scalar store. Uses the ordinary scalar memory path.
pto.store %value, %ptr[%offset] : !pto.ptr<T, space>, T
```

- **semantics:** Store one scalar element to `%ptr + %offset`.

```text
effective_element = ptr + offset
memory[effective_element] = value
```

- **inputs:** `%value` is the scalar element to write. `%ptr` is a
  `!pto.ptr<T, space>` or memref. `%offset` is an `index` element offset.
- **outputs:** None.
- **constraints and limitations:** `%value` type must match the pointer element
  type. This op does not accept cache-control clauses; use `pto.stg` for GM
  scalar stores that need `l1cache(...)` or `l2cache(...)`.

##### `pto.ldg`

- **syntax:** `%value = pto.ldg %ptr[%offset] l1cache(...)? l2cache(...)? attr-dict : !pto.ptr<T, gm> -> T`
- **accepted forms:**

```mlir
// GM load with default cache controls: l1cache(cache) and l2cache(nmfv).
%value = pto.ldg %gm[%offset] : !pto.ptr<T, gm> -> T

// GM load with an explicit L1 cache control.
%value = pto.ldg %gm[%offset] l1cache(cache) : !pto.ptr<T, gm> -> T

// GM load with explicit L1 and L2 cache controls.
%value = pto.ldg %gm[%offset] l1cache(uncache) l2cache(nmpref) : !pto.ptr<T, gm> -> T
```

- **semantics:** Load one element from GM at `%ptr + %offset` using the
  selected cache controls.
- **inputs:** `%ptr` is a `!pto.ptr<T, gm>`. `%offset` is an `index` element
  offset, not a byte offset.  For ``!pto.ptr<vector<2xf32>, gm>``, offset 1
  advances by 8 bytes (2 × sizeof(f32)).
- **attributes:** `l1cache` may be `l1cache(cache)` or `l1cache(uncache)` and
  defaults to `cache`. `l2cache(...)` uses the load L2 cache table and defaults
  to `nmfv`.
- **outputs:** One value of type `T`.
- **constraints and limitations:** `pto.ldg` supports 8/16/32/64-bit integer
  values, `f16`, `bf16`, `f32`, `f64`, `fp8`, `hif8`, and packed vectors
  `vector<2xf16>`, `vector<2xbf16>`, `vector<2xf32>`,
  `vector<2/4/8xf8E4M3FN>`, `vector<2/4/8xf8E5M2>`,
  `vector<2xi8>`, `vector<2xi16>`, `vector<2xi32>`,
  and `!pto.hif8x2`.  Vector loads from GM
  use the same-width load path as scalars (e.g. ``vector<2xf32>`` uses a 64-bit
  GM load) and reinterpret the loaded bits as the requested vector type.  The
  effective address for ``vector<2xf32>`` must satisfy 8-byte alignment
  (enforced by the call-site contract; the op does not carry an alignment
  operand).

##### `pto.stg`

- **syntax:** `pto.stg %value, %ptr[%offset] l1cache(...)? l2cache(...)? attr-dict : !pto.ptr<T, gm>, T`
- **accepted forms:**

```mlir
// GM store with default cache controls: l1cache(cache) and l2cache(nmfv).
pto.stg %value, %gm[%offset] : !pto.ptr<T, gm>, T

// GM store with an explicit L1 cache control.
pto.stg %value, %gm[%offset] l1cache(cache) : !pto.ptr<T, gm>, T

// GM store with explicit L1 and L2 cache controls.
pto.stg %value, %gm[%offset] l1cache(uncache) l2cache(wtsred) : !pto.ptr<T, gm>, T
```

- **semantics:** Store one element to GM at `%ptr + %offset` using the
  selected cache controls.
- **inputs:** `%value` is the element to write. `%ptr` is a
  `!pto.ptr<T, gm>`. `%offset` is an `index` element offset
  (same element-level semantics as `pto.ldg`).
- **attributes:** `l1cache` may be `l1cache(cache)` or `l1cache(uncache)` and
  defaults to `cache`. `l2cache(...)` uses the store/atomic L2 cache table and
  defaults to `nmfv`.
- **outputs:** None.
- **constraints and limitations:** `%value` type must match the pointer element
  type.  Supported types and alignment requirements are the same as `pto.ldg`
  (see above).

Example:

```mlir
%tx = pto.get_tid_x : i32
%idx = arith.index_castui %tx : i32 to index
%loaded = pto.load %gm[%idx] : !pto.ptr<i32, gm> -> i32
%sum = arith.addi %loaded, %tx : i32
pto.store %sum, %gm[%idx] : !pto.ptr<i32, gm>, i32
```

---

#### Atomic Memory Ops

Atomic ops update one scalar or supported packed memory location and return the
old value observed by the current workitem. The read, update, and returned old
value form one atomic read-modify-write at `%ptr`.

##### `pto.atomic_exch` / `pto.atomic_add` / `pto.atomic_sub`

- **syntax:**
```mlir
%old = pto.atomic_exch %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
%old = pto.atomic_add  %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
%old = pto.atomic_sub  %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
```
- **accepted forms:**

```mlir
// Signed integer atomic with default nmfv L2 cache.
%old = pto.atomic_add %ptr, %value signed : !pto.ptr<i32, space>, i32 -> i32

// Signed integer atomic with an explicit store/atomic L2 cache.
%old = pto.atomic_add %ptr, %value l2cache(wtsred) signed : !pto.ptr<i32, space>, i32 -> i32

// Floating-point atomic. Floating-point atomics do not take signedness.
%old = pto.atomic_add %ptr, %value l2cache(nmfv) : !pto.ptr<f32, space>, f32 -> f32

// Packed two-lane atomic. Packed atomics do not take signedness.
%old = pto.atomic_add %ptr, %value : !pto.ptr<vector<2xf16>, space>, vector<2xf16> -> vector<2xf16>
```
- **semantics:**
```text
old = *ptr
atomic_exch: *ptr = value
atomic_add:  *ptr = old + value
atomic_sub:  *ptr = old - value
return old
```
- **inputs:** `%ptr` is `!pto.ptr<T, gm>` or `!pto.ptr<T, ub>`. `%value` is
  `i32`, `i64`, `f16`, `bf16`, `f32`, `vector<2xf16>`, or
  `vector<2xbf16>`.
  `l2cache(...)` selects the store/atomic L2 cache control and defaults to
  `nmfv`.
- **outputs:** `%old` has the same type as `%value`. For packed
  `vector<2xf16>` and `vector<2xbf16>` atomics on beta.1, `%old` must be left
  unused; beta.1 can compile the packed atomic update but cannot compile a
  consumed packed old-value result.
- **constraints and limitations:** UB-space atomics do not support `i64`.
  Floating-point and packed atomics do not accept `signed` or `unsigned`.
  Packed atomics must be placed inside a `pto.simt_entry` function on beta.1.

##### `pto.atomic_min` / `pto.atomic_max`

- **syntax:**
```mlir
%old = pto.atomic_min %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
%old = pto.atomic_max %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
```
- **accepted forms:**

```mlir
// Signed integer comparison.
%old = pto.atomic_min %ptr, %value signed : !pto.ptr<i32, space>, i32 -> i32

// Unsigned integer comparison.
%old = pto.atomic_min %ptr, %value unsigned : !pto.ptr<i32, space>, i32 -> i32

// Floating-point comparison. Floating-point atomics do not take signedness.
%old = pto.atomic_min %ptr, %value l2cache(nmlv) : !pto.ptr<f32, space>, f32 -> f32

// Packed two-lane comparison.
%old = pto.atomic_min %ptr, %value : !pto.ptr<vector<2xbf16>, space>, vector<2xbf16> -> vector<2xbf16>
```
- **semantics:**
```text
old = *ptr
atomic_min: *ptr = min(old, value)
atomic_max: *ptr = max(old, value)
return old
```
- **inputs:** Same as `pto.atomic_add`. For integer values, `signed` or
  `unsigned` selects the comparison interpretation.
- **outputs:** `%old` has the same type as `%value`. For packed
  `vector<2xf16>` and `vector<2xbf16>` atomics on beta.1, `%old` must be left
  unused; beta.1 can compile the packed atomic update but cannot compile a
  consumed packed old-value result.
- **constraints and limitations:** Floating-point and packed atomics do not
  accept signedness. UB-space atomics do not support `i64`. Packed atomics
  must be placed inside a `pto.simt_entry` function on beta.1.

##### `pto.atomic_and` / `pto.atomic_or` / `pto.atomic_xor`

- **syntax:**
```mlir
%old = pto.atomic_and %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
%old = pto.atomic_or  %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
%old = pto.atomic_xor %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
```
- **accepted forms:**

```mlir
// Unsigned bitwise atomic with default nmfv L2 cache.
%old = pto.atomic_and %ptr, %value unsigned : !pto.ptr<i32, space>, i32 -> i32

// Signedness is accepted for integer authoring consistency; the bit operation
// itself is bitwise and does not reinterpret arithmetic magnitude.
%old = pto.atomic_and %ptr, %value l2cache(napw) signed : !pto.ptr<i32, space>, i32 -> i32
```
- **semantics:**
```text
old = *ptr
atomic_and: *ptr = old & value
atomic_or:  *ptr = old | value
atomic_xor: *ptr = old ^ value
return old
```
- **inputs:** `%ptr` points to an integer scalar element. `%value` is `i32` or
  `i64`.
- **outputs:** `%old` has the same type as `%value`.
- **constraints and limitations:** Bitwise atomics require integer types.
  UB-space bitwise atomics do not support `i64`.

##### `pto.atomic_cas`

- **syntax:** `%old = pto.atomic_cas %ptr, %compare, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T`
- **accepted forms:**

```mlir
// Integer CAS with default nmfv L2 cache.
%old = pto.atomic_cas %ptr, %compare, %value signed : !pto.ptr<i32, space>, i32 -> i32

// Integer CAS with an explicit store/atomic L2 cache.
%old = pto.atomic_cas %ptr, %compare, %value l2cache(wbhred) signed : !pto.ptr<i32, space>, i32 -> i32

// Floating-point CAS. Floating-point atomics do not take signedness.
%old = pto.atomic_cas %ptr, %compare, %value : !pto.ptr<f32, space>, f32 -> f32

// Packed two-lane CAS. Packed atomics do not take signedness.
%old = pto.atomic_cas %ptr, %compare, %value : !pto.ptr<vector<2xbf16>, space>, vector<2xbf16> -> vector<2xbf16>
```

- **semantics:**
```text
old = *ptr
if old == compare:
  *ptr = value
return old
```
- **inputs:** `%ptr` is the atomic address. `%compare` is the expected old
  value. `%value` is the replacement value.
- **outputs:** `%old` is the value observed before the conditional update. For
  packed `vector<2xf16>` and `vector<2xbf16>` CAS on beta.1, `%old` must be
  left unused; beta.1 can compile the packed CAS update but cannot compile a
  consumed packed old-value result.
- **constraints and limitations:** `%compare`, `%value`, pointer element type,
  and result type must match. `T` is `i32`, `i64`, `f32`,
  `vector<2xf16>`, or `vector<2xbf16>`; UB-space `i64` is not supported.
  Packed CAS must be placed inside a `pto.simt_entry` function on beta.1.

When multiple workitems target the same address, each workitem observes one
serialized old value from the total order chosen by the target. Algorithms must
not rely on any particular tie order beyond atomicity.

---

#### SIMT Scalar Math Ops

##### `pto.prmt`

- **syntax:** `%r = pto.prmt %lhs, %rhs, %selector : i32, i32, i32 -> i32`
- **semantics:** Build the `i32` result byte-by-byte from the eight source bytes
  in `%lhs:%rhs` according to `%selector`.
- **inputs:** `%lhs` and `%rhs` provide the source bytes. `%selector` selects
  which source byte is copied into each destination byte.
- **outputs:** One `i32` result.
- **constraints and limitations:** All operands and the result are `i32`.

##### `pto.mulhi`

- **syntax:**
```mlir
%s32 = pto.mulhi %lhs, %rhs signed : i32, i32 -> i32
%u32 = pto.mulhi %lhs, %rhs unsigned : i32, i32 -> i32
%s64 = pto.mulhi %lhs64, %rhs64 signed : i64, i64 -> i64
%u64 = pto.mulhi %lhs64, %rhs64 unsigned : i64, i64 -> i64
```
- **semantics:**
```text
N = bitwidth(lhs)
if signed:
  product = signed_N(lhs) * signed_N(rhs)
else:
  product = unsigned_N(lhs) * unsigned_N(rhs)
result = high_N_bits(product)
```
- **inputs:** `%lhs` and `%rhs` are scalar integer operands with the same type.
- **outputs:** One scalar integer result with the same type as the inputs.
- **attributes:** The required `signed` or `unsigned` clause selects whether
  the operands are interpreted as signed two's-complement values or unsigned
  values before forming the double-width product.
- **constraints and limitations:** The operands and result must all be `i32` or
  all be `i64`.

##### `pto.mul_i32toi64`

- **syntax:**
```mlir
%s = pto.mul_i32toi64 %lhs, %rhs signed : i32, i32 -> i64
%u = pto.mul_i32toi64 %lhs, %rhs unsigned : i32, i32 -> i64
```
- **semantics:**
```text
if signed:
  result = sign_extend_i64(lhs) * sign_extend_i64(rhs)
else:
  result = zero_extend_i64(lhs) * zero_extend_i64(rhs)
```
- **inputs:** `%lhs` and `%rhs` are `i32` scalar operands.
- **outputs:** One `i64` widened-product result.
- **attributes:** The required `signed` or `unsigned` clause selects the
  extension rule before multiplication.
- **constraints and limitations:** The operand types are fixed to `i32`, and
  the result type is fixed to `i64`.

##### `pto.absf`

- **syntax:** `%r = pto.absf %x : T -> T`
- **semantics:** Return `abs(x)`. For `vector<2xT>`, absolute value is applied
  independently to each element.
- **inputs:** `%x` is an `f32` scalar, `vector<2xf16>`, or `vector<2xbf16>`.
- **outputs:** One value with the same type as `%x`.
- **constraints and limitations:** Scalar `f16` and scalar `bf16` are not
  accepted by this op; use the packed form only for `vector<2xT>`.

##### `pto.sqrt`

- **syntax:** `%r = pto.sqrt %x : T -> T`
- **semantics:** Return `sqrt(x)`. For `vector<2xT>`, square root is applied
  independently to each element.
- **inputs:** `%x` is `f16`, `f32`, or `vector<2xf16>`.
- **outputs:** One value with the same type as `%x`.
- **constraints and limitations:** `T` is `f16`, `f32`, or `vector<2xf16>`.

##### `pto.exp`

- **syntax:** `%r = pto.exp %x : T -> T`
- **semantics:** Return the natural exponential `e ** x`. For `vector<2xT>`,
  exponentiation is applied independently to each element.
- **inputs:** `%x` is an `f16` scalar, `f32` scalar, or `vector<2xf16>`.
- **outputs:** One value with the same type as `%x`.
- **constraints and limitations:** `T` is `f16`, `f32`, or `vector<2xf16>`.
  Overflow, underflow, infinities, and NaNs follow the target floating-point
  rules.

##### `pto.log`

- **syntax:** `%r = pto.log %x : T -> T`
- **semantics:** Return the natural logarithm `ln(x)`. For `vector<2xT>`,
  logarithm is applied independently to each element.
- **inputs:** `%x` is an `f16` scalar, `f32` scalar, or `vector<2xf16>`.
- **outputs:** One value with the same type as `%x`.
- **constraints and limitations:** `T` is `f16`, `f32`, or `vector<2xf16>`.
  For real-number semantics, each element should be positive; non-positive
  inputs follow the target floating-point rules.

##### `pto.pow`

- **syntax:** `%r = pto.pow %a, %b : T, T -> T`
- **semantics:** Return `%a ** %b`. For `vector<2xT>`, power is applied
  independently to each element pair.
- **inputs:** `%a` is the base and `%b` is the exponent. Both operands have the
  same type.
- **outputs:** One value with the same type as the inputs.
- **constraints and limitations:** `T` is `f16`, `f32`, or `vector<2xf16>`.
  Exceptional inputs follow the target floating-point rules.

##### `pto.ceil`

- **syntax:** `%r = pto.ceil %x : T -> T`
- **semantics:** Return the smallest integral floating value not less than
  `%x`.
- **inputs:** `%x` is an `f16`, `bf16`, or `f32` scalar.
- **outputs:** One scalar with the same type as `%x`.
- **constraints and limitations:** `T` is `f16`, `bf16`, or `f32`.

##### `pto.floor`

- **syntax:** `%r = pto.floor %x : T -> T`
- **semantics:** Return the largest integral floating value not greater than
  `%x`.
- **inputs:** `%x` is an `f16`, `bf16`, or `f32` scalar.
- **outputs:** One scalar with the same type as `%x`.
- **constraints and limitations:** `T` is `f16`, `bf16`, or `f32`.

##### `pto.rint`

- **syntax:** `%r = pto.rint %x : T -> T`
- **semantics:** Return the integral floating value selected by the target's
  current floating rounding rule.
- **inputs:** `%x` is an `f16`, `bf16`, or `f32` scalar.
- **outputs:** One scalar with the same type as `%x`.
- **constraints and limitations:** `T` is `f16`, `bf16`, or `f32`.

##### `pto.round`

- **syntax:** `%r = pto.round %x : T -> T`
- **semantics:** Return the nearest integral floating value using the target
  round operation's tie rule.
- **inputs:** `%x` is an `f16`, `bf16`, or `f32` scalar.
- **outputs:** One scalar with the same type as `%x`.
- **constraints and limitations:** `T` is `f16`, `bf16`, or `f32`.

##### `pto.fmin`

- **syntax:** `%r = pto.fmin %a, %b : T, T -> T`
- **semantics:** Return the floating minimum of `%a` and `%b`.
- **inputs:** `%a` and `%b` have the same type.
- **outputs:** One value with the same type as the inputs.
- **constraints and limitations:** `T` is `f32`, `bf16`, `vector<2xf16>`, or
  `vector<2xbf16>`. For vector types, the minimum is computed element-wise. NaN
  handling follows the target floating-point minimum rule.

##### `pto.fmax`

- **syntax:** `%r = pto.fmax %a, %b : T, T -> T`
- **semantics:** Return the floating maximum of `%a` and `%b`.
- **inputs:** `%a` and `%b` have the same type.
- **outputs:** One value with the same type as the inputs.
- **constraints and limitations:** `T` is `f32`, `bf16`, `vector<2xf16>`, or
  `vector<2xbf16>`. For vector types, the maximum is computed element-wise. NaN
  handling follows the target floating-point maximum rule.

##### `pto.fma`

- **syntax:** `%r = pto.fma %a, %b, %acc : T, T, T -> T`
- **semantics:** Return fused `a * b + acc` with one final rounding.
- **inputs:** `%a`, `%b`, and `%acc` have the same type.
- **outputs:** One value with the same type as the inputs.
- **constraints and limitations:** `T` is `f16`, `bf16`, `f32`,
  `vector<2xf16>`, or `vector<2xbf16>`. For vector types, fused multiply-add is
  computed element-wise.

---

#### SIMT Conversion Op

##### `pto.convert`

- **syntax:** `%dst = pto.convert %src round(R) sat|nosat [signed|unsigned] : SrcType -> DstType`
- **semantics:** Convert one scalar or packed two-element value from `SrcType` to
  `DstType` using the specified rounding, saturation, and signedness controls.

```mlir
%as_f32 = pto.convert %i round(r) nosat signed : i32 -> f32
%as_i32 = pto.convert %f round(z) sat signed : f32 -> i32
%as_h2 = pto.convert %f2 round(r) nosat : vector<2xf32> -> vector<2xf16>
```

- **inputs:** `%src` is `i32`, `i64`, `f16`, `bf16`, `f32`,
  `vector<2xf16>`, `vector<2xbf16>`, or `vector<2xf32>`.
  `round(R)` selects the rounding rule. `sat` or `nosat` selects whether
  finite overflow is clamped to the destination range. `signed` or `unsigned`
  is required when converting to or from an integer type and is omitted for
  floating-to-floating and packed vector conversion.
- **outputs:** `%dst` is `i32`, `i64`, `f16`, `bf16`, `f32`,
  `vector<2xf16>`, `vector<2xbf16>`, or `vector<2xf32>`.
- **constraints and limitations:** Integer-to-integer conversion is not
  supported by `pto.convert`. Scalar floating-to-floating conversion supports
  `f32`, `f16`, and `bf16` source/destination pairs. `i64` source conversion is
  supported only to `f32`; conversion to `i64` is supported only from `f32`.
  `i32` can convert to `f32`, `f16`, or `bf16`, with `signed` or `unsigned`
  selecting the source interpretation. Floating-to-integer conversion supports
  `i32` destinations, plus `f32 -> i64`, and requires `sat`. Packed conversion
  supports only
  `vector<2xf32> -> vector<2xf16>`, `vector<2xf16> -> vector<2xf32>`,
  `vector<2xf32> -> vector<2xbf16>`, and
  `vector<2xbf16> -> vector<2xf32>`.

Rounding selectors:

| Selector | Meaning |
|----------|---------|
| `round(r)` | Round to nearest, ties to even |
| `round(a)` | Round away from zero |
| `round(f)` | Round toward minus infinity |
| `round(c)` | Round toward plus infinity |
| `round(z)` | Round toward zero |
| `round(o)` | Round to odd |
| `round(h)` | Cast-ceil mode for the target conversion slice that supports it |

Saturation selectors:

| Selector | Meaning |
|----------|---------|
| `nosat` | Do not clamp finite overflow to the destination range |
| `sat` | Clamp finite overflow to the destination range |

---

#### SIMT Entry Synchronization and State Ops

##### `pto.syncthreads`

- **syntax:** `pto.syncthreads attr-dict`
- **semantics:** Synchronize all active workitems in the current SIMT entry.
  Memory effects issued before the barrier by participating workitems are
  ordered before memory effects issued after the barrier by those workitems.
- **inputs:** None.
- **outputs:** None.
- **constraints and limitations:** `pto.syncthreads` must appear inside a
  function marked with `pto.simt_entry`. It synchronizes workitems in the
  active SIMT entry; it is not a substitute for outer pipeline synchronization
  between vector, MTE, cube, and scalar host-visible effects.

Example:

```mlir
func.func @body(%ub: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {
  %tx = pto.get_tid_x : i32
  %idx = arith.index_castui %tx : i32 to index
  pto.store %tx, %ub[%idx] : !pto.ptr<i32, ub>, i32
  pto.syncthreads
  %v = pto.load %ub[%idx] : !pto.ptr<i32, ub> -> i32
  pto.store %v, %ub[%idx] : !pto.ptr<i32, ub>, i32
  return
}
```

##### `pto.threadfence` / `pto.threadfence_block`

- **syntax:** `pto.threadfence attr-dict` or `pto.threadfence_block attr-dict`
- **semantics:** Issue a memory fence for memory effects from the current SIMT
  workitem. `pto.threadfence` uses the target workitem fence operation;
  `pto.threadfence_block` uses the target block-scoped workitem fence
  operation.
- **inputs:** None.
- **outputs:** None.
- **constraints and limitations:** These ops must appear inside a function
  marked with `pto.simt_entry`. They order memory effects but do not by
  themselves make other workitems wait; use `pto.syncthreads` when a workitem
  barrier is required.

##### `pto.keep` / `pto.resume`

- **syntax:**

```mlir
pto.keep %value {slot = N : i64} : T
%value = pto.resume {slot = N : i64} : T
```

- **semantics:** Preserve and restore one per-workitem scalar payload across
  adjacent SIMT entry calls in the same outer launch sequence. `pto.keep`
  records the current workitem's `%value` in logical slot `N`; `pto.resume`
  restores the value for the same logical workitem from logical slot `N`.

```text
for each active workitem:
  keep(slot, value) stores value in that workitem's slot
  resume(slot) returns the value stored in the same workitem's slot
```

- **inputs:** `pto.keep` takes one scalar `%value` of type `T`.
- **outputs:** `pto.resume` returns one scalar value of type `T`.
- **attributes:** `slot` is a non-negative `i64` logical slot identifier in
  the range `[0, 122]`.
- **supported types:** `T` may be any signless integer scalar with bit width up
  to 64 bits, `f16`, `bf16`, or `f32`.
- **constraints and limitations:** Both ops must appear inside functions marked
  with `pto.simt_entry`. A `pto.resume` group must be the first non-constant
  operation group in its SIMT entry. A `pto.keep` group must be the final
  operation group before optional `pto.syncthreads` and `func.return`. Slot
  storage words must not overlap within one `pto.resume` group or one
  `pto.keep` group. A value resumed from a slot should use the same type as the
  value kept into that slot.
- **slot allocation rule:** Users allocate slots explicitly. Values up to 32
  bits and supported floating-point values consume only `slot`. A 64-bit
  integer value consumes `slot` and `slot + 1`; therefore its slot must be even
  and must leave room for the second word. Because slot mapping is explicit, a
  later SIMT entry may resume only the subset of preserved slots that it needs
  without changing the location of those slots.

Example:

```mlir
func.func @stage0(%dst: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {
  %tx = pto.get_tid_x : i32
  %idx = arith.index_castui %tx : i32 to index
  pto.store %tx, %dst[%idx] : !pto.ptr<i32, ub>, i32
  pto.keep %tx {slot = 0 : i64} : i32
  pto.syncthreads
  return
}

func.func @stage1(%dst: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {
  %tx0 = pto.resume {slot = 0 : i64} : i32
  %tx = pto.get_tid_x : i32
  %idx = arith.index_castui %tx : i32 to index
  %sum = arith.addi %tx0, %tx : i32
  pto.store %sum, %dst[%idx] : !pto.ptr<i32, ub>, i32
  return
}
```

---

#### Outer Pipeline Synchronization and Ordering

SIMT body execution is sequenced as a function call from the outer kernel. Use
the existing PTO pipeline synchronization ops around the SIMT call when data is
produced or consumed by other pipelines.

```mlir
pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32
func.call @body(%ub_out) : (!pto.ptr<i32, ub>) -> ()
pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.mte_ub_gm %ub_out, %gm_out, %len
  nburst(%n, %src_stride, %dst_stride) l2_cache_ctl(%l2_cache_ctl)
  : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i64, i64
```

For pipeline synchronization semantics, see
[`01-pipeline-sync.md`](#micro-01-pipeline-sync). Do not use pipeline barriers as a
substitute for lane collectives: vote, shuffle, redux, and atomic ops are the
SIMT-specific mechanisms documented in this chapter.

---

#### Complete Minimal Example

```mlir
module attributes {pto.target_arch = "a5",
                   pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @simt_store_tid_kernel(%out: !pto.ptr<i32, gm>)
      attributes {pto.aicore} {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %dim_z = arith.constant 1 : i32
    %dim_y = arith.constant 1 : i32
    %dim_x = arith.constant 32 : i32

    %ub_out = pto.castptr %c0_i64 : i64 -> !pto.ptr<i32, ub>
    pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32
    func.call @simt_write(%ub_out) : (!pto.ptr<i32, ub>) -> ()

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %ub_out, %out, %c128_i64
      nburst(%c32_i64, %c128_i64, %c128_i64) l2_cache_ctl(%c0_i64)
      : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }

  func.func @simt_write(%dst: !pto.ptr<i32, ub>)
      attributes {pto.simt_entry} {
    %tx = pto.get_tid_x : i32
    %ty = pto.get_tid_y : i32
    %tz = pto.get_tid_z : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %ty_shift = arith.shli %ty, %c8_i32 : i32
    %tz_shift = arith.shli %tz, %c16_i32 : i32
    %xy = arith.ori %tx, %ty_shift : i32
    %xyz = arith.ori %xy, %tz_shift : i32
    %lane_base = arith.muli %ty, %c32_i32 : i32
    %idx_i32 = arith.addi %lane_base, %tx : i32
    %idx = arith.index_castui %idx_i32 : i32 to index
    pto.store %xyz, %dst[%idx] : !pto.ptr<i32, ub>, i32
    return
  }
}
```

<a id="micro-18-special-scalar"></a>

### 18. Special Scalar Operations

> **Category:** PTO scalar query, pointer/address, and scalar-memory operations
> **Dialect:** `pto`

Special Scalar operations provide the PTO-specific scalar facilities used
around vector and tile code. They query the current kernel execution instance,
construct and adjust typed pointers, access one scalar element through the
scalar pipeline, and perform ordinary AICore GM accesses that bypass the local
L1 data cache.

This group does not include shared scalar arithmetic, which remains in
[Arith](#micro-14-shared-arith), or SIMT workitem operations, which remain in
[SIMT Ops](#micro-17-simt). An operation with a scalar operand but a vector result,
such as `pto.vadds`, belongs to [Vec-Scalar Ops](#micro-08-vec-scalar-ops).

---

#### Operation Summary

| Family | Operations | Purpose |
|--------|------------|---------|
| Kernel execution queries | `pto.get_block_idx`, `pto.get_subblock_idx`, `pto.get_block_num`, `pto.get_subblock_num` | Query the block or subblock identity and launch extent visible to the current kernel instance |
| Typed pointer/address operations | `pto.castptr`, `pto.addptr` | Construct, reinterpret, and offset `!pto.ptr` values |
| Scalar-pipeline memory | `pto.load_scalar`, `pto.store_scalar` | Read or write one element through the general scalar-memory interface |
| AICore scalar GM L1-bypass | `pto.ld_dev`, `pto.st_dev` | Read or write one integer GM element while bypassing the local L1 data cache |

---

#### Common Pointer and Offset Rules

Memory operations in this chapter use the typed pointer form
`!pto.ptr<T, space>`:

- `T` is the element type stored at the pointed-to location;
- `space` identifies the memory space, such as `gm` or `ub`;
- a pointer carries an address, element type, and memory-space interpretation,
  but no tensor shape or stride metadata;
- every `%offset` operand in this chapter has type `index` and is measured in
  elements of `T`, not bytes.

For a base address `base`, element type `T`, and element offset `offset`, the
effective byte address is:

```text
effective_address = base + offset * sizeof(T)
```

For example, offset `3` on `!pto.ptr<i32, gm>` selects the element beginning
12 bytes after the base address.

---

#### Kernel Execution Query Operations

These nullary, side-effect-free operations expose the block-level execution
state visible to the current PTO kernel instance. They return `i64` values and
do not perform memory access, synchronization, tiling, or work partitioning by
themselves.

They are distinct from the `pto.get_tid_*`, `pto.get_block_idx_*`, and related
SIMT workitem queries documented in [SIMT Ops](#micro-17-simt).

##### `pto.get_block_idx`

- **Purpose:** Return the linear block index of the current kernel instance.
- **Syntax:**

  ```mlir
  %block = pto.get_block_idx
  ```

- **Operands and attributes:** None.
- **Result:** One `i64` block index.
- **Semantics:** The result identifies the current block in the range
  `[0, block_num)`, where `block_num` is the value returned by
  `pto.get_block_num` for the same launch.

```text
block = current_block_index
0 <= block < block_num
```

##### `pto.get_subblock_idx`

- **Purpose:** Return the subblock index visible to the current kernel
  instance.
- **Syntax:**

  ```mlir
  %subblock = pto.get_subblock_idx
  ```

- **Operands and attributes:** None.
- **Result:** One `i64` subblock index.
- **Semantics:** The result identifies the current subblock in the range
  `[0, subblock_num)`, where `subblock_num` is returned by
  `pto.get_subblock_num` for the same execution instance.

```text
subblock = current_subblock_index
0 <= subblock < subblock_num
```

##### `pto.get_block_num`

- **Purpose:** Return the total number of blocks in the current kernel launch.
- **Syntax:**

  ```mlir
  %block_num = pto.get_block_num
  ```

- **Operands and attributes:** None.
- **Result:** One `i64` block count.
- **Semantics:** The result is the launch-wide block count used to interpret
  `pto.get_block_idx`.

##### `pto.get_subblock_num`

- **Purpose:** Return the number of subblocks visible to the current execution
  instance.
- **Syntax:**

  ```mlir
  %subblock_num = pto.get_subblock_num
  ```

- **Operands and attributes:** None.
- **Result:** One `i64` subblock count.
- **Semantics:** The result is the subblock count used to interpret
  `pto.get_subblock_idx`.

##### Block Partitioning Example

The following example assigns a disjoint 2048-element GM window to each
block:

```mlir
%block = pto.get_block_idx
%block_num = pto.get_block_num
%block_len = arith.constant 2048 : index
%block_as_index = arith.index_cast %block : i64 to index
%block_offset = arith.muli %block_as_index, %block_len : index
%block_in = pto.addptr %gm_in, %block_offset
  : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
%block_out = pto.addptr %gm_out, %block_offset
  : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
```

The query operations report the launch state; the surrounding arithmetic and
pointer operations define the actual partitioning policy.

---

#### Typed Pointer and Address Operations

##### `pto.castptr`

- **Purpose:** Explicitly convert between an integer address, a typed PTO
  pointer, or a memref base address without moving data.
- **Syntax:**

  ```mlir
  %result = pto.castptr %input : input-type -> result-type
  ```

- **Operands:** `%input` is an integer, a memref, or `!pto.ptr<T, space>`.
- **Result:** An integer or `!pto.ptr<T, space>` according to the selected form.
- **Attributes:** None.
- **Legal forms:**

  | Input | Result | Meaning |
  |-------|--------|---------|
  | integer | `!pto.ptr<T, space>` | Interpret the integer as an address in `space` |
  | `!pto.ptr<T, space>` | integer | Expose the pointer address as an integer |
  | `!pto.ptr<S, space>` | `!pto.ptr<T, space>` | Reinterpret the element type while preserving the address and memory space |
  | `memref<..., space>` | `!pto.ptr<T, space>` | Extract the aligned base address and represent it as a PTO pointer |

- **Constraints:** Integer-to-integer and memref-to-integer forms are invalid.
  Pointer-to-pointer casts must preserve the PTO memory space. A memref with an
  explicit PTO memory space must be cast to the same space. The operation does
  not dereference the address and does not change the referenced bytes.

```text
result.address = input.address
result.space = requested_space
result.element_type = requested_element_type
```

Examples:

```mlir
%gm_i32 = pto.castptr %addr : i64 -> !pto.ptr<i32, gm>
%gm_i8 = pto.castptr %gm_i32
  : !pto.ptr<i32, gm> -> !pto.ptr<i8, gm>
%addr_again = pto.castptr %gm_i8 : !pto.ptr<i8, gm> -> i64
```

##### `pto.addptr`

- **Purpose:** Produce a pointer displaced from a typed base pointer.
- **Syntax:**

  ```mlir
  %result = pto.addptr %ptr, %offset
    : !pto.ptr<T, space> -> !pto.ptr<T, space>
  ```

- **Operands:** `%ptr` is `!pto.ptr<T, space>` and `%offset` is `index`.
- **Result:** A pointer with exactly the same element type and memory space as
  `%ptr`.
- **Attributes:** None.
- **Semantics:** `%offset` is a signed element displacement. Positive values
  advance the pointer and negative values move it toward lower addresses.

```text
result.address = ptr.address + offset * sizeof(T)
result.element_type = T
result.space = space
```

- **Constraints:** The result type must exactly match the input pointer type.
  `pto.addptr` does not access memory and does not perform bounds checking.

Example:

```mlir
%c16 = arith.constant 16 : index
%tail = pto.addptr %base, %c16
  : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
```

`%tail` points 16 `f32` elements, or 64 bytes, after `%base`.

---

#### Scalar-Pipeline Memory Operations

`pto.load_scalar` and `pto.store_scalar` access one element through the general
scalar-memory interface. The pointer element type and memory space determine
the accessed value type and storage domain.

##### `pto.load_scalar`

- **Purpose:** Read one scalar element from a typed PTO pointer.
- **Syntax:**

  ```mlir
  %value = pto.load_scalar %ptr[%offset]
    : !pto.ptr<T, space> -> T
  ```

- **Operands:** `%ptr` is `!pto.ptr<T, space>` and `%offset` is an element
  offset of type `index`.
- **Result:** One scalar value of type `T`.
- **Attributes:** None.
- **Semantics:** Read the element at `ptr + offset` through the scalar pipeline.

```text
value = memory[ptr.address + offset * sizeof(T)] as T
```

- **Constraints:** The result type must exactly match the pointer element type.
  This op returns a scalar, not a `!pto.vreg` value, and has no vector load
  distribution or mask clauses.

##### `pto.store_scalar`

- **Purpose:** Write one scalar element through a typed PTO pointer.
- **Syntax:**

  ```mlir
  pto.store_scalar %value, %ptr[%offset]
    : !pto.ptr<T, space>, T
  ```

- **Operands:** `%value` has type `T`, `%ptr` is `!pto.ptr<T, space>`, and
  `%offset` is an element offset of type `index`.
- **Results:** None.
- **Attributes:** None.
- **Semantics:** Write `%value` to the element at `ptr + offset` through the
  scalar pipeline.

```text
memory[ptr.address + offset * sizeof(T)] = value
```

- **Constraints:** `%value` must exactly match the pointer element type. This
  op writes one scalar element and has no vector store distribution or mask
  clauses.

Example round trip in UB:

```mlir
%c7 = arith.constant 7 : index
%value = pto.load_scalar %ub[%c7] : !pto.ptr<i32, ub> -> i32
pto.store_scalar %value, %ub_out[%c7] : !pto.ptr<i32, ub>, i32
```

---

#### AICore Scalar GM L1-Bypass Operations

`pto.ld_dev` and `pto.st_dev` are the ordinary AICore scalar GM access pair
for accesses that must bypass the local L1 data cache. They are not SIMT
operations and must not be substituted with `pto.ldg` or `pto.stg`, whose
execution scope and cache-control contract are different.

##### Common Contract

- the pointer must be `!pto.ptr<T, gm>`;
- `T` must be one of `i8`, `i16`, `i32`, or `i64`;
- `%offset` has type `index` and is measured in elements of `T`;
- load result and store value types must exactly match `T`;
- no `l1cache` or `l2cache` policy attribute is accepted;
- the op must appear in an ordinary AICore entry function, outside both a
  `pto.simt_entry` function and `pto.section.simt`;
- the supported target profile is A5 with CANN output version 9.0.0 official
  or newer.

Both operations are non-atomic. They do not imply synchronization, memory
ordering, cache invalidation, cache writeback, or an L2 cache policy. Programs
that combine these accesses with another cached path must provide any required
synchronization and cache maintenance separately. Cache behavior beyond the
local L1 data cache is target-defined.

##### `pto.ld_dev`

- **Purpose:** Read one integer scalar from GM while bypassing the local L1
  data cache.
- **Syntax:**

  ```mlir
  %value = pto.ld_dev %ptr[%offset] : !pto.ptr<T, gm> -> T
  ```

- **Operands:** `%ptr` is `!pto.ptr<T, gm>` and `%offset` is an element offset
  of type `index`.
- **Result:** One value of type `T` containing exactly the bytes read from GM.
- **Attributes:** None.
- **Semantics:** Read `sizeof(T)` bytes from the selected GM element. No sign
  extension, zero extension, truncation, or numeric conversion is part of the
  observable operation semantics.

```text
address = ptr.address + offset * sizeof(T)
value = GM[address : address + sizeof(T)] as T
```

##### `pto.st_dev`

- **Purpose:** Write one integer scalar to GM while bypassing the local L1 data
  cache.
- **Syntax:**

  ```mlir
  pto.st_dev %value, %ptr[%offset] : !pto.ptr<T, gm>, T
  ```

- **Operands:** `%value` has type `T`, `%ptr` is `!pto.ptr<T, gm>`, and
  `%offset` is an element offset of type `index`.
- **Results:** None.
- **Attributes:** None.
- **Semantics:** Write exactly `sizeof(T)` bytes from `%value` to the selected
  GM element.

```text
address = ptr.address + offset * sizeof(T)
GM[address : address + sizeof(T)] = value as bytes
```

##### Nonzero-Offset Example

```mlir
%c3 = arith.constant 3 : index
%value = pto.ld_dev %src[%c3] : !pto.ptr<i32, gm> -> i32
pto.st_dev %value, %dst[%c3] : !pto.ptr<i32, gm>, i32
```

Both operations access element 3, which begins 12 bytes after the corresponding
`i32` base address. The load and store bypass the local L1 data cache; they do
not establish ordering with other memory operations.

---

#### Choosing a Scalar Memory Operation

| Requirement | Operation family |
|-------------|------------------|
| General typed scalar access through the scalar-memory interface | `pto.load_scalar`, `pto.store_scalar` |
| Ordinary AICore integer GM access that bypasses local L1 | `pto.ld_dev`, `pto.st_dev` |
| SIMT workitem scalar memory access | See [SIMT Ops](#micro-17-simt) |

## Supported Data Types

| Type | Bits | vreg Lanes | Description |
|------|------|-----------|-------------|
| `i8` / `si8` / `ui8` | 8 | 256 | Signless/signed/unsigned 8-bit integer |
| `i16` / `si16` / `ui16` | 16 | 128 | Signless/signed/unsigned 16-bit integer |
| `f16` | 16 | 128 | IEEE 754 half precision |
| `bf16` | 16 | 128 | Brain floating point |
| `i32` / `si32` / `ui32` | 32 | 64 | Signless/signed/unsigned 32-bit integer |
| `f32` | 32 | 64 | IEEE 754 single precision |
| `i64` / `si64` / `ui64` | 64 | 32 | Signless/signed/unsigned 64-bit integer |

## Common Patterns

### Softmax (Numerically Stable)

```mlir
// 1. Find max
%max_vec = pto.vcmax %logits, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
pto.vsts %max_vec, %ub_tmp[%c0], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
%max_bc = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// 2. exp(x - max) using fused op
%exp = pto.vexpdif %logits, %max_bc, %mask, "ODD" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// 3. Sum
%sum = pto.vcadd %exp, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
pto.vsts %sum, %ub_tmp[%c0], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
%sum_bc = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// 4. Divide
%softmax = pto.vdiv %exp, %sum_bc, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

### ReLU Variants

```mlir
// Standard ReLU
%relu = pto.vrelu %input, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Leaky ReLU (scalar alpha)
%lrelu = pto.vlrelu %input, %alpha, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>

// Parametric ReLU (per-element alpha)
%prelu = pto.vprelu %input, %alpha_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

```

### Data Layout Conversion

```mlir
// AoS → SoA (deinterleave)
%x, %y = pto.vldsx2 %ub_xy[%offset], "DINTLV_B32" : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

// SoA → AoS (interleave)
pto.vstsx2 %x, %y, %ub_xy[%offset], "INTLV_B32", %all_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>
```

---

## Quick Reference by Category

### Memory Operations

| Operation | Group | Description |
|-----------|-------|-------------|
| GM→UB DMA | 2 | `pto.mte_gm_ub` |
| UB→GM DMA | 2 | `pto.mte_ub_gm` |
| UB→UB / UB→L1 copy | 2 | `pto.mte_ub_ub`, `pto.mte_ub_l1` |
| GM→L1 | 16 | `pto.mte_gm_l1`, `pto.mte_gm_l1_frac` |
| L1→UB | 16 | `pto.mte_l1_ub` |
| L1→BT | 16 | `pto.mte_l1_bt` |
| L1→FB | 16 | `pto.mte_l1_fb` |
| L1→L0A / L1→L0B | 16 | `pto.mte_l1_l0a`, `pto.mte_l1_l0b`, `pto.mte_l1_l0a_mx`, `pto.mte_l1_l0b_mx` |
| L0C→L1 / GM / UB (FIXPIPE MTE) | 16 | `pto.mte_l0c_l1`, `pto.mte_l0c_gm`, `pto.mte_l0c_ub` |
| Contiguous Load | 3 | `pto.vlds` with `NORM` dist |
| Broadcast Load | 3 | `pto.vlds` with `BRC` family dist |
| Gather | 3 | `pto.vgather2`, `pto.vgatherb` |
| Contiguous Store | 3 | `pto.vsts` with `NORM_B8` / `NORM_B16` / `NORM_B32` dist |
| Scatter | 3 | `pto.vscatter` |
| Scalar GM access bypassing local L1 data cache | 18 | `pto.ld_dev`, `pto.st_dev` |

### Compute Operations

| Operation | Group | Description |
|-----------|-------|-------------|
| Element-wise Arithmetic | 6, 7 | `pto.vadd`, `pto.vmul`, `pto.vabs`, etc. |
| Scalar Operations | 8 | `pto.vadds`, `pto.vmuls`, etc. |
| Transcendental | 6 | `pto.vexp`, `pto.vln`, `pto.vsqrt`, etc. |
| Reduction | 10 | `pto.vcadd`, `pto.vcmax`, `pto.vcmin` |
| Cube matmul family (zero-init / accumulate / bias-init; shared clauses `unit_flag`, `disable_gemv`, `sat`, `tf32_mode`, `n_dir`) | 16 | `pto.mad`, `pto.mad_acc`, `pto.mad_bias`, `pto.mad_mx`, `pto.mad_mx_acc`, `pto.mad_mx_bias` |
| Comparison | 11 | `pto.vcmp`, `pto.vcmps` |
| Selection | 11 | `pto.vsel`, `pto.vselr` |

### Type & Data Manipulation

| Operation | Group | Description |
|-----------|-------|-------------|
| Type Conversion | 9 | `pto.vcvt`, `pto.vbitcast`, `pto.pbitcast` |
| Interleave/Deinterleave | 12 | `pto.vintlv`, `pto.vdintlv` |
| Interleave/Deinterleave (not A5) | 12 | `pto.vintlvv2`, `pto.vdintlvv2` |

### Synchronization

| Operation | Group | Description |
|-----------|-------|-------------|
| Intra-core Sync | 1 | `pto.set_flag`, `pto.wait_flag` |
| Pipeline Buffer Sync | 1 | `pto.get_buf`, `pto.rls_buf` |
| Memory Barrier / Cache Maintenance | 1 | `pto.mem_bar`, `pto.dsb`, `pto.dcci` |

### Scalar & Control Operations

Group 14 covers shared MLIR scalar arithmetic. Group 18 catalogs PTO scalar
queries, pointer/address operations, and scalar-memory operations. SIMT scalar
operations remain in Group 17, while
shared structured-control semantics remain in Group 15.

| Operation | Group | Description |
|-----------|-------|-------------|
| Scalar Constants | 14 | `arith.constant` |
| Scalar Integer / Index Arithmetic | 14 | `arith.addi`, `arith.subi`, `arith.muli`, `arith.divsi`, `arith.remui`, `arith.ceildivsi`, etc. |
| Scalar Floating-Point Arithmetic | 14 | `arith.addf`, `arith.subf`, `arith.mulf`, `arith.divf`, `arith.maximumf`, etc. |
| Scalar Compare & Select | 14 | `arith.cmpi`, `arith.cmpf`, `arith.select` |
| Scalar Casts / Width Changes | 14 | `arith.index_cast`, `arith.index_castui`, `arith.extsi`, `arith.extui`, `arith.trunci`, `arith.sitofp`, etc. |
| Scalar Bitwise / Shift Ops | 14 | `arith.andi`, `arith.ori`, `arith.xori`, `arith.shli`, `arith.shrsi`, `arith.shrui`, etc. |
| Kernel Execution Queries | 18 | `pto.get_block_idx`, `pto.get_subblock_idx`, `pto.get_block_num`, `pto.get_subblock_num` |
| Typed Pointer / Address Operations | 18 | `pto.castptr`, `pto.addptr` |
| Scalar-Pipeline Memory | 18 | `pto.load_scalar`, `pto.store_scalar` |
| AICore Scalar GM L1-Bypass | 18 | `pto.ld_dev`, `pto.st_dev` |
| SIMT Scalar Memory / Atomics | 17 | `pto.load`, `pto.store`, `pto.ldg`, `pto.stg`, `pto.atomic_*` |
| SIMT Scalar Math / Conversion | 17 | `pto.prmt`, `pto.mulhi`, `pto.sqrt`, `pto.exp`, `pto.fma`, `pto.convert`, etc. |
| Counted Loops | 15 | `scf.for` |
| Conditional Regions | 15 | `scf.if`, `scf.yield` |
| Break-like Structured Loops | 15 | `scf.while`, `scf.condition`, `scf.yield` |

### Cube Operation Surface

- `pto.mte_l1_bt`
- `pto.mte_l1_fb`
- `pto.mte_gm_l1`
- `pto.mte_gm_l1_frac`
- `pto.mte_l1_ub`
- `pto.mte_l1_l0a`
- `pto.mte_l1_l0b`
- `pto.mte_l1_l0a_mx`
- `pto.mte_l1_l0b_mx`
- `pto.mad`
- `pto.mad_acc`
- `pto.mad_bias`
- `pto.mad_mx`
- `pto.mad_mx_acc`
- `pto.mad_mx_bias`
- `pto.mte_l0c_l1`
- `pto.mte_l0c_gm`
- `pto.mte_l0c_ub`

## Part IV: PTO Tile Instruction

PTO Tile Instruction is a high-performance instruction surface built on top of PTO micro Instruction. Each tile instruction encapsulates a tile-granular pattern — DMA between GM and on-chip buffers, vector arithmetic over a whole tile, reductions, broadcast / expansion, selection, padding — and internally expands to a sequence of micro-instruction primitives (`pto.vlds`, `pto.vsts`, `pto.vadd`, mask ops, sync flags, …).

The full PTO Tile Instruction reference starts from [Tile and PTO Tile Instruction overview](PTO-tile-Instruction-SPEC.md#tile-01-tile-overview). It covers:

- [Tile and PTO Tile Instruction overview](PTO-tile-Instruction-SPEC.md#tile-01-tile-overview) — tile concept, on-chip placement, physical shape vs valid region, conventions
- [Types & Attributes](PTO-tile-Instruction-SPEC.md#tile-02-types-and-attributes) — `!pto.tile_buf`, `!pto.tensor_view`, address spaces, layout, pad
- [Pointer & View](PTO-tile-Instruction-SPEC.md#tile-03-pointer-and-view) — tensor views, partitions, tile allocation, valid-shape updates
- [DMA Data Movement](PTO-tile-Instruction-SPEC.md#tile-04-dma-data-movement) — `pto.tload` / `pto.tstore`
- [Vector Arithmetic](PTO-tile-Instruction-SPEC.md#tile-05-vector-arithmetic) — `pto.tadd / tsub / tmul / tdiv / tmax / tmin`, tile-scalar forms, unary math, activations
- [Reductions](PTO-tile-Instruction-SPEC.md#tile-06-reduction-ops), [Partial Elementwise](PTO-tile-Instruction-SPEC.md#tile-07-partial-elementwise), [Bitwise & Shift](PTO-tile-Instruction-SPEC.md#tile-08-bitwise-shift-ops), [Type Conversion](PTO-tile-Instruction-SPEC.md#tile-09-type-conversion), [Broadcast & Expansion](PTO-tile-Instruction-SPEC.md#tile-10-broadcast-and-expansion-ops), [Selection](PTO-tile-Instruction-SPEC.md#tile-11-selection-ops), [Fill & Padding](PTO-tile-Instruction-SPEC.md#tile-12-fill-and-padding-ops)

For the boundary between Tile Instruction and the micro instruction surface (when to drop into `pto.vecscope` and how `pto.tile_buf_addr` bridges the two), see [Tile and PTO Tile Instruction overview §1.10](PTO-tile-Instruction-SPEC.md#110-mixing-pto-tile-instruction-and-pto-micro-instruction).
