# PTO Virtual micro Instruction (`pto.vmi`)

- v0.1: Doc init. Per-op reference for all `pto.vmi` ops, with syntax,
  semantics, operand tables, lowering notes, and lit-test examples.

- v0.2: Sync with PTOAS `origin/master` (2026-09-16): conversion contract
  matrix, unified/vreg bitwise interface split, carry ops, grouped
  reductions, and MERGE status corrections.

**Status:** draft. This document is the **instruction reference** for the unified
`pto.vmi` surface. It documents every user-facing op with concrete MLIR syntax,
per-operand tables, C-style semantics, and lowering-to-`pto.mi` guidance. 

Category A/B/C is defined inline in §1.4 below (self-contained in this doc).
Whenever this doc references a physical `pto.mi` op, it means the micro-SPEC.

[toc]

---

## Part I: Architecture Overview

### 1.1 Position in the Stack

`pto.vmi` sits between high-level programming models (TileLang, pto-dsl) and
the physical `pto.mi` ISA. It exposes **logically contiguous vectors** and
**elementwise compute intent**; the physical SIMD register layout (interleave,
parity, width, part, pack, dist tokens) is held and propagated by `pto.as` and
is invisible to the user.

```
TileLang  T.parallel(N) { C[i] = cast<i32>(A[i]) + B[i] }
   │  (direct translation, elementwise semantics preserved)
   ▼
pto.vmi   %w = pto.vmi.vcvt %a; %c = pto.vmi.vadd %w, %b
   │  (pto.as: layout-assignment + lowering)
   ▼
pto.mi    vcvt EVEN/ODD + two-way vadd + vstsx2 INTLV_B32
```

- **Upper → vmi**: `T.parallel`'s logical iteration space translates directly
  to `pto.vmi` logical vector ops — elementwise → Category A op, `T.cast` →
  a `vcvt` with no explicit `part`, logical length `N` →
  `!pto.vmi.vreg<N×T>`, "all active" → auto-generated tail predicate.
- **vmi → pto.mi**: `pto.as` performs layout inference + unification +
  materialization, lowering logical vectors to concrete `pto.mi` instructions
  (including `part/pack/interleave/dist`). At `K=1` this degenerates to
  zero-overhead pass-through.

**Relationship to the `pto.mi` architecture — inherited, not redefined.**
`pto.vmi` is strictly a **micro-instruction-level** abstraction; it does **not**
redefine the architecture. Every architectural definition of the underlying
`pto.mi` target is inherited verbatim from the
[PTO micro Instruction SPEC](./PTO-micro-Instruction-SPEC.md): the hardware
pipeline model (MTE2 / Vector / MTE3 under the Decoupled Access-Execute scheme),
the Vector Lane (VLane) organization and the `!pto.vreg` / `!pto.mask` physical
constants, the memory hierarchy and address spaces (`!pto.ptr<T, gm/ub>`, 256 KB
UB), and the synchronization model. In particular, the **Vector Function (VF)**
definition is taken unchanged: the `__VEC_SCOPE__` execution-scope contract —
scalar–vector fork-launch semantics, the `pto.vecscope` /
`pto.strict_vecscope` region ops, the strict-vs-default region-argument
distinction, the single-vector-interval legality rule, and the prohibition on
nested vector intervals — is exactly as the micro-SPEC's *Execution Scopes
(`__VEC_SCOPE__`)* section defines it. `pto.vmi` introduces **no new scope op**
and changes **no scope semantics**: a `pto.vmi` program is still enclosed by
exactly one `pto.vecscope` / `pto.strict_vecscope` region.

**Scope of replacement — micro-instruction groups §3–§13 only.** `pto.vmi`
performs abstraction and replacement **only at the micro-instruction level**, and
only over the vector compute / UB↔vreg data-movement surface of `pto.mi`. It
unifies and replaces micro-instruction groups **§3 (Vector Load/Store)** through
**§13 (DSA/SFU Ops)** — that is, §3 Vector Load/Store, §4 Predicate Load/Store,
§5 Materialization & Predicate Ops, §6 Unary Vector Ops, §7 Binary Vector Ops,
§8 Vec-Scalar Ops, §9 Conversion Ops, §10 Reduction Ops, §11 Compare & Select,
§12 Data Rearrangement, and §13 DSA/SFU Ops — folding their per-op physical-layout
variants (interleave / parity / part / pack / dist tokens) into a single logical
surface that `pto.as` lowers back to concrete `pto.mi`.

Everything outside that range remains **untouched `pto.mi`**, exactly as the
micro-SPEC defines it, and is outside `pto.vmi`'s replacement scope:

- §1 Pipeline Synchronization (`pto.set_flag` / `pto.wait_flag` /
  `pto.pipe_barrier` / `pto.get_buf` / `pto.rls_buf`)
- §2 DMA Copy Programming (`pto.mte_gm_ub` / `pto.mte_ub_gm` /
  `pto.mte_ub_ub` / `pto.mte_ub_l1`)
- §14 shared `arith` and §15 shared `scf` — the scalar / control-flow glue
  surrounding the VF
- §16 Cube Matrix Multiply — the cube pipe, orthogonal to the vector pipe

The partition is therefore clean: `pto.vmi` replaces the **vector-pipe compute
and UB↔vreg data-movement** instructions (§3–§13) at the micro-instruction level,
while the surrounding synchronization, DMA, scalar-control, and cube layers stay
as authored `pto.mi`.

### 1.2 Logical vs Physical

A `pto.vmi` value is **logical** — a flat sequence of `L` lanes of type `T`.
Its physical backing is `K` hardware vector registers (256B / 2048-bit each):

```
K = ⌈ L · bitwidth(T) / 2048 ⌉
```

At `K=1` and full-width (no partial lanes), one `pto.vmi.vreg` maps 1:1 to
one `pto.vreg`. At `K>1`, the logical value fans out across `K` physical
registers with a layout descriptor (`#pto.vmi.layout`) tracking the mapping.

**Physical constants (A5 vector pipe):**

```
vector register file : 32 architectural vregs, 256 B (2048 bit) each
predicate file       : 8  architectural pregs, 256 bit each, 1 bit controls 1 byte
VLane                : 32 B sub-lane; 8 VLanes per vreg
E_v = 32 / sizeof(T) : lanes per VLane     (f32 → 8, f16/bf16 → 16, i8 → 32)
```

### 1.3 Type System

#### `!pto.vmi.vreg<L×T>`

Logical vector register. `L` is the logical lane count; `T` is the element type.

| T | bits | E_v (lanes per physical vreg) | Legal L multiples |
|---|---|---|---|
| `f32` / `i32` / `ui32` / `si32` | 32 | 64 | 64 |
| `f16` / `bf16` / `i16` / `ui16` / `si16` | 16 | 128 | 64 |
| `i8` / `ui8` / `si8` / `fp8_e4m3` / `fp8_e5m2` | 8 | 256 | 64 |

- **Full vector**: `L · bitwidth(T) == N · 2048` (integer multiple of 256B).
- **Compact/partial vector**: `L · bitwidth(T) < 2048` — still backed by one
  physical vreg (256B); only the low `L` logical slots are valid. Physical
  slots outside the logical value are `pad/undef` and must be masked out.

**Common logical ↔ physical mappings:**

| Logical type | Byte size | K | Physical vregs | Valid slots per vreg |
|---|---:|---:|---:|---|
| `V<256×f32>` | 1024B | 4 | 4 | 64 f32 each, all valid |
| `V<256×f16>` | 512B | 2 | 2 | 128 f16 each, all valid |
| `V<256×i8>` | 256B | 1 | 1 | 256 i8, all valid |
| `V<128×f32>` | 512B | 2 | 2 | 64 f32 each, all valid |
| `V<64×f16>` | 128B | 1 | 1 | low 64 f16 valid |
| `V<64×i8>` | 64B | 1 | 1 | low 64 i8 valid |

#### `V<256×f32>`: 4 physical regs (K=4)

**Logical view**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x254 │ x255 │
└────┴────┴────┴─────┴──────┴──────┘
                  256 lane
```

**Physical view (contiguous)** - 4 physical regs, each BlockLane = 32B = 8 f32 lanes

```text
   BL0       BL1               BL7
┌───────┬───────┬───┬───────┐
│ x0..7 │ x8..15│...│x56..63│
└───────┴───────┴───┴───────┘
            P0 (256B)

   BL0        BL1              BL7
┌────────┬────────┬───┬─────────┐
│x64..71 │x72..79 │...│x120..127│
└────────┴────────┴───┴─────────┘
            P1 (256B)

   BL0          BL1             BL7
┌──────────┬──────────┬───┬──────────┐
│x128..135 │x136..143 │...│x184..191 │
└──────────┴──────────┴───┴──────────┘
            P2 (256B)

   BL0          BL1             BL7
┌──────────┬──────────┬───┬──────────┐
│x192..199 │x200..207 │...│x248..255 │
└──────────┴──────────┴───┴──────────┘
            P3 (256B)
```

**Physical view (non-contiguous, parity EVEN/ODD)** - even lanes in P0/P2, odd lanes in P1/P3 (typical source: `V<256×f16> -> V<256×f32>` widening preserves parity; all 4 regs carry 64 valid lanes each)

```text
 P0 (chunk0 EVEN)   P1 (chunk0 ODD)    P2 (chunk1 EVEN)   P3 (chunk1 ODD)
┌────┬────┬─────┐ ┌────┬────┬─────┐ ┌──────┬──────┬─────┐ ┌──────┬──────┬─────┐
│ x0 │ x2 │x126 │ │ x1 │ x3 │x127 │ │ x128 │ x130 │x254 │ │ x129 │ x131 │x255 │
└────┴────┴─────┘ └────┴────┴─────┘ └──────┴──────┴─────┘ └──────┴──────┴─────┘
    64 lane            64 lane            64 lane            64 lane
```

> Restore contiguous: `INTLV_B32(P0, P1) -> [x0..x127]`, `INTLV_B32(P2, P3) -> [x128..x255]`, then concatenate in chunk order.

**Physical view (non-contiguous, P0/P1/P2/P3)** - 4-way stride-4 interleave: every 4 logical elements land in one reg each (`x0,x4,...` -> P0; `x1,x5,...` -> P1; `x2,x6,...` -> P2; `x3,x7,...` -> P3); all 4 regs carry 64 valid lanes each (corresponds to the sub_part / part_T 4-way axis)

```text
   P0                   P1                   P2                   P3
┌────┬────┬─────┐  ┌────┬────┬─────┐  ┌────┬────┬─────┐  ┌────┬────┬─────┐
│ x0 │ x4 │x252 │  │ x1 │ x5 │x253 │  │ x2 │ x6 │x254 │  │ x3 │ x7 │x255 │
└────┴────┴─────┘  └────┴────┴─────┘  └────┴────┴─────┘  └────┴────┴─────┘
     64 lane              64 lane              64 lane              64 lane
```

#### `V<256×f16>`: 2 physical regs (K=2)

**Logical view**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x254 │ x255 │
└────┴────┴────┴─────┴──────┴──────┘
                  256 lane
```

**Physical view (contiguous)** - 2 physical regs, each BlockLane = 32B = 16 fp16 lanes

```text
   BL0           BL1               BL7
┌──────────┬──────────┬───┬───────────┐
│ x0..x15  │ x16..x31 │...│x112..x127 │
└──────────┴──────────┴───┴───────────┘
                  P0 (256B)

   BL0           BL1               BL7
┌──────────┬──────────┬───┬───────────┐
│x128..x143│x144..x159│...│x240..x255 │
└──────────┴──────────┴───┴───────────┘
                  P1 (256B)
```

**Physical view (non-contiguous, parity EVEN/ODD)** - even lanes in P0, odd lanes in P1 (e.g. after `DINTLV_B16` load or `vdintlv` preserves parity; both regs carry 128 valid lanes each)

```text
   P0 (EVEN)                                   P1 (ODD)
┌────┬────┬────┬─────┬──────┬──────┐  ┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x2 │ x4 │ ... │ x252 │ x254 │  │ x1 │ x3 │ x5 │ ... │ x253 │ x255 │
└────┴────┴────┴─────┴──────┴──────┘  └────┴────┴────┴─────┴──────┴──────┘
   128 even lanes valid                     128 odd lanes valid
```

#### `V<256×i8>`: 1 physical reg (K=1)

**Logical view**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x254 │ x255 │
└────┴────┴────┴─────┴──────┴──────┘
                  256 lane
```

**Physical view (contiguous)** - 1 physical reg, each BlockLane = 32B = 32 i8 lanes

```text
   BL0          BL1                  BL7
┌─────────────┬─────────────┬───┬──────────────┐
│ x0 ... x31  │ x32 ... x63 │...│x224 ... x255 │
└─────────────┴─────────────┴───┴──────────────┘
                   P0 (256B)
```

#### `V<128×f32>`: 2 physical regs (K=2)

**Logical view**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x127 │ x128 │
└────┴────┴────┴─────┴──────┴──────┘
                  128 lane
```

**Physical view (contiguous)** - 2 physical regs, each BlockLane = 32B = 8 f32 lanes

```text
   BL0       BL1               BL7
┌───────┬───────┬───┬───────┐
│ x0..7 │ x8..15│...│x56..63│
└───────┴───────┴───┴───────┘
            P0 (256B)

   BL0        BL1              BL7
┌────────┬────────┬───┬─────────┐
│x64..71 │x72..79 │...│x120..127│
└────────┴────────┴───┴─────────┘
            P1 (256B)
```

**Physical view (non-contiguous, parity EVEN/ODD)** - even lanes in P0, odd lanes in P1

```text
 P0 (chunk0 EVEN)   P1 (chunk0 ODD) 
┌────┬────┬─────┐ ┌────┬────┬─────┐
│ x0 │ x2 │x126 │ │ x1 │ x3 │x127 │
└────┴────┴─────┘ └────┴────┴─────┘
    64 lane            64 lane      
```

#### `V<64×fp16>`: 1 partial physical reg (K=1, low 64 lanes valid)

**Logical view**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x62  │ x63  │
└────┴────┴────┴─────┴──────┴──────┘
                  64 lane
```

**Physical view (contiguous)** - 1 physical reg, low 64 lanes valid, each BlockLane = 16 fp16 lanes

```text
   BL0          BL1         BL2          BL3          BL4   BL5   BL6   BL7
┌──────────┬──────────┬──────────┬──────────┬──────┬──────┬──────┬──────┐
│ x0..x15  │ x16..x31 │ x32..x47 │ x48..x63 │      │      │      │      │
└──────────┴──────────┴──────────┴──────────┴──────┴──────┴──────┴──────┘
<------------- 128B logical payload -------------><---- 128B outside logical value ---->
                          P0 (256B)
```

**Physical view (non-contiguous, part EVEN/ODD)** - single `V<64×fp32> -> V<64×fp16>` narrowing carrier: the 64 valid fp16 sit on even/odd positions of the 128 physical lanes

```text
   EVEN carrier (phys lanes 0,2,...,126 valid)
┌────┬───┬────┬───┬─────┬─────┬───┬─────┬───┐
│ x0 │ _ │ x1 │ _ │ ... │ x62 │ _ │ x63 │ _ │
└────┴───┴────┴───┴─────┴─────┴───┴─────┴───┘
```


#### `V<64×fp8>`: 1 partial physical reg (K=1, low 64 lanes valid)

**Logical view**

```text
┌────┬────┬────┬─────┬──────┬──────┐
│ x0 │ x1 │ x2 │ ... │ x62  │ x63  │
└────┴────┴────┴─────┴──────┴──────┘
                  64 lane
```

**Physical view (contiguous)** - 1 physical reg, low 64 lanes valid, each BlockLane = 32 fp8 lanes

```text
      BL0           BL1          BL2   BL3   BL4   BL5   BL6   BL7
┌─────────────┬─────────────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ x0 ... x31  │ x32 ... x63 │      │      │      │      │      │      │
└─────────────┴─────────────┴──────┴──────┴──────┴──────┴──────┴──────┘
<-- 64B logical payload  --><------- 192B outside logical value ------>
                          P0 (256B)
```

**Physical view (non-contiguous, sub_part P0)** - from `V<64×fp32> -> V<64×fp8>` via `vcvt`: instead of placing the low 64B contiguously, the 0th byte of each 4B group holds the valid fp8 (`PK4_B32` extraction target)

```text
P0: 256B fp8 carrier, viewed as 64 groups × 4B, only the P0 slot is valid per group
┌────────────┬────────────┬─────┬─────────────┐
│ x0  _  _  _│ x1  _  _  _│ ... │ x63  _  _  _│
│ P0 P1 P2 P3│ P0 P1 P2 P3│     │ P0 P1 P2 P3 │
└────────────┴────────────┴─────┴─────────────┘
   grp0          grp1              grp63
```

#### `!pto.vmi.mask<L>`

Virtual predicate mask. Each logical mask lane corresponds to one logical
vector lane (`L` must match the governed vreg's `L`).

### 1.4 Category A / B / C

Every VMI op belongs to one of three lowering categories that determine how
`pto.as` handles its physical layout:

| Category | Layout relationship | `pto.as` behavior | Output layout |
|---|---|---|---|
| **A — Layout-passthrough** | Does not modify register layout | Fan-out: emit the same `pto.mi` op once per physical reg (`K × op`); mask follows per-reg (with `ppack`/`punpack` as needed) | Unchanged: preserves input parity/half/sub-part layout |
| **B — Layout-rewritable** | Modifies layout predictably | Fan-out along other axes; instantiate matching modes (`PART_EVEN/ODD`, `Bin_N0/N1`, `PK`/`UNPK`, `INTLV`/`DINTLV`) | Rewritten to the op's natural output layout |
| **C — Contiguous-required** | Requires stride-1 contiguous input (no in-place mode satisfies it) | `pto.as` inserts `.contiguous()` materialization (store+reload or explicit repack) before the op | Flattened contiguous chunk (`is_contiguous`) |

> **C-class note:** C-class ops cannot tolerate a non-contiguous physical
layout — any parity/half/sub-part arrangement must first be materialized to
contiguous before the op runs. `pto.as` therefore treats a C-class op as a
**layout barrier**: upstream A/B ops may keep their compact layout right up to
the C-class boundary, where a `.contiguous()` is forced. This is why
gather/scatter and sort are Category C, while elementwise compute is A.

### 1.5 Mask & Predication (`pmode`)

All compute ops accept an optional governing mask operand `[pmode]`. The mask
is a `!pto.vmi.mask<L>` with the same `L` as the data operand.

**`pmode` values:**

| `pmode` | Inactive lane behavior | Default? |
|---|---|---|
| `"zero"` | Inactive lanes produce 0 (hardware-native ZEROING) | ✓ (default) |
| `"merge"` | Inactive lanes preserve the destination's prior value — **not implemented yet** | |

On A5, MERGE is **not implemented yet**: the hardware predicates only in ZEROING mode, so the
compiler has to synthesize merge as a predicate complement plus a `vor`/`vsel` blend
of the zeroed result with the old destination (see Appendix C). On A6,
some ops support native MERGE.

**A5 load restriction**: `vload` has **no** mask operand — A5 loads are
unpredicated. A logical tail mask associated with a load is never lowered as a
"masked load"; `pto.as` migrates it to the consuming compute op, the store, or
shortens the load length. `vstore` **is** predicated on A5.

### 1.6 The `group` Attribute

Reduce ops (`vcadd`, `vcmax`, `vcmin`) and broadcast (`vbrc`) accept an
optional `{group=C}` attribute where `C` is the **number of groups** (not the
per-group lane count):

- **Reduce**: Splits `L` lanes into `C` groups, each producing one scalar.
  Output is `V<C×T>` — a compact vector of `C` scalars.
- **Broadcast**: Takes a compact `V<C×T>` and fans each scalar back across
  `L/C` lanes, producing `V<L×T>`.

Legal `C` values: `1`, `2`, `4`, `8` (must divide `L`; must match the result
type's `C`).

**`group → Category` decision table** (W = bytes per sub-group):

| W vs BlockLane (32B) | Category | Lowering |
|---|---|---|
| `W == 32B` (sub-group = 1 VLane) | B | `vcgadd`/`vcgmax`/`vcgmin` — one op per reg, no cross-reg combine |
| `W > 32B`, aligned | B | Fold `(k-1)× vadd/vmax/vmin` then `vcg*` |
| Unaligned | C | Materialize → contiguous → reduce |

## Part II: Instruction Reference

### Group Index

| # | Group | Ops | Category | Mask |
|---|---|---|---|---|
| 1 | **Load / Store** | `vload`, `vstore` | A (+B on dintlv/intlv) | load: none; store: `Pg` |
| 2 | **Index-gen** | `vci` | A | none |
| 3 | **Eltwise Compute** | `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`, `vabs`, `vneg`, `vrelu`, `vexp`, `vln`, `vsqrt`, `vand`, `vor`, `vxor`, `vnot`, `vshl`, `vshr`, `vadds`, `vmuls`, `vmaxs`, `vmins`, `vshls`, `vshrs`, `vcmp`, `vcmps`, `vsel`, `vselr` | A | `Pg` (except `vselr`: none) , `vaddc`, `vaddcs`, `vsubc`, `vsubcs` |
| 4 | **Broadcast** | `vbrc` | A (ungrouped) / B (grouped) | none |
| 5 | **Reduce** | `vcadd`, `vcmax`, `vcmin` | B (VLane-aligned) / C (unaligned) | `Pg req` |
| 6 | **Convert** | `vcvt`, `vinterpret_cast` | B / A | `Pg` / none |
| 7 | **SFU** | `vexpdif`, `vaxpy`, `vlrelu`, `vprelu`, `vmull`, `vmula`, `vchist`, `vdhist`, `vgather`, `vscatter` | A (fused, `vmull`) / B (`vchist`, `vdhist`) / C (gather/scatter) | `Pg` (`vchist`/`vdhist`/SFU) / `Pg` (gather/scatter) |
| 8 | **Predicate Ops** | `create_mask`, `create_group_mask` | gen | gen |
| 9 | **Data Rearrange** | `vintlv`, `vdintlv` | A | `Pg` |

---

## Group 1: Load / Store

> **Category:** A (+B on `dintlv`/`intlv`). **Mask:** load none (A5 loads are unpredicated), store `Pg`.
>
> `vload`/`vstore` are logical memory ops. **`[dist_mode]` explicitly declares
> the access pattern**, defaulting to `continuous` (contiguous); the optional
> modes are `dintlv` (deinterleaved dual load), `brc` (broadcast) and `intlv` (interleaved dual store).

### `pto.vmi.vload`

- **semantics:** Load elements of type `T` from UB into a logical vector
  register starting at `%source + %offset` (element offset). The default
  (`continuous`) is a contiguous stride-1 read:

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = ub[base + offset + i];
  ```

  The access pattern is not always contiguous: depending on the attributes
  (`{dist_mode}`, `{group = C}` with a `stride` operand, or
  `%block_stride`), the load may instead read in a strided/scattered
  fashion (e.g. per-row stride for group mode, 32B-block stride for
  block-stride mode), deinterleave the source, or broadcast. The exact pattern is
  determined by these mutually exclusive attributes (see attributes and
  lowering below).

- **syntax:**
  ```mlir
  %result = pto.vmi.vload %source[%offset] : !pto.ptr<T, ub> -> !pto.vmi.vreg<L×T>
  ```
- **syntax (`group`):**
  ```mlir
  // strided group load: C rows of L/C elements, row g at base + g*stride
  %result = pto.vmi.vload %source[%offset], %stride {group = C}
      : !pto.ptr<T, ub>, index -> !pto.vmi.vreg<L×T>
  ```
- **syntax (block-stride):**
  ```mlir
  // block-strided load: %block_stride is a dynamic i16 operand (no mask)
  %result = pto.vmi.vload %source[%offset], %block_stride
      : !pto.ptr<T, ub>, i16 -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `source` | `!pto.ptr<T, ub>` | UB base pointer |
  | `offset` | `index` | Element offset from base |
  | `stride` | `index` | Per-row stride (element units); required with `{group}`, invalid otherwise |
  | `block_stride` | `i16` | 32B-block stride between scattered blocks (block-stride mode); mutually exclusive with `{group}` |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Loaded logical vector |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `dist_mode` | `"continuous"`, `"dintlv"`, `"brc"` | `"continuous"` | Memory access pattern |
  | `group` | positive integer | *(none)* | Strided group load arity; mutually exclusive with `dist_mode`; requires `stride` |
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-lane behavior (applied at consumer, not on load) |

- **lowering to `pto.mi`:**
  - **dist-mode** `vload` and `vstore` accept an optional `{dist_mode = "..."}` attribute
declaring the memory access pattern. Default is `"continuous"`.

  | `dist_mode` | Physical lowering |
  |---|---|
  | `"continuous"` | Known 32B-aligned addresses use aligned access; other effective UB addresses require a proven safe physical read range for unaligned access |
  | `"dintlv"` | `K × pto.vldsx2 {dist="DINTLV_B*"}` (deinterleaved dual load; suffix from `Ptr<T>`) |
  | `"brc"` | `1 × pto.vlds {dist="BRC_B*"}` or `BRC_BLK`; broadcast-axis (1-reg backing, replicate-read) |

  **Short continuous access.** Address alignment and readable memory size
  are separate requirements. Let `A` be the effective byte address
  `source + offset * sizeof(T)` and `P = L * sizeof(T)` be the logical payload.

  - `L = 1` reads one element and requires only element alignment and that
    element to be readable.
  - For `L > 1` and `P <= 32`, bounded block access requires a provably
    32B-aligned `A`. The caller must provide the entire readable interval
    `[A, A + 32)`, even when the logical payload is smaller than 32 bytes.
  - For `32 < P < 256` with `P` a multiple of 32, bounded block access also
    requires a provably 32B-aligned `A`, and the caller must provide the
    entire readable interval `[A, A + P)`.
  - A dynamic offset is accepted by the bounded block path when its effective
    address can be proven aligned. Unknown alignment is not an alignment
    guarantee. Partial extra blocks or unproven alignment require an independent
    safe physical-read proof for another supported access sequence; otherwise
    compilation rejects the load. A raw pointer alone provides no allocation
    extent for that proof.

  Address proofs use the existing address-space ABI alignment contract for
  pointer block arguments; callers must satisfy that contract. An arbitrary
  integer cast to a pointer does not establish alignment.

  Limiting an access to one block does not remove its address alignment
  requirement. A consumer mask does not shorten the physical read range.
  Full-carrier loads retain their existing rules. These are physical access
  requirements; the separate PTODSL lane whitelist is `1/2/4/8/64/128/256`.

  **Group mode** (`{group = C}` + `stride`) has two sub-cases, decided by the
  relation between `result.L` and `C`:
  - **Full-group load** (`result.L > C`): each group loads `L/C` elements,
    row-strided tile load: `C·(L/C) = L` elements across `C` rows,
    each row `g` at offset `base + g·stride`.
  - **Slot load** (`result.L == C`): each group loads **1 scalar** into the
    corresponding slot, producing a compact `V<C×T>`. This is the
    dual of group reduce — reduce
    folds lanes into slots, slot load reads those slots back into a vreg.
    `C ∈ {1, 2, 4, 8}`. Not combinable with `dist_mode`.

  **Block-stride mode** (`%block_stride` operand): 2D-tile block-strided load.
  Memory is read in 32B blocks with block `blk` at
  `base + blk * block_stride` (scattered access); the internal repeat stride
  defaults to 0. `%block_stride` is a dynamic `i16` operand. A5 loads are
  unpredicated, so an implicit all-active mask is applied. Not combinable
  with `dist_mode` or `group`.

  `B*` suffix is derived from `Ptr<T>` element width: `f32/i32 → B32`, `f16/bf16/i16 → B16`, `i8/fp8 → B8`.

- **examples:**

  ```mlir
  // Continuous load (default dist_mode): UB → vreg
  %v = pto.vmi.vload %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64×f32>
  // → pto.as: Ptr<f32> → B32, dist_mode=continuous → pto.mi.vlds {dist="NORM"}
  // Slot load: 1 scalar per group → compact V<8×f32> (reads back reduce output)
  %s = pto.vmi.vload %ub[%off], %stride {group = 8}
      : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<8×f32>
  // → each of 8 groups loads 1 scalar into its slot
  // Full-group load: 8 rows × 8 elements, stride 64
  %t = pto.vmi.vload %ub[%off], %stride {group = 8}
      : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<64×f32>
  // → 8 rows of 8 elements, row g at base + g*stride
  // Block-strided load: block_stride = 8 (dynamic i16 operand, no mask)
  %vb = pto.vmi.vload %ub[%off], %c8_i16
      : !pto.ptr<f32, ub>, i16 -> !pto.vmi.vreg<64×f32>
  // → block-strided load (block=8), all lanes active

  // Broadcast load: scalar/block replicate into vreg
  %vb = pto.vmi.vload %ub[%offset] {dist_mode = "brc"} : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64×f32>
  // → pto.as: Ptr<f32> → B32, dist_mode=brc → pto.mi.vlds {dist="BRC_B32"}
  ```

- **notes:**
  - **A5 loads are unpredicated.** A tail mask associated with a `vload` is
    never lowered as a masked load. It migrates to the consuming compute op or
    to a `vstore`.
  - Continuous loads support effective UB addresses that are not 32B-aligned
    only when the complete physical read range of a supported unaligned access
    can be proven safe. This range may exceed the logical payload. Alignment
    state is managed internally and is not part of the VMI programming model.
  - `dist_mode` and layout inference are orthogonal: `pto.as` may still
    rewrite the physical layout of a `continuous` load to serve a downstream
    consumer (e.g. a grouped reduce).
  - The `pmode` attribute on `vload` governs the result lane behavior at the
    *consumer*, not on the load itself.

- **attention:**
  - **`{group}`, `%block_stride`, and `{dist_mode}` are mutually exclusive.**
    Specifying more than one at once is rejected by `pto.as`.
  - **`stride` operand is bound to `{group}`.** It is required with
    `{group = C}` and invalid otherwise; `block_stride` is bound to the
    block-stride mode and invalid otherwise. `vload` has no mask operand in
    any mode (A5 loads are unpredicated).

### `pto.vmi.vstore`

- **semantics:** Store elements from a vector register to UB starting at
  `%dest + %offset` (element offset). The default (`continuous`) is a
  contiguous stride-1 write; only lanes where `mask[i] != 0` are written
  (A5 stores are predicated):

  ```c
  for (int i = 0; i < L; i++)
      if (mask[i])
          ub[base + offset + i] = src[i];
  ```

  The access pattern is not always contiguous: depending on the attributes
  (`{dist_mode}`, `{group = C}` with a `stride` operand, or
  `%block_stride`), the store may instead write in a strided/scattered
  fashion (e.g. per-row stride for group mode, 32B-block stride for
  block-stride mode). The exact pattern is determined by these mutually
  exclusive attributes (see attributes and lowering below).

- **syntax:**
  ```mlir
  pto.vmi.vstore %value, %dest[%offset], %mask : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, !pto.vmi.mask<L>
  ```
- **syntax (`group`):**
  ```mlir
  // strided group store: C rows of L/C elements, row g at base + g*stride (no mask)
  pto.vmi.vstore %value, %dest[%offset], %stride {group = C}
      : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, index
  ```
- **syntax (block-stride):**
  ```mlir
  // block-strided store: %block_stride is a dynamic i16 operand (mask required)
  pto.vmi.vstore %value, %dest[%offset], %block_stride, %mask
      : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, i16, !pto.vmi.mask<L>
  ```
- **operands:**

 | Operand | Type | Description |
 |---|---|---|
  | `value` | `!pto.vmi.vreg<L×T>` | Vector value to store |
  | `dest` | `!pto.ptr<T, ub>` | UB destination base pointer |
  | `offset` | `index` | Element offset from base |
  | `stride` | `index` | Per-row stride (element units); required with `{group}`, invalid otherwise |
  | `block_stride` | `i16` | 32B-block stride between scattered blocks (block-stride mode); mutually exclusive with `{group}` |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate (variadic: 0 or 1) |

- **results:** *(none)*

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `dist_mode` | `"continuous"`, `"intlv"` | `"continuous"` | Memory access pattern |
  | `group` | positive integer | *(none)* | Strided group store arity; mutually exclusive with `dist_mode`; requires `stride`; forbids `mask` |
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-lane behavior: `"zero"` (default) stores 0; `"merge"` skips write on inactive lanes |

- **lowering to `pto.mi`:**
  - **dist-mode** `vload` and `vstore` accept an optional `{dist_mode = "..."}` attribute
declaring the memory access pattern. Default is `"continuous"`.

  | `dist_mode` | Physical lowering |
  |---|---|
  | `"continuous"` | Known 32B-aligned addresses use the aligned fast path; unmasked accesses to other effective UB addresses use an unaligned sequence with lowering-managed alignment state |
  | `"intlv"` | `K × pto.vstsx2 {dist="INTLV_B*"}` (interleaved dual store; suffix from `Ptr<T>`) |

  **Group mode** (`{group = C}` + `stride`): row-strided tile store. Not combinable with
  `dist_mode` or `mask` (group stores are unpredicated).

  **Block-stride mode** (`%block_stride` operand): 2D-tile block-strided store.
  Memory is written in 32B blocks with block `blk` at
  `base + blk * block_stride` (scattered access); the internal repeat stride
  defaults to 0. `%block_stride` is a dynamic `i16` operand. An explicit
  `mask` is applied; if absent an implicit all-active mask is used. Not
  combinable with `dist_mode` or `group`.

  Continuous unmasked stores support effective UB addresses that are not
  32B-aligned, including a final prefix tail. Alignment state and the required
  final flush are managed by lowering and are not part of the VMI programming
  model. Explicit sparse-mask stores still use the predicated store path and
  require their target alignment constraints to be statically provable;
  otherwise the access is unsupported.

- **examples:**

  ```mlir
  // Continuous store (default): vreg → UB, masked
  pto.vmi.vstore %v, %ub_out[%offset], %mask : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, !pto.vmi.mask<64>
  // → pto.as: Ptr<f32> → B32, dist_mode=continuous → pto.mi.vsts {dist="NORM_B32"}
  ```

  ```mlir
  // Group (strided) store: 8 rows × 8 elements, stride 64 (no mask)
  pto.vmi.vstore %tile, %ub_out[%off], %stride {group = 8}
      : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, index
  // → 8 rows of 8 elements, row g at base + g*stride
  // Block-strided store: block_stride = 8 (dynamic i16 operand + mask)
  pto.vmi.vstore %v, %ub_out[%off], %c8_i16, %mask
      : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, i16, !pto.vmi.mask<64>
  // → block-strided store (block=8), governed by mask
  ```



---

## Group 2: Index-gen

> **Category:** A. **Mask:** none.
>
> Index materialization. Produces an index vector; the single physical reg
> backing is replicate-read until a Category B/C edge needs the expanded form.

### `pto.vmi.vci`

- **semantics:** Generate a per-lane index/counter vector from a single scalar base such as `[base, base±1, base±2, ...]`, lane `i` gets `base + i` (ASC) or `base - i` (DESC). It is the index source for `vgather`/`vscatter` offsets.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = base + (order == "ASC" ? i : -i);
  ```

  With `group=C>1`, each group of `S=L/C` lanes restarts the ramp:

  ```c
  dst[g*S + j] = base + (order == "ASC" ? j : -j);
  ```

  `group=1` is normalized to ordinary continuous `iota`, so it has exactly the
  same semantics and tail support as omitting `group`. Group-periodic iota is
  an internal contiguous-only producer; layout assignment inserts
  `ensure_layout` when a consumer requests a deinterleaved layout.

- **syntax:**
  ```mlir
  %result = pto.vmi.vci %base {order = "ASC", group = 2} : T -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `base` | scalar (`i8`/`i16`/`i32`, `f16`/`f32`) | Starting value |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Index vector |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `order` | `"ASC"`, `"DESC"` | `"ASC"` | Index generation direction |
  | `group` | positive integer | omitted | Number of equal groups. `1` is equivalent to omitted; values greater than one restart the ramp per group. |

- **lowering to `pto.mi`:**
  ```
  1 × pto.vci {ASC/DESC} per chunk
  ```
  `#mi = 1/chunk`, `dep = 1`.

- **datatypes:** `i8`/`i16`/`i32`, `f16`, `f32`. For every element type,
  the legal lane counts `L` of the result are `1, 2, 4, 8, 64, 128, 256`.

- **example:**
  ```mlir
  // Ascending i32 indices for a gather base
  %idx = pto.vmi.vci %c0 {order = "ASC"} : i32 -> !pto.vmi.vreg<64×i32>
  // Descending f32 ramp
  %ramp = pto.vmi.vci %c10 {order = "DESC"} : f32 -> !pto.vmi.vreg<64×f32>
  ```

- **example:**
  ```mlir
  %idx = pto.vmi.vci %base {order = "ASC"} : i32 -> !pto.vmi.vreg<64×i32>
  // → pto.as: pto.vci {order="ASC"}, one op per physical chunk
  ```

---

## Group 3: Eltwise Compute

> **Category:** A (layout-passthrough) for ordinary per-lane operations;
> `vselr` is classified separately as Category C below. **Mask:** `Pg`
> (optional governing predicate, except `vselr` which has none).
>
> Pure per-lane ops. Layout passes through unchanged. An operand whose
> cardinality along an axis is 1 becomes a broadcast (replicate-read, never
> expanded to `K` copies). Under the `K ≤ 4` core profile these fan out as
> fully-unrolled straight-line code.

### 3.1 Binary Arithmetic

#### `pto.vmi.vadd` / `pto.vmi.vsub`

- **semantics:** Unified fp/int elementwise add / subtract.

  ```c
  for (int i = 0; i < N; i++)
      dst[i] = mask[i] ? lhs[i] + rhs[i] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vadd %lhs, %rhs, %mask {pmode = "zero"} : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `lhs` | `!pto.vmi.vreg<L×T>` | First operand |
  | `rhs` | `!pto.vmi.vreg<L×T>` | Second operand |
  | `mask` | `!pto.vmi.mask<L>` (variadic) | Governing predicate (0 or 1) |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Elementwise result |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-lane behavior |

- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vadd / pto.vsub  (+ mask per reg, ppack/punpack if needed)
  ```
  `#mi = K`, `dep = 1`, util = 100%.

#### `pto.vmi.vmul`

- **semantics:** Unified floating-point/integer elementwise multiply.
- **syntax:** Same operand, mask, result, and `pmode` model as
  `pto.vmi.vadd` / `pto.vmi.vsub`.
- **datatypes:** `i16`, `i32`, `f16`, `bf16`, `f32`. The A5 vector multiply
  family has no 8-bit integer form.
- **lowering to `pto.mi`:** `K × pto.vmul`.

- **example:**
  ```mlir
  // fp32 add with deinterleaved layout
  %sum = pto.vmi.vadd %a, %b
      : !pto.vmi.vreg<128×f32>,
        !pto.vmi.vreg<128×f32>
      -> !pto.vmi.vreg<128×f32>
  // → pto.as: 2 × pto.vadd (EVEN/ODD), each with create_mask all-active mask

  // Masked add with merge mode
  %s = pto.vmi.vadd %a, %b, %mask {pmode = "merge"}
      : !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64×f32>
  ```

#### `pto.vmi.vaddc` / `pto.vmi.vsubc` / `pto.vmi.vaddcs` / `pto.vmi.vsubcs`

Carry-chain integer arithmetic is exposed as multi-result VMI operations so the
frontend can preserve the hardware carry instruction instead of expanding the
operation into an add/compare/select sequence.

```mlir
%sum, %carry = pto.vmi.vaddc %lhs, %rhs, %mask
    : !pto.vmi.vreg<Lxui32>, !pto.vmi.vreg<Lxui32>, !pto.vmi.mask<L>
    -> !pto.vmi.vreg<Lxui32>, !pto.vmi.mask<L>
%difference, %borrow_free = pto.vmi.vsubc %lhs, %rhs, %mask
    : !pto.vmi.vreg<Lxui32>, !pto.vmi.vreg<Lxui32>, !pto.vmi.mask<L>
    -> !pto.vmi.vreg<Lxui32>, !pto.vmi.mask<L>
%next, %carry2 = pto.vmi.vaddcs %lhs, %rhs, %carry, %mask
    : !pto.vmi.vreg<Lxui32>, !pto.vmi.vreg<Lxui32>, !pto.vmi.mask<L>, !pto.vmi.mask<L>
    -> !pto.vmi.vreg<Lxui32>, !pto.vmi.mask<L>
%difference2, %borrow_free2 = pto.vmi.vsubcs %lhs, %rhs, %carry, %mask
    : !pto.vmi.vreg<Lxui32>, !pto.vmi.vreg<Lxui32>, !pto.vmi.mask<L>, !pto.vmi.mask<L>
    -> !pto.vmi.vreg<Lxui32>, !pto.vmi.mask<L>
```

These operations require matching 32-bit integer data values. The execution
mask, carry-in (for `vaddcs`/`vsubcs`), and carry-out use the same logical lane
count and layout as the data ports, and all physical mask parts must use `b32`
granularity. They lower one-to-N to the corresponding VPTO carry operation.
For subtraction, the carry predicate is **not-borrow**: the comparison is
unsigned, `carry[i] = 1` means no borrow occurred, and `carry[i] = 0` means a
borrow occurred. In `vsubcs`, a carry-in of 0 propagates a borrow.

#### `pto.vmi.vdiv`

- **semantics:** Elementwise floating-point divide.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? lhs[i] / rhs[i] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vdiv %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `f16`, `f32` only
- **lowering to `pto.mi`:**
  ```
  K × pto.vdiv
  ```
  `#mi = K`, `dep = 1`.

#### `pto.vmi.vmax` / `pto.vmi.vmin`

- **semantics:** Elementwise maximum / minimum (unified fp/int).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? max(lhs[i], rhs[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vmax %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vmax / pto.vmin
  ```
  `#mi = K`, `dep = 1`.

### 3.2 Unary Arithmetic & Activation

#### `pto.vmi.vabs`

- **semantics:** Elementwise absolute value (unified fp/int).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? abs(src[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vabs %src, %mask {pmode = "zero"} : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `si8`, `si16`, `si32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vabs (si8/si16/si32/f16/f32)
  K × sign-bit clear (bf16)
  ```
  BF16 has no direct A5 vector-absolute instruction, so VMI implements it by
  clearing each element's sign bit. `dep = 1`.

#### `pto.vmi.vneg`

- **semantics:** Elementwise negate: `0 - x`.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? -src[i] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vneg %src, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `i8`–`i32`, `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vneg
  ```
  `#mi = K`, `dep = 1`.

#### `pto.vmi.vrelu`

- **semantics:** Elementwise ReLU: `max(0, x)`.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? max(0, src[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vrelu %src, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `si32`, `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vrelu
  ```
  `#mi = K`, `dep = 1`.

#### `pto.vmi.vexp` / `pto.vmi.vln` / `pto.vmi.vsqrt`

- **semantics:** Elementwise transcendental: exponential, natural logarithm, square root.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? exp(src[i]) : (pmode_merge ? dst_old[i] : 0);   // vexp
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? ln(src[i])  : (pmode_merge ? dst_old[i] : 0);   // vln
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? sqrt(src[i]) : (pmode_merge ? dst_old[i] : 0);  // vsqrt
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vexp %src, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `f16`, `f32` only
- **lowering to `pto.mi`:**
  ```
  K × pto.vexp / pto.vln / pto.vsqrt
  ```
  `#mi = K`, `dep = 1`.

### 3.3 Bitwise Ops

#### `pto.vmi.vand` / `pto.vmi.vor` / `pto.vmi.vxor`

- **semantics:** Elementwise bitwise AND / OR / XOR. Operands and result are
  vregs by default. These ops also accept mask-typed operands, performing a
  per-lane predicate boolean op and yielding a mask. When the operands are
  masks (predicate type), no governing `mask` operand may be given — a mask
  operand would be ambiguous with the predicate data operands themselves.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (lhs[i] & rhs[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  // vreg operands (optional governing mask)
  %r = pto.vmi.vand %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>

  // mask operands (no governing mask)
  %r = pto.vmi.vand %lhs, %rhs : !pto.vmi.mask<L>, !pto.vmi.mask<L> -> !pto.vmi.mask<L>
  %r = pto.vmi.vxor %lhs, %rhs : !pto.vmi.mask<L>, !pto.vmi.mask<L> -> !pto.vmi.mask<L>
  ```
- **datatypes:** `i8`–`i32` (integer bitwise); `pred` (per-lane boolean op)
- **interface split:** the two forms are lowered through two separate
  interfaces. vreg operands become the vreg-interface ops
  `pto.vmi.andi` / `pto.vmi.ori` / `pto.vmi.xori`, which are vreg-only and keep
  the governing `mask`; mask operands become `pto.vmi.mask_and` /
  `pto.vmi.mask_or` / `pto.vmi.mask_xor`. `pto.as` performs this split in
  `vmi-lower-unified-to-legacy`.
- **lowering to `pto.mi`:**
  ```
  K × pto.vand / pto.vor / pto.vxor
  ```
  `#mi = K`, `dep = 1`. The governing `mask` of the vreg form becomes the
  predication operand of each part.

#### `pto.vmi.vnot`

- **semantics:** Elementwise bitwise NOT. Operand and result are vregs by
  default. This op also accepts a mask-typed operand, performing a per-lane
  predicate complement and yielding a mask. When the operand is a mask
  (predicate type), no governing `mask` operand may be given — a mask operand
  would be ambiguous with the predicate data operand itself.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? ~src[i] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  // vreg operand (optional governing mask)
  %r = pto.vmi.vnot %src, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>

  // mask operand (no governing mask)
  %r = pto.vmi.vnot %src : !pto.vmi.mask<L> -> !pto.vmi.mask<L>
  ```
- **datatypes:** `i8`–`i32`; `pred` (predicate complement)
- **interface split:** the vreg form becomes `pto.vmi.not` (vreg-only, keeps the
  governing `mask`), the mask form becomes `pto.vmi.mask_not`; `pto.as`
  performs this split in `vmi-lower-unified-to-legacy`.
- **lowering to `pto.mi`:**
  ```
  K × pto.vnot
  ```
  `#mi = K`, `dep = 1`. The governing `mask` of the vreg form becomes the
  predication operand of each part.

### 3.4 Shift Ops

#### `pto.vmi.vshl` / `pto.vmi.vshr`

- **semantics:** Elementwise left shift (`vshl`) or right shift (`vshr`). The shift count is per-lane from `rhs`. For `vshr`, signed elements use arithmetic right shift, while unsigned and signless elements use logical right shift.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (lhs[i] << rhs[i]) : (pmode_merge ? dst_old[i] : 0);  // vshl
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (lhs[i] >> rhs[i]) : (pmode_merge ? dst_old[i] : 0);  // vshr (signed: arithmetic; unsigned: logical)
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vshl %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `si8`/`si16`/`si32` and `ui8`/`ui16`/`ui32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vshl / pto.vshr
  ```
  `#mi = K`, `dep = 1`.

### 3.5 Vec-Scalar Ops

Vec-scalar ops broadcast a scalar to all lanes (R6 implicit broadcast). The
scalar type must match the vector element type.

#### `pto.vmi.vadds` / `pto.vmi.vmaxs` / `pto.vmi.vmins`

- **semantics:** Elementwise vector-scalar add / max / min.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? src[i] + scalar : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vadds %src, %scalar, %mask {pmode = "merge"} : !pto.vmi.vreg<L×T>, T, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.vreg<L×T>` | Vector operand |
  | `scalar` | `T` | Scalar (implicitly broadcast to all lanes) |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Elementwise result |

- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vadds / pto.vmaxs / pto.vmins
  ```
  `#mi = K`, `dep = 1`. No extra reg for scalar.

- **example:**
  ```mlir
  %shifted = pto.vmi.vadds %x, %bias, %mask
      : !pto.vmi.vreg<64×f32>, f32, !pto.vmi.mask<64> -> !pto.vmi.vreg<64×f32>
  ```

#### `pto.vmi.vmuls`

- **semantics:** Elementwise vector-scalar multiply with the same scalar
  broadcast, mask, and `pmode` model as the other vector-scalar operations.
- **datatypes:** `i16`, `i32`, `f16`, `f32`.
- **lowering to `pto.mi`:** `K × pto.vmuls`.
- **example:**
  ```mlir
  %scaled = pto.vmi.vmuls %x, %scale, %mask
      : !pto.vmi.vreg<64×f32>, f32, !pto.vmi.mask<64> -> !pto.vmi.vreg<64×f32>
  ```

#### `pto.vmi.vshls` / `pto.vmi.vshrs`

- **semantics:** Elementwise vector-scalar shift.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (src[i] << scalar) : (pmode_merge ? dst_old[i] : 0);  // vshls
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (src[i] >> scalar) : (pmode_merge ? dst_old[i] : 0);  // vshrs
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vshls %src, %shift, %mask : !pto.vmi.vreg<L×T>, i16, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `si8`/`si16`/`si32` and `ui8`/`ui16`/`ui32`. The uniform shift
  amount is a `ui16` value independent of `T` and should be in the
  range `[0, bitwidth(T))`. For `vshrs`, signed elements use arithmetic right
  shift, while unsigned and signless elements use logical right shift.
- **lowering to `pto.mi`:**
  ```
  K × pto.vshls / pto.vshrs
  ```
  `#mi = K`, `dep = 1`.

### 3.6 Compare & Select

#### `pto.vmi.vcmp`

- **semantics:** Elementwise compare → predicate mask. The `seed` mask is the
  governing predicate `Pg`: where `seed[i] = 0` the result lane is 0 (zeroing);
  where `seed[i] = 1` the comparison is evaluated.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = seed[i] ? cmp(lhs[i], rhs[i]) : 0;
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vcmp %lhs, %rhs, %seed {cmp = "lt"} : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.mask<L>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `lhs` | `!pto.vmi.vreg<L×T>` | First operand |
  | `rhs` | `!pto.vmi.vreg<L×T>` | Second operand |
  | `seed` | `!pto.vmi.mask<L>` | Governing predicate (required) |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.mask<L>` | Predicate mask (same L, granularity derived from T) |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `cmp` | `eq`, `ne`, `lt`, `le`, `gt`, `ge` | *(required)* | Comparison mode (fp unordered / integer; integer signedness comes from the element type: `siN` vs `iN`/`uiN`) |
  | | `oeq`, `one`, `olt`, `ole`, `ogt`, `oge` | | FP ordered forms |
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-lane behavior |

- **datatypes:** `i8`/`si8`/`ui8` – `i32`/`si32`/`ui32`, `f16`, `bf16`, `f32`.
  Integer signedness is taken from the element type; signless `iN` is treated
  as unsigned (equivalent to `uiN`).
- **lowering to `pto.mi`:**
  ```
  K × pto.vcmp {cmp_mode}
  ```
  `#mi = K`, `dep = 1`. +1 preg per live mask result.

- **example:**
  ```mlir
  // f32 less-than compare over deinterleaved layout
  %lt = pto.vmi.vcmp %a, %b, %seed {cmp = "lt"}
      : !pto.vmi.vreg<128×f32>,
        !pto.vmi.vreg<128×f32>,
        !pto.vmi.mask<128×b32>
      -> !pto.vmi.mask<128×b32>
  // → pto.as: 2 × pto.vcmp "lt" (EVEN/ODD), each with per-reg seed mask

  // i32 unsigned greater-than-or-equal (signless integers use unsigned semantics)
  %ge = pto.vmi.vcmp %a, %b, %seed {cmp = "ge"}
      : !pto.vmi.vreg<128×i32>, !pto.vmi.vreg<128×i32>, !pto.vmi.mask<128×b32>
      -> !pto.vmi.mask<128×b32>
  // si32 signed greater-than-or-equal (signedness carried by the `si32` element type)
  %ge = pto.vmi.vcmp %a, %b, %seed {cmp = "ge"}
      : !pto.vmi.vreg<128×si32>, !pto.vmi.vreg<128×si32>, !pto.vmi.mask<128×b32>
      -> !pto.vmi.mask<128×b32>
  // ui32 unsigned greater-than-or-equal (same `cmp = "ge"`; signedness from `ui32`)
  %uge = pto.vmi.vcmp %ua, %ub, %seed {cmp = "ge"}
      : !pto.vmi.vreg<128×ui32>, !pto.vmi.vreg<128×ui32>, !pto.vmi.mask<128×b32>
      -> !pto.vmi.mask<128×b32>
  // bf16 contiguous equality compare (K=1)
  %eq = pto.vmi.vcmp %a, %b, %seed {cmp = "eq"}
      : !pto.vmi.vreg<128×bf16>, !pto.vmi.vreg<128×bf16>, !pto.vmi.mask<128×b16>
      -> !pto.vmi.mask<128×b16>
  ```

#### `pto.vmi.vcmps`

- **semantics:** Elementwise vector-scalar compare → predicate mask.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = seed[i] ? cmp(src[i], scalar) : 0;
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vcmps %src, %scalar, %seed {cmp = "ge"} : !pto.vmi.vreg<L×T>, T, !pto.vmi.mask<L> -> !pto.vmi.mask<L>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.vreg<L×T>` | Vector operand |
  | `scalar` | `T` | Scalar to compare against |
  | `seed` | `!pto.vmi.mask<L>` | Governing predicate (required) |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.mask<L>` | Predicate mask |

- **attributes:** Same `cmp` / `pmode` as `vcmp`.
- **datatypes:** `i8`/`si8`/`ui8` – `i32`/`si32`/`ui32`, `f16`, `bf16`, `f32`.
  Integer signedness is taken from the element type; signless `iN` is treated
  as unsigned (equivalent to `uiN`). The scalar operand's element type must
  match the vector's, so signedness is consistent on both operands.
- **lowering to `pto.mi`:**
  ```
  K × pto.vcmps {cmp_mode}
  ```
  `#mi = K`, `dep = 1`.

- **example:**
  ```mlir
  %ges = pto.vmi.vcmps %a, %c0, %seed {cmp = "ge"}
      : !pto.vmi.vreg<64×f32>, f32, !pto.vmi.mask<64> -> !pto.vmi.mask<64>
  ```

#### `pto.vmi.vsel`

- **semantics:** Per-lane selection driven by a predicate mask.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? true_val[i] : false_val[i];
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vsel %mask, %true_val, %false_val {pmode = "zero"} : !pto.vmi.mask<L>, !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `mask` | `!pto.vmi.mask<L>` | Selector predicate (required) |
  | `true_val` | `!pto.vmi.vreg<L×T>` | Value when mask[i] = 1 |
  | `false_val` | `!pto.vmi.vreg<L×T>` | Value when mask[i] = 0 |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Selected result |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Result handling when selector inactive: `"merge"` retains `false_value` lanes |

- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vsel
  ```
  `#mi = K`, `dep = 1`.

- **example:**
  ```mlir
  %out = pto.vmi.vsel %mask, %x, %y {pmode = "zero"}
      : !pto.vmi.mask<256×b16>, !pto.vmi.vreg<256×ui16>, !pto.vmi.vreg<256×ui16>
      -> !pto.vmi.vreg<256×ui16>
  ```

#### `pto.vmi.vselr`

- **layout contract:** Category C (contiguous-required). Source, index, and
  result use contiguous layout; an arbitrary input layout is not passed through
  this operation. Compilation may materialize a contiguous representation at
  this boundary. IR that reaches this operation with an assigned
  non-contiguous layout is unsupported.

- **semantics:** Dynamic lane permutation: `result[i] = source[index[i]]`.

  ```c
  for (int i = 0; i < N; i++)
      dst[i] = src[index[i]];
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vselr %source, %index : !pto.vmi.vreg<N×T>, !pto.vmi.vreg<N×index_T> -> !pto.vmi.vreg<N×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `source` | `!pto.vmi.vreg<N×T>` | Source vector to select from |
  | `index` | `!pto.vmi.vreg<N×index_T>` | Per-lane source lane index |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<N×T>` | Permuted result |

- **datatypes:** 8-, 16-, and 32-bit integer or floating-point source/result
  elements; `index_T` must be an integer type with the same storage width as
  `T`.
- **constraints:** Source, index, and result have the same lane count. The
  supported lane counts are `N ∈ {64, 128, 256}` for 8-bit elements,
  `N ∈ {64, 128}` for 16-bit elements, and `N = 64` for 32-bit elements.
  Every `index[i]` must identify a valid logical source lane; behavior is
  unspecified for an out-of-range index.

- **notes:**
  - This is the permute/gather class — it is the register-resident realization
    of a grouped broadcast.
  - `vselr` takes no mask; the index vector encodes the permutation directly.
  - `vselrv2` is not available on A5 and does not add other supported shapes.

- **example:**
  ```mlir
  %r = pto.vmi.vselr %src, %idx
      : !pto.vmi.vreg<128×f16>, !pto.vmi.vreg<128×i16> -> !pto.vmi.vreg<128×f16>
  ```

### 3.7 Borrow Ops (Not Provided)

borrow arithmetic  is **not provided** on the current surface. It will be added directly
as `i64` element-wise ops once the `i64` support plan is finalized and the
hardware path is confirmed. Until then, widening to `i64` scalar emulation
or fusing at the `pto.mi` layer is the workaround.

---

## Group 4: Broadcast

> **Category:** A (ungrouped scalar→vector), B (grouped `{group}`).
> **Mask:** none.
>
> `vbrc` is the logical scalar→vector / compact→full broadcast. The ungrouped
> form (single scalar fanned over `L` lanes) is cheap (`vdup`); the grouped form
> (per-group scalar fan-back) has no single native instruction and is a
> cost-model decision.

### `pto.vmi.vbrc`

- **semantics:** Broadcast a scalar or group-slot compact value across lanes.

  **Ungrouped:** One value replicated to all `L` lanes.
  ```c
  for (int i = 0; i < L; i++)
      dst[i] = src[0];
  ```

  **Grouped (`{group = C}`):** Each of the `C` compact scalar slots is
  fanned back across `L/C` lanes.
  ```c
  int gs = L / C;  // lanes per group
  for (int g = 0; g < C; g++)
      for (int i = 0; i < gs; i++)
          dst[g * gs + i] = src[g];
  ```

- **syntax:**
  ```mlir
  // Ungrouped: scalar → full vector
  %r = pto.vmi.vbrc %scalar : f32 -> !pto.vmi.vreg<64×f32>

  // Ungrouped: 1-lane vreg → full vector
  %r = pto.vmi.vbrc %val : !pto.vmi.vreg<1×f32> -> !pto.vmi.vreg<256×f32>

  // Grouped: compact group-slot → dense vector
  %r = pto.vmi.vbrc %source {group = 128} : !pto.vmi.vreg<128×f32> -> !pto.vmi.vreg<1024×f32>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `value` | `T` (scalar) or `!pto.vmi.vreg<C×T>` | Broadcast source |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Broadcast result |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `group` | positive integer | *(none — ungrouped)* | Number of group slots; must equal `input.L` for group mode |

- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **Bounded grouped vectors:** For the one-carrier lengths documented in
  [Reduce](05-reduce.md), a group count of at most eight that divides `L`
  broadcasts compact source slot `g` into logical lanes
  `[g * L/C, (g + 1) * L/C)`. The A5 VPTO backend retains established native
  layouts when available and uses a dense register-selection fallback otherwise.
  The source can come from a short load or a grouped reduction. Padding lanes
  in the physical register are not logical results.
- **Group slots and output layout:** Source-slot spacing and broadcast-output
  spacing describe different values. For `ui16 L=64, group=8`, native integer
  addition first normalizes its eight 32-bit sums into eight consecutive
  16-bit group slots. Broadcast selects and repeats those values into the
  preferred `ls(2)` output, which a `PK_B32` store writes as 64 logical
  elements. This path uses `vpack`, `vselr`, and a packing store; it does not
  require a separate source-slot unpack after normalization. Directly storing
  the eight group values instead uses the short-vector store path. Retaining
  strided native sums for consumers to select is a
  [follow-up exploration](05-reduce.md#follow-up-exploration-retain-strided-native-sums).
- **lowering to `pto.mi`:**

  | Form | Physical lowering | `#mi` | `dep` |
  |---|---|---|---|
  | Ungrouped (scalar) | `1 × pto.vdup` (register-resident), or `vsts`+`vlds BRC_*` (UB roundtrip) | `1` | `1` |
  | Ungrouped (1-lane vreg) | `1 × pto.vdup {position="LOWEST"}` per physical reg | `K` | `1` |
  | Grouped (`{group}`) | **Cost-model decision**: UB roundtrip (`vsts` partials + `vlds BRC_BLK`) **or** `vselr` gather **or** masked recompute | varies | 2–3 |

- **examples:**
  ```mlir
  // Ungrouped: scalar → full vector
  %bc = pto.vmi.vbrc %maxe : f32 -> !pto.vmi.vreg<64×f32>
  // → pto.as: pto.vdup %maxe (one op, register-resident)

  // Ungrouped: 1-lane vreg → full vector (rank-0 broadcast)
  %bc = pto.vmi.vbrc %scalar : !pto.vmi.vreg<1×f32> -> !pto.vmi.vreg<256×f32>
  // → pto.as: 4 × pto.vdup {position="LOWEST"} (K=4)

  // Grouped: 128 compact slots → 1024-lane dense vector
  %bc = pto.vmi.vbrc %source {group = 128}
      : !pto.vmi.vreg<128×f32> -> !pto.vmi.vreg<1024×f32>
  // → pto.as: 16 × pto.vselr (vselr gather realization)
  ```

- **notes:**
  - Fused `reduce→broadcast` (`vcadd`+`vbrc`) is the recognized fusion pattern:
    `pto.as` emits them back-to-back and keeps the result as a broadcast axis
    rather than materializing `K` copies.
  - Prefer `vdup` over a UB `BRC` reload for a single scalar.
  - Grouped broadcast has **no single native `pto.mi` op** — `pto.as` picks
    UB roundtrip (default, `vsts` partials + `vlds BRC_BLK`), `vselr` gather
    (when group count and K are tiny), or masked recompute (very small groups).

---

## Group 5: Reduce

> **Category:** B (VLane-aligned), C (unaligned sub-VLane).
> **Mask:** `Pg req` (governing mask is a required operand).
>
> Reduction ops collapse lanes into compact scalars, governed by a mask.
> `{group=C}` controls the number of sub-groups. Inactive lane behavior:
> `vcadd` treats inactive as 0; `vcmax`/`vcmin` treat inactive as `-∞`/`+∞`
> (fp) or type min/max (int).

The A5 VPTO backend supports `group = 1, 2, 4, 8` when the group count
divides `L`, for the following one-carrier shapes:

| Element type | Supported logical `L` |
|---|---|
| 8-bit integers (signless, signed, unsigned) | `1, 2, 4, 8, 64, 128, 256` |
| 16-bit integers and `f16` | `1, 2, 4, 8, 64, 128` |
| 32-bit integers and `f32` | `1, 2, 4, 8, 64` |

The compiler prefers executable native layouts, retaining the established
broadcast and memory paths. Where these are unavailable, a dense fallback
intersects each group's logical lane range with the governing mask and packs
the scalar results for grouped stores or broadcasts. Dynamic masks may contain
holes, empty groups, or partial final groups; inactive lanes retain the
identities described above.

A5 has no executable 8-bit row or VCG reduction form. Eight-bit integer inputs are
extended to 16 bits before reducing, with `L = 256` unpacked into two halves
inside instruction lowering. An empty min/max group retains the original
8-bit type's identity across extension. For integer singleton groups
(`group = L`), selection between the input and its identity replaces reduction.
Floating singleton groups retain reduction semantics for NaNs and signed zero.
Integer add results are narrowed modulo the result element width; native
16-bit VCG sums are packed from 32-bit results into 16-bit group slots.
Floating-point addition still requires `reassoc`.

The dense fallback is bounded by one 256-byte input carrier and at most eight
result slots. Existing executable 16/32-bit multi-carrier layouts remain
available; this does not add arbitrary `L` values or a general multi-carrier
8-bit fallback. Unsupported reduction requests are rejected with
`VMI-UNSUPPORTED` and the element type, `VL`, group count, and bytes per group,
before layout assignment and again during final conversion.

#### A5 integer result widths and group slots

The logical result keeps the input element width, but the hardware's sum may
be wider. In particular, **both** 16-bit integer `vcgadd` and `vcadd` produce
32-bit sums. Their different packing steps reflect how many independent
results are produced by each instruction:

| Physical path | Hardware result for 16-bit integer input | VMI result handling |
|---|---|---|
| Native `vcgadd` | Eight 32-bit sums in one register | Take each sum's low 16 bits with `vpack LOWER`, yielding consecutive `gs(8)` slots |
| Native `vcgmax` / `vcgmin` | Eight consecutive 16-bit extrema | Keep the values at their original width; no sum-narrowing pack |
| Compact `vcadd` | One 32-bit sum in lane zero per invocation | Use the widened result type, combine partial sums if needed, then select each group's low bits into the result packet |

For example, a 32-bit sum of `131091` has low/high 16-bit halves `19` and `2`.
Two such VCG sums appear as `[19, 2, 19, 2]` in a 16-bit view. The logical
16-bit results must be `[19, 19]`, not `[19, 2]`. The high halves are part of
the hardware sums; they are not additional groups or necessarily zero.

In the compact path, `getRowResultType()` selects the widened integer sum
type. After any partial sums are combined, a register bitcast exposes the
original-width low lane. `buildCompactPacket()` uses only that lane from each
group and assembles the logical slots. The bitcast itself does not pack or
clear the other physical lanes. Compact max/min likewise consume only the
extremum value, not the index returned by the row max/min instruction. For
8-bit inputs, the logical result is narrowed back to 8 bits after extension
and reduction. All integer sum narrowing follows modulo arithmetic at the
logical result width; it is not saturation. Floating-point reductions keep
their existing result types and semantics. See the physical contracts in
[Reduction Ops](./PTO-micro-Instruction-SPEC.md).

#### Follow-up exploration: retain strided native sums

A possible optimization is to expose a native 16-bit integer VCG sum packet
as `gs(8, 2)`: eight group slots whose values occupy 16-bit positions
`0, 2, 4, ..., 14`. Consumers could select those positions directly and pack
only when needed. This is a follow-up exploration, not a new result layout
selected by the current reduction implementation; the existing `gs(8)`
contract and its normalization remain in place.

Existing `gs(8)` / `gs(8, 2)` conversion rows provide part of the machinery,
but do not prove that every consumer can materialize the new producer layout.
A separate change must distinguish integer addition and element width in the
reduction capability rules, then audit grouped stores, broadcasts, subsequent
reductions, casts, masks, and multi-part partial-sum combines. Early validation,
layout selection, and final conversion must agree on the executable cases.

The investigation should compare complete producer/consumer chains, including
direct group stores and reduce-broadcast-store sequences. For example,
`ui16 L=64, group=8` produces eight group values; the preferred `ls(2)` store
for 64 elements describes the subsequent broadcast output, not those eight
input slots. It does not by itself establish a redundant pack/unpack pair.
Strided source slots can also add broadcast-index arithmetic. Measure `vpack`
counts together with total instructions, dependencies, register pressure, and
device timings, and rerun the integer overflow, empty/tail/holey-mask, direct
store, and broadcast device cases before claiming a benefit.

### `pto.vmi.vcadd`

- **semantics:** Masked add-reduction. When `{group=C}` is absent, reduces all
  `L` active lanes to a single scalar (`V<1×T>`).

  ```c
  // Without group: full reduction to scalar
  T sum = 0;
  for (int i = 0; i < L; i++)
      if (mask[i]) sum += src[i];
  dst[0] = sum;

  // With {group=C}: per-group reduction
  int gs = L / C;  // lanes per group
  for (int g = 0; g < C; g++) {
      T sum = 0;
      for (int i = 0; i < gs; i++)
          if (mask[g*gs + i]) sum += src[g*gs + i];
      dst[g] = sum;
  }
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vcadd %src, %mask {group = C, reassoc} : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<C×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.vreg<L×T>` | Source vector |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate (required) |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<C×T>` | Compact scalar vector (`C = 1` if no group) |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `group` | `1`, `2`, `4`, `8` | `1` (full reduce) | Number of sub-groups |
  | `reassoc` | *(unit attr)* | *(absent)* | Permit reassociation (**required** for fp sources) |
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-result behavior |

- **datatypes:** full reduce — `i32`, `f16`/`f32`; grouped reduce — `i8`/`i16`/`i32`, `f16`/`f32`
- **lowering to `pto.mi`:**

  | Group / W | Category | Physical lowering | `#mi` | `dep` |
  |---|---|---|---|---|
  | No group (`C=1`), `K=1` | B | `1 × pto.vcadd` | `1` | `1` |
  | No group, `K>1` (fold) | B | `(K-1) × vadd` + `1 × vcadd` | `K` | `K` |
  | No group, `K>1` (partial) | B | `K × vcadd` + combine | `K` | `1+⌈log₂K⌉` |
  | `group=8` (W=32B, VLane-aligned) | B | `K × pto.vcgadd` | `K` | `1` |
  | `group=2/4` (W=64B/128B aligned) | B | `(k-1) × vadd` fold + `vcgadd` | `K+k-1` | `k` |

- **example:**
  ```mlir
  // Full sum reduction (to scalar)
  %sum = pto.vmi.vcadd %x, %mask {reassoc}
      : !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<1×f32>

  // Grouped: 256-lane → 8 groups of 32, each VLane-aligned (W=32B)
  %sums = pto.vmi.vcadd %x, %mask {group = 8, reassoc}
      : !pto.vmi.vreg<256×f16>, !pto.vmi.mask<256> -> !pto.vmi.vreg<8×f16>
  ```

### `pto.vmi.vcmax` / `pto.vmi.vcmin`

- **semantics:** Masked max/min reduction.

  ```c
  // vcmax: inactive lanes treated as -∞
  T best = -INF;
  for (int i = 0; i < L; i++)
      if (mask[i]) best = max(best, src[i]);
  dst[0] = best;

  // vcmin: inactive lanes treated as +∞
  T best = +INF;
  for (int i = 0; i < L; i++)
      if (mask[i]) best = min(best, src[i]);
  dst[0] = best;
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vcmax %src, %mask {group = C} : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<C×T>
  ```
- **operands:** Same as `vcadd` (without `reassoc`).
- **results:** Same as `vcadd`.
- **attributes:** `group`, `pmode` (same as `vcadd`, no `reassoc`).
- **datatypes:** `i8`/`i16`/`i32`, `f16`/`f32`.
- **lowering to `pto.mi`:**

  | Group / W | Physical lowering |
  |---|---|
  | No group, fold | `(K-1) × vmax` + `1 × vcmax` |
  | VLane-aligned | `K × pto.vcgmax` / `K × pto.vcgmin` |

- **example:**
  ```mlir
  // Full max reduction
  %mx = pto.vmi.vcmax %x, %mask
      : !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<1×f32>

  // Grouped: 8-sub-group max (MX block-scale exponent pattern)
  %maxe = pto.vmi.vcmax %exp, %mask {group = 8}
      : !pto.vmi.vreg<256×ui16>, !pto.vmi.mask<256> -> !pto.vmi.vreg<8×ui16>
  ```

---

## Group 6: Convert

> **Category:** B (`vcvt`), A (`vinterpret_cast`).
> **Mask:** `Pg` (`vcvt`), none (`vinterpret_cast`).
>
> One logical `vcvt` whose target dtype IS the layout. `pto.as` expands it into
> the dtype-specific cast chain + part/width staging + matching store
> distribution. The author never spells `EVEN`/`ODD`, `P0`–`P3`, `PK`/`UNPK`,
> or `VL/2` addresses.

### `pto.vmi.vcvt`

- **semantics:** Unified elementwise type conversion. The conversion direction
  is derived from the source and destination element types; the verifier
  dispatches to one of seven kinds:

  1. **FpWiden** — `fp → fp`, `|dst| > |src|` (e.g. `f16 → f32`,
     `bf16 → f32`, `fp8_e4m3 → f16`, `f4x2 → bf16x2`).

  2. **FpNarrow** — `fp → fp`, `|dst| < |src|` (e.g. `f32 → f16`,
     `f32 → bf16`, `f32 → fp8_e4m3`, `bf16x2 → f4x2`). Same-width `fp → fp`
     (`|dst| == |src|`, e.g. `bf16 → f16`, `f16 → bf16`).

  3. **FpToSi** — `fp → signed int`. Supported pairs follow the contract
     table `lookupVMIFpToSiContract`: `f32→si32`, `f16→si16`, `f32→si16`,
     `f16→si8`, `f16→si32` (nosat), `bf16→si32`.

  4. **FpToUi** — `fp → unsigned int`. Supported pairs follow the contract
     table `lookupVMIFpToUIContract`: currently `f16→u8`.

  5. **SiToFp** — `signed int → fp`. The currently supported pairs are
     `si32 → f32` and `si8 → f16` (see the conversion contract matrix below).

  6. **IntWiden** — `int → int`, `|dst| > |src|`.

  7. **IntNarrow** — `int → int`, `|dst| < |src|`.

- **syntax:**
  ```mlir
  %r = pto.vmi.vcvt %src {rounding = "H", saturate = "SAT"} : !pto.vmi.vreg<L×T_src> -> !pto.vmi.vreg<L×T_dst>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.vreg<L×T_src>` | Source vector |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T_dst>` | Converted vector (same `L`, different `T`) |

- **attributes:**

  | Attribute | Values | Valid for | Description |
  |---|---|---|---|
  | `rounding` | `"R"` (nearest-even), `"A"` (away-from-zero), `"H"` (half-up), `"Z"` (toward-zero); for the `bf16x2→f4x2` contract pair the allowed set is `"R"`,`"A"`,`"F"` (floor), `"C"` (ceil), `"Z"` (toward-zero) — `"H"` is **rejected** | fp narrowing | Rounding mode |
  | `saturate` | `"SAT"`, `"NOSAT"` | required for fp-narrow / int-narrow; for fp→si / fp→ui the requirement follows the vcvt contract's `requiresSat` (e.g. `f16→si8` required, `f16→si32` **forbidden** — no overflow possible; same-width `bf16→f16` required, same-width `f16→bf16` **forbidden**); the `bf16x2→f4x2` narrow has `requiresSat=false` — any `saturate` is **forbidden**; `si32→si8` int-narrow accepts only `"NOSAT"` | For signed destinations, `SAT` clamps to `[min, max]`; for unsigned or signless destinations, it clamps to `[0, max]`. `NOSAT` performs a direct bit truncation of the result representation. |

- **datatypes:** Floating-point source and destination types are
  `{f32, f16, bf16, fp8_e4m3, fp8_e5m2}`. Integer types are
  `{si32, si16, si8, i32, i16, i8, ui32, ui16, ui8}`. The explicitly signed
  `si*` types are required when an integer is converted to floating point
  (`SiToFp`). Signless `i*` and unsigned `ui*` types cannot be used as
  `SiToFp` sources. Packed carrier types
  `{!pto.bf16x2, !pto.f4E1M2x2, !pto.f4E2M1x2}` are valid only for the
  bf16x2↔f4x2 fp-to-fp pair (see contract `lookupVMIFpToFpContract`).
  `bf16x2` is **conversion-only** — it may not appear as a compute element
  type (`vfadd`/`vfmul`/`vcmp`/...).

#### Conversion contract matrix

| Direction | Source → destination | `rounding` | `saturate` |
|---|---|---|---|
| FpWiden | `f16→f32`, `bf16→f32`, `fp8_e4m3→f16`, `fp8_e4m3→bf16`, `fp8_e4m3→f32`, `fp8_e5m2→f16`, `fp8_e5m2→bf16`, `fp8_e5m2→f32` | forbidden | forbidden |
| FpWiden (packed) | `f4E1M2x2→bf16x2`, `f4E2M1x2→bf16x2` | forbidden | forbidden |
| FpNarrow | `f32→f16`, `f32→bf16`, `f32→fp8_e4m3`, `f32→fp8_e5m2`, `f16→fp8_e4m3`, `f16→fp8_e5m2`, `bf16→fp8_e4m3`, `bf16→fp8_e5m2` | optional: `R`/`A`/`H`/`Z` (lowering defaults to `R`; `A` for hif8 targets) | required: `SAT`/`NOSAT` |
| FpNarrow (same width) | `bf16→f16` | optional: `R`/`A`/`H`/`Z` | required: `SAT`/`NOSAT` |
| FpNarrow (same width) | `f16→bf16` | optional: `R`/`A`/`H`/`Z` | forbidden |
| FpNarrow (packed) | `bf16x2→f4E1M2x2`, `bf16x2→f4E2M1x2` | optional: `R`/`A`/`F`/`C`/`Z` (`H` rejected) | forbidden |
| FpToSi | `f32→si32`, `f32→si16`, `f16→si16`, `f16→si8`, `bf16→si32` | optional: `R`/`A`/`F`/`C`/`Z` | required: `SAT`/`NOSAT` |
| FpToSi | `f16→si32` | optional: `R`/`A`/`F`/`C`/`Z` | forbidden |
| FpToUi | `f16→ui8`  | optional: `R`/`A`/`F`/`C`/`Z` | required: `SAT`/`NOSAT` |
| SiToFp | `si32→f32`, `si8→f16`  | forbidden | forbidden |
| IntWiden | any `si8/si16/si32`, `ui8/ui16/ui32`, pair with a wider destination (e.g. `ui8→ui16`, `si16→si32`,); same-width is rejected | forbidden | forbidden |
| IntNarrow | any `si8/si16/si32`, `ui8/ui16/ui32`, pair with a narrower destination (e.g. `ui32→ui8`, `si32→si16`) | forbidden | required: `SAT`/`NOSAT`; `si32→si8` accepts only `NOSAT` |

- **lowering to `pto.mi`:**

  | Conversion | Physical lowering | `#mi` | `dep` |
  |---|---|---|---|
  | SiToFp (`si32→f32`, current) | same-width `vcvt`, no part | `K` | `1` |
  | SiToFp (`si8→f16`, current) | widen `vcvt` EVEN/ODD | `2K` | `2` |
  | 16↔32 (radix-2) | `2K × vcvt EVEN/ODD` + predicate `ppack`/`punpack` companion | `2K` | `2` |
  | 8↔32 (radix-4) | widen: `UNPK_B8` + `vintlv` + `vcvt P0` + `punpack`; narrow: `PK4_B32` store (or `vselr` gather) + `ppack` | `2–3` | `2–3` |
  | f32→fp8 quant | `1 cast` + `PK4_B32` | `K` | `1` |
  | f32→int8 quant | 3-stage cast + `PK4_B32` | `~3K` | `3` |
  | fp↔fp same-width (`bf16→f16`, `f16→bf16`) | `K × vcvt` (1:1, no part) | `K` | `1` |
  | fp→si / fp→ui | per contract pair: same-width 1:1, widen EVEN/ODD, narrow EVEN/ODD+Vor | `K`–`~3K` | `2`–`3` |
  | int↔int (same width) | `K × vtrc` or `K × vcvt` | `K` | `1` |
  | `bf16x2→f4x2` narrow (32→8) | source viewed as raw `bf16` lanes (2 bf16/bf16x2); `vcvt{P0}` 1:1, `rnd` set, **no sat**; reuse prior pairing `vbitcast` when present | `K` | `1` |
  | `f4x2→bf16x2` widen (8→32) | `vcvt{P0}` produces `bf16` lanes; result-side `vbitcast` reinterprets them as `bf16x2`; no rnd, no sat | `K` | `1` |

  The width-family rows above do not imply that every source/destination
  signedness combination is exposed for `SiToFp`; use the conversion contract matrix as
  the normative list for signed-integer-to-floating-point conversions.

- **example:**
  ```mlir
  // fp16 → fp32 widen (radix-2, produces parity EVEN/ODD)
  %w = pto.vmi.vcvt %a
      : !pto.vmi.vreg<128×f16>
      -> !pto.vmi.vreg<128×f32>
  // → pto.as: 2 × pto.vcvt EVEN/ODD + ppack (parity companion)

  // fp32 → fp16 narrow with half-up rounding
  %n = pto.vmi.vcvt %y {rounding = "H", saturate = "SAT"}
      : !pto.vmi.vreg<64×f32> -> !pto.vmi.vreg<64×f16>

  // ui8 -> i16 unsigned extension
  %z = pto.vmi.vcvt %a
      : !pto.vmi.vreg<256×ui8> -> !pto.vmi.vreg<256×i16>

  // f32 → fp8 quantized narrow (saturate required)
  %q = pto.vmi.vcvt %s {saturate = "SAT"}
      : !pto.vmi.vreg<64×f32> -> !pto.vmi.vreg<64×fp8_e4m3>

  // i32 → i8 int-narrow without saturation (wrap on overflow)
  %t = pto.vmi.vcvt %v {saturate = "NOSAT"}
      : !pto.vmi.vreg<64×i32> -> !pto.vmi.vreg<64×i8>

  // f32 → si32 fp-to-si (saturate required)
  %r = pto.vmi.vcvt %x {saturate = "SAT"}
      : !pto.vmi.vreg<64×f32> -> !pto.vmi.vreg<64×si32>

  // bf16 → f16 same-width fp-to-fp (VPTO contract pair, routed via FpNarrow;
  // saturate required)
  %h = pto.vmi.vcvt %g {saturate = "SAT"}
      : !pto.vmi.vreg<128×bf16> -> !pto.vmi.vreg<128×f16>

  // f16 → u8 fp-to-ui (unsigned; contract pair, saturate required)
  %u = pto.vmi.vcvt %x {saturate = "SAT"}
      : !pto.vmi.vreg<128×f16> -> !pto.vmi.vreg<128×ui8>

  // si32 → f32 signed-integer to floating-point
  %sf = pto.vmi.vcvt %s
      : !pto.vmi.vreg<64×si32> -> !pto.vmi.vreg<64×f32>

  // si8 → f16 signed-integer to floating-point
  %hf = pto.vmi.vcvt %b8
      : !pto.vmi.vreg<128×si8> -> !pto.vmi.vreg<128×f16>

  // bf16x2 → f4x2 quantized narrow (rounding required; saturate forbidden;
  // bf16x2 arrives via a physical-noop vinterpret_cast pairing of 2 bf16 lanes)
  %pair = pto.vmi.vinterpret_cast %b
      : !pto.vmi.vreg<128×bf16> -> !pto.vmi.vreg<64×!pto.bf16x2>
  %q4 = pto.vmi.vcvt %pair {rounding = "R"}
      : !pto.vmi.vreg<64×!pto.bf16x2> -> !pto.vmi.vreg<64×!pto.f4E1M2x2>

  // f4x2 → bf16x2 dequant widen (no rounding, no saturate; bf16x2 is the
  // only legal bf16 carrier for f4 dequant; bare f4x2→bf16 is rejected)
  %d = pto.vmi.vcvt %f4
      : !pto.vmi.vreg<64×!pto.f4E1M2x2> -> !pto.vmi.vreg<64×!pto.bf16x2>
  ```

- **notes:**
  - `vcvt` **does not change lane count** — `src.L == dst.L` always. The
    physical register count `K` changes because `bitwidth(T)` changes.
  - The `part`/`parity`/`width` axes are lowering-only; the user never writes
    `EVEN`/`ODD`/`P0..P3`.
  - Radix-4 (8↔32) is **not** a stacked predicate chain and **not** a UB
    roundtrip; the 1↔4 lane spread rides data load/store distribution
    (`UNPK_B*`/`PK4_B32`) or a `vselr` byte-gather.
  - `bf16x2` is **conversion-only**: it is rejected as an element type by all
    compute verifiers (`vfadd`/`vfmul`/`vfma`/`vcmp`/`vcmps`/...). The only
    way to produce/consume `bf16x2` is via `vcvt` against `f4x2`, or a
    bit-conserving `vinterpret_cast` against `bf16` lanes.
  - The `bf16x2↔f4x2` pair is the only f4 conversion path exposed at VMI. The
    physical `pto.vcvt` consumes/produces raw `bf16` lanes; the `bf16x2`
    packaging is a `vbitcast` view inserted by lowering (`vinterpret_cast`
    from `128×bf16` to `64×!pto.bf16x2` is a physical no-op pairing).

### `pto.vmi.vinterpret_cast`

- **semantics:** Bitwise reinterpretation of a vector register — same bits,
  different element type. No data movement. The lane count may change so long
  as the total number of bits is conserved.

  ```c
  // Same bits, reinterpreted element-by-element
  memcpy(&dst, &src, L * sizeof(T_src));
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vinterpret_cast %src : !pto.vmi.vreg<L×T_src> -> !pto.vmi.vreg<L×T_dst>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.vreg<L×T_src>` | Source vector |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T_dst>` | Bit-reinterpreted vector |

- **attributes:** *(none)*
- **datatypes:** Any `T_src`, `T_dst` (including packed PTO types `!pto.bf16x2`, `!pto.f4E1M2x2`, `!pto.f4E2M1x2`) with `L · bitwidth(T_src) == L · bitwidth(T_dst)`
- **lowering to `pto.mi`:**
  ```
  K × pto.vbitcast (or no-op if same physical layout)
  ```
  `#mi = 0` or `K`, `dep = 0` or `1`.

- **notes:**
  - **Category A** — layout-transparent, no new axis produced.
  - This is **not** `vcvt` — no dtype cast chain, no `part`/`parity`/`width`
    axis, no `[pmode]`.
  - The user must ensure semantic legality (e.g., `f32` → `i32` bitcast is
    valid; `f32` → `f16` is not — use `vcvt` for that).

- **example:**
  ```mlir
  %r = pto.vmi.vinterpret_cast %a : !pto.vmi.vreg<64×f32> -> !pto.vmi.vreg<64×i32>
  ```

---

## Group 7: SFU

> **Category:** A (fused arithmetic, `vmull`), B (`vchist`, `vdhist`), C (gather/scatter).
> **Mask:** `Pg` on all except sort-like ops.
>
> Special-function / domain-accelerator ops. Mixed categories: `vchist`
> produces a `half` axis (B); `vdhist` yields a plain per-bin count (B);
> gather/scatter are Category C tile/permute ops; fused activation/arithmetic
> ops (including `vmull`, whose 64-bit product is split into a pair of `i32`
> results at the VMI surface) are Category A `vreg→vreg`.

### 7.1 Fused Arithmetic

#### `pto.vmi.vexpdif`

- **semantics:** Fused `exp(x − max)` for softmax numerical stability.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? exp(x[i] - max[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %e = pto.vmi.vexpdif %x, %max, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×f32>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.vreg<L×T>` | Input (`f16` or `f32`) |
  | `max` | `!pto.vmi.vreg<L×T>` | Subtracted max with the same type as `x` |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×f32>` | `exp(x − max)` (always `f32`) |

- **attributes:** `pmode` (`"zero"` / `"merge"`), default `"zero"`
- **datatypes:** `x` and `max`: matching `f16` or `f32`; result: `f32`
- **lowering to `pto.mi`:**
  ```
  f32: K × pto.vexpdif
  f16: 2K × pto.vexpdif
  ```
  Fuses `vsub` + `vexp`.

- **example:**
  ```mlir
  %e = pto.vmi.vexpdif %x, %max, %mask
      : !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64>
      -> !pto.vmi.vreg<64×f32>
  ```

#### `pto.vmi.vaxpy`

- **semantics:** Fused `α·x + y` (scale-add). Single hardware instruction.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (alpha * x[i] + acc[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %y = pto.vmi.vaxpy %x, %acc, %alpha, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, T, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.vreg<L×T>` | Input vector |
  | `acc` | `!pto.vmi.vreg<L×T>` | Accumulator (`y`) |
  | `alpha` | `T` (float scalar) | Scale factor |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | `α·x + acc` |

- **datatypes:** `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vaxpy
  ```
  `#mi = K`, `dep = 1`. Fuses `vmuls` + `vadd`.

#### `pto.vmi.vlrelu`

- **semantics:** Leaky ReLU: `y = x > 0 ? x : slope × x`. The slope is a
  scalar shared across all lanes.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (src[i] > 0 ? src[i] : slope * src[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %y = pto.vmi.vlrelu %x, %slope, %mask : !pto.vmi.vreg<L×T>, T, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.vreg<L×T>` | Input |
  | `slope` | `T` (float scalar) | Negative-slope multiplier |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **datatypes:** `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vlrelu
  ```
  `#mi = K`, `dep = 1`.

#### `pto.vmi.vprelu`

- **semantics:** Parametric ReLU: `y = max(x, 0) + alpha × min(x, 0)`. The
  `alpha` is a per-lane parameter vector (not a shared scalar).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (max(src[i], 0) + alpha[i] * min(src[i], 0)) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %y = pto.vmi.vprelu %x, %alpha, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.vreg<L×T>` | Input |
  | `alpha` | `!pto.vmi.vreg<L×T>` | Per-lane negative-slope parameter |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **datatypes:** `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vprelu
  ```
  `#mi = K`, `dep = 1`.

#### `pto.vmi.vmull`

- **semantics:** Widening 32-bit × 32-bit → 64-bit integer multiply. At the
  VMI surface the 64-bit product is **split into a pair of `i32` results**:
  `%low` carries the lower 32 bits and `%high` carries the upper 32 bits.
  This matches the shape of `pto.mi.vmull` one-to-one, so no `width` axis is
  introduced at the VMI layer. Signedness is inherited from the inputs
  (`i32 → (i32, i32)` uses arithmetic shift for the high half;
  `ui32 → (ui32, ui32)` uses logical shift).

  ```c
  for (int i = 0; i < L; i++) {
      // signed variant; use uint64_t for the ui32 form
      int64_t r = (int64_t)lhs[i] * (int64_t)rhs[i];
      low [i] = mask[i] ? (int32_t)(r & 0xFFFFFFFF)
                        : (pmode_merge ? low_old [i] : 0);
      high[i] = mask[i] ? (int32_t)(r >> 32)
                        : (pmode_merge ? high_old[i] : 0);
  }
  ```

- **syntax:**
  ```mlir
  %low, %high = pto.vmi.vmull %lhs, %rhs, %mask
      : !pto.vmi.vreg<L×i32>, !pto.vmi.vreg<L×i32>, !pto.vmi.mask<L>
        -> !pto.vmi.vreg<L×i32>, !pto.vmi.vreg<L×i32>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `a` | `!pto.vmi.vreg<L×i32>` | First operand |
  | `b` | `!pto.vmi.vreg<L×i32>` | Second operand |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `low`  | `!pto.vmi.vreg<L×i32>` | Lower 32 bits of the per-lane 64-bit product |
  | `high` | `!pto.vmi.vreg<L×i32>` | Upper 32 bits of the per-lane 64-bit product (arithmetic shift for `i32`; logical shift for `ui32`) |

- **attributes:**

  | Attribute | Type | Default | Description |
  |---|---|---|---|
  | `pmode` | `StrAttr` (`"zero"` \| `"merge"`) | `"zero"` | Predication mode. `"merge"` preserves the previous `low`/`high` lane values on inactive lanes; on A5 this is **not implemented**  (see [Appendix C](10-appendices.md)). |

- **datatypes:** `i32 → (i32, i32)`, `ui32 → (ui32, ui32)` (both result vregs share the input signedness).
- **lowering to `pto.mi`:**
  ```
  for k in [0, K):
      (low_k, high_k) = pto.mi.vmull(lhs_k, rhs_k, mask_k)
  ```
  `#mi = K`, `dep = 1`. Structurally 1:1 with `pto.mi.vmull`

- **example:**
  ```mlir
  %lo, %hi = pto.vmi.vmull %lhs, %rhs, %mask
      : !pto.vmi.vreg<64×i32>, !pto.vmi.vreg<64×i32>, !pto.vmi.mask<64>
        -> !pto.vmi.vreg<64×i32>, !pto.vmi.vreg<64×i32>
  ```

#### `pto.vmi.vmula`

- **semantics:** Fused multiply-add: `acc = acc + lhs × rhs`. Single hardware
  instruction. The accumulator is both an input and output (writes back).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (acc[i] + lhs[i] * rhs[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %acc1 = pto.vmi.vmula %acc, %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `acc` | `!pto.vmi.vreg<L×T>` | Accumulator (read-modify-write) |
  | `lhs` | `!pto.vmi.vreg<L×T>` | First multiply operand |
  | `rhs` | `!pto.vmi.vreg<L×T>` | Second multiply operand |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vmula
  ```
  `#mi = K`, `dep = 1`. Fuses `vmul` + `vadd`.

- **example:**
  ```mlir
  %acc1 = pto.vmi.vmula %acc, %a, %b, %mask
      : !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>,
        !pto.vmi.mask<64> -> !pto.vmi.vreg<64×f32>
  ```

### 7.2 Histogram

#### `pto.vmi.vchist`

`N` is the source/mask lane count; PTODSL requires both operands to have the
same `N` in `1/2/4/8/64/128/256`. Other counts, including `96`, raise `ValueError`
before the operation is constructed. Compiler-internal VMI types remain general.
`B` is the accumulator/result bin count: `128` or `256`, independent of `N`.
For example, `N=64, B=128` and `N=64, B=256` are both supported. Raw UB `ui8`
loads at 64/128 lanes use bounded 2/4-block reads at aligned addresses.

- **semantics:** **Cumulative histogram** over 8-bit source lanes
  (interpreted as unsigned). Counts per-bin occurrences over `%src` on top
  of a carry-in accumulator `%acc`. `B=256` returns bins 0–255 using
  Bin_N0 + Bin_N1; `B=128` returns bins 0–127 using Bin_N0 only. The result
  is a logical vector of `B` bins; the low/high split is a physical detail.
  The 128-bin form covers the full source range when samples are `< 128`.

  ```c
  // N source samples; B output bins (128 or 256)
  uint16_t dhist[256] = {0};
  for (int i = 0; i < N; i++)
      if (mask[i])
          dhist[src[i]]++;
  uint16_t chist[B];
  uint16_t cumulative = 0;
  for (int b = 0; b < B; b++) {
      cumulative += dhist[b];
      chist[b] = acc[b] + cumulative;
  }
  // B=128: Bin_N0 only; B=256: Bin_N0 and Bin_N1
  ```

- **syntax:**
  ```mlir
  // output is Bin_N0 + Bin_N1
  %h = pto.vmi.vchist %acc, %src, %mask
      : !pto.vmi.vreg<256xui16>, !pto.vmi.vreg<256xui8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<256xui16>

  // output is Bin_N0 when the source lanes are known to be < 128
  %h = pto.vmi.vchist %acc, %src, %mask
      : !pto.vmi.vreg<128xui16>, !pto.vmi.vreg<256xui8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<128xui16>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `acc`  | `!pto.vmi.vreg<B×{ui16\|i16}>` | Carry-in accumulator; same shape as `result` (256-bin Bin_N0+Bin_N1, or 128-bin Bin_N0-only). Element type is `ui16` or signless `i16` (interpreted as unsigned). |
  | `src`  | `!pto.vmi.vreg<N×{ui8\|i8}>` | Source lanes to be binned; 8-bit element type is `ui8` or signless `i8` (interpreted as unsigned). |
  | `mask` | `!pto.vmi.mask<N×pred>` | Governing predicate over source lanes. Does not gate `acc`. |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<B×{ui16\|i16}>` | Cumulative counts on top of `acc` (`B=128`: Bin_N0; `B=256`: Bin_N0+Bin_N1). Element type is `ui16` or signless `i16` (interpreted as unsigned). |

- **datatypes:** Source bin index: `ui8` or signless `i8`. Accumulator / result:
  `ui16` or signless `i16`. All are interpreted as
  unsigned; signed types (`si8` / `si16`) are rejected by the verifier.
- **lowering to `pto.mi`:**
  ```
  B=128: chistv2 Bin_N0 accumulator chain
  B=256: chistv2 Bin_N0 and Bin_N1 accumulator chains
  ```
  For `K = ceil(N / 256)` physical source chunks, this emits `K * (B / 128)`
  histogram instructions, excluding masks and memory operations. Thus public
  PTODSL sizes emit one instruction for `B=128`, or two for `B=256`. Bin_N1
  uses global cumulative semantics; no software prefix compensation is needed.

  The source operand must be `contiguous`. Raw UB inputs support one-lane
  scalar loads, aligned single-block short reads, aligned exact 64/128-byte multi-block
  reads, and full 256-byte register reads. The lowering intersects the input
  predicate with logical validity, so padding lanes never contribute.

- **example:**
  ```mlir
  // Cumulative histogram, full 256-bin (Bin_N0 + Bin_N1) output
  %h = pto.vmi.vchist %acc, %src, %mask
      : !pto.vmi.vreg<256xui16>, !pto.vmi.vreg<256xui8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<256xui16>
  // → two physical chistv2 instructions, one per 128-bin result part

  // N=64 samples, B=128 bins (sample values known to be < 128)
  %h0 = pto.vmi.vchist %acc0, %src64, %mask64
      : !pto.vmi.vreg<128xui16>, !pto.vmi.vreg<64xui8>, !pto.vmi.mask<64xpred>
     -> !pto.vmi.vreg<128xui16>

  // signless i16/i8 also accepted (interpreted as unsigned; acc and result must match)
  %hs = pto.vmi.vchist %acc, %src, %mask
      : !pto.vmi.vreg<256xi16>, !pto.vmi.vreg<256xi8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<256xi16>
  ```

#### `pto.vmi.vdhist`

`N` is the source/mask lane count; PTODSL requires both operands to have the
same `N` in `1/2/4/8/64/128/256`. Other counts, including `96`, raise `ValueError`
before the operation is constructed. Compiler-internal VMI types remain general.
`B` is the accumulator/result bin count: `128` or `256`, independent of `N`.
For example, `N=64, B=128` and `N=64, B=256` are both supported. Raw UB `ui8`
loads at 64/128 lanes use bounded 2/4-block reads at aligned addresses.

- **semantics:** **Distribution histogram** over 8-bit source lanes
  (interpreted as unsigned). Counts per-bin occurrences over `%src` on top
  of a carry-in accumulator `%acc`, yielding a logical vector of `B` per-bin
  counts. `B=128` returns bins 0–127; `B=256` returns all bins 0–255. The
  128-bin form covers the full source range when samples are `< 128`.

  ```c
  // N source samples; B output bins (128 or 256)
  uint16_t dhist[B];
  for (int b = 0; b < B; b++) dhist[b] = acc[b];     // carry-in
  for (int i = 0; i < N; i++)
      if (mask[i] && src[i] < B)
          dhist[src[i]]++;
  ```

- **syntax:**
  ```mlir
  // 256-bin full output
  %d = pto.vmi.vdhist %acc, %src, %mask
      : !pto.vmi.vreg<256xui16>, !pto.vmi.vreg<256xui8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<256xui16>

  // 128-bin output when the source lanes are known to be < 128
  %d = pto.vmi.vdhist %acc, %src, %mask
      : !pto.vmi.vreg<128xui16>, !pto.vmi.vreg<256xui8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<128xui16>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `acc`  | `!pto.vmi.vreg<B×{ui16\|i16}>` | Carry-in accumulator; same shape as `result` (`B=256`: bins 0–255; `B=128`: bins 0–127). Element type is `ui16` or signless `i16` (interpreted as unsigned). |
  | `src`  | `!pto.vmi.vreg<N×{ui8\|i8}>` | Source lanes to be binned; 8-bit element type is `ui8` or signless `i8` (interpreted as unsigned). |
  | `mask` | `!pto.vmi.mask<N×pred>` | Governing predicate over source lanes. Does not gate `acc`. |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<B×{ui16\|i16}>` | Plain per-bin count vector on top of `acc` (`B=256`: bins 0–255; `B=128`: bins 0–127). Element type is `ui16` or signless `i16` (interpreted as unsigned). |

- **datatypes:** Source bin index: `ui8` or signless `i8`. Accumulator / result:
  `ui16` or signless `i16`. All are interpreted as
  unsigned; signed types (`si8` / `si16`) are rejected by the verifier.
- **lowering to `pto.mi`:**
  ```
  B=128: dhistv2 Bin_N0 accumulator chain
  B=256: dhistv2 Bin_N0 and Bin_N1 accumulator chains
  ```
  For `K = ceil(N / 256)` physical source chunks, this emits `K * (B / 128)`
  histogram instructions, excluding masks and memory operations. Thus public
  PTODSL sizes emit one instruction for `B=128`, or two for `B=256`. The
  source and mask must be contiguous, and lowering intersects the b8 user
  mask with logical validity so padding lanes never contribute.

- **example:**
  ```mlir
  // Distribution histogram, plain per-bin count (256-bin full)
  %d = pto.vmi.vdhist %acc, %src, %mask
      : !pto.vmi.vreg<256xui16>, !pto.vmi.vreg<256xui8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<256xui16>

  // N=64 samples, B=128 bins (sample values known to be < 128)
  %d0 = pto.vmi.vdhist %acc0, %src64, %mask64
      : !pto.vmi.vreg<128xui16>, !pto.vmi.vreg<64xui8>, !pto.vmi.mask<64xpred>
     -> !pto.vmi.vreg<128xui16>

  // signless i16/i8 also accepted (interpreted as unsigned; acc and result must match)
  %ds = pto.vmi.vdhist %acc, %src, %mask
      : !pto.vmi.vreg<256xi16>, !pto.vmi.vreg<256xi8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<256xi16>
  ```

### 7.3 Gather / Scatter

> **Category C** — contiguous-required. `pto.as` materializes `.contiguous()`
> before these ops if the input layout is non-contiguous.

#### `pto.vmi.vgather`

- **semantics:** Indexed gather from UB at B32/B16 granularity. For each active
  lane `i`, load `src[offsets[i]]`.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? ub[base + offsets[i]] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  // B32 path
  %g = pto.vmi.vgather %src, %offsets, %mask
      : !pto.ptr<T, ub>, !pto.vmi.vreg<L×i32>, !pto.vmi.mask<L×b32> -> !pto.vmi.vreg<L×T>   // T in {i32,ui32,f32}

  // B16 path
  %g = pto.vmi.vgather %src, %offsets, %mask
      : !pto.ptr<T, ub>, !pto.vmi.vreg<L×ui16>, !pto.vmi.mask<L×b16> -> !pto.vmi.vreg<L×T>   // T in {i16,ui16,f16,bf16}
  %g = pto.vmi.vgather %src, %offsets, %mask
      : !pto.ptr<i8, ub>, !pto.vmi.vreg<L×ui16>, !pto.vmi.mask<L×b16> -> !pto.vmi.vreg<L×i16>
  %g = pto.vmi.vgather %src, %offsets, %mask
      : !pto.ptr<ui8, ub>, !pto.vmi.vreg<L×ui16>, !pto.vmi.mask<L×b16> -> !pto.vmi.vreg<L×ui16>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.ptr<T, ub>` | UB base pointer |
  | `offsets` | `!pto.vmi.vreg<L×i32>` or `!pto.vmi.vreg<L×ui16>` | Per-lane element offset |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:** `!pto.vmi.vreg<L×T>`
- **attributes:** `pmode`
- **datatypes:** B32 -- `i32`/`ui32`/`f32`; B16 -- `i16`/`ui16`/`f16`/`bf16`,
  plus `i8`/`ui8` -> `i16`/`ui16` zero-extension.
- **lowering:** B16 -> `K × pto.vgather2`; B32 -> `K × pto.vgather2_bc`.
  A statically all-active mask omits the trailing `vsel`.

#### `pto.vmi.vscatter`

- **semantics:** Indexed scatter to UB. For each active lane `i`,
  write `value[i]` to `dest[offsets[i]]`.

  ```c
  for (int i = 0; i < L; i++)
      if (mask[i])
          ub[base + offsets[i]] = value[i];
  ```

- **syntax:**
  ```mlir
  pto.vmi.vscatter %value, %dest, %offsets, %mask : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, !pto.vmi.vreg<L×i32>, !pto.vmi.mask<L>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `value` | `!pto.vmi.vreg<L×T>` | Values to scatter |
  | `dest` | `!pto.ptr<T, ub>` | UB destination base pointer |
  | `offsets` | `!pto.vmi.vreg<L×i32>` or `!pto.vmi.vreg<L×ui16>` | Per-lane element offset |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:** *(none)*
- **attributes:** `pmode`
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vscatter
  ```
  `#mi = K`, `dep = 1`.

  Value, offsets and mask use unit-stride `contiguous` layouts. The PTODSL
  public logical lane whitelist is `1/2/4/8/64/128/256`; invalid counts such as
  96 are rejected in Python. Internal VMI types remain general.

  B32/B16 support partial chunks by intersecting the user mask with logical
  validity. Full chunks do not need an extra intersection. B8 uses dense data
  and a logical b8 mask. Lowering zero-unpacks each low/high 128-byte half into
  the low bytes of B16 request slots, bitcasts back to the original byte type,
  unpacks the matching predicate half to b16, and masks the last request group.
  The number of physical scatters is `ceil(L/64)` for B32 and `ceil(L/128)`
  for B16/B8. Active indices must be valid and pairwise distinct.

- **example:**
  ```mlir
  pto.vmi.vscatter %v, %dest, %offsets, %mask
      : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, !pto.vmi.vreg<64×i32>, !pto.vmi.mask<64>
  ```

---

## Group 8: Predicate Generation Ops

> **Category:** gen (mask producers — take no input mask).
> **Mask in:** none (they generate masks).
>
> Mask generation is expressed with two ops: `create_mask` (prefix / first-N
> tail) and `create_group_mask` (grouped prefix / grouped first-N tail). Mask
> granularity (`b8`/`b16`/`b32`) is derived from the result type, not spelled in
> the op name.
>
> `create_mask` takes a single `index` operand `active_lanes`. When
> `active_lanes ≥ L` it yields an all-active mask; when `active_lanes = N < L`
> it yields a first-N tail mask. `create_group_mask` repeats the first-N pattern
> within each of `num_groups` equal groups (group size `group_size`).

```mlir
%act  = arith.minsi %rem, %cL   // min(rem, L)
%aidx = arith.index_cast %act   // i32 -> index
%mask = pto.vmi.create_mask %aidx : index -> !pto.vmi.mask<128×b32>
%next = arith.subi %rem, %act   // rem - min(rem, L)
```

### `pto.vmi.create_mask`

- **syntax:**
  ```mlir
  %m = pto.vmi.create_mask %active_lanes : index -> !pto.vmi.mask<L>
  ```
- **semantics:** Create a predicate mask where the first `active_lanes` logical
  lanes are active and the rest are inactive. `active_lanes ≥ L` produces an
  all-active mask; `active_lanes = N` produces a first-N tail mask.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = (i < active_lanes) ? 1 : 0;
  ```

- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `active_lanes` | `index` | Number of leading active lanes |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.mask<L>` | Predicate mask |

- **example:**
  ```mlir
  // All-active mask (active_lanes >= L)
  %all = pto.vmi.create_mask %c128 : index -> !pto.vmi.mask<128×b32>

  // First-N tail mask (N = 64)
  %tail = pto.vmi.create_mask %c64 : index -> !pto.vmi.mask<128×b32>
  ```

### `pto.vmi.create_group_mask`

- **syntax:**
  ```mlir
  %m = pto.vmi.create_group_mask %active_elems_per_group {num_groups = C, group_size = S}
      : index -> !pto.vmi.mask<L>
  ```
- **semantics:** Create a grouped predicate mask. The mask is divided into
  `num_groups` equal groups of `group_size` lanes each; lane `i` is active iff
  `(i % group_size) < active_elems_per_group`. When
  `active_elems_per_group ≥ group_size` all lanes are active within every group
  (grouped all-active); otherwise the first `active_elems_per_group` lanes are
  active within each group (grouped first-N tail).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = ((i % group_size) < active_elems_per_group) ? 1 : 0;
  ```

- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `active_elems_per_group` | `index` | Active lanes within each group |

- **attributes:**

  | Attribute | Values | Description |
  |---|---|---|
  | `num_groups` | positive integer | Number of equal groups |
  | `group_size` | positive integer | Lanes per group (`L / num_groups`) |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.mask<L>` | Grouped predicate mask |

- **example:**
  ```mlir
  // Grouped all-active: 8 groups, group size 32, all lanes active per group
  %all = pto.vmi.create_group_mask %c32 {num_groups = 8, group_size = 32}
      : index -> !pto.vmi.mask<256×b32>

  // Grouped first-N tail: first 25 lanes per group, 8 groups
  %tail = pto.vmi.create_group_mask %c25 {num_groups = 8, group_size = 32}
      : index -> !pto.vmi.mask<256×b32>
  ```

`num_groups` is logically legal for any positive divisor of the result mask
lane count. A backend may impose a narrower materialization limit separately.

### Mask Boolean Ops (`vand` / `vor` / `vxor` / `vnot` on masks)

The elementwise bitwise ops are reused directly on mask operands, treated as a
per-lane bit-wise boolean op on the predicate.

- **example:**
  ```mlir
  // Predicate boolean ops on masks
  %and = pto.vmi.vand %lt, %gt
      : !pto.vmi.mask<128xpred>, !pto.vmi.mask<128xpred>
      -> !pto.vmi.mask<128xpred>
  %or = pto.vmi.vor %lt, %gt
      : !pto.vmi.mask<128xpred>, !pto.vmi.mask<128xpred>
      -> !pto.vmi.mask<128xpred>
  %xor = pto.vmi.vxor %lt, %gt
      : !pto.vmi.mask<128xpred>, !pto.vmi.mask<128xpred>
      -> !pto.vmi.mask<128xpred>
  %not = pto.vmi.vnot %lt
      : !pto.vmi.mask<128xpred> -> !pto.vmi.mask<128xpred>
  ```

- **lowering:** mask logic is normalised onto the mask interface by `pto.as` in
  `vmi-lower-unified-to-legacy`: `vand` / `vor` / `vxor` / `vnot` on masks
  become `pto.vmi.mask_and` / `mask_or` / `mask_xor` / `mask_not`, which lower
  to `pto.pand` / `por` / `pxor` / `pnot`. The vreg form of the same spellings
  becomes the vreg interface (`pto.vmi.andi` / `ori` / `xori` / `not`), which
  keeps its governing predicate mask.

---

## Group 9: Data Rearrange

> **Category:** A (layout-transparent). **Mask:** `Pg`.
>
> In-register data movement and permutation. No UB access. `vintlv`/`vdintlv`
> are per-lane, dtype-preserving ops that do not change vreg layout — the output
> has the same `L` and `T` as the inputs. Commonly used for real+imaginary and
> value+index interleaving within a single vector register.

### `pto.vmi.vintlv`

- **semantics:** Interleave two source vectors by even/odd lanes.

  ```c
  // low  = {lhs[0], rhs[0], lhs[1], rhs[1], ..., lhs[L/2-1], rhs[L/2-1]}
  // high = {lhs[L/2], rhs[L/2], lhs[L/2+1], rhs[L/2+1], ...}
  for (int i = 0; i < L/2; i++) {
      lo[2*i]     = lhs[i];
      lo[2*i + 1] = rhs[i];
      hi[2*i]     = lhs[L/2 + i];
      hi[2*i + 1] = rhs[L/2 + i];
  }
  ```

- **syntax:**
  ```mlir
  %lo, %hi = pto.vmi.vintlv %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `lhs` | `!pto.vmi.vreg<L×T>` | First source (provides low-half even slots) |
  | `rhs` | `!pto.vmi.vreg<L×T>` | Second source (provides low-half odd slots) |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `low` | `!pto.vmi.vreg<L×T>` | Even-odd interleaved low half |
  | `high` | `!pto.vmi.vreg<L×T>` | Even-odd interleaved high half |

- **attributes:** `pmode`
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vintlv
  ```
  `#mi = K`, `dep = 1`. Layout-transparent (Category A).

- **example:**
  ```mlir
  %lo, %hi = pto.vmi.vintlv %a, %b, %mask
      : !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64>
      -> !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>
  ```

### `pto.vmi.vdintlv`

- **semantics:** Deinterleave a paired-source by even/odd lanes (AoS → SoA).

  ```c
  // even = {lhs[0], lhs[2], ..., lhs[L-2], rhs[0], rhs[2], ..., rhs[L-2]}
  // odd  = {lhs[1], lhs[3], ..., lhs[L-1], rhs[1], rhs[3], ..., rhs[L-1]}
  for (int i = 0; i < L/2; i++) {
      even[i]       = lhs[2*i];      // even-indexed slots of lhs
      even[L/2 + i] = rhs[2*i];      // even-indexed slots of rhs
      odd[i]        = lhs[2*i + 1];  // odd-indexed slots of lhs
      odd[L/2 + i]  = rhs[2*i + 1];  // odd-indexed slots of rhs
  }
  ```

- **syntax:**
  ```mlir
  %even, %odd = pto.vmi.vdintlv %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>
  ```
- **operands:** Same shape as `vintlv`.
- **results:** Same shape as `vintlv` (two `!pto.vmi.vreg<L×T>`).
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vdintlv
  ```
  `#mi = K`, `dep = 1`.

- **example:**
  ```mlir
  %even, %odd = pto.vmi.vdintlv %x, %y, %mask
      : !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64>
      -> !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>
  ```

- **notes:**
  - `vintlv` and `vdintlv` are inverses: `vdintlv(vintlv(a, b))` recovers `(a, b)`.
  - Both are Category A — they do **not** change vreg layout (parity/half/width
    axes pass through unchanged).
  - Common use cases: real+imaginary interleave, value+index pair manipulation,
    complex number arithmetic.

---

## Appendix A: Unified Ops Index

| # | Op | Group | Category | Brief |
|---|---|---|---|---|
| 1 | `pto.vmi.vload` | 1: Load/Store | A | Logical vector load from UB |
| 2 | `pto.vmi.vstore` | 1: Load/Store | A | Logical vector store to UB |
| 3 | `pto.vmi.vci` | 2: Index-gen | A | Lane-index vector generation |
| 4 | `pto.vmi.vadd` | 3: Eltwise | A | Elementwise add (fp+int unified) |
| 5 | `pto.vmi.vsub` | 3: Eltwise | A | Elementwise subtract |
| 6 | `pto.vmi.vmul` | 3: Eltwise | A | Elementwise multiply |
| 7 | `pto.vmi.vdiv` | 3: Eltwise | A | Elementwise divide (fp only) |
| 8 | `pto.vmi.vmax` | 3: Eltwise | A | Elementwise maximum |
| 9 | `pto.vmi.vmin` | 3: Eltwise | A | Elementwise minimum |
| 10 | `pto.vmi.vabs` | 3: Eltwise | A | Elementwise absolute value |
| 11 | `pto.vmi.vneg` | 3: Eltwise | A | Elementwise negate |
| 12 | `pto.vmi.vrelu` | 3: Eltwise | A | Elementwise ReLU |
| 13 | `pto.vmi.vexp` | 3: Eltwise | A | Elementwise exponential |
| 14 | `pto.vmi.vln` | 3: Eltwise | A | Elementwise natural log |
| 15 | `pto.vmi.vsqrt` | 3: Eltwise | A | Elementwise square root |
| 16 | `pto.vmi.vand` | 3: Eltwise | A | Elementwise bitwise AND |
| 17 | `pto.vmi.vor` | 3: Eltwise | A | Elementwise bitwise OR |
| 18 | `pto.vmi.vxor` | 3: Eltwise | A | Elementwise bitwise XOR |
| 19 | `pto.vmi.vnot` | 3: Eltwise | A | Elementwise bitwise NOT |
| 20 | `pto.vmi.vshl` | 3: Eltwise | A | Elementwise left shift |
| 21 | `pto.vmi.vshr` | 3: Eltwise | A | Elementwise right shift (arithmetic for signed, logical for unsigned) |
| 22 | `pto.vmi.vadds` | 3: Eltwise | A | Vector-scalar add |
| 23 | `pto.vmi.vmuls` | 3: Eltwise | A | Vector-scalar multiply |
| 24 | `pto.vmi.vmaxs` | 3: Eltwise | A | Vector-scalar maximum |
| 25 | `pto.vmi.vmins` | 3: Eltwise | A | Vector-scalar minimum |
| 26 | `pto.vmi.vshls` | 3: Eltwise | A | Vector-scalar shift left |
| 27 | `pto.vmi.vshrs` | 3: Eltwise | A | Vector-scalar shift right |
| 28 | `pto.vmi.vcmp` | 3: Eltwise | A | Elementwise compare → mask |
| 29 | `pto.vmi.vcmps` | 3: Eltwise | A | Vector-scalar compare → mask |
| 30 | `pto.vmi.vsel` | 3: Eltwise | A | Predicate select |
| 31 | `pto.vmi.vselr` | 3: Eltwise | C | Dynamic lane select; contiguous, supported logical shape |
| 32 | `pto.vmi.vbrc` | 4: Broadcast | A/B | Broadcast scalar/group-slot |
| 33 | `pto.vmi.vcadd` | 5: Reduce | B | Add-reduction |
| 34 | `pto.vmi.vcmax` | 5: Reduce | B | Max-reduction |
| 35 | `pto.vmi.vcmin` | 5: Reduce | B | Min-reduction |
| 36 | `pto.vmi.vcvt` | 6: Convert | B | Unified type conversion |
| 37 | `pto.vmi.vinterpret_cast` | 6: Convert | A | Bitwise reinterpret |
| 38 | `pto.vmi.vexpdif` | 7: SFU | A | Fused exp(x−max) |
| 39 | `pto.vmi.vaxpy` | 7: SFU | A | Fused α·x+y |
| 40 | `pto.vmi.vlrelu` | 7: SFU | A | Leaky ReLU |
| 41 | `pto.vmi.vprelu` | 7: SFU | A | Parametric ReLU |
| 42 | `pto.vmi.vmull` | 7: SFU | A | Widening 32×32 multiply, split into (`low`, `high`) `i32` pair |
| 43 | `pto.vmi.vmula` | 7: SFU | A | Fused multiply-add |
| 44 | `pto.vmi.vchist` | 7: SFU | B | Cumulative histogram (half-axis) |
| 45 | `pto.vmi.vdhist` | 7: SFU | B | Distribution histogram (plain per-bin) |
| 46 | `pto.vmi.vgather` | 7: SFU | C | Indexed gather (B32/B16) |
| 47 | `pto.vmi.vscatter` | 7: SFU | C | Indexed scatter |
| 48 | `pto.vmi.create_mask` | 8: Predicate | gen | Prefix / first-N tail mask |
| 49 | `pto.vmi.create_group_mask` | 8: Predicate | gen | Grouped predicate mask |
| 50 | `pto.vmi.vintlv` | 9: Rearrange | A | Interleave two vectors |
| 51 | `pto.vmi.vdintlv` | 9: Rearrange | A | Deinterleave two vectors |
| 52 | `pto.vmi.vaddc` | 3: Eltwise | A | 32-bit integer add with per-lane carry output |
| 53 | `pto.vmi.vsubc` | 3: Eltwise | A | 32-bit integer subtract with per-lane not-borrow output |
| 54 | `pto.vmi.vaddcs` | 3: Eltwise | A | 32-bit integer add with carry input and output |
| 55 | `pto.vmi.vsubcs` | 3: Eltwise | A | 32-bit integer subtract with carry input and output |

## Appendix C: MERGE Mode on A5

On A5, the hardware predicates only in **ZEROING** mode (inactive lanes → 0).
MERGE mode is **not implemented yet**:

Until emulated or native MERGE support lands, write the merge explicitly with
`vsel` against the old destination value:

```mlir
// Explicit merge:  dst = Pg ? op(a, b) : dst_old
%new = pto.vmi.<op> %a, %b, %pg           // ZEROING: inactive lanes → 0
%dst = pto.vmi.vsel %pg, %new, %dst_old   // keep old value on inactive lanes
```

Once emulation is implemented, the compiler is expected to expand MERGE as
`vnot` + zeroing op + `vand`/`vor` (cost: `+1 vnot` per distinct `Pg`, plus
`+K vsel`/`vor`); on A6, merge-capable ops are expected to take the mode
natively and collapse to the single predicated op.
