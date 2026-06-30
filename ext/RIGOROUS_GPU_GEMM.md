# Rigorous GPU GEMM for BallArithmetic.jl — experiment notes

Goal: compute `MMul4` (the rigorous ball-arithmetic matrix product) on NVIDIA
GPUs, exploring several routes — a directed-rounding cuTile kernel, a
TF32-tensor-core "Ozaki" scheme via cuBLAS, and INT8 Ozaki-Scheme-I/II. All
experiments below were run on an **RTX 4060 Ti** (Ada, AD106, consumer); the
scripts live in `experiments/cutile_probe/`.

> **Attribution.** `experiments/cutile_probe/gemmul8_port.jl` is a **Julia /
> CUDA.jl port of RIKEN R-CCS's [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8)**
> (Ozaki Scheme II, MIT License, © 2025- RIKEN R-CCS). The MIT notice is
> reproduced in that file's header. See the References section.

`MMul4` recap: `C2 = mA·mB + rC` (RoundUp), `C1 = mA·mB − rC` (RoundDown),
`rC = |mA|·rB + rA·(|mB|+rB)` (RoundUp), `mC = (C1+C2)/2`, `rC = mC − C1`. The
result is a ball matrix guaranteed to contain the true product.

---

## Route 1 — directed-rounding cuTile kernel (`CuTileExt.jl`)

cuTile's `@fpmode rounding_mode=…` lets a kernel pick the FP rounding direction,
which is exactly what directed-rounding ball arithmetic needs.

**Finding (blocker for tensor cores):** `@fpmode` is honored by the *elementwise*
tile ops (`addf`, `mulf`, `subf`, …) but **silently ignored** by the tile-level
`mma` / `muladd(tile,tile,tile)` matmul — tensor-core MMA is hardware-fixed to
round-to-nearest-even, and `encode_MmaFOp!` never receives the fpmode attribute.
Running the same tile matmul under `PosInf`, `NegInf`, `NearestEven` gives
**bitwise-identical** output (`probe.jl`).

→ Filed upstream as **JuliaGPU/cuTile.jl issue #229** (ask: docs / an IRError on
misuse / eventual plumbing). Draft in `experiments/cutile_probe/issue_draft.md`.

**Workaround that works:** express the matmul as **outer-product accumulation**
of `(tm×1)×(1×tn)` tiles inside the `@fpmode` scope — these lower to `mulf`/`addf`
which *do* honor the mode. This is what `CuTileExt.jl` implements: three
`@fpmode` scopes (RoundUp for `C2`/`rC`, RoundDown for `C1`, RoundUp for the
final mid/rad). Validated rigorously — every output ball overlaps the CPU
`MMul4` (`probe_mmul4.jl`, `smoke_test.jl`, 8 cases). It is correct, but bypasses
tensor cores.

### Float64 benchmark — GPU (CuTileExt) vs CPU (multithreaded OpenBLAS)

`bench_mmul4.jl`, median of 5 runs, overlap-verified each size:

| Size | CPU | GPU | GPU vs CPU |
|------|-----|-----|-----------|
| 128³ | 0.34 ms | 0.31 ms | 1.09× (tie, overhead-bound) |
| 256³ | 1.36 ms | 2.19 ms | 0.62× |
| 512³ | 11.2 ms | 16.6 ms | 0.67× |
| 1024³ | 55.5 ms | 118.9 ms | 0.47× |
| 2048³ | 431 ms | 934 ms | 0.46× |

CPU throughput climbs to ~80 GF/s as threading amortizes; the GPU plateaus at
~37 GF/s (FP64 compute-bound). **Float64 rigorous GEMM is a net loss on consumer
NVIDIA hardware** — two compounding reasons:

1. No tensor cores (forced by the `@fpmode`/`mma` limitation above).
2. Consumer Ada FP64 runs at 1/64 of FP32 and has **no** FP64 tensor cores.

This is hardware-specific: FP64 tensor cores exist only on the datacenter dies
(A100/GA100, H100/GH100, Blackwell). On an A100 the *same* outer-product kernel
would be limited by ~9.7 TF of FP64 (½ the FP32 rate) instead of the consumer
1/64 cap — comfortably above the CPU baseline. Note: the A40 on toeplitz is
GA102-based and has the consumer FP64 penalty, so it would **not** help Float64.

---

## Route 2 — TF32-Ozaki ball GEMM via cuBLAS

Idea: TF32 tensor cores round FP32 inputs to an 11-bit mantissa (their ~5e-4
error) but **accumulate in FP32**. Pre-split each input into TF32-exact slices
so the tensor core can't round them; then every per-element product (11×11 = 22
bits) is exact in the FP32 accumulator, and only the K-fold sum rounds —
recovering ~FP32 accuracy at tensor-core throughput. No custom kernel needed:
it's just cuBLAS GEMMs.

```
A = A0 + A1 + eA   (Veltkamp split, s=13 → A0,A1 each 11 significant bits = TF32-exact)
A·B ≈ A0·B0 + A0·B1 + A1·B0 + A1·B1     (four TF32 GEMMs, eA·… dropped & bounded)
```

### Gotcha: `*` never uses TF32

`A * B` for `Float32` CuArrays dispatches to plain `cublasSgemm`, which **always**
computes in full FP32 and ignores `math_mode`. TF32 is only reachable by calling
`CUDA.CUBLAS.gemmEx!('N','N', 1f0, A, B, 0f0, C)` under
`CUDACore.math_mode!(FAST_MATH)` (→ compute type `CUBLAS_COMPUTE_32F_FAST_TF32`,
with `math_precision() == :TensorFloat32`). This explains an earlier "TF32 not
engaging" red herring: through `*`, FP32 and "TF32" are bitwise identical at all
sizes. (`probe_tf32_ozaki.jl`.)

### Accuracy — Ozaki recovers near-FP32 on TF32

Relative error vs a Float64 reference (`probe_tf32_ozaki.jl`):

| Size | FP32 | naive TF32 | Ozaki-4 | Ozaki-3 (drop A1·B1) |
|------|------|-----------|---------|----------------------|
| 256³ | 2.2e-7 | 3.0e-4 | 1.1e-6 | 1.1e-6 |
| 512³ | 5.4e-7 | 3.2e-4 | 2.5e-6 | 2.5e-6 |
| 1024³ | 2.7e-7 | 2.6e-4 | 4.0e-6 | 4.0e-6 |
| 2048³ | 4.1e-7 | 3.2e-4 | 8.6e-6 | 8.6e-6 |

Ozaki improves the TF32 center by **30–300×** (3e-4 → ~1e-6), to within ~20× of
full FP32. The `A1·B1` term is negligible (Ozaki-3 == Ozaki-4) → **3 GEMMs
suffice**. The error grows like √K (the FP32 accumulation of the split residual),
as expected.

### Rigor — verified enclosure

Radius `γ·(|A|·|B|)` with `γ = 2·2⁻²² + (K+4)·2⁻²⁴`, `(|A|·|B|)` computed in FP32
and inflated to a rigorous upper bound. Checked against a 200-bit **BigFloat**
truth (`probe_tf32_rigor.jl`):

| Size | enclosure | center max err | radius (median rel) |
|------|-----------|----------------|---------------------|
| 256³ | **OK — all entries contained** | 7.8e-5 | 2.4e-4 |
| 512³ | **OK — all entries contained** | 2.7e-4 | 6.7e-4 |

The radius is dominated by the `|A|·|B| / |A·B|` amplification (≈√K for `randn`
inputs), which is intrinsic to *any* ball/interval GEMM — an FP32 `MMul4` carries
the same factor. So the balls are about as tight as an FP32 `MMul4`, with the
center produced on the TF32 path.

### Timing — net loss on the 4060 Ti

`probe_tf32_rigor.jl`, median of 6 runs:

| Size | FP32 gemm | TF32 gemm | Ozaki center (3–4 gemm) | center + radius |
|------|-----------|-----------|-------------------------|-----------------|
| 1024³ | 0.21 ms | 0.14 ms | 0.81 ms (3.8× FP32) | 1.11 ms (5.2× FP32) |
| 2048³ | 1.37 ms | 0.92 ms | 5.27 ms (3.8× FP32) | 7.03 ms (5.1× FP32) |
| 4096³ | 10.5 ms | 7.0 ms | 34.4 ms (3.3× FP32) | 46.7 ms (4.5× FP32) |

On this card a single TF32 gemm is only **~1.5× faster** than FP32, so 3–4 TF32
gemms cost **~3.3–3.8×** a single FP32 gemm — a clear loss. The whole premise
needs TF32 ≫ FP32 to pay off, and that ratio is small on consumer Ada.

---

## Route 3 — INT8 tensor cores via Ozaki Scheme II (recommended direction)

**Reference:** K. Ozaki, Y. Uchino, T. Imamura, *"Ozaki Scheme II: A GEMM-oriented
emulation of floating-point matrix multiplication using an integer modular
technique"*, arXiv:2504.08009 (2025).

This route sidesteps **both** blockers above. The key property:
**INT8×INT8 → INT32 tensor-core products are EXACT** — integer MMA does no
rounding at all (provided the accumulation stays within INT32 range). So:

* Route 1's blocker (tensor-core MMA ignores `@fpmode`) is irrelevant — there is
  no rounding to direct.
* Route 2's blocker (TF32 round-to-nearest, only ~1.5× FP32 on consumer Ada) is
  gone — exact INT8 is *much* faster than FP32 on consumer cards.

**Method.** Scale FP matrices to integers with exact power-of-two diagonal
scaling `C = AB = D⁻¹(DA)(BE)E⁻¹`; reduce modulo a set of coprime moduli
`mᵢ ∈ [191,256]`; compute one **exact** INT8 GEMM per modulus (`cublasGemmEx`,
INT8→INT32, with `m,n,k` multiples of 4); reconstruct the exact integer product
via the **Chinese Remainder Theorem**; scale back. The *only* error is the
deliberate input truncation, controlled by the number of moduli `s` (k bits,
`k = ⌊log₂(M/2q)/2⌋`). FP64-equivalent accuracy needs `s ≈ 14–16` GEMMs
(vs ~28–35 for the original FP-splitting Ozaki Scheme I).

**Why this is the right fit for ball arithmetic.** The integer products are
exact and the CRT reconstruction is exact, so the computed center is the *exact*
product of the truncated inputs. The radius only has to cover the input
truncation `A = A' + ΔA` (a known, a-priori bound) — there is **no accumulation
rounding to bound**. Adding moduli shrinks the radius arbitrarily. This is
strictly tighter and cleaner than the TF32-Ozaki radius of Route 2.

**Why it should win even on consumer hardware.** Consumer cards have fast INT8
tensor cores but crippled FP64. Paper results:

| Hardware | FP64-equiv via INT8-Ozaki | native FP64 | speedup |
|----------|---------------------------|-------------|---------|
| RTX 4090 (consumer Ada) | 7.4–9.8 TFLOPS (s=14–15) | 1.29 TFLOPS | ~6–7× |
| GH200 | 56.6–80.2 TFLOPS (s=14) | 61.9 TFLOPS (FP64 TC) | ~comparable |

The RTX 4090 is the same consumer-Ada situation as the **RTX 4060 Ti** here, so
INT8-Ozaki is the route most likely to make rigorous **Float64** GEMM a *win* on
this card — beating both native GPU FP64 and plausibly the multithreaded CPU.

The INT8 GEMM primitive is reached through the same path we use elsewhere:
`CUDA.CUBLAS.gemmEx!('N','N', Int32(1), A_int8, B_int8, Int32(0), C_int32)`
(sig `(Int8,Int32)` → `CUBLAS_COMPUTE_32I`, requires `m,n,k % 4 == 0`). Verified
**bit-exact** vs an integer CPU reference.

### Prototype results (slice-based Scheme I, this repo)

Implemented as a slice split (base `2^7`, per-row/col power-of-two scaling),
exact INT8 GEMMs for pairs `i+j ≤ T`, FP64 recombination. Scripts:
`probe_int8_ozaki.jl` (accuracy), `probe_int8_bench.jl` (timing),
`probe_int8_rigor.jl` (enclosure), `probe_int8_streams.jl` (breakdown).

**Accuracy** (relerr vs 250-bit BigFloat, randn inputs) — tunable via `T`:

| s, T | #GEMMs | 256³ | 512³ |
|------|--------|------|------|
| native FP64 gemm | 1 | 1.1e-15 | 1.4e-15 |
| 8, 8 | 28 | 4.1e-14 | 5.4e-14 |
| 8, 10 | 43 | **4.6e-16** | **5.4e-16** |
| 8, 16 | 64 | 4.6e-16 | 5.4e-16 |

→ 43 GEMMs **match/beat native FP64**; full reconstruction at 64.

**Timing vs native FP64 gemm (RTX 4060 Ti)** — INT8-Ozaki *wins at scale*:

| Size | native FP64 | single INT8 | INT8/FP64 per-gemm | Ozaki T=8 (28) | Ozaki T=10 (43) |
|------|-------------|-------------|--------------------|----------------|------------------|
| 512³ | 1.0 ms | 0.04 ms | 26× | 0.48× | 0.38× |
| 1024³ | 7.6 ms | 0.09 ms | 82× | 0.97× | 0.82× |
| 2048³ | 56 ms | 0.47 ms | 120× | **1.13×** | 0.92× |
| 4096³ | 413 ms | 3.3 ms | 126× | **1.65×** | **1.28×** |

A single INT8 GEMM is up to **126× faster** than FP64; crossover ~1–2k. (More
modest than the paper's 6–7× because this is Scheme I with 28–43 GEMMs, not
Scheme II's ~16, and the FP64 recombination is unoptimized — see below.)

**Timing vs the CPU** (`probe_int8_vs_cpu.jl`, 6-thread OpenBLAS). NOTE: this
compares the GPU **midpoint** product against the CPU **full `MMul4`** (mid +
radius) — *not* apples-to-apples; it overstates the GPU win. For the corrected
full-`MMul4`-vs-full-`MMul4` figure (~6–8×) see the GEMMul8-port section. With
that caveat, the midpoint-vs-CPU-MMul4 table:

| Size | CPU FP64 gemm (non-rig.) | CPU MMul4 (rig.) | GPU INT8-Ozaki midpoint (rig.) | ratio |
|------|--------------------------|-------------------|--------------------------------|-------|
| 1024³ | 8.0 ms | 61 ms | 9.4 ms | 6.6× |
| 2048³ | 57 ms | 661 ms | 61 ms | 10.8× |
| 4096³ | 452 ms | 2926 ms | 326 ms | 9.0× |

The CPU `MMul4` is itself ~8× a plain gemm (directed-rounding GEMMs through the
slow ConsistentFPCSR BLAS). The GPU midpoint is also ≥ the CPU's *non-rigorous*
gemm (1.0× → 1.8×). This reverses Route 1, where CuTileExt Float64 *lost* to the
CPU by ~2×. (Again: the full GPU `MMul4` adds a cheap radius → ~6–8× overall.)

**Rigor** — verified enclosure of the exact `A·B` vs 300-bit BigFloat:
**all entries contained** at 256³/512³. Two radius bounds:
- *Loose* (`probe_int8_rigor.jl`): `loA·fullB + fullA·loB`, ~10⁶× the true error
  (over-counts non-dropped pairs `i+j ≤ T`).
- *Tight* (`probe_int8_rigor_tight.jl`, **done**): the per-`i` tail-sum form
  `Σ_i x_i·Ytail_{T-i}` in FP32 (≤`s` extra GEMMs) + a rigorous recombination
  bound (`Σ|terms|·u`) + residual bound (row/col sums, no GEMM). At T=10 this
  gives **radius 1.1e-14 rel, only ~20–24× the true center error** — tighter than
  a native FP64 `MMul4` (~1e-13), and at the `√K` cancellation floor that any
  rigorous ball GEMM pays.

| Size | T | radius (rel) | radius/err | gemms |
|------|---|--------------|------------|-------|
| 256³ | 8 | 6.8e-13 | 16.6× | 28 INT8 + 8 FP32 |
| 256³ | 10 | 1.1e-14 | 23.8× | 43 INT8 + 6 FP32 |
| 512³ | 10 | 1.1e-14 | 20.0× | 43 INT8 + 6 FP32 |

**Bottleneck / parallelism** (`probe_int8_streams.jl`, 4096³, 43 GEMMs):

| Stage | Time |
|-------|------|
| split (FP64→INT8) | 93 ms |
| 43 GEMMs sequential | 150 ms |
| 43 GEMMs, 4 / 8 streams | 150 / 152 ms (**no speedup**) |
| FP64 recombination only | 60 ms |
| full (interleaved) | 209 ms |

Key findings: (1) **spreading GEMMs over streams does nothing** — one INT8 GEMM
already saturates the tensor cores at this size, so concurrent streams just
time-slice. (2) The FP64 **split + recombine (153 ms) rivals the GEMMs (150 ms)**
and the interleaved pipeline does *not* overlap them (209 ≈ 150 + 60). The real
wins are: overlap the memory-bound recombine with the next tensor-core GEMM
(2-stream producer/consumer); **group recombination by `i+j`** (sum Int32 in
integer, one FP64 convert+scale per diagonal → ~15 reduces, not 43); and move to
**Scheme II** (CRT, ~16 GEMMs) to cut both GEMM and recombination cost.

### Scheme II (CRT) — implemented & verified, but reconstruction-bound

The CRT variant (arXiv:2504.08009) replaces the `~T²/2` slice-pair GEMMs with
**one GEMM per modulus**: scale to integer `A',B'`; for each of `s` coprime
moduli `mₜ∈[191,256]` reduce to balanced INT8 residues, GEMM exactly to INT32,
take `mod mₜ` → `X mod mₜ`; reconstruct the exact integer `X = A'·B'` by CRT
(Garner), then `C = 2^(σA+σB−2k)·X`. (`probe_int8_schemeII.jl`,
`probe_schemeII_bench.jl`.)

**Correctness — confirmed.** Sharp threshold at `M = Πmₜ > 2q`:

| s, k | #GEMMs | 256³ | 512³ |
|------|--------|------|------|
| 12, 53 | 12 | 1.0e0 (fail, M<2q) | 1.0e0 (fail) |
| 13, 53 | 13 | 9.8e-1 (fail) | 1.0e0 (fail) |
| **14, 53** | **14** | **4.7e-16** | **6.0e-16** |
| 15, 53 | 15 | 4.7e-16 | 6.0e-16 |

→ Full FP64 with **14 GEMMs vs Scheme I's 43** — the promised 3× GEMM cut.

**Performance — a net loss as implemented (~100× slower than Scheme I):**

| Stage (1024³) | Time | (2048³) |
|---------------|------|---------|
| Scheme I (43 gemms, full) | 9.4 ms | 62 ms |
| Scheme II — GPU GEMM portion (14) | 235 ms | 498 ms |
| Scheme II — CPU Int128 reconstruction | 711 ms | 3719 ms |
| Scheme II — total | 947 ms | 4217 ms |

The 14 GEMMs themselves are only ~1.3 ms; this CPU-reconstruction version is
dominated by **Int64 residue arithmetic**, **14 host transfers**, and the **CPU
Int128 Garner reconstruction**.

**On-GPU reconstruction (done — `probe_schemeII_gpu.jl`).** Moved everything to
the GPU: residues in **Float64** (no Int64), **Int32 Garner** digits on-GPU, and
the big-integer combine + balanced reduction in a **double-double CUDA kernel**
(≈106 bits — the FP64 result needs only X's top ~53 bits, and balancing `X−M` in
dd absorbs the leading cancellation). Correct to full FP64 (relerr 4.7e-16 at
s=14). This cut Scheme II from ~100× slower to **~2× slower** than Scheme I — but
**still not a win on this card:**

| Size | native FP64 | Scheme I (43) | Scheme II GPU (14) |
|------|-------------|---------------|--------------------|
| 1024³ | 7.6 ms | 9.8 ms | 28.3 ms (0.35× Scheme I) |
| 2048³ | 61 ms | 63 ms | 120 ms (0.53×) |
| 4096³ | 415 ms | 326 ms | 521 ms (0.63×) |

Breakdown (2048³): the 14 GEMMs are ~6.6 ms; the cost is **Float64 residue
passes (~43 ms)**, **Garner (s²=196 elementwise launches, ~20 ms)**, and
**digit-stacking + dd kernel (~49 ms)**. So Scheme II is entirely
reconstruction-bound: the GEMM-count win (43→14) is tiny in absolute terms here,
and the unfused reconstruction swamps it. It improves with size (0.35×→0.63×) and
would need **heavy kernel fusion** (fuse residue computation, fuse Garner+dd into
one kernel) to win — which is what the RIKEN reference (GEMMul8) and the papers
do, and why their 6–7× appears on datacenter cards (A100/GH200/B200) where INT8
GEMMs dominate and reconstruction is fused. **On the RTX 4060 Ti, Scheme I (all-
GPU, simple FP64 recombination) remains the faster path.**

### Full `MMul4` (BallMatrix × BallMatrix) — implemented & verified

The probes above enclose the *point* product `mA·mB`. A true `MMul4` (see
`src/types/MMul/MMul4.jl`) also propagates the **input radii**: its directed
rounding of `mA·mB` (up for `C2`, down for `C1`) simultaneously captures the
product's rounding error *and* `rC_input = |mA|·rB + rA·(|mB|+rB)`. The GPU
version splits these two roles cleanly (`probe_int8_mmul4.jl`):

```
m     = INT8-Ozaki(mA, mB)                    exact FP64 midpoint  (expensive)
ρ     = rigorous bound on |m − mA·mB|          (Ozaki truncation + recomb)
rC_in = upper_bound(|mA|·rB + rA·(|mB|+rB))    FP32 GEMMs, inflated up  (cheap)
C     = ( m ,  ρ + rC_in )                     rounded up
```

The midpoint is where accuracy matters (→ exact INT8-Ozaki); `rC_in` only needs
to be a valid **upper bound**, so it's a single low-precision (FP32, inflated)
GEMM — the input radii are typically ~1e-16 anyway. (Nice symmetry: TF32, useless
for the accurate midpoint, is fine here.) **Verified to contain the exact ball
product** vs 300-bit BigFloat at input-radius levels 0, 1e-12, 1e-8, 1e-4 — the
ball radius transitions correctly from ρ-dominated (tiny inputs) to
input-radius-dominated (`5.7e-4` rel at radius 1e-4). With the tight ρ
(`probe_int8_rigor_tight.jl`), the point-input radius is ~1.1e-14 rel.

## Bottom line & where this pays off

* **Correctness & rigor: established** for the routes tested. Both produce
  verified enclosures (CuTileExt overlaps CPU `MMul4`; TF32-Ozaki contained by
  BigFloat truth).
* **Routes 1 & 2 on the RTX 4060 Ti (and A40): not worth it.** Float64 (Route 1)
  loses to the multithreaded CPU (no FP64 tensor cores, 1/64 FP32 rate);
  TF32-Ozaki (Route 2) loses because consumer TF32 is only ~1.5× FP32, so 3–4
  GEMMs ≈ 3.8×.
> **Midpoint vs full `MMul4`.** Most benchmarks below time the *midpoint*
> product `mA·mB` (the expensive GEMM). A full `MMul4` also returns the radius
> `rC`. The radius is cheap: the midpoint-error part is row/col-sum bounds (no
> GEMM, since the CRT is exact), and the input-radius part `|mA|·rB+rA·(|mB|+rB)`
> is ≤2 FP32 GEMMs (only when radii ≠ 0). The *full* GPU `MMul4` built on the port
> (`probe_gemmul8_mmul4.jl`, verified to contain the exact ball product vs
> BigFloat) beats CPU `MMul4` by **~6–8×** — see that section. Midpoint-only
> speedups quoted elsewhere are larger because they omit the (cheap) radius.

* **Route 3 (INT8-Ozaki Scheme I) is a winner — prototyped and validated.**
  Exact integer tensor-core products make it both rigorous *and* fast. Matches
  FP64 accuracy with verified BigFloat enclosure; the *midpoint* beats native
  GPU FP64 (1.1–1.65× at ≥2048³). This reverses Route 1, which lost to the CPU.
* **Scheme II (CRT): the Julia port of GEMMul8 (`gemmul8_port.jl`) is the fastest.**
  A hand-rolled on-GPU Garner version was ~2× slower than Scheme I, but porting
  GEMMul8's *fused* reconstruction (precomputed CRT weights, no Garner) **plus
  fused residue extraction** brought the *midpoint* to **beat Scheme I at every
  size (1.18–1.91×)** and native FP64 by up to 2.45×, full FP64. The **full
  `MMul4`** on it beats CPU `MMul4` by **~6–8×** (verified rigorous).
* **Datacenter cards** help all routes (A100/H100 add real FP64 tensor cores and
  ~8× TF32/FP32), but Route 3 (Scheme I)'s advantage does **not** depend on them.
  Scheme II's advantage *does* — it wins where INT8 GEMMs dominate and the
  reconstruction is fused (A100/GH200/B200), which is the papers' regime.

### Julia port of RIKEN's GEMMul8 (`gemmul8_port.jl`)

[GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) (MIT, © 2025- RIKEN R-CCS) is
the authors' reference Ozaki-II implementation. **`gemmul8_port.jl` is a Julia /
CUDA.jl port of its INT8 DGEMM path** (the MIT notice is reproduced in the file
header). Reading the reference pinpointed exactly the fusion our hand-rolled
Scheme II lacked:

- **No Garner.** Reconstruction is the *direct* CRT with **precomputed weights**
  `qPᵢ = qᵢ·Pᵢ` baked as double-double. The `hi` part is rounded to a **uniform
  absolute level** `2^(E−52+⌈log₂ρ⌉)` (E = exponent of P, ρ = Σ⌊pᵢ/2⌋) so the
  accumulation `Σ qhiᵢ·aᵢ` is **error-free** (lands exactly in 53 bits) — this
  uniform-absolute truncation (not per-value relative) is the subtle bit.
- Accumulate with two FMAs/term, then `X = C64f − P·rint(C64f/P)` (rint ⇒ signed,
  no separate balance step), then the diagonal rescale — **all in one fused
  per-element kernel** (port of GEMMul8 `invscal_device`).

Two fusions, each ~halving the time, brought it to full FP64 (relerr 4.3e-16,
s=14) and a clear win at every size:
- **Fused reconstruction** (precomputed CRT weights + 1 kernel, no Garner):
  2048³ 120→60 ms; 4096³ 521→268 ms.
- **Fused residue extraction** (`fused_residues_kernel!`, port of GEMMul8
  `extract_A_lo`): one pass over A and one over B emit *all* moduli residues
  (vs 14 passes each). 2048³ 60→35 ms; 4096³ 268→169 ms.

*Midpoint* product timing (the GEMM only):

| Size | native FP64 | Scheme I (43) | GEMMul8-port (14) | vs Scheme I | vs FP64 |
|------|-------------|---------------|-------------------|-------------|---------|
| 1024³ | 7.7 ms | 9.4 ms | 7.9 ms | 1.18× | 0.97× |
| 2048³ | 58 ms | 61 ms | 35 ms | 1.73× | 1.64× |
| 4096³ | 413 ms | 323 ms | **169 ms** | 1.91× | **2.45×** |

So the midpoint **beats Scheme I at every size** (1.18–1.91×) and native FP64 by
up to **2.45×**, full FP64. (The paper's 6–7× needs datacenter INT8/FP64 ratios;
2.45× is the consumer-Ada ceiling here.)

#### Full `MMul4` on the port (`probe_gemmul8_mmul4.jl`)

A *full* `MMul4` = midpoint (port) + radius `ρ_trunc + rC_in`. `ρ_trunc` is the
input-scaling truncation `|δA|·|mB|+|mA|·|δB|`; since the CRT midpoint is exact,
`|δA| ≤ 2^(σA−k−1)` is per-row constant, so this is **row/col sums × broadcast,
no GEMM**. `rC_in = |mA|·rB+rA·(|mB|+rB)` is ≤2 FP32 GEMMs, only when radii ≠ 0.
**Verified to contain the exact ball product** vs 300-bit BigFloat at input-radius
levels 0 / 1e-10 / 1e-5 (radius 5.2e-15 rel at zero input radius — FP64-tight).

| Size | CPU `MMul4` | GPU `MMul4` (port) | speedup |
|------|-------------|--------------------|---------|
| 1024³ | 87 ms | 13 ms | ~6.6× |
| 2048³ | 672 ms | 83 ms | ~8× |
| 4096³ | 2897 ms | 410 ms | ~7× |

So the honest full-`MMul4` win over CPU is **~6–8×** (adding the radius + the
full-pipeline overhead roughly halves the midpoint-only figure). Still the
fastest rigorous-capable `MMul4` here, and a *true* ball×ball product.

`num_moduli` is the accuracy/speed knob (SGEMM/CGEMM 2–13, DGEMM/ZGEMM 2–20);
error is governed by it → boundable for a rigorous radius. Remaining port work:
add SGEMM and **CGEMM/ZGEMM** (complex, from arXiv:2512.08321); wrap as the
`MMul4` overload (midpoint via the port + ρ from `num_moduli` + FP32 input-radius
term).

## Recommended next steps

1. ✅ **DONE — tightened the INT8-Ozaki radius** (`probe_int8_rigor_tight.jl`):
   per-`i` tail-sum bound in FP32 + rigorous recombination/residual bounds. Radius
   1.1e-14 rel at T=10, ~20× the true error (the `√K` rigor floor), tighter than a
   native FP64 `MMul4`.
2. ✅ **DONE — Scheme II reconstruction ported on-GPU** (`probe_schemeII_gpu.jl`):
   Float64 residues + Int32 Garner + double-double combine kernel, full FP64,
   ~2× slower than Scheme I (reconstruction-bound on consumer HW; would need
   kernel fusion to win — see GEMMul8 below). Remaining Scheme-I pipeline tweaks:
   group recombination by `i+j`, overlap recombine with the next GEMM (2-stream).
3. **For production / complex / datacenter, wrap [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8)**
   (RIKEN's fused Ozaki-II, incl. CGEMM/ZGEMM) for the midpoint + add the rigorous
   radius, rather than hand-rolling fused kernels.
4. **Wrap as a `MMul4` overload** for `Float64`/`Float32` (and complex) ball
   matrices on `CuArray`, dispatched like the CuTileExt overload; pick `T`/`s`
   from the requested radius. Add `m,n,k % 4 == 0` padding.
5. Re-run `bench_mmul4.jl` (Float64, CuTileExt) and `probe_tf32_rigor.jl`
   (TF32-Ozaki) on an **A100/H100** to confirm the Route-1/2 crossover. The A40
   on toeplitz is *not* sufficient (GA102 FP64 penalty). Needs Julia ≥ 1.11 for
   cuTile (toeplitz ships 1.8.1 — ask Leonardo to load a recent Julia).
6. Keep `CuTileExt.jl` (outer-product, directed-rounding) as the correct,
   tensor-core-free fallback.

## References

- Ozaki, Uchino, Imamura, *"Ozaki Scheme II: A GEMM-oriented emulation of FP
  matmul using an integer modular technique"*, arXiv:2504.08009 (2025).
- Uchino, Ma, Imamura, Ozaki, Gutsche, *"Emulation of Complex Matrix
  Multiplication based on the Chinese Remainder Theorem"*, arXiv:2512.08321
  (2025) — CGEMM/ZGEMM on INT8; 4–6.5× over cuBLAS on B200.
- RIKEN-RCCS, **GEMMul8**, https://github.com/RIKEN-RCCS/GEMMul8 — reference
  Ozaki-II (CUDA/HIP; SGEMM/DGEMM/CGEMM/ZGEMM; `num_moduli`, `fastmode`).

## Script index (`experiments/cutile_probe/`)

| Script | Purpose |
|--------|---------|
| `probe.jl` | Shows tile `mma` ignores `@fpmode` (Up==Dn bitwise). |
| `probe_elementwise.jl` | Outer-product honors `@fpmode` (one-sided bracketing). |
| `probe_mmul4.jl` | Full `MMul4` on cuTile validated vs CPU. |
| `smoke_test.jl` | Extension loads + `*` dispatch overlaps CPU (8 cases). |
| `bench_mmul4.jl` | Float64 CuTileExt GPU vs CPU benchmark. |
| `probe_tf32_ozaki.jl` | TF32 engagement + Ozaki accuracy recovery. |
| `probe_tf32_rigor.jl` | TF32-Ozaki rigorous radius (BigFloat-checked) + timing. |
| `probe_int8_ozaki.jl` | INT8-Ozaki accuracy vs BigFloat (slice count sweep). |
| `probe_int8_bench.jl` | INT8-Ozaki vs native FP64 timing (beats FP64 at ≥2048³). |
| `probe_int8_rigor.jl` | INT8-Ozaki rigorous ball (loose bound), BigFloat check. |
| `probe_int8_rigor_tight.jl` | Tight radius: per-i tail sums (1.1e-14 rel, ~20× err). |
| `probe_int8_streams.jl` | Time breakdown; GEMM-stream concurrency (no speedup). |
| `probe_int8_vs_cpu.jl` | INT8-Ozaki GPU vs CPU FP64 gemm and rigorous CPU MMul4. |
| `probe_int8_schemeII.jl` | Scheme II (CRT) correctness vs BigFloat (14 gemms → FP64). |
| `probe_schemeII_bench.jl` | Scheme II GEMM/reconstruction breakdown vs Scheme I (CPU recon). |
| `probe_schemeII_gpu.jl` | Scheme II with on-GPU double-double CRT (Garner) reconstruction. |
| `probe_schemeII_gpu_bench.jl` | GPU Scheme II (Garner) vs Scheme I vs native FP64. |
| `gemmul8_port.jl` | **Julia port of RIKEN GEMMul8** (fused direct-CRT reconstruction). |
| `gemmul8_bench.jl` | GEMMul8 port vs Scheme I vs native FP64 (beats both at 4096³). |
| `probe_gemmul8_mmul4.jl` | **Full ball MMul4** on the port (mid+radius); enclosure + vs CPU MMul4 (~6–8×). |
| `probe_int8_mmul4.jl` | Full GPU MMul4 (ball×ball): INT8 midpoint + FP32 radius. |
| `issue_draft.md` | Body of cuTile.jl issue #229. |
