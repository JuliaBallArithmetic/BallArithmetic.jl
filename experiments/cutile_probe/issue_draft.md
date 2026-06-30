## `@fpmode rounding_mode` is silently ignored by the tile-level `mma` / `muladd` matmul

### Summary

`ct.@fpmode rounding_mode=ct.Rounding.PosInf` (and `NegInf`) has no effect on the result of tile-level matrix multiplications produced by `muladd(a::Tile, b::Tile, acc::Tile)` (which lowers to the `mma` intrinsic). Running the same matmul under `PosInf`, `NegInf`, and `NearestEven` produces **bitwise-identical** outputs.

The mode *does* take effect when the matmul is rewritten as elementwise tile ops (`mulf` + `addf`), which is consistent with my reading of `arithmetic.jl` (each elementwise intrinsic passes `fpmode_kwargs(ctx)` to its encoder) versus `core.jl` where `encode_MmaFOp!` is called without any rounding/mode attribute.

This matters for rigorous numerics: I'm writing a CUDA backend for [BallArithmetic.jl](https://github.com/orkolorko/BallArithmetic.jl) that needs directed-rounding matrix multiplication (Miyajima/Rump-style verified GEMM); without `@fpmode` reaching the `mma` op, the entire library has to fall back to elementwise accumulation, losing the tensor-core path.

### Environment

- cuTile 0.3.0
- Julia 1.12.6
- CUDA driver 580.142 (CUDA Runtime via `CUDA_Runtime_jll`)
- GPU: NVIDIA GeForce RTX 4060 Ti (Ada, sm_8.9)
- Linux 6.17.0

### Minimal reproducer

```julia
using CUDACore
using cuTile: cuTile
import cuTile as ct
using Random, Printf

function gemm_kernel!(A::ct.TileArray{T,2}, B::ct.TileArray{T,2}, C::ct.TileArray{T,2},
                     tm::Int, tn::Int, tk::Int, rmode) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    num_k = ct.num_tiles(A, 2, (tm, tk))

    acc = zeros(T, tm, tn)

    # NO TFloat32 conversion — full IEEE precision so the rounding-mode
    # question is well-posed.
    ct.@fpmode rounding_mode=rmode flush_to_zero=false begin
        for k in Int32(1):num_k
            a = ct.load(A; index=(bid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.Zero)
            b = ct.load(B; index=(k, bid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.Zero)
            acc = muladd(a, b, acc)
        end
    end
    ct.store(C; index=(bid_m, bid_n), tile=acc)
    return nothing
end

function run_one(::Type{T}, M, N, K, tm, tn, tk, rmode) where {T}
    A = CuArray(rand(MersenneTwister(1), T, M, K))
    B = CuArray(rand(MersenneTwister(2), T, K, N))
    C = CuArray{T}(undef, M, N)
    grid = (cld(M, tm), cld(N, tn))
    @cuda backend=cuTile blocks=grid gemm_kernel!(A, B, C,
        ct.Constant(tm), ct.Constant(tn), ct.Constant(tk), ct.Constant(rmode))
    CUDACore.synchronize()
    return Array(C)
end

function probe(::Type{T}; M=128, N=128, K=512, tm=32, tn=32, tk=32) where {T}
    C_up = run_one(T, M, N, K, tm, tn, tk, ct.Rounding.PosInf)
    C_dn = run_one(T, M, N, K, tm, tn, tk, ct.Rounding.NegInf)
    C_ne = run_one(T, M, N, K, tm, tn, tk, ct.Rounding.NearestEven)
    diff = C_up .- C_dn
    println(@sprintf("%-10s nonzero gap entries / total: %d / %d   max gap: %g",
        T, count(>(0), diff), length(C_up), maximum(diff)))
    println(@sprintf("           Up == Dn (bitwise): %s", C_up == C_dn))
    println(@sprintf("           Up == NearestEven : %s", C_up == C_ne))
end

probe(Float32)
probe(Float64)
```

Output:

```
Float32    nonzero gap entries / total: 0 / 16384   max gap: 0
           Up == Dn (bitwise): true
           Up == NearestEven : true
Float64    nonzero gap entries / total: 0 / 16384   max gap: 0
           Up == Dn (bitwise): true
           Up == NearestEven : true
```

### Expected behaviour

With `K=512` random `[0,1)` inputs, each output entry is a sum of 512 products; in `Float32` the gap between `PosInf` and `NegInf` should be on the order of `K · eps(Float32) · mean ≈ 3e-5`. Concretely, when I rewrite the same matmul using only elementwise tile ops (an outer-product accumulation `acc + a_col * b_row` with `(tm × 1) × (1 × tn)` loads inside the same `@fpmode` scope), I get strictly one-sided bracketing on every entry and the expected gap magnitude. So `@fpmode` *is* propagating to `mulf`/`addf` — just not to `mma`.

### Root cause (apparent)

`src/compiler/intrinsics/core.jl:442` — `emit_intrinsic!(ctx, ::typeof(Intrinsics.mma), args)` ends with

```julia
encode_MmaFOp!(cb, acc.type_id, lhs.v, rhs.v, acc.v)
```

with no `fpmode_kwargs(ctx)...` (compare with e.g. `emit_intrinsic!` for `addf` at `src/compiler/intrinsics/arithmetic.jl:430` which does forward them). So even if the user sets `rounding_mode`, the attribute never reaches `mmaf`/`mmai`.

I realize this likely reflects the underlying hardware reality: NVIDIA tensor-core MMA instructions are fixed to round-to-nearest-even, and Tile IR may not currently expose a directed-rounding variant. But silently ignoring the mode set by the surrounding `@fpmode` scope is surprising — at minimum it would be helpful to either:

1. Throw an `IRError` at codegen time when an `mma` op is emitted inside a scope whose `rounding_mode` is not `NearestEven`/`Approx`/inherited-default, or
2. Document explicitly in the `@fpmode` docstring and on `Intrinsics.mma` that the rounding-mode attribute does not reach the matmul intrinsic, and recommend the elementwise rewrite for users who need directed rounding, or
3. Plumb the attribute through `encode_MmaFOp!` to a future Tile-IR-level directed-rounding mma (when Tile IR / hardware supports it — e.g. some Blackwell modes).

### Workaround I'm using

Express the matmul as outer-product accumulation:

```julia
ct.@fpmode rounding_mode=ct.Rounding.PosInf flush_to_zero=false begin
    for k in Int32(1):Int32(K)
        a_col = ct.load(A; index=(bid_m, k), shape=(tm, 1), padding_mode=ct.PaddingMode.Zero)
        b_row = ct.load(B; index=(k, bid_n), shape=(1, tn), padding_mode=ct.PaddingMode.Zero)
        acc = acc + a_col * b_row
    end
end
```

This works (verified rigorously bracketing the CPU `setrounding` result), but bypasses tensor cores and costs ~5× throughput on Float32 (no penalty on Float64 since consumer Ada has no Float64 tensor cores anyway).

### Asks (in priority order)

1. **At minimum:** documentation on `@fpmode`, `muladd(::Tile,::Tile,::Tile)`, and `Intrinsics.mma` noting that the rounding mode is not honored by tile matmul, so users know they need the elementwise rewrite.
2. **Better:** an `IRError` (or compile-time warning) when `mma` is emitted under a non-default `rounding_mode` scope, so the silent miss can't happen.
3. **Best, longer-term:** wire `fpmode_kwargs` through to `encode_MmaFOp!` so that when/if Tile IR exposes a directed-rounding mma (Blackwell-class hardware, software fallback, etc.), it Just Works.

Happy to send a PR for (1) or (2) if there's interest.
