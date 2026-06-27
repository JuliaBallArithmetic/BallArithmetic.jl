# Probe: does cuTile's @fpmode actually round every FMA in a tile-level matmul?
#
# Strategy: pick A, B with many addends so accumulated rounding error is
# visible, then compute C = A*B under @fpmode rounding_mode=PosInf and
# rounding_mode=NegInf using a Float32 (and Float64) tile accumulator.
#
# Pass criteria:
#   1. C_up[i,j] >= C_dn[i,j] for every entry (one-sidedness)
#   2. C_up[i,j] >  C_dn[i,j] for many entries (mode is not a no-op)
#   3. CPU midpoint (RoundNearest) lies in [C_dn, C_up] for every entry
#
# If any of these fails, @fpmode is not rigorous enough to back MMul4.

using CUDACore
using cuTile: cuTile
import cuTile as ct
using Random, Printf, LinearAlgebra

function gemm_kernel!(A::ct.TileArray{T,2}, B::ct.TileArray{T,2}, C::ct.TileArray{T,2},
                     tm::Int, tn::Int, tk::Int, rmode) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    num_k = ct.num_tiles(A, 2, (tm, tk))

    acc = zeros(T, tm, tn)

    # Note: NO TFloat32 conversion. Keep full IEEE precision so the
    # rounding mode is a meaningful question.
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
    return Array(A), Array(B), Array(C)
end

function probe(::Type{T}; M=128, N=128, K=512, tm=32, tn=32, tk=32) where {T}
    println("--- $(T): $(M)x$(K) * $(K)x$(N), tiles $(tm)x$(tn)x$(tk) ---")
    A, B, C_up = run_one(T, M, N, K, tm, tn, tk, ct.Rounding.PosInf)
    _, _, C_dn = run_one(T, M, N, K, tm, tn, tk, ct.Rounding.NegInf)
    _, _, C_ne = run_one(T, M, N, K, tm, tn, tk, ct.Rounding.NearestEven)

    diff = C_up .- C_dn
    one_sided = all(>=(0), diff)
    nz = count(>(0), diff)
    max_gap = maximum(diff)
    cpu_ref = A * B  # CPU reference, ordinary rounding

    bracket_ok = all(C_dn .<= cpu_ref .<= C_up)
    ne_in_bracket = all(C_dn .<= C_ne .<= C_up)

    println(@sprintf("  one-sided   (C_up >= C_dn always)        : %s", one_sided))
    println(@sprintf("  nonzero gap entries / total              : %d / %d", nz, length(C_up)))
    println(@sprintf("  max gap                                  : %g", max_gap))
    println(@sprintf("  CPU ref in [C_dn, C_up]                  : %s", bracket_ok))
    println(@sprintf("  NearestEven in [C_dn, C_up]              : %s", ne_in_bracket))
    println()
    return (one_sided=one_sided, nz=nz, max_gap=max_gap, bracket=bracket_ok)
end

println("CUDA device: ", CUDACore.name(CUDACore.device()))
println()
probe(Float32)
probe(Float64)
