# Probe v2: bypass the `mma` tile-matmul intrinsic (whose codegen ignores
# `@fpmode`) and compute matmul as accumulated elementwise tile mulf/addf.
# Those *do* forward the fpmode kwargs to the op (see arithmetic.jl), so
# this path should be rigorously rounded.

using CUDACore
using cuTile: cuTile
import cuTile as ct
using Random, Printf

# Compute one (tm × tn) output tile by accumulating outer products
#   acc += A[:, k] * B[k, :]^T
# using elementwise mulf/addf inside @fpmode. No tile-level muladd.
function gemm_elem_kernel!(A::ct.TileArray{T,2}, B::ct.TileArray{T,2}, C::ct.TileArray{T,2},
                           tm::Int, tn::Int, K::Int, rmode) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)

    acc = zeros(T, tm, tn)

    ct.@fpmode rounding_mode=rmode flush_to_zero=false begin
        for k in Int32(1):Int32(K)
            a_col = ct.load(A; index=(bid_m, k), shape=(tm, 1), padding_mode=ct.PaddingMode.Zero)
            b_row = ct.load(B; index=(k, bid_n), shape=(1, tn), padding_mode=ct.PaddingMode.Zero)
            acc = acc + a_col * b_row        # broadcast outer product via elementwise mulf+addf
        end
    end
    ct.store(C; index=(bid_m, bid_n), tile=acc)
    return nothing
end

function run_one(::Type{T}, M, N, K, tm, tn, rmode) where {T}
    A = CuArray(rand(MersenneTwister(1), T, M, K))
    B = CuArray(rand(MersenneTwister(2), T, K, N))
    C = CuArray{T}(undef, M, N)
    grid = (cld(M, tm), cld(N, tn))
    @cuda backend=cuTile blocks=grid gemm_elem_kernel!(A, B, C,
        ct.Constant(tm), ct.Constant(tn), ct.Constant(K), ct.Constant(rmode))
    CUDACore.synchronize()
    return Array(A), Array(B), Array(C)
end

function probe(::Type{T}; M=64, N=64, K=512, tm=16, tn=16) where {T}
    println("--- $(T): $(M)x$(K) * $(K)x$(N), tile $(tm)x$(tn), elementwise ---")
    A, B, C_up = run_one(T, M, N, K, tm, tn, ct.Rounding.PosInf)
    _, _, C_dn = run_one(T, M, N, K, tm, tn, ct.Rounding.NegInf)
    _, _, C_ne = run_one(T, M, N, K, tm, tn, ct.Rounding.NearestEven)

    diff = C_up .- C_dn
    one_sided = all(>=(0), diff)
    nz = count(>(0), diff)
    max_gap = maximum(diff)

    ne_in_bracket = all(C_dn .<= C_ne .<= C_up)

    println(@sprintf("  one-sided   (C_up >= C_dn always)        : %s", one_sided))
    println(@sprintf("  nonzero gap entries / total              : %d / %d", nz, length(C_up)))
    println(@sprintf("  max gap                                  : %g", max_gap))
    println(@sprintf("  NearestEven in [C_dn, C_up]              : %s", ne_in_bracket))
    println()
end

println("CUDA device: ", CUDACore.name(CUDACore.device()))
println()
probe(Float32)
probe(Float64)
