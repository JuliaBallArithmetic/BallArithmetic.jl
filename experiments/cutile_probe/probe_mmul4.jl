# End-to-end probe of a full MMul4 kernel on cuTile.
#
# Reproduces the MMul4 algorithm from src/types/MMul/MMul4.jl on the GPU
# using elementwise tile ops inside @fpmode scopes, and compares against
# the CPU implementation entry by entry.

using CUDACore
using cuTile: cuTile
import cuTile as ct
using Random, Printf, Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io=devnull)  # load BallArithmetic
using BallArithmetic
Pkg.activate(@__DIR__; io=devnull)                       # back to probe env

# One thread block computes one (tm × tn) tile of the output mid/rad.
# Implements MMul4(BallMatrix, BallMatrix) -> BallMatrix.
function _mmul4_kernel!(mC::ct.TileArray{T,2}, rC::ct.TileArray{T,2},
                       mA::ct.TileArray{T,2}, rA::ct.TileArray{T,2},
                       mB::ct.TileArray{T,2}, rB::ct.TileArray{T,2},
                       tm::Int, tn::Int) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    K = size(mA, 2)

    acc_up = zeros(T, tm, tn)   # will become C2 = (mA*mB)_up + rC
    acc_r  = zeros(T, tm, tn)   # rC = |mA|·rB + rA·(|mB|+rB), all RoundUp

    ct.@fpmode rounding_mode=ct.Rounding.PosInf flush_to_zero=false begin
        for k in Int32(1):Int32(K)
            mA_col = ct.load(mA; index=(bid_m, k), shape=(tm, 1), padding_mode=ct.PaddingMode.Zero)
            mB_row = ct.load(mB; index=(k, bid_n), shape=(1, tn), padding_mode=ct.PaddingMode.Zero)
            rA_col = ct.load(rA; index=(bid_m, k), shape=(tm, 1), padding_mode=ct.PaddingMode.Zero)
            rB_row = ct.load(rB; index=(k, bid_n), shape=(1, tn), padding_mode=ct.PaddingMode.Zero)

            absA = cuTile.Intrinsics.absf(mA_col)
            absB = cuTile.Intrinsics.absf(mB_row)
            acc_up = acc_up + mA_col * mB_row
            acc_r  = acc_r  + absA * rB_row + rA_col * (absB + rB_row)
        end
        acc_up = acc_up + acc_r       # C2
    end

    acc_dn = zeros(T, tm, tn)
    ct.@fpmode rounding_mode=ct.Rounding.NegInf flush_to_zero=false begin
        for k in Int32(1):Int32(K)
            mA_col = ct.load(mA; index=(bid_m, k), shape=(tm, 1), padding_mode=ct.PaddingMode.Zero)
            mB_row = ct.load(mB; index=(k, bid_n), shape=(1, tn), padding_mode=ct.PaddingMode.Zero)
            acc_dn = acc_dn + mA_col * mB_row
        end
        acc_dn = acc_dn - acc_r        # C1
    end

    ct.@fpmode rounding_mode=ct.Rounding.PosInf flush_to_zero=false begin
        mid_tile = (acc_dn + acc_up) * T(0.5)
        rad_tile = mid_tile - acc_dn
        ct.store(mC; index=(bid_m, bid_n), tile=mid_tile)
        ct.store(rC; index=(bid_m, bid_n), tile=rad_tile)
    end
    return nothing
end

function mmul4_gpu(mA::Matrix{T}, rA::Matrix{T}, mB::Matrix{T}, rB::Matrix{T};
                   tm=16, tn=16) where {T}
    dmA = CuArray(mA); drA = CuArray(rA)
    dmB = CuArray(mB); drB = CuArray(rB)
    dmC = CuArray{T}(undef, size(mA, 1), size(mB, 2))
    drC = CuArray{T}(undef, size(mA, 1), size(mB, 2))
    grid = (cld(size(mA,1), tm), cld(size(mB,2), tn))
    @cuda backend=cuTile blocks=grid _mmul4_kernel!(dmC, drC, dmA, drA, dmB, drB,
        ct.Constant(tm), ct.Constant(tn))
    CUDACore.synchronize()
    return Array(dmC), Array(drC)
end

function check(::Type{T}, M, N, K; tm=16, tn=16) where {T}
    rng = MersenneTwister(42)
    mA = rand(rng, T, M, K) .- T(0.5)
    rA = rand(rng, T, M, K) .* T(1e-3)
    mB = rand(rng, T, K, N) .- T(0.5)
    rB = rand(rng, T, K, N) .* T(1e-3)

    A = BallMatrix(mA, rA)
    B = BallMatrix(mB, rB)
    C_cpu = A * B   # MMul4 CPU
    mC_cpu, rC_cpu = mid(C_cpu), rad(C_cpu)

    mC_gpu, rC_gpu = mmul4_gpu(mA, rA, mB, rB; tm=tm, tn=tn)

    cpu_lo = mC_cpu .- rC_cpu
    cpu_hi = mC_cpu .+ rC_cpu
    gpu_lo = mC_gpu .- rC_gpu
    gpu_hi = mC_gpu .+ rC_gpu

    gpu_encloses_cpu = all(gpu_lo .<= cpu_lo) && all(cpu_hi .<= gpu_hi)
    cpu_encloses_gpu = all(cpu_lo .<= gpu_lo) && all(gpu_hi .<= cpu_hi)
    overlap = all(max.(gpu_lo, cpu_lo) .<= min.(gpu_hi, cpu_hi))

    mid_diff = maximum(abs.(mC_gpu .- mC_cpu))
    rad_ratio_max = maximum(rC_gpu ./ max.(rC_cpu, eps(T)))
    rad_ratio_min = minimum(rC_gpu ./ max.(rC_cpu, eps(T)))

    println(@sprintf("  %s  M=%d N=%d K=%d tm=%d tn=%d", T, M, N, K, tm, tn))
    println(@sprintf("    overlap (every entry)         : %s", overlap))
    println(@sprintf("    GPU encloses CPU              : %s", gpu_encloses_cpu))
    println(@sprintf("    CPU encloses GPU              : %s", cpu_encloses_gpu))
    println(@sprintf("    |mC_gpu - mC_cpu|_inf         : %g", mid_diff))
    println(@sprintf("    rC_gpu / rC_cpu range         : [%g, %g]",
                     rad_ratio_min, rad_ratio_max))
end

println("Device: ", CUDACore.name(CUDACore.device()))
println()

println("--- Float32 ---")
check(Float32, 16, 16, 64;  tm=16, tn=16)
check(Float32, 64, 64, 256; tm=16, tn=16)
check(Float32, 64, 64, 256; tm=32, tn=32)

println("\n--- Float64 ---")
check(Float64, 16, 16, 64;  tm=16, tn=16)
check(Float64, 64, 64, 256; tm=16, tn=16)
check(Float64, 64, 64, 256; tm=32, tn=32)
