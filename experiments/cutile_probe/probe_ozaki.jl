# Probe: Ozaki-style splitting for a more-accurate GPU GEMM.
#
# Idea: split Float32 A, B into 12-bit "high" and "low" halves via Veltkamp.
# Each per-element product A_hi[i,k] * B_hi[k,j] has ≤ 24 mantissa bits
# (12+12), so the per-multiply rounding error is zero in Float32. Only the
# K-fold accumulation contributes rounding error. Four CUBLAS GEMMs:
#   C ≈ A_hi*B_hi + A_hi*B_lo + A_lo*B_hi + A_lo*B_lo
#
# Pass criterion: |C_ozaki - true| should be measurably smaller than
# |C_naive - true|, where C_naive is a single Float32 CUBLAS GEMM and
# `true` is BigFloat A*B rounded back to Float32.

using CUDACore
using Random, Printf, LinearAlgebra

const SPLIT_BITS = 12
const SPLIT_C    = Float32(2^SPLIT_BITS + 1)

# Veltkamp split on the device. After this:
#   * A = A_hi + A_lo exactly (Float32),
#   * A_hi, A_lo each have ≤ (24 − SPLIT_BITS) = 12 mantissa bits.
function veltkamp_split_gpu(A::CuArray{Float32})
    γ    = SPLIT_C .* A
    A_hi = γ .- (γ .- A)
    A_lo = A .- A_hi
    return A_hi, A_lo
end

function ozaki_gemm(dA::CuArray{Float32}, dB::CuArray{Float32})
    A_hi, A_lo = veltkamp_split_gpu(dA)
    B_hi, B_lo = veltkamp_split_gpu(dB)
    return (A_hi * B_hi) .+ (A_hi * B_lo) .+ (A_lo * B_hi) .+ (A_lo * B_lo)
end

naive_gemm(dA::CuArray{Float32}, dB::CuArray{Float32}) = dA * dB

# Ground truth: BigFloat matmul, rounded back to Float32 for comparison.
function reference(A::Matrix{Float32}, B::Matrix{Float32})
    setprecision(BigFloat, 128) do
        Float32.(BigFloat.(A) * BigFloat.(B))
    end
end

# Reference at full BigFloat precision (no rounding back), for the bound.
function reference_big(A::Matrix{Float32}, B::Matrix{Float32})
    setprecision(BigFloat, 128) do
        BigFloat.(A) * BigFloat.(B)
    end
end

function run_probe(M, N, K; seed=42)
    rng = MersenneTwister(seed)
    A = randn(rng, Float32, M, K)
    B = randn(rng, Float32, K, N)
    dA = CuArray(A); dB = CuArray(B)

    C_naive_d = naive_gemm(dA, dB)
    C_ozaki_d = ozaki_gemm(dA, dB)
    CUDACore.synchronize()
    C_naive = Array(C_naive_d)
    C_ozaki = Array(C_ozaki_d)

    C_true_big = reference_big(A, B)

    err_naive_inf = maximum(abs.(BigFloat.(C_naive) .- C_true_big))
    err_ozaki_inf = maximum(abs.(BigFloat.(C_ozaki) .- C_true_big))

    # Higham-style entrywise bound on a Float32 GEMM with IEEE rounding:
    #   |fl(A*B) - A*B|_ij ≤ K * eps(Float32) * (|A|·|B|)_ij  (leading order)
    AB_abs = abs.(BigFloat.(A)) * abs.(BigFloat.(B))
    higham = Float32(K) * eps(Float32) .* AB_abs
    bound_max = maximum(higham)

    @printf("  M=%d N=%d K=%d\n", M, N, K)
    @printf("    naive Float32 max err     = %g\n", Float64(err_naive_inf))
    @printf("    Ozaki k=12   max err      = %g\n", Float64(err_ozaki_inf))
    @printf("    improvement               = %.2fx\n",
        Float64(err_naive_inf / max(err_ozaki_inf, eps(BigFloat))))
    @printf("    K·eps·max(|A||B|) bound   = %g\n", Float64(bound_max))
    @printf("    naive_err / bound         = %.3f\n", Float64(err_naive_inf / bound_max))
    @printf("    ozaki_err / bound         = %.3g\n", Float64(err_ozaki_inf / bound_max))
end

println("Device: ", CUDACore.name(CUDACore.device()))
println("Initial math_mode: ", CUDACore.math_mode(),
        "   math_precision: ", CUDACore.math_precision())
println()

for mm in (CUDACore.DEFAULT_MATH, CUDACore.FAST_MATH, CUDACore.PEDANTIC_MATH)
    CUDACore.math_mode!(mm)
    println("============================================================")
    println("math_mode = $mm")
    println("============================================================")
    for (M, N, K) in [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
        run_probe(M, N, K)
        println()
    end
end
