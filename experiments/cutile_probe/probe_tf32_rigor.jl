# Probe: rigorous TF32-Ozaki ball GEMM — timing + verified enclosure.
#
# Center: 4-GEMM Ozaki on TF32 tensor cores (recovers ~FP32 accuracy).
# Radius: a-priori bound  γ · (|A|·|B|)  with
#         γ = 2·ε_split + (K+4)·u_f32,   ε_split = 2^-22 (2-slice residual),
#         u_f32 = 2^-24.  (|A|·|B|) computed in FP32 then inflated to a
#         rigorous upper bound. This is the same radius an FP32 MMul4 would
#         give — but the center is produced on the fast TF32 path.
#
# Checks:
#   * enclosure vs BigFloat truth (true rigor) at a modest size,
#   * timing: 1×FP32 gemm  vs  4×TF32 gemm + recombination (+ radius cost).

using CUDACore, CUDA
using Random, Printf, LinearAlgebra, Statistics

const gemmEx! = CUDA.CUBLAS.gemmEx!
function gemm_mode(A, B, mode)
    C = similar(A, size(A, 1), size(B, 2))
    old = CUDACore.math_mode(); CUDACore.math_mode!(mode)
    try; gemmEx!('N','N', 1.0f0, A, B, 0.0f0, C); finally; CUDACore.math_mode!(old); end
    return C
end
tf32_gemm(A, B) = gemm_mode(A, B, CUDACore.FAST_MATH)
fp32_gemm(A, B) = gemm_mode(A, B, CUDACore.DEFAULT_MATH)

const VC = Float32(2^13 + 1)
tf32_hi(A) = (g = VC .* A; g .- (g .- A))
function split2(A)
    A0 = tf32_hi(A); r = A .- A0; A1 = tf32_hi(r)
    return A0, A1
end

# Ozaki center on TF32: A ≈ A0+A1, B ≈ B0+B1, four TF32 GEMMs recombined.
function ozaki_center(dA, dB)
    A0, A1 = split2(dA); B0, B1 = split2(dB)
    return tf32_gemm(A0,B0) .+ tf32_gemm(A0,B1) .+ tf32_gemm(A1,B0) .+ tf32_gemm(A1,B1)
end

# Rigorous radius γ·(|A|·|B|), with (|A|·|B|) over-estimated.
function rigor_radius(dA, dB, K)
    u   = 2.0f0^-24
    γ   = 2.0f0 * 2.0f0^-22 + Float32(K + 4) * u
    AB  = fp32_gemm(abs.(dA), abs.(dB))          # ≈ |A|·|B|, round-to-nearest
    ABhi = AB .* (1.0f0 + Float32(2K) * u)       # inflate to rigorous upper bound
    return γ .* ABhi
end

function timeit(f; nwarm=2, nrep=6)
    for _ in 1:nwarm; f(); CUDACore.synchronize(); end
    ts = Float64[]
    for _ in 1:nrep
        t0 = time_ns(); f(); CUDACore.synchronize()
        push!(ts, (time_ns()-t0)/1e9)
    end
    return sort(ts)[cld(length(ts),2)]
end

function run(M, N, K; check_rigor=false, seed=1)
    rng = MersenneTwister(seed)
    A = randn(rng, Float32, M, K); B = randn(rng, Float32, K, N)
    dA = CuArray(A); dB = CuArray(B)

    mC = Array(ozaki_center(dA, dB))
    rC = Array(rigor_radius(dA, dB, K)); CUDACore.synchronize()

    # Timings
    t_fp32 = timeit(() -> fp32_gemm(dA, dB))
    t_tf32 = timeit(() -> tf32_gemm(dA, dB))
    t_oz   = timeit(() -> ozaki_center(dA, dB))
    t_full = timeit(() -> (ozaki_center(dA, dB); rigor_radius(dA, dB, K)))

    @printf("  M=%d N=%d K=%d\n", M, N, K)
    @printf("    timing:  FP32 %.2f ms | TF32 %.2f ms | Ozaki-center %.2f ms (%.2fx FP32) | center+radius %.2f ms (%.2fx FP32)\n",
        t_fp32*1e3, t_tf32*1e3, t_oz*1e3, t_oz/t_fp32, t_full*1e3, t_full/t_fp32)
    @printf("    radius:  median rel = %.2e   (FP32 K·u ref = %.2e)\n",
        median(vec(rC) ./ max.(abs.(vec(mC)), 1f-30)), K * 2.0^-24)

    if check_rigor
        setprecision(BigFloat, 200) do
            P = BigFloat.(A) * BigFloat.(B)
            lo = BigFloat.(mC) .- BigFloat.(rC)
            hi = BigFloat.(mC) .+ BigFloat.(rC)
            inside = all(lo .<= P .<= hi)
            margin = minimum(min.(P .- lo, hi .- P))   # >=0 means enclosed
            ctr_err = maximum(abs.(BigFloat.(mC) .- P))
            @printf("    ENCLOSURE vs BigFloat: %s   min margin = %.2e   center max err = %.2e\n",
                inside ? "OK (all entries contained)" : "FAILED", Float64(margin), Float64(ctr_err))
        end
    end
    println()
end

println("Device: ", CUDACore.name(CUDACore.device()), "\n")
run(256, 256, 256; check_rigor=true)
run(512, 512, 512; check_rigor=true)
for (M,N,K) in [(1024,1024,1024), (2048,2048,2048), (4096,4096,4096)]
    run(M, N, K)
end
