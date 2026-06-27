# Probe: rigorous TF32-Ozaki GEMM.
#
# Three questions:
#   1. Does FAST_MATH actually dispatch TF32 tensor cores, and at what size?
#      (Earlier sweep showed no TF32 effect at K<=256.) Signal: naive-TF32
#      error jumps to ~K*2^-11 relative, far above the FP32-GEMM error.
#   2. Is the 2-slice Veltkamp split TF32-exact? If A0,A1 are TF32-exact, a
#      TF32-compute GEMM of them must equal an FP32-compute GEMM bitwise.
#   3. Does 4-GEMM Ozaki recover FP32-quality accuracy on the TF32 path?
#
# Reference is Float64 CPU matmul (accurate enough to separate TF32's ~1e-2
# error from FP32's ~1e-4), avoiding the BigFloat-at-K=2048 OOM from before.

using CUDACore, CUDA
using Random, Printf, LinearAlgebra

# IMPORTANT: `A * B` for Float32 CuArrays dispatches to plain cublasSgemm,
# which ALWAYS computes in full FP32 and ignores math_mode — so TF32 never
# engages through `*`. To reach the TF32 tensor cores we must call gemmEx!
# explicitly under FAST_MATH (compute type CUBLAS_COMPUTE_32F_FAST_TF32).
const gemmEx! = CUDA.CUBLAS.gemmEx!

function gemm_mode(A, B, mode)
    C = similar(A, size(A, 1), size(B, 2))
    old = CUDACore.math_mode()
    CUDACore.math_mode!(mode)
    try
        gemmEx!('N', 'N', 1.0f0, A, B, 0.0f0, C)
    finally
        CUDACore.math_mode!(old)
    end
    return C
end
tf32_gemm(A, B) = gemm_mode(A, B, CUDACore.FAST_MATH)      # TF32 tensor cores
fp32_gemm(A, B) = gemm_mode(A, B, CUDACore.DEFAULT_MATH)    # full FP32

# --- TF32 round via Veltkamp split (s = 13 -> hi has 24-13 = 11 significant
# bits, exactly TF32-representable). -----------------------------------------
const VC = Float32(2^13 + 1)
tf32_hi(A) = (g = VC .* A; g .- (g .- A))

# A = A0 + A1 + eA, with A0, A1 TF32-exact and eA the (FP32-exact) residual.
function split2(A)
    A0 = tf32_hi(A)
    r  = A .- A0
    A1 = tf32_hi(r)
    eA = r .- A1
    return A0, A1, eA
end

with_mode(f, m) = (old = CUDACore.math_mode(); CUDACore.math_mode!(m);
                   try f() finally CUDACore.math_mode!(old) end)

function run(M, N, K; seed = 1)
    rng = MersenneTwister(seed)
    A = randn(rng, Float32, M, K)
    B = randn(rng, Float32, K, N)
    dA = CuArray(A); dB = CuArray(B)

    # References
    C64 = Float64.(A) * Float64.(B)                       # accurate center
    relerr(C) = maximum(abs.(Float64.(C) .- C64)) /
                maximum(abs.(C64))

    # 1. FP32 baseline (full Float32 compute, no tensor cores).
    C_fp32 = Array(fp32_gemm(dA, dB)); CUDACore.synchronize()

    # 2. Naive TF32 (inputs rounded to TF32 by the compute type).
    C_tf32 = Array(tf32_gemm(dA, dB)); CUDACore.synchronize()

    # 3. Split + Ozaki on the TF32 tensor cores.
    A0, A1, _ = split2(dA)
    B0, B1, _ = split2(dB)

    # TF32-exactness check: gemmEx! of TF32-exact A0,B0 under TF32 compute must
    # equal the FULL-FP32 compute of the same product (only accumulation
    # differs, FP32 in both). If equal, the tensor core did not round A0/B0.
    exact = Array(tf32_gemm(A0, B0)) == Array(fp32_gemm(A0, B0))

    C_oz  = tf32_gemm(A0,B0) .+ tf32_gemm(A0,B1) .+ tf32_gemm(A1,B0) .+ tf32_gemm(A1,B1)
    C_oz3 = tf32_gemm(A0,B0) .+ tf32_gemm(A0,B1) .+ tf32_gemm(A1,B0)   # drop A1*B1
    CUDACore.synchronize()
    C_oz = Array(C_oz); C_oz3 = Array(C_oz3)

    @printf("  M=%d N=%d K=%d\n", M, N, K)
    @printf("    FP32   relerr = %.3e\n", relerr(C_fp32))
    @printf("    TF32   relerr = %.3e   (TF32 engaged: %s)\n",
            relerr(C_tf32), relerr(C_tf32) > 10 * relerr(C_fp32) ? "YES" : "no/maybe")
    @printf("    A0 TF32-exact under TF32 GEMM: %s\n", exact)
    @printf("    Ozaki-4 relerr = %.3e\n", relerr(C_oz))
    @printf("    Ozaki-3 relerr = %.3e   (dropped A1*B1)\n", relerr(C_oz3))
    println()
end

println("Device: ", CUDACore.name(CUDACore.device()))
println("eps(Float32) = ", eps(Float32), "   2^-11 (TF32 unit) = ", 2.0^-11)
println()
for (M, N, K) in [(256,256,256), (512,512,512), (1024,1024,1024), (2048,2048,2048)]
    run(M, N, K)
end
