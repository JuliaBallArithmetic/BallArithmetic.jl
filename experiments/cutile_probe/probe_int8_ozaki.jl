# Probe: INT8-Ozaki GEMM (slice-based, the Scheme-I precursor to arXiv:2504.08009).
#
# INT8 x INT8 -> INT32 tensor-core products are EXACT. We split each FP64
# matrix into base-2^7 integer "slices", do exact INT8 GEMMs, and recombine.
#
#   A[r,k] = 2^σA[r] * ahat[r,k],   ahat = Σ_i Di_A[i] 2^(-7 i)   (Di_A INT8)
#   B[k,c] = 2^σB[c] * bhat[k,c],   bhat = Σ_j Dj_B[j] 2^(-7 j)
#   C = Diag(2^σA) * [ Σ_{i,j} 2^(-7(i+j)) (Di_A[i] · Dj_B[j]) ] * Diag(2^σB)
#
# Each (Di_A[i] · Dj_B[j]) is an exact INT8 tensor-core GEMM. Truncating to
# i+j <= T drops only terms of size <= 2^(-7T) -> a-priori bound (rigor probe
# is in probe_int8_rigor.jl). Here we check exactness/accuracy vs BigFloat.

using CUDACore, CUDA
using Random, Printf, LinearAlgebra

const gemmI8! = CUDA.CUBLAS.gemmEx!   # Int8*Int8 -> Int32, CUBLAS_COMPUTE_32I
const B_BITS  = 7
const BASE    = 2.0^B_BITS            # 128

# Per-row (dim=2) scaling exponent so |ahat| <= 1/2.  Returns σ::CuArray (Mx1).
function scale_exp(A; dims)
    mx = maximum(abs.(A); dims=dims)
    σ  = ceil.(log2.(max.(mx, floatmin(Float64)))) .+ 1.0
    σ  = ifelse.(mx .== 0, 0.0, σ)     # zero rows/cols -> no scaling
    return σ
end

# Split scaled matrix Ahat (|Ahat|<=1/2) into `s` INT8 slice matrices.
function slices(Ahat, s)
    D = Vector{CuArray{Int8,2}}(undef, s)
    t = copy(Ahat)
    for i in 1:s
        t .*= BASE
        di = round.(t)
        D[i] = Int8.(di)
        t .-= di
    end
    return D
end

# INT8-Ozaki product. s slices each side; include pairs with i+j <= T.
function int8_ozaki(dA, dB; s=8, T=2s)
    σA = scale_exp(dA; dims=2)          # M x 1
    σB = scale_exp(dB; dims=1)          # 1 x N
    Ahat = dA .* (2.0 .^ (.-σA))
    Bhat = dB .* (2.0 .^ (.-σB))
    DA = slices(Ahat, s)
    DB = slices(Bhat, s)

    M, N = size(dA, 1), size(dB, 2)
    S = CUDA.zeros(Float64, M, N)
    Cij = CUDA.zeros(Int32, M, N)
    ngemm = 0
    for i in 1:s, j in 1:s
        i + j <= T || continue
        fill!(Cij, Int32(0))
        gemmI8!('N','N', Int32(1), DA[i], DB[j], Int32(0), Cij)
        S .+= Float64.(Cij) .* (2.0^(-B_BITS*(i+j)))
        ngemm += 1
    end
    C = S .* (2.0 .^ σA) .* (2.0 .^ σB)
    return C, ngemm
end

function run(M, N, K; seed=1)
    rng = MersenneTwister(seed)
    A = randn(rng, Float64, M, K); B = randn(rng, Float64, K, N)
    dA = CuArray(A); dB = CuArray(B)

    P = setprecision(BigFloat, 250) do; BigFloat.(A) * BigFloat.(B); end
    relerr(C) = Float64(maximum(abs.(BigFloat.(C) .- P)) / maximum(abs.(P)))

    fp64_gpu = Array(dA * dB)                 # native FP64 gemm (round-to-nearest)
    @printf("  M=%d N=%d K=%d\n", M, N, K)
    @printf("    native FP64 gpu relerr = %.3e\n", relerr(fp64_gpu))
    for (s, T) in [(4, 4), (6, 6), (8, 8), (8, 10), (8, 16)]
        C, ng = int8_ozaki(dA, dB; s=s, T=T); CUDACore.synchronize()
        @printf("    int8-ozaki s=%-2d T=%-2d (%2d gemms) relerr = %.3e\n",
                s, T, ng, relerr(Array(C)))
    end
    println()
end

println("Device: ", CUDACore.name(CUDACore.device()))
println("eps(Float64) = ", eps(Float64), "\n")
run(256, 256, 256)
run(512, 512, 512)
