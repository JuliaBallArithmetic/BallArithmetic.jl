# Probe: Ozaki Scheme II (CRT) — INT8 GEMM, one per modulus, reconstructed via
# the Chinese Remainder Theorem. arXiv:2504.08009.
#
# Scale A,B to integer matrices A' = round(2^k ahat), B' = round(2^k bhat)
# (per-row/col power-of-two scaling, ahat,bhat in [-1/2,1/2]). The exact integer
# product X = A'·B' has |X| <= K·2^(2k-2) = q.  Pick s coprime moduli m_t with
# M = Πm_t > 2q.  For each t:
#     A'_t = balanced(A' mod m_t) in INT8,  B'_t likewise,
#     C_t  = A'_t · B'_t      (exact INT8->INT32 tensor-core GEMM),
#     r_t  = C_t mod m_t  ≡  X mod m_t .
# CRT (Garner) reconstructs X exactly from {r_t}; then C = 2^(σA+σB-2k) X.
#
# Cost: s GEMMs (vs Scheme-I's ~T^2/2 = 43). Reconstruction here is on CPU
# (Garner + Int128) — O(M·N·s^2), independent of K; GPU-side is future work.

using CUDACore, CUDA
using Random, Printf, LinearAlgebra

const MODULI  = Int[256,255,253,251,247,239,233,229,227,223,217,211,199,197,193]
const gemmI8! = CUDA.CUBLAS.gemmEx!

scale_exp(A; dims) = (mx = maximum(abs.(A); dims=dims);
    ifelse.(mx .== 0, 0.0, ceil.(log2.(max.(mx, floatmin(Float64)))) .+ 1.0))

# Balanced residue of an Int64 matrix mod m, as INT8 in [-128,127].
function residue_i8(Ap, m)
    r  = mod.(Ap, Int64(m))
    rb = ifelse.(2 .* r .>= m, r .- m, r)   # -> [-m/2, m/2)
    return Int8.(rb)
end

function schemeII(A, B; s=15, k=53)
    M_, N_ = size(A,1), size(B,2)
    dA = CuArray(A); dB = CuArray(B)
    σA = scale_exp(dA; dims=2); σB = scale_exp(dB; dims=1)
    Ap = Int64.(round.((2.0^k) .* (dA .* (2.0 .^ (.-σA)))))   # |A'| <= 2^(k-1)
    Bp = Int64.(round.((2.0^k) .* (dB .* (2.0 .^ (.-σB)))))
    mods = MODULI[1:s]

    R = Vector{Matrix{Int64}}(undef, s)
    Cij = CUDA.zeros(Int32, M_, N_)
    for (t,m) in enumerate(mods)
        At = residue_i8(Ap, m); Bt = residue_i8(Bp, m)
        fill!(Cij, Int32(0))
        gemmI8!('N','N', Int32(1), At, Bt, Int32(0), Cij)
        R[t] = Int64.(Array(mod.(Cij, Int32(m))))     # r_t = X mod m_t
    end

    # --- Garner mixed-radix reconstruction (CPU) ---
    minv = [j < t ? invmod(mods[j], mods[t]) : 0 for j in 1:s, t in 1:s]
    c = Vector{Matrix{Int64}}(undef, s)
    for t in 1:s
        ct = copy(R[t])
        for j in 1:t-1
            ct = mod.((ct .- c[j]) .* minv[j,t], mods[t])
        end
        c[t] = ct
    end
    Mbig = prod(Int128.(mods))
    X = Int128.(c[s])
    for t in s-1:-1:1
        X = X .* Int128(mods[t]) .+ c[t]
    end
    X = ifelse.(X .>= Mbig ÷ 2, X .- Mbig, X)        # balanced -> signed
    C = Float64.(X) .* (2.0^(-2k)) .* Array(2.0 .^ σA) .* Array(2.0 .^ σB)
    return C, length(mods)
end

function run(M, N, K; ks=[53])
    rng = MersenneTwister(1)
    A = randn(rng, Float64, M, K); B = randn(rng, Float64, K, N)
    P = setprecision(BigFloat, 300) do; BigFloat.(A) * BigFloat.(B); end
    relerr(C) = Float64(maximum(abs.(BigFloat.(C) .- P)) / maximum(abs.(P)))
    @printf("  M=%d N=%d K=%d   (native FP64 gpu relerr = %.2e)\n",
            M, N, K, relerr(Array(CuArray(A)*CuArray(B))))
    for k in ks, s in [12, 13, 14, 15]
        C, ng = schemeII(A, B; s=s, k=k)
        @printf("    Scheme II  s=%-2d k=%-2d (%2d gemms)  relerr = %.3e\n", s, k, ng, relerr(C))
    end
    println()
end

println("Device: ", CUDACore.name(CUDACore.device()))
println("eps(Float64) = ", eps(Float64), "\n")
run(256, 256, 256)
run(512, 512, 512)
