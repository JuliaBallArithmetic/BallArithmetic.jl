# Task 3: Scheme II (CRT) with a FULLY ON-GPU reconstruction.
#
# Fixes the three things that made the CPU version ~100x slower than Scheme I:
#   * residues computed in Float64 (NOT Int64 — consumer GPUs run Int64 ~1/32 rate),
#   * Garner mixed-radix digits computed on-GPU in Int32,
#   * the big-integer combine + balanced reduction done in DOUBLE-DOUBLE in a
#     single CUDA kernel (no Int128, no host transfers).
#
# Why double-double suffices: the exact integer X = A'·B' is ~2^114, but the FP64
# result needs only its top ~53 bits. Reconstructing X·2^-2k = Σ_t c_t·(P_t·2^-2k)
# in dd (~106 bits) keeps enough; the balanced step (subtract M·2^-2k when
# X >= M/2) is a dd subtraction whose ~15-bit leading cancellation still leaves
# ~90 bits — comfortably > 53.

using CUDACore, CUDA
using Random, Printf, LinearAlgebra

const MODULI  = Int[256,255,253,251,247,239,233,229,227,223,217,211,199,197,193]
const gemmI8! = CUDA.CUBLAS.gemmEx!

scale_exp(A; dims) = (mx = maximum(abs.(A); dims=dims);
    ifelse.(mx .== 0, 0.0, ceil.(log2.(max.(mx, floatmin(Float64)))) .+ 1.0))

# Balanced residue of Float64 integer-valued Ap (|Ap|<=2^52) mod m, as INT8.
function residue_i8_f64(Ap, m)
    r = Ap .- m .* floor.(Ap ./ m)        # exact: m*floor(...) <= 2^52
    r = ifelse.(r .< 0, r .+ m, r)
    r = ifelse.(r .>= m, r .- m, r)        # now r in [0,m)
    rb = ifelse.(2 .* r .>= m, r .- m, r)  # balanced -> [-m/2, m/2)
    return Int8.(rb)
end

# --- double-double combine kernel ---
@inline function dd_add(ah, al, bh, bl)
    s = ah + bh; bb = s - ah
    e = (ah - (s - bb)) + (bh - bb) + al + bl
    sh = s + e; sl = e - (sh - s)
    return sh, sl
end

function combine_kernel!(out, C3, wh, wl, Wh, Wl, Whalf, scaleA, scaleB, M, N, s)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= M * N
        r = (idx - 1) % M + 1
        c = (idx - 1) ÷ M + 1
        acch = 0.0; accl = 0.0
        @inbounds for t in 1:s
            ct = C3[t, r, c]
            ph = ct * wh[t]
            pe = fma(ct, wh[t], -ph) + ct * wl[t]
            acch, accl = dd_add(acch, accl, ph, pe)
        end
        if acch >= Whalf                       # X >= M/2  ->  signed: subtract M
            acch, accl = dd_add(acch, accl, -Wh, -Wl)
        end
        @inbounds out[r, c] = (acch + accl) * scaleA[r] * scaleB[c]
    end
    return nothing
end

function schemeII_gpu(A, B; s=14, k=53)
    M_, N_ = size(A,1), size(B,2)
    dA = CuArray(A); dB = CuArray(B)
    σA = scale_exp(dA; dims=2); σB = scale_exp(dB; dims=1)
    Ap = round.((2.0^k) .* (dA .* (2.0 .^ (.-σA))))    # Float64, |.|<=2^(k-1)
    Bp = round.((2.0^k) .* (dB .* (2.0 .^ (.-σB))))
    mods = MODULI[1:s]

    # one exact INT8 GEMM per modulus -> r_t = X mod m_t  (Int32, on GPU)
    R = Vector{CuArray{Int32,2}}(undef, s)
    Cij = CUDA.zeros(Int32, M_, N_)
    for (t,m) in enumerate(mods)
        At = residue_i8_f64(Ap, m); Bt = residue_i8_f64(Bp, m)
        fill!(Cij, Int32(0)); gemmI8!('N','N', Int32(1), At, Bt, Int32(0), Cij)
        R[t] = mod.(Cij, Int32(m))
    end

    # Garner mixed-radix digits, on GPU in Int32
    minv = [j < t ? Int32(invmod(mods[j], mods[t])) : Int32(0) for j in 1:s, t in 1:s]
    cdig = Vector{CuArray{Int32,2}}(undef, s)
    for t in 1:s
        ct = copy(R[t]); mt = Int32(mods[t])
        for j in 1:t-1
            ct = mod.((ct .- cdig[j]) .* minv[j,t], mt)
        end
        cdig[t] = ct
    end

    # stack digits into (s, M, N) Float64 for the kernel
    C3 = CUDA.zeros(Float64, s, M_, N_)
    for t in 1:s
        @views C3[t, :, :] .= Float64.(cdig[t])
    end

    # dd weights w_t = P_t * 2^-2k  and  W = M * 2^-2k  (computed in BigFloat)
    wh = zeros(Float64, s); wl = zeros(Float64, s)
    Mbig = prod(BigInt.(mods))
    setprecision(BigFloat, 400) do
        scale = BigFloat(2)^(-2k)
        P = BigInt(1)
        for t in 1:s
            w = BigFloat(P) * scale
            wh[t] = Float64(w); wl[t] = Float64(w - wh[t])
            P *= mods[t]
        end
        global Wbf = BigFloat(Mbig) * scale
    end
    Wh = Float64(Wbf); Wl = Float64(Wbf - Wh); Whalf = Float64(Wbf / 2)

    out = CUDA.zeros(Float64, M_, N_)
    sA = vec(2.0 .^ σA); sB = vec(2.0 .^ σB)
    dwh = CuArray(wh); dwl = CuArray(wl)
    threads = 256; blocks = cld(M_*N_, threads)
    @cuda threads=threads blocks=blocks combine_kernel!(out, C3, dwh, dwl,
        Wh, Wl, Whalf, sA, sB, M_, N_, s)
    return out
end

function run(M, N, K; s=14, k=53, seed=1)
    rng = MersenneTwister(seed)
    A = randn(rng, Float64, M, K); B = randn(rng, Float64, K, N)
    P = setprecision(BigFloat, 300) do; BigFloat.(A) * BigFloat.(B); end
    relerr(C) = Float64(maximum(abs.(BigFloat.(C) .- P)) / maximum(abs.(P)))
    C = Array(schemeII_gpu(A, B; s=s, k=k)); CUDACore.synchronize()
    @printf("  %d^3  s=%d k=%d:  relerr = %.3e\n", M, s, k, relerr(C))
end

println("Device: ", CUDACore.name(CUDACore.device()), "\n")
for s in (13, 14, 15); run(256, 256, 256; s=s); end
for s in (14, 15);     run(512, 512, 512; s=s); end
