# gemmul8_port.jl — a Julia port of RIKEN R-CCS's GEMMul8 (Ozaki Scheme II).
#
# This is a Julia/CUDA.jl port of the INT8 DGEMM path of GEMMul8:
#     https://github.com/RIKEN-RCCS/GEMMul8   (MIT License, (c) 2025- RIKEN R-CCS)
# based on:
#     Ozaki, Uchino, Imamura, "Ozaki Scheme II", arXiv:2504.08009 (2025).
#
# Original GEMMul8 license (MIT) reproduced per its terms:
#   MIT License. Copyright (c) 2025- RIKEN R-CCS. Permission is hereby granted,
#   free of charge, ... THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY ...
#   (full text: /tmp/GEMMul8/LICENSE and the upstream repository).
#
# Key fusion ported from GEMMul8 (vs the earlier hand-rolled probe_schemeII_gpu.jl):
#   * NO Garner mixed-radix. The reconstruction uses the DIRECT CRT formula with
#     PRECOMPUTED weights qPi = q_i·P_i (P_i=P/p_i, q_i=P_i^{-1} mod p_i), stored
#     as double-double, accumulated with two FMAs/term (GEMMul8 accumulator_double2),
#     then the balanced reduction  X = C64f - P·rint(C64f/P)  (rint => signed),
#     and the diagonal rescale — ALL in ONE fused per-element kernel
#     (GEMMul8 invscal_device).

using CUDACore, CUDA
using Random, Printf, LinearAlgebra

const MODULI  = Int[256,255,253,251,247,239,233,229,227,223,217,211,199,197,193]
const gemmI8! = CUDA.CUBLAS.gemmEx!

scale_exp(A; dims) = (mx = maximum(abs.(A); dims=dims);
    ifelse.(mx .== 0, 0.0, ceil.(log2.(max.(mx, floatmin(Float64)))) .+ 1.0))

# Fused residue extraction (port of GEMMul8 extract_A_lo / scaling): read each
# input element ONCE, form the scaled integer A' = round(X·2^(k-σ)), and write
# all `s` balanced-INT8 residues A' mod p_i.  `lo` is (rows, cols, s); `sft` is
# the per-row (A) or per-col (B) shift exponent; `rowwise` selects which.
function fused_residues_kernel!(lo, X, sft, mods, nrows, sz, s, rowwise, k)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= sz
        @inbounds begin
            r = (idx - 1) % nrows + 1
            c = (idx - 1) ÷ nrows + 1
            e = rowwise ? sft[r] : sft[c]
            ap = round(X[idx] * exp2(Float64(k) - e))      # A' = round(X·2^(k-σ))
            for i in 1:s
                m  = mods[i]
                rr = ap - m * floor(ap / m)                # A' mod m  in [0,m)
                rr = rr < 0 ? rr + m : rr
                rr = rr >= m ? rr - m : rr
                rb = (2 * rr >= m) ? rr - m : rr           # balance -> [-m/2, m/2)
                lo[idx + (i - 1) * sz] = unsafe_trunc(Int8, rb)
            end
        end
    end
    return nothing
end

function fused_residues!(lo, X, sft, mods_d, nrows, s, k, rowwise)
    sz = length(X)
    threads = 256; blocks = cld(sz, threads)
    @cuda threads=threads blocks=blocks fused_residues_kernel!(
        lo, X, sft, mods_d, nrows, sz, s, rowwise, k)
    return lo
end

# --- fused inverse-scaling kernel (port of GEMMul8 invscal_device) ---
# Cmid layout (sizeC, s): residue of modulus i at linear offset idx + (i-1)*sizeC.
function invscal_kernel!(C, Cmid, qhi, qlo, Px, Py, invP, scaleA, scaleB, M, sizeC, s)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= sizeC
        col = (idx - 1) ÷ M + 1
        row = (idx - 1) % M + 1
        chi = 0.0; clo = 0.0
        @inbounds for i in 1:s
            a = Float64(Cmid[idx + (i-1)*sizeC])
            chi = fma(qhi[i], a, chi)      # error-free part (GEMMul8 .x)
            clo = fma(qlo[i], a, clo)      # correction part  (GEMMul8 .y)
        end
        quot = round(invP * chi)                       # rint(C64f / P)
        crt  = fma(Py, quot, fma(Px, quot, chi) + clo) # C64f - P*quot  (P = -prod)
        @inbounds C[idx] = crt * scaleA[row] * scaleB[col]
    end
    return nothing
end

function gemmul8_dgemm(A, B; num_moduli=14, k=53)
    M_, N_, K_ = size(A,1), size(B,2), size(A,2)
    sizeC = M_ * N_; s = num_moduli; mods = MODULI[1:s]
    dA = CuArray(A); dB = CuArray(B)
    σA = scale_exp(dA; dims=2); σB = scale_exp(dB; dims=1)

    # fused residue extraction: one pass over A and one over B produce ALL moduli
    mods_d = CuArray(Int32.(mods))
    Alo = CuArray{Int8}(undef, M_, K_, s); Blo = CuArray{Int8}(undef, K_, N_, s)
    fused_residues!(Alo, dA, vec(σA), mods_d, M_, s, k, true)   # per-row shift
    fused_residues!(Blo, dB, vec(σB), mods_d, K_, s, k, false)  # per-col shift

    # one INT8 GEMM per modulus; reduce mod p_i directly into the Cmid buffer
    Cmid = CUDA.zeros(Int32, sizeC, s)
    Cij  = CUDA.zeros(Int32, M_, N_)
    for (i,m) in enumerate(mods)
        At = @view Alo[:, :, i]; Bt = @view Blo[:, :, i]
        fill!(Cij, Int32(0)); gemmI8!('N','N', Int32(1), At, Bt, Int32(0), Cij)
        @views Cmid[:, i] .= vec(mod.(Cij, Int32(m)))
    end

    # precompute CRT weights (host) as double-double, GEMMul8-style: the hi part
    # is qPi truncated to (53 - ceil(log2(rho))) bits so the running accumulation
    # Σ qhi·a_i is EXACT (error-free); lo captures the remainder.  rho bounds the
    # accumulated residue magnitude.
    qhi = zeros(s); qlo = zeros(s)
    P = prod(BigInt.(mods))
    rho = sum(div(m, 2) for m in mods)
    guard = ceil(Int, log2(rho))
    E = ndigits(P, base = 2) - 1            # floor(log2(P)) = exponent of largest weight
    L = E - 52 + guard                      # round all weights to multiples of 2^L
    twoL = BigInt(1) << L
    for i in 1:s
        Pi  = P ÷ mods[i]
        qi  = invmod(mod(Pi, mods[i]), mods[i])
        qPi = qi * Pi                                  # exact BigInt weight
        n   = fld(qPi + (twoL >> 1), twoL)             # round to nearest multiple of 2^L
        qhi[i] = Float64(n) * 2.0^L                    # exact: n < 2^(52-guard)
        qlo[i] = Float64(qPi - n * twoL)               # remainder
    end
    Px = 0.0; Py = 0.0
    setprecision(BigFloat, 400) do
        Pneg = -BigFloat(P); Px = Float64(Pneg); Py = Float64(Pneg - Px)
    end
    invP = Float64(1 / BigFloat(P))

    out = CUDA.zeros(Float64, M_, N_)
    scaleA = vec(2.0 .^ (σA .- k)); scaleB = vec(2.0 .^ (σB .- k))
    dqhi = CuArray(qhi); dqlo = CuArray(qlo)
    threads = 256; blocks = cld(sizeC, threads)
    @cuda threads=threads blocks=blocks invscal_kernel!(out, Cmid, dqhi, dqlo,
        Px, Py, invP, scaleA, scaleB, M_, sizeC, s)
    return out
end

function run(M, N, K; s=14)
    rng = MersenneTwister(1)
    A = randn(rng, Float64, M, K); B = randn(rng, Float64, K, N)
    P = setprecision(BigFloat, 300) do; BigFloat.(A) * BigFloat.(B); end
    relerr(C) = Float64(maximum(abs.(BigFloat.(C) .- P)) / maximum(abs.(P)))
    C = Array(gemmul8_dgemm(A, B; num_moduli=s)); CUDACore.synchronize()
    @printf("  %d^3  num_moduli=%d:  relerr = %.3e\n", M, s, relerr(C))
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Device: ", CUDACore.name(CUDACore.device()), "\n")
    for s in (13, 14, 15); run(256, 256, 256; s=s); end
    run(512, 512, 512; s=14)
end
