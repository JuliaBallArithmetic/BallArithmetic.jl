# Task 1: TIGHT rigorous radius for the INT8-Ozaki midpoint product.
#
# The loose bound (probe_int8_rigor.jl) used  loA·fullB + fullA·loB, which pairs
# every i>m slice with ALL j — over-counting many pairs with i+j<=T that are NOT
# dropped (e.g. (i=6,j=1) for T=10). Those non-dropped, high-magnitude terms
# dominated the bound (~1e6x the true error).
#
# Tight bound: the dropped set is exactly {(i,j): i+j>T}, so
#     Σ_{i+j>T} x_i y_j  =  Σ_i x_i · ( Σ_{j>T-i} y_j )  =  Σ_i x_i · Ytail_{T-i}
# with x_i = 2^-7i|DA_i|, y_j = 2^-7j|DB_j|.  Computed in FP32 (it's only an upper
# bound) — a handful of GEMMs.  Plus: rigorous recombination bound (Σ|terms|·u)
# and a residual bound (row/col sums, no GEMM).

using CUDACore, CUDA
using Random, Printf, LinearAlgebra

const gemmI8! = CUDA.CUBLAS.gemmEx!
const B_BITS = 7; const BASE = 2.0^B_BITS
const U64 = eps(Float64); const U32 = Float64(eps(Float32))

scale_exp(A; dims) = (mx = maximum(abs.(A); dims=dims);
    ifelse.(mx .== 0, 0.0, ceil.(log2.(max.(mx, floatmin(Float64)))) .+ 1.0))
function slices(Ahat, s)
    D = Vector{CuArray{Int8,2}}(undef, s); t = copy(Ahat)
    for i in 1:s; t .*= BASE; di = round.(t); D[i] = Int8.(di); t .-= di; end
    return D
end

function ozaki_midpoint_tight(dmA, dmB; s=8, T=10)
    M_, N_, K_ = size(dmA,1), size(dmB,2), size(dmA,2)
    σA = scale_exp(dmA; dims=2); σB = scale_exp(dmB; dims=1)
    scaleA = 2.0 .^ σA; scaleB = 2.0 .^ σB
    DA = slices(dmA .* (2.0 .^ (.-σA)), s); DB = slices(dmB .* (2.0 .^ (.-σB)), s)

    # center + Σ|terms| for the recombination bound
    S    = CUDA.zeros(Float64, M_, N_)
    Sabs = CUDA.zeros(Float64, M_, N_)
    Cij  = CUDA.zeros(Int32, M_, N_); ng = 0
    for i in 1:s, j in 1:s
        i+j <= T || continue
        fill!(Cij, Int32(0)); gemmI8!('N','N', Int32(1), DA[i], DB[j], Int32(0), Cij)
        term = Float64.(Cij) .* (2.0^(-B_BITS*(i+j)))
        S .+= term; Sabs .+= abs.(term); ng += 1
    end
    m = (S .* scaleA) .* scaleB

    # tight dropped-tail bound in FP32
    xs = [Float32(2.0^(-B_BITS*i)) .* abs.(Float32.(DA[i])) for i in 1:s]   # M×K
    ys = [Float32(2.0^(-B_BITS*j)) .* abs.(Float32.(DB[j])) for j in 1:s]   # K×N
    Ytail = Vector{CuArray{Float32,2}}(undef, s+1)                # Ytail[p+1]=Σ_{j>p} y_j
    Ytail[s+1] = CUDA.zeros(Float32, K_, N_)
    for p in s-1:-1:0; Ytail[p+1] = Ytail[p+2] .+ ys[p+1]; end
    bnd32 = CUDA.zeros(Float32, M_, N_); ngb = 0
    for i in 1:s
        p = T - i
        p >= s && continue          # no dropped j for this i
        bnd32 .+= xs[i] * Ytail[p+1]    # FP32 GEMM
        ngb += 1
    end
    infl = 1.0 + Float64(2K_ + 8) * U32
    tail = (Float64.(bnd32) .* infl)

    # residual bound (slice leftover beyond s): 2^-7s-1 · (rowsumA + colsumB)
    fullA32 = reduce(.+, xs); fullB32 = reduce(.+, ys)
    rowsumA = Float64.(sum(fullA32; dims=2))      # M×1
    colsumB = Float64.(sum(fullB32; dims=1))      # 1×N
    resid = (2.0^(-B_BITS*s - 1)) .* (rowsumA .+ colsumB) .* (1.0 + Float64(2K_)*U32)

    # assemble ρ (all rounded up generously)
    dropped = ((tail .+ resid) .* scaleA) .* scaleB
    recomb  = Float64(ng + 3) * U64 .* ((Sabs .* scaleA) .* scaleB)
    ρ = (dropped .+ recomb) .* (1.0 + 8U64)
    return m, ρ, ng, ngb
end

function run(M, N, K; s=8, T=10, seed=1)
    rng = MersenneTwister(seed)
    A = randn(rng, Float64, M, K); B = randn(rng, Float64, K, N)
    dA = CuArray(A); dB = CuArray(B)
    m, ρ, ng, ngb = ozaki_midpoint_tight(dA, dB; s=s, T=T); CUDACore.synchronize()
    mC = Array(m); rC = Array(ρ)
    setprecision(BigFloat, 300) do
        P = BigFloat.(A) * BigFloat.(B)
        lo = BigFloat.(mC) .- BigFloat.(rC); hi = BigFloat.(mC) .+ BigFloat.(rC)
        inside = all(lo .<= P .<= hi)
        ctr_err = maximum(abs.(BigFloat.(mC) .- P))
        relrad  = Float64(maximum(BigFloat.(rC)) / maximum(abs.(P)))
        ratio   = Float64(maximum(BigFloat.(rC)) / max(ctr_err, eps(BigFloat)))
        @printf("  %d^3 s=%d T=%d (%d int8 + %d fp32 gemms): ENCLOSE=%s  ctr_err=%.2e  radius_rel=%.2e  radius/err=%.1fx\n",
            M, s, T, ng, ngb, inside ? "OK" : "FAIL", Float64(ctr_err), relrad, ratio)
    end
end

println("Device: ", CUDACore.name(CUDACore.device()), "\n")
for T in (8, 10, 12); run(256, 256, 256; T=T); end
for T in (8, 10, 12); run(512, 512, 512; T=T); end
