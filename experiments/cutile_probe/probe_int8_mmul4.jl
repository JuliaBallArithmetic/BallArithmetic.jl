# Probe: a full GPU MMul4 (BallMatrix × BallMatrix → BallMatrix) built on
# INT8-Ozaki for the midpoint and a LOW-PRECISION (FP32) radius term.
#
#   m     = INT8-Ozaki(mA, mB)               exact FP64 midpoint product
#   ρ     = rigorous bound on |m − mA·mB|     (Ozaki truncation + recomb error)
#   rC_in = upper_bound(|mA|·rB + rA·(|mB|+rB))  ← FP32 GEMMs, inflated up
#   C     = ( m ,  ρ + rC_in )                 rounded up
#
# Validation: the result must contain the EXACT ball product per entry, whose
# true center is Σ_k mA·mB and true half-width is Σ_k (|mA|rB + rA|mB| + rA rB).
# We check containment of that interval (computed in BigFloat).

using CUDACore, CUDA
using Random, Printf, LinearAlgebra

const gemmI8! = CUDA.CUBLAS.gemmEx!
const B_BITS = 7; const BASE = 2.0^B_BITS
const U64 = eps(Float64); const U32 = eps(Float32)

scale_exp(A; dims) = (mx = maximum(abs.(A); dims=dims);
    ifelse.(mx .== 0, 0.0, ceil.(log2.(max.(mx, floatmin(Float64)))) .+ 1.0))
function slices(Ahat, s)
    D = Vector{CuArray{Int8,2}}(undef, s); t = copy(Ahat)
    for i in 1:s; t .*= BASE; di = round.(t); D[i] = Int8.(di); t .-= di; end
    return D
end

# Midpoint product (center m) + rigorous bound ρ on |m − mA·mB|.  (Scheme I.)
function ozaki_midpoint(dmA, dmB; s=8, T=10)
    M_, N_, K_ = size(dmA,1), size(dmB,2), size(dmA,2)
    σA = scale_exp(dmA; dims=2); σB = scale_exp(dmB; dims=1)
    DA = slices(dmA .* (2.0 .^ (.-σA)), s); DB = slices(dmB .* (2.0 .^ (.-σB)), s)
    S = CUDA.zeros(Float64, M_, N_); Cij = CUDA.zeros(Int32, M_, N_); ng = 0
    for i in 1:s, j in 1:s
        i+j <= T || continue
        fill!(Cij, Int32(0)); gemmI8!('N','N', Int32(1), DA[i], DB[j], Int32(0), Cij)
        S .+= Float64.(Cij) .* (2.0^(-B_BITS*(i+j))); ng += 1
    end
    scaleA = 2.0 .^ σA; scaleB = 2.0 .^ σB
    m = (S .* scaleA) .* scaleB

    # ρ: dropped slice-pairs (i+j>T) + recombination rounding (see probe_int8_rigor)
    mm = fld(T,2); resid = 2.0^(-B_BITS*s - 1)
    fullA = CUDA.zeros(Float64,M_,K_); loA = CUDA.zeros(Float64,M_,K_)
    fullB = CUDA.zeros(Float64,K_,N_); loB = CUDA.zeros(Float64,K_,N_)
    for i in 1:s
        w = 2.0^(-B_BITS*i)
        fullA .+= w.*abs.(Float64.(DA[i])); fullB .+= w.*abs.(Float64.(DB[i]))
        if i > mm; loA .+= w.*abs.(Float64.(DA[i])); loB .+= w.*abs.(Float64.(DB[i])); end
    end
    fullA .+= resid; fullB .+= resid; loA .+= resid; loB .+= resid
    infl = 1.0 + (K_+2)*U64
    bnd = (loA*fullB).*infl; bnd .+= (fullA*loB).*infl
    ρ = (((bnd .* scaleA) .* scaleB) .* infl) .+ Float64(ng+3)*U64 .* abs.(m)
    return m, ρ
end

# Low-precision (FP32) rigorous upper bound on |mA|·rB + rA·(|mB|+rB).
function radius_term_fp32(dmA, drA, dmB, drB)
    K_ = size(dmA, 2)
    amA = abs.(Float32.(dmA)); amB = abs.(Float32.(dmB))
    rA32 = Float32.(drA); rB32 = Float32.(drB)
    t1 = amA * rB32                    # |mA|·rB    (FP32 gemm, round-to-nearest)
    t2 = rA32 * (amB .+ rB32)          # rA·(|mB|+rB)
    raw = Float64.(t1 .+ t2)
    # inflate to a rigorous upper bound: FP32 accumulation (K·u32), the (|mB|+rB)
    # add, and the Float32 down-conversion of |mA|,|mB| (relative u32 each).
    infl = 1.0 + Float64(2K_ + 8) * Float64(U32)
    return raw .* infl
end

function gpu_mmul4(mA, rA, mB, rB; s=8, T=10)
    dmA=CuArray(mA); drA=CuArray(rA); dmB=CuArray(mB); drB=CuArray(rB)
    m, ρ = ozaki_midpoint(dmA, dmB; s=s, T=T)
    rC_in = radius_term_fp32(dmA, drA, dmB, drB)
    rC = (ρ .+ rC_in) .* (1.0 + 8U64)          # round up
    CUDACore.synchronize()
    return Array(m), Array(rC)
end

function run(M, N, K; radlevel=1e-10, s=8, T=10, seed=1)
    rng = MersenneTwister(seed)
    mA = randn(rng, Float64, M, K); rA = abs.(randn(rng, M, K)) .* radlevel
    mB = randn(rng, Float64, K, N); rB = abs.(randn(rng, K, N)) .* radlevel

    mC, rC = gpu_mmul4(mA, rA, mB, rB; s=s, T=T)

    # Exact ball product per entry, in BigFloat.
    inside, minmargin, maxrad_rel = setprecision(BigFloat, 300) do
        bmA=BigFloat.(mA); brA=BigFloat.(rA); bmB=BigFloat.(mB); brB=BigFloat.(rB)
        ctr = bmA * bmB
        hw  = abs.(bmA) * brB .+ brA * abs.(bmB) .+ brA * brB
        lo_t = ctr .- hw; hi_t = ctr .+ hw
        lo_g = BigFloat.(mC) .- BigFloat.(rC); hi_g = BigFloat.(mC) .+ BigFloat.(rC)
        ins = all((lo_g .<= lo_t) .& (hi_g .>= hi_t))
        mg  = Float64(minimum(min.(lo_t .- lo_g, hi_g .- hi_t)))   # >=0 if contained
        mr  = Float64(maximum(BigFloat.(rC)) / maximum(abs.(ctr)))
        (ins, mg, mr)
    end
    @printf("  %d^3  rad~%.0e  s=%d T=%d:  CONTAINS exact ball = %s   min margin = %.2e   radius(rel)=%.2e\n",
            M, radlevel, s, T, inside ? "YES" : "NO", minmargin, maxrad_rel)
end

println("Device: ", CUDACore.name(CUDACore.device()), "\n")
for rl in (0.0, 1e-12, 1e-8, 1e-4)
    run(256, 256, 256; radlevel=rl)
end
run(512, 512, 512; radlevel=1e-10)
