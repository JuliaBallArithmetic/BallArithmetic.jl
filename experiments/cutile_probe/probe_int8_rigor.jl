# Probe: rigorous INT8-Ozaki ball GEMM — verified enclosure of the exact A*B.
#
# Center: truncated INT8-Ozaki (pairs i+j <= T), computed in FP64.
# Radius: a-priori bound on
#   (a) dropped slice-pairs (i+j > T) and slice residual, and
#   (b) FP64 recombination rounding.
# Since INT8 products are EXACT, the true product is
#   A*B = 2^(σA+σB) Σ_{all i,j} 2^(-7(i+j)) (DA_i · DB_j)   (exact),
# so the error is purely the omitted terms + FP64 add rounding — both bounded.
#
# i+j > T  ==>  i > m or j > m  (m = floor(T/2)), hence
#   |A*B - center| <= 2^(σA+σB) ( loA·fullB + fullA·loB )
# with loA = Σ_{i>m} 2^(-7i)|DA_i| + residual,  fullA = Σ_i 2^(-7i)|DA_i| + residual.

using CUDACore, CUDA
using Random, Printf, LinearAlgebra

const gemmI8! = CUDA.CUBLAS.gemmEx!
const B_BITS  = 7
const BASE    = 2.0^B_BITS
const U        = eps(Float64)

scale_exp(A; dims) = (mx = maximum(abs.(A); dims=dims);
    ifelse.(mx .== 0, 0.0, ceil.(log2.(max.(mx, floatmin(Float64)))) .+ 1.0))

# Return slice matrices AND the final residual t (Ahat = Σ D_i 2^-7i + t·2^-7s).
function slices(Ahat, s)
    D = Vector{CuArray{Int8,2}}(undef, s)
    t = copy(Ahat)
    for i in 1:s
        t .*= BASE; di = round.(t); D[i] = Int8.(di); t .-= di
    end
    return D, t
end

function int8_ozaki_ball(dA, dB; s=8, T=8)
    M, N, K = size(dA,1), size(dB,2), size(dA,2)
    σA = scale_exp(dA; dims=2); σB = scale_exp(dB; dims=1)
    DA, _ = slices(dA .* (2.0 .^ (.-σA)), s)
    DB, _ = slices(dB .* (2.0 .^ (.-σB)), s)

    # --- center: kept pairs i+j <= T, accumulated in FP64 ---
    S = CUDA.zeros(Float64, M, N); Cij = CUDA.zeros(Int32, M, N); ngemm = 0
    for i in 1:s, j in 1:s
        i + j <= T || continue
        fill!(Cij, Int32(0)); gemmI8!('N','N', Int32(1), DA[i], DB[j], Int32(0), Cij)
        S .+= Float64.(Cij) .* (2.0^(-B_BITS*(i+j))); ngemm += 1
    end
    scaleA = 2.0 .^ σA; scaleB = 2.0 .^ σB
    center = (S .* scaleA) .* scaleB

    # --- radius ---
    m = fld(T, 2)
    resid = 2.0^(-B_BITS*s - 1)            # bound on |slice residual| (scaled units)
    fullA = CUDA.zeros(Float64, M, K); loA = CUDA.zeros(Float64, M, K)
    fullB = CUDA.zeros(Float64, K, N); loB = CUDA.zeros(Float64, K, N)
    for i in 1:s
        w = 2.0^(-B_BITS*i)
        fullA .+= w .* abs.(Float64.(DA[i])); fullB .+= w .* abs.(Float64.(DB[i]))
        if i > m
            loA .+= w .* abs.(Float64.(DA[i])); loB .+= w .* abs.(Float64.(DB[i]))
        end
    end
    fullA .+= resid; fullB .+= resid; loA .+= resid; loB .+= resid

    # two nonneg matmuls; inflate to a rigorous upper bound for FP64 gemm rounding
    infl = 1.0 + (K + 2) * U
    bound = (loA * fullB) .* infl
    bound .+= (fullA * loB) .* infl
    dropped = ((bound .* scaleA) .* scaleB) .* infl

    rec = Float64(ngemm + 3) * U .* abs.(center)     # recombination rounding
    radius = (dropped .+ rec) .* (1.0 + 8U)          # small safety margin
    return center, radius, ngemm
end

function run(M, N, K; s=8, T=8, seed=1)
    rng = MersenneTwister(seed)
    A = randn(rng, Float64, M, K); B = randn(rng, Float64, K, N)
    dA = CuArray(A); dB = CuArray(B)
    c, r, ng = int8_ozaki_ball(dA, dB; s=s, T=T); CUDACore.synchronize()
    mC = Array(c); rC = Array(r)
    setprecision(BigFloat, 300) do
        P = BigFloat.(A) * BigFloat.(B)
        lo = BigFloat.(mC) .- BigFloat.(rC); hi = BigFloat.(mC) .+ BigFloat.(rC)
        inside = all(lo .<= P .<= hi)
        ctr_err = maximum(abs.(BigFloat.(mC) .- P))
        relrad  = Float64(maximum(BigFloat.(rC)) / maximum(abs.(P)))
        tightness = Float64(maximum(BigFloat.(rC)) / max(ctr_err, eps(BigFloat)))
        @printf("  M=%d N=%d K=%d  s=%d T=%d (%d gemms)\n", M, N, K, s, T, ng)
        @printf("    ENCLOSURE: %s   center max err = %.2e   radius (rel) = %.2e   radius/err = %.1fx\n",
            inside ? "OK" : "FAILED", Float64(ctr_err), relrad, tightness)
    end
end

println("Device: ", CUDACore.name(CUDACore.device()), "\n")
run(256, 256, 256; T=8)
run(256, 256, 256; T=10)
run(512, 512, 512; T=8)
run(512, 512, 512; T=10)
