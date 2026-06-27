# Full BallMatrix MMul4 built on the GEMMul8 port (gemmul8_dgemm midpoint).
# Answers: once we add the radius, is the GPU MMul4 still a big win vs CPU MMul4?
#
#   m      = gemmul8_dgemm(mA, mB)               exact-CRT FP64 midpoint
#   ρ_trunc = |δA|·|mB| + |mA|·|δB| (+ tiny)      input-scaling truncation; since
#             |δA| <= 2^(σA-k-1) is per-ROW constant, this is row/col sums × bcast
#             (NO GEMM). Plus u·|m| for the final FP64 rounding.
#   rC_in  = |mA|·rB + rA·(|mB|+rB)               input-radius prop. (FP32, only if
#                                                 radii != 0; 2 GEMMs)
#   C = ( m , ρ_trunc + rC_in ) rounded up

using CUDACore, CUDA, Random, Printf, LinearAlgebra
using BallArithmetic
include("gemmul8_port.jl")

const U64 = eps(Float64); const U32 = Float64(eps(Float32))

function gemmul8_mmul4(mA, rA, mB, rB; num_moduli=14, k=53)
    M_, N_, K_ = size(mA,1), size(mB,2), size(mA,2)
    m = gemmul8_dgemm(mA, mB; num_moduli=num_moduli, k=k)      # midpoint (port)
    dmA = CuArray(mA); dmB = CuArray(mB)
    σA = scale_exp(dmA; dims=2); σB = scale_exp(dmB; dims=1)

    # --- midpoint truncation bound (no GEMM) ---
    absA = abs.(dmA); absB = abs.(dmB)
    rowsumA = sum(absA; dims=2) .* (1 + K_*U64)     # M×1, rigorous upper bound
    colsumB = sum(absB; dims=1) .* (1 + K_*U64)     # 1×N
    δA = 2.0 .^ (σA .- k .- 1)                      # M×1  (>= |δA| per row)
    δB = 2.0 .^ (σB .- k .- 1)                      # 1×N
    ρ_trunc = (δA .* colsumB) .+ (rowsumA .* δB) .+ (Float64(K_) .* (δA .* δB))

    # --- input-radius term (FP32 GEMMs, skipped if radii are zero) ---
    if any(rA .!= 0) || any(rB .!= 0)
        drA = CuArray(Float32.(rA)); drB = CuArray(Float32.(rB))
        amA = abs.(Float32.(dmA)); amB = abs.(Float32.(dmB))
        t1 = amA * drB; t2 = drA * (amB .+ drB)
        rC_in = Float64.(t1 .+ t2) .* (1 + (2K_+8)*U32)
    else
        rC_in = CUDA.zeros(Float64, M_, N_)
    end

    rC = (ρ_trunc .+ rC_in .+ (U64 .* abs.(m))) .* (1 + 8U64)
    CUDACore.synchronize()
    return Array(m), Array(rC)
end

gt(f;nw=2,nr=5)=(for _ in 1:nw;f();CUDACore.synchronize();end;ts=Float64[];for _ in 1:nr;t0=time_ns();f();CUDACore.synchronize();push!(ts,(time_ns()-t0)/1e9);end;sort(ts)[cld(nr,2)])
ct(f;nw=1,nr=3)=(for _ in 1:nw;f();end;ts=Float64[];for _ in 1:nr;t0=time_ns();f();push!(ts,(time_ns()-t0)/1e9);end;sort(ts)[cld(nr,2)])

# correctness: contain the exact ball product
function verify(M,N,K; radlevel=1e-10, seed=1)
    rng=MersenneTwister(seed)
    mA=randn(rng,M,K); rA=abs.(randn(rng,M,K)).*radlevel
    mB=randn(rng,K,N); rB=abs.(randn(rng,K,N)).*radlevel
    mC,rC = gemmul8_mmul4(mA,rA,mB,rB)
    setprecision(BigFloat,300) do
        ctr=BigFloat.(mA)*BigFloat.(mB)
        hw =abs.(BigFloat.(mA))*BigFloat.(rB) .+ BigFloat.(rA)*abs.(BigFloat.(mB)) .+ BigFloat.(rA)*BigFloat.(rB)
        lo_t=ctr.-hw; hi_t=ctr.+hw
        lo_g=BigFloat.(mC).-BigFloat.(rC); hi_g=BigFloat.(mC).+BigFloat.(rC)
        ins=all((lo_g.<=lo_t).&(hi_g.>=hi_t))
        @printf("  %d^3 rad~%.0e: CONTAINS exact ball=%s  radius_rel=%.2e\n",
            M, radlevel, ins ? "YES" : "NO", Float64(maximum(BigFloat.(rC))/maximum(abs.(ctr))))
    end
end

function bench(M,N,K)
    rng=MersenneTwister(1)
    mA=randn(rng,M,K); mB=randn(rng,K,N)
    Ab=BallMatrix(mA); Bb=BallMatrix(mB)              # zero-radius balls
    rZ=zeros(M,K); rZ2=zeros(K,N)
    t_cpu = ct(()->BallArithmetic.MMul4(Ab,Bb))
    t_gpu = gt(()->gemmul8_mmul4(mA,rZ,mB,rZ2))
    # nonzero-radius GPU cost
    rA=abs.(randn(rng,M,K)).*1e-10; rB=abs.(randn(rng,K,N)).*1e-10
    t_gpu_r = gt(()->gemmul8_mmul4(mA,rA,mB,rB))
    @printf("  %d^3:  CPU MMul4 %8.2f ms | GPU MMul4 (port) zero-rad %7.2f ms (%.1fx) | nonzero-rad %7.2f ms (%.1fx)\n",
        M, t_cpu*1e3, t_gpu*1e3, t_cpu/t_gpu, t_gpu_r*1e3, t_cpu/t_gpu_r)
end

println("Device: ", CUDACore.name(CUDACore.device()))
println("\n== full MMul4 enclosure check ==")
for rl in (0.0, 1e-10, 1e-5); verify(256,256,256; radlevel=rl); end
println("\n== full MMul4: GPU (port) vs CPU ==")
for s in (1024,2048,4096); bench(s,s,s); end
