# Radius matmul on INT8 tensor cores: a rigorous UPPER bound on U·V (U,V >= 0)
# via ONE exact INT8 GEMM with ceil-rounded slices, vs the FP32 approach.
#
# Scale per row/col so Ũ,Ṽ <= 1/2; U8 = ceil(Ũ·2^7) (so U8·2^(σU-7) >= U); the
# INT8 GEMM is exact (INT32), and Σ_k (U8·2^(σU-7))(V8·2^(σV-7)) >= Σ_k U·V.
# We only need an upper bound (looser radius = slightly fatter ball), so the
# 7-bit truncation is fine — and INT8 tensor cores beat FP32 ALUs ~3x.

using CUDACore, CUDA, Random, Printf, LinearAlgebra
const gemmI8! = CUDA.CUBLAS.gemmEx!

# rigorous elementwise upper bound on U*V, U (M×K)>=0, V (K×N)>=0, via 1 INT8 gemm
function int8_upper_product(dU, dV)
    M_, K_ = size(dU); N_ = size(dV, 2)
    mxU = maximum(dU; dims=2); mxV = maximum(dV; dims=1)
    σU = ifelse.(mxU .== 0, 0.0, ceil.(log2.(max.(mxU, floatmin(Float64)))) .+ 1.0)
    σV = ifelse.(mxV .== 0, 0.0, ceil.(log2.(max.(mxV, floatmin(Float64)))) .+ 1.0)
    U8 = Int8.(ceil.(dU .* (2.0 .^ (7 .- σU))))    # in [0,64]
    V8 = Int8.(ceil.(dV .* (2.0 .^ (7 .- σV))))
    C  = CUDA.zeros(Int32, M_, N_)
    gemmI8!('N','N', Int32(1), U8, V8, Int32(0), C)
    return (Float64.(C) .* (2.0 .^ (σU .- 7)) .* (2.0 .^ (σV .- 7))) .* (1 + 4eps())
end

# FP32 reference upper bound (round-to-nearest + inflate)
function fp32_upper_product(dU, dV)
    K_ = size(dU, 2)
    (Float64.(Float32.(dU) * Float32.(dV))) .* (1 + Float64(2K_+4) * Float64(eps(Float32)))
end

gt(f;nw=2,nr=6)=(for _ in 1:nw;f();CUDACore.synchronize();end;ts=Float64[];for _ in 1:nr;t0=time_ns();f();CUDACore.synchronize();push!(ts,(time_ns()-t0)/1e9);end;sort(ts)[cld(nr,2)])

function run(M, N, K; seed=1)
    rng = MersenneTwister(seed)
    U = abs.(randn(rng, M, K)); V = abs.(randn(rng, K, N))   # nonneg
    dU = CuArray(U); dV = CuArray(V)
    b8 = Array(int8_upper_product(dU, dV)); CUDACore.synchronize()
    bf = Array(fp32_upper_product(dU, dV))
    setprecision(BigFloat, 200) do
        true_p = BigFloat.(U) * BigFloat.(V)
        ok8 = all(BigFloat.(b8) .>= true_p)
        okf = all(BigFloat.(bf) .>= true_p)
        loose8 = Float64(maximum(BigFloat.(b8) ./ true_p))
        loosef = Float64(maximum(BigFloat.(bf) ./ true_p))
        t8 = gt(() -> int8_upper_product(dU, dV))
        tf = gt(() -> fp32_upper_product(dU, dV))
        @printf("  %d^3: INT8 upper-bound=%s (max overest %.3fx, %.2f ms) | FP32=%s (%.4fx, %.2f ms)  speedup %.1fx\n",
            M, ok8 ? "valid" : "INVALID", loose8, t8*1e3, okf ? "valid" : "INVALID", loosef, tf*1e3, tf/t8)
        flush(stdout)
    end
end

# speed only (no BigFloat), at larger sizes
function speed(M, N, K)
    rng = MersenneTwister(1); dU = CuArray(abs.(randn(rng,M,K))); dV = CuArray(abs.(randn(rng,K,N)))
    t8 = gt(() -> int8_upper_product(dU, dV)); tf = gt(() -> fp32_upper_product(dU, dV))
    @printf("  %d^3 speed: INT8 %.2f ms | FP32 %.2f ms  -> %.1fx faster\n", M, t8*1e3, tf*1e3, tf/t8); flush(stdout)
end

println("Device: ", CUDACore.name(CUDACore.device()), "\n")
println("-- rigor + looseness (BigFloat-checked) --")
for s in (256, 512); run(s, s, s); end
println("-- speed (INT8 tensor cores vs FP32 ALU) --")
for s in (1024, 2048, 4096); speed(s, s, s); end
