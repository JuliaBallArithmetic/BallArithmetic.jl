# Benchmark: INT8-Ozaki FP64-emulation vs native FP64 on the GPU, and vs CPU.
# Question (from arXiv:2504.08009): does exact-INT8-tensor-core emulation beat
# native FP64 on consumer Ada, as the paper reports ~6-7x on the RTX 4090?

using CUDACore, CUDA
using Random, Printf, LinearAlgebra

const gemmI8! = CUDA.CUBLAS.gemmEx!
const B_BITS  = 7
const BASE    = 2.0^B_BITS

function scale_exp(A; dims)
    mx = maximum(abs.(A); dims=dims)
    σ  = ceil.(log2.(max.(mx, floatmin(Float64)))) .+ 1.0
    return ifelse.(mx .== 0, 0.0, σ)
end
function slices(Ahat, s)
    D = Vector{CuArray{Int8,2}}(undef, s)
    t = copy(Ahat)
    for i in 1:s
        t .*= BASE; di = round.(t); D[i] = Int8.(di); t .-= di
    end
    return D
end
function int8_ozaki(dA, dB; s=8, T=2s)
    σA = scale_exp(dA; dims=2); σB = scale_exp(dB; dims=1)
    DA = slices(dA .* (2.0 .^ (.-σA)), s)
    DB = slices(dB .* (2.0 .^ (.-σB)), s)
    M, N = size(dA, 1), size(dB, 2)
    S = CUDA.zeros(Float64, M, N); Cij = CUDA.zeros(Int32, M, N)
    for i in 1:s, j in 1:s
        i + j <= T || continue
        fill!(Cij, Int32(0))
        gemmI8!('N','N', Int32(1), DA[i], DB[j], Int32(0), Cij)
        S .+= Float64.(Cij) .* (2.0^(-B_BITS*(i+j)))
    end
    return (S .* (2.0 .^ σA)) .* (2.0 .^ σB)
end

function timeit(f; nwarm=3, nrep=8)
    for _ in 1:nwarm; f(); CUDACore.synchronize(); end
    ts = Float64[]
    for _ in 1:nrep
        t0 = time_ns(); f(); CUDACore.synchronize(); push!(ts, (time_ns()-t0)/1e9)
    end
    return sort(ts)[cld(length(ts),2)]
end

function run(M, N, K; seed=1)
    rng = MersenneTwister(seed)
    A = randn(rng, Float64, M, K); B = randn(rng, Float64, K, N)
    dA = CuArray(A); dB = CuArray(B)
    dAi = CuArray(rand(rng, Int8.(-64:64), M, K)); dBi = CuArray(rand(rng, Int8.(-64:64), K, N))
    dCi = CUDA.zeros(Int32, M, N)
    gf = 2.0 * M * N * K / 1e9

    t_fp64 = timeit(() -> dA * dB)
    t_i8   = timeit(() -> gemmI8!('N','N', Int32(1), dAi, dBi, Int32(0), dCi))
    t_oz28 = timeit(() -> int8_ozaki(dA, dB; s=8, T=8))
    t_oz43 = timeit(() -> int8_ozaki(dA, dB; s=8, T=10))

    @printf("  M=%d N=%d K=%d\n", M, N, K)
    @printf("    native FP64 gemm        %8.3f ms  (%6.1f GF/s)\n", t_fp64*1e3, gf/t_fp64)
    @printf("    single INT8 gemm        %8.3f ms  (%6.1f GF/s)  -> INT8 is %.1fx FP64/gemm\n",
            t_i8*1e3, gf/t_i8, t_fp64/t_i8)
    @printf("    INT8-Ozaki s8 T8  (28)  %8.3f ms  -> %.2fx native FP64\n", t_oz28*1e3, t_fp64/t_oz28)
    @printf("    INT8-Ozaki s8 T10 (43)  %8.3f ms  -> %.2fx native FP64\n", t_oz43*1e3, t_fp64/t_oz43)
    println()
end

println("Device: ", CUDACore.name(CUDACore.device()), "\n")
for (M,N,K) in [(512,512,512), (1024,1024,1024), (2048,2048,2048), (4096,4096,4096)]
    run(M, N, K)
end
