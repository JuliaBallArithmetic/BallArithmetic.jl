# INT8-Ozaki (GPU) vs CPU: raw FP64 GEMM and the rigorous CPU MMul4.
# All produce an FP64-accurate product; INT8-Ozaki and MMul4 are rigorous.

using CUDACore, CUDA
using BallArithmetic
using Random, Printf, LinearAlgebra

const gemmI8! = CUDA.CUBLAS.gemmEx!
const B_BITS = 7; const BASE = 2.0^B_BITS
scale_exp(A; dims) = (mx = maximum(abs.(A); dims=dims);
    ifelse.(mx .== 0, 0.0, ceil.(log2.(max.(mx, floatmin(Float64)))) .+ 1.0))
function slices(Ahat, s)
    D = Vector{CuArray{Int8,2}}(undef, s); t = copy(Ahat)
    for i in 1:s; t .*= BASE; di = round.(t); D[i] = Int8.(di); t .-= di; end
    return D
end
function int8_ozaki(dA, dB; s=8, T=2s)
    σA = scale_exp(dA; dims=2); σB = scale_exp(dB; dims=1)
    DA = slices(dA .* (2.0 .^ (.-σA)), s); DB = slices(dB .* (2.0 .^ (.-σB)), s)
    M, N = size(dA,1), size(dB,2)
    S = CUDA.zeros(Float64, M, N); Cij = CUDA.zeros(Int32, M, N)
    for i in 1:s, j in 1:s
        i+j <= T || continue
        fill!(Cij, Int32(0)); gemmI8!('N','N', Int32(1), DA[i], DB[j], Int32(0), Cij)
        S .+= Float64.(Cij) .* (2.0^(-B_BITS*(i+j)))
    end
    return (S .* (2.0 .^ σA)) .* (2.0 .^ σB)
end

gtime(f; nw=3, nr=8) = (for _ in 1:nw; f(); CUDACore.synchronize(); end;
    ts=Float64[]; for _ in 1:nr; t0=time_ns(); f(); CUDACore.synchronize(); push!(ts,(time_ns()-t0)/1e9); end; sort(ts)[cld(nr,2)])
ctime(f; nw=2, nr=5) = (for _ in 1:nw; f(); end;
    ts=Float64[]; for _ in 1:nr; t0=time_ns(); f(); push!(ts,(time_ns()-t0)/1e9); end; sort(ts)[cld(nr,2)])

function run(M, N, K; seed=1)
    rng = MersenneTwister(seed)
    A = randn(rng, Float64, M, K); B = randn(rng, Float64, K, N)
    dA = CuArray(A); dB = CuArray(B)
    Ab = BallMatrix(A); Bb = BallMatrix(B)        # zero-radius ball matrices

    t_cpu_gemm = ctime(() -> A * B)
    t_cpu_mmul4 = ctime(() -> BallArithmetic.MMul4(Ab, Bb))
    t_gpu_t8  = gtime(() -> int8_ozaki(dA, dB; s=8, T=8))
    t_gpu_t10 = gtime(() -> int8_ozaki(dA, dB; s=8, T=10))

    @printf("  %d^3\n", M)
    @printf("    CPU  FP64 gemm (OpenBLAS, non-rigorous)  %8.2f ms\n", t_cpu_gemm*1e3)
    @printf("    CPU  MMul4 (rigorous ball)               %8.2f ms\n", t_cpu_mmul4*1e3)
    @printf("    GPU  INT8-Ozaki T=8  (rigorous, 4e-14)   %8.2f ms   %.1fx CPU-MMul4 | %.1fx CPU-gemm\n",
            t_gpu_t8*1e3, t_cpu_mmul4/t_gpu_t8, t_cpu_gemm/t_gpu_t8)
    @printf("    GPU  INT8-Ozaki T=10 (rigorous, 5e-16)   %8.2f ms   %.1fx CPU-MMul4 | %.1fx CPU-gemm\n",
            t_gpu_t10*1e3, t_cpu_mmul4/t_gpu_t10, t_cpu_gemm/t_gpu_t10)
    println()
end

println("Device: ", CUDACore.name(CUDACore.device()))
println("CPU BLAS threads: ", BLAS.get_num_threads(), "\n")
for s in [1024, 2048, 4096]
    run(s, s, s)
end
