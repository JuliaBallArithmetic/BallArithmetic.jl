# Benchmark the FULLY ON-GPU Scheme II (14 INT8 GEMMs + GPU Garner + dd kernel)
# against Scheme I (43 INT8 GEMMs) and native FP64. Does the GEMM-count win now
# translate to wall-clock, with reconstruction on the GPU?

using CUDACore, CUDA
using Random, Printf, LinearAlgebra
include("probe_schemeII_gpu.jl")   # reuse schemeII_gpu + helpers

const BB = 7; const BASEv = 2.0^BB
function slices_(Ahat, s)
    D = Vector{CuArray{Int8,2}}(undef, s); t = copy(Ahat)
    for i in 1:s; t .*= BASEv; di = round.(t); D[i] = Int8.(di); t .-= di; end
    return D
end
function schemeI_(dA, dB; s=8, T=10)
    σA = scale_exp(dA; dims=2); σB = scale_exp(dB; dims=1)
    DA = slices_(dA .* (2.0 .^ (.-σA)), s); DB = slices_(dB .* (2.0 .^ (.-σB)), s)
    M_, N_ = size(dA,1), size(dB,2); S = CUDA.zeros(Float64,M_,N_); Cij = CUDA.zeros(Int32,M_,N_)
    for i in 1:s, j in 1:s
        i+j <= T || continue
        fill!(Cij, Int32(0)); gemmI8!('N','N', Int32(1), DA[i], DB[j], Int32(0), Cij)
        S .+= Float64.(Cij) .* (2.0^(-BB*(i+j)))
    end
    return (S .* (2.0 .^ σA)) .* (2.0 .^ σB)
end
gtime(f;nw=2,nr=6)=(for _ in 1:nw;f();CUDACore.synchronize();end;ts=Float64[];for _ in 1:nr;t0=time_ns();f();CUDACore.synchronize();push!(ts,(time_ns()-t0)/1e9);end;sort(ts)[cld(nr,2)])

function bench(M, N, K)
    rng = MersenneTwister(1)
    A = randn(rng, Float64, M, K); B = randn(rng, Float64, K, N)
    dA = CuArray(A); dB = CuArray(B)
    t_fp64 = gtime(() -> dA * dB)
    t_I    = gtime(() -> schemeI_(dA, dB; s=8, T=10))
    t_II   = gtime(() -> schemeII_gpu(A, B; s=14))
    @printf("  %d^3:  native FP64 %7.2f ms | Scheme I (43) %7.2f ms | Scheme II GPU (14) %7.2f ms  (%.2fx Scheme I, %.2fx FP64)\n",
        M, t_fp64*1e3, t_I*1e3, t_II*1e3, t_I/t_II, t_fp64/t_II)
end

println("\n== Scheme II GPU benchmark ==")
for s in [1024, 2048, 4096]; bench(s, s, s); end
