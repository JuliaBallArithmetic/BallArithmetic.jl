# Where does INT8-Ozaki time actually go, and can the GEMMs run concurrently?
#
# Breakdown at a fixed size:
#   (a) split cost (FP64 -> INT8 slices)
#   (b) 43 INT8 GEMMs only, sequential (one stream)
#   (c) 43 INT8 GEMMs only, spread over N streams (test concurrency)
#   (d) full pipeline: GEMM + FP64 recombination interleaved (current probe)
#   (e) FP64 recombination only

using CUDACore, CUDA
using Random, Printf

const gemmI8! = CUDA.CUBLAS.gemmEx!
const B_BITS = 7; const BASE = 2.0^B_BITS

scale_exp(A; dims) = (mx = maximum(abs.(A); dims=dims);
    ifelse.(mx .== 0, 0.0, ceil.(log2.(max.(mx, floatmin(Float64)))) .+ 1.0))
function slices(Ahat, s)
    D = Vector{CuArray{Int8,2}}(undef, s); t = copy(Ahat)
    for i in 1:s; t .*= BASE; di = round.(t); D[i] = Int8.(di); t .-= di; end
    return D
end

function timeit(f; nwarm=3, nrep=8)
    for _ in 1:nwarm; f(); CUDACore.synchronize(); end
    ts = Float64[]
    for _ in 1:nrep
        t0 = time_ns(); f(); CUDACore.synchronize(); push!(ts, (time_ns()-t0)/1e9)
    end
    return sort(ts)[cld(length(ts),2)]
end

function run(M, N, K; s=8, T=10)
    rng = MersenneTwister(1)
    dA = CuArray(randn(rng, Float64, M, K)); dB = CuArray(randn(rng, Float64, K, N))
    σA = scale_exp(dA; dims=2); σB = scale_exp(dB; dims=1)
    DA = slices(dA .* (2.0 .^ (.-σA)), s); DB = slices(dB .* (2.0 .^ (.-σB)), s)
    pairs = [(i,j) for i in 1:s for j in 1:s if i+j <= T]
    ng = length(pairs)

    # Pre-allocated Int32 output buffers (one per pair, for the concurrent case)
    Couts = [CUDA.zeros(Int32, M, N) for _ in 1:ng]
    Cij   = CUDA.zeros(Int32, M, N)
    S     = CUDA.zeros(Float64, M, N)

    t_split = timeit(() -> (slices(dA .* (2.0 .^ (.-σA)), s); slices(dB .* (2.0 .^ (.-σB)), s)))

    # (b) GEMMs only, sequential, default stream
    t_seq = timeit(function ()
        for (p,(i,j)) in enumerate(pairs)
            gemmI8!('N','N', Int32(1), DA[i], DB[j], Int32(0), Couts[p])
        end
    end)

    # (c) GEMMs only, spread over nstreams
    function gemms_streams(nstreams)
        streams = [CuStream() for _ in 1:nstreams]
        for (p,(i,j)) in enumerate(pairs)
            CUDA.stream!(streams[(p % nstreams) + 1]) do
                gemmI8!('N','N', Int32(1), DA[i], DB[j], Int32(0), Couts[p])
            end
        end
        foreach(CUDA.synchronize, streams)
    end
    t_str4 = timeit(() -> gemms_streams(4))
    t_str8 = timeit(() -> gemms_streams(8))

    # (d) full pipeline (current probe): gemm + FP64 accumulate interleaved
    t_full = timeit(function ()
        fill!(S, 0.0)
        for (i,j) in pairs
            fill!(Cij, Int32(0)); gemmI8!('N','N', Int32(1), DA[i], DB[j], Int32(0), Cij)
            S .+= Float64.(Cij) .* (2.0^(-B_BITS*(i+j)))
        end
    end)

    # (e) recombination only (GEMM results precomputed)
    t_rec = timeit(function ()
        fill!(S, 0.0)
        for (p,(i,j)) in enumerate(pairs)
            S .+= Float64.(Couts[p]) .* (2.0^(-B_BITS*(i+j)))
        end
    end)

    @printf("  M=%d N=%d K=%d   %d gemms\n", M, N, K, ng)
    @printf("    split (FP64->INT8)        %7.2f ms\n", t_split*1e3)
    @printf("    %d GEMMs, sequential       %7.2f ms\n", ng, t_seq*1e3)
    @printf("    %d GEMMs, 4 streams        %7.2f ms\n", ng, t_str4*1e3)
    @printf("    %d GEMMs, 8 streams        %7.2f ms\n", ng, t_str8*1e3)
    @printf("    FP64 recombination only   %7.2f ms\n", t_rec*1e3)
    @printf("    full (gemm+recomb interl) %7.2f ms\n", t_full*1e3)
    println()
end

println("Device: ", CUDACore.name(CUDACore.device()), "\n")
run(1024, 1024, 1024)
run(2048, 2048, 2048)
run(4096, 4096, 4096)
