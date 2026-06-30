# Benchmark: rigorous MMul4 (ball-arithmetic GEMM), CPU Float64 vs GPU Float64.
#
# Both paths run the SAME algorithm (directed-rounding outer-product
# accumulation): CPU via SetRounding/OpenBLASConsistentFPCSR, GPU via the
# CuTileExt kernel (@fpmode outer-product). We time only the matmul, with a
# warmup to exclude compilation, and verify the GPU result still overlaps the
# CPU result before reporting timings.

using Random, Printf, LinearAlgebra
using CUDACore
using BallArithmetic
using cuTile  # triggers CuTileExt

const have_ext = Base.get_extension(BallArithmetic, :CuTileExt) !== nothing
@assert have_ext "CuTileExt failed to load"

# Median wall time (seconds) of `f()` over `nrep` runs after `nwarm` warmups.
function timeit(f; nwarm=2, nrep=5)
    for _ in 1:nwarm
        f()
    end
    ts = Float64[]
    for _ in 1:nrep
        t0 = time_ns()
        f()
        push!(ts, (time_ns() - t0) / 1e9)
    end
    return sort(ts)[cld(length(ts), 2)]
end

function make_ball(::Type{T}, M, K) where {T}
    rng = MersenneTwister(7)
    m = rand(rng, T, M, K) .- T(0.5)
    r = rand(rng, T, M, K) .* T(1e-3)
    return m, r
end

function bench(::Type{T}, M, N, K) where {T}
    mA, rA = make_ball(T, M, K)
    mB, rB = make_ball(T, K, N)

    A_cpu = BallMatrix(mA, rA)
    B_cpu = BallMatrix(mB, rB)
    A_gpu = BallMatrix(CuArray(mA), CuArray(rA))
    B_gpu = BallMatrix(CuArray(mB), CuArray(rB))

    # Correctness check (overlap) before timing.
    C_cpu = BallArithmetic.MMul4(A_cpu, B_cpu)
    C_gpu = BallArithmetic.MMul4(A_gpu, B_gpu)
    CUDACore.synchronize()
    mC_c, rC_c = mid(C_cpu), rad(C_cpu)
    mC_g, rC_g = Array(mid(C_gpu)), Array(rad(C_gpu))
    overlap = all(max.(mC_g .- rC_g, mC_c .- rC_c) .<=
                  min.(mC_g .+ rC_g, mC_c .+ rC_c))

    t_cpu = timeit() do
        BallArithmetic.MMul4(A_cpu, B_cpu)
    end
    t_gpu = timeit() do
        C = BallArithmetic.MMul4(A_gpu, B_gpu)
        CUDACore.synchronize()
        C
    end

    # 2 GEMMs worth of flops in MMul4's dominant cost (mid + radius accum).
    gflop = 2.0 * 2.0 * M * N * K / 1e9
    @printf("  %dx%dx%d  CPU %8.2f ms  GPU %8.2f ms  speedup %5.2fx  | CPU %6.1f GF/s  GPU %7.1f GF/s  | overlap=%s\n",
        M, N, K, t_cpu*1e3, t_gpu*1e3, t_cpu/t_gpu,
        gflop/t_cpu, gflop/t_gpu, overlap)
end

println("Device: ", CUDACore.name(CUDACore.device()))
println("BLAS threads: ", BLAS.get_num_threads())
println("\nFloat64 rigorous MMul4 — CPU vs GPU")
println("="^96)
for (M, N, K) in [(128,128,128), (256,256,256), (512,512,512),
                  (1024,1024,1024), (2048,2048,2048)]
    bench(Float64, M, N, K)
end
