# Smoke test: load BallArithmetic + cuTile together, exercise the
# extension via the high-level BallMatrix * BallMatrix interface, and
# verify the result overlaps the CPU MMul4 result.

using Random, Printf, LinearAlgebra
using CUDACore
using BallArithmetic
using cuTile  # triggers CuTileExt load

const have_ext = Base.get_extension(BallArithmetic, :CuTileExt) !== nothing
@info "Extension loaded?" have_ext

function compare(::Type{T}, M, N, K) where {T}
    rng = MersenneTwister(7)
    mA = rand(rng, T, M, K) .- T(0.5)
    rA = rand(rng, T, M, K) .* T(1e-3)
    mB = rand(rng, T, K, N) .- T(0.5)
    rB = rand(rng, T, K, N) .* T(1e-3)

    A_cpu = BallMatrix(mA, rA)
    B_cpu = BallMatrix(mB, rB)
    C_cpu = A_cpu * B_cpu

    A_gpu = BallMatrix(CuArray(mA), CuArray(rA))
    B_gpu = BallMatrix(CuArray(mB), CuArray(rB))
    C_gpu = A_gpu * B_gpu

    @assert mid(C_gpu) isa CuArray "GPU dispatch did not happen — mid is $(typeof(mid(C_gpu)))"

    mC_c, rC_c = mid(C_cpu), rad(C_cpu)
    mC_g, rC_g = Array(mid(C_gpu)), Array(rad(C_gpu))

    cpu_lo, cpu_hi = mC_c .- rC_c, mC_c .+ rC_c
    gpu_lo, gpu_hi = mC_g .- rC_g, mC_g .+ rC_g
    overlap_ok = all(max.(gpu_lo, cpu_lo) .<= min.(gpu_hi, cpu_hi))

    mid_diff = maximum(abs.(mC_g .- mC_c))
    rad_diff = maximum(abs.(rC_g .- rC_c))

    println(@sprintf("  %s M=%d N=%d K=%d: overlap=%s  |Δmid|=%g  |Δrad|=%g",
        T, M, N, K, overlap_ok, mid_diff, rad_diff))
    return overlap_ok
end

function compare_complex(::Type{T}, M, N, K) where {T}
    rng = MersenneTwister(11)
    mA = (rand(rng, T, M, K) .- T(0.5)) .+ im .* (rand(rng, T, M, K) .- T(0.5))
    rA = rand(rng, T, M, K) .* T(1e-3)
    mB = (rand(rng, T, K, N) .- T(0.5)) .+ im .* (rand(rng, T, K, N) .- T(0.5))
    rB = rand(rng, T, K, N) .* T(1e-3)

    A_cpu = BallMatrix(mA, rA)
    B_cpu = BallMatrix(mB, rB)
    C_cpu = A_cpu * B_cpu

    A_gpu = BallMatrix(CuArray(mA), CuArray(rA))
    B_gpu = BallMatrix(CuArray(mB), CuArray(rB))
    C_gpu = A_gpu * B_gpu

    @assert mid(C_gpu) isa CuArray

    mC_c, rC_c = mid(C_cpu), rad(C_cpu)
    mC_g, rC_g = Array(mid(C_gpu)), Array(rad(C_gpu))

    overlap_ok = all(abs.(mC_g .- mC_c) .<= rC_c .+ rC_g)
    println(@sprintf("  Complex{%s} M=%d N=%d K=%d: overlap=%s  |Δmid|=%g  |Δrad|=%g",
        T, M, N, K, overlap_ok,
        maximum(abs.(mC_g .- mC_c)),
        maximum(abs.(rC_g .- rC_c))))
    return overlap_ok
end

println("Device: ", CUDACore.name(CUDACore.device()))
println()
results = Bool[]
for sz in [(16, 16, 64), (64, 64, 256), (128, 96, 320)]
    push!(results, compare(Float32, sz...))
    push!(results, compare(Float64, sz...))
end
for sz in [(16, 16, 64), (64, 64, 256)]
    # NB: complex Float32 BallMatrix * BallMatrix has a pre-existing bug on
    # CPU (im * BallMatrix{Float32,Float32} constructs a mismatched BallMatrix
    # type). Skip Float32 here — the GPU kernel itself handles complex via
    # the same dispatch chain.
    push!(results, compare_complex(Float64, sz...))
end
all_ok = all(results)

if all_ok
    println("\n✓ All cases overlapped CPU MMul4")
    exit(0)
else
    println("\n✗ At least one case did not overlap")
    exit(1)
end
