# # Circle Certification Benchmark: Precision Comparison
#
# This benchmark compares different precision levels for certifying the
# resolvent norm around a circle in the complex plane.
#
# Methods compared:
# - Float64 (Miyajima): Direct ball arithmetic SVD certification
# - Double64: Extended precision (~106 bits) with BigFloat certification
# - MultiFloat: Similar to Double64, using MultiFloats.jl
# - BigFloat: Full arbitrary precision (slowest, reference accuracy)
#
# Note on warm-start: Ogita refinement assumes the initial SVD is accurate
# for THE SAME matrix. Using SVD from (A - z₁I) to refine (A - z₂I) fails
# because the initial error is O(1), not O(ε_machine). Warm-start is only
# meaningful for refining the same matrix with more iterations.

include("benchmark_common.jl")
include("benchmark_spectral_gap.jl")

# ## Circle Certification with Warm-Start

struct CircleCertificationResult
    num_points::Int
    times_cold::Vector{Float64}
    times_warm::Vector{Float64}
    bounds_cold::Vector{Float64}
    bounds_warm::Vector{Float64}
    iterations_cold::Vector{Int}
    iterations_warm::Vector{Int}
    total_cold::Float64
    total_warm::Float64
    speedup::Float64
end

"""Extended result including all precision comparisons."""
struct ExtendedCircleResult
    bf::CircleCertificationResult       # BigFloat results
    # Double64 results
    d64_cold::Union{Nothing, Tuple{Vector{Float64}, Vector{Float64}, Vector{Int}}}
    d64_warm::Union{Nothing, Tuple{Vector{Float64}, Vector{Float64}, Vector{Int}}}
    d64_total_cold::Float64
    d64_total_warm::Float64
    d64_speedup::Float64
    bf_vs_d64_speedup::Float64          # How much faster is D64 than BigFloat?
    # MultiFloat results
    mf_cold::Union{Nothing, Tuple{Vector{Float64}, Vector{Float64}, Vector{Int}}}
    mf_warm::Union{Nothing, Tuple{Vector{Float64}, Vector{Float64}, Vector{Int}}}
    mf_total_cold::Float64
    mf_total_warm::Float64
    mf_speedup::Float64
    # Float64-only results (no refinement - direct Miyajima)
    f64_times::Union{Nothing, Vector{Float64}}
    f64_bounds::Union{Nothing, Vector{Float64}}
    f64_total::Float64
end

"""
    certify_circle_cold(A, center, radius, num_points; use_optimal_iterations=false)

Certify resolvent at `num_points` around circle, recomputing SVD from scratch each time.
"""
function certify_circle_cold(T_mid::Matrix, center::Complex, radius::Real, num_points::Int;
                             use_optimal_iterations::Bool=false)
    θs = range(0, 2π, length=num_points+1)[1:end-1]
    zs = [center + radius * exp(im * θ) for θ in θs]

    times = Float64[]
    bounds = Float64[]
    iterations = Int[]

    for z in zs
        t0 = time()
        Ashift = T_mid - z * I
        U, S, V = svd(Ashift)
        result = ogita_svd_refine(Ashift, U, S, V;
                                  max_iterations=10,
                                  check_convergence=!use_optimal_iterations,
                                  use_optimal_iterations=use_optimal_iterations)
        push!(times, time() - t0)

        if result.converged
            σ_min = result.Σ[end, end] - result.residual_norm
            push!(bounds, σ_min > 0 ? 1.0 / Float64(σ_min) : Inf)
        else
            push!(bounds, Inf)
        end
        push!(iterations, result.iterations)
    end

    return times, bounds, iterations
end

"""
    certify_circle_warm(A, center, radius, num_points; use_optimal_iterations=false)

Certify resolvent at `num_points` around circle, using warm-start from previous point.
"""
function certify_circle_warm(T_mid::Matrix, center::Complex, radius::Real, num_points::Int;
                             use_optimal_iterations::Bool=false)
    θs = range(0, 2π, length=num_points+1)[1:end-1]
    zs = [center + radius * exp(im * θ) for θ in θs]

    times = Float64[]
    bounds = Float64[]
    iterations = Int[]

    # First point: cold start
    z = zs[1]
    t0 = time()
    Ashift = T_mid - z * I
    U, S, V = svd(Ashift)
    result = ogita_svd_refine(Ashift, U, S, V;
                              max_iterations=10,
                              check_convergence=!use_optimal_iterations,
                              use_optimal_iterations=use_optimal_iterations)
    push!(times, time() - t0)

    if result.converged
        σ_min = result.Σ[end, end] - result.residual_norm
        push!(bounds, σ_min > 0 ? 1.0 / Float64(σ_min) : Inf)
        # Extract refined SVD for warm-start (convert back to Float64!)
        U_warm = convert.(ComplexF64, result.U)
        V_warm = convert.(ComplexF64, result.V)
        S_warm = convert.(Float64, diag(result.Σ))
    else
        push!(bounds, Inf)
        U_warm, S_warm, V_warm = U, S, V
    end
    push!(iterations, result.iterations)

    # Remaining points: warm start from previous
    for i in 2:length(zs)
        z = zs[i]
        t0 = time()
        Ashift = T_mid - z * I

        # Use previous SVD as starting point
        result = ogita_svd_refine(Ashift, U_warm, S_warm, V_warm;
                                  max_iterations=10,
                                  check_convergence=!use_optimal_iterations,
                                  use_optimal_iterations=use_optimal_iterations)
        push!(times, time() - t0)

        if result.converged
            σ_min = result.Σ[end, end] - result.residual_norm
            push!(bounds, σ_min > 0 ? 1.0 / Float64(σ_min) : Inf)
            # Update warm-start for next point (convert back to Float64!)
            U_warm = convert.(ComplexF64, result.U)
            V_warm = convert.(ComplexF64, result.V)
            S_warm = convert.(Float64, diag(result.Σ))
        else
            push!(bounds, Inf)
            # Fall back to cold SVD for next warm-start
            U_new, S_new, V_new = svd(Ashift)
            U_warm, S_warm, V_warm = U_new, S_new, V_new
        end
        push!(iterations, result.iterations)
    end

    return times, bounds, iterations
end

# ## Double64 Circle Certification (Fast!)
#
# These use ogita_svd_refine_fast from DoubleFloatsExt for ~10-30x speedup.

"""
    certify_circle_cold_d64(T_mid, center, radius, num_points)

Cold-start circle certification using Double64 refinement.
Much faster than BigFloat for the bulk of iterations.
"""
function certify_circle_cold_d64(T_mid::Matrix, center::Complex, radius::Real, num_points::Int)
    if !HAS_DOUBLEFLOATS
        error("DoubleFloats.jl not available")
    end

    θs = range(0, 2π, length=num_points+1)[1:end-1]
    zs = [center + radius * exp(im * θ) for θ in θs]

    times = Float64[]
    bounds = Float64[]
    iterations = Int[]

    for z in zs
        t0 = time()
        Ashift = T_mid - z * I
        U, S, V = svd(Ashift)
        result = ogita_svd_refine_fast(Ashift, U, S, V;
                                       max_iterations=2,
                                       certify_with_bigfloat=true,
                                       bigfloat_precision=256)
        push!(times, time() - t0)

        if result.converged
            σ_min = result.Σ[end, end] - result.residual_norm
            push!(bounds, σ_min > 0 ? 1.0 / Float64(σ_min) : Inf)
        else
            push!(bounds, Inf)
        end
        push!(iterations, result.iterations)
    end

    return times, bounds, iterations
end

"""
    certify_circle_warm_d64(T_mid, center, radius, num_points)

Warm-start circle certification using Double64 refinement.
"""
function certify_circle_warm_d64(T_mid::Matrix, center::Complex, radius::Real, num_points::Int)
    if !HAS_DOUBLEFLOATS
        error("DoubleFloats.jl not available")
    end

    θs = range(0, 2π, length=num_points+1)[1:end-1]
    zs = [center + radius * exp(im * θ) for θ in θs]

    times = Float64[]
    bounds = Float64[]
    iterations = Int[]

    # First point: cold start
    z = zs[1]
    t0 = time()
    Ashift = T_mid - z * I
    U, S, V = svd(Ashift)
    result = ogita_svd_refine_fast(Ashift, U, S, V;
                                   max_iterations=2,
                                   certify_with_bigfloat=true,
                                   bigfloat_precision=256)
    push!(times, time() - t0)

    if result.converged
        σ_min = result.Σ[end, end] - result.residual_norm
        push!(bounds, σ_min > 0 ? 1.0 / Float64(σ_min) : Inf)
        # Extract for warm-start (convert back to Float64 for next SVD call)
        U_warm = convert.(ComplexF64, result.U)
        V_warm = convert.(ComplexF64, result.V)
        S_warm = convert.(Float64, diag(result.Σ))
    else
        push!(bounds, Inf)
        U_warm, S_warm, V_warm = U, S, V
    end
    push!(iterations, result.iterations)

    # Remaining points: warm start
    for i in 2:length(zs)
        z = zs[i]
        t0 = time()
        Ashift = T_mid - z * I

        # Use previous SVD as starting point
        result = ogita_svd_refine_fast(Ashift, U_warm, S_warm, V_warm;
                                       max_iterations=2,
                                       certify_with_bigfloat=true,
                                       bigfloat_precision=256)
        push!(times, time() - t0)

        if result.converged
            σ_min = result.Σ[end, end] - result.residual_norm
            push!(bounds, σ_min > 0 ? 1.0 / Float64(σ_min) : Inf)
            U_warm = convert.(ComplexF64, result.U)
            V_warm = convert.(ComplexF64, result.V)
            S_warm = convert.(Float64, diag(result.Σ))
        else
            push!(bounds, Inf)
            U_new, S_new, V_new = svd(Ashift)
            U_warm, S_warm, V_warm = U_new, S_new, V_new
        end
        push!(iterations, result.iterations)
    end

    return times, bounds, iterations
end

# ## MultiFloat Circle Certification
#
# These use ogita_svd_refine_multifloat from MultiFloatsExt.

"""
    certify_circle_cold_mf(T_mid, center, radius, num_points; precision=:x2)

Cold-start circle certification using MultiFloat refinement.
"""
function certify_circle_cold_mf(T_mid::Matrix, center::Complex, radius::Real, num_points::Int;
                                precision::Symbol=:x2)
    if !HAS_MULTIFLOATS
        error("MultiFloats.jl not available")
    end

    θs = range(0, 2π, length=num_points+1)[1:end-1]
    zs = [center + radius * exp(im * θ) for θ in θs]

    times = Float64[]
    bounds = Float64[]
    iterations = Int[]

    for z in zs
        t0 = time()
        Ashift = T_mid - z * I
        U, S, V = svd(Ashift)
        result = ogita_svd_refine_multifloat(Ashift, U, S, V;
                                              precision=precision,
                                              max_iterations=2,
                                              certify_with_bigfloat=true,
                                              bigfloat_precision=256)
        push!(times, time() - t0)

        if result.converged
            σ_min = result.Σ[end, end] - result.residual_norm
            push!(bounds, σ_min > 0 ? 1.0 / Float64(σ_min) : Inf)
        else
            push!(bounds, Inf)
        end
        push!(iterations, result.iterations)
    end

    return times, bounds, iterations
end

"""
    certify_circle_warm_mf(T_mid, center, radius, num_points; precision=:x2)

Warm-start circle certification using MultiFloat refinement.
"""
function certify_circle_warm_mf(T_mid::Matrix, center::Complex, radius::Real, num_points::Int;
                                precision::Symbol=:x2)
    if !HAS_MULTIFLOATS
        error("MultiFloats.jl not available")
    end

    θs = range(0, 2π, length=num_points+1)[1:end-1]
    zs = [center + radius * exp(im * θ) for θ in θs]

    times = Float64[]
    bounds = Float64[]
    iterations = Int[]

    # First point: cold start
    z = zs[1]
    t0 = time()
    Ashift = T_mid - z * I
    U, S, V = svd(Ashift)
    result = ogita_svd_refine_multifloat(Ashift, U, S, V;
                                          precision=precision,
                                          max_iterations=2,
                                          certify_with_bigfloat=true,
                                          bigfloat_precision=256)
    push!(times, time() - t0)

    if result.converged
        σ_min = result.Σ[end, end] - result.residual_norm
        push!(bounds, σ_min > 0 ? 1.0 / Float64(σ_min) : Inf)
        # Extract for warm-start
        U_warm = convert.(ComplexF64, result.U)
        V_warm = convert.(ComplexF64, result.V)
        S_warm = convert.(Float64, diag(result.Σ))
    else
        push!(bounds, Inf)
        U_warm, S_warm, V_warm = U, S, V
    end
    push!(iterations, result.iterations)

    # Remaining points: warm start
    for i in 2:length(zs)
        z = zs[i]
        t0 = time()
        Ashift = T_mid - z * I

        result = ogita_svd_refine_multifloat(Ashift, U_warm, S_warm, V_warm;
                                              precision=precision,
                                              max_iterations=2,
                                              certify_with_bigfloat=true,
                                              bigfloat_precision=256)
        push!(times, time() - t0)

        if result.converged
            σ_min = result.Σ[end, end] - result.residual_norm
            push!(bounds, σ_min > 0 ? 1.0 / Float64(σ_min) : Inf)
            U_warm = convert.(ComplexF64, result.U)
            V_warm = convert.(ComplexF64, result.V)
            S_warm = convert.(Float64, diag(result.Σ))
        else
            push!(bounds, Inf)
            U_new, S_new, V_new = svd(Ashift)
            U_warm, S_warm, V_warm = U_new, S_new, V_new
        end
        push!(iterations, result.iterations)
    end

    return times, bounds, iterations
end

# ## Pure Float64 Certification (Miyajima baseline)
#
# This is the fastest but least accurate method - direct SVD certification
# without any extended precision refinement.

"""
    certify_circle_float64(T_mid, center, radius, num_points)

Float64-only circle certification using Miyajima's method.
No warm-start benefit (each point is independent).
"""
function certify_circle_float64(T_mid::Matrix, center::Complex, radius::Real, num_points::Int)
    θs = range(0, 2π, length=num_points+1)[1:end-1]
    zs = [center + radius * exp(im * θ) for θ in θs]

    times = Float64[]
    bounds = Float64[]

    for z in zs
        t0 = time()
        A = BallMatrix(T_mid) - z * I
        result = rigorous_svd(A)
        push!(times, time() - t0)

        # Smallest singular value is last (sorted descending)
        σ_min = result.singular_values[end]
        σ_min_lower = BallArithmetic.inf(σ_min)
        if σ_min_lower > 0
            push!(bounds, 1.0 / Float64(σ_min_lower))
        else
            push!(bounds, Inf)
        end
    end

    return times, bounds
end

"""
    run_circle_warmstart_benchmark(; n, k, gap, nonnormality, center, radius, num_points,
                                    use_optimal_iterations, verbose)

Compare cold vs warm-start circle certification.
"""
function run_circle_warmstart_benchmark(;
        n::Int=50,
        k::Int=5,
        gap::Real=0.5,
        nonnormality::Real=0.1,
        center::Complex=1.0+0.0im,
        radius::Real=0.15,
        num_points::Int=32,
        use_optimal_iterations::Bool=false,
        verbose::Bool=true)

    opt_str = use_optimal_iterations ? " (optimal iterations)" : " (check convergence)"
    verbose && println("\n" * "="^70)
    verbose && println("CIRCLE WARM-START BENCHMARK" * opt_str)
    verbose && println("n=$n, k=$k, gap=$gap, ν=$nonnormality")
    verbose && println("Circle: center=$center, radius=$radius, points=$num_points")
    if use_optimal_iterations
        verbose && println("Using optimal iterations: $(ogita_iterations_for_precision(256)) for 256-bit precision")
    end
    verbose && println("="^70)

    # Create matrix with spectral gap
    T = make_spectral_gap_matrix(n; cluster_size=k, cluster_center=center,
                                  cluster_radius=0.02, gap=gap, nonnormality=nonnormality)
    T_mid = T  # Already Float64

    # Cold certification
    verbose && println("\n--- Cold start (recompute SVD each point) ---")
    times_cold, bounds_cold, iters_cold = certify_circle_cold(T_mid, center, radius, num_points;
                                                               use_optimal_iterations)
    total_cold = sum(times_cold)
    verbose && @printf("  Total time: %.2fs\n", total_cold)
    verbose && @printf("  Mean time per point: %.3fs\n", mean(times_cold))
    verbose && @printf("  Mean iterations: %.1f\n", mean(iters_cold))

    # Warm certification
    verbose && println("\n--- Warm start (reuse SVD from previous point) ---")
    times_warm, bounds_warm, iters_warm = certify_circle_warm(T_mid, center, radius, num_points;
                                                               use_optimal_iterations)
    total_warm = sum(times_warm)
    verbose && @printf("  Total time: %.2fs\n", total_warm)
    verbose && @printf("  Mean time per point: %.3fs\n", mean(times_warm))
    verbose && @printf("  Mean iterations: %.1f\n", mean(iters_warm))

    speedup = total_cold / total_warm
    verbose && println("\n--- Summary ---")
    verbose && @printf("  Speedup: %.2fx\n", speedup)
    verbose && @printf("  Iteration reduction: %.1f → %.1f (%.0f%% fewer)\n",
                       mean(iters_cold), mean(iters_warm),
                       100 * (1 - mean(iters_warm) / mean(iters_cold)))

    # Check bounds are consistent
    bound_diff = maximum(abs.(bounds_cold .- bounds_warm) ./ bounds_cold)
    verbose && @printf("  Max relative bound difference: %.2e\n", bound_diff)

    bf_result = CircleCertificationResult(num_points, times_cold, times_warm,
                                          bounds_cold, bounds_warm, iters_cold, iters_warm,
                                          total_cold, total_warm, speedup)

    # Double64 certification (if available)
    d64_cold_data = nothing
    d64_warm_data = nothing
    d64_total_cold = Inf
    d64_total_warm = Inf
    d64_speedup = 1.0
    bf_vs_d64_speedup = 1.0

    if HAS_DOUBLEFLOATS
        verbose && println("\n" * "-"^50)
        verbose && println("DOUBLE64 CERTIFICATION")
        verbose && println("-"^50)

        # D64 Cold
        verbose && println("\n--- Double64 Cold start ---")
        times_d64_cold, bounds_d64_cold, iters_d64_cold = certify_circle_cold_d64(
            T_mid, center, radius, num_points)
        d64_total_cold = sum(times_d64_cold)
        verbose && @printf("  Total time: %.2fs\n", d64_total_cold)
        verbose && @printf("  Mean time per point: %.3fs\n", mean(times_d64_cold))

        # D64 Warm
        verbose && println("\n--- Double64 Warm start ---")
        times_d64_warm, bounds_d64_warm, iters_d64_warm = certify_circle_warm_d64(
            T_mid, center, radius, num_points)
        d64_total_warm = sum(times_d64_warm)
        verbose && @printf("  Total time: %.2fs\n", d64_total_warm)
        verbose && @printf("  Mean time per point: %.3fs\n", mean(times_d64_warm))

        d64_speedup = d64_total_cold / d64_total_warm
        bf_vs_d64_speedup = total_cold / d64_total_cold  # BigFloat cold vs D64 cold

        verbose && println("\n--- Double64 Summary ---")
        verbose && @printf("  D64 warm-start speedup: %.2fx\n", d64_speedup)
        verbose && @printf("  BigFloat vs Double64 (cold): %.1fx faster\n", bf_vs_d64_speedup)
        verbose && @printf("  BigFloat vs Double64 (warm): %.1fx faster\n", total_warm / d64_total_warm)

        # Verify bounds are consistent with BigFloat
        d64_bf_diff = maximum(abs.(bounds_d64_cold .- bounds_cold) ./ bounds_cold)
        verbose && @printf("  Max D64 vs BF bound difference: %.2e\n", d64_bf_diff)

        d64_cold_data = (times_d64_cold, bounds_d64_cold, iters_d64_cold)
        d64_warm_data = (times_d64_warm, bounds_d64_warm, iters_d64_warm)
    else
        verbose && println("\n[Double64 not available - install DoubleFloats.jl]")
    end

    # MultiFloat certification (if available)
    mf_cold_data = nothing
    mf_warm_data = nothing
    mf_total_cold = Inf
    mf_total_warm = Inf
    mf_speedup = 1.0

    if HAS_MULTIFLOATS
        verbose && println("\n" * "-"^50)
        verbose && println("MULTIFLOAT (x2) CERTIFICATION")
        verbose && println("-"^50)

        # MF Cold
        verbose && println("\n--- MultiFloat Cold start ---")
        times_mf_cold, bounds_mf_cold, iters_mf_cold = certify_circle_cold_mf(
            T_mid, center, radius, num_points; precision=:x2)
        mf_total_cold = sum(times_mf_cold)
        verbose && @printf("  Total time: %.2fs\n", mf_total_cold)
        verbose && @printf("  Mean time per point: %.3fs\n", mean(times_mf_cold))

        # MF Warm
        verbose && println("\n--- MultiFloat Warm start ---")
        times_mf_warm, bounds_mf_warm, iters_mf_warm = certify_circle_warm_mf(
            T_mid, center, radius, num_points; precision=:x2)
        mf_total_warm = sum(times_mf_warm)
        verbose && @printf("  Total time: %.2fs\n", mf_total_warm)
        verbose && @printf("  Mean time per point: %.3fs\n", mean(times_mf_warm))

        mf_speedup = mf_total_cold / mf_total_warm

        verbose && println("\n--- MultiFloat Summary ---")
        verbose && @printf("  MF warm-start speedup: %.2fx\n", mf_speedup)
        verbose && @printf("  BigFloat vs MultiFloat (cold): %.1fx faster\n", total_cold / mf_total_cold)
        verbose && @printf("  BigFloat vs MultiFloat (warm): %.1fx faster\n", total_warm / mf_total_warm)

        # Verify bounds are consistent
        mf_bf_diff = maximum(abs.(bounds_mf_cold .- bounds_cold) ./ bounds_cold)
        verbose && @printf("  Max MF vs BF bound difference: %.2e\n", mf_bf_diff)

        mf_cold_data = (times_mf_cold, bounds_mf_cold, iters_mf_cold)
        mf_warm_data = (times_mf_warm, bounds_mf_warm, iters_mf_warm)
    else
        verbose && println("\n[MultiFloats not available - install MultiFloats.jl]")
    end

    # Pure Float64 certification (Miyajima baseline)
    verbose && println("\n" * "-"^50)
    verbose && println("FLOAT64 CERTIFICATION (Miyajima baseline)")
    verbose && println("-"^50)

    times_f64, bounds_f64 = certify_circle_float64(T_mid, center, radius, num_points)
    f64_total = sum(times_f64)
    verbose && @printf("  Total time: %.2fs\n", f64_total)
    verbose && @printf("  Mean time per point: %.3fs\n", mean(times_f64))

    # Compare with BigFloat
    verbose && println("\n--- Float64 vs BigFloat ---")
    verbose && @printf("  BigFloat cold / Float64: %.1fx slower\n", total_cold / f64_total)
    verbose && @printf("  BigFloat warm / Float64: %.1fx slower\n", total_warm / f64_total)

    # Check bound quality (Float64 may be worse)
    f64_bf_diff = maximum(abs.(bounds_f64 .- bounds_cold) ./ bounds_cold)
    verbose && @printf("  Max F64 vs BF bound difference: %.2e\n", f64_bf_diff)
    if any(bounds_f64 .> 10 .* bounds_cold)
        verbose && println("  WARNING: Float64 bounds significantly worse than BigFloat!")
    end

    # Overall comparison table
    verbose && println("\n" * "="^70)
    verbose && println("OVERALL COMPARISON")
    verbose && println("="^70)
    verbose && println("\nTotal certification time for $num_points points:")
    verbose && @printf("  Float64 (Miyajima):    %7.3fs  (baseline)\n", f64_total)
    verbose && @printf("  BigFloat cold:         %7.3fs  (%.1fx slower)\n", total_cold, total_cold/f64_total)
    verbose && @printf("  BigFloat warm:         %7.3fs  (%.1fx slower)\n", total_warm, total_warm/f64_total)
    if HAS_DOUBLEFLOATS
        verbose && @printf("  Double64 cold:         %7.3fs  (%.1fx slower)\n", d64_total_cold, d64_total_cold/f64_total)
        verbose && @printf("  Double64 warm:         %7.3fs  (%.1fx slower)\n", d64_total_warm, d64_total_warm/f64_total)
    end
    if HAS_MULTIFLOATS
        verbose && @printf("  MultiFloat cold:       %7.3fs  (%.1fx slower)\n", mf_total_cold, mf_total_cold/f64_total)
        verbose && @printf("  MultiFloat warm:       %7.3fs  (%.1fx slower)\n", mf_total_warm, mf_total_warm/f64_total)
    end

    # Bound accuracy comparison (lower = tighter = better)
    verbose && println("\n" * "="^70)
    verbose && println("BOUND ACCURACY COMPARISON (lower = tighter = better)")
    verbose && println("="^70)

    # Use BigFloat cold as reference (most accurate)
    ref_bounds = bounds_cold
    ref_mean = mean(ref_bounds)
    ref_max = maximum(ref_bounds)

    verbose && println("\nMethod                    Mean Bound    Max Bound    Overhead vs BF")
    verbose && println("-"^70)
    verbose && @printf("BigFloat cold (ref)       %.4e    %.4e    1.00x\n", ref_mean, ref_max)
    verbose && @printf("BigFloat warm             %.4e    %.4e    %.2fx\n",
                       mean(bounds_warm), maximum(bounds_warm), mean(bounds_warm)/ref_mean)

    if HAS_DOUBLEFLOATS && d64_cold_data !== nothing
        d64_cold_bounds = d64_cold_data[2]
        d64_warm_bounds = d64_warm_data[2]
        verbose && @printf("Double64 cold             %.4e    %.4e    %.2fx\n",
                           mean(d64_cold_bounds), maximum(d64_cold_bounds), mean(d64_cold_bounds)/ref_mean)
        verbose && @printf("Double64 warm             %.4e    %.4e    %.2fx\n",
                           mean(d64_warm_bounds), maximum(d64_warm_bounds), mean(d64_warm_bounds)/ref_mean)
    end

    if HAS_MULTIFLOATS && mf_cold_data !== nothing
        mf_cold_bounds = mf_cold_data[2]
        mf_warm_bounds = mf_warm_data[2]
        verbose && @printf("MultiFloat cold           %.4e    %.4e    %.2fx\n",
                           mean(mf_cold_bounds), maximum(mf_cold_bounds), mean(mf_cold_bounds)/ref_mean)
        verbose && @printf("MultiFloat warm           %.4e    %.4e    %.2fx\n",
                           mean(mf_warm_bounds), maximum(mf_warm_bounds), mean(mf_warm_bounds)/ref_mean)
    end

    verbose && @printf("Float64 (Miyajima)        %.4e    %.4e    %.2fx\n",
                       mean(bounds_f64), maximum(bounds_f64), mean(bounds_f64)/ref_mean)

    # Show point-by-point comparison for a few points
    verbose && println("\nPoint-by-point bounds (first 4 points):")
    verbose && println("-"^70)
    verbose && print("Point   BigFloat        ")
    HAS_DOUBLEFLOATS && d64_cold_data !== nothing && verbose && print("Double64        ")
    HAS_MULTIFLOATS && mf_cold_data !== nothing && verbose && print("MultiFloat      ")
    verbose && println("Miyajima")
    verbose && println("-"^70)

    for i in 1:min(4, num_points)
        verbose && @printf("  %2d    %.4e    ", i, bounds_cold[i])
        if HAS_DOUBLEFLOATS && d64_cold_data !== nothing
            verbose && @printf("%.4e    ", d64_cold_data[2][i])
        end
        if HAS_MULTIFLOATS && mf_cold_data !== nothing
            verbose && @printf("%.4e    ", mf_cold_data[2][i])
        end
        verbose && @printf("%.4e\n", bounds_f64[i])
    end

    return ExtendedCircleResult(bf_result, d64_cold_data, d64_warm_data,
                                d64_total_cold, d64_total_warm, d64_speedup, bf_vs_d64_speedup,
                                mf_cold_data, mf_warm_data, mf_total_cold, mf_total_warm, mf_speedup,
                                times_f64, bounds_f64, f64_total)
end

# ## Simple Cold-Start Accuracy Comparison
#
# This function provides a clean comparison of cold-start methods only,
# since warm-start doesn't work as expected with Ogita refinement.

"""
    compare_precision_methods(; n, center, radius, num_points, verbose)

Compare certification accuracy and timing across all precision levels.
Uses only cold-start (fresh SVD for each point).
"""
function compare_precision_methods(;
        n::Int=50,
        center::Complex=1.0+0.0im,
        radius::Real=0.15,
        num_points::Int=16,
        nonnormality::Real=0.5,
        verbose::Bool=true)

    verbose && println("\n" * "="^70)
    verbose && println("PRECISION COMPARISON (Cold-Start Only)")
    verbose && println("n=$n, ν=$nonnormality, circle: r=$radius, $num_points points")
    verbose && println("="^70)

    # Create test matrix
    T = make_spectral_gap_matrix(n; cluster_size=5, cluster_center=center,
                                  cluster_radius=0.02, gap=0.5, nonnormality=nonnormality)

    # Run each method
    methods = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()

    # Float64 Miyajima
    verbose && println("\n--- Float64 (Miyajima) ---")
    t0 = time()
    times_f64, bounds_f64 = certify_circle_float64(T, center, radius, num_points)
    total_f64 = time() - t0
    methods["Float64"] = (times_f64, bounds_f64)
    verbose && @printf("  Total: %.2fs, Mean bound: %.4e\n", total_f64, mean(bounds_f64))

    # BigFloat
    verbose && println("\n--- BigFloat ---")
    t0 = time()
    times_bf, bounds_bf, _ = certify_circle_cold(T, center, radius, num_points;
                                                  use_optimal_iterations=true)
    total_bf = time() - t0
    methods["BigFloat"] = (times_bf, bounds_bf)
    verbose && @printf("  Total: %.2fs, Mean bound: %.4e\n", total_bf, mean(bounds_bf))

    # Double64
    if HAS_DOUBLEFLOATS
        verbose && println("\n--- Double64 ---")
        t0 = time()
        times_d64, bounds_d64, _ = certify_circle_cold_d64(T, center, radius, num_points)
        total_d64 = time() - t0
        methods["Double64"] = (times_d64, bounds_d64)
        verbose && @printf("  Total: %.2fs, Mean bound: %.4e\n", total_d64, mean(bounds_d64))
    end

    # MultiFloat
    if HAS_MULTIFLOATS
        verbose && println("\n--- MultiFloat ---")
        t0 = time()
        times_mf, bounds_mf, _ = certify_circle_cold_mf(T, center, radius, num_points)
        total_mf = time() - t0
        methods["MultiFloat"] = (times_mf, bounds_mf)
        verbose && @printf("  Total: %.2fs, Mean bound: %.4e\n", total_mf, mean(bounds_mf))
    end

    # Summary table
    verbose && println("\n" * "="^70)
    verbose && println("SUMMARY")
    verbose && println("="^70)
    verbose && println("\nMethod          Time(s)   Mean Bound    Max Bound     vs BigFloat")
    verbose && println("-"^70)

    ref_mean = mean(bounds_bf)
    ref_max = maximum(bounds_bf)

    for (name, key) in [("BigFloat", "BigFloat"), ("Double64", "Double64"),
                        ("MultiFloat", "MultiFloat"), ("Float64 (Miy)", "Float64")]
        if haskey(methods, key)
            times, bounds = methods[key]
            total = sum(times)
            bmean = mean(bounds)
            bmax = maximum(bounds)
            overhead = bmean / ref_mean
            verbose && @printf("%-14s  %6.2f    %.4e    %.4e    %.2fx\n",
                               name, total, bmean, bmax, overhead)
        end
    end

    return methods
end

# ## Varying Circle Parameters

function run_warmstart_parameter_study(; verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("WARM-START PARAMETER STUDY")
    verbose && println("="^70)

    results = Dict{String, ExtendedCircleResult}()

    # Vary number of points (more points = closer together = better warm-start)
    verbose && println("\n### Varying number of points ###")
    for np in [8, 16, 32, 64]
        verbose && println("\n--- $np points ---")
        r = run_circle_warmstart_benchmark(; num_points=np, verbose=false)
        results["points_$np"] = r
        verbose && @printf("  BF Speedup: %.2fx, Iter reduction: %.0f%%\n",
                          r.bf.speedup, 100 * (1 - mean(r.bf.iterations_warm) / mean(r.bf.iterations_cold)))
    end

    # Vary radius (smaller radius = points closer together)
    verbose && println("\n### Varying circle radius ###")
    for rad in [0.05, 0.1, 0.2, 0.4]
        verbose && println("\n--- radius=$rad ---")
        r = run_circle_warmstart_benchmark(; radius=rad, num_points=32, verbose=false)
        results["radius_$rad"] = r
        verbose && @printf("  BF Speedup: %.2fx, Iter reduction: %.0f%%\n",
                          r.bf.speedup, 100 * (1 - mean(r.bf.iterations_warm) / mean(r.bf.iterations_cold)))
    end

    # Vary matrix size
    verbose && println("\n### Varying matrix size ###")
    for n in [30, 50, 70, 100]
        verbose && println("\n--- n=$n ---")
        r = run_circle_warmstart_benchmark(; n=n, k=5, num_points=32, verbose=false)
        results["size_$n"] = r
        verbose && @printf("  BF Speedup: %.2fx, Cold=%.1fs, Warm=%.1fs\n",
                          r.bf.speedup, r.bf.total_cold, r.bf.total_warm)
    end

    return results
end

# ## Generate LaTeX

function warmstart_circle_to_latex(result::ExtendedCircleResult, param_results;
                                   output_file="benchmark_warmstart_circle.tex")
    io = IOBuffer()
    bf = result.bf  # BigFloat results

    println(io, raw"""
\documentclass[11pt]{article}
\usepackage{booktabs,pgfplots,xcolor}
\pgfplotsset{compat=1.18}
\definecolor{bfcold}{RGB}{0,100,180}
\definecolor{bfwarm}{RGB}{180,0,0}
\definecolor{d64cold}{RGB}{0,150,100}
\definecolor{d64warm}{RGB}{200,100,0}
\definecolor{f64}{RGB}{128,128,128}
\begin{document}
\section*{Warm-Start Circle Certification: Precision Comparison}

When certifying the resolvent $\|(zI-A)^{-1}\|$ at multiple points around a circle,
adjacent points have similar shifted matrices. Using the SVD from point $i$ as
initial guess for Ogita refinement at point $i+1$ reduces iterations needed.

We compare four precision levels:
\begin{itemize}
\item \textbf{Float64}: Direct Miyajima SVD certification (fastest, least accurate)
\item \textbf{Double64}: Extended precision ($\sim$106 bits) with BigFloat certification
\item \textbf{MultiFloat}: Similar to Double64, using MultiFloats.jl
\item \textbf{BigFloat}: Full arbitrary precision (slowest, most accurate)
\end{itemize}

\subsection*{Time per Point (BigFloat)}
\begin{tikzpicture}
\begin{axis}[
    xlabel={Point index around circle},
    ylabel={Time (s)},
    legend pos=north east,
    width=0.9\textwidth,
    height=0.4\textwidth,
    grid=major,
]
""")

    print(io, "\\addplot[bfcold, thick] coordinates {")
    for (i, t) in enumerate(bf.times_cold)
        @printf(io, "(%d, %.4f) ", i, t)
    end
    println(io, "};")
    println(io, "\\addlegendentry{BigFloat Cold}")

    print(io, "\\addplot[bfwarm, thick] coordinates {")
    for (i, t) in enumerate(bf.times_warm)
        @printf(io, "(%d, %.4f) ", i, t)
    end
    println(io, "};")
    println(io, "\\addlegendentry{BigFloat Warm}")

    # Add Double64 if available
    if result.d64_cold !== nothing
        print(io, "\\addplot[d64cold, thick, dashed] coordinates {")
        for (i, t) in enumerate(result.d64_cold[1])
            @printf(io, "(%d, %.4f) ", i, t)
        end
        println(io, "};")
        println(io, "\\addlegendentry{Double64 Cold}")

        print(io, "\\addplot[d64warm, thick, dashed] coordinates {")
        for (i, t) in enumerate(result.d64_warm[1])
            @printf(io, "(%d, %.4f) ", i, t)
        end
        println(io, "};")
        println(io, "\\addlegendentry{Double64 Warm}")
    end

    # Add Float64 baseline
    if result.f64_times !== nothing
        print(io, "\\addplot[f64, thick, dotted] coordinates {")
        for (i, t) in enumerate(result.f64_times)
            @printf(io, "(%d, %.4f) ", i, t)
        end
        println(io, "};")
        println(io, "\\addlegendentry{Float64 (Miyajima)}")
    end

    println(io, raw"""
\end{axis}
\end{tikzpicture}

\subsection*{Overall Comparison}
""")

    # Main comparison table
    println(io, "\\begin{tabular}{lrrr}")
    println(io, "\\toprule")
    println(io, "Method & Total Time & Speedup vs BF Cold & Warm Speedup \\\\\\midrule")

    # Float64 baseline
    if result.f64_total < Inf
        @printf(io, "Float64 (Miyajima) & %.3f s & %.1f\\texttimes & -- \\\\\n",
                result.f64_total, bf.total_cold / result.f64_total)
    end

    # Double64
    if result.d64_total_cold < Inf
        @printf(io, "Double64 Cold & %.3f s & %.1f\\texttimes & -- \\\\\n",
                result.d64_total_cold, bf.total_cold / result.d64_total_cold)
        @printf(io, "Double64 Warm & %.3f s & %.1f\\texttimes & %.2f\\texttimes \\\\\n",
                result.d64_total_warm, bf.total_cold / result.d64_total_warm, result.d64_speedup)
    end

    # MultiFloat
    if result.mf_total_cold < Inf
        @printf(io, "MultiFloat Cold & %.3f s & %.1f\\texttimes & -- \\\\\n",
                result.mf_total_cold, bf.total_cold / result.mf_total_cold)
        @printf(io, "MultiFloat Warm & %.3f s & %.1f\\texttimes & %.2f\\texttimes \\\\\n",
                result.mf_total_warm, bf.total_cold / result.mf_total_warm, result.mf_speedup)
    end

    # BigFloat
    @printf(io, "BigFloat Cold & %.3f s & 1.0\\texttimes & -- \\\\\n", bf.total_cold)
    @printf(io, "BigFloat Warm & %.3f s & %.1f\\texttimes & %.2f\\texttimes \\\\\n",
            bf.total_warm, bf.total_cold / bf.total_warm, bf.speedup)

    println(io, "\\bottomrule\\end{tabular}")

    println(io, raw"""

\subsection*{Iterations per Point (BigFloat)}
\begin{tikzpicture}
\begin{axis}[
    xlabel={Point index around circle},
    ylabel={Ogita iterations},
    legend pos=north east,
    width=0.9\textwidth,
    height=0.4\textwidth,
    grid=major,
    ymin=0,
]
""")

    print(io, "\\addplot[bfcold, mark=*, thick] coordinates {")
    for (i, it) in enumerate(bf.iterations_cold)
        @printf(io, "(%d, %d) ", i, it)
    end
    println(io, "};")
    println(io, "\\addlegendentry{Cold start}")

    print(io, "\\addplot[bfwarm, mark=square*, thick] coordinates {")
    for (i, it) in enumerate(bf.iterations_warm)
        @printf(io, "(%d, %d) ", i, it)
    end
    println(io, "};")
    println(io, "\\addlegendentry{Warm start}")

    println(io, raw"""
\end{axis}
\end{tikzpicture}

\subsection*{BigFloat Statistics}
""")

    println(io, "\\begin{tabular}{lrr}")
    println(io, "\\toprule")
    println(io, "Metric & Cold Start & Warm Start \\\\\\midrule")
    @printf(io, "Total time & %.2f s & %.2f s \\\\\n", bf.total_cold, bf.total_warm)
    @printf(io, "Mean time/point & %.3f s & %.3f s \\\\\n",
            mean(bf.times_cold), mean(bf.times_warm))
    @printf(io, "Mean iterations & %.1f & %.1f \\\\\n",
            mean(bf.iterations_cold), mean(bf.iterations_warm))
    @printf(io, "\\textbf{Speedup} & \\multicolumn{2}{c}{\\textbf{%.2f\\texttimes}} \\\\\n", bf.speedup)
    println(io, "\\bottomrule\\end{tabular}")

    # Parameter study tables
    println(io, "\n\\subsection*{Parameter Study (BigFloat)}")

    println(io, "\n\\paragraph{Effect of number of points}")
    println(io, "\\begin{tabular}{rrrr}")
    println(io, "\\toprule")
    println(io, "Points & Cold (s) & Warm (s) & Speedup \\\\\\midrule")
    for np in [8, 16, 32, 64]
        key = "points_$np"
        if haskey(param_results, key)
            r = param_results[key].bf
            @printf(io, "%d & %.2f & %.2f & %.2f\\texttimes \\\\\n",
                    np, r.total_cold, r.total_warm, r.speedup)
        end
    end
    println(io, "\\bottomrule\\end{tabular}")

    println(io, "\n\\paragraph{Effect of circle radius}")
    println(io, "\\begin{tabular}{rrrr}")
    println(io, "\\toprule")
    println(io, "Radius & Cold (s) & Warm (s) & Speedup \\\\\\midrule")
    for rad in [0.05, 0.1, 0.2, 0.4]
        key = "radius_$rad"
        if haskey(param_results, key)
            r = param_results[key].bf
            @printf(io, "%.2f & %.2f & %.2f & %.2f\\texttimes \\\\\n",
                    rad, r.total_cold, r.total_warm, r.speedup)
        end
    end
    println(io, "\\bottomrule\\end{tabular}")

    println(io, "\n\\paragraph{Effect of matrix size}")
    println(io, "\\begin{tabular}{rrrr}")
    println(io, "\\toprule")
    println(io, "\$n\$ & Cold (s) & Warm (s) & Speedup \\\\\\midrule")
    for n in [30, 50, 70, 100]
        key = "size_$n"
        if haskey(param_results, key)
            r = param_results[key].bf
            @printf(io, "%d & %.2f & %.2f & %.2f\\texttimes \\\\\n",
                    n, r.total_cold, r.total_warm, r.speedup)
        end
    end
    println(io, "\\bottomrule\\end{tabular}")

    println(io, "\n\\end{document}")

    content = String(take!(io))
    open(output_file, "w") do f
        write(f, content)
    end
    println("LaTeX written to: $output_file")
    return output_file
end

# ## Main

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running warm-start benchmark with all precision levels...")
    result = run_circle_warmstart_benchmark(; num_points=32)

    println("\nRunning parameter study...")
    param_results = run_warmstart_parameter_study()

    println("\nGenerating LaTeX output...")
    warmstart_circle_to_latex(result, param_results)
end
