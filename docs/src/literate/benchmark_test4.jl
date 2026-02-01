# # Test 4: Warm-Start Effectiveness
#
# When certifying multiple points along a contour, we can reuse the SVD
# from a nearby point as a starting guess. This test measures:
#
# - Cache hit rate: How often the warm-start succeeds
# - Cache misses: When we need to recompute from scratch
# - Fallbacks: When warm-start fails and we fall back to cold start
# - Overall speedup from the caching strategy

include("benchmark_common.jl")

# ## Test Parameters

const TEST4_N = 50
const TEST4_NUM_POINTS = 32
const TEST4_CIRCLE_RADIUS = 0.1
const TEST4_DISTANCE_THRESHOLD = 0.05  # Max distance for warm-start

# ## Cache Statistics Structure

struct WarmStartStats
    times_cold::Vector{Float64}
    times_warm::Vector{Float64}
    cache_hits::Int
    cache_misses::Int
    fallbacks::Int
    total_cold::Float64
    total_warm::Float64
    speedup::Float64
    hit_rate::Float64
end

# ## Run Test

"""
    run_test4(; n, num_points, circle_radius, distance_threshold, verbose)

Measure warm-start effectiveness when certifying points along a circular contour.

Returns `WarmStartStats` with detailed cache statistics.
"""
function run_test4(; n=TEST4_N, num_points=TEST4_NUM_POINTS,
                    circle_radius=TEST4_CIRCLE_RADIUS,
                    distance_threshold=TEST4_DISTANCE_THRESHOLD,
                    verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("TEST 4: Warm-start effectiveness along contour")
    verbose && println("="^70)

    # Create test matrix
    eigs = eigenvalues_single_dominant(n; λ₁=1.0+0.0im, spacing=0.2)
    T = make_test_matrix(n; eigenvalues=eigs, nonnormality=1.0)
    A = BallMatrix(T)

    # Points along contour
    center = 1.0 + 0.0im
    θs = range(0, 2π, length=num_points+1)[1:end-1]
    zs = [center + circle_radius * exp(im * θ) for θ in θs]

    # === Cold Start (no caching) ===
    verbose && println("\n--- Cold start (recompute SVD each time) ---")
    times_cold = Float64[]
    T_mid = mid.(A)

    for (i, z) in enumerate(zs)
        t0 = time()
        Ashift = T_mid - z * I  # Regular matrix, not BallMatrix
        U, S, V = svd(Ashift)
        ogita_svd_refine(Ashift, U, S, V; max_iterations=5)
        push!(times_cold, time() - t0)
    end

    total_cold = sum(times_cold)
    verbose && @printf("  Total: %.4fs, Mean: %.4fs, Std: %.4fs\n",
                       total_cold, mean(times_cold), std(times_cold))

    # === Warm Start (with caching) ===
    verbose && println("\n--- Warm start (reuse SVD from nearby points) ---")
    times_warm = Float64[]
    cache_hits = 0
    cache_misses = 0
    fallbacks = 0

    # Initialize cache with first point
    z = zs[1]
    Ashift = T_mid - z * I  # Regular matrix
    U_cache, S_cache, V_cache = svd(Ashift)
    z_cache = z

    t0 = time()
    result = ogita_svd_refine(Ashift, U_cache, S_cache, V_cache; max_iterations=5)
    push!(times_warm, time() - t0)
    cache_misses += 1  # First point is always a miss

    verbose && @printf("  Point %3d: cold start (first point)\n", 1)

    # Process remaining points
    for i in 2:length(zs)
        z = zs[i]
        dist = abs(z - z_cache)

        t0 = time()
        Ashift = T_mid - z * I  # Regular matrix

        if dist <= distance_threshold
            # Try warm start
            result_warm = ogita_svd_refine(Ashift, U_cache, S_cache, V_cache;
                                           max_iterations=5)

            if result_warm.converged
                # Warm start succeeded
                cache_hits += 1
                push!(times_warm, time() - t0)
                verbose && @printf("  Point %3d: cache HIT  (dist=%.4f)\n", i, dist)
            else
                # Warm start failed, fallback to cold
                fallbacks += 1
                U_new, S_new, V_new = svd(Ashift)
                ogita_svd_refine(Ashift, U_new, S_new, V_new; max_iterations=5)
                push!(times_warm, time() - t0)

                # Update cache
                U_cache, S_cache, V_cache = U_new, S_new, V_new
                z_cache = z
                verbose && @printf("  Point %3d: FALLBACK  (dist=%.4f, warm failed)\n", i, dist)
            end
        else
            # Distance too large, cold start
            cache_misses += 1
            U_new, S_new, V_new = svd(Ashift)
            ogita_svd_refine(Ashift, U_new, S_new, V_new; max_iterations=5)
            push!(times_warm, time() - t0)

            # Update cache
            U_cache, S_cache, V_cache = U_new, S_new, V_new
            z_cache = z
            verbose && @printf("  Point %3d: cache MISS (dist=%.4f > threshold)\n", i, dist)
        end
    end

    total_warm = sum(times_warm)
    speedup = total_cold / total_warm
    hit_rate = cache_hits / num_points

    verbose && println("\n--- Summary ---")
    verbose && @printf("  Total time (cold):  %.4fs\n", total_cold)
    verbose && @printf("  Total time (warm):  %.4fs\n", total_warm)
    verbose && @printf("  Speedup:            %.2fx\n", speedup)
    verbose && println()
    verbose && @printf("  Cache hits:         %d (%.1f%%)\n", cache_hits, 100*hit_rate)
    verbose && @printf("  Cache misses:       %d\n", cache_misses)
    verbose && @printf("  Fallbacks:          %d\n", fallbacks)

    return WarmStartStats(times_cold, times_warm, cache_hits, cache_misses,
                          fallbacks, total_cold, total_warm, speedup, hit_rate)
end

# ## Threshold Sensitivity Analysis

"""
    analyze_threshold_sensitivity(; thresholds, kwargs...)

Test different distance thresholds to find optimal caching strategy.
"""
function analyze_threshold_sensitivity(; thresholds=[0.01, 0.02, 0.05, 0.1, 0.2],
                                        n=TEST4_N, num_points=TEST4_NUM_POINTS,
                                        circle_radius=TEST4_CIRCLE_RADIUS)
    println("\n--- Threshold Sensitivity Analysis ---")

    results = Dict{Float64, WarmStartStats}()

    for thresh in thresholds
        stats = run_test4(; n=n, num_points=num_points, circle_radius=circle_radius,
                           distance_threshold=thresh, verbose=false)
        results[thresh] = stats
        @printf("  threshold=%.3f: speedup=%.2fx, hit_rate=%.1f%%, fallbacks=%d\n",
                thresh, stats.speedup, 100*stats.hit_rate, stats.fallbacks)
    end

    return results
end

# ## Generate LaTeX Output

function test4_to_latex(stats::WarmStartStats; threshold_results=nothing,
                        output_file="benchmark_test4.tex")
    io = IOBuffer()

    println(io, raw"""
\documentclass[11pt]{article}
\usepackage{booktabs,xcolor,pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\section*{Test 4: Warm-Start Effectiveness}
""")

    # Summary statistics
    println(io, "\\subsection*{Cache Statistics}")
    println(io, "\\begin{tabular}{lr}")
    println(io, "\\toprule")
    println(io, "Metric & Value \\\\\\midrule")
    @printf(io, "Total points & %d \\\\\n", length(stats.times_warm))
    @printf(io, "Cache hits & %d (%.1f\\%%) \\\\\n", stats.cache_hits, 100*stats.hit_rate)
    @printf(io, "Cache misses & %d \\\\\n", stats.cache_misses)
    @printf(io, "Fallbacks & %d \\\\\n", stats.fallbacks)
    println(io, "\\midrule")
    @printf(io, "Total time (cold) & %.4f s \\\\\n", stats.total_cold)
    @printf(io, "Total time (warm) & %.4f s \\\\\n", stats.total_warm)
    @printf(io, "\\textbf{Speedup} & \\textbf{%.2f\\texttimes} \\\\\n", stats.speedup)
    println(io, "\\bottomrule\\end{tabular}\n")

    # Time per point plot
    println(io, raw"""
\subsection*{Time per Point}
\begin{tikzpicture}
\begin{axis}[
    xlabel={Point index},
    ylabel={Time (s)},
    legend pos=north east,
    width=0.9\textwidth,
    height=0.4\textwidth,
    grid=major,
]
""")

    print(io, "\\addplot[blue, mark=*, thick, mark size=1pt] coordinates {")
    for (i, t) in enumerate(stats.times_cold)
        @printf(io, "(%d, %.4f) ", i, t)
    end
    println(io, "};")
    println(io, "\\addlegendentry{Cold start}")

    print(io, "\\addplot[red, mark=square*, thick, mark size=1pt] coordinates {")
    for (i, t) in enumerate(stats.times_warm)
        @printf(io, "(%d, %.4f) ", i, t)
    end
    println(io, "};")
    println(io, "\\addlegendentry{Warm start}")

    println(io, raw"""
\end{axis}
\end{tikzpicture}
""")

    # Threshold sensitivity
    if !isnothing(threshold_results)
        println(io, "\\subsection*{Threshold Sensitivity}")
        println(io, "\\begin{tabular}{rrrr}")
        println(io, "\\toprule")
        println(io, "Threshold & Speedup & Hit Rate & Fallbacks \\\\\\midrule")

        for thresh in sort(collect(keys(threshold_results)))
            s = threshold_results[thresh]
            @printf(io, "%.3f & %.2f\\texttimes & %.1f\\%% & %d \\\\\n",
                    thresh, s.speedup, 100*s.hit_rate, s.fallbacks)
        end
        println(io, "\\bottomrule\\end{tabular}\n")
    end

    println(io, "\\end{document}")

    content = String(take!(io))
    open(output_file, "w") do f
        write(f, content)
    end
    println("LaTeX written to: $output_file")
    return output_file
end

# ## Main

if abspath(PROGRAM_FILE) == @__FILE__
    stats = run_test4()
    threshold_results = analyze_threshold_sensitivity()
    test4_to_latex(stats; threshold_results=threshold_results)
end
