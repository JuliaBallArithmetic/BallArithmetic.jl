# # Test 5: Parallel Scaling
#
# This test measures how certification scales with multiple workers.
# We compare:
# - Serial execution (1 worker)
# - Parallel execution with 2, 4 workers
# - Different certification methods (Miyajima, Ogita, Parametric)

include("benchmark_common.jl")
using Distributed

# ## Test Parameters

const TEST5_N = 50
const TEST5_NUM_POINTS = 64
const TEST5_CIRCLE_RADIUS = 0.1
const TEST5_MAX_WORKERS = 4

# ## Parallel Scaling Results

struct ParallelScalingResult
    worker_counts::Vector{Int}
    times::Vector{Float64}
    speedups::Vector{Float64}
    efficiencies::Vector{Float64}
end

# ## Serial Baseline

"""
    run_serial_baseline(A, circle; method, verbose)

Run certification serially to establish baseline timing.
"""
function run_serial_baseline(A::BallMatrix, circle::CertificationCircle;
                             method::Symbol=:ogita, verbose=true)
    zs = points_on(circle)
    times = Float64[]

    verbose && println("  Running serial certification ($(length(zs)) points)...")

    for z in zs
        t0 = time()
        if method == :miyajima
            certify_miyajima(A, z)
        elseif method == :ogita
            certify_ogita(A, z)
        elseif method == :parametric
            certify_parametric(A, z, config_v2(); name="V2")
        end
        push!(times, time() - t0)
    end

    return sum(times)
end

# ## Parallel Execution

"""
    run_parallel_certification(A, circle, worker_ids; method, verbose)

Run certification in parallel using the distributed framework.
"""
function run_parallel_certification(A::BallMatrix, circle::CertificationCircle,
                                    worker_ids::Vector{Int};
                                    method::Symbol=:ogita, verbose=true)
    verbose && println("  Running parallel certification ($(length(worker_ids)) workers)...")

    log_io = IOBuffer()

    t0 = time()
    try
        if method == :parametric
            run_certification(A, circle; η=0.5, log_io=log_io,
                             use_parametric=true, parametric_config=config_v2())
        else
            run_certification(A, circle; η=0.5, log_io=log_io,
                             use_ogita_cache=(method == :ogita))
        end
    catch e
        verbose && println("  Warning during parallel run: $e")
    end

    return time() - t0
end

# ## Run Test

"""
    run_test5(; n, num_points, circle_radius, max_workers, verbose)

Test parallel scaling with different worker counts.

Returns `ParallelScalingResult` with timing and efficiency data.
"""
function run_test5(; n=TEST5_N, num_points=TEST5_NUM_POINTS,
                    circle_radius=TEST5_CIRCLE_RADIUS,
                    max_workers=TEST5_MAX_WORKERS,
                    verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("TEST 5: Parallel Scaling")
    verbose && println("="^70)

    # Create test matrix
    eigs = eigenvalues_single_dominant(n; λ₁=1.0+0.0im, spacing=0.2)
    T = make_test_matrix(n; eigenvalues=eigs, nonnormality=1.0)
    A = BallMatrix(T)
    circle = CertificationCircle(1.0 + 0.0im, circle_radius; samples=num_points)

    worker_counts = Int[]
    times = Float64[]

    # Serial baseline
    verbose && println("\n--- Serial (1 worker) ---")
    t_serial = run_serial_baseline(A, circle; method=:ogita, verbose=verbose)
    push!(worker_counts, 1)
    push!(times, t_serial)
    verbose && @printf("  Time: %.4fs\n", t_serial)

    # Parallel runs
    for nw in [2, max_workers]
        verbose && println("\n--- $nw workers ---")

        # Check/add workers
        current = nworkers()
        if current < nw
            addprocs(nw - current)
            @everywhere using BallArithmetic
        end

        worker_ids = workers()[1:nw]
        t_parallel = run_parallel_certification(A, circle, worker_ids;
                                                method=:ogita, verbose=verbose)
        push!(worker_counts, nw)
        push!(times, t_parallel)

        speedup = t_serial / t_parallel
        efficiency = speedup / nw
        verbose && @printf("  Time: %.4fs, Speedup: %.2fx, Efficiency: %.1f%%\n",
                           t_parallel, speedup, 100*efficiency)
    end

    # Compute metrics
    speedups = [t_serial / t for t in times]
    efficiencies = [s / w for (s, w) in zip(speedups, worker_counts)]

    return ParallelScalingResult(worker_counts, times, speedups, efficiencies)
end

# ## Run Without Starting Workers (for testing)

"""
    run_test5_serial_only(; kwargs...)

Run only the serial portion of the test (no worker management).
"""
function run_test5_serial_only(; n=TEST5_N, num_points=TEST5_NUM_POINTS,
                                circle_radius=TEST5_CIRCLE_RADIUS,
                                verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("TEST 5: Serial Baseline Only")
    verbose && println("="^70)

    eigs = eigenvalues_single_dominant(n; λ₁=1.0+0.0im, spacing=0.2)
    T = make_test_matrix(n; eigenvalues=eigs, nonnormality=1.0)
    A = BallMatrix(T)
    circle = CertificationCircle(1.0 + 0.0im, circle_radius; samples=num_points)

    verbose && println("\n--- Serial execution ---")
    t_serial = run_serial_baseline(A, circle; method=:ogita, verbose=verbose)
    verbose && @printf("  Time: %.4fs\n", t_serial)

    return t_serial
end

# ## Generate LaTeX Output

function test5_to_latex(results::ParallelScalingResult; output_file="benchmark_test5.tex")
    io = IOBuffer()

    println(io, raw"""
\documentclass[11pt]{article}
\usepackage{booktabs,pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\section*{Test 5: Parallel Scaling}
""")

    # Results table
    println(io, "\\subsection*{Scaling Results}")
    println(io, "\\begin{tabular}{rrrr}")
    println(io, "\\toprule")
    println(io, "Workers & Time (s) & Speedup & Efficiency \\\\\\midrule")

    for i in eachindex(results.worker_counts)
        @printf(io, "%d & %.3f & %.2f\\texttimes & %.1f\\%% \\\\\n",
                results.worker_counts[i], results.times[i],
                results.speedups[i], 100*results.efficiencies[i])
    end
    println(io, "\\bottomrule\\end{tabular}\n")

    # Speedup plot
    println(io, raw"""
\subsection*{Speedup vs Workers}
\begin{tikzpicture}
\begin{axis}[
    xlabel={Number of workers},
    ylabel={Speedup},
    legend pos=north west,
    width=0.7\textwidth,
    height=0.5\textwidth,
    grid=major,
    xmin=0,
    ymin=0,
]
""")

    # Ideal scaling line
    max_w = maximum(results.worker_counts)
    println(io, "\\addplot[gray, dashed, thick] coordinates {(0, 0) ($max_w, $max_w)};")
    println(io, "\\addlegendentry{Ideal}")

    # Actual speedup
    print(io, "\\addplot[blue, mark=*, thick] coordinates {")
    for (w, s) in zip(results.worker_counts, results.speedups)
        @printf(io, "(%d, %.2f) ", w, s)
    end
    println(io, "};")
    println(io, "\\addlegendentry{Measured}")

    println(io, raw"""
\end{axis}
\end{tikzpicture}
""")

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
    # Note: For full parallel test, run with: julia -p 4 benchmark_test5.jl
    if nworkers() > 1
        results = run_test5()
        test5_to_latex(results)
    else
        println("Running serial baseline only (no workers available)")
        println("For full parallel test, run with: julia -p 4 benchmark_test5.jl")
        t_serial = run_test5_serial_only()
    end
end
