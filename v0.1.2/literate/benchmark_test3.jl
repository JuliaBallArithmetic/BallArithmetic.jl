# # Test 3: Parametric Configuration Comparison
#
# This test provides a detailed comparison of the four parametric configurations:
#
# | Config | D⁻¹ Estimator | Coupling | Combiner |
# |--------|---------------|----------|----------|
# | V1 | TriBacksub | None (product) | CombinerV1 |
# | V2 | TriBacksub | AR Solve | CombinerV2 |
# | V2.5 | TriBacksub | Off-diag Direct | CombinerV2p5 |
# | V3 | NeumannCollatz | AR Solve | CombinerV2 |
#
# The goal is to identify which combination provides the best trade-off
# between bound tightness and computation time, and potentially discover
# a superior V4 configuration.

include("benchmark_common.jl")

# ## Test Parameters

const TEST3_SIZES = [30, 50, 70, 100, 150]
const TEST3_NONNORMALITIES = [0.5, 1.0, 2.0]
const TEST3_CIRCLE_RADIUS = 0.05

# ## Extended Configuration Set
#
# Beyond the standard V1-V3, we can test additional combinations.

struct ConfigSpec
    name::String
    config::ResolventBoundConfig
    description::String
end

function get_all_configs()
    return [
        ConfigSpec("V1", config_v1(), "TriBacksub + Product + CombinerV1"),
        ConfigSpec("V2", config_v2(), "TriBacksub + AR + CombinerV2"),
        ConfigSpec("V2.5", config_v2p5(), "TriBacksub + OffDiag + CombinerV2p5"),
        ConfigSpec("V3", config_v3(), "NeumannCollatz + AR + CombinerV2"),
        # Additional experimental configs could be added here
    ]
end

# ## Run Test

"""
    run_test3(; sizes, nonnormalities, circle_radius, verbose)

Compare all parametric configurations across multiple matrix sizes.

Returns a dictionary mapping `(n, ν, config_name)` to `BenchmarkResult`.
"""
function run_test3(; sizes=TEST3_SIZES, nonnormalities=TEST3_NONNORMALITIES,
                    circle_radius=TEST3_CIRCLE_RADIUS, verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("TEST 3: Parametric Configuration Comparison")
    verbose && println("="^70)

    results = Dict{Tuple{Int, Float64, String}, BenchmarkResult}()
    configs = get_all_configs()

    for n in sizes
        for nn in nonnormalities
            verbose && println("\n--- n=$n, ν=$nn ---")

            eigs = eigenvalues_single_dominant(n; λ₁=1.0+0.0im, spacing=0.2)
            T = make_test_matrix(n; eigenvalues=eigs, nonnormality=nn)
            A = BallMatrix(T)
            z = 1.0 + circle_radius + 0.0im

            # Parametric configs
            for spec in configs
                r = certify_parametric(A, z, spec.config; name=spec.name)
                results[(n, nn, spec.name)] = r
                if verbose
                    status = r.success ? "OK" : "FAIL"
                    @printf("  %-8s: bound=%.4e, time=%.4fs [%s]\n",
                            spec.name, r.bound, r.time_seconds, status)
                end
            end

            # Baseline methods for comparison
            r_miy = certify_miyajima(A, z)
            r_ogi = certify_ogita(A, z)
            results[(n, nn, "Miyajima")] = r_miy
            results[(n, nn, "Ogita")] = r_ogi
            if verbose
                @printf("  %-8s: bound=%.4e, time=%.4fs [%s]\n",
                        "Miyajima", r_miy.bound, r_miy.time_seconds, r_miy.success ? "OK" : "FAIL")
                @printf("  %-8s: bound=%.4e, time=%.4fs [%s]\n",
                        "Ogita", r_ogi.bound, r_ogi.time_seconds, r_ogi.success ? "OK" : "FAIL")
            end
        end
    end

    return results
end

# ## Analysis Functions

"""
    compute_bound_ratios(results)

Compute ratio of each method's bound to the tightest bound for each (n, ν).
"""
function compute_bound_ratios(results)
    # Group by (n, ν)
    groups = Dict{Tuple{Int,Float64}, Dict{String,Float64}}()
    for ((n, nn, method), r) in results
        key = (n, nn)
        if !haskey(groups, key)
            groups[key] = Dict{String,Float64}()
        end
        if r.success
            groups[key][method] = r.bound
        end
    end

    # Compute ratios
    ratios = Dict{Tuple{Int,Float64,String}, Float64}()
    for ((n, nn), bounds) in groups
        if !isempty(bounds)
            best = minimum(values(bounds))
            for (method, bound) in bounds
                ratios[(n, nn, method)] = bound / best
            end
        end
    end

    return ratios
end

"""
    rank_methods(results)

Rank methods by average bound tightness and average time.
"""
function rank_methods(results)
    methods = unique([k[3] for k in keys(results)])

    avg_ratio = Dict{String, Float64}()
    avg_time = Dict{String, Float64}()
    success_rate = Dict{String, Float64}()

    ratios = compute_bound_ratios(results)

    for method in methods
        method_results = [(k, v) for (k, v) in results if k[3] == method]
        n_total = length(method_results)
        n_success = count(r -> r[2].success, method_results)

        success_rate[method] = n_success / n_total

        times = [r[2].time_seconds for r in method_results if r[2].success]
        avg_time[method] = isempty(times) ? Inf : mean(times)

        method_ratios = [v for (k, v) in ratios if k[3] == method]
        avg_ratio[method] = isempty(method_ratios) ? Inf : mean(method_ratios)
    end

    return (avg_ratio=avg_ratio, avg_time=avg_time, success_rate=success_rate)
end

# ## Generate LaTeX Output

function test3_to_latex(results; output_file="benchmark_test3.tex")
    io = IOBuffer()

    println(io, raw"""
\documentclass[11pt]{article}
\usepackage{booktabs,xcolor,pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{okgreen}{RGB}{0,128,0}
\definecolor{failred}{RGB}{200,0,0}
\definecolor{best}{RGB}{0,100,0}
\begin{document}
\section*{Test 3: Parametric Configuration Comparison}
""")

    sizes = sort(unique([k[1] for k in keys(results)]))
    nns = sort(unique([k[2] for k in keys(results)]))
    methods = sort(unique([k[3] for k in keys(results)]))

    # Ranking table
    ranking = rank_methods(results)
    println(io, "\\subsection*{Method Rankings}")
    println(io, "\\begin{tabular}{lrrr}")
    println(io, "\\toprule")
    println(io, "Method & Avg Bound Ratio & Avg Time (s) & Success Rate \\\\\\midrule")

    sorted_methods = sort(collect(ranking.avg_ratio), by=x->x[2])
    for (method, ratio) in sorted_methods
        if isfinite(ratio)
            @printf(io, "%s & %.3f & %.4f & %.0f\\%% \\\\\n",
                    method, ratio, ranking.avg_time[method], 100*ranking.success_rate[method])
        end
    end
    println(io, "\\bottomrule\\end{tabular}\n")

    # Bound comparison at ν=1.0
    println(io, "\\subsection*{Certified Bounds at \$\\nu=1.0\$}")
    println(io, "\\begin{tabular}{l|" * "r"^length(sizes) * "}")
    println(io, "\\toprule")
    print(io, "Method")
    for n in sizes
        print(io, " & \$n=$n\$")
    end
    println(io, " \\\\\\midrule")

    for method in methods
        print(io, method)
        for n in sizes
            key = (n, 1.0, method)
            if haskey(results, key)
                r = results[key]
                if r.success
                    @printf(io, " & %.2e", r.bound)
                else
                    print(io, " & \\textcolor{failred}{--}")
                end
            else
                print(io, " & --")
            end
        end
        println(io, " \\\\")
    end
    println(io, "\\bottomrule\\end{tabular}\n")

    # Timing comparison
    println(io, "\\subsection*{Timing at \$\\nu=1.0\$ (seconds)}")
    println(io, "\\begin{tabular}{l|" * "r"^length(sizes) * "}")
    println(io, "\\toprule")
    print(io, "Method")
    for n in sizes
        print(io, " & \$n=$n\$")
    end
    println(io, " \\\\\\midrule")

    for method in methods
        print(io, method)
        for n in sizes
            key = (n, 1.0, method)
            if haskey(results, key)
                r = results[key]
                if r.success
                    @printf(io, " & %.3f", r.time_seconds)
                else
                    print(io, " & --")
                end
            else
                print(io, " & --")
            end
        end
        println(io, " \\\\")
    end
    println(io, "\\bottomrule\\end{tabular}\n")

    # Scaling plot
    println(io, raw"""
\subsection*{Timing Scaling with Matrix Size}
\begin{tikzpicture}
\begin{axis}[
    xlabel={Matrix size $n$},
    ylabel={Time (s)},
    legend pos=north west,
    width=0.9\textwidth,
    height=0.5\textwidth,
    grid=major,
    ymode=log,
]
""")

    colors = ["blue", "red", "green!60!black", "orange", "purple", "cyan"]
    for (i, method) in enumerate(methods)
        print(io, "\\addplot[$(colors[mod1(i, length(colors))]), mark=*, thick] coordinates {")
        for n in sizes
            key = (n, 1.0, method)
            if haskey(results, key) && results[key].success
                @printf(io, "(%d, %.4f) ", n, results[key].time_seconds)
            end
        end
        println(io, "};")
        println(io, "\\addlegendentry{$method}")
    end

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
    results = run_test3()
    test3_to_latex(results)
end
