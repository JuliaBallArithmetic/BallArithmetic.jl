# # Test 1: Varying Matrix Size and Nonnormality
#
# This test examines how different certification methods perform as we vary:
# - Matrix dimension ``n \in \{20, 40, 60, 80, 100\}``
# - Nonnormality strength ``\nu \in \{0.1, 0.5, 1.0, 2.0, 5.0\}``
#
# The test matrix is upper triangular with a dominant eigenvalue at ``\lambda_1 = 1``.
# The certification point is at ``z = 1 + r`` where ``r`` is the circle radius.

include("benchmark_common.jl")

# ## Test Parameters

const TEST1_SIZES = [20, 40, 60, 80, 100]
const TEST1_NONNORMALITIES = [0.1, 0.5, 1.0, 2.0, 5.0]
const TEST1_CIRCLE_RADIUS = 0.05

# ## Run Test

"""
    run_test1(; sizes, nonnormalities, circle_radius, verbose)

Test certification methods across varying matrix sizes and nonnormality strengths.

Returns a dictionary mapping `(n, ν, method_name)` to `BenchmarkResult`.
"""
function run_test1(; sizes=TEST1_SIZES, nonnormalities=TEST1_NONNORMALITIES,
                    circle_radius=TEST1_CIRCLE_RADIUS, verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("TEST 1: Varying matrix size and nonnormality strength")
    verbose && println("="^70)

    results = Dict{Tuple{Int, Float64, String}, BenchmarkResult}()

    for n in sizes
        for nn in nonnormalities
            verbose && println("\n--- n=$n, nonnormality=$nn ---")

            # Create test matrix
            eigs = eigenvalues_single_dominant(n; λ₁=1.0+0.0im, spacing=0.2)
            T = make_test_matrix(n; eigenvalues=eigs, nonnormality=nn)
            A = BallMatrix(T)
            z = 1.0 + circle_radius + 0.0im

            # Methods to test
            methods = [
                ("Miyajima", () -> certify_miyajima(A, z)),
                ("Ogita", () -> certify_ogita(A, z)),
                ("Double64", () -> certify_double64(A, z)),
                ("Hybrid", () -> certify_hybrid(A, z)),
                ("MF-x2", () -> certify_multifloat(A, z; precision=:x2)),
                ("V1", () -> certify_parametric(A, z, config_v1(); name="V1")),
                ("V2", () -> certify_parametric(A, z, config_v2(); name="V2")),
                ("V2.5", () -> certify_parametric(A, z, config_v2p5(); name="V2.5")),
                ("V3", () -> certify_parametric(A, z, config_v3(); name="V3")),
            ]

            for (name, method_fn) in methods
                r = method_fn()
                results[(n, nn, name)] = r
                if verbose
                    status = r.success ? "OK" : "FAIL"
                    @printf("  %-15s: bound=%.4e, time=%.4fs [%s]\n",
                            name, r.bound, r.time_seconds, status)
                end
            end
        end
    end

    return results
end

# ## Generate LaTeX Output

function test1_to_latex(results; output_file="benchmark_test1.tex")
    io = IOBuffer()

    println(io, raw"""
\documentclass[11pt]{article}
\usepackage{booktabs,xcolor}
\definecolor{okgreen}{RGB}{0,128,0}
\definecolor{failred}{RGB}{200,0,0}
\begin{document}
\section*{Test 1: Size and Nonnormality Scaling}
""")

    sizes = sort(unique([k[1] for k in keys(results)]))
    nns = sort(unique([k[2] for k in keys(results)]))
    methods = sort(unique([k[3] for k in keys(results)]))

    # Timing tables
    println(io, "\\subsection*{Timing (seconds)}")
    for method in methods
        println(io, "\n\\paragraph{$method}")
        println(io, "\\begin{tabular}{l|" * "r"^length(sizes) * "}")
        println(io, "\\toprule")
        print(io, "\$\\nu\$ \\textbackslash{} \$n\$")
        for n in sizes
            print(io, " & $n")
        end
        println(io, " \\\\\\midrule")

        for nn in nns
            @printf(io, "%.1f", nn)
            for n in sizes
                key = (n, nn, method)
                if haskey(results, key)
                    r = results[key]
                    if r.success
                        @printf(io, " & %.3f", r.time_seconds)
                    else
                        print(io, " & \\textcolor{failred}{fail}")
                    end
                else
                    print(io, " & --")
                end
            end
            println(io, " \\\\")
        end
        println(io, "\\bottomrule\\end{tabular}\n")
    end

    # Bound quality tables
    println(io, "\\subsection*{Certified Bounds}")
    for method in methods
        println(io, "\n\\paragraph{$method}")
        println(io, "\\begin{tabular}{l|" * "r"^length(sizes) * "}")
        println(io, "\\toprule")
        print(io, "\$\\nu\$ \\textbackslash{} \$n\$")
        for n in sizes
            print(io, " & $n")
        end
        println(io, " \\\\\\midrule")

        for nn in nns
            @printf(io, "%.1f", nn)
            for n in sizes
                key = (n, nn, method)
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
    results = run_test1()
    test1_to_latex(results)
end
