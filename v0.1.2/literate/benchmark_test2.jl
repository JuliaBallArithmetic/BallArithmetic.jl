# # Test 2: Varying Circle Radius (Float64 Breakdown)
#
# As the certification radius decreases toward the eigenvalue, Float64
# methods eventually fail due to numerical precision limits.
#
# This test identifies:
# - The breakdown radius for each Float64 method
# - Required BigFloat precision for small radii
# - Trade-offs between precision and computation time

include("benchmark_common.jl")

# ## Test Parameters

const TEST2_N = 50
const TEST2_RADII = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]
const TEST2_BIGFLOAT_PRECISIONS = [128, 256, 512]

# ## Run Test

"""
    run_test2(; n, radii, bigfloat_precisions, verbose)

Test certification methods at decreasing circle radii.

Returns a dictionary mapping `(radius, method_name)` to `BenchmarkResult`.
"""
function run_test2(; n=TEST2_N, radii=TEST2_RADII,
                    bigfloat_precisions=TEST2_BIGFLOAT_PRECISIONS, verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("TEST 2: Varying circle radius (Float64 breakdown)")
    verbose && println("="^70)

    results = Dict{Tuple{Float64, String}, BenchmarkResult}()

    # Create matrix with isolated dominant eigenvalue
    eigs = eigenvalues_single_dominant(n; λ₁=1.0+0.0im, spacing=0.3)
    T = make_test_matrix(n; eigenvalues=eigs, nonnormality=1.0)
    A = BallMatrix(T)

    for radius in radii
        verbose && println("\n--- radius = $radius ---")
        z = 1.0 + radius + 0.0im

        # Float64 methods
        float64_methods = [
            ("Miyajima", () -> certify_miyajima(A, z)),
            ("Ogita", () -> certify_ogita(A, z)),
            ("V1", () -> certify_parametric(A, z, config_v1(); name="V1")),
            ("V2", () -> certify_parametric(A, z, config_v2(); name="V2")),
            ("V2.5", () -> certify_parametric(A, z, config_v2p5(); name="V2.5")),
            ("V3", () -> certify_parametric(A, z, config_v3(); name="V3")),
        ]

        for (name, method_fn) in float64_methods
            r = method_fn()
            results[(radius, name)] = r
            if verbose
                status = r.success ? "OK" : "FAIL"
                @printf("  %-20s: bound=%.4e, time=%.4fs [%s]\n",
                        name, r.bound, r.time_seconds, status)
            end
        end

        # BigFloat methods for small radii
        if radius <= 1e-4
            for prec in bigfloat_precisions
                r = certify_bigfloat(A, z; precision=prec)
                results[(radius, r.method)] = r
                if verbose
                    status = r.success ? "OK" : "FAIL"
                    @printf("  %-20s: bound=%.4e, time=%.4fs [%s]\n",
                            r.method, r.bound, r.time_seconds, status)
                end
            end
        end
    end

    return results
end

# ## Analysis: Find Breakdown Radii

"""
    find_breakdown_radii(results)

For each method, find the smallest radius at which it still succeeds.
"""
function find_breakdown_radii(results)
    methods = unique([k[2] for k in keys(results)])
    breakdown = Dict{String, Float64}()

    for method in methods
        radii_for_method = [k[1] for k in keys(results) if k[2] == method && results[k].success]
        if !isempty(radii_for_method)
            breakdown[method] = minimum(radii_for_method)
        else
            breakdown[method] = Inf
        end
    end

    return breakdown
end

# ## Generate LaTeX Output

function test2_to_latex(results; output_file="benchmark_test2.tex")
    io = IOBuffer()

    println(io, raw"""
\documentclass[11pt]{article}
\usepackage{booktabs,xcolor,pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{okgreen}{RGB}{0,128,0}
\definecolor{failred}{RGB}{200,0,0}
\begin{document}
\section*{Test 2: Circle Radius and Float64 Breakdown}
""")

    radii = sort(unique([k[1] for k in keys(results)]), rev=true)
    methods = sort(unique([k[2] for k in keys(results)]))

    # Success/failure matrix
    println(io, "\\subsection*{Success by Radius}")
    println(io, "\\begin{tabular}{l|" * "c"^length(radii) * "}")
    println(io, "\\toprule")
    print(io, "Method")
    for r in radii
        @printf(io, " & \$10^{%d}\$", round(Int, log10(r)))
    end
    println(io, " \\\\\\midrule")

    for method in methods
        print(io, replace(method, "_" => "\\_"))
        for r in radii
            key = (r, method)
            if haskey(results, key)
                res = results[key]
                if res.success
                    print(io, " & \\textcolor{okgreen}{\\checkmark}")
                else
                    print(io, " & \\textcolor{failred}{\\texttimes}")
                end
            else
                print(io, " & --")
            end
        end
        println(io, " \\\\")
    end
    println(io, "\\bottomrule\\end{tabular}\n")

    # Breakdown radii summary
    breakdown = find_breakdown_radii(results)
    println(io, "\\subsection*{Breakdown Radii}")
    println(io, "\\begin{tabular}{lr}")
    println(io, "\\toprule")
    println(io, "Method & Smallest Successful Radius \\\\\\midrule")
    for (method, radius) in sort(collect(breakdown), by=x->x[2])
        if isfinite(radius)
            @printf(io, "%s & \$%.0e\$ \\\\\n", replace(method, "_" => "\\_"), radius)
        else
            println(io, "$(replace(method, "_" => "\\_")) & -- \\\\")
        end
    end
    println(io, "\\bottomrule\\end{tabular}\n")

    # Bound vs radius plot
    println(io, raw"""
\subsection*{Bound Quality vs Radius}
\begin{tikzpicture}
\begin{loglogaxis}[
    xlabel={Circle radius},
    ylabel={Resolvent bound},
    legend pos=north east,
    width=0.9\textwidth,
    height=0.5\textwidth,
    grid=major,
    x dir=reverse,
]
""")

    colors = ["blue", "red", "green!60!black", "orange", "purple", "cyan", "brown", "pink", "gray"]
    for (i, method) in enumerate(methods)
        print(io, "\\addplot[$(colors[mod1(i, length(colors))]), mark=*, thick] coordinates {")
        for r in radii
            key = (r, method)
            if haskey(results, key) && results[key].success
                @printf(io, "(%.2e, %.4e) ", r, results[key].bound)
            end
        end
        println(io, "};")
        println(io, "\\addlegendentry{$(replace(method, "_" => "\\_"))}")
    end

    println(io, raw"""
\end{loglogaxis}
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
    results = run_test2()
    test2_to_latex(results)
end
