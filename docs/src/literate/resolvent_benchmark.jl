# # Resolvent Certification Methods: Comparative Benchmark
#
# This benchmark compares different methods for certifying resolvent bounds
# ``\|(zI - A)^{-1}\|_2`` in the BallArithmetic.jl package.
#
# ## Methods Compared
#
# 1. **Pure Miyajima SVD**: Direct SVD enclosure via Miyajima's method
# 2. **Ogita + Miyajima**: Newton refinement with Miyajima certification
# 3. **Parametric Sylvester (V1-V3)**: Block-diagonalization with various estimators
# 4. **BigFloat variants**: Extended precision for small radii
#
# ## Test Suite Overview
#
# - **Test 1**: Vary matrix size and nonnormality strength
# - **Test 2**: Vary circle radius (Float64 breakdown point)
# - **Test 3**: Compare parametric configurations
# - **Test 4**: Warm-start effectiveness with cache statistics
# - **Test 5**: Parallel scaling

using BallArithmetic
using BallArithmetic.CertifScripts
using LinearAlgebra
using Printf
using Statistics
using Random

# ## Matrix Generators
#
# We create upper triangular test matrices with controllable eigenvalue
# structure and nonnormality strength.

"""
    make_test_matrix(n; eigenvalues, nonnormality=1.0, seed=42)

Create upper triangular test matrix with specified eigenvalues.
The `nonnormality` parameter scales the off-diagonal entries.
"""
function make_test_matrix(n::Int; eigenvalues::Vector{<:Number},
                          nonnormality::Real=1.0, seed::Int=42)
    Random.seed!(seed)
    @assert length(eigenvalues) == n
    T = zeros(ComplexF64, n, n)
    for i in 1:n
        T[i, i] = eigenvalues[i]
    end
    for i in 1:n, j in (i+1):n
        T[i, j] = nonnormality * (randn() + im * randn()) / sqrt(2)
    end
    return T
end

"""Create eigenvalues with single dominant one at λ₁."""
function eigenvalues_single_dominant(n; λ₁=1.0+0.0im, spacing=0.1)
    eigs = Vector{ComplexF64}(undef, n)
    eigs[1] = λ₁
    for i in 2:n
        eigs[i] = λ₁ - spacing * i + 0.05im * (i - n/2)
    end
    return eigs
end

"""Create eigenvalues with a cluster near center."""
function eigenvalues_cluster(n; center=1.0+0.0im, cluster_size=3,
                             cluster_radius=0.01, spacing=0.1)
    @assert cluster_size <= n
    eigs = Vector{ComplexF64}(undef, n)
    for i in 1:cluster_size
        θ = 2π * (i - 1) / cluster_size
        eigs[i] = center + cluster_radius * exp(im * θ)
    end
    for i in (cluster_size+1):n
        eigs[i] = center - spacing * (i - cluster_size) + 0.05im * (i - n/2)
    end
    return eigs
end

# ## Result Structure

struct BenchmarkResult
    method::String
    bound::Float64
    time_seconds::Float64
    success::Bool
    precision_bits::Int
    extra::Dict{Symbol, Any}
end

BenchmarkResult(m, b, t, s, p) = BenchmarkResult(m, b, t, s, p, Dict{Symbol,Any}())

# ## Certification Wrappers

function certify_miyajima(A::BallMatrix, z::Number)
    t0 = time()
    try
        result = rigorous_svd(A - z * I)
        bound = 1.0 / Float64(inf(result.σ_min))
        return BenchmarkResult("Miyajima", bound, time() - t0, true, 53)
    catch
        return BenchmarkResult("Miyajima", Inf, time() - t0, false, 53)
    end
end

function certify_ogita(A::BallMatrix, z::Number; iterations=5)
    t0 = time()
    try
        Ashift = A - z * I
        Amid = mid.(Ashift)
        U, S, V = svd(Amid)
        result = ogita_svd_refine(Ashift, U, S, V; max_iterations=iterations)
        if result.certified
            bound = 1.0 / Float64(inf(result.σ_min))
            return BenchmarkResult("Ogita", bound, time() - t0, true, 53)
        else
            return BenchmarkResult("Ogita", Inf, time() - t0, false, 53)
        end
    catch
        return BenchmarkResult("Ogita", Inf, time() - t0, false, 53)
    end
end

function certify_parametric(A::BallMatrix, z::Number, config::ResolventBoundConfig;
                           k::Union{Nothing,Int}=nothing, name::String="")
    t0 = time()
    try
        T = mid.(A)
        n = size(T, 1)
        k_use = isnothing(k) ? max(1, n ÷ 4) : k
        precomp = sylvester_resolvent_precompute(T, k_use)
        R = solve_sylvester_oracle(precomp)
        result = parametric_resolvent_bound(precomp, T, z, config; R=R)
        method_name = isempty(name) ? "Param-$(config.combiner)" : name
        if result.success && isfinite(result.resolvent_bound)
            return BenchmarkResult(method_name, result.resolvent_bound, time() - t0, true, 53)
        else
            return BenchmarkResult(method_name, Inf, time() - t0, false, 53)
        end
    catch
        method_name = isempty(name) ? "Param-$(config.combiner)" : name
        return BenchmarkResult(method_name, Inf, time() - t0, false, 53)
    end
end

function certify_bigfloat(A::BallMatrix, z::Number; precision=128, iterations=10)
    t0 = time()
    try
        setprecision(BigFloat, precision)
        Amid = mid.(A)
        T_bf = convert.(Complex{BigFloat}, Amid)
        z_bf = Complex{BigFloat}(z)
        Ashift_bf = T_bf - z_bf * I
        U64, S64, V64 = svd(Complex{Float64}.(Ashift_bf))
        U_bf = convert.(Complex{BigFloat}, U64)
        S_bf = convert.(BigFloat, S64)
        V_bf = convert.(Complex{BigFloat}, V64)
        A_ball_bf = BallMatrix(Ashift_bf)
        result = ogita_svd_refine(A_ball_bf, U_bf, S_bf, V_bf; max_iterations=iterations)
        if result.certified
            bound = Float64(1.0 / inf(result.σ_min))
            return BenchmarkResult("BigFloat-$precision", bound, time() - t0, true, precision)
        else
            return BenchmarkResult("BigFloat-$precision", Inf, time() - t0, false, precision)
        end
    catch
        return BenchmarkResult("BigFloat-$precision", Inf, time() - t0, false, precision)
    end
end

# ## Test 1: Varying Matrix Size and Nonnormality
#
# We test how methods scale with matrix dimension and nonnormality strength.

function run_test1(; sizes=[20, 40, 60, 80], nonnormalities=[0.1, 0.5, 1.0, 2.0],
                    circle_radius=0.05, verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("TEST 1: Varying matrix size and nonnormality strength")
    verbose && println("="^70)

    results = Dict{Tuple{Int, Float64, String}, BenchmarkResult}()

    for n in sizes
        for nn in nonnormalities
            verbose && println("\n--- n=$n, nonnormality=$nn ---")
            eigs = eigenvalues_single_dominant(n; λ₁=1.0+0.0im, spacing=0.2)
            T = make_test_matrix(n; eigenvalues=eigs, nonnormality=nn)
            A = BallMatrix(T)
            z = 1.0 + circle_radius + 0.0im

            for (method_fn, args) in [
                (certify_miyajima, (A, z)),
                (certify_ogita, (A, z)),
                (() -> certify_parametric(A, z, config_v1(); name="V1"), ()),
                (() -> certify_parametric(A, z, config_v2(); name="V2"), ()),
                (() -> certify_parametric(A, z, config_v2p5(); name="V2.5"), ()),
                (() -> certify_parametric(A, z, config_v3(); name="V3"), ()),
            ]
                r = isempty(args) ? method_fn() : method_fn(args...)
                results[(n, nn, r.method)] = r
                if verbose
                    status = r.success ? "OK" : "FAIL"
                    @printf("  %-15s: bound=%.4e, time=%.4fs [%s]\n",
                            r.method, r.bound, r.time_seconds, status)
                end
            end
        end
    end
    return results
end

# ## Test 2: Varying Circle Radius (Float64 Breakdown)
#
# As the certification radius decreases, Float64 methods eventually fail.
# We compare with BigFloat at various precisions.

function run_test2(; n=50, radii=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8],
                    bigfloat_precisions=[128, 256], verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("TEST 2: Varying circle radius (Float64 breakdown)")
    verbose && println("="^70)

    results = Dict{Tuple{Float64, String}, BenchmarkResult}()
    eigs = eigenvalues_single_dominant(n; λ₁=1.0+0.0im, spacing=0.3)
    T = make_test_matrix(n; eigenvalues=eigs, nonnormality=1.0)
    A = BallMatrix(T)

    for radius in radii
        verbose && println("\n--- radius=$radius ---")
        z = 1.0 + radius + 0.0im

        # Float64 methods
        for method_fn in [certify_miyajima, certify_ogita,
                          (a, z) -> certify_parametric(a, z, config_v2(); name="V2")]
            r = method_fn(A, z)
            results[(radius, r.method)] = r
            if verbose
                status = r.success ? "OK" : "FAIL"
                @printf("  %-20s: bound=%.4e, time=%.4fs [%s]\n",
                        r.method, r.bound, r.time_seconds, status)
            end
        end

        # BigFloat for small radii
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

# ## Test 3: Parametric Configuration Comparison
#
# Compare all four parametric configurations on the same problems.

function run_test3(; sizes=[30, 50, 70], nonnormality=1.0, circle_radius=0.05, verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("TEST 3: Comparing parametric configurations")
    verbose && println("="^70)

    results = Dict{Tuple{Int, String}, BenchmarkResult}()
    configs = [("V1", config_v1()), ("V2", config_v2()),
               ("V2.5", config_v2p5()), ("V3", config_v3())]

    for n in sizes
        verbose && println("\n--- n=$n ---")
        eigs = eigenvalues_single_dominant(n; λ₁=1.0+0.0im, spacing=0.2)
        T = make_test_matrix(n; eigenvalues=eigs, nonnormality=nonnormality)
        A = BallMatrix(T)
        z = 1.0 + circle_radius + 0.0im

        for (name, config) in configs
            r = certify_parametric(A, z, config; name=name)
            results[(n, name)] = r
            if verbose
                status = r.success ? "OK" : "FAIL"
                @printf("  Config %-6s: bound=%.4e, time=%.4fs [%s]\n",
                        name, r.bound, r.time_seconds, status)
            end
        end

        # Also baseline methods
        for method_fn in [certify_miyajima, certify_ogita]
            r = method_fn(A, z)
            results[(n, r.method)] = r
            if verbose
                status = r.success ? "OK" : "FAIL"
                @printf("  %-12s: bound=%.4e, time=%.4fs [%s]\n",
                        r.method, r.bound, r.time_seconds, status)
            end
        end
    end
    return results
end

# ## Test 4: Warm-Start Effectiveness
#
# Measure speedup from reusing SVD between nearby certification points.
# We track cache hits, misses, and fallbacks.

struct WarmStartStats
    times_cold::Vector{Float64}      # Times without warm-start
    times_warm::Vector{Float64}      # Times with warm-start
    cache_hits::Int
    cache_misses::Int
    fallbacks::Int
    speedup::Float64
end

function run_test4(; n=50, num_points=20, circle_radius=0.1, verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("TEST 4: Warm-start effectiveness along contour")
    verbose && println("="^70)

    eigs = eigenvalues_single_dominant(n; λ₁=1.0+0.0im, spacing=0.2)
    T = make_test_matrix(n; eigenvalues=eigs, nonnormality=1.0)
    A = BallMatrix(T)

    center = 1.0 + 0.0im
    θs = range(0, 2π, length=num_points+1)[1:end-1]
    zs = [center + circle_radius * exp(im * θ) for θ in θs]

    # Cold start: recompute SVD each time
    verbose && println("\n--- Without warm-start (cold) ---")
    times_cold = Float64[]
    for z in zs
        t0 = time()
        certify_ogita(A, z)
        push!(times_cold, time() - t0)
    end
    total_cold = sum(times_cold)
    verbose && @printf("  Total: %.4fs, Mean: %.4fs\n", total_cold, mean(times_cold))

    # Warm start: reuse previous SVD
    verbose && println("\n--- With warm-start ---")
    times_warm = Float64[]
    cache_hits = 0
    cache_misses = 0
    fallbacks = 0

    # First point: cold start
    Ashift = A - zs[1] * I
    Amid = mid.(Ashift)
    U, S, V = svd(Amid)
    t0 = time()
    result = ogita_svd_refine(Ashift, U, S, V; max_iterations=5)
    push!(times_warm, time() - t0)
    cache_misses += 1  # First point is always a miss

    # Subsequent points: warm start from previous
    prev_z = zs[1]
    for i in 2:length(zs)
        z = zs[i]
        t0 = time()
        Ashift = A - z * I

        # Check if warm-start is beneficial (points are close)
        dist = abs(z - prev_z)
        if dist < 0.5 * circle_radius
            # Use warm-start: previous U, S, V as starting point
            result_warm = ogita_svd_refine(Ashift, U, S, V; max_iterations=5)
            if result_warm.certified
                cache_hits += 1
                # Note: in full implementation, we'd extract refined U,S,V
            else
                # Warm start failed, fall back to cold
                Amid = mid.(Ashift)
                U, S, V = svd(Amid)
                result_warm = ogita_svd_refine(Ashift, U, S, V; max_iterations=5)
                fallbacks += 1
            end
        else
            # Points too far apart, cold start
            Amid = mid.(Ashift)
            U, S, V = svd(Amid)
            result_warm = ogita_svd_refine(Ashift, U, S, V; max_iterations=5)
            cache_misses += 1
        end
        push!(times_warm, time() - t0)
        prev_z = z
    end

    total_warm = sum(times_warm)
    speedup = total_cold / total_warm
    verbose && @printf("  Total: %.4fs, Mean: %.4fs\n", total_warm, mean(times_warm))
    verbose && println("\n--- Cache Statistics ---")
    verbose && @printf("  Hits: %d, Misses: %d, Fallbacks: %d\n",
                       cache_hits, cache_misses, fallbacks)
    verbose && @printf("  Hit rate: %.1f%%\n", 100.0 * cache_hits / num_points)
    verbose && @printf("  Speedup: %.2fx\n", speedup)

    return WarmStartStats(times_cold, times_warm, cache_hits, cache_misses,
                          fallbacks, speedup)
end

# ## Test 5: Parallel Scaling
#
# Compare serial vs parallel certification with different worker counts.

function run_test5_serial(; n=50, num_points=32, circle_radius=0.1, verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("TEST 5: Parallel scaling (serial baseline)")
    verbose && println("="^70)

    eigs = eigenvalues_single_dominant(n; λ₁=1.0+0.0im, spacing=0.2)
    T = make_test_matrix(n; eigenvalues=eigs, nonnormality=1.0)
    A = BallMatrix(T)

    circle = CertificationCircle(1.0 + 0.0im, circle_radius; samples=num_points)
    log_io = IOBuffer()

    verbose && println("\n--- Serial execution ---")
    t0 = time()
    try
        run_certification(A, circle; η=0.5, log_io=log_io)
    catch e
        verbose && println("  Warning: $e")
    end
    t_serial = time() - t0
    verbose && @printf("  Time: %.4fs\n", t_serial)

    return t_serial
end

# Note: Parallel test requires separate invocation with workers loaded

# ## Run All Benchmarks

function run_all_benchmarks(; verbose=true)
    println("="^70)
    println("RESOLVENT CERTIFICATION BENCHMARK SUITE")
    println("="^70)

    results = Dict{Symbol, Any}()

    results[:test1] = run_test1(; verbose=verbose)
    results[:test2] = run_test2(; verbose=verbose)
    results[:test3] = run_test3(; verbose=verbose)
    results[:test4] = run_test4(; verbose=verbose)
    results[:test5_serial] = run_test5_serial(; verbose=verbose)

    return results
end

# ## Generate LaTeX Tables

function results_to_latex(results; output_file="resolvent_benchmark.tex")
    io = IOBuffer()

    println(io, raw"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{geometry}
\geometry{margin=2.5cm}
\usepackage{xcolor}

\definecolor{okgreen}{RGB}{0,128,0}
\definecolor{failred}{RGB}{200,0,0}

\title{Resolvent Certification Methods: Comparative Benchmark}
\author{BallArithmetic.jl}
\date{\today}

\begin{document}
\maketitle
""")

    # Test 1 table
    if haskey(results, :test1)
        test1 = results[:test1]
        sizes = sort(unique([k[1] for k in keys(test1)]))
        nns = sort(unique([k[2] for k in keys(test1)]))
        methods = unique([k[3] for k in keys(test1)])

        println(io, "\n\\section{Test 1: Size and Nonnormality}\n")
        for method in methods
            println(io, "\\subsection*{$method}")
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
                    if haskey(test1, key)
                        r = test1[key]
                        if r.success
                            @printf(io, " & %.3f", r.time_seconds)
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
    end

    # Test 2 table
    if haskey(results, :test2)
        test2 = results[:test2]
        radii = sort(unique([k[1] for k in keys(test2)]), rev=true)
        methods2 = unique([k[2] for k in keys(test2)])

        println(io, "\n\\section{Test 2: Radius Breakdown}\n")
        println(io, "\\begin{tabular}{l|" * "c"^length(radii) * "}")
        println(io, "\\toprule")
        print(io, "Method")
        for r in radii
            @printf(io, " & \$10^{%d}\$", round(Int, log10(r)))
        end
        println(io, " \\\\\\midrule")
        for method in methods2
            print(io, replace(method, "_" => "\\_"))
            for r in radii
                key = (r, method)
                if haskey(test2, key)
                    res = test2[key]
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
    end

    # Test 4 warm-start
    if haskey(results, :test4)
        ws = results[:test4]
        println(io, "\n\\section{Test 4: Warm-Start Effectiveness}\n")
        println(io, "\\begin{itemize}")
        @printf(io, "\\item Cache hits: %d\n", ws.cache_hits)
        @printf(io, "\\item Cache misses: %d\n", ws.cache_misses)
        @printf(io, "\\item Fallbacks: %d\n", ws.fallbacks)
        @printf(io, "\\item Speedup: %.2f\\texttimes\n", ws.speedup)
        println(io, "\\end{itemize}\n")
    end

    println(io, "\n\\end{document}")

    content = String(take!(io))
    open(output_file, "w") do f
        write(f, content)
    end
    println("LaTeX written to: $output_file")
    return output_file
end

# ## Main Entry Point

if abspath(PROGRAM_FILE) == @__FILE__
    results = run_all_benchmarks()
    results_to_latex(results)
end
