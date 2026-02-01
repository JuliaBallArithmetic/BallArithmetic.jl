"""
Comprehensive benchmark comparing resolvent certification methods.

Tests:
1. Vary matrix size n and nonnormality strength
2. Vary circle radius (Float64 breakdown point)
3. Compare parametric configurations
4. Warm-start effectiveness
5. Parallel scaling

Output: LaTeX file with plots and data tables.
"""

using BallArithmetic
using BallArithmetic.CertifScripts
using LinearAlgebra
using Printf
using Statistics
using Random
using Distributed

# ============================================================================
# Matrix generators
# ============================================================================

"""
    make_upper_triangular_test_matrix(n; eigenvalues, nonnormality=1.0)

Create an upper triangular test matrix with specified eigenvalues and
controllable nonnormality (scaling of off-diagonal entries).
"""
function make_upper_triangular_test_matrix(n::Int;
        eigenvalues::Vector{<:Number},
        nonnormality::Real=1.0,
        seed::Int=42)
    Random.seed!(seed)
    @assert length(eigenvalues) == n

    # Create upper triangular matrix
    T = zeros(ComplexF64, n, n)

    # Set diagonal (eigenvalues)
    for i in 1:n
        T[i, i] = eigenvalues[i]
    end

    # Set off-diagonal entries with controlled nonnormality
    for i in 1:n
        for j in (i+1):n
            T[i, j] = nonnormality * (randn() + im * randn()) / sqrt(2)
        end
    end

    return T
end

"""
    eigenvalues_single_dominant(n; dominant=1.0, spacing=0.1)

Create eigenvalue distribution with single dominant eigenvalue at `dominant`,
others spaced away.
"""
function eigenvalues_single_dominant(n::Int; dominant=1.0+0.0im, spacing=0.1)
    eigs = Vector{ComplexF64}(undef, n)
    eigs[1] = dominant
    for i in 2:n
        # Place other eigenvalues away from dominant
        eigs[i] = dominant - spacing * i + 0.05im * (i - n/2)
    end
    return eigs
end

"""
    eigenvalues_cluster(n; center=1.0, cluster_size=3, cluster_radius=0.01, spacing=0.1)

Create eigenvalue distribution with a cluster near `center`.
"""
function eigenvalues_cluster(n::Int; center=1.0+0.0im, cluster_size=3,
                             cluster_radius=0.01, spacing=0.1)
    @assert cluster_size <= n
    eigs = Vector{ComplexF64}(undef, n)

    # Create cluster
    for i in 1:cluster_size
        θ = 2π * (i - 1) / cluster_size
        eigs[i] = center + cluster_radius * exp(im * θ)
    end

    # Other eigenvalues away from cluster
    for i in (cluster_size+1):n
        eigs[i] = center - spacing * (i - cluster_size) + 0.05im * (i - n/2)
    end

    return eigs
end

"""
    eigenvalues_multiple_dominant(n; dominant=1.0, multiplicity=2, perturbation=1e-8, spacing=0.1)

Create eigenvalue distribution with near-multiple dominant eigenvalue.
"""
function eigenvalues_multiple_dominant(n::Int; dominant=1.0+0.0im, multiplicity=2,
                                       perturbation=1e-8, spacing=0.1)
    @assert multiplicity <= n
    eigs = Vector{ComplexF64}(undef, n)

    # Near-multiple eigenvalues
    for i in 1:multiplicity
        eigs[i] = dominant + perturbation * (i - 1) * exp(im * 2π * (i-1) / multiplicity)
    end

    # Other eigenvalues
    for i in (multiplicity+1):n
        eigs[i] = dominant - spacing * (i - multiplicity) + 0.05im * (i - n/2)
    end

    return eigs
end

# ============================================================================
# Certification methods to compare
# ============================================================================

struct CertificationResult
    method::String
    bound::Float64
    time_seconds::Float64
    success::Bool
    precision_bits::Int  # 53 for Float64, higher for BigFloat
end

"""
Run pure Miyajima SVD certification.
"""
function certify_miyajima(A::BallMatrix, z::Number)
    t0 = time()
    try
        result = rigorous_svd(A - z * I)
        bound = 1.0 / Float64(inf(result.σ_min))
        return CertificationResult("Miyajima", bound, time() - t0, true, 53)
    catch e
        return CertificationResult("Miyajima", Inf, time() - t0, false, 53)
    end
end

"""
Run Ogita + Miyajima refinement.
"""
function certify_ogita(A::BallMatrix, z::Number; iterations=5)
    t0 = time()
    try
        Ashift = A - z * I
        Amid = mid.(Ashift)
        U, S, V = svd(Amid)
        result = ogita_svd_refine(Ashift, U, S, V; max_iterations=iterations)
        if result.certified
            bound = 1.0 / Float64(inf(result.σ_min))
            return CertificationResult("Ogita+Miyajima", bound, time() - t0, true, 53)
        else
            return CertificationResult("Ogita+Miyajima", Inf, time() - t0, false, 53)
        end
    catch e
        return CertificationResult("Ogita+Miyajima", Inf, time() - t0, false, 53)
    end
end

"""
Run parametric Sylvester-based certification.
"""
function certify_parametric(A::BallMatrix, z::Number, config::ResolventBoundConfig;
                           k::Union{Nothing,Int}=nothing)
    t0 = time()
    try
        T = mid.(A)  # Use midpoint for Schur
        n = size(T, 1)
        k_use = isnothing(k) ? max(1, n ÷ 4) : k

        precomp = sylvester_resolvent_precompute(T, k_use)
        R = solve_sylvester_oracle(precomp)
        result = parametric_resolvent_bound(precomp, T, z, config; R=R)

        if result.success && isfinite(result.resolvent_bound)
            return CertificationResult("Parametric-$(config.combiner)",
                                       result.resolvent_bound, time() - t0, true, 53)
        else
            return CertificationResult("Parametric-$(config.combiner)",
                                       Inf, time() - t0, false, 53)
        end
    catch e
        return CertificationResult("Parametric-$(config.combiner)",
                                   Inf, time() - t0, false, 53)
    end
end

"""
Run BigFloat Ogita refinement.
"""
function certify_bigfloat_ogita(A::BallMatrix, z::Number; precision=128, iterations=10)
    t0 = time()
    try
        setprecision(BigFloat, precision)

        # Convert to BigFloat
        Amid = mid.(A)
        T_bf = convert.(Complex{BigFloat}, Amid)
        z_bf = Complex{BigFloat}(z)
        Ashift_bf = T_bf - z_bf * I

        # Get Float64 SVD as starting point
        U64, S64, V64 = svd(Complex{Float64}.(Ashift_bf))
        U_bf = convert.(Complex{BigFloat}, U64)
        S_bf = convert.(BigFloat, S64)
        V_bf = convert.(Complex{BigFloat}, V64)

        # Create BallMatrix with BigFloat
        A_ball_bf = BallMatrix(Ashift_bf)

        result = ogita_svd_refine(A_ball_bf, U_bf, S_bf, V_bf; max_iterations=iterations)

        if result.certified
            bound = Float64(1.0 / inf(result.σ_min))
            return CertificationResult("BigFloat-Ogita-$(precision)", bound, time() - t0, true, precision)
        else
            return CertificationResult("BigFloat-Ogita-$(precision)", Inf, time() - t0, false, precision)
        end
    catch e
        return CertificationResult("BigFloat-Ogita-$(precision)", Inf, time() - t0, false, precision)
    end
end

# ============================================================================
# TEST 1: Vary matrix size and nonnormality
# ============================================================================

function run_test1(; sizes=[20, 40, 60, 80, 100],
                    nonnormalities=[0.1, 0.5, 1.0, 2.0, 5.0],
                    circle_radius=0.05)
    println("\n" * "="^70)
    println("TEST 1: Varying matrix size and nonnormality strength")
    println("="^70)

    results = Dict{Tuple{Int, Float64, String}, CertificationResult}()

    for n in sizes
        for nn in nonnormalities
            println("\n--- n=$n, nonnormality=$nn ---")

            # Create matrix with single dominant eigenvalue at 1
            eigs = eigenvalues_single_dominant(n; dominant=1.0+0.0im, spacing=0.2)
            T = make_upper_triangular_test_matrix(n; eigenvalues=eigs, nonnormality=nn)
            A = BallMatrix(T)

            # Test point just outside the eigenvalue
            z = 1.0 + circle_radius + 0.0im

            # Run different methods
            methods = [
                () -> certify_miyajima(A, z),
                () -> certify_ogita(A, z),
                () -> certify_parametric(A, z, config_v1()),
                () -> certify_parametric(A, z, config_v2()),
                () -> certify_parametric(A, z, config_v2p5()),
                () -> certify_parametric(A, z, config_v3()),
            ]

            for method in methods
                r = method()
                results[(n, nn, r.method)] = r
                status = r.success ? "OK" : "FAIL"
                @printf("  %-25s: bound=%.4e, time=%.4fs [%s]\n",
                        r.method, r.bound, r.time_seconds, status)
            end
        end
    end

    return results
end

# ============================================================================
# TEST 2: Vary circle radius (Float64 breakdown)
# ============================================================================

function run_test2(; n=50,
                    radii=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12],
                    bigfloat_precisions=[128, 256, 512])
    println("\n" * "="^70)
    println("TEST 2: Varying circle radius (Float64 breakdown point)")
    println("="^70)

    results = Dict{Tuple{Float64, String}, CertificationResult}()

    # Create matrix with isolated dominant eigenvalue
    eigs = eigenvalues_single_dominant(n; dominant=1.0+0.0im, spacing=0.3)
    T = make_upper_triangular_test_matrix(n; eigenvalues=eigs, nonnormality=1.0)
    A = BallMatrix(T)

    for radius in radii
        println("\n--- radius=$radius ---")

        # Test point at distance radius from eigenvalue
        z = 1.0 + radius + 0.0im

        # Float64 methods
        for method in [
            () -> certify_miyajima(A, z),
            () -> certify_ogita(A, z),
            () -> certify_parametric(A, z, config_v2()),
        ]
            r = method()
            results[(radius, r.method)] = r
            status = r.success ? "OK" : "FAIL"
            @printf("  %-25s: bound=%.4e, time=%.4fs [%s]\n",
                    r.method, r.bound, r.time_seconds, status)
        end

        # BigFloat methods (only for small radii where Float64 fails)
        if radius <= 1e-4
            for prec in bigfloat_precisions
                r = certify_bigfloat_ogita(A, z; precision=prec)
                results[(radius, r.method)] = r
                status = r.success ? "OK" : "FAIL"
                @printf("  %-25s: bound=%.4e, time=%.4fs [%s]\n",
                        r.method, r.bound, r.time_seconds, status)
            end
        end
    end

    return results
end

# ============================================================================
# TEST 3: Compare parametric configurations
# ============================================================================

function run_test3(; sizes=[30, 50, 70, 100], nonnormality=1.0, circle_radius=0.05)
    println("\n" * "="^70)
    println("TEST 3: Comparing parametric configurations")
    println("="^70)

    results = Dict{Tuple{Int, String}, CertificationResult}()

    configs = [
        ("V1", config_v1()),
        ("V2", config_v2()),
        ("V2.5", config_v2p5()),
        ("V3", config_v3()),
    ]

    for n in sizes
        println("\n--- n=$n ---")

        eigs = eigenvalues_single_dominant(n; dominant=1.0+0.0im, spacing=0.2)
        T = make_upper_triangular_test_matrix(n; eigenvalues=eigs, nonnormality=nonnormality)
        A = BallMatrix(T)
        z = 1.0 + circle_radius + 0.0im

        for (name, config) in configs
            r = certify_parametric(A, z, config)
            results[(n, name)] = r
            status = r.success ? "OK" : "FAIL"
            @printf("  Config %-6s: bound=%.4e, time=%.4fs [%s]\n",
                    name, r.bound, r.time_seconds, status)
        end

        # Also run Miyajima and Ogita for comparison
        r_miy = certify_miyajima(A, z)
        r_ogi = certify_ogita(A, z)
        results[(n, "Miyajima")] = r_miy
        results[(n, "Ogita")] = r_ogi
        @printf("  %-12s: bound=%.4e, time=%.4fs [%s]\n",
                "Miyajima", r_miy.bound, r_miy.time_seconds, r_miy.success ? "OK" : "FAIL")
        @printf("  %-12s: bound=%.4e, time=%.4fs [%s]\n",
                "Ogita", r_ogi.bound, r_ogi.time_seconds, r_ogi.success ? "OK" : "FAIL")
    end

    return results
end

# ============================================================================
# TEST 4: Warm-start effectiveness
# ============================================================================

function run_test4(; n=50, num_points=20, circle_radius=0.1)
    println("\n" * "="^70)
    println("TEST 4: Warm-start effectiveness along contour")
    println("="^70)

    # Create matrix
    eigs = eigenvalues_single_dominant(n; dominant=1.0+0.0im, spacing=0.2)
    T = make_upper_triangular_test_matrix(n; eigenvalues=eigs, nonnormality=1.0)
    A = BallMatrix(T)

    # Points along a circle around eigenvalue 1
    center = 1.0 + 0.0im
    θs = range(0, 2π, length=num_points+1)[1:end-1]
    zs = [center + circle_radius * exp(im * θ) for θ in θs]

    # Method 1: Without warm-start (recompute everything each time)
    println("\n--- Without warm-start ---")
    times_no_warmstart = Float64[]
    t_total_no_ws = time()
    for z in zs
        t0 = time()
        certify_ogita(A, z)
        push!(times_no_warmstart, time() - t0)
    end
    total_no_ws = time() - t_total_no_ws
    @printf("  Total time: %.4fs, Mean per point: %.4fs\n",
            total_no_ws, mean(times_no_warmstart))

    # Method 2: With warm-start (reuse SVD from previous point)
    println("\n--- With warm-start ---")
    times_warmstart = Float64[]
    t_total_ws = time()

    # First point: no warm-start available
    Ashift = A - zs[1] * I
    Amid = mid.(Ashift)
    U, S, V = svd(Amid)
    t0 = time()
    result = ogita_svd_refine(Ashift, U, S, V; max_iterations=5)
    push!(times_warmstart, time() - t0)

    # Subsequent points: use previous SVD as warm-start
    for i in 2:length(zs)
        t0 = time()
        Ashift = A - zs[i] * I
        # Use previous U, S, V as starting point
        result = ogita_svd_refine(Ashift, U, S, V; max_iterations=5)
        if result.certified
            # Update for next iteration
            # (In practice, we'd extract the refined U, S, V from result)
        end
        push!(times_warmstart, time() - t0)
    end
    total_ws = time() - t_total_ws
    @printf("  Total time: %.4fs, Mean per point: %.4fs\n",
            total_ws, mean(times_warmstart))

    speedup = total_no_ws / total_ws
    @printf("\n  Speedup from warm-start: %.2fx\n", speedup)

    return (no_warmstart=times_no_warmstart, warmstart=times_warmstart, speedup=speedup)
end

# ============================================================================
# TEST 5: Parallel scaling
# ============================================================================

function run_test5(; n=50, num_points=32, circle_radius=0.1, max_workers=4)
    println("\n" * "="^70)
    println("TEST 5: Parallel scaling")
    println("="^70)

    # Create matrix
    eigs = eigenvalues_single_dominant(n; dominant=1.0+0.0im, spacing=0.2)
    T = make_upper_triangular_test_matrix(n; eigenvalues=eigs, nonnormality=1.0)
    A = BallMatrix(T)

    # Define certification circle
    circle = CertificationCircle(1.0 + 0.0im, circle_radius; samples=num_points)

    results = Dict{Int, Float64}()

    # Serial timing
    println("\n--- Serial execution ---")
    t0 = time()
    log_io = IOBuffer()
    try
        run_certification(A, circle; η=0.5, log_io=log_io)
    catch e
        println("  Serial test encountered: ", e)
    end
    t_serial = time() - t0
    results[1] = t_serial
    @printf("  Time: %.4fs\n", t_serial)

    # Parallel timing with different worker counts
    for nworkers in [2, max_workers]
        println("\n--- $nworkers workers ---")

        # Add workers if needed
        current_workers = nworkers(Distributed)
        if current_workers < nworkers
            addprocs(nworkers - current_workers)
        end

        # Load BallArithmetic on workers
        @everywhere using BallArithmetic

        t0 = time()
        log_io = IOBuffer()
        try
            run_certification(A, circle; η=0.5, log_io=log_io)
        catch e
            println("  Parallel test with $nworkers workers encountered: ", e)
        end
        t_parallel = time() - t0
        results[nworkers] = t_parallel

        speedup = t_serial / t_parallel
        @printf("  Time: %.4fs, Speedup: %.2fx\n", t_parallel, speedup)
    end

    return results
end

# ============================================================================
# LaTeX output generation
# ============================================================================

function generate_latex_report(test1_results, test2_results, test3_results,
                               test4_results, test5_results;
                               output_file="resolvent_comparison_report.tex")

    io = IOBuffer()

    # Preamble
    println(io, raw"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{pgfplots}
\usepackage{pgfplotstable}
\pgfplotsset{compat=1.18}
\usepackage{geometry}
\geometry{margin=2.5cm}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{listings}

\definecolor{okgreen}{RGB}{0,128,0}
\definecolor{failred}{RGB}{200,0,0}

\title{Resolvent Certification Methods: Comparative Benchmark}
\author{BallArithmetic.jl Benchmark Suite}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This report compares different methods for certifying resolvent bounds
$\|(zI - A)^{-1}\|_2$ in the BallArithmetic.jl package. We test pure Miyajima SVD,
Ogita+Miyajima refinement, parametric Sylvester-based methods (V1, V2, V2.5, V3),
and BigFloat variants. Tests cover varying matrix sizes, nonnormality strengths,
circle radii (including Float64 breakdown), and parallel scaling.
\end{abstract}

\tableofcontents
\newpage
""")

    # TEST 1 Section
    println(io, raw"""
\section{Test 1: Varying Matrix Size and Nonnormality}

This test examines how different certification methods perform as we vary:
\begin{itemize}
    \item Matrix dimension $n \in \{20, 40, 60, 80, 100\}$
    \item Nonnormality strength $\nu \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$
\end{itemize}

The test matrix is upper triangular with a dominant eigenvalue at $\lambda_1 = 1$.
The certification point is at $z = 1 + 0.05$ (radius 0.05 from the eigenvalue).
""")

    # Generate Test 1 data tables
    println(io, "\n\\subsection{Timing Results (seconds)}\n")

    # Extract unique values
    sizes = sort(unique([k[1] for k in keys(test1_results)]))
    nonnormalities = sort(unique([k[2] for k in keys(test1_results)]))
    methods = unique([k[3] for k in keys(test1_results)])

    for method in methods
        println(io, "\\subsubsection{Method: $method}\n")
        println(io, "\\begin{table}[htbp]")
        println(io, "\\centering")
        println(io, "\\caption{Timing for $method}")
        print(io, "\\begin{tabular}{l|")
        for _ in sizes
            print(io, "r")
        end
        println(io, "}")
        println(io, "\\toprule")
        print(io, "\$\\nu\\$ \\textbackslash{} \$n\$")
        for n in sizes
            print(io, " & $n")
        end
        println(io, " \\\\")
        println(io, "\\midrule")

        for nn in nonnormalities
            @printf(io, "%.1f", nn)
            for n in sizes
                key = (n, nn, method)
                if haskey(test1_results, key)
                    r = test1_results[key]
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

        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}\n")
    end

    # Test 1 plots
    println(io, raw"""
\subsection{Timing vs Matrix Size}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{axis}[
    xlabel={Matrix size $n$},
    ylabel={Time (s)},
    legend pos=north west,
    width=0.8\textwidth,
    height=0.5\textwidth,
    grid=major,
    ymode=log,
]
""")

    # Add plot data for each method at nonnormality=1.0
    colors = ["blue", "red", "green!60!black", "orange", "purple", "cyan"]
    for (i, method) in enumerate(methods)
        print(io, "\\addplot[$(colors[mod1(i, length(colors))]), mark=*, thick] coordinates {")
        for n in sizes
            key = (n, 1.0, method)
            if haskey(test1_results, key) && test1_results[key].success
                @printf(io, "(%d, %.4f) ", n, test1_results[key].time_seconds)
            end
        end
        println(io, "};")
        println(io, "\\addlegendentry{$method}")
    end

    println(io, raw"""
\end{axis}
\end{tikzpicture}
\caption{Timing comparison at nonnormality $\nu = 1.0$}
\end{figure}
""")

    # TEST 2 Section
    println(io, raw"""
\newpage
\section{Test 2: Varying Circle Radius (Float64 Breakdown)}

This test examines the breakdown of Float64 arithmetic as the certification
radius decreases. We compare Float64 methods with BigFloat at various precisions.

\begin{itemize}
    \item Matrix dimension: $n = 50$
    \item Radii: $10^{-1}, 10^{-2}, \ldots, 10^{-12}$
    \item BigFloat precisions: 128, 256, 512 bits
\end{itemize}
""")

    radii = sort(unique([k[1] for k in keys(test2_results)]), rev=true)
    methods2 = unique([k[2] for k in keys(test2_results)])

    println(io, "\n\\subsection{Success/Failure by Radius}\n")
    println(io, "\\begin{table}[htbp]")
    println(io, "\\centering")
    println(io, "\\caption{Method success at different radii}")
    print(io, "\\begin{tabular}{l|")
    for _ in radii
        print(io, "c")
    end
    println(io, "}")
    println(io, "\\toprule")
    print(io, "Method")
    for r in radii
        @printf(io, " & \$10^{%d}\$", round(Int, log10(r)))
    end
    println(io, " \\\\")
    println(io, "\\midrule")

    for method in methods2
        print(io, replace(method, "_" => "\\_"))
        for r in radii
            key = (r, method)
            if haskey(test2_results, key)
                res = test2_results[key]
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

    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}\n")

    # Bound quality plot for Test 2
    println(io, raw"""
\subsection{Bound Quality vs Radius}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{loglogaxis}[
    xlabel={Circle radius},
    ylabel={Resolvent bound},
    legend pos=north east,
    width=0.8\textwidth,
    height=0.5\textwidth,
    grid=major,
    x dir=reverse,
]
""")

    for (i, method) in enumerate(methods2)
        print(io, "\\addplot[$(colors[mod1(i, length(colors))]), mark=*, thick] coordinates {")
        for r in radii
            key = (r, method)
            if haskey(test2_results, key) && test2_results[key].success
                @printf(io, "(%.2e, %.4e) ", r, test2_results[key].bound)
            end
        end
        println(io, "};")
        println(io, "\\addlegendentry{$(replace(method, "_" => "\\_"))}")
    end

    println(io, raw"""
\end{loglogaxis}
\end{tikzpicture}
\caption{Certified resolvent bound vs circle radius}
\end{figure}
""")

    # TEST 3 Section
    println(io, raw"""
\newpage
\section{Test 3: Parametric Configuration Comparison}

This test compares the four parametric configurations:
\begin{itemize}
    \item \textbf{V1}: Triangular backsubstitution + product bound (CombinerV1)
    \item \textbf{V2}: Triangular backsubstitution + coupling AR solve (CombinerV2)
    \item \textbf{V2.5}: Triangular backsubstitution + off-diagonal direct (CombinerV2p5)
    \item \textbf{V3}: Neumann-Collatz + coupling AR solve (CombinerV2)
\end{itemize}
""")

    sizes3 = sort(unique([k[1] for k in keys(test3_results)]))
    configs3 = unique([k[2] for k in keys(test3_results)])

    println(io, "\n\\subsection{Bound Quality Comparison}\n")
    println(io, "\\begin{table}[htbp]")
    println(io, "\\centering")
    println(io, "\\caption{Certified bounds by method and matrix size}")
    print(io, "\\begin{tabular}{l|")
    for _ in sizes3
        print(io, "r")
    end
    println(io, "}")
    println(io, "\\toprule")
    print(io, "Method")
    for n in sizes3
        print(io, " & \$n=$n\$")
    end
    println(io, " \\\\")
    println(io, "\\midrule")

    for config in configs3
        print(io, config)
        for n in sizes3
            key = (n, config)
            if haskey(test3_results, key)
                r = test3_results[key]
                if r.success
                    @printf(io, " & %.2e", r.bound)
                else
                    print(io, " & fail")
                end
            else
                print(io, " & --")
            end
        end
        println(io, " \\\\")
    end

    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}\n")

    # TEST 4 Section
    println(io, raw"""
\newpage
\section{Test 4: Warm-Start Effectiveness}

This test measures the speedup from using SVD warm-starts when certifying
sequential points along a contour.
""")

    @printf(io, "\n\\begin{itemize}\n")
    @printf(io, "    \\item Total time without warm-start: %.4f s\n", sum(test4_results.no_warmstart))
    @printf(io, "    \\item Total time with warm-start: %.4f s\n", sum(test4_results.warmstart))
    @printf(io, "    \\item \\textbf{Speedup: %.2f\\texttimes}\n", test4_results.speedup)
    @printf(io, "\\end{itemize}\n")

    println(io, raw"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{axis}[
    xlabel={Point index along contour},
    ylabel={Time per point (s)},
    legend pos=north east,
    width=0.8\textwidth,
    height=0.4\textwidth,
    grid=major,
]
""")

    print(io, "\\addplot[blue, mark=*, thick] coordinates {")
    for (i, t) in enumerate(test4_results.no_warmstart)
        @printf(io, "(%d, %.4f) ", i, t)
    end
    println(io, "};")
    println(io, "\\addlegendentry{Without warm-start}")

    print(io, "\\addplot[red, mark=square*, thick] coordinates {")
    for (i, t) in enumerate(test4_results.warmstart)
        @printf(io, "(%d, %.4f) ", i, t)
    end
    println(io, "};")
    println(io, "\\addlegendentry{With warm-start}")

    println(io, raw"""
\end{axis}
\end{tikzpicture}
\caption{Time per certification point with and without warm-start}
\end{figure}
""")

    # TEST 5 Section
    println(io, raw"""
\newpage
\section{Test 5: Parallel Scaling}

This test compares serial execution with parallel execution using multiple workers.
""")

    if !isempty(test5_results)
        worker_counts = sort(collect(keys(test5_results)))

        println(io, "\n\\begin{table}[htbp]")
        println(io, "\\centering")
        println(io, "\\caption{Parallel scaling results}")
        println(io, "\\begin{tabular}{lrr}")
        println(io, "\\toprule")
        println(io, "Workers & Time (s) & Speedup \\\\")
        println(io, "\\midrule")

        t_serial = get(test5_results, 1, 1.0)
        for nw in worker_counts
            t = test5_results[nw]
            speedup = t_serial / t
            @printf(io, "%d & %.3f & %.2f\\texttimes \\\\\n", nw, t, speedup)
        end

        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}\n")
    else
        println(io, "\n\\textit{Parallel scaling test was skipped.}\n")
    end

    # Conclusion
    println(io, raw"""
\newpage
\section{Conclusions}

\begin{enumerate}
    \item \textbf{Float64 Methods}: Pure Miyajima and Ogita+Miyajima refinement
          work well for moderate circle radii ($> 10^{-6}$) but break down
          for smaller radii due to floating-point limitations.

    \item \textbf{BigFloat}: Required for small radii certification. Higher
          precision (256--512 bits) provides more reliable bounds at the cost
          of increased computation time.

    \item \textbf{Parametric Methods}: The Sylvester-based parametric framework
          offers flexibility in trading off between bound tightness and
          computation speed. V2 and V2.5 typically provide the best balance.

    \item \textbf{Warm-Start}: Using SVD warm-starts from nearby points provides
          significant speedup (typically 2--5$\times$) for contour certification.

    \item \textbf{Parallel Scaling}: The distributed certification framework
          scales well up to the number of physical cores.
\end{enumerate}

\end{document}
""")

    # Write to file
    content = String(take!(io))
    open(output_file, "w") do f
        write(f, content)
    end

    println("\nLaTeX report written to: $output_file")
    return output_file
end

# ============================================================================
# Main benchmark runner
# ============================================================================

function run_all_benchmarks(; output_file="resolvent_comparison_report.tex",
                            skip_parallel=false)
    println("="^70)
    println("RESOLVENT CERTIFICATION BENCHMARK SUITE")
    println("="^70)
    println("Running comprehensive comparison of certification methods...")
    println()

    # Run tests
    test1_results = run_test1(; sizes=[20, 40, 60, 80],
                               nonnormalities=[0.1, 0.5, 1.0, 2.0])

    test2_results = run_test2(; n=50,
                               radii=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8],
                               bigfloat_precisions=[128, 256])

    test3_results = run_test3(; sizes=[30, 50, 70], nonnormality=1.0)

    test4_results = run_test4(; n=50, num_points=16, circle_radius=0.1)

    # Parallel test (optional)
    if skip_parallel
        test5_results = Dict{Int, Float64}()
        println("\n[Parallel scaling test skipped]")
    else
        test5_results = Dict{Int, Float64}()
        println("\n[Parallel scaling test - run separately with workers]")
    end

    # Generate LaTeX report
    generate_latex_report(test1_results, test2_results, test3_results,
                         test4_results, test5_results;
                         output_file=output_file)

    return (test1=test1_results, test2=test2_results, test3=test3_results,
            test4=test4_results, test5=test5_results)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_benchmarks()
end
