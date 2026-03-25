# # Run All Resolvent Benchmarks
#
# Master script to run all benchmark tests and generate the combined LaTeX report.
#
# Usage:
#   julia run_benchmarks.jl           # Run all tests
#   julia run_benchmarks.jl 1         # Run only test 1
#   julia run_benchmarks.jl 1 3       # Run tests 1 and 3
#   julia -p 4 run_benchmarks.jl 5    # Run test 5 with 4 workers

using Dates

# Determine which tests to run
if isempty(ARGS)
    tests_to_run = [1, 2, 3, 4, 5]
else
    tests_to_run = parse.(Int, ARGS)
end

println("="^70)
println("RESOLVENT CERTIFICATION BENCHMARK SUITE")
println("Date: ", Dates.now())
println("Tests to run: ", tests_to_run)
println("="^70)

# Run selected tests
results = Dict{Int, Any}()

if 1 in tests_to_run
    println("\n>>> Running Test 1...")
    include("benchmark_test1.jl")
    results[1] = run_test1()
    test1_to_latex(results[1])
end

if 2 in tests_to_run
    println("\n>>> Running Test 2...")
    include("benchmark_test2.jl")
    results[2] = run_test2()
    test2_to_latex(results[2])
end

if 3 in tests_to_run
    println("\n>>> Running Test 3...")
    include("benchmark_test3.jl")
    results[3] = run_test3()
    test3_to_latex(results[3])
end

if 4 in tests_to_run
    println("\n>>> Running Test 4...")
    include("benchmark_test4.jl")
    results[4] = run_test4()
    threshold_results = analyze_threshold_sensitivity()
    test4_to_latex(results[4]; threshold_results=threshold_results)
end

if 5 in tests_to_run
    println("\n>>> Running Test 5...")
    include("benchmark_test5.jl")
    using Distributed
    if nworkers() > 1
        results[5] = run_test5()
        test5_to_latex(results[5])
    else
        println("  [Serial only - run with 'julia -p N' for parallel test]")
        results[5] = run_test5_serial_only()
    end
end

# Generate combined report
println("\n" * "="^70)
println("GENERATING COMBINED REPORT")
println("="^70)

function generate_combined_report(results; output_file="resolvent_benchmark_report.tex")
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
\usepackage{hyperref}
\usepackage{xcolor}

\definecolor{okgreen}{RGB}{0,128,0}
\definecolor{failred}{RGB}{200,0,0}

\title{Resolvent Certification Methods:\\A Comparative Study}
\author{BallArithmetic.jl Benchmark Suite}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This report presents a comprehensive comparison of methods for certifying
resolvent bounds $\|(zI - A)^{-1}\|_2$ in rigorous numerical linear algebra.
We compare pure SVD enclosure (Miyajima), Newton-refinement (Ogita),
and parametric Sylvester-based methods across varying matrix sizes,
nonnormality strengths, and certification radii. We also analyze
warm-start caching and parallel scaling efficiency.
\end{abstract}

\tableofcontents
\newpage

\section{Introduction}

Rigorous resolvent bounds are essential for:
\begin{itemize}
    \item Pseudospectra computation
    \item Eigenvalue sensitivity analysis
    \item Stability verification of dynamical systems
    \item Verified numerical computation
\end{itemize}

This benchmark compares four families of methods:
\begin{enumerate}
    \item \textbf{Miyajima SVD}: Direct singular value enclosure
    \item \textbf{Ogita refinement}: Newton iteration with Miyajima certification
    \item \textbf{Parametric Sylvester}: Block-diagonalization with various estimators
    \item \textbf{BigFloat}: Extended precision for small radii
\end{enumerate}

\section{Test Results}

Individual test results are in separate files:
\begin{itemize}
    \item \texttt{benchmark\_test1.tex}: Size and nonnormality scaling
    \item \texttt{benchmark\_test2.tex}: Radius breakdown analysis
    \item \texttt{benchmark\_test3.tex}: Parametric configuration comparison
    \item \texttt{benchmark\_test4.tex}: Warm-start effectiveness
    \item \texttt{benchmark\_test5.tex}: Parallel scaling
\end{itemize}

\section{Conclusions}

[To be filled based on test results]

\end{document}
""")

    content = String(take!(io))
    open(output_file, "w") do f
        write(f, content)
    end
    println("Combined report written to: $output_file")
end

generate_combined_report(results)

println("\n" * "="^70)
println("BENCHMARK COMPLETE")
println("="^70)
println("Results: ", keys(results))
