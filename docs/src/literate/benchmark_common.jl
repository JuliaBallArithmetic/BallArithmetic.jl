# # Common Utilities for Resolvent Benchmarks
#
# This file provides shared matrix generators, result structures, and
# certification wrappers used across all benchmark tests.

using BallArithmetic
using BallArithmetic.CertifScripts
using LinearAlgebra
using Printf
using Statistics
using Random

# ## Matrix Generators

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
    certified::Bool
    precision_bits::Int
    extra::Dict{Symbol, Any}
end

BenchmarkResult(m, b, t, s, p) = BenchmarkResult(m, b, t, s, p, Dict{Symbol,Any}())

# ## Certification Wrappers

function certify_miyajima(A::BallMatrix, z::Number)
    t0 = time()
    try
        result = rigorous_svd(A - z * I)
        # Singular values are sorted descending, so last is smallest
        σ_min = result.singular_values[end]
        bound = 1.0 / Float64(inf(σ_min))
        return BenchmarkResult("Miyajima", bound, time() - t0, true, 53)
    catch e
        return BenchmarkResult("Miyajima", Inf, time() - t0, false, 53)
    end
end

function certify_ogita(A::BallMatrix, z::Number; iterations=5)
    t0 = time()
    try
        Amid = mid.(A) - z * I  # Regular matrix (not BallMatrix)
        U, S, V = svd(Amid)
        result = ogita_svd_refine(Amid, U, S, V; max_iterations=iterations)
        if result.converged
            # Σ is a Diagonal{BigFloat}, residual_norm gives the error bound
            σ_min_approx = result.Σ[end, end]
            # Rigorous lower bound: σ_min ≥ σ_min_approx - residual_norm
            σ_min_lower = σ_min_approx - result.residual_norm
            if σ_min_lower > 0
                bound = 1.0 / Float64(σ_min_lower)
                return BenchmarkResult("Ogita", bound, time() - t0, true, 53)
            else
                return BenchmarkResult("Ogita", Inf, time() - t0, false, 53)
            end
        else
            return BenchmarkResult("Ogita", Inf, time() - t0, false, 53)
        end
    catch e
        return BenchmarkResult("Ogita", Inf, time() - t0, false, 53)
    end
end

function certify_parametric(A::BallMatrix, z::Number, config::ResolventBoundConfig;
                           k::Union{Nothing,Int}=nothing, name::String="")
    t0 = time()
    method_name = isempty(name) ? "Param-$(config.coupling_estimator)" : name
    try
        T = mid.(A)
        n = size(T, 1)
        k_use = isnothing(k) ? max(1, n ÷ 4) : k
        # Use convenience function that handles precomputation
        _, result = parametric_resolvent_bound(T, k_use, Complex(z), config)
        if result.certified && isfinite(result.resolvent_bound)
            return BenchmarkResult(method_name, result.resolvent_bound, time() - t0, true, 53)
        else
            return BenchmarkResult(method_name, Inf, time() - t0, false, 53)
        end
    catch e
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
        result = ogita_svd_refine(Ashift_bf, U_bf, S_bf, V_bf;
                                  max_iterations=iterations, precision_bits=precision)
        if result.converged
            σ_min_approx = result.Σ[end, end]
            σ_min_lower = σ_min_approx - result.residual_norm
            if σ_min_lower > 0
                bound = Float64(1.0 / σ_min_lower)
                return BenchmarkResult("BigFloat-$precision", bound, time() - t0, true, precision)
            else
                return BenchmarkResult("BigFloat-$precision", Inf, time() - t0, false, precision)
            end
        else
            return BenchmarkResult("BigFloat-$precision", Inf, time() - t0, false, precision)
        end
    catch e
        return BenchmarkResult("BigFloat-$precision", Inf, time() - t0, false, precision)
    end
end

# ## LaTeX helpers

function format_result_table(io::IO, data::Dict, row_key::Function, col_key::Function,
                             value_fn::Function; title::String="")
    rows = sort(unique([row_key(k) for k in keys(data)]))
    cols = sort(unique([col_key(k) for k in keys(data)]))

    !isempty(title) && println(io, "\\subsection*{$title}")
    println(io, "\\begin{tabular}{l|" * "r"^length(cols) * "}")
    println(io, "\\toprule")
    print(io, " ")
    for c in cols
        print(io, " & $c")
    end
    println(io, " \\\\\\midrule")

    for r in rows
        print(io, r)
        for c in cols
            val = value_fn(data, r, c)
            print(io, " & $val")
        end
        println(io, " \\\\")
    end
    println(io, "\\bottomrule\\end{tabular}\n")
end
