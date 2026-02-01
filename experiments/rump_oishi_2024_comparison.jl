# Comparison experiment: SVD-based vs Rump-Oishi 2024 Schur complement method
# for pseudospectra certification
#
# This script compares methods for computing rigorous lower bounds on
# the minimum singular value of (T - zI) where T is upper triangular (Schur form).
#
# Methods compared:
# 1. Full SVD (current approach) - rigorous_svd()
# 2. Rump-Oishi 2024 triangular bound - rump_oishi_2024_triangular_bound()
# 3. Backward singular value bound - backward_singular_value_bound()
# 4. Schur complement method (on real representation) - rump_oishi_2024_sigma_min_bound()

using BallArithmetic
using LinearAlgebra
using Printf
using Statistics
using Random

#==============================================================================#
# Schur reordering utilities
#==============================================================================#

"""
    reorder_schur_by_distance(T::Matrix, z::Complex)

Reorder a Schur matrix T so that eigenvalues are sorted by increasing distance
to z. This puts the closest eigenvalue in the top-left corner, which is optimal
for the Rump-Oishi 2024 method.

Returns (T_reordered, Z) where T_reordered = Z' * T * Z.
"""
function reorder_schur_by_distance(T::Matrix{CT}, z::Complex) where {CT}
    n = size(T, 1)
    T_work = copy(T)
    Z = Matrix{CT}(I, n, n)

    # Get eigenvalues (diagonal of T)
    λ = diag(T_work)
    distances = abs.(λ .- z)

    # Bubble sort by distance (stable, preserves Schur form)
    for i in 1:n-1
        for j in n-1:-1:i
            if distances[j] > distances[j+1]
                # Swap eigenvalues j and j+1 using Givens rotation
                _swap_schur_eigenvalues!(T_work, Z, j)
                distances[j], distances[j+1] = distances[j+1], distances[j]
            end
        end
    end

    return T_work, Z
end

"""
Swap adjacent eigenvalues j and j+1 in Schur form using Givens rotation.
"""
function _swap_schur_eigenvalues!(T::Matrix{CT}, Z::Matrix{CT}, j::Int) where {CT}
    n = size(T, 1)

    # For 1x1 blocks (real or complex eigenvalues treated as 1x1)
    # Use the standard Schur swap algorithm

    a = T[j, j]
    b = T[j, j+1]
    c = T[j+1, j+1]

    # Solve Sylvester equation: (c - a) * x = b
    # Then construct Givens rotation
    if abs(c - a) > 1e-14
        x = b / (a - c)
        r = sqrt(1 + abs2(x))
        cs = 1 / r
        sn = x / r
    else
        # Eigenvalues are equal or very close, use QR
        cs = 1.0
        sn = 0.0
    end

    # Build Givens rotation matrix G
    G = [cs conj(sn); -sn cs]

    # Apply: T <- G' * T * G
    T[j:j+1, :] = G' * T[j:j+1, :]
    T[:, j:j+1] = T[:, j:j+1] * G

    # Update Z: Z <- Z * G
    Z[:, j:j+1] = Z[:, j:j+1] * G

    # Clean up numerical noise
    T[j+1, j] = zero(CT)
end

#==============================================================================#
# Comparison functions
#==============================================================================#

"""
    compare_methods(T::BallMatrix, z::Complex, m::Int; verbose=false)

Compare SVD-based and Rump-Oishi 2024 methods for computing σ_min(T - zI).

Returns a NamedTuple with timing and bound information.
"""
function compare_methods(T::BallMatrix{FT}, z::Complex, m::Int; verbose=false) where {FT}
    n = size(T, 1)

    # Form T - zI
    TmzI = T - z * BallMatrix(Matrix{Complex{FT}}(I, n, n))

    # Method 1: Full SVD (current approach)
    time_svd = @elapsed begin
        svd_result = rigorous_svd(TmzI)
        σ_min_svd = svd_result.singular_values[end]
    end
    σ_min_svd_lower = mid(σ_min_svd) - rad(σ_min_svd)
    σ_min_svd_upper = mid(σ_min_svd) + rad(σ_min_svd)

    # Method 2: Rump-Oishi 2024 Schur complement
    # Convert complex to real by treating as 2n x 2n real matrix
    # For now, work with the modulus for comparison
    TmzI_real = BallMatrix(abs.(mid(TmzI)), abs.(rad(TmzI)))

    time_ro2024 = @elapsed begin
        ro_result = rump_oishi_2024_sigma_min_bound(TmzI_real, m)
    end

    # True σ_min for reference
    σ_true = svdvals(mid(TmzI))[end]

    if verbose
        println("  SVD method:        σ_min ∈ [$σ_min_svd_lower, $σ_min_svd_upper] (time: $(round(time_svd*1000, digits=2)) ms)")
        if ro_result.verified
            println("  Rump-Oishi 2024:   σ_min ≥ $(ro_result.sigma_min_lower) (time: $(round(time_ro2024*1000, digits=2)) ms)")
        else
            println("  Rump-Oishi 2024:   FAILED (contraction=$(ro_result.schur_contraction))")
        end
        println("  True σ_min:        $σ_true")
    end

    return (
        σ_true = σ_true,
        svd_lower = σ_min_svd_lower,
        svd_upper = σ_min_svd_upper,
        svd_time = time_svd,
        ro2024_lower = ro_result.verified ? ro_result.sigma_min_lower : 0.0,
        ro2024_verified = ro_result.verified,
        ro2024_time = time_ro2024,
        ro2024_used_fast_gamma = ro_result.used_fast_gamma,
        ro2024_contraction = ro_result.schur_contraction,
        block_size = m
    )
end

"""
    compare_methods_triangular(T_mid::Matrix, z::Complex, m::Int; verbose=false)

Compare multiple methods for computing σ_min(T - zI) where T is upper triangular.

Methods:
1. Full SVD (current pseudospectra approach)
2. Triangular backward bound
3. Rump-Oishi 2024 triangular psi-bound
4. Schur complement on real 2n×2n representation

Note: For complex T - zI, the Schur complement methods use the real representation:
    [Re(M)  -Im(M)]
    [Im(M)   Re(M)]
which preserves singular values.
"""
function compare_methods_triangular(T_mid::Matrix{CT}, z::CT, m::Int; verbose=false) where {CT<:Complex}
    n = size(T_mid, 1)
    FT = real(CT)

    # Form T - zI (upper triangular complex)
    TmzI_mid = T_mid - z * I
    TmzI = BallMatrix(TmzI_mid, zeros(FT, n, n))

    # True σ_min for reference
    σ_true = svdvals(TmzI_mid)[end]

    results = Dict{Symbol, Any}()
    results[:σ_true] = σ_true
    results[:block_size] = m

    # Method 1: Full SVD on complex matrix (current approach)
    time_svd = @elapsed begin
        svd_result = rigorous_svd(TmzI)
        σ_min_svd = svd_result.singular_values[end]
    end
    σ_min_svd_lower = mid(σ_min_svd) - rad(σ_min_svd)
    results[:svd_lower] = σ_min_svd_lower
    results[:svd_time] = time_svd

    # Convert to real 2n × 2n representation for triangular methods
    # The complex upper triangular becomes a real upper quasi-triangular
    Re_M = real.(TmzI_mid)
    Im_M = imag.(TmzI_mid)

    # For purely upper triangular complex, the real representation is:
    # [Re(T-zI)   -Im(T-zI)]
    # [Im(T-zI)    Re(T-zI)]
    # This is NOT upper triangular, so triangular-specific methods don't apply directly

    # Method 2: Schur complement on real representation
    M_real = [Re_M -Im_M; Im_M Re_M]
    TmzI_real = BallMatrix(M_real, zeros(FT, 2n, 2n))

    m_real = min(2 * m, 2n - 1)  # Scale block size for 2n × 2n matrix
    schur_result = nothing
    time_schur = @elapsed begin
        try
            schur_result = rump_oishi_2024_sigma_min_bound(TmzI_real, m_real)
        catch e
            # May fail due to singular submatrix
        end
    end
    if schur_result !== nothing
        results[:schur_lower] = schur_result.verified ? schur_result.sigma_min_lower : 0.0
        results[:schur_verified] = schur_result.verified
        results[:schur_used_fast_gamma] = schur_result.used_fast_gamma
        results[:schur_contraction] = schur_result.schur_contraction
    else
        results[:schur_lower] = 0.0
        results[:schur_verified] = false
        results[:schur_used_fast_gamma] = false
        results[:schur_contraction] = Inf
    end
    results[:schur_time] = time_schur

    # Method 3: Oishi 2023 on real representation (for comparison)
    oishi_result = nothing
    time_oishi = @elapsed begin
        try
            oishi_result = oishi_2023_sigma_min_bound(TmzI_real, m_real)
        catch e
            # May fail due to singular submatrix
        end
    end
    if oishi_result !== nothing
        results[:oishi2023_lower] = oishi_result.verified ? oishi_result.sigma_min_lower : 0.0
        results[:oishi2023_verified] = oishi_result.verified
    else
        results[:oishi2023_lower] = 0.0
        results[:oishi2023_verified] = false
    end
    results[:oishi2023_time] = time_oishi

    # Method 4: For real upper triangular part only (approximate)
    # Use backward bound on just the real part as a quick estimate
    Re_ball = BallMatrix(UpperTriangular(Re_M), zeros(FT, n, n))
    backward_lower = 0.0
    time_backward = @elapsed begin
        try
            backward_bounds = backward_singular_value_bound(Re_ball)
            backward_lower = 1.0 / sup(backward_bounds[1])  # σ_min ≥ 1/‖T⁻¹‖
        catch e
            backward_lower = 0.0
        end
    end
    results[:backward_lower] = backward_lower
    results[:backward_time] = time_backward

    if verbose
        println("  True σ_min:        $(round(σ_true, sigdigits=6))")
        println("  SVD method:        σ_min ≥ $(round(σ_min_svd_lower, sigdigits=6)) (time: $(round(time_svd*1000, digits=2)) ms)")

        if schur_result !== nothing && schur_result.verified
            ratio = σ_min_svd_lower / schur_result.sigma_min_lower
            println("  RO2024 Schur:      σ_min ≥ $(round(schur_result.sigma_min_lower, sigdigits=6)) (time: $(round(time_schur*1000, digits=2)) ms) ratio=$(round(ratio, digits=3))")
            println("                     (fast γ: $(schur_result.used_fast_gamma), contraction: $(round(schur_result.schur_contraction, sigdigits=3)))")
        elseif schur_result !== nothing
            println("  RO2024 Schur:      FAILED (contraction=$(round(schur_result.schur_contraction, sigdigits=3)))")
        else
            println("  RO2024 Schur:      ERROR (singular matrix)")
        end

        if oishi_result !== nothing && oishi_result.verified
            println("  Oishi 2023:        σ_min ≥ $(round(oishi_result.sigma_min_lower, sigdigits=6)) (time: $(round(time_oishi*1000, digits=2)) ms)")
        elseif oishi_result !== nothing
            println("  Oishi 2023:        FAILED")
        else
            println("  Oishi 2023:        ERROR (singular matrix)")
        end

        if backward_lower > 0
            println("  Backward (Re):     σ_min ≥ $(round(backward_lower, sigdigits=6)) (time: $(round(time_backward*1000, digits=2)) ms)")
        end
    end

    return results
end

#==============================================================================#
# Test cases
#==============================================================================#

"""
Generate a random Schur matrix with specified eigenvalue distribution.
"""
function random_schur_matrix(n::Int, λ_generator; seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    # Generate eigenvalues
    λ = λ_generator(n)

    # Create upper triangular matrix with eigenvalues on diagonal
    T = diagm(0 => λ)

    # Add random strictly upper triangular part
    for i in 1:n
        for j in i+1:n
            T[i, j] = 0.1 * randn(ComplexF64)
        end
    end

    return T
end

"""
Run experiment with a specific test case.
"""
function run_experiment(name::String, T_mid::Matrix{ComplexF64}, z::ComplexF64;
                        block_sizes=[2, 5, 10, 20], reorder=true)
    n = size(T_mid, 1)
    println("\n" * "="^70)
    println("Experiment: $name")
    println("Matrix size: $n × $n")
    println("Query point: z = $z")
    println("="^70)

    # Optionally reorder Schur matrix
    if reorder
        T_reordered, _ = reorder_schur_by_distance(T_mid, z)
        println("\nEigenvalues reordered by distance to z (closest first)")

        # Show first few eigenvalue distances
        λ = diag(T_reordered)
        distances = abs.(λ .- z)
        println("First 5 distances: ", round.(distances[1:min(5,n)], digits=4))
        println("Last 5 distances:  ", round.(distances[max(1,n-4):n], digits=4))
    else
        T_reordered = T_mid
        println("\nNo reordering applied")
    end

    results = Dict[]

    for m in block_sizes
        if m >= n
            continue
        end
        println("\n--- Block size m = $m ---")
        result = compare_methods_triangular(T_reordered, z, m; verbose=true)
        push!(results, result)
    end

    return results
end

#==============================================================================#
# Main experiments
#==============================================================================#

function main()
    println("╔══════════════════════════════════════════════════════════════════════╗")
    println("║  Rump-Oishi 2024 vs SVD: Pseudospectra Certification Comparison      ║")
    println("╚══════════════════════════════════════════════════════════════════════╝")

    all_results = Dict{String, Vector}()

    # Experiment 1: Well-separated eigenvalues
    println("\n" * "▶"^35)
    println("EXPERIMENT 1: Well-separated eigenvalues")
    T1 = random_schur_matrix(50, n -> [complex(i, 0.1*i) for i in 1:n]; seed=42)
    z1 = 5.0 + 0.5im  # Close to eigenvalue 5
    all_results["well_separated"] = run_experiment(
        "Well-separated eigenvalues", T1, z1;
        block_sizes=[2, 5, 10, 15, 20]
    )

    # Experiment 2: Clustered eigenvalues
    println("\n" * "▶"^35)
    println("EXPERIMENT 2: Clustered eigenvalues")
    T2 = random_schur_matrix(50, n -> [complex(1 + 0.01*i, 0.01*i) for i in 1:n]; seed=42)
    z2 = 1.2 + 0.2im  # Near the cluster
    all_results["clustered"] = run_experiment(
        "Clustered eigenvalues", T2, z2;
        block_sizes=[2, 5, 10, 15, 20]
    )

    # Experiment 3: Large matrix
    println("\n" * "▶"^35)
    println("EXPERIMENT 3: Larger matrix (n=100)")
    T3 = random_schur_matrix(100, n -> [complex(i, sin(i/10)) for i in 1:n]; seed=42)
    z3 = 50.0 + 0.5im
    all_results["large"] = run_experiment(
        "Large matrix (n=100)", T3, z3;
        block_sizes=[5, 10, 20, 30, 40]
    )

    # Experiment 4: Effect of reordering
    println("\n" * "▶"^35)
    println("EXPERIMENT 4: Effect of eigenvalue reordering")
    T4 = random_schur_matrix(50, n -> [complex(i, 0.1*randn()) for i in 1:n]; seed=42)
    z4 = 25.0 + 0.0im  # Middle eigenvalue

    println("\n--- WITHOUT reordering ---")
    results_no_reorder = run_experiment(
        "No reordering", T4, z4;
        block_sizes=[5, 10, 15], reorder=false
    )

    println("\n--- WITH reordering ---")
    results_with_reorder = run_experiment(
        "With reordering", T4, z4;
        block_sizes=[5, 10, 15], reorder=true
    )

    all_results["no_reorder"] = results_no_reorder
    all_results["with_reorder"] = results_with_reorder

    # Experiment 5: Timing comparison for various sizes
    println("\n" * "▶"^35)
    println("EXPERIMENT 5: Timing comparison across matrix sizes")
    timing_results = Dict[]

    for n in [20, 50, 100, 150, 200]
        println("\n--- n = $n ---")
        T = random_schur_matrix(n, i -> [complex(j, 0.1*j) for j in 1:i]; seed=42)
        z = complex(n/2, 0.5)
        T_reordered, _ = reorder_schur_by_distance(T, z)

        m = min(20, n-1)
        result = compare_methods_triangular(T_reordered, z, m; verbose=true)
        result[:n] = n
        push!(timing_results, result)
    end

    all_results["timing"] = timing_results

    # Summary
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)

    println("\nTiming comparison (n vs time in ms):")
    println("   n  |  SVD time  | Schur time | Speedup | Schur verified")
    println("------|------------|------------|---------|---------------")
    for r in timing_results
        if r[:schur_verified]
            speedup = r[:svd_time] / r[:schur_time]
            @printf(" %4d |  %8.2f  |  %8.2f  |  %5.2fx |  Yes\n",
                    r[:n], r[:svd_time]*1000, r[:schur_time]*1000, speedup)
        else
            @printf(" %4d |  %8.2f  |  %8.2f  |   N/A   |  No (γ=%.2f)\n",
                    r[:n], r[:svd_time]*1000, r[:schur_time]*1000, r[:schur_contraction])
        end
    end

    println("\nBound quality comparison (SVD vs Schur complement):")
    for (name, results) in all_results
        if name == "timing"
            continue
        end
        println("\n$name:")
        for r in results
            if r[:schur_verified]
                ratio = r[:svd_lower] / r[:schur_lower]
                @printf("  m=%2d: SVD=%.6f, Schur=%.6f, ratio=%.3f, true=%.6f\n",
                        r[:block_size], r[:svd_lower], r[:schur_lower], ratio, r[:σ_true])
            else
                @printf("  m=%2d: SVD=%.6f, Schur=FAILED (γ=%.2f), true=%.6f\n",
                        r[:block_size], r[:svd_lower], r[:schur_contraction], r[:σ_true])
            end
        end
    end

    # Key findings
    println("\n" * "="^70)
    println("KEY FINDINGS")
    println("="^70)

    # Count successes
    total_tests = 0
    schur_successes = 0
    svd_tighter = 0

    for (name, results) in all_results
        if name == "timing"
            continue
        end
        for r in results
            total_tests += 1
            if r[:schur_verified]
                schur_successes += 1
                if r[:svd_lower] > r[:schur_lower]
                    svd_tighter += 1
                end
            end
        end
    end

    println("\n1. Schur complement method success rate: $schur_successes / $total_tests")
    if schur_successes > 0
        println("2. SVD gives tighter bound in $svd_tighter / $schur_successes verified cases")
    end
    println("3. The Schur complement method is designed for diagonally dominant matrices")
    println("   and may not be optimal for general Schur forms from eigenvalue problems.")

    return all_results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
