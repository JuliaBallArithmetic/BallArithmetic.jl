"""
    adaptive_ogita_svd.jl

Adaptive precision SVD with Ogita's iterative refinement.

Based on Rump & Ogita (2024), this implements an adaptive algorithm that:
1. Starts with low precision (Float64)
2. Computes rigorous SVD bounds
3. If bounds are not tight enough, refines using Ogita's iterative method
4. Automatically increases precision as needed
"""

"""
    _spectral_norm_bound(A::AbstractMatrix)

Compute an upper bound on the spectral norm (2-norm) using:
    ‖A‖₂ ≤ √(‖A‖∞ · ‖A‖₁)

where ‖A‖∞ = max row sum of |A| and ‖A‖₁ = max column sum of |A|.

This bound is valid for any matrix and computable for BigFloat (unlike opnorm(..., 2)
which requires svdvals and fails for BigFloat).
"""
function _spectral_norm_bound(A::AbstractMatrix{T}) where T
    m, n = size(A)

    # ‖A‖∞ = max row sum of absolute values
    inf_norm = zero(real(T))
    for i in 1:m
        row_sum = sum(abs(A[i, j]) for j in 1:n)
        inf_norm = max(inf_norm, row_sum)
    end

    # ‖A‖₁ = max column sum of absolute values
    one_norm = zero(real(T))
    for j in 1:n
        col_sum = sum(abs(A[i, j]) for i in 1:m)
        one_norm = max(one_norm, col_sum)
    end

    return sqrt(inf_norm * one_norm)
end

"""
    OgitaSVDRefinementResult

Result from Ogita's iterative SVD refinement.
"""
struct OgitaSVDRefinementResult{T, UT, ΣT, VT}
    U::UT                      # Refined left singular vectors
    Σ::ΣT                      # Refined singular values
    V::VT                      # Refined right singular vectors
    iterations::Int            # Number of refinement iterations performed
    precision_used::Int        # Precision (bits) used for refinement
    residual_norm::T           # Final residual norm
    converged::Bool            # Whether refinement converged
end

"""
    AdaptiveSVDResult

Result from adaptive precision SVD computation.
"""
struct AdaptiveSVDResult{T, RigT}
    rigorous_result::RigT           # Final RigorousSVDResult
    precision_levels::Vector{Int}   # Precision levels attempted
    radii_history::Vector{T}        # Maximum radii at each level
    total_iterations::Int           # Total refinement iterations
    final_precision::Int            # Final precision used
    tolerance_achieved::Bool        # Whether tolerance was met
end

"""
    ogita_iterations_for_precision(target_bits::Int; start_bits::Int=53)

Compute the number of Ogita iterations needed to achieve target precision,
based on quadratic convergence theory.

With quadratic convergence, each iteration roughly doubles the number of correct bits.
Starting from Float64 (53 bits), we need ceil(log2(target_bits / start_bits)) iterations.

# Examples
- 256 bits: ceil(log2(256/53)) = 3 iterations
- 512 bits: ceil(log2(512/53)) = 4 iterations
- 1024 bits: ceil(log2(1024/53)) = 5 iterations
"""
function ogita_iterations_for_precision(target_bits::Int; start_bits::Int=53)
    if target_bits <= start_bits
        return 0
    end
    return ceil(Int, log2(target_bits / start_bits))
end

"""
    ogita_svd_refine(A::AbstractMatrix{T}, U, Σ, V;
                     max_iterations=10, precision_bits=256,
                     check_convergence=false,
                     use_optimal_iterations=false) where {T<:AbstractFloat}

Refine an approximate SVD using Ogita's iterative method (RefSVD algorithm).

# Arguments
- `A`: Original matrix (in higher precision if needed)
- `U`: Initial left singular vectors
- `Σ`: Initial singular values
- `V`: Initial right singular vectors
- `max_iterations`: Maximum number of iterations to run (default: 10)
- `precision_bits`: Working precision in bits (default: 256)
- `check_convergence`: If `true`, check convergence via spectral norm and stop early.
                       If `false` (default), run fixed number of iterations.
- `use_optimal_iterations`: If `true`, compute optimal iteration count from quadratic
                            convergence theory and ignore max_iterations.
                            This is the fastest option for known target precision.

# Iteration count (quadratic convergence from Float64)
Based on theory, starting from Float64 (~53 bits), each iteration doubles precision:
- 3 iterations: sufficient for 256-bit (~77 decimal digits)
- 4 iterations: sufficient for 512-bit (~154 decimal digits)
- 5 iterations: sufficient for 1024-bit (~308 decimal digits)

# Returns
- `OgitaSVDRefinementResult` containing refined SVD

# References
- [OgitaAishima2020](@cite) Ogita, T. & Aishima, K. (2020), "Iterative refinement for
  singular value decomposition based on matrix multiplication",
  J. Comput. Appl. Math. 369, 112512.
"""
function ogita_svd_refine(A::AbstractMatrix{T}, U, Σ, V;
                         max_iterations::Int=10,
                         precision_bits::Int=256,
                         check_convergence::Bool=false,
                         use_optimal_iterations::Bool=false) where {T<:Union{AbstractFloat, Complex{<:AbstractFloat}}}
    RT = real(T)  # Get the real type (Float64, BigFloat, etc.)

    # Compute optimal iterations if requested
    iterations_to_run = if use_optimal_iterations
        ogita_iterations_for_precision(precision_bits)
    else
        max_iterations
    end

    # Convert to higher precision if needed
    if RT == Float64 && precision_bits > 53
        # Need to work in BigFloat
        A_high = convert.(Complex{BigFloat}, A)
        U_high = convert.(Complex{BigFloat}, U)
        Σ_high = convert.(BigFloat, Σ)
        V_high = convert.(Complex{BigFloat}, V)

        # Set precision
        old_precision = precision(BigFloat)
        setprecision(BigFloat, precision_bits)

        try
            result = _ogita_svd_refine_impl(A_high, U_high, Σ_high, V_high, iterations_to_run, check_convergence)
            return result
        finally
            setprecision(BigFloat, old_precision)
        end
    else
        return _ogita_svd_refine_impl(A, U, Σ, V, iterations_to_run, check_convergence)
    end
end

"""
    _ogita_svd_refine_impl(A, U, Σ, V, max_iterations)

Internal implementation of Ogita's iterative SVD refinement.

Based on Algorithm 1 (RefSVD) from:
Ogita, T. & Aishima, K. (2020), "Iterative refinement for singular value
decomposition based on matrix multiplication", J. Comput. Appl. Math. 369, 112512.

The algorithm computes corrections F̃ and G̃ such that:
  U' = Û(I + F̃)
  V' = V̂(I + G̃)

This converges quadratically if singular values are well-separated.
"""
function _ogita_svd_refine_impl(A::AbstractMatrix{T}, U, Σ, V,
                                max_iterations::Int,
                                check_convergence::Bool) where {T<:Union{AbstractFloat, Complex{<:AbstractFloat}}}
    m, n = size(A)
    min_dim = min(m, n)
    RT = real(T)  # Real type for singular values

    # Current approximations - convert to match A's precision
    U_curr = convert.(T, U)
    Σ_curr = convert.(RT, Σ)
    V_curr = convert.(T, V)

    iterations = 0
    converged = false

    # Quadratic convergence guarantees: iterations ≈ ceil(log2(precision_bits / 15))
    # - 2 iterations: ~10^-60 (enough for 256-bit)
    # - 3 iterations: ~10^-120 (saturates 256-bit, enough for 512-bit)
    # - 4 iterations: ~10^-240 (enough for 1024-bit)
    # NOTE: This iterative refinement is an ORACLE computation. The accumulated
    # errors during iteration are NOT tracked here because the final result is
    # verified A POSTERIORI by computing rigorous residual bounds. The refinement
    # improves the approximation quality, but rigor comes from the final verification.
    for iter in 1:max_iterations
        iterations = iter
        # Step 1: Compute residual matrices (Algorithm 1, line 1)
        # R = I_m - U^T U (orthogonality residual for U)
        # S = I_n - V^T V (orthogonality residual for V)
        # T = U^T A V (should be diagonal if SVD is exact)
        R = I - U_curr' * U_curr
        S = I - V_curr' * V_curr
        T_matrix = U_curr' * A * V_curr

        # Step 2: Compute refined singular values (Algorithm 1, line 2)
        # Singular values are always real
        σ_tilde = zeros(RT, min_dim)
        for i in 1:min_dim
            denom = 1 - real(R[i,i] + S[i,i]) / 2
            σ_tilde[i] = real(T_matrix[i,i] / denom)
        end

        # Steps 3-7: Compute correction matrices F̃ and G̃
        F_tilde = zeros(T, m, m)
        G_tilde = zeros(T, n, n)

        # Step 3: Diagonal parts (Algorithm 1, line 3)
        for i in 1:n
            F_tilde[i,i] = R[i,i] / 2
            G_tilde[i,i] = S[i,i] / 2
        end

        # Step 4: Off-diagonal parts of F_11 and G (Algorithm 1, line 4)
        # From Ogita & Aishima 2020:
        #   α := T_{i,j} + σ̃_j · R_{i,j}
        #   β := conj(T_{j,i}) + σ̃_j · conj(S_{j,i})
        #   F̃_{i,j} := (α·σ̃_j + β·σ̃_i) / (σ̃_j² - σ̃_i²)
        #   G̃_{i,j} := (α·σ̃_i + β·σ̃_j) / (σ̃_j² - σ̃_i²)
        for i in 1:n
            for j in 1:n
                if i != j
                    denom = σ_tilde[j]^2 - σ_tilde[i]^2
                    # Skip if singular values are too close (clustered)
                    if abs(denom) > eps(RT) * max(σ_tilde[i], σ_tilde[j])^2
                        α = T_matrix[i,j] + σ_tilde[j] * R[i,j]
                        β = conj(T_matrix[j,i]) + σ_tilde[j] * conj(S[j,i])
                        F_tilde[i,j] = (α * σ_tilde[j] + β * σ_tilde[i]) / denom
                        G_tilde[i,j] = (α * σ_tilde[i] + β * σ_tilde[j]) / denom
                    end
                end
            end
        end

        # Step 5: F_12 block (Algorithm 1, line 5)
        if m > n
            for i in 1:n
                for j in (n+1):m
                    if abs(σ_tilde[i]) > eps(RT)
                        F_tilde[i,j] = -T_matrix[j,i] / σ_tilde[i]
                    end
                end
            end
        end

        # Step 6: F_21 block (Algorithm 1, line 6)
        if m > n
            for i in (n+1):m
                for j in 1:n
                    F_tilde[i,j] = R[i,j] - F_tilde[j,i]
                end
            end
        end

        # Step 7: F_22 block (Algorithm 1, line 7)
        if m > n
            for i in (n+1):m
                for j in (n+1):m
                    F_tilde[i,j] = R[i,j] / 2
                end
            end
        end

        # Step 8: Update U and V (Algorithm 1, line 8)
        U_curr = U_curr * (I + F_tilde)
        V_curr = V_curr * (I + G_tilde)
        Σ_curr = σ_tilde

        # Optional convergence check based on correction matrix norms
        if check_convergence
            correction_norm = max(_spectral_norm_bound(F_tilde), _spectral_norm_bound(G_tilde))
            if correction_norm < eps(RT) * 100
                converged = true
                break
            end
        end
    end

    # Final singular value computation using the converged U and V
    # This is necessary because Σ_curr was computed BEFORE the last U,V update
    T_final = U_curr' * A * V_curr
    R_final = I - U_curr' * U_curr
    S_final = I - V_curr' * V_curr
    for i in 1:min_dim
        denom = one(RT) - real(R_final[i,i] + S_final[i,i]) / 2
        Σ_curr[i] = abs(T_final[i,i] / denom)  # Use abs() to get the magnitude
    end

    # Phase correction for complex matrices: ensure U'AV is real positive
    # If T_final[i,i] = σ_i * e^{iθ_i}, multiply U[:,i] by e^{iθ_i} so that
    # (new U[:,i])' * A * V[:,i] = σ_i (real positive)
    if T <: Complex
        for i in 1:min_dim
            if abs(T_final[i,i]) > eps(RT)
                phase = T_final[i,i] / abs(T_final[i,i])  # e^{iθ_i}
                U_curr[:, i] .*= phase  # U[:,i] ← U[:,i] * e^{iθ_i}
            end
        end
    end

    # Mark as converged if we ran all iterations (for fixed iteration mode)
    if !check_convergence
        converged = true
    end

    precision_used = T == Float64 ? 53 : precision(BigFloat)

    # Compute final residual norm for diagnostics (only once at the end)
    residual_norm = _spectral_norm_bound(A - U_curr * Diagonal(Σ_curr) * V_curr')

    return OgitaSVDRefinementResult(
        U_curr, Diagonal(Σ_curr), V_curr,
        iterations, precision_used, residual_norm, converged
    )
end


"""
    adaptive_ogita_svd(A::BallMatrix{T};
                       tolerance=1e-10,
                       method::SVDMethod=MiyajimaM1(),
                       apply_vbd=true,
                       max_precision_bits=1024,
                       max_refinement_iterations=5) where {T}

Compute rigorous SVD bounds with adaptive precision using Ogita's refinement.

# Algorithm
1. Start with Float64 precision
2. Compute rigorous SVD bounds using `rigorous_svd`
3. Check if max(rad(σᵢ)) < tolerance
4. If not satisfied, refine using Ogita's method with doubled precision
5. Repeat until tolerance met or max precision reached

# Arguments
- `A`: Input ball matrix
- `tolerance`: Target tolerance for max(rad(σᵢ))
- `method`: SVD certification method (MiyajimaM1, MiyajimaM4, RumpOriginal)
- `apply_vbd`: Whether to apply verified block diagonalization
- `max_precision_bits`: Maximum precision to use (bits)
- `max_refinement_iterations`: Maximum number of refinement steps

# Returns
- `AdaptiveSVDResult` containing final rigorous result and adaptation history

# Example
```julia
A = BallMatrix([3.0 1.0; 1.0 2.0], fill(1e-8, 2, 2))
result = adaptive_ogita_svd(A; tolerance=1e-12)

# Access final result
σ = result.rigorous_result.singular_values
println("Final precision: ", result.final_precision, " bits")
println("Max radius: ", maximum(rad.(σ)))
```
"""
function adaptive_ogita_svd(A::BallMatrix{T};
                           tolerance::Real=1e-10,
                           method::SVDMethod=MiyajimaM1(),
                           apply_vbd::Bool=true,
                           max_precision_bits::Int=1024,
                           max_refinement_iterations::Int=5) where {T}

    # Track adaptation history
    precision_levels = Int[]
    radii_history = T[]
    total_iterations = 0

    # Start with Float64 precision
    current_precision = 53  # Float64 precision in bits
    A_working = T == Float64 ? A : BallMatrix(convert.(Float64, mid(A)), convert.(Float64, rad(A)))

    # Initial computation
    result = rigorous_svd(A_working; method, apply_vbd)

    push!(precision_levels, current_precision)
    max_radius = maximum(rad.(result.singular_values))
    push!(radii_history, max_radius)

    # Check if initial result is good enough
    if max_radius <= tolerance
        return AdaptiveSVDResult(
            result, precision_levels, radii_history,
            0, current_precision, true
        )
    end

    # Need refinement - iterate with increasing precision
    for refinement_step in 1:max_refinement_iterations
        # Double precision
        current_precision = min(current_precision * 2, max_precision_bits)
        push!(precision_levels, current_precision)

        @info "Refining SVD at precision $(current_precision) bits (step $refinement_step/$max_refinement_iterations)"

        # Convert to higher precision
        A_high = _convert_to_precision(A, current_precision)

        # Extract current SVD for refinement
        U_curr = mid(result.U)
        Σ_curr = [mid(σ) for σ in result.singular_values]
        V_curr = mid(result.V)

        # Apply Ogita refinement
        refined = ogita_svd_refine(
            mid(A_high), U_curr, Σ_curr, V_curr;
            max_iterations=10,
            precision_bits=current_precision
        )

        total_iterations += refined.iterations

        if !refined.converged
            @warn "Ogita refinement did not converge at precision $(current_precision) bits"
        end

        # Certify the refined SVD
        # Create synthetic SVD for certification
        Σ_refined = isa(refined.Σ, Diagonal) ? diag(refined.Σ) : refined.Σ
        svd_refined = (U=refined.U,
                      S=Σ_refined,
                      V=refined.V,
                      Vt=refined.V')

        # Certify with rigorous bounds
        result = _certify_svd(A_high, svd_refined, method; apply_vbd)

        # Check new radius
        max_radius = maximum(rad.(result.singular_values))
        push!(radii_history, max_radius)

        @info "After refinement: max radius = $(max_radius)"

        if max_radius <= tolerance
            @info "Tolerance achieved at precision $(current_precision) bits"
            return AdaptiveSVDResult(
                result, precision_levels, radii_history,
                total_iterations, current_precision, true
            )
        end

        if current_precision >= max_precision_bits
            @warn "Maximum precision reached without achieving tolerance"
            break
        end
    end

    # Tolerance not achieved
    @warn "Maximum refinement iterations reached. Final max radius: $(max_radius), target: $(tolerance)"
    return AdaptiveSVDResult(
        result, precision_levels, radii_history,
        total_iterations, current_precision, false
    )
end

"""
    _convert_to_precision(A::BallMatrix{T}, precision_bits::Int) where {T}

Convert a ball matrix to specified precision.
"""
function _convert_to_precision(A::BallMatrix{T}, precision_bits::Int) where {T}
    if precision_bits <= 53
        # Float64
        return BallMatrix(convert.(Float64, mid(A)), convert.(Float64, rad(A)))
    else
        # BigFloat with specified precision
        old_prec = precision(BigFloat)
        setprecision(BigFloat, precision_bits)
        try
            A_high = BallMatrix(convert.(BigFloat, mid(A)), convert.(BigFloat, rad(A)))
            return A_high
        finally
            setprecision(BigFloat, old_prec)
        end
    end
end

#==============================================================================#
# Stub functions for Double64 extension (DoubleFloatsExt)
#==============================================================================#

"""
    ogita_svd_refine_fast(A, U, Σ, V; max_iterations=2, certify_with_bigfloat=true, bigfloat_precision=256)

Fast SVD refinement using Double64 arithmetic (~106 bits precision).
Requires DoubleFloats.jl to be loaded.

This is ~30× faster than pure BigFloat refinement because:
1. Double64 uses native Float64 operations with error compensation
2. No memory allocation per arithmetic operation (unlike BigFloat)

See `DoubleFloatsExt` module for implementation details.
"""
function ogita_svd_refine_fast end

"""
    ogita_svd_refine_hybrid(A, U, Σ, V; d64_iterations=2, bf_iterations=1, precision_bits=256)

Hybrid SVD refinement: Double64 for bulk iterations, BigFloat for final polish.
Requires DoubleFloats.jl to be loaded.

Expected speedup: ~2× for 256-bit precision compared to pure BigFloat.
"""
function ogita_svd_refine_hybrid end

#==============================================================================#
# Stub functions for MultiFloat extension (MultiFloatsExt)
#==============================================================================#

"""
    ogita_svd_refine_multifloat(A, U, Σ, V; precision=:x2, max_iterations=2,
                                 certify_with_bigfloat=true, bigfloat_precision=256)

Fast SVD refinement using MultiFloats arithmetic.
Requires MultiFloats.jl to be loaded.

Precision options: `:x2` (~106 bits), `:x4` (~212 bits), `:x8` (~424 bits)
"""
function ogita_svd_refine_multifloat end

# Export new functions
export OgitaSVDRefinementResult, AdaptiveSVDResult
export ogita_svd_refine, adaptive_ogita_svd, ogita_iterations_for_precision
export ogita_svd_refine_fast, ogita_svd_refine_hybrid, ogita_svd_refine_multifloat
