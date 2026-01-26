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
    ogita_svd_refine(A::AbstractMatrix{T}, U, Σ, V;
                     max_iterations=10, precision_bits=256) where {T<:AbstractFloat}

Refine an approximate SVD using Ogita's iterative method.

# Arguments
- `A`: Original matrix (in higher precision if needed)
- `U`: Initial left singular vectors
- `Σ`: Initial singular values
- `V`: Initial right singular vectors
- `max_iterations`: Maximum number of iterations (default: 10)
- `precision_bits`: Working precision in bits (default: 256)

# Returns
- `OgitaSVDRefinementResult` containing refined SVD

# Algorithm
TODO: Implement Ogita's specific algorithm from RuOg24a.pdf

The general structure for iterative SVD refinement:
1. Compute residual R = A - U*Σ*V'
2. Solve correction equations for ΔU, ΔΣ, ΔV
3. Update: U += ΔU, Σ += ΔΣ, V += ΔV
4. Re-orthogonalize U and V if needed
5. Repeat until convergence

# References
- Rump, S.M. & Ogita, T. (2024), "..." (RuOg24a.pdf)
"""
function ogita_svd_refine(A::AbstractMatrix{T}, U, Σ, V;
                         max_iterations::Int=10,
                         precision_bits::Int=256) where {T<:AbstractFloat}
    # TODO: This is a placeholder. The user needs to provide specifics from RuOg24a.pdf
    # for the exact Ogita algorithm

    # Convert to higher precision if needed
    if T == Float64 && precision_bits > 53
        # Need to work in BigFloat
        A_high = convert.(BigFloat, A)
        U_high = convert.(BigFloat, U)
        Σ_high = convert.(BigFloat, Σ)
        V_high = convert.(BigFloat, V)

        # Set precision
        old_precision = precision(BigFloat)
        setprecision(BigFloat, precision_bits)

        try
            result = _ogita_svd_refine_impl(A_high, U_high, Σ_high, V_high, max_iterations)
            return result
        finally
            setprecision(BigFloat, old_precision)
        end
    else
        return _ogita_svd_refine_impl(A, U, Σ, V, max_iterations)
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
                                max_iterations::Int) where {T<:AbstractFloat}
    m, n = size(A)
    min_dim = min(m, n)

    # Current approximations
    U_curr = copy(U)
    Σ_curr = copy(Σ)
    V_curr = copy(V)

    converged = false
    iterations = 0
    residual_norm = T(Inf)

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
        σ_tilde = zeros(T, min_dim)
        for i in 1:min_dim
            denom = 1 - (R[i,i] + S[i,i]) / 2
            if abs(denom) < eps(T)
                @warn "Ogita refinement: near-singular denominator at i=$i"
                converged = false
                break
            end
            σ_tilde[i] = T_matrix[i,i] / denom
        end

        # Check convergence via residual
        residual_norm = opnorm(A - U_curr * Diagonal(σ_tilde) * V_curr', 2)
        if residual_norm < eps(T) * opnorm(A, 2) * 100
            converged = true
            Σ_curr = σ_tilde
            break
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
        for i in 1:n
            for j in 1:n
                if i != j
                    # Check for distinct singular values
                    if abs(σ_tilde[j]^2 - σ_tilde[i]^2) < eps(T) * max(σ_tilde[i], σ_tilde[j])^2
                        @warn "Ogita refinement: clustered singular values at i=$i, j=$j"
                        converged = false
                        break
                    end

                    α = T_matrix[i,j] + σ_tilde[j] * R[i,j]
                    β = T_matrix[j,i] + σ_tilde[j] * S[i,j]

                    F_tilde[i,j] = (α * σ_tilde[j] + β * σ_tilde[i]) / (σ_tilde[j]^2 - σ_tilde[i]^2)
                    G_tilde[i,j] = (α * σ_tilde[i] + β * σ_tilde[j]) / (σ_tilde[j]^2 - σ_tilde[i]^2)
                end
            end
        end

        if !converged
            break
        end

        # Step 5: F_12 block (Algorithm 1, line 5)
        if m > n
            for i in 1:n
                for j in (n+1):m
                    if abs(σ_tilde[i]) < eps(T)
                        @warn "Ogita refinement: near-zero singular value"
                        converged = false
                        break
                    end
                    F_tilde[i,j] = -T_matrix[j,i] / σ_tilde[i]
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

        # Check if corrections are small enough
        if max(opnorm(F_tilde, 2), opnorm(G_tilde, 2)) < eps(T) * 100
            converged = true
            break
        end
    end

    precision_used = T == Float64 ? 53 : precision(BigFloat)

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

# Export new functions
export OgitaSVDRefinementResult, AdaptiveSVDResult
export ogita_svd_refine, adaptive_ogita_svd
