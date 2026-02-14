# Triangular matrix inverse norm bounds and similarity conditioning
#
# These utilities compute rigorous upper bounds on inverse norms
# of triangular matrices without forming the inverse explicitly.

"""
    triangular_inverse_inf_norm_bound(U::AbstractMatrix{T}) where {T}

Compute an upper bound on ‖U⁻¹‖_∞ for upper triangular U using backward recursion.

For upper triangular U, define y ∈ ℝᵐ:
- y[m] = 1/|u[m,m]|
- for i = m-1:-1:1: y[i] = (1 + Σⱼ |u[i,j]| * y[j]) / |u[i,i]|

Then ‖U⁻¹‖_∞ ≤ max(y).

# Returns
- Upper bound on ‖U⁻¹‖_∞, or Inf if U is singular (zero diagonal)
"""
function triangular_inverse_inf_norm_bound(U::AbstractMatrix{CT}) where {CT}
    m = size(U, 1)
    m == size(U, 2) || throw(DimensionMismatch("U must be square"))

    T = real(CT)
    y = zeros(T, m)

    # Backward recursion
    for i in m:-1:1
        diag_i = abs(U[i, i])
        if diag_i ≤ eps(T) * norm(U, Inf)
            return T(Inf)  # Singular or nearly singular
        end

        # Compute sum of |u[i,j]| * y[j] for j > i
        row_sum = zero(T)
        for j in (i+1):m
            row_sum += abs(U[i, j]) * y[j]
        end

        y[i] = (one(T) + row_sum) / diag_i
    end

    return maximum(y)
end

"""
    triangular_inverse_one_norm_bound(U::AbstractMatrix{T}) where {T}

Compute an upper bound on ‖U⁻¹‖₁ for upper triangular U.

Uses the identity ‖U⁻¹‖₁ = ‖(Uᵀ)⁻¹‖_∞ and applies forward recursion.

# Returns
- Upper bound on ‖U⁻¹‖₁, or Inf if U is singular
"""
function triangular_inverse_one_norm_bound(U::AbstractMatrix{CT}) where {CT}
    m = size(U, 1)
    m == size(U, 2) || throw(DimensionMismatch("U must be square"))

    T = real(CT)
    y = zeros(T, m)

    # Forward recursion (equivalent to backward on Uᵀ)
    for i in 1:m
        diag_i = abs(U[i, i])
        if diag_i ≤ eps(T) * norm(U, Inf)
            return T(Inf)  # Singular or nearly singular
        end

        # Compute sum of |u[j,i]| * y[j] for j < i
        col_sum = zero(T)
        for j in 1:(i-1)
            col_sum += abs(U[j, i]) * y[j]
        end

        y[i] = (one(T) + col_sum) / diag_i
    end

    return maximum(y)
end

"""
    triangular_inverse_two_norm_bound(U::AbstractMatrix{T}) where {T}

Compute an upper bound on ‖U⁻¹‖₂ for upper triangular U.

Uses ‖U⁻¹‖₂ ≤ √(‖U⁻¹‖₁ · ‖U⁻¹‖_∞).

# Returns
- Upper bound on ‖U⁻¹‖₂, or Inf if U is singular
"""
function triangular_inverse_two_norm_bound(U::AbstractMatrix{CT}) where {CT}
    T = real(CT)
    norm_inf = triangular_inverse_inf_norm_bound(U)
    if !isfinite(norm_inf)
        return T(Inf)
    end

    norm_one = triangular_inverse_one_norm_bound(U)
    if !isfinite(norm_one)
        return T(Inf)
    end

    return sqrt(norm_one * norm_inf)
end

#==============================================================================#
# Similarity transformation conditioning
#==============================================================================#

"""
    psi_squared(μ::T) where {T}

Compute ψ(μ)² where ψ(μ) is the 2-norm of the unit block triangular matrix S(X).

For S(X) = [I, -X; 0, I], we have ‖S‖₂ = ‖S⁻¹‖₂ = ψ(‖X‖₂).

The formula is:
    ψ(μ)² = 1 + μ²/2 + (μ/2)·√(μ² + 4)

This equals κ₂(S(X)) when ‖X‖₂ = μ.

# Arguments
- `μ::T`: Upper bound on ‖X‖₂

# Returns
- ψ(μ)² = κ₂(S(X))
"""
function psi_squared(μ::T) where {T<:AbstractFloat}
    if μ ≤ zero(T)
        return one(T)
    end

    μ_sq = μ * μ
    sqrt_term = sqrt(μ_sq + T(4))

    return one(T) + μ_sq / T(2) + (μ / T(2)) * sqrt_term
end

"""
    similarity_condition_number(X::AbstractMatrix{T}) where {T}

Compute κ₂(S(X)) for the similarity transformation S(X) = [I, -X; 0, I].

Uses the cheap bound ‖X‖₂ ≤ √(‖X‖₁ · ‖X‖_∞).

# Returns
- Upper bound on κ₂(S(X))
"""
function similarity_condition_number(X::AbstractMatrix{CT}) where {CT}
    T = real(CT)

    # Cheap 2-norm bound
    norm1 = opnorm(X, 1)
    norminf = opnorm(X, Inf)
    μ = sqrt(norm1 * norminf)

    return psi_squared(T(μ))
end
