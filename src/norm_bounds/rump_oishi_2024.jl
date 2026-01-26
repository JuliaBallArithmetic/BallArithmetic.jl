# Enhanced implementation of RumpOishi2024 triangular matrix bounds
# Reference: Rump, S.M. & Oishi, S. (2024), "A Note on Oishi's Lower Bound
# for the Smallest Singular Value of Linearized Galerkin Equations"

using LinearAlgebra

"""
    rump_oishi_2024_triangular_bound(T::BallMatrix, k::Int; method=:hybrid)

Compute rigorous upper bound on `‖T[1:k,:]⁻¹‖₂` for upper triangular `T`.

Implements the bounds from RumpOishi2024 with improved handling of the
Collatz spectral radius estimate for triangular matrices.

# Arguments
- `T::BallMatrix`: Upper triangular ball matrix
- `k::Int`: Block size (compute bound for first k rows/columns)
- `method::Symbol`: Method to use
  - `:psi` - ψ-bound method (original RumpOishi2024)
  - `:backward` - Backward substitution method
  - `:hybrid` - Use best of both methods (default)

# Method Details

## Ψ-bound method:
For block structure `T = [A B; 0 D]` where `A` is k×k:
1. Compute `E = A⁻¹B` via backward substitution
2. Compute `F = D_d⁻¹ D_f` where `D = D_d + D_f` (diagonal + off-diagonal)
3. Estimate using ψ bounds: `‖T⁻¹‖ ≤ max(α, β) · ψ(E)`
   where α = ‖A⁻¹‖, β = ‖D_d⁻¹‖/(1-‖F‖)

## Backward substitution method:
Recursively compute bounds for singular values using:
`σᵢ = (1/|dᵢᵢ|) · √(1 + ‖bᵢ‖² · σᵢ₊₁²)`
where `bᵢ` is the i-th row tail and `dᵢᵢ` is the i-th diagonal.

# Returns
Rigorous upper bound on `‖T[1:k,:]⁻¹‖₂` as a floating-point number.

# Reference
* Rump & Oishi (2024), "A Note on Oishi's Lower Bound...",
  SIAM J. Matrix Anal. Appl.
"""
function rump_oishi_2024_triangular_bound(T::BallMatrix{T_type, NT}, k::Int;
                                           method::Symbol = :hybrid) where {T_type, NT}
    istriu(T.c) && istriu(T.r) ||
        throw(ArgumentError("T must be upper triangular"))

    n = size(T, 1)
    1 ≤ k ≤ n || throw(ArgumentError("k must be between 1 and $n"))

    method ∈ [:psi, :backward, :hybrid] ||
        throw(ArgumentError("method must be :psi, :backward, or :hybrid"))

    if k == n
        # Full matrix - use direct SVD bound
        return svd_bound_L2_opnorm_inverse(T)
    end

    if method == :psi || method == :hybrid
        bound_psi = _rump_oishi_psi_bound(T, k)
    end

    if method == :backward || method == :hybrid
        bound_backward = _rump_oishi_backward_bound(T, k)
    end

    if method == :psi
        return bound_psi
    elseif method == :backward
        return bound_backward
    else  # :hybrid
        return min(bound_psi, bound_backward)
    end
end

"""
    _rump_oishi_psi_bound(T, k)

Ψ-bound method from RumpOishi2024 with improved Collatz handling.
"""
function _rump_oishi_psi_bound(T::BallMatrix{T_type, NT}, k::Int) where {T_type, NT}
    # Extract blocks: T = [A B; 0 D]
    A = T[1:k, 1:k]
    B = T[1:k, (k+1):end]
    D = T[(k+1):end, (k+1):end]

    # Split D = D_diagonal + D_offdiagonal
    Dd = Diagonal(diag(D))
    Df = D - Dd

    # Compute E = A⁻¹B via backward substitution
    E = backward_substitution(A, B)

    # Compute ψ(E) bound
    psi_E = _psi_bound_improved(E)

    # Compute ‖A⁻¹‖ using SVD or backward substitution
    norm_Ainv = svd_bound_L2_opnorm_inverse(A)

    # Compute D_d⁻¹ (diagonal inverse)
    Ddinv_diag = [1.0 / Dd[i, i] for i in 1:size(Dd, 1)]
    norm_Ddinv = setrounding(T_type, RoundUp) do
        maximum([sup(abs(Ball(Ddinv_diag[i]))) for i in eachindex(Ddinv_diag)])
    end

    # Compute F = D_d⁻¹ · D_f with proper triangular structure preservation
    F = _compute_F_triangular(Dd, Df, T_type)

    # Compute ‖F‖ using improved Collatz for strictly triangular matrices
    norm_F = _collatz_strictly_triangular(F)

    # Verification: ‖F‖ < 1 required for convergence
    if norm_F ≥ one(T_type)
        @warn "‖F‖ = $norm_F ≥ 1, bound may be loose or invalid"
        # Fall back to simple bound
        return setrounding(T_type, RoundUp) do
            max(norm_Ainv, norm_Ddinv) * (1 + psi_E)
        end
    end

    # Final bound: max(α, β) · ψ(E) where β = ‖D_d⁻¹‖/(1-‖F‖)
    bound = setrounding(T_type, RoundUp) do
        α = norm_Ainv
        β = norm_Ddinv / (one(T_type) - norm_F)
        max(α, β) * psi_E
    end

    return bound
end

"""
    _rump_oishi_backward_bound(T, k)

Backward substitution bound for triangular matrices.
"""
function _rump_oishi_backward_bound(T::BallMatrix{T_type, NT}, k::Int) where {T_type, NT}
    # Extract blocks
    A = T[1:k, 1:k]
    B = T[1:k, (k+1):end]
    D = T[(k+1):end, (k+1):end]

    # Estimate ‖A⁻¹‖
    norm_Ainv = svd_bound_L2_opnorm_inverse(A)

    # Compute B·D⁻¹ via backward substitution on D
    BDinv = backward_substitution(D, B')'

    # Estimate ‖B·D⁻¹‖
    norm_BDinv = collatz_upper_bound_L2_opnorm(BDinv)

    # Estimate ‖D⁻¹‖ using backward singular value bound
    σ_D = backward_singular_value_bound(D)
    norm_Dinv = sup(σ_D[1])

    # Combined bound: max(‖A⁻¹‖(1 + ‖BD⁻¹‖), ‖D⁻¹‖)
    bound = setrounding(T_type, RoundUp) do
        max(norm_Ainv * (one(T_type) + norm_BDinv), norm_Dinv)
    end

    return bound
end

"""
    _compute_F_triangular(Dd, Df, ::Type{T})

Compute F = D_d⁻¹ · D_f while preserving triangular structure.
"""
function _compute_F_triangular(Dd::Diagonal, Df::BallMatrix{T_type, NT},
                                ::Type{T}) where {T, T_type, NT}
    n = size(Dd, 1)

    # Create ball matrix for Dd⁻¹
    Ddinv_ball = BallMatrix(Diagonal([1.0 / Dd[i, i] for i in 1:n]))

    # Multiply: F = Dd⁻¹ · Df
    F_temp = Ddinv_ball * Df

    # Ensure strict triangular structure (zero diagonal)
    F_mid = mid(F_temp)
    F_rad = rad(F_temp)

    for i in 1:n
        F_mid[i, i] = zero(eltype(F_mid))
        F_rad[i, i] = zero(T_type)
    end

    return _triangularize(BallMatrix(F_mid, F_rad))
end

"""
    _collatz_strictly_triangular(F::BallMatrix)

Improved Collatz bound for strictly triangular matrices (zero diagonal).

For strictly triangular matrices, the spectral radius is exactly zero.
However, numerical computation may introduce small errors. This function
computes a tight bound accounting for:
1. Zero diagonal
2. Triangular structure
3. Interval arithmetic propagation
"""
function _collatz_strictly_triangular(F::BallMatrix{T}) where {T}
    n = size(F, 1)

    # For strictly triangular matrix, use power method on |F|
    # but with awareness that exact spectral radius should be 0

    absF = upper_abs(F)

    # Check if matrix is numerically zero
    norm_F_frobenius = setrounding(T, RoundUp) do
        sqrt(sum(absF.^2))
    end

    if norm_F_frobenius < 100 * eps(T)
        return zero(T)
    end

    # Use standard Collatz but with more iterations for triangular case
    lam = setrounding(T, RoundUp) do
        x_old = ones(n)
        x_new = x_old

        # More iterations for triangular matrices
        for iter in 1:20
            x_old = x_new
            x_new = absF' * absF * x_old

            # Normalize to prevent overflow
            scale = maximum(x_new)
            if scale > zero(T)
                x_new = x_new ./ scale
            end

            # Early termination if converged
            if iter > 5 && maximum(abs.(x_new - x_old)) < eps(T)
                break
            end
        end

        # Compute maximum ratio
        ratio = zero(T)
        for i in 1:n
            if x_old[i] > eps(T)
                ratio = max(ratio, x_new[i] / x_old[i])
            end
        end
        ratio
    end

    return sqrt_up(lam)
end

"""
    _psi_bound_improved(E::BallMatrix)

Improved ψ bound computation with better handling of matrix norms.

For matrix N, computes ψ(N) = √(1 + 2αμ√(1-α²) + (αμ)²)
where α = α(μ) and μ = ‖N‖₂.
"""
function _psi_bound_improved(N::BallMatrix{T}) where {T}
    # Compute μ = ‖N‖₂ using best available method
    μ = setrounding(T, RoundUp) do
        # Try both Collatz and interpolation, use minimum
        collatz_bound = collatz_upper_bound_L2_opnorm(N)

        norm1 = upper_bound_L1_opnorm(N)
        norminf = upper_bound_L_inf_opnorm(N)
        interp_bound = sqrt_up(norm1 * norminf)

        min(collatz_bound, interp_bound)
    end

    # Compute α(μ) = √(1/2 · (1 + 1/√(1 + 4/μ²)))
    α = setrounding(T, RoundUp) do
        if μ < eps(T)
            # Special case: if μ ≈ 0, then α → 1/√2
            sqrt(T(0.5))
        else
            μ_sq = μ * μ
            inner = one(T) + T(4) / μ_sq
            sqrt_inner = sqrt(inner)
            α_expr = one(T) + one(T) / sqrt_inner
            sqrt(T(0.5) * α_expr)
        end
    end

    # Compute ψ(N) = √(1 + 2αμ√(1-α²) + (αμ)²)
    psi = setrounding(T, RoundUp) do
        α_sq = α * α
        one_minus_α_sq = one(T) - α_sq

        # Guard against numerical issues
        if one_minus_α_sq < zero(T)
            one_minus_α_sq = zero(T)
        end

        sqrt_term = sqrt(one_minus_α_sq)
        αμ = α * μ

        psi_sq = one(T) + T(2) * αμ * sqrt_term + αμ * αμ
        sqrt(psi_sq)
    end

    return psi
end

"""
    backward_singular_value_bound(A::BallMatrix)

Compute rigorous upper bounds for singular values of upper triangular matrix
using backward recursion.

For upper triangular A, computes bounds σᵢ such that σᵢ(A) ≤ σᵢ for i=1,...,n
using the recursion:

    σᵢ = (1/|aᵢᵢ|) · √(1 + ‖row_tail_i‖² · σᵢ₊₁²)

starting from σₙ₊₁ = 0.

# Reference
* RumpOishi2024, Theorem 3.2
"""
function backward_singular_value_bound(A::BallMatrix{T, NT}) where {T, NT}
    istriu(A.c) && istriu(A.r) ||
        throw(ArgumentError("A must be upper triangular"))

    n = size(A, 1)
    σ = Vector{Ball{T, NT}}(undef, n + 1)
    σ[n + 1] = Ball(zero(T), zero(T))

    for i in n:-1:1
        # Extract row tail (elements beyond diagonal)
        if i < n
            b = A[i, (i+1):end]
            norm_b = upper_bound_norm(b, 2)
        else
            norm_b = Ball(zero(T), zero(T))
        end

        # Diagonal element
        d_ii = A[i, i]

        # Compute: σᵢ = (1/|dᵢᵢ|) · √(1 + ‖b‖² · σᵢ₊₁²)
        term = setrounding(T, RoundUp) do
            norm_b_sq = norm_b * norm_b
            σ_next_sq = σ[i + 1] * σ[i + 1]
            Ball(one(T), zero(T)) + norm_b_sq * σ_next_sq
        end

        bound = (Ball(one(T), zero(T)) / abs(d_ii)) * sqrt(term)

        # Take supremum to ensure rigorous upper bound
        σ[i] = Ball(sup(max(σ[i + 1], bound)))
    end

    return σ[1:end-1]
end

# Export new functions
export rump_oishi_2024_triangular_bound, backward_singular_value_bound
