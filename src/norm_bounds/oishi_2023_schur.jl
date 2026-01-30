# Implementation of Oishi 2023: Lower bounds for the smallest singular values
# of generalized asymptotic diagonal dominant matrices
#
# Reference: Oishi, S. (2023), "Lower bounds for the smallest singular values of
# generalized asymptotic diagonal dominant matrices", Japan J. Indust. Appl. Math.
# 40:1569-1585, https://doi.org/10.1007/s13160-023-00596-5
#
# Enhanced with Rump-Oishi 2024 improvements:
# Reference: Rump, S.M. & Oishi, S. (2024), "A Note on Oishi's Lower Bound for
# the Smallest Singular Value of Linearized Galerkin Equations"

"""
    Oishi2023Result{T}

Result from the Oishi 2023 Schur complement bound computation.

# Fields
- `sigma_min_lower`: Rigorous lower bound on the minimum singular value
- `G_inv_upper`: Upper bound on ‖G⁻¹‖₂ (= 1/σ_min)
- `A_inv_norm`: Upper bound on ‖A⁻¹‖₂
- `A_inv_B_norm`: Upper bound on ‖A⁻¹B‖₂
- `C_A_inv_norm`: Upper bound on ‖CA⁻¹‖₂
- `schur_contraction`: Upper bound on ‖D_d⁻¹(D_f - CA⁻¹B)‖₂
- `verified`: Whether all conditions of Theorem 1 are satisfied
- `block_size`: The block size m used
"""
struct Oishi2023Result{T}
    sigma_min_lower::T
    G_inv_upper::T
    A_inv_norm::T
    A_inv_B_norm::T
    C_A_inv_norm::T
    schur_contraction::T
    verified::Bool
    block_size::Int
end

"""
    RumpOishi2024Result{T}

Result from the Rump-Oishi 2024 improved Schur complement bound computation.

This improves upon Oishi 2023 by:
1. Removing the conditions ‖A⁻¹B‖ < 1 and ‖CA⁻¹‖ < 1
2. Using the exact ψ(N) formula instead of 1/(1-‖N‖)
3. Optionally using a fast γ bound to avoid expensive matrix products

# Fields
- `sigma_min_lower`: Rigorous lower bound on the minimum singular value
- `G_inv_upper`: Upper bound on ‖G⁻¹‖₂ (= 1/σ_min)
- `A_inv_norm`: Upper bound on ‖A⁻¹‖₂
- `psi_A_inv_B`: ψ(‖A⁻¹B‖₂) factor
- `psi_C_A_inv`: ψ(‖CA⁻¹‖₂) factor
- `schur_contraction`: Upper bound on ‖D_d⁻¹(D_f - CA⁻¹B)‖₂
- `used_fast_gamma`: Whether the fast γ bound was used
- `verified`: Whether the method succeeded (only requires schur_contraction < 1)
- `block_size`: The block size m used

# Reference
Rump, S.M. & Oishi, S. (2024), "A Note on Oishi's Lower Bound for the Smallest
Singular Value of Linearized Galerkin Equations"
"""
struct RumpOishi2024Result{T}
    sigma_min_lower::T
    G_inv_upper::T
    A_inv_norm::T
    psi_A_inv_B::T
    psi_C_A_inv::T
    schur_contraction::T
    used_fast_gamma::Bool
    verified::Bool
    block_size::Int
end

"""
    oishi_2023_sigma_min_bound(G::BallMatrix{T}, m::Int) where {T}

Compute a rigorous lower bound on the minimum singular value of G using
the Schur complement method from Oishi (2023).

# Arguments
- `G::BallMatrix`: The matrix to analyze
- `m::Int`: Block size for the partition (A is m×m)

# Method (Theorem 1 from Oishi 2023)

Partition G as:
```
G = [A  B]
    [C  D]
```
where A ∈ M_m, B ∈ M_{m,n-m}, C ∈ M_{n-m,m}, D ∈ M_{n-m,n-m}.

Let D_d and D_f be the diagonal and off-diagonal parts of D.

If the following conditions hold:
1. ‖A⁻¹B‖₂ < 1
2. ‖CA⁻¹‖₂ < 1
3. ‖D_d⁻¹(D_f - CA⁻¹B)‖₂ < 1

Then G is invertible and:

```
‖G⁻¹‖₂ ≤ max{‖A⁻¹‖₂, ‖D_d⁻¹‖₂/(1 - ‖D_d⁻¹(D_f - CA⁻¹B)‖₂)}
         / ((1 - ‖A⁻¹B‖₂)(1 - ‖CA⁻¹‖₂))
```

Since σ_min(G) = 1/‖G⁻¹‖₂, this gives a lower bound on σ_min.

# Returns
`Oishi2023Result` containing the bounds and verification status.

# Reference
Oishi, S. (2023), Japan J. Indust. Appl. Math. 40:1569-1585
"""
function oishi_2023_sigma_min_bound(G::BallMatrix{T}, m::Int) where {T}
    n = size(G, 1)
    size(G, 1) == size(G, 2) || throw(ArgumentError("G must be square"))
    1 ≤ m < n || throw(ArgumentError("m must satisfy 1 ≤ m < n"))

    # Extract blocks: G = [A B; C D]
    A = G[1:m, 1:m]
    B = G[1:m, (m+1):n]
    C = G[(m+1):n, 1:m]
    D = G[(m+1):n, (m+1):n]

    # Split D into diagonal and off-diagonal parts
    D_mid = mid(D)
    D_rad = rad(D)
    Dd_diag = diag(D_mid)

    # Check if A and D_d are invertible (diagonal elements non-zero)
    # For rigorous computation, we need the diagonal to be bounded away from zero

    # Compute ‖A⁻¹‖₂ using SVD (with fallback for BigFloat)
    A_inv_norm = try
        svd_bound_L2_opnorm_inverse(A)
    catch
        # SVD computation may fail for BigFloat or other types; return unverified result
        return Oishi2023Result{T}(
            zero(T), T(Inf), T(Inf), T(Inf), T(Inf), T(Inf), false, m
        )
    end

    # Compute A⁻¹B
    A_inv_B = _solve_ball_system(A, B)
    A_inv_B_norm = collatz_upper_bound_L2_opnorm(A_inv_B)

    # Compute CA⁻¹ = (A⁻ᵀC')'
    # We solve A'X = C' then take X'
    A_inv_T_CT = _solve_ball_system_transpose(A, C)
    C_A_inv_norm = collatz_upper_bound_L2_opnorm(A_inv_T_CT)

    # Compute D_d⁻¹ norm (diagonal inverse)
    Dd_inv_norm = setrounding(T, RoundUp) do
        max_inv = zero(T)
        for i in 1:length(Dd_diag)
            d_i = Dd_diag[i]
            r_i = D_rad[i, i]
            # Lower bound on |d_i|
            d_lower = abs(d_i) - r_i
            if d_lower ≤ zero(T)
                return T(Inf)  # Diagonal element could be zero
            end
            max_inv = max(max_inv, one(T) / d_lower)
        end
        max_inv
    end

    if !isfinite(Dd_inv_norm)
        return Oishi2023Result{T}(
            zero(T), T(Inf), A_inv_norm, A_inv_B_norm, C_A_inv_norm,
            T(Inf), false, m
        )
    end

    # Compute D_f = D - D_d (off-diagonal part)
    Df_mid = copy(D_mid)
    Df_rad = copy(D_rad)
    for i in 1:size(D, 1)
        Df_mid[i, i] = zero(T)
        Df_rad[i, i] = zero(T)
    end
    Df = BallMatrix(Df_mid, Df_rad)

    # Compute CA⁻¹B
    C_A_inv_B = C * A_inv_B

    # Compute D_f - CA⁻¹B
    Df_minus_CAinvB = Df - C_A_inv_B

    # Compute D_d⁻¹(D_f - CA⁻¹B)
    # This is diagonal scaling: each row i multiplied by 1/D_d[i,i]
    Dd_inv_term = _diagonal_scale_left(Dd_diag, D_rad, Df_minus_CAinvB)

    # Compute ‖D_d⁻¹(D_f - CA⁻¹B)‖₂
    schur_contraction = collatz_upper_bound_L2_opnorm(Dd_inv_term)

    # Check conditions of Theorem 1
    cond1 = A_inv_B_norm < one(T)
    cond2 = C_A_inv_norm < one(T)
    cond3 = schur_contraction < one(T)

    verified = cond1 && cond2 && cond3

    if !verified
        return Oishi2023Result{T}(
            zero(T), T(Inf), A_inv_norm, A_inv_B_norm, C_A_inv_norm,
            schur_contraction, false, m
        )
    end

    # Compute the bound from Theorem 1 (Equation 12)
    G_inv_upper = setrounding(T, RoundUp) do
        # Numerator: max{‖A⁻¹‖₂, ‖D_d⁻¹‖₂/(1 - schur_contraction)}
        term1 = A_inv_norm
        term2 = Dd_inv_norm / (one(T) - schur_contraction)
        numerator = max(term1, term2)

        # Denominator: (1 - ‖A⁻¹B‖₂)(1 - ‖CA⁻¹‖₂)
        denominator = (one(T) - A_inv_B_norm) * (one(T) - C_A_inv_norm)

        numerator / denominator
    end

    # Lower bound on σ_min = 1/‖G⁻¹‖₂
    sigma_min_lower = setrounding(T, RoundDown) do
        one(T) / G_inv_upper
    end

    return Oishi2023Result{T}(
        sigma_min_lower, G_inv_upper, A_inv_norm, A_inv_B_norm,
        C_A_inv_norm, schur_contraction, true, m
    )
end

"""
    _solve_ball_system(A::BallMatrix, B::BallMatrix)

Solve AX = B for X using the midpoint approximation and error bounds.
Returns a BallMatrix enclosure of X.
"""
function _solve_ball_system(A::BallMatrix{T}, B::BallMatrix{T}) where {T}
    # Approximate solution using midpoints
    A_mid = mid(A)
    B_mid = mid(B)

    X_approx = A_mid \ B_mid

    # Compute residual R = B - A*X_approx
    R = B - A * BallMatrix(X_approx)

    # Estimate error bound
    # ‖X - X_approx‖ ≤ ‖A⁻¹‖ * ‖R‖
    A_inv_norm = svd_bound_L2_opnorm_inverse(A)
    R_norm = collatz_upper_bound_L2_opnorm(R)

    error_bound = setrounding(T, RoundUp) do
        A_inv_norm * R_norm
    end

    # Return X with inflated radius
    X_rad = setrounding(T, RoundUp) do
        fill(error_bound, size(X_approx))
    end

    return BallMatrix(X_approx, X_rad)
end

"""
    _solve_ball_system_transpose(A::BallMatrix, C::BallMatrix)

Solve A'X = C' for X, returning X' (which equals CA⁻¹).
"""
function _solve_ball_system_transpose(A::BallMatrix{T}, C::BallMatrix{T}) where {T}
    # A'X = C' means X = A⁻ᵀC', so X' = CA⁻¹
    A_mid = mid(A)
    C_mid = mid(C)

    # Solve A'X = C'
    X_approx = A_mid' \ C_mid'

    # X' = CA⁻¹
    CA_inv_approx = X_approx'

    # Compute residual
    R = C - BallMatrix(CA_inv_approx) * A

    # Estimate error
    A_inv_norm = svd_bound_L2_opnorm_inverse(A)
    R_norm = collatz_upper_bound_L2_opnorm(R)

    error_bound = setrounding(T, RoundUp) do
        A_inv_norm * R_norm
    end

    X_rad = fill(error_bound, size(CA_inv_approx))

    return BallMatrix(CA_inv_approx, X_rad)
end

"""
    _diagonal_scale_left(d::Vector, d_rad::Matrix, M::BallMatrix)

Compute diag(1./d) * M with rigorous bounds, where d are the diagonal elements
and d_rad contains the radii of the diagonal.
"""
function _diagonal_scale_left(d::Vector{T}, d_rad::AbstractMatrix{T},
                               M::BallMatrix{T}) where {T}
    m, n_cols = size(M)
    length(d) == m || throw(DimensionMismatch("d must have length equal to rows of M"))

    M_mid = mid(M)
    M_rad = rad(M)

    result_mid = similar(M_mid)
    result_rad = similar(M_rad)

    for i in 1:m
        d_i = d[i]
        r_i = d_rad[i, i]

        # Interval for 1/d[i]
        d_lower = abs(d_i) - r_i
        d_upper = abs(d_i) + r_i

        if d_lower ≤ zero(T)
            # Cannot bound the inverse
            for j in 1:n_cols
                result_mid[i, j] = zero(T)
                result_rad[i, j] = T(Inf)
            end
        else
            # 1/d is in [1/d_upper, 1/d_lower] if d > 0
            # or [-1/d_lower, -1/d_upper] if d < 0
            inv_mid = one(T) / d_i
            inv_rad = setrounding(T, RoundUp) do
                max(abs(one(T)/d_lower - inv_mid), abs(one(T)/d_upper - inv_mid))
            end

            for j in 1:n_cols
                m_ij = M_mid[i, j]
                r_ij = M_rad[i, j]

                # (inv_mid ± inv_rad) * (m_ij ± r_ij)
                result_mid[i, j] = inv_mid * m_ij
                result_rad[i, j] = setrounding(T, RoundUp) do
                    abs(inv_mid) * r_ij + inv_rad * abs(m_ij) + inv_rad * r_ij
                end
            end
        end
    end

    return BallMatrix(result_mid, result_rad)
end

"""
    oishi_2023_optimal_block_size(G::BallMatrix{T}; max_m::Int=100) where {T}

Find the optimal block size m that gives the tightest bound on σ_min.

Returns a tuple (best_m, best_result).
"""
function oishi_2023_optimal_block_size(G::BallMatrix{T}; max_m::Int=100) where {T}
    n = size(G, 1)
    max_m = min(max_m, n - 1)

    best_m = 1
    best_result = oishi_2023_sigma_min_bound(G, 1)

    for m in 2:max_m
        result = oishi_2023_sigma_min_bound(G, m)
        if result.verified && result.sigma_min_lower > best_result.sigma_min_lower
            best_m = m
            best_result = result
        end
    end

    return (best_m, best_result)
end

#==============================================================================#
# Rump-Oishi 2024 Improvements
#==============================================================================#

"""
    psi_schur_factor(μ::T) where {T}

Compute ψ(μ) from Lemma 1.2 of Rump-Oishi 2024.

For a block triangular matrix H = [I, -N; 0, I] with ‖N‖ = μ, this computes
the exact value of ‖H‖ = ‖H⁻¹‖.

The formula is:
    α = √(½(1 + 1/√(1 + 4/μ²)))
    ψ(μ) = √(1 + 2αμ√(1-α²) + α²μ²)

This replaces the weaker bound 1/(1-μ) used in Oishi 2023, and works for
any μ ≥ 0 (no restriction μ < 1).

# Returns
ψ(μ) as an upper bound (computed with directed rounding).
"""
function psi_schur_factor(μ::T) where {T<:AbstractFloat}
    if μ ≤ zero(T)
        return one(T)
    end

    setrounding(T, RoundUp) do
        # α = √(½(1 + 1/√(1 + 4/μ²)))
        μ_sq = μ * μ
        inner = one(T) + T(4) / μ_sq
        sqrt_inner = sqrt(inner)
        α_sq = T(0.5) * (one(T) + one(T) / sqrt_inner)
        α = sqrt(α_sq)

        # ψ(μ) = √(1 + 2αμ√(1-α²) + α²μ²)
        one_minus_α_sq = one(T) - α_sq
        # Guard against numerical issues (α ∈ (0,1) so this should be positive)
        if one_minus_α_sq < zero(T)
            one_minus_α_sq = zero(T)
        end
        sqrt_term = sqrt(one_minus_α_sq)
        αμ = α * μ

        psi_sq = one(T) + T(2) * αμ * sqrt_term + αμ * αμ
        sqrt(psi_sq)
    end
end

"""
    pi_norm(M::BallMatrix{T}) where {T}

Compute π(M) = √(‖M‖₁ · ‖M‖∞), an upper bound on ‖M‖₂.

This is a cheap bound useful for the fast γ estimate in Rump-Oishi 2024.
"""
function pi_norm(M::BallMatrix{T}) where {T}
    norm1 = upper_bound_L1_opnorm(M)
    norminf = upper_bound_L_inf_opnorm(M)
    setrounding(T, RoundUp) do
        sqrt(norm1 * norminf)
    end
end

"""
    _fast_gamma_bound(A::BallMatrix{T}, B::BallMatrix{T}, C::BallMatrix{T},
                      Dd_diag::Vector{T}, D_rad::Matrix{T}, Df::BallMatrix{T}) where {T}

Compute the fast γ bound from equation (9) of Rump-Oishi 2024:

    γ = π(Dd⁻¹Df) + π(Dd⁻¹C) · π(A⁻¹B)

where π(N) = √(‖N‖₁ · ‖N‖∞).

This avoids computing the expensive product Dd⁻¹(Df - CA⁻¹B) and reduces
complexity from O((n-m)²m) to O((n-m)m²).

# Returns
A tuple (γ, A_inv_B) where A_inv_B is the computed A⁻¹B (reused later).
Returns (Inf, nothing) if computation fails.
"""
function _fast_gamma_bound(A::BallMatrix{T}, B::BallMatrix{T}, C::BallMatrix{T},
                           Dd_diag::Vector{T}, D_rad::AbstractMatrix{T},
                           Df::BallMatrix{T}) where {T}
    # Compute Dd⁻¹Df using diagonal scaling
    Dd_inv_Df = _diagonal_scale_left(Dd_diag, D_rad, Df)
    pi_Dd_inv_Df = pi_norm(Dd_inv_Df)

    if !isfinite(pi_Dd_inv_Df)
        return (T(Inf), nothing)
    end

    # Compute Dd⁻¹C
    Dd_inv_C = _diagonal_scale_left(Dd_diag, D_rad, C)
    pi_Dd_inv_C = pi_norm(Dd_inv_C)

    if !isfinite(pi_Dd_inv_C)
        return (T(Inf), nothing)
    end

    # Compute A⁻¹B
    A_inv_B = _solve_ball_system(A, B)
    pi_A_inv_B = pi_norm(A_inv_B)

    if !isfinite(pi_A_inv_B)
        return (T(Inf), nothing)
    end

    # γ = π(Dd⁻¹Df) + π(Dd⁻¹C) · π(A⁻¹B)
    γ = setrounding(T, RoundUp) do
        pi_Dd_inv_Df + pi_Dd_inv_C * pi_A_inv_B
    end

    return (γ, A_inv_B)
end

"""
    rump_oishi_2024_sigma_min_bound(G::BallMatrix{T}, m::Int;
                                     try_fast_gamma::Bool=true) where {T}

Compute a rigorous lower bound on the minimum singular value of G using
the improved Schur complement method from Rump-Oishi (2024).

# Arguments
- `G::BallMatrix`: The matrix to analyze
- `m::Int`: Block size for the partition (A is m×m)
- `try_fast_gamma::Bool=true`: Whether to try the fast γ bound first

# Improvements over Oishi 2023

1. **Removes conditions 1 & 2**: No longer requires ‖A⁻¹B‖ < 1 and ‖CA⁻¹‖ < 1.
   Only condition 3 (‖Dd⁻¹(Df - CA⁻¹B)‖ < 1) is needed.

2. **Uses exact ψ(N) formula**: Instead of 1/(1-‖N‖), uses
   ψ(μ) = √(1 + 2αμ√(1-α²) + α²μ²)
   which is tighter and works for any μ ≥ 0.

3. **Fast γ bound** (optional): Uses π(N) = √(‖N‖₁·‖N‖∞) to quickly check
   if the method will succeed, avoiding expensive matrix products.

# Method (Theorem 1.3 from Rump-Oishi 2024)

Partition G as:
```
G = [A  B]
    [C  D]
```

If ‖Dd⁻¹(Df - CA⁻¹B)‖ < 1, then:

```
‖G⁻¹‖ ≤ max{‖A⁻¹‖, ‖Dd⁻¹‖/(1 - ‖Dd⁻¹(Df - CA⁻¹B)‖)} · ψ(‖A⁻¹B‖) · ψ(‖CA⁻¹‖)
```

# Returns
`RumpOishi2024Result` containing the bounds and verification status.

# Reference
Rump, S.M. & Oishi, S. (2024), "A Note on Oishi's Lower Bound for the Smallest
Singular Value of Linearized Galerkin Equations"
"""
function rump_oishi_2024_sigma_min_bound(G::BallMatrix{T}, m::Int;
                                          try_fast_gamma::Bool=true) where {T}
    n = size(G, 1)
    size(G, 1) == size(G, 2) || throw(ArgumentError("G must be square"))
    1 ≤ m < n || throw(ArgumentError("m must satisfy 1 ≤ m < n"))

    # Extract blocks: G = [A B; C D]
    A = G[1:m, 1:m]
    B = G[1:m, (m+1):n]
    C = G[(m+1):n, 1:m]
    D = G[(m+1):n, (m+1):n]

    # Split D into diagonal and off-diagonal parts
    D_mid = mid(D)
    D_rad = rad(D)
    Dd_diag = diag(D_mid)

    # Build Df (off-diagonal part of D)
    Df_mid = copy(D_mid)
    Df_rad = copy(D_rad)
    for i in 1:size(D, 1)
        Df_mid[i, i] = zero(T)
        Df_rad[i, i] = zero(T)
    end
    Df = BallMatrix(Df_mid, Df_rad)

    # Compute ‖A⁻¹‖₂
    A_inv_norm = try
        svd_bound_L2_opnorm_inverse(A)
    catch
        return RumpOishi2024Result{T}(
            zero(T), T(Inf), T(Inf), T(Inf), T(Inf), T(Inf), false, false, m
        )
    end

    # Compute ‖Dd⁻¹‖ (maximum of 1/|d_i|)
    Dd_inv_norm = setrounding(T, RoundUp) do
        max_inv = zero(T)
        for i in 1:length(Dd_diag)
            d_i = Dd_diag[i]
            r_i = D_rad[i, i]
            d_lower = abs(d_i) - r_i
            if d_lower ≤ zero(T)
                return T(Inf)
            end
            max_inv = max(max_inv, one(T) / d_lower)
        end
        max_inv
    end

    if !isfinite(Dd_inv_norm)
        return RumpOishi2024Result{T}(
            zero(T), T(Inf), A_inv_norm, T(Inf), T(Inf), T(Inf), false, false, m
        )
    end

    # Try fast γ bound first if requested
    used_fast_gamma = false
    A_inv_B = nothing
    schur_contraction = T(Inf)

    if try_fast_gamma
        γ, A_inv_B_computed = _fast_gamma_bound(A, B, C, Dd_diag, D_rad, Df)
        if γ < one(T)
            # Fast bound succeeded! Check if max term is ‖A⁻¹‖
            schur_term = setrounding(T, RoundUp) do
                Dd_inv_norm / (one(T) - γ)
            end
            if schur_term ≤ A_inv_norm
                # The maximum is ‖A⁻¹‖, so we can use the fast γ bound
                used_fast_gamma = true
                schur_contraction = γ
                A_inv_B = A_inv_B_computed
            end
        end
    end

    # If fast gamma didn't work or wasn't tried, compute full Schur complement
    if !used_fast_gamma
        # Compute A⁻¹B if not already computed
        if A_inv_B === nothing
            A_inv_B = _solve_ball_system(A, B)
        end

        # Compute CA⁻¹B
        C_A_inv_B = C * A_inv_B

        # Compute Df - CA⁻¹B
        Df_minus_CAinvB = Df - C_A_inv_B

        # Compute Dd⁻¹(Df - CA⁻¹B)
        Dd_inv_term = _diagonal_scale_left(Dd_diag, D_rad, Df_minus_CAinvB)

        # Compute ‖Dd⁻¹(Df - CA⁻¹B)‖₂
        schur_contraction = collatz_upper_bound_L2_opnorm(Dd_inv_term)
    end

    # Check main condition
    if schur_contraction ≥ one(T)
        return RumpOishi2024Result{T}(
            zero(T), T(Inf), A_inv_norm, T(Inf), T(Inf),
            schur_contraction, used_fast_gamma, false, m
        )
    end

    # Compute ‖A⁻¹B‖₂ and ψ(‖A⁻¹B‖₂)
    A_inv_B_norm = collatz_upper_bound_L2_opnorm(A_inv_B)
    psi_A_inv_B = psi_schur_factor(A_inv_B_norm)

    # Compute ‖CA⁻¹‖₂ and ψ(‖CA⁻¹‖₂)
    A_inv_T_CT = _solve_ball_system_transpose(A, C)
    C_A_inv_norm = collatz_upper_bound_L2_opnorm(A_inv_T_CT)
    psi_C_A_inv = psi_schur_factor(C_A_inv_norm)

    # Compute the bound from Theorem 1.3
    G_inv_upper = setrounding(T, RoundUp) do
        # max{‖A⁻¹‖, ‖Dd⁻¹‖/(1 - schur_contraction)}
        term1 = A_inv_norm
        term2 = Dd_inv_norm / (one(T) - schur_contraction)
        max_term = max(term1, term2)

        # Multiply by ψ factors
        max_term * psi_A_inv_B * psi_C_A_inv
    end

    # Lower bound on σ_min = 1/‖G⁻¹‖₂
    sigma_min_lower = setrounding(T, RoundDown) do
        one(T) / G_inv_upper
    end

    return RumpOishi2024Result{T}(
        sigma_min_lower, G_inv_upper, A_inv_norm, psi_A_inv_B,
        psi_C_A_inv, schur_contraction, used_fast_gamma, true, m
    )
end

"""
    rump_oishi_2024_optimal_block_size(G::BallMatrix{T}; max_m::Int=100,
                                        try_fast_gamma::Bool=true) where {T}

Find the optimal block size m that gives the tightest bound on σ_min
using the Rump-Oishi 2024 method.

Returns a tuple (best_m, best_result).
"""
function rump_oishi_2024_optimal_block_size(G::BallMatrix{T}; max_m::Int=100,
                                             try_fast_gamma::Bool=true) where {T}
    n = size(G, 1)
    max_m = min(max_m, n - 1)

    best_m = 1
    best_result = rump_oishi_2024_sigma_min_bound(G, 1; try_fast_gamma=try_fast_gamma)

    for m in 2:max_m
        result = rump_oishi_2024_sigma_min_bound(G, m; try_fast_gamma=try_fast_gamma)
        if result.verified && result.sigma_min_lower > best_result.sigma_min_lower
            best_m = m
            best_result = result
        end
    end

    return (best_m, best_result)
end
