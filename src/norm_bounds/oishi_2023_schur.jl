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
- `certified`: Whether all conditions of Theorem 1 are satisfied
- `block_size`: The block size m used

The flag is called `certified` since version 0.4 and was called `verified` in 0.3.0; the
old name still resolves, with a deprecation warning, and [`iscertified`](@ref) reads it in either
version.
"""
struct Oishi2023Result{T}
    sigma_min_lower::T
    G_inv_upper::T
    A_inv_norm::T
    A_inv_B_norm::T
    C_A_inv_norm::T
    schur_contraction::T
    certified::Bool
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
- `certified`: Whether the method succeeded (only requires schur_contraction < 1)
- `block_size`: The block size m used

# Reference
S. M. Rump and S. Oishi, Japan J. Indust. Appl. Math. **41**(2) (2024), 1097-1104
([RumpOishi2024](@cite)), Lemma 1.2 and Theorem 1.3.

The flag is called `certified` since version 0.4 and was called `verified` in 0.3.0; the
old name still resolves, with a deprecation warning, and [`iscertified`](@ref) reads it in either
version.
"""
struct RumpOishi2024Result{T}
    sigma_min_lower::T
    G_inv_upper::T
    A_inv_norm::T
    psi_A_inv_B::T
    psi_C_A_inv::T
    schur_contraction::T
    used_fast_gamma::Bool
    certified::Bool
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
S. Oishi, *Lower bounds for the smallest singular values of generalized asymptotic diagonal
dominant matrices*, Japan J. Indust. Appl. Math. **40** (2023), 1569-1585,
doi 10.1007/s13160-023-00596-5 ([Oishi2023](@cite)), Theorem 1.
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
    Dd_inv_norm = _inverse_diagonal_norm(Dd_diag, D_rad)

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
    # the denominators are rounded down and the quotients up
    schur_term = div_up(Dd_inv_norm, sub_down(one(T), schur_contraction))
    denominator = mul_down(sub_down(one(T), A_inv_B_norm), sub_down(one(T), C_A_inv_norm))
    G_inv_upper = div_up(max(A_inv_norm, schur_term), denominator)

    # Lower bound on σ_min = 1/‖G⁻¹‖₂
    sigma_min_lower = div_down(one(T), G_inv_upper)

    return Oishi2023Result{T}(
        sigma_min_lower, G_inv_upper, A_inv_norm, A_inv_B_norm,
        C_A_inv_norm, schur_contraction, true, m
    )
end

"""
    _solve_ball_system(A::BallMatrix, B::BallMatrix)

An enclosure of `A⁻¹B`, as a `BallMatrix` whose midpoint `X̃` is the floating-point solution of
`mid(A) X = mid(B)` and each of whose radii is `‖A⁻¹‖₂ ‖R‖₂` for `R` the residual `B − A X̃`
computed in ball arithmetic.

The solve itself is not trusted: for every `M` of the ball `A` and `N` of the ball `B` the exact
solution `X` of `M X = N` satisfies `X − X̃ = M⁻¹(N − M X̃)`, and `N − M X̃` is enclosed by `R`, so
`‖X − X̃‖₂ ≤ ‖M⁻¹‖₂ ‖R‖₂`. The modulus of an entry of a matrix is at most its ℓ² operator norm,
so that single number bounds every entry of `X − X̃` and is used as the radius of each. The two
factors are bounded uniformly over the balls by [`svd_bound_L2_opnorm_inverse`](@ref) and
[`collatz_upper_bound_L2_opnorm`](@ref), and their product is rounded up. The radius is `Inf`
when either factor is, which is what happens when the ball `A` may contain a singular matrix.
"""
function _solve_ball_system(A::BallMatrix{T}, B::BallMatrix{T}) where {T}
    # Approximate solution using midpoints
    # NOTE: The `\` solve here is an ORACLE - the approximate solution is used
    # as a starting point, and its quality is verified A POSTERIORI via the
    # residual-based error bound below. No directed rounding needed for the solve.
    A_mid = mid(A)
    B_mid = mid(B)

    X_approx = A_mid \ B_mid

    # Compute residual R = B - A*X_approx
    R = B - A * BallMatrix(X_approx)

    # Estimate error bound
    # ‖X - X_approx‖ ≤ ‖A⁻¹‖ * ‖R‖
    A_inv_norm = svd_bound_L2_opnorm_inverse(A)
    R_norm = collatz_upper_bound_L2_opnorm(R)

    error_bound = mul_up(A_inv_norm, R_norm)

    # Return X with inflated radius
    X_rad = fill(error_bound, size(X_approx))

    return BallMatrix(X_approx, X_rad)
end

"""
    _solve_ball_system_transpose(A::BallMatrix, C::BallMatrix)

An enclosure of `C A⁻¹`, obtained as [`_solve_ball_system`](@ref) does but from the other side:
the midpoint is the floating-point solution `X̃'` of `mid(A)' X = mid(C)'`, the residual
`R = C − X̃' A` is computed in ball arithmetic, and each radius is `‖A⁻¹‖₂ ‖R‖₂`, which bounds
every entry of `C M⁻¹ − X̃'` for every `M` of the ball `A` and every `C` of its ball since
`C M⁻¹ − X̃' = (C − X̃' M) M⁻¹`.
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

    error_bound = mul_up(A_inv_norm, R_norm)

    X_rad = fill(error_bound, size(CA_inv_approx))

    return BallMatrix(CA_inv_approx, X_rad)
end

"""
    _diagonal_scale_left(d::Vector, d_rad::Matrix, M::BallMatrix)

An enclosure of `Δ⁻¹ N`, where `Δ` is the diagonal matrix whose `i`-th entry lies in the ball of
centre `d[i]` and radius `d_rad[i, i]` and `N` is any matrix of the ball `M`. Row `i` is the
product of the ball `inv(Ball(d[i], d_rad[i, i]))` with the balls of row `i` of `M`, so the
rounding of the inverse and of the products is carried in the radii.

`d_rad` is passed as the radius matrix of the block whose diagonal `d` is, and only its diagonal
is read. A row whose ball may contain zero gets radius `Inf` rather than raising: no bound of
`Δ⁻¹` exists there, and that is reported through the enclosure.
"""
function _diagonal_scale_left(d::Vector{<:Union{T, Complex{T}}}, d_rad::AbstractMatrix{T},
                               M::BallMatrix{T}) where {T}
    m, n_cols = size(M)
    length(d) == m || throw(DimensionMismatch("d must have length equal to rows of M"))
    M_mid, M_rad = mid(M), rad(M)
    result_mid = similar(M_mid)
    result_rad = similar(M_rad)
    for i in 1:m
        if sub_down(abs_down(d[i]), d_rad[i, i]) ≤ zero(T)
            # the diagonal entry may vanish: no bound of the inverse
            result_mid[i, :] .= zero(T)
            result_rad[i, :] .= T(Inf)
            continue
        end
        # the product of balls accounts for the rounding of the midpoint
        dinv = inv(Ball(d[i], d_rad[i, i]))
        for j in 1:n_cols
            x = dinv * Ball(M_mid[i, j], M_rad[i, j])
            result_mid[i, j], result_rad[i, j] = mid(x), rad(x)
        end
    end
    return BallMatrix(result_mid, result_rad)
end

"""
    _inverse_diagonal_norm(d, d_rad)

An upper bound of `max_i 1/|x_i|` over `|x_i − d[i]| ≤ d_rad[i, i]`, the 2-norm of the inverse of
the diagonal matrix; `Inf` where an entry may vanish. The lower bound of `|x_i|` is rounded down.
"""
function _inverse_diagonal_norm(d::Vector{<:Union{T, Complex{T}}},
        d_rad::AbstractMatrix{T}) where {T}
    bound = zero(T)
    for i in eachindex(d)
        lower = sub_down(abs_down(d[i]), d_rad[i, i])
        lower ≤ zero(T) && return T(Inf)
        bound = max(bound, div_up(one(T), lower))
    end
    return bound
end

"""
    oishi_2023_optimal_block_size(G::BallMatrix{T}; max_m::Int=100) where {T}

[`oishi_2023_sigma_min_bound`](@ref) for every block size `m = 1, …, min(max_m, n − 1)`, and the
`m` whose certified lower bound of `σ_min(G)` is the largest, as `(best_m, best_result)`. Each
certified result is a valid bound, so the largest is; when none is certified, `best_result` is
the uncertified one for `m = 1`. The statement and its reference are those of
`oishi_2023_sigma_min_bound`.
"""
function oishi_2023_optimal_block_size(G::BallMatrix{T}; max_m::Int=100) where {T}
    n = size(G, 1)
    max_m = min(max_m, n - 1)

    best_m = 1
    best_result = oishi_2023_sigma_min_bound(G, 1)

    for m in 2:max_m
        result = oishi_2023_sigma_min_bound(G, m)
        if result.certified && result.sigma_min_lower > best_result.sigma_min_lower
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

Compute ψ(μ) from Lemma 1.2 of [RumpOishi2024](@cite).

For a block triangular matrix H = [I, -N; 0, I] with ‖N‖ = μ, this computes
the exact value of ‖H‖ = ‖H⁻¹‖.

The paper's formula (5) is

    α = √(½(1 + 1/√(1 + 4/μ²))),      ψ(μ) = √(1 + 2αμ√(1-α²) + α²μ²).

It is evaluated here in the equal form `ψ(μ) = (μ + √(μ² + 4))/2`: `β = α²` solves
`β² − β + 1/(4 + μ²) = 0` (the proof of Lemma 1.2), so `α√(1-α²) = 1/√(4 + μ²)` and, with
`t = μ/√(μ² + 4)`, `β = (1 + t)/2`, whence `ψ² = 1 + 2t + μ²(1 + t)/2 = ((μ + √(μ² + 4))/2)²`.
Formula (5) is the maximum over `α` of the quantity (6), so an `α` carrying a rounding error
gives a value below `ψ`; the closed form has no such step.

This replaces the weaker bound 1/(1-μ) used in Oishi 2023, and works for any μ ≥ 0. By (10) of
the paper an upper bound of `‖N‖` may be passed for `μ`.

# Returns
An upper bound of ψ(μ).
"""
function psi_schur_factor(μ::T) where {T<:AbstractFloat}
    μ ≤ zero(T) && return one(T)
    # ψ(μ) = (μ + √(μ² + 4))/2, increasing in μ, every operation rounded up
    return add_up(μ, sqrt_up(add_up(mul_up(μ, μ), T(4)))) / 2
end

"""
    pi_norm(M::BallMatrix{T}) where {T}

An upper bound of the ℓ² operator norm of every matrix of the ball `M`, `π(M) = √(‖M‖₁‖M‖_∞)`,
from the largest column sum and the largest row sum of the moduli of its entries. It is the bound
written `π` in [RumpOishi2024](@cite) and used there for the cheap `γ` of equation (9); the same
inequality appears among the immediate bounds of Section 3 of [Rump2011](@cite).
"""
function pi_norm(M::BallMatrix{T}) where {T}
    norm1 = upper_bound_L1_opnorm(M)
    norminf = upper_bound_L_inf_opnorm(M)
    return sqrt_up(mul_up(norm1, norminf))
end

"""
    _fast_gamma_bound(A::BallMatrix{T}, B::BallMatrix{T}, C::BallMatrix{T},
                      Dd_diag::Vector{T}, D_rad::Matrix{T}, Df::BallMatrix{T}) where {T}

Equation (9) of [RumpOishi2024](@cite):

    ‖D_d⁻¹(D_f − CA⁻¹B)‖₂ ≤ π(D_d⁻¹D_f) + π(D_d⁻¹C) π(A⁻¹B) =: γ,

with `π(N) = √(‖N‖₁‖N‖_∞)` ([`pi_norm`](@ref)) an upper bound of `‖N‖₂`. So `γ < 1` establishes
the hypothesis (7) of Theorem 1.3 without forming the product `CA⁻¹B`, which brings the cost of
the test from `O((n−m)²m)` to `O((n−m)m²)` operations.

The three factors are enclosed by [`_diagonal_scale_left`](@ref) and
[`_solve_ball_system`](@ref), so the inequality is established for every matrix of the ball `G`;
the sum and the product are rounded up.

# Returns

`(γ, A_inv_B)`, the second being the enclosure of `A⁻¹B` that the caller reuses, or
`(Inf, nothing)` when one of the three bounds is not finite.
"""
function _fast_gamma_bound(A::BallMatrix{T}, B::BallMatrix{T}, C::BallMatrix{T},
                           Dd_diag::Vector{<:Union{T, Complex{T}}}, D_rad::AbstractMatrix{T},
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
    γ = add_up(pi_Dd_inv_Df, mul_up(pi_Dd_inv_C, pi_A_inv_B))

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
S. M. Rump and S. Oishi, *A note on Oishi's lower bound for the smallest singular value of
linearized Galerkin equations*, Japan J. Indust. Appl. Math. **41**(2) (2024), 1097-1104,
doi 10.1007/s13160-024-00645-7 ([RumpOishi2024](@cite)), Lemma 1.2 and Theorem 1.3.
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
    Dd_inv_norm = _inverse_diagonal_norm(Dd_diag, D_rad)

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
            schur_term = div_up(Dd_inv_norm, sub_down(one(T), γ))
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
    schur_term = div_up(Dd_inv_norm, sub_down(one(T), schur_contraction))
    G_inv_upper = mul_up(mul_up(max(A_inv_norm, schur_term), psi_A_inv_B), psi_C_A_inv)

    # Lower bound on σ_min = 1/‖G⁻¹‖₂
    sigma_min_lower = div_down(one(T), G_inv_upper)

    return RumpOishi2024Result{T}(
        sigma_min_lower, G_inv_upper, A_inv_norm, psi_A_inv_B,
        psi_C_A_inv, schur_contraction, used_fast_gamma, true, m
    )
end

"""
    rump_oishi_2024_optimal_block_size(G::BallMatrix{T}; max_m::Int=100,
                                        try_fast_gamma::Bool=true) where {T}

[`rump_oishi_2024_sigma_min_bound`](@ref) for every block size `m = 1, …, min(max_m, n − 1)`, and
the `m` whose certified lower bound of `σ_min(G)` is the largest, as `(best_m, best_result)`.
Each certified result is a valid bound, so the largest is; when none is certified, `best_result`
is the uncertified one for `m = 1`. The statement and its reference are those of
`rump_oishi_2024_sigma_min_bound`.
"""
function rump_oishi_2024_optimal_block_size(G::BallMatrix{T}; max_m::Int=100,
                                             try_fast_gamma::Bool=true) where {T}
    n = size(G, 1)
    max_m = min(max_m, n - 1)

    best_m = 1
    best_result = rump_oishi_2024_sigma_min_bound(G, 1; try_fast_gamma=try_fast_gamma)

    for m in 2:max_m
        result = rump_oishi_2024_sigma_min_bound(G, m; try_fast_gamma=try_fast_gamma)
        if result.certified && result.sigma_min_lower > best_result.sigma_min_lower
            best_m = m
            best_result = result
        end
    end

    return (best_m, best_result)
end
