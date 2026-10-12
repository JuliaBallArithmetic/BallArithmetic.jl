# Sufficient conditions for the regularity and for the singularity of an interval matrix, after
#
#   J. Horáček, Interval linear and nonlinear systems, doctoral thesis, Charles University,
#   Prague, 2019: Theorem 4.2, Definition 4.24 and Theorems 11.12 and 11.13.
#
# A ball matrix `A` with midpoint `A_c` and radius `A_Δ` is regular when every matrix of the ball
# is nonsingular, and singular when the ball contains a singular matrix. Every routine here
# evaluates a sufficient condition with its rounding accounted for: `true` is a proof, `false`
# means that the condition was not established.

"""
    RegularityResult{T}

Result of a regularity test.

# Fields
- `certified::Bool`: `true` when regularity is proved; `false` when the test is inconclusive
- `method::Symbol`: the condition that was tried last
- `certificate::T`: the quantity compared in that condition, as described by each routine

The conditions are sufficient: `certified = false` does not say that the matrix is singular.
"""
struct RegularityResult{T}
    certified::Bool
    method::Symbol
    certificate::T
end

# an upper bound of ‖M⁻¹‖₂ for a point matrix M, Inf where none is obtained
function _inverse_norm_or_inf(M::AbstractMatrix, ::Type{T}) where {T}
    try
        b = svd_bound_L2_opnorm_inverse(BallMatrix(M))
        return (isfinite(b) && b > 0) ? T(b) : T(Inf)
    catch
        return T(Inf)
    end
end

"""
    is_regular_sufficient_condition(A::BallMatrix; condition = 1) -> RegularityResult

Theorem 11.12 of Horáček (2019): the interval matrix `A` is regular if

1. `λ_max(A_Δᵀ A_Δ) < λ_min(A_cᵀ A_c)` (`condition = 1`), or
2. `A_cᵀ A_c − ‖A_Δᵀ A_Δ‖ I` is positive definite for some consistent matrix norm
   (`condition = 2`, evaluated with the ∞-norm).

Both are evaluated through singular values: `λ_min(A_cᵀ A_c) = σ_min(A_c)² ≥ 1/b²` with
`b ≥ ‖A_c⁻¹‖₂` from [`svd_bound_L2_opnorm_inverse`](@ref), and `λ_max(A_Δᵀ A_Δ) ≤ ν²` with
`ν ≥ ‖A_Δ‖₂` from [`upper_bound_L2_opnorm`](@ref), or, for the second condition, an upper bound
of `‖A_Δᵀ A_Δ‖_∞` computed in ball arithmetic. The `certificate` is a lower bound of the
difference of the two sides, and regularity is proved when it is positive.

The thesis states the theorem for real interval matrices. For a complex ball matrix the first
condition still implies regularity, since `σ_min(A_c + E) ≥ σ_min(A_c) − ‖E‖₂` and
`‖E‖₂ ≤ ‖A_Δ‖₂` for `|E| ≤ A_Δ`; the second is weaker than the first.
"""
function is_regular_sufficient_condition(A::BallMatrix{T}; condition::Integer = 1)::RegularityResult{T} where {T}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("the matrix must be square"))
    condition in (1, 2) || throw(ArgumentError("condition must be 1 or 2"))
    b = _inverse_norm_or_inf(mid(A), T)
    isfinite(b) || return RegularityResult(false, :sufficient_condition, T(-Inf))
    λmin_c = div_down(one(T), mul_up(b, b))
    Δ = BallMatrix(rad(A))
    λmax_Δ = if condition == 1
        ν = upper_bound_L2_opnorm(Δ)
        mul_up(ν, ν)
    else
        upper_bound_L_inf_opnorm(Δ' * Δ)
    end
    separation = sub_down(λmin_c, λmax_Δ)
    return RegularityResult(separation > 0, :sufficient_condition, separation)
end

"""
    is_regular_spectral_radius(A::BallMatrix; R = inv(mid(A))) -> RegularityResult

Theorem 4.2 of Horáček (2019): a square interval matrix `A` is regular if for some real matrix
`R`

    ϱ(|I − R A_c| + |R| A_Δ) < 1.

`R` defaults to the floating-point inverse of the midpoint, which need not be exact: the theorem
holds for any `R`. The matrix `|I − R A_c| + |R| A_Δ` is bounded above entrywise in ball
arithmetic and its spectral radius by [`collatz_upper_bound`](@ref); the `certificate` is that
bound, and regularity is proved when it is below one. With `R = A_c⁻¹` the condition is
`ϱ(|A_c⁻¹| A_Δ) < 1`, strong regularity (Theorem 4.33 of the thesis).
"""
function is_regular_spectral_radius(A::BallMatrix{T}; R = nothing)::RegularityResult{T} where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("the matrix must be square"))
    Ac = mid(A)
    if R === nothing
        R = try
            inv(Ac)
        catch
            return RegularityResult(false, :spectral_radius, T(Inf))
        end
    end
    all(isfinite, R) || return RegularityResult(false, :spectral_radius, T(Inf))
    bR = BallMatrix(Matrix(R))
    defect = upper_abs(bR * BallMatrix(Ac) - I)
    spread = upper_abs(BallMatrix(upper_abs(bR)) * BallMatrix(rad(A)))
    M = BallMatrix(add_up.(defect, spread))
    ϱ = collatz_upper_bound(M)
    return RegularityResult(ϱ < 1, :spectral_radius, T(ϱ))
end

# min over the rows of mig(a_ii) − Σ_{k ≠ i} mag(a_ik), a lower bound of it
function _row_dominance_margin(A::BallMatrix{T}) where {T}
    n = size(A, 1)
    mag = upper_abs(A)
    margin = T(Inf)
    for i in 1:n
        mig = sub_down(abs_down(mid(A)[i, i]), rad(A)[i, i])
        s = zero(T)
        for k in 1:n
            k == i || (s = add_up(s, mag[i, k]))
        end
        margin = min(margin, sub_down(mig, s))
    end
    return margin
end

"""
    is_regular_diagonal_dominance(A::BallMatrix) -> RegularityResult

Definition 4.24 of Horáček (2019): `A` is strictly diagonally dominant when

    mig(a_ii) > Σ_{k ≠ i} mag(a_ik)        for every i,

with `mig` and `mag` the smallest and the largest modulus over an entry. Such a matrix is an
H-matrix and therefore regular (Section 4.5 of the thesis). The `certificate` is a lower bound
of the smallest difference of the two sides over the rows.
"""
function is_regular_diagonal_dominance(A::BallMatrix{T})::RegularityResult{T} where {T}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("the matrix must be square"))
    margin = _row_dominance_margin(A)
    return RegularityResult(margin > 0, :diagonal_dominance, margin)
end

"""
    is_regular_gershgorin(A::BallMatrix) -> RegularityResult

Regularity from the Geršgorin discs of the rows or of the columns: if zero lies outside every
row disc of every matrix of the ball, or outside every column disc, the matrix is regular. For
the rows this is strict diagonal dominance ([`is_regular_diagonal_dominance`](@ref)); for the
columns it is the same condition on the transpose. The `certificate` is the larger of the two
margins.
"""
function is_regular_gershgorin(A::BallMatrix{T})::RegularityResult{T} where {T}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("the matrix must be square"))
    At = BallMatrix(Matrix(transpose(mid(A))), Matrix(transpose(rad(A))))
    margin = max(_row_dominance_margin(A), _row_dominance_margin(At))
    return RegularityResult(margin > 0, :gershgorin, margin)
end

"""
    is_regular(A::BallMatrix; methods = [:diagonal_dominance, :gershgorin, :h_matrix,
        :sufficient_condition, :spectral_radius], verbose = false) -> RegularityResult

Try the sufficient conditions for regularity in the given order and return the first that
proves it, or the last result when none does. `:h_matrix` is [`is_H_matrix`](@ref) (an H-matrix
is regular, Section 4.4 of Horáček (2019)); the others are the routines of this file.
"""
function is_regular(A::BallMatrix{T};
        methods::Vector{Symbol} = [:diagonal_dominance, :gershgorin, :h_matrix,
            :sufficient_condition, :spectral_radius],
        verbose::Bool = false)::RegularityResult{T} where {T}
    last_result = RegularityResult(false, :none, T(NaN))
    for method in methods
        result = if method == :sufficient_condition
            is_regular_sufficient_condition(A)
        elseif method == :spectral_radius
            is_regular_spectral_radius(A)
        elseif method == :gershgorin
            is_regular_gershgorin(A)
        elseif method == :diagonal_dominance
            is_regular_diagonal_dominance(A)
        elseif method == :h_matrix
            RegularityResult(is_H_matrix(A), :h_matrix, T(NaN))
        else
            throw(ArgumentError("unknown regularity test $method"))
        end
        verbose && println(method, ": ", result.certified ? "regular" : "inconclusive",
            " (certificate ", result.certificate, ")")
        result.certified && return result
        last_result = result
    end
    return last_result
end

"""
    is_singular_sufficient_condition(A::BallMatrix; conditions = 1:4) -> Bool

Theorem 11.13 of Horáček (2019): a real interval matrix `A` is singular, that is, it contains a
singular matrix, if at least one of the following holds:

1. `λ_max(A_cᵀ A_c) ≤ λ_min(A_Δᵀ A_Δ)`;
2. `max_j (|A_c⁻¹| A_Δ)_jj ≥ 1`;
3. `(A_Δ − |A_c|)⁻¹ ≥ 0`;
4. `A_Δᵀ A_Δ − A_cᵀ A_c` is positive semidefinite.

Returns `true` when one of the conditions listed in `conditions` is proved, `false` otherwise,
which does not mean that the matrix is regular. How each is evaluated:

1. as `‖A_c‖₂ ‖A_Δ⁻¹‖₂ ≤ 1` with upper bounds of the two norms;
2. with an enclosure of `A_c⁻¹` by [`verifylss`](@ref), the moduli of its entries bounded from
   below and the products with `A_Δ` rounded down;
3. with an enclosure of the inverse of the ball matrix `A_Δ − |A_c|` (the subtraction is
   rounded) by `verifylss`, all of whose lower ends must be nonnegative; an inverse with an
   entry equal to zero is therefore not recognised;
4. with enclosures of the eigenvalues of the symmetric ball matrix `A_Δᵀ A_Δ − A_cᵀ A_c` by
   `verifyeigall(...; method = :rumplange2023)`, all of whose lower ends must be nonnegative.
"""
function is_singular_sufficient_condition(A::BallMatrix{T}; conditions = 1:4) where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("the matrix must be square"))
    eltype(mid(A)) <: Real ||
        throw(ArgumentError("Theorem 11.13 is stated for real interval matrices"))
    all(c -> c in 1:4, conditions) || throw(ArgumentError("conditions must be among 1:4"))
    Ac, AΔ = mid(A), rad(A)
    Id = BallMatrix(Matrix{T}(I, n, n))
    for c in conditions
        proved = try
            if c == 1
                b = _inverse_norm_or_inf(AΔ, T)
                isfinite(b) && mul_up(upper_bound_L2_opnorm(BallMatrix(Ac)), b) <= 1
            elseif c == 2
                sol = verifylss(BallMatrix(Ac), Id)
                sol.certified && _has_diagonal_at_least_one(sol.solution, AΔ)
            elseif c == 3
                sol = verifylss(BallMatrix(AΔ) - BallMatrix(abs.(Ac)), Id)
                X = sol.solution
                sol.certified && all(sub_down(mid(X)[i], rad(X)[i]) >= 0 for i in eachindex(mid(X)))
            else
                Δ, C = BallMatrix(AΔ), BallMatrix(Ac)
                S = _symmetric_part_ball(Δ' * Δ - C' * C, identity)
                r = verifyeigall(S; method = :rumplange2023)
                all(r.certified) && sum(length, r.clusters) == n &&
                    all(sub_down(real(r.centers[k]), r.radii[k]) >= 0 for k in eachindex(r.radii))
            end
        catch
            false
        end
        proved && return true
    end
    return false
end

# whether some diagonal entry of |X| A_Δ is proved to be at least one, X an enclosure of A_c⁻¹
function _has_diagonal_at_least_one(X::BallMatrix{T}, AΔ::AbstractMatrix{T}) where {T}
    n = size(AΔ, 1)
    for j in 1:n
        d = zero(T)
        for k in 1:n
            lower = sub_down(abs_down(mid(X)[j, k]), rad(X)[j, k])
            lower > 0 && (d = add_down(d, mul_down(lower, AΔ[k, j])))
        end
        d >= 1 && return true
    end
    return false
end

export RegularityResult
export is_regular_sufficient_condition, is_regular_spectral_radius, is_regular_gershgorin,
       is_regular_diagonal_dominance
export is_regular, is_singular_sufficient_condition
