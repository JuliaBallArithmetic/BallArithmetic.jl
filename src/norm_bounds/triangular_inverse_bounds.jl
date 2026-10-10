# Upper bounds for the norm of the inverse of an upper triangular matrix, without forming the
# inverse, and the norm of the unit block triangular similarity S(X) = [I X; 0 I].
#
# Every operation is rounded outward with the emulated directed operations; no rounding mode is
# changed.

# |x| is exact for a real float; for a complex one it is rounded in the direction asked
_abs_lo(x::Real) = abs(x)
_abs_lo(x::Complex) = abs_down(x)
_abs_hi(x::Real) = abs(x)
_abs_hi(x::Complex) = abs_up(x)

# The two recursions behind the bounds below. `dlo[i]` is a lower bound of |u_ii| and `absU[i, j]`,
# j > i, an upper bound of |u_ij|; the entries of `absU` on and below the diagonal are not read.
# Returns upper bounds of (‖U⁻¹‖_∞, ‖U⁻¹‖₁), or (Inf, Inf) when some `dlo[i]` is not positive.
#
# Proof. Let M be the matrix with m_ii = dlo[i], m_ij = −absU[i, j] for j > i, and zero below the
# diagonal. Back substitution gives, by induction on the columns from the diagonal outward,
# |U⁻¹| ≤ M⁻¹ entrywise, and M⁻¹ ≥ 0. Hence ‖U⁻¹‖_∞ ≤ ‖M⁻¹e‖_∞ with e the vector of ones, and
# y = M⁻¹e is the solution of dlo[i]·y_i = 1 + Σ_{j>i} absU[i, j]·y_j, computed from the last row
# upward. Each y_i is rounded up, which keeps it above the exact solution because the recursion is
# increasing in the y_j already computed. The 1-norm is the ∞-norm of the transpose, a lower
# triangular matrix, and the same argument runs from the first row downward.
function _triangular_inverse_bounds(dlo::AbstractVector{T}, absU::AbstractMatrix{T}) where {T}
    m = length(dlo)
    all(d -> isfinite(d) && d > 0, dlo) || return T(Inf), T(Inf)
    y = zeros(T, m)
    for i in m:-1:1
        s = one(T)
        for j in (i + 1):m
            s = add_up(s, mul_up(absU[i, j], y[j]))
        end
        y[i] = div_up(s, dlo[i])
    end
    w = zeros(T, m)
    for j in 1:m
        s = one(T)
        for i in 1:(j - 1)
            s = add_up(s, mul_up(absU[i, j], w[i]))
        end
        w[j] = div_up(s, dlo[j])
    end
    return maximum(y; init = zero(T)), maximum(w; init = zero(T))
end

# the data of the recursion for a matrix whose entries are exact
function _triangular_data(U::AbstractMatrix)
    size(U, 1) == size(U, 2) || throw(DimensionMismatch("U must be square"))
    istriu(U) || throw(ArgumentError("U must be upper triangular"))
    return [_abs_lo(float(U[i, i])) for i in axes(U, 1)], _abs_hi.(float.(U))
end

"""
    triangular_inverse_inf_norm_bound(U)

An upper bound of `‖U⁻¹‖_∞` for an upper triangular matrix `U` with exact entries, `Inf` when a
diagonal entry is zero. With `y` the solution of

    |u_ii| y_i = 1 + Σ_{j>i} |u_ij| y_j,    i = m, m−1, …, 1,

the bound is `max_i y_i`: `y = M⁻¹e` for the matrix `M` with diagonal `|u_ii|` and off-diagonal
entries `−|u_ij|`, and `|U⁻¹| ≤ M⁻¹` entrywise. The recursion is evaluated with every operation
rounded up and the moduli on the diagonal rounded down.
"""
triangular_inverse_inf_norm_bound(U::AbstractMatrix) =
    _triangular_inverse_bounds(_triangular_data(U)...)[1]

"""
    triangular_inverse_one_norm_bound(U)

An upper bound of `‖U⁻¹‖₁` for an upper triangular matrix `U` with exact entries: the bound of
[`triangular_inverse_inf_norm_bound`](@ref) for the transpose.
"""
triangular_inverse_one_norm_bound(U::AbstractMatrix) =
    _triangular_inverse_bounds(_triangular_data(U)...)[2]

"""
    triangular_inverse_two_norm_bound(U)

An upper bound of `‖U⁻¹‖₂` for an upper triangular matrix `U` with exact entries, by
`‖M‖₂ ≤ √(‖M‖₁‖M‖_∞)` applied to the two bounds above.
"""
triangular_inverse_two_norm_bound(U::AbstractMatrix) =
    _two_norm_from_one_inf(_triangular_inverse_bounds(_triangular_data(U)...)...)

function _two_norm_from_one_inf(a::T, b::T) where {T}
    (isfinite(a) && isfinite(b)) || return T(Inf)
    return sqrt_up(mul_up(a, b))
end

"""
    psi_squared(μ)

An upper bound of `ψ(μ)² = 1 + μ²/2 + (μ/2)√(μ² + 4)`, rounded up.

For `S(X) = [I X; 0 I]` and `μ = ‖X‖₂`, `‖S(X)‖₂ = ‖S(X)⁻¹‖₂ = ψ(μ)`, so `ψ(μ)²` is the condition
number of `S(X)` in the spectral norm: with the singular value decomposition of `X` the matrix
`S(X)*S(X)` splits into blocks `[1 σ; σ 1 + σ²]`, whose largest eigenvalue is
`1 + σ²/2 + (σ/2)√(σ² + 4)`, increasing in `σ`; and `S(X)⁻¹ = S(−X)`. Being increasing, the
function may be evaluated at an upper bound of `‖X‖₂`.
"""
function psi_squared(μ::T) where {T <: AbstractFloat}
    μ ≤ zero(T) && return one(T)
    μ2 = mul_up(μ, μ)
    return add_up(add_up(one(T), div_up(μ2, T(2))),
        mul_up(div_up(μ, T(2)), sqrt_up(add_up(μ2, T(4)))))
end

"""
    similarity_condition_number(X)

An upper bound of `κ₂(S(X))` for `S(X) = [I X; 0 I]`: [`psi_squared`](@ref) at
`upper_bound_L2_opnorm` of `X`. `X` may be a matrix with exact entries or a `BallMatrix`.
"""
similarity_condition_number(X::AbstractMatrix) = similarity_condition_number(BallMatrix(float.(X)))
similarity_condition_number(X::BallMatrix) = psi_squared(upper_bound_L2_opnorm(X))
