# Enclosures of the determinant of an interval matrix, after
#
#   J. Horáček, Interval linear and nonlinear systems, doctoral thesis, Charles University,
#   Prague, 2019, Chapter 8: Sections 8.4.1 (Gaussian elimination), 8.4.2 (Gerschgorin discs),
#   8.4.3 (Hadamard's inequality) and 8.4.4 (Cramer's rule).
#
# Not implemented from that chapter: monotonicity checking (8.4.5), preconditioning (8.4.6), the
# verified determinant of a real matrix (8.5) and the special cases of Section 8.6.
#
# Every routine returns a ball that contains det(A) for every matrix A of the input ball.

"""
    DeterminantResult{T}

Result of a determinant enclosure.

# Fields
- `determinant::Ball`: a ball containing `det(A)` for every matrix `A` of the input; radius `Inf`
  where the method gives no bound
- `method::Symbol`: the method used
"""
struct DeterminantResult{BT <: Ball}
    determinant::BT
    method::Symbol
end

# the ball that says nothing, with midpoint of type CT (real or complex)
_unbounded_determinant(::Type{CT}) where {CT} = Ball(zero(CT), real(CT)(Inf))

"""
    det_hadamard(A::BallMatrix) -> DeterminantResult

Section 8.4.3 of Horáček (2019): by Hadamard's inequality, for every matrix of the interval
matrix `A`,

    |det(A)| ≤ d = ∏_i ( Σ_j mag(A_ji)² )^(1/2),

the product of the largest Euclidean norms of the columns. The same holds with the rows, and
the smaller of the two bounds is returned, as the ball of centre zero and radius `d`. The sign
of the determinant cannot be detected in this way. Real or complex input.
"""
function det_hadamard(A::BallMatrix{T}) where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("the matrix must be square"))
    mag = upper_abs(A)
    function norms_product(byrow::Bool)
        d = one(T)
        for i in 1:n
            s = zero(T)
            for j in 1:n
                m = byrow ? mag[i, j] : mag[j, i]
                s = add_up(s, mul_up(m, m))
            end
            d = mul_up(d, sqrt_up(s))
        end
        return d
    end
    d = min(norms_product(false), norms_product(true))
    return DeterminantResult(Ball(zero(T), d), :hadamard)
end

# Figures 8.3 and 8.4 of the thesis: an interval containing every product of k real numbers of
# [a, b], and with them every product in which pairs of them are replaced by pairs of complex
# conjugates lying in discs centred on the real axis inside [a, b], whose product is the square
# of a modulus between the smallest and the largest modulus of [a, b].
function _bunch_product_bounds(a::T, b::T, k::Int) where {T}
    if a > 0                                   # c) nonnegative: [a^k, b^k]
        return pow_down(a, k), pow_up(b, k)
    elseif b < 0                               # b1), b2) nonpositive
        lo, hi = pow_down(-b, k), pow_up(-a, k)          # bounds of the modulus
        return iseven(k) ? (lo, hi) : (-hi, -lo)
    elseif -a <= b                             # d1), d2) contains 0, |a| ≤ |b|: [a b^(k-1), b^k]
        return -mul_up(-a, pow_up(b, k - 1)), pow_up(b, k)
    elseif iseven(k)                           # d3) contains 0, |a| > |b|: [a^(k-1) b, a^k]
        return -mul_up(pow_up(-a, k - 1), b), pow_up(-a, k)
    else                                       # d4) [a^k, a^(k-1) b]
        return -pow_up(-a, k), mul_up(pow_up(-a, k - 1), b)
    end
end

"""
    det_gershgorin(A::BallMatrix) -> DeterminantResult

Section 8.4.2 of Horáček (2019), for a real interval matrix. Every eigenvalue of every matrix of
`A` lies in an interval Gerschgorin disc, of centre `mid(A_ii)` and radius
`rad(A_ii) + Σ_{j≠i} mag(A_ij)`, and a group of `k` discs whose union is connected (a "bunch")
contains `k` eigenvalues. With `[a, b]` the projection of a bunch on the real axis, the product
of its `k` eigenvalues lies in the interval of Figures 8.3 and 8.4 of the thesis:

- `[a, b]` for one disc;
- `[a^k, b^k]` when `a > 0`; when `b < 0`, `[a^k, b^k]` for odd `k` and `[b^k, a^k]` for even `k`;
- when `a ≤ 0 ≤ b` and `|a| ≤ |b|`, `[a b^(k−1), b^k]`;
- when `a ≤ 0 ≤ b` and `|a| > |b|`, `[a^(k−1) b, a^k]` for even `k` and `[a^k, a^(k−1) b]` for
  odd `k`.

Complex eigenvalues come in conjugate pairs and the product of a pair, the square of its
modulus, lies between the squares of the smallest and the largest modulus of `[a, b]`
(Figure 8.5), so the same intervals hold. The determinant is enclosed by the product of the
intervals of the bunches. The ends of the discs and the powers are rounded outwards.

The thesis remarks that the overestimation may be large, and that the method suits tight
intervals or a matrix close to a diagonal one. Throws on complex input, where the discs are not
centred on the real axis.
"""
function det_gershgorin(A::BallMatrix{T}) where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("the matrix must be square"))
    eltype(mid(A)) <: Real ||
        throw(ArgumentError("det_gershgorin is for real interval matrices"))
    mag = upper_abs(A)
    lo, hi = Vector{T}(undef, n), Vector{T}(undef, n)
    for i in 1:n
        r = rad(A)[i, i]
        for j in 1:n
            j == i || (r = add_up(r, mag[i, j]))
        end
        lo[i], hi[i] = sub_down(mid(A)[i, i], r), add_up(mid(A)[i, i], r)
    end
    all(isfinite, lo) && all(isfinite, hi) ||
        return DeterminantResult(_unbounded_determinant(T), :gershgorin)
    order = sortperm(lo)
    det = Ball(one(T), zero(T))
    a, b, k = lo[order[1]], hi[order[1]], 1
    for idx in order[2:end]
        if lo[idx] <= b                      # the disc meets the current bunch
            b = max(b, hi[idx])
            k += 1
        else
            det = det * _ball_from_bounds(_bunch_product_bounds(a, b, k)...)
            a, b, k = lo[idx], hi[idx], 1
        end
    end
    det = det * _ball_from_bounds(_bunch_product_bounds(a, b, k)...)
    return DeterminantResult(det, :gershgorin)
end

"""
    det_cramer(A::BallMatrix) -> DeterminantResult

Section 8.4.4 of Horáček (2019). By Cramer's rule the first coefficient of the solution of
`A x = e₁` is `x₁ = det(A_{2:n})/det(A)`, with `A_{2:n}` the matrix without its first row and
column, so that (8.3)

    det(A) ⊆ det(A_{2:n}) / x₁

for an enclosure `x₁` of that coefficient over the interval matrix. The rule is applied
recursively down to order one. The enclosure of `x₁` is that of [`verifylss`](@ref). The method
works when no enclosure of `x₁` in the recursion contains zero and every solve is certified;
otherwise the returned ball has radius `Inf`.
"""
function det_cramer(A::BallMatrix{T}) where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("the matrix must be square"))
    CT = eltype(mid(A))
    det = Ball(one(CT), zero(T))
    M = A
    for m in n:-1:2
        e1 = zeros(CT, m)
        e1[1] = one(CT)
        sol = try
            verifylss(M, BallVector(e1))
        catch
            return DeterminantResult(_unbounded_determinant(CT), :cramer)
        end
        sol.certified || return DeterminantResult(_unbounded_determinant(CT), :cramer)
        x1 = sol.solution[1]
        sub_down(abs_down(mid(x1)), rad(x1)) > 0 ||
            return DeterminantResult(_unbounded_determinant(CT), :cramer)
        det = det / x1
        M = M[2:m, 2:m]
    end
    return DeterminantResult(det * M[1, 1], :cramer)
end

"""
    det_cofactor(A::BallMatrix) -> DeterminantResult

The Laplace expansion of the determinant along the first row, evaluated in ball arithmetic. It
is not a method of the thesis; its cost is `n!`, and it is used by [`interval_det`](@ref) for
orders up to three. For order two each entry occurs once in the formula, so the only
overestimation is that of the ball product.
"""
function det_cofactor(A::BallMatrix{T}) where {T}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("the matrix must be square"))
    return DeterminantResult(_cofactor_expansion(A), :cofactor)
end

function _cofactor_expansion(A::BallMatrix{T}) where {T}
    n = size(A, 1)
    n == 1 && return A[1, 1]
    n == 2 && return A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    total = A[1, 1] * _cofactor_expansion(A[2:n, 2:n])
    for j in 2:n
        term = A[1, j] * _cofactor_expansion(A[2:n, [k for k in 1:n if k != j]])
        total = isodd(j) ? total + term : total - term
    end
    return total
end

"""
    interval_det(A::BallMatrix; method = :auto) -> DeterminantResult

An enclosure of the determinant of the interval matrix `A` by

- `:gaussian_elimination`: Section 8.4.1 of Horáček (2019), the product of the diagonal of the
  row echelon form ([`interval_gaussian_elimination_det`](@ref));
- `:gershgorin`: [`det_gershgorin`](@ref);
- `:hadamard`: [`det_hadamard`](@ref);
- `:cramer`: [`det_cramer`](@ref);
- `:cofactor`: [`det_cofactor`](@ref);
- `:auto`: the cofactor expansion for orders up to three; above, Gaussian elimination for a
  real matrix and `:cramer` for a complex one, the elimination being for real matrices.
"""
function interval_det(A::BallMatrix{T}; method::Symbol = :auto) where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("the matrix must be square"))
    if method == :auto
        method = n <= 3 ? :cofactor :
                 eltype(mid(A)) <: Real ? :gaussian_elimination : :cramer
    end
    if method == :hadamard
        return det_hadamard(A)
    elseif method == :gershgorin
        return det_gershgorin(A)
    elseif method == :gaussian_elimination
        return DeterminantResult(interval_gaussian_elimination_det(A), :gaussian_elimination)
    elseif method == :cramer
        return det_cramer(A)
    elseif method == :cofactor
        return det_cofactor(A)
    end
    throw(ArgumentError("unknown determinant method $method"))
end

"""
    contains_zero(det_result::DeterminantResult) -> Bool

Whether zero lies in the enclosure of the determinant. `false` proves that the interval matrix
is regular; `true` proves nothing.
"""
function contains_zero(det_result::DeterminantResult)
    d = det_result.determinant
    return !(sub_down(abs_down(mid(d)), rad(d)) > 0)
end

export DeterminantResult
export det_hadamard, det_gershgorin, det_cramer, det_cofactor
export interval_det, contains_zero
