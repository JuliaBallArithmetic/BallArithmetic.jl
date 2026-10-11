# The Hansen-Bliek-Rohn-Ning-Kearfott-Neumaier enclosure of the solution set of a square interval
# linear system, after
#
#   J. Horáček, Interval linear and nonlinear systems, doctoral thesis, Charles University,
#   Prague, 2019, Section 5.6.2, Theorem 5.12 (which the thesis takes from Ning and Kearfott).

"""
    HBRResult{VT}

Result of [`hbr_method`](@ref).

# Fields
- `solution`: a box containing the solution set when `success` is `true`; infinite radii
  otherwise
- `success`: whether the hypotheses of Theorem 5.12 were verified and the enclosure computed
"""
struct HBRResult{VT}
    solution::VT
    success::Bool
end

_hbr_failure(::Type{T}, n) where {T} = HBRResult(BallVector(zeros(T, n), fill(T(Inf), n)), false)

"""
    hbr_method(A::BallMatrix, b::BallVector; preconditioner = nothing) -> HBRResult

Theorem 5.12 of Horáček (2019). Let `A x = b` be a real square interval system with `A` an
H-matrix, `⟨A⟩` its comparison matrix,

    u = ⟨A⟩⁻¹ mag(b),    d_i = (⟨A⟩⁻¹)_ii,    α_i = ⟨A⟩_ii − 1/d_i,    β_i = u_i/d_i − mag(b_i).

Then the solution set is contained in the box with components

    x_i = (b_i + [−β_i, β_i]) / (A_ii + [−α_i, α_i]).

When the midpoint of `A` is diagonal the box is the hull (the thesis, after the theorem).

The theorem is applied to the preconditioned system `(CA) x = Cb`, formed in ball arithmetic,
whose solution set contains that of `A x = b` (Section 5.4). `preconditioner = nothing` takes
`C` the floating-point inverse of the midpoint; a real matrix is used as given;
[`hbr_method_simple`](@ref) applies the theorem to `A x = b` itself.

How it is verified: that the matrix is an H-matrix by [`is_H_matrix`](@ref); `⟨A⟩⁻¹` is enclosed
by [`verifylss`](@ref), as the thesis suggests with the ε-inflation method; `α_i` and `β_i`
are replaced by upper bounds, which enlarges the box, since the quotient grows with both.
`success` is `false` where one of these steps fails.
"""
function hbr_method(A::BallMatrix{T}, b::BallVector{T}; preconditioner = nothing) where {T}
    n = _require_real_system(A, b, "hbr_method")
    C = preconditioner === nothing ? _midpoint_inverse(A) : preconditioner
    C === nothing && return _hbr_failure(T, n)
    bC = BallMatrix(Matrix{T}(C))
    return _hbr_theorem_5_12(bC * A, bC * b)
end

"""
    hbr_method_simple(A::BallMatrix, b::BallVector) -> HBRResult

Theorem 5.12 of Horáček (2019) applied to `A x = b` without preconditioning; `A` must then be
an H-matrix itself. See [`hbr_method`](@ref).
"""
function hbr_method_simple(A::BallMatrix{T}, b::BallVector{T}) where {T}
    _require_real_system(A, b, "hbr_method_simple")
    return _hbr_theorem_5_12(A, b)
end

function _hbr_theorem_5_12(A::BallMatrix{T}, b::BallVector{T}) where {T}
    n = size(A, 1)
    (all(isfinite, mid(A)) && all(isfinite, rad(A)) && is_H_matrix(A)) ||
        return _hbr_failure(T, n)
    # the comparison matrix as a ball matrix: mig on the diagonal, −mag off it, the sums rounded
    Ac, Ar = mid(A), rad(A)
    comparison = [i == j ? Ball(abs(Ac[i, i])) - Ball(Ar[i, i]) :
                  -(Ball(abs(Ac[i, j])) + Ball(Ar[i, j])) for i in 1:n, j in 1:n]
    M = BallMatrix(mid.(comparison), rad.(comparison))
    sol = try
        verifylss(M, BallMatrix(Matrix{T}(I, n, n)))
    catch
        return _hbr_failure(T, n)
    end
    sol.certified || return _hbr_failure(T, n)
    Minv = sol.solution
    magb = [Ball(abs(mid(b)[i])) + Ball(rad(b)[i]) for i in 1:n]
    u = Minv * BallVector(mid.(magb), rad.(magb))
    xc, xr = Vector{T}(undef, n), Vector{T}(undef, n)
    for i in 1:n
        d = Minv[i, i]
        inf(d) > 0 || return _hbr_failure(T, n)
        α = max(sup(comparison[i, i] - inv(d)), zero(T))
        β = max(sup(u[i] / d - magb[i]), zero(T))
        # the quotient of the two intervals by its ends: the ball quotient would overestimate it
        # and lose the hull of the diagonal case
        rn, rd = add_up(rad(b)[i], β), add_up(Ar[i, i], α)
        x = _interval_quotient(sub_down(mid(b)[i], rn), add_up(mid(b)[i], rn),
            sub_down(Ac[i, i], rd), add_up(Ac[i, i], rd))
        x === nothing && return _hbr_failure(T, n)
        xc[i], xr[i] = mid(x), rad(x)
    end
    all(isfinite, xr) || return _hbr_failure(T, n)
    return HBRResult(BallVector(xc, xr), true)
end

# A ball containing [nl, nu]/[dl, du], its ends rounded outwards; nothing where the denominator
# contains zero.
function _interval_quotient(nl::T, nu::T, dl::T, du::T) where {T}
    if du < 0
        nl, nu, dl, du = -nu, -nl, -du, -dl
    end
    dl > 0 || return nothing
    lo = nl >= 0 ? div_down(nl, du) : div_down(nl, dl)
    hi = nu >= 0 ? div_up(nu, dl) : div_up(nu, du)
    return _ball_from_bounds(lo, hi)
end

export HBRResult
export hbr_method, hbr_method_simple
