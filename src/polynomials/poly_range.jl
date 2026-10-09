# Certified range enclosure of a real polynomial p on an interval [a,b].
#
# Two algorithms, both rigorous on the BallArithmetic backend, for comparison:
#
#   A. `range_bisect`   — interval branch-and-bound (Moore–Skelboe). Subdivide
#      [a,b], evaluate p in ball arithmetic on each subinterval, refine the box
#      that drives the current extreme, prune boxes that cannot hold it. Cost
#      grows with the requested tolerance; degrades gracefully; no derivative.
#
#   B. `range_critical` — analytic critical-point method. The extrema of a C¹
#      function on [a,b] sit at the endpoints or at interior stationary points
#      p'(x)=0. Find ALL roots of p' as the eigenvalues of its companion matrix,
#      certified by `verifyeigall` (Rump 2022: when every cluster is certified the
#      discs enclose EVERY root, so none is missed), keep the discs that reach the
#      real axis inside [a,b], and
#      hull the ball-evaluations of p at the endpoints and those stationary
#      enclosures. Accuracy is decoupled from a tolerance (one eigenproblem), and
#      is essentially exact — limited only by the eigenvalue-enclosure width.
#
# Polynomials are real coefficient vectors in ASCENDING order:
# `coeffs = [c₀, c₁, …, c_d]` ↦ p(x) = Σ cₖ xᵏ (the `evalpoly` convention).
#
# Moved here from RigorousPseudospectra, where it did not belong; it is a separate problem and may
# later become a package of its own.

export range_bisect, range_critical, ball_horner, PolyRange

"""
    PolyRange{T}

Certified range enclosure of a polynomial on `[a,b]`. With `min`/`max` the true
extrema of `p` over `[a,b]`, the four fields are sound brackets

    min ∈ [min_lo, min_hi],    max ∈ [max_lo, max_hi],

so the true range `[min, max] ⊆ [min_lo, max_hi]` (the outer enclosure), and each
extremum is pinned to a bracket of width `min_hi-min_lo` / `max_hi-max_lo` (the
residual overestimation). `method` is `:bisection` or `:critical`; `diag` carries
the method-specific cost/quality record.
"""
struct PolyRange{T <: AbstractFloat}
    min_lo::T
    min_hi::T
    max_lo::T
    max_hi::T
    method::Symbol
    diag::NamedTuple
end

function Base.show(io::IO, R::PolyRange)
    print(io,
        "PolyRange($(R.method): range ⊆ [", R.min_lo, ", ", R.max_hi,
        "]  min∈[", R.min_lo, ",", R.min_hi, "] max∈[", R.max_lo, ",", R.max_hi, "])")
end

# Rigorous Horner evaluation of a real polynomial at a ball x (outward-rounded
# throughout, so the returned ball encloses {p(t) : t ∈ x}).
function ball_horner(coeffs::AbstractVector{T}, x::Ball) where {T <: Real}
    acc = Ball(zero(T)) * x          # ball zero of the right (possibly complex) type
    @inbounds for k in lastindex(coeffs):-1:firstindex(coeffs)
        acc = acc * x + Ball(coeffs[k])
    end
    return acc
end

# ----------------------------------------------------------------------------
# A. branch-and-bound (Moore–Skelboe) for the global MINIMUM of p on [a,b].
# Returns (m_lo, m_hi, neval): a certified bracket m_lo ≤ min ≤ m_hi.
#   m_lo = min over live boxes of inf(p(box))     (lower bound, monotone ↑)
#   m_hi = min over sampled points of sup(p(pt))  (upper bound, an attained value)
# We always split the box with the smallest inf(p(box)) (the one holding the
# best candidate minimiser and driving m_lo), and discard boxes whose inf exceeds
# m_hi (they provably cannot contain the global minimum).
# ----------------------------------------------------------------------------
function _bnb_min(coeffs::AbstractVector{T}, a::T, b::T;
        tol::T, maxeval::Int) where {T <: AbstractFloat}
    boxval(lo, hi) = ball_horner(coeffs, Ball((lo + hi) / 2, (hi - lo) / 2))
    ptsup(x) = sup(ball_horner(coeffs, Ball(x)))

    F0 = boxval(a, b)
    items = [(a, b, inf(F0), sup(F0))]          # (lo, hi, inf p, sup p)
    m_hi = min(ptsup(a), ptsup(b))
    neval = 3
    minwidth(lo, hi) = 8 * eps(max(abs(lo), abs(hi), one(T)))

    while neval < maxeval
        i = argmin(j -> items[j][3], eachindex(items))
        lo, hi, fl, _ = items[i]
        m_hi = min(m_hi, ptsup((lo + hi) / 2));
        neval += 1
        (m_hi - fl <= tol) && break
        (hi - lo <= minwidth(lo, hi)) && break  # cannot refine the binding box further
        deleteat!(items, i)
        mid = (lo + hi) / 2
        for (l, h) in ((lo, mid), (mid, hi))
            F = boxval(l, h);
            neval += 1
            infF = inf(F)
            infF <= m_hi && push!(items, (l, h, infF, sup(F)))   # prune
        end
    end
    m_lo = minimum(t -> t[3], items)
    return m_lo, m_hi, neval
end

"""
    range_bisect(coeffs, a, b; tol = 1e-8, maxeval = 100_000) -> PolyRange

Algorithm A — certified range of `p` (ascending `coeffs`) on `[a,b]` by interval
branch-and-bound. `tol` is the target width of each extreme's bracket; `maxeval`
caps the ball evaluations. The maximum is obtained from `-p`.
"""
function range_bisect(coeffs::AbstractVector{<:Real}, a::Real, b::Real;
        tol::Real = 1e-8, maxeval::Integer = 100_000)
    T = float(promote_type(eltype(coeffs), typeof(a), typeof(b)))
    c = T.(coeffs);
    aa, bb = T(a), T(b)
    aa <= bb || error("need a ≤ b (got a=$a, b=$b)")
    mlo, mhi, n1 = _bnb_min(c, aa, bb; tol = T(tol), maxeval = Int(maxeval))
    nlo, nhi, n2 = _bnb_min(-c, aa, bb; tol = T(tol), maxeval = Int(maxeval))  # max(p) = -min(-p)
    return PolyRange(mlo, mhi, -nhi, -nlo, :bisection, (; evals = n1 + n2))
end

# ----------------------------------------------------------------------------
# B. critical-point method via certified roots of p'.
# ----------------------------------------------------------------------------

# Frobenius companion of a monic polynomial (ascending `mon`, mon[end]==1):
# eigenvalues = roots. Last column = −[mon₀ … mon_{d−1}], unit subdiagonal.
function _companion(mon::AbstractVector{T}) where {T}
    d = length(mon) - 1
    C = zeros(T, d, d)
    @inbounds for i in 2:d
        C[i, i - 1] = one(T)
    end
    @inbounds for i in 1:d
        C[i, d] = -mon[i]
    end
    return C
end

"""
    range_critical(coeffs, a, b; maxiter = 20) -> PolyRange

Algorithm B — certified range of `p` (ascending `coeffs`) on `[a,b]` by enclosing
the stationary points. Forms `p'`, encloses the roots of its companion matrix
with [`verifyeigall`](@ref) (`maxiter` interval iterations), keeps every
inclusion disc that meets the real axis inside `[a,b]`, and hulls the
ball-evaluations of `p` at the endpoints and those stationary enclosures.

The candidate set is exhaustive only when every cluster is certified
(`spectrum_covered`); when it is not, the range is computed by
[`range_bisect`](@ref) instead and `diag.covered` is false.

`diag` reports `ncrit` (real stationary enclosures used), `maxdisc` (largest
root-disc radius: large means the companion was hard to enclose, and the bound
loosens but stays rigorous), and `covered`.
"""
function range_critical(coeffs::AbstractVector{<:Real}, a::Real, b::Real;
        maxiter::Integer = 20)
    T = float(promote_type(eltype(coeffs), typeof(a), typeof(b)))
    c = T.(coeffs);
    aa, bb = T(a), T(b)
    aa <= bb || error("need a ≤ b (got a=$a, b=$b)")
    deg = length(c) - 1

    # endpoints are always candidate extrema
    balls = [ball_horner(c, Ball(aa)), ball_horner(c, Ball(bb))]
    ncrit = 0
    maxdisc = zero(T)
    covered = true

    if deg >= 2                                # p' has degree ≥ 1 ⇒ may have roots
        d = T[k * c[k + 1] for k in 1:deg]     # p'(x) = Σ k cₖ x^{k-1}, ascending
        m = length(d)                          # trim EXACT high-order zeros only (sound)
        while m > 1 && d[m] == 0
            m -= 1
        end
        if m >= 2                              # genuine derivative of degree ≥ 1
            # the monic coefficients d[i]/d[m] are rounded, so they enter the companion as balls
            # of radius |d[i]/d[m]| u, and the enclosure covers the roots of the exact p'
            monc = d[1:m] ./ d[m]
            Cm = _companion(monc)
            Cr = zeros(T, size(Cm))
            Cr[:, end] .= [mul_up(abs(x), eps(T) / 2) for x in monc[1:(end - 1)]]
            vr = verifyeigall(BallMatrix(Cm, Cr); maxiter)
            if !vr.spectrum_covered
                covered = false
                Rb = range_bisect(coeffs, a, b)
                return PolyRange(Rb.min_lo, Rb.min_hi, Rb.max_lo, Rb.max_hi, :critical,
                    (; ncrit = 0, maxdisc = T(NaN), covered))
            end
            for i in eachindex(vr.clusters)
                ctr = complex(vr.centers[i])
                r = T(vr.radii[i])
                maxdisc = max(maxdisc, r)
                im = abs(imag(ctr))
                im <= r || continue            # disc clears the real axis ⇒ no real root
                # the chord of the disc on the real axis, half-width rounded up
                s = sqrt_up(max(zero(T), sub_up(mul_up(r, r), mul_down(im, im))))
                lo = max(sub_down(real(ctr), s), aa)
                hi = min(add_up(real(ctr), s), bb)
                lo <= hi || continue           # stationary enclosure outside [a,b]
                ncrit += 1
                push!(balls, ball_horner(c, ball_hull(Ball(lo), Ball(hi))))
            end
        end
    end

    # min = minₖ p(region k), each value ∈ [inf bₖ, sup bₖ] ⇒ min ∈ [minₖ inf, minₖ sup];
    # symmetrically for max. (No subdivision needed — the candidate set is exhaustive.)
    infs = inf.(balls)
    sups = sup.(balls)
    return PolyRange(minimum(infs), minimum(sups), maximum(infs), maximum(sups),
        :critical, (; ncrit, maxdisc, covered))
end
