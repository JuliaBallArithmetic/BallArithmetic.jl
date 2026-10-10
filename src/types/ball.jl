"""
    Ball{T, CT}

Closed floating-point ball with midpoint type `CT` and radius type `T`.
Each value represents the set `{ c + δ : |δ| ≤ r }` where `c::CT` is the
stored midpoint and `r::T ≥ 0` is the radius. Both real and complex
midpoints are supported as long as the radius is expressed in the
underlying real field. The type behaves as a number and participates in
arithmetic with rigorous outward rounding.
"""
struct Ball{T <: AbstractFloat, CT <: Union{T, Complex{T}}} <: Number
    c::CT
    r::T
end

BallF64 = Ball{Float64, Float64}
BallComplexF64 = Ball{Float64, ComplexF64}

"""
    ±(c, r)

Shorthand constructor for `Ball(c, r)`. The operator mirrors the common
mathematical notation `c ± r` for centered intervals.
"""
±(c, r) = Ball(c, r)

# ---------------------------------------------------------------------------------------------
# Helpers for enclosures. None of them changes the rounding mode: they are built on the emulated
# directed operations of `rounding.jl`.
# ---------------------------------------------------------------------------------------------

# `x` converted to `T`, not below `x`. Comparisons between Julia's real types are exact, so the
# nearest conversion is stepped up once when it fell below.
function _convert_up(::Type{T}, x::Real) where {T <: AbstractFloat}
    x isa T && return x
    y = T(x)
    return (isnan(y) || y >= x) ? y : nextfloat(y)
end

# A bound on |m − c| when `m` is `c` converted to floating point. Zero when the conversion was
# exact, which `==` decides exactly. Otherwise each real conversion is one rounding to nearest for
# a float, an integer or an irrational constant, and at most three (numerator, denominator,
# quotient) for a rational, so 4u|m| per component covers it, plus the subnormal spacing.
function _conversion_error(m::Union{T, Complex{T}}, c::Number) where {T <: AbstractFloat}
    (m == c) && return zero(T)
    return add_up(mul_up(mul_up(T(2), machine_epsilon(T)), abs_up(m)), mul_up(T(2), subnormal_min(T)))
end

# A bound on the rounding error of a nearest-rounded product whose computed value is `c`. A real
# product errs by at most u|c|, or half the subnormal spacing under underflow; `machine_epsilon`
# is 2u. For a complex product,
#
#   N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM, Philadelphia,
#   2002, doi 10.1137/1.9780898718027, Lemma 3.5:
#       fl(xy) = xy(1 + δ),   |δ| ≤ √2 γ₂,     γ_n = n u / (1 − n u),
#
# about 2.83u of the exact product. In terms of the computed one, |fl(xy) − xy| ≤ √2γ₂|xy| and
# |xy| ≤ |c| + |fl(xy) − xy| give √2γ₂/(1 − √2γ₂)·|c| < 4u|c| (this step is ours), and the four
# real products may each underflow, which the lemma excludes.
_product_roundoff(c::T) where {T <: AbstractFloat} =
    add_up(subnormal_min(T), mul_up(machine_epsilon(T), abs(c)))
_product_roundoff(c::Complex{T}) where {T <: AbstractFloat} =
    add_up(mul_up(T(4), subnormal_min(T)), mul_up(mul_up(T(2), machine_epsilon(T)), abs_up(c)))

# The real ball [lo, hi]: the centre is rounded up from the midpoint and the radius up from
# centre − lo, which also reaches hi, since centre − lo ≥ (hi − lo)/2. `inv` and `sqrt` of a real
# ball below build their result the same way.
function _ball_from_bounds(lo::T, hi::T) where {T <: AbstractFloat}
    c = add_up(lo, mul_up(one(T) / T(2), sub_up(hi, lo)))
    return Ball(c, sub_up(c, lo))
end

"""
    Ball(c, r)

Construct a ball whose midpoint is `c` and radius is `r`. Both arguments are converted to
floating point; the radius is rounded up, and the rounding of the midpoint, when `c` is not
exactly representable, is added to it, so the ball contains the ball it was asked for.
"""
function Ball(c, r)
    m = float(c)
    T = typeof(real(m))
    return Ball(m, add_up(_convert_up(T, r), _conversion_error(m, c)))
end

"""
    Ball(c::Number)

The ball around `float(c)` containing `c`: radius zero when `c` is exactly representable, and a
bound on the rounding of the conversion otherwise (`Ball(1//3)`, `Ball(π)`).
"""
function Ball(c::Number)
    m = float(c)
    return Ball(m, _conversion_error(m, c))
end

"""
    Ball(x::Ball)

Identity conversion that returns `x` unchanged. This overload allows
`Ball` to participate seamlessly in generic code that may attempt to
reconstruct elements via the type constructor.
"""
Ball(x::Ball) = x

"""
    mid(x)

Return the midpoint of `x`. For plain numbers the midpoint is the value
itself, while for balls the stored center is returned.
"""
mid(x::Ball) = x.c
mid(x::Number) = x

"""
    rad(x)

Return the radius associated with `x`. Numbers default to a zero radius,
and balls return their stored uncertainty.
"""
rad(x::Ball) = x.r
rad(::T) where {T <: Number} = zero(float(real(T)))

# rad and mid for collections of Balls
"""
    rad(v::AbstractVector{<:Ball})

Return a vector of radii for a collection of balls.
"""
rad(v::AbstractVector{<:Ball}) = [x.r for x in v]

"""
    rad(M::AbstractMatrix{<:Ball})

Return a matrix of radii for a collection of balls.
"""
rad(M::AbstractMatrix{<:Ball}) = [x.r for x in M]

"""
    mid(v::AbstractVector{<:Ball})

Return a vector of midpoints for a collection of balls.
"""
mid(v::AbstractVector{<:Ball}) = [x.c for x in v]

"""
    mid(M::AbstractMatrix{<:Ball})

Return a matrix of midpoints for a collection of balls.
"""
mid(M::AbstractMatrix{<:Ball}) = [x.c for x in M]

"""
    midtype(::Ball)

Return the type used to store the midpoint component of a `Ball`. This
is useful for allocating arrays that mirror the internal layout of a
ball or a collection of balls.
"""
midtype(::Ball{T, CT}) where {T, CT} = CT
midtype(::Type{Ball{T, CT}}) where {T, CT} = CT
midtype(::Type{Ball}) = Float64

"""
    radtype(x)

Return the floating-point type used to store radii for `x`. The helper
accepts either a ball instance or the associated type, mirroring the
behaviour of [`midtype`](@ref).
"""
radtype(::Ball{T, CT}) where {T, CT} = T
radtype(::Type{Ball{T, CT}}) where {T, CT} = T
radtype(::Type{Ball}) = Float64

"""
    sup(x::Ball)

Return the supremum (upper endpoint) of the set represented by `x` by
evaluating `mid(x) + rad(x)` with outward rounding.
"""
sup(x::Ball) = @up x.c + x.r

"""
    inf(x::Ball)

Return the infimum (lower endpoint) of the set represented by `x` by
evaluating `mid(x) - rad(x)` with downward rounding.
"""
inf(x::Ball) = @down x.c - x.r

Base.show(io::IO, ::MIME"text/plain", x::Ball) = print(io, x.c, " ± ", x.r)

#################
# SET OPERATIONS #
#################

"""
    ball_hull(a::Ball, b::Ball)

Return the smallest ball that contains both `a` and `b`. For real centres
the function encloses the convex hull on the real line. When the midpoints
are complex the result encloses both discs while keeping the centre as
close as possible to one of the inputs so that subsequent operations remain
stable.
"""
function ball_hull(a::Ball{T, T}, b::Ball{T, T}) where {T}
    return _ball_from_bounds(min(inf(a), inf(b)), max(sup(a), sup(b)))
end

function ball_hull(a::Ball{T, Complex{T}}, b::Ball{T, Complex{T}}) where {T}
    distance = dist_up(mid(a), mid(b))
    coverage_from_a = add_up(distance, rad(b))
    coverage_from_b = add_up(distance, rad(a))

    option_a = max(rad(a), coverage_from_a)
    option_b = max(rad(b), coverage_from_b)

    if option_a <= option_b
        return Ball(mid(a), option_a)
    else
        return Ball(mid(b), option_b)
    end
end

"""
    intersect_ball(a::Ball, b::Ball)

Return the intersection of the real balls `a` and `b`. When the balls do
not overlap the function returns `nothing` to indicate that the
intersection is empty.
"""
function intersect_ball(a::Ball{T, T}, b::Ball{T, T}) where {T}
    lower = max(inf(a), inf(b))
    upper = min(sup(a), sup(b))
    if lower > upper
        return nothing
    end
    return _ball_from_bounds(lower, upper)
end

###############
# CONVERSIONS #
###############

"""
    Base.convert(::Type{Ball{T, CT}}, x::Ball)

Convert a ball to the same enclosure expressed with alternative midpoint
and radius types. This is typically used when promoting collections of
balls to a common numeric representation.
"""
function Base.convert(::Type{Ball{T, CT}}, x::Ball) where {T, CT}
    x isa Ball{T, CT} && return x
    # the radius is rounded up, and the rounding of the midpoint to the new type is added to it:
    # converting 1/3 from BigFloat to Float64 moves the centre by about 1e-17
    m = convert(CT, mid(x))
    return Ball(m, add_up(_convert_up(T, rad(x)), _conversion_error(m, mid(x))))
end

"""
    Base.convert(::Type{Ball{T, CT}}, c::Number)

Embed a plain number into a ball whose midpoint is `c` converted to `CT`. The radius is zero when
that conversion is exact and a bound on its rounding otherwise.
"""
function Base.convert(::Type{Ball{T, CT}}, c::Number) where {T, CT}
    m = convert(CT, c)
    return Ball(m, _conversion_error(m, c))
end
Base.convert(::Type{Ball}, c::Number) = Ball(c)

# Single-argument parametric constructor — delegates to convert so that
# promote_type + T(x) works (used by GKWExperiments and other downstream code).
Ball{T, CT}(x::Number) where {T <: AbstractFloat, CT} = convert(Ball{T, CT}, x)

# Conversion from Ball to plain numeric types (extracts midpoint)
function Base.convert(::Type{T}, x::Ball{T, T}) where {T <: AbstractFloat}
    throw(DomainError(x, "This conversion breaks rigour"))
end
function Base.convert(::Type{T}, x::Ball) where {T <: AbstractFloat}
    throw(DomainError(x, "This conversion breaks rigour"))
end
Base.Float64(x::Ball) = throw(DomainError(x, "This conversion breaks rigour"))
Base.Float32(x::Ball) = throw(DomainError(x, "This conversion breaks rigour"))
function (::Type{T})(x::Ball) where {T <: AbstractFloat}
    throw(DomainError(x, "This conversion breaks rigour"))
end

#########################
# ARITHMETIC OPERATIONS #
#########################

"""
    +(x::Ball)

Return `x` unchanged. Unary plus exists for completeness so that generic
numeric code can treat balls like other scalar types.
"""
Base.:+(x::Ball) = x

"""
    -(x::Ball)

Negate the midpoint of `x` while keeping the radius unchanged. The
result encloses the additive inverse of the represented set.
"""
Base.:-(x::Ball) = Ball(-x.c, x.r)

"""
    Base.:+(x::Ball, y::Ball)

Combine two balls using addition and enlarge the radius so that the
result remains a rigorous enclosure. The midpoint is the rounded sum and
the radius accounts for both operands plus floating-point roundoff.
"""
function Base.:+(x::Ball{T}, y::Ball{T}) where {T}
    c = mid(x) + mid(y)
    ϵ = machine_epsilon(T)
    r = add_up(add_up(mul_up(ϵ, abs_up(c)), rad(x)), rad(y))
    Ball(c, r)
end

"""
    Base.:-(x::Ball, y::Ball)

Combine two balls using subtraction and enlarge the radius so that the
result remains a rigorous enclosure. The midpoint is the rounded
difference and the radius accounts for both operands plus floating-point
roundoff.
"""
function Base.:-(x::Ball{T}, y::Ball{T}) where {T}
    c = mid(x) - mid(y)
    ϵ = machine_epsilon(T)
    r = add_up(add_up(mul_up(ϵ, abs_up(c)), rad(x)), rad(y))
    Ball(c, r)
end

"""
    *(x::Ball, y::Ball)

Multiply two balls and return the enclosure of the product. The midpoint
is the product of the midpoints, whereas the radius collects propagated
uncertainty from both operands and the intrinsic rounding error of the
operation.
"""
function Base.:*(x::Ball{T}, y::Ball{T}) where {T}
    c = mid(x) * mid(y)
    # r = roundoff(c) + ((|mid(x)| + rad(x)) * rad(y) + rad(x) * |mid(y)|), the moduli bounded
    # above; the roundoff of a complex product is larger than that of a real one
    abs_mx = abs_up(mid(x))
    abs_my = abs_up(mid(y))
    term1 = _product_roundoff(c)
    term2 = mul_up(add_up(abs_mx, rad(x)), rad(y))
    term3 = mul_up(rad(x), abs_my)
    r = add_up(term1, add_up(term2, term3))
    Ball(c, r)
end

"""
    inv(x::Ball)

Return the multiplicative inverse of a real ball. The method throws an
`ArgumentError` when the interval straddles zero, since no rigorous
inverse can be produced in that case.
"""
function Base.inv(y::Ball{T, T}) where {T <: AbstractFloat}
    my, ry = mid(y), rad(y)
    ry < abs(my) || throw(ArgumentError("Ball $y contains zero."))
    one_T = one(T)
    half_T = one_T / T(2)  # Exact in binary floating point (1 and 0.5 are representable)
    c1 = div_down(one_T, add_up(abs(my), ry))
    c2 = div_up(one_T, sub_down(abs(my), ry))
    c = add_up(c1, mul_up(half_T, sub_up(c2, c1)))
    r = sub_up(c, c1)
    Ball(copysign(c, my), r)
end

"""
    inv(x::Ball{T, Complex{T}})

Return the multiplicative inverse of a complex ball. The method throws an `ArgumentError` when the
ball is not proved to exclude zero. The image of the disc `B(m, r)` under `1/z` is the disc
`B(conj(m)/D, r/D)` with `D = |m|² − r²`. `D` is bracketed by `D_lo ≤ D ≤ D_hi` with directed
operations and `D_lo > 0` is required; the centre is computed with `D_lo`, and the radius is
`r/D_lo` plus the distance `|m|(1/D_lo − 1/D_hi)` between that centre and the exact one, plus the
rounding of its two divisions.
"""
function Base.inv(y::Ball{T, Complex{T}}) where {T <: AbstractFloat}
    my, ry = mid(y), rad(y)
    a, b = real(my), imag(my)
    n2_lo = add_down(mul_down(a, a), mul_down(b, b))
    n2_hi = add_up(mul_up(a, a), mul_up(b, b))
    D_lo = sub_down(n2_lo, mul_up(ry, ry))
    D_hi = sub_up(n2_hi, mul_down(ry, ry))
    D_lo > 0 || throw(ArgumentError("Ball $y contains zero."))
    c = complex(a / D_lo, -b / D_lo)
    spread = mul_up(abs_up(my), sub_up(div_up(one(T), D_lo), div_down(one(T), D_hi)))
    roundoff = add_up(mul_up(machine_epsilon(T), abs_up(c)), mul_up(T(2), subnormal_min(T)))
    r = add_up(div_up(ry, D_lo), add_up(spread, roundoff))
    Ball(c, r)
end

"""
    /(x::Ball, y::Ball)

Divide `x` by `y` by multiplying with the inverse of `y`. The operation
inherits the same domain restrictions as [`inv`](@ref).
"""
Base.:/(x::Ball, y::Ball) = x * inv(y)

"""
    sqrt(x::Ball)

Principal square root of a non-negative real ball. The method verifies
that the enclosure stays within the domain of the square root and then
propagates rounding errors to produce a rigorous result.
"""
function Base.sqrt(y::Ball{T}) where {T <: AbstractFloat}
    my, ry = mid(y), rad(y)
    ry < my || throw(DomainError("Ball $y contains zero."))
    half_T = one(T) / T(2)  # Exact in binary floating point
    c1 = sqrt_down(sub_down(my, ry))
    c2 = sqrt_up(add_up(my, ry))
    c = add_up(c1, mul_up(half_T, sub_up(c2, c1)))
    r = sub_up(c, c1)
    Ball(c, r)
end

"""
    abs(x::Ball)

Return a real ball that encloses `|z|` for every `z` in `x`, that is the interval
`[max(0, |c| − r), |c| + r]`. For a real ball that does not contain zero this is the ball of
centre `|c|` and the same radius, exactly. For a complex ball the modulus of the centre is
bounded below and above, since `hypot` is not correctly rounded.
"""
function Base.abs(x::Ball{T, T}) where {T <: AbstractFloat}
    abs(x.c) > x.r && return Ball(abs(x.c), x.r)
    return _ball_from_bounds(zero(T), add_up(abs(x.c), x.r))
end
function Base.abs(x::Ball{T, Complex{T}}) where {T <: AbstractFloat}
    lo = max(zero(T), sub_down(abs_down(x.c), x.r))
    return _ball_from_bounds(lo, add_up(abs_up(x.c), x.r))
end

"""
    conj(x::Ball)

Complex conjugate of a ball. The midpoint is conjugated while the radius
remains unchanged.
"""
Base.conj(x::Ball) = Ball(conj(x.c), x.r)

"""
    in(x::Number, B::Ball)

Return `true` when the scalar `x` is proved to lie in the ball `B`: an upper bound of `|c − x|` is
compared with the radius. A number that is not a float is first enclosed by `Ball(x)`.
"""
Base.in(x::Union{AbstractFloat, Complex{<:AbstractFloat}}, B::Ball) = dist_up(B.c, x) <= B.r
Base.in(x::Number, B::Ball) = _ball_in(Ball(x), B)

# B1 ⊆ B2, proved: an upper bound of the distance of the centres plus the inner radius against
# the outer radius
_ball_in(B1::Ball, B2::Ball) = add_up(dist_up(B1.c, B2.c), B1.r) <= B2.r

"""
    in(B₁::Ball{T}, B₂::Ball{T})

Check whether the enclosure `B₁` is fully contained in `B₂`. The test
expands the endpoints using outward rounding to ensure a rigorous
decision.
"""
function Base.in(B1::Ball{T, T}, B2::Ball{T, T}) where {T <: AbstractFloat}
    upper = add_up(B1.c, B1.r) <= add_down(B2.c, B2.r)
    lower = sub_up(B2.c, B2.r) <= sub_down(B1.c, B1.r)
    return lower && upper
end

"""
    in(B₁::Ball{T, Complex{T}}, B₂::Ball{T, Complex{T}})

Containment test for complex balls. The check reduces the problem to the
real case by comparing the distance between midpoints with the radii of
the two enclosures.
"""
function Base.in(
        B1::Ball{T, Complex{T}}, B2::Ball{T, Complex{T}}) where {T <: AbstractFloat}
    return _ball_in(B1, B2)
end

"""
    in0(B₁::Ball, B₂::Ball)

Return `true` when `B₁` is contained in the **interior** of `B₂`, which for balls is
`|mid(B₁) − mid(B₂)| + rad(B₁) < rad(B₂)`. The left-hand side is bounded above, the distance of
the centres by `dist_up` and the sum by `add_up`, and compared against the stored radius, which is
exact, so a `true` answer is a proof of the containment. (The modulus of a complex difference is
not obtained from `abs` under an upward rounding mode: `abs` is `hypot`, which does not honour it,
and the difference itself rounds a negative component toward zero.)

This is the predicate Rump writes `in0` and defines in footnote 19 of

> S. M. Rump, *Verification methods: rigorous results using floating-point arithmetic*,
> Acta Numerica **19** (2010) 287-449, doi 10.1017/S096249291000005X,

as "checks `X ⊂ int(Y)` componentwise". Every self-mapping test in the Krawczyk-Moore-Rump
line needs the interior, not the closure: a fixed point on the boundary satisfies `X ⊆ Y`
without giving the contraction the argument rests on, so [`in`](@ref), which is not strict, is
not a substitute.
"""
function in0(B1::Ball{T, NT1}, B2::Ball{T, NT2}) where {T <: AbstractFloat, NT1, NT2}
    return add_up(dist_up(B1.c, B2.c), B1.r) < B2.r
end

#==============================================================================#
# Comparison operators for Ball
#==============================================================================#

"""
    isless(a::Ball, b::Ball)

Compare two balls by their midpoints. This provides a total ordering for
sorting and comparison operations. For rigorous "certainly less than"
semantics, use `sup(a) < inf(b)`.
"""
Base.isless(a::Ball{T, T}, b::Ball{T, T}) where {T <: AbstractFloat} = isless(a.c, b.c)

"""
    isless(a::Ball, b::Number)

Compare a ball with a number by comparing the ball's midpoint.
"""
Base.isless(a::Ball{T, T}, b::Number) where {T <: AbstractFloat} = isless(a.c, b)
Base.isless(a::Number, b::Ball{T, T}) where {T <: AbstractFloat} = isless(a, b.c)

"""
    <(a::Ball, b::Ball)

Compare two balls. Returns true if the ball midpoints satisfy a < b.
"""
Base.:(<)(a::Ball{T, T}, b::Ball{T, T}) where {T <: AbstractFloat} = a.c < b.c
Base.:(<)(a::Ball{T, T}, b::Number) where {T <: AbstractFloat} = a.c < b
Base.:(<)(a::Number, b::Ball{T, T}) where {T <: AbstractFloat} = a < b.c

"""
    <=(a::Ball, b::Ball)

Compare two balls. Returns true if the ball midpoints satisfy a <= b.
"""
Base.:(<=)(a::Ball{T, T}, b::Ball{T, T}) where {T <: AbstractFloat} = a.c <= b.c
Base.:(<=)(a::Ball{T, T}, b::Number) where {T <: AbstractFloat} = a.c <= b
Base.:(<=)(a::Number, b::Ball{T, T}) where {T <: AbstractFloat} = a <= b.c

"""
    >(a::Ball, b::Ball)

Compare two balls by midpoints.
"""
Base.:(>)(a::Ball{T, T}, b::Ball{T, T}) where {T <: AbstractFloat} = a.c > b.c
Base.:(>)(a::Ball{T, T}, b::Number) where {T <: AbstractFloat} = a.c > b
Base.:(>)(a::Number, b::Ball{T, T}) where {T <: AbstractFloat} = a > b.c

"""
    >=(a::Ball, b::Ball)

Compare two balls by midpoints.
"""
Base.:(>=)(a::Ball{T, T}, b::Ball{T, T}) where {T <: AbstractFloat} = a.c >= b.c
Base.:(>=)(a::Ball{T, T}, b::Number) where {T <: AbstractFloat} = a.c >= b
Base.:(>=)(a::Number, b::Ball{T, T}) where {T <: AbstractFloat} = a >= b.c
