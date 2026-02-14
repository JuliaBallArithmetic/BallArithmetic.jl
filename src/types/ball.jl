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

"""
    Ball(c, r)

Construct a ball whose midpoint is `c` and radius is `r`. Both arguments
are converted to floating-point values so that subsequent arithmetic
obeys the package's rounding assumptions.
"""
Ball(c, r) = Ball(float(c), float(r))

"""
    Ball(c::Number)

Create a degenerate ball representing the exact value `c`. The midpoint
is stored as `float(c)` and the radius is zero.
"""
Ball(c::T) where {T <: Number} = Ball(float(c), zero(float(real(T))))

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
    lower = min(inf(a), inf(b))
    upper = max(sup(a), sup(b))
    center = setrounding(T, RoundNearest) do
        (lower + upper) / 2
    end
    radius = setrounding(T, RoundUp) do
        (upper - lower) / 2
    end
    return Ball(center, radius)
end

function ball_hull(a::Ball{T, Complex{T}}, b::Ball{T, Complex{T}}) where {T}
    center_a = Ball(mid(a))
    center_b = Ball(mid(b))
    distance = abs(center_a - center_b)

    coverage_from_a = setrounding(T, RoundUp) do
        add_up(add_up(distance.c, distance.r), rad(b))
    end
    coverage_from_b = setrounding(T, RoundUp) do
        add_up(add_up(distance.c, distance.r), rad(a))
    end

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
    center = setrounding(T, RoundNearest) do
        (lower + upper) / 2
    end
    radius = setrounding(T, RoundUp) do
        (upper - lower) / 2
    end
    return Ball(center, radius)
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
    new_rad = setrounding(T, RoundUp) do
        convert(T, rad(x))
    end
    Ball(convert(CT, mid(x)), new_rad)
end

"""
    Base.convert(::Type{Ball{T, CT}}, c::Number)

Embed a plain number into a ball with zero radius whose midpoint matches
`c` converted to `CT`.
"""
Base.convert(::Type{Ball{T, CT}}, c::Number) where {T, CT} = Ball(convert(CT, c), zero(T))
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
    r = add_up(add_up(mul_up(ϵ, abs(c)), rad(x)), rad(y))
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
    r = add_up(add_up(mul_up(ϵ, abs(c)), rad(x)), rad(y))
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
    ϵ = machine_epsilon(T)
    η_val = subnormal_min(T)
    # r = (η + ϵ * |c|) + ((|mid(x)| + rad(x)) * rad(y) + rad(x) * |mid(y)|)
    abs_mx = abs(mid(x))
    abs_my = abs(mid(y))
    term1 = add_up(η_val, mul_up(ϵ, abs(c)))
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
function Base.inv(y::Ball{T}) where {T <: AbstractFloat}
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

# Base.abs(x::Ball) = Ball(max(0, sub_down(abs(mid(x)), rad(x))), add_up(abs(mid(x)), rad(x)))
#
"""
    abs(x::Ball)

Return a ball that encloses the absolute value of `x`. When the interval
does not cross zero, the midpoint is simply the absolute value of the
stored center; otherwise the result widens to account for the possible
sign change.
"""
function Base.abs(x::Ball)
    if abs(x.c) > x.r
        return Ball(abs(x.c), x.r)
    else
        # NOTE: Division by 2 is exact in IEEE 754 binary floating point
        # (just decrements the exponent), so no setrounding needed here.
        val = add_up(abs(x.c), x.r) / 2
        return Ball(val, val)
    end
end

"""
    conj(x::Ball)

Complex conjugate of a ball. The midpoint is conjugated while the radius
remains unchanged.
"""
Base.conj(x::Ball) = Ball(conj(x.c), x.r)

"""
    in(x::Number, B::Ball)

Return `true` if the scalar `x` is contained in the ball `B`.
"""
Base.in(x::Number, B::Ball) = abs(B.c - x) <= B.r

"""
    in(B₁::Ball{T}, B₂::Ball{T})

Check whether the enclosure `B₁` is fully contained in `B₂`. The test
expands the endpoints using outward rounding to ensure a rigorous
decision.
"""
function Base.in(B1::Ball{T, T}, B2::Ball{T, T}) where {T <: AbstractFloat}
    upper = (@up B1.c + B1.r) <= (@down B2.c + B2.r)
    lower = (@up B2.c - B2.r) <= (@down B1.c - B1.r)
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
    center1 = Ball(B1.c)
    center2 = Ball(B2.c)

    d = abs(center2 - center1)

    if B2.r >= add_up(add_up(d.c, d.r), B1.r)
        return true
    else
        return false
    end
end

"""
    inv(x::Ball{T, Complex{T}})

Return the multiplicative inverse of a complex ball by using the identity
`x⁻¹ = conj(x) / |x|²`. The helper relies on the existing real-valued
operations defined above to keep the enclosure rigorous.
"""
function Base.inv(x::Ball{T, Complex{T}}) where {T <: AbstractFloat}
    return conj(x) / (abs(x)^2)
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
