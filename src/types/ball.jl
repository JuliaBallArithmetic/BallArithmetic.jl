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

"""
    midtype(::Ball)

Return the type used to store the midpoint component of a `Ball`. This
is useful for allocating arrays that mirror the internal layout of a
ball or a collection of balls.
"""
midtype(::Ball{T, CT}) where {T, CT} = CT
radtype(::Ball{T, CT}) where {T, CT} = T
midtype(::Type{Ball{T, CT}}) where {T, CT} = CT
radtype(::Type{Ball{T, CT}}) where {T, CT} = T
midtype(::Type{Ball}) = Float64
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
    Ball(convert(CT, mid(x)), convert(T, rad(x)))
end

"""
    Base.convert(::Type{Ball{T, CT}}, c::Number)

Embed a plain number into a ball with zero radius whose midpoint matches
`c` converted to `CT`.
"""
Base.convert(::Type{Ball{T, CT}}, c::Number) where {T, CT} = Ball(convert(CT, c), zero(T))
Base.convert(::Type{Ball}, c::Number) = Ball(c)

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

for op in (:+, :-)
    @eval begin
        function Base.$op(x::Ball, y::Ball)
            c = $op(mid(x), mid(y))
            r = @up (ϵp * abs(c) + rad(x)) + rad(y)
            Ball(c, r)
        end
    end
end

"""
    *(x::Ball, y::Ball)

Multiply two balls and return the enclosure of the product. The midpoint
is the product of the midpoints, whereas the radius collects propagated
uncertainty from both operands and the intrinsic rounding error of the
operation.
"""
function Base.:*(x::Ball, y::Ball)
    c = mid(x) * mid(y)
    r = @up (η + ϵp * abs(c)) + ((abs(mid(x)) + rad(x)) * rad(y) + rad(x) * abs(mid(y)))
    Ball(c, r)
end

"""
    inv(x::Ball)

Return the multiplicative inverse of a real ball. The method throws an
`ArgumentError` when the interval straddles zero, since no rigorous
inverse can be produced in that case.
"""
# TODO: this probably is incorrect for complex balls
function Base.inv(y::Ball{<:AbstractFloat})
    my, ry = mid(y), rad(y)
    ry < abs(my) || throw(ArgumentError("Ball $y contains zero."))
    c1 = @down 1.0 / (abs(my) + ry)
    c2 = @up 1.0 / (abs(my) - ry)
    c = @up c1 + 0.5 * (c2 - c1)
    r = @up c - c1
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
function Base.sqrt(y::Ball{Float64}) where {T}
    my, ry = mid(y), rad(y)
    ry < my || throw(DomainError("Ball $y contains zero."))
    c1 = sqrt_down(@down my - ry)
    c2 = sqrt_up(@up my + ry)
    c = @up c1 + 0.5 * (c2 - c1)
    r = @up c - c1
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
