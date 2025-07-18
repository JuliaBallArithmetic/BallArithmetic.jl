struct Ball{T <: AbstractFloat, CT <: Union{T, Complex{T}}} <: Number
    c::CT
    r::T
end

BallF64 = Ball{Float64, Float64}
BallComplexF64 = Ball{Float64, ComplexF64}

±(c, r) = Ball(c, r)
Ball(c, r) = Ball(float(c), float(r))
Ball(c::T) where {T <: Number} = Ball(float(c), zero(float(real(T))))
Ball(x::Ball) = x

mid(x::Ball) = x.c
rad(x::Ball) = x.r
mid(x::Number) = x
rad(::T) where {T <: Number} = zero(float(real(T)))

midtype(::Ball{T, CT}) where {T, CT} = CT
radtype(::Ball{T, CT}) where {T, CT} = T
midtype(::Type{Ball{T, CT}}) where {T, CT} = CT
radtype(::Type{Ball{T, CT}}) where {T, CT} = T
midtype(::Type{Ball}) = Float64
radtype(::Type{Ball}) = Float64

sup(x::Ball) = @up x.c + x.r
inf(x::Ball) = @down x.c - x.r

Base.show(io::IO, ::MIME"text/plain", x::Ball) = print(io, x.c, " ± ", x.r)

###############
# CONVERSIONS #
###############

function Base.convert(::Type{Ball{T, CT}}, x::Ball) where {T, CT}
    Ball(convert(CT, mid(x)), convert(T, rad(x)))
end
Base.convert(::Type{Ball{T, CT}}, c::Number) where {T, CT} = Ball(convert(CT, c), zero(T))
Base.convert(::Type{Ball}, c::Number) = Ball(c)

#########################
# ARITHMETIC OPERATIONS #
#########################

Base.:+(x::Ball) = x
Base.:-(x::Ball) = Ball(-x.c, x.r)
for op in (:+, :-)
    @eval function Base.$op(x::Ball, y::Ball)
        c = $op(mid(x), mid(y))
        r = @up (ϵp * abs(c) + rad(x)) + rad(y)
        Ball(c, r)
    end
end

function Base.:*(x::Ball, y::Ball)
    c = mid(x) * mid(y)
    r = @up (η + ϵp * abs(c)) + ((abs(mid(x)) + rad(x)) * rad(y) + rad(x) * abs(mid(y)))
    Ball(c, r)
end

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

Base.:/(x::Ball, y::Ball) = x * inv(y)

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
function Base.abs(x::Ball)
    if abs(x.c) > x.r
        return Ball(abs(x.c), x.r)
    else
        val = add_up(abs(x.c), x.r) / 2
        return Ball(val, val)
    end
end

Base.conj(x::Ball) = Ball(conj(x.c), x.r)
Base.in(x::Number, B::Ball) = abs(B.c - x) <= B.r

function Base.in(B1::Ball{T, T}, B2::Ball{T, T}) where {T <: AbstractFloat}
    upper = (@up B1.c + B1.r) <= (@down B2.c + B2.r)
    lower = (@up B2.c - B2.r) <= (@down B1.c - B1.r)
    return lower && upper
end

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

function Base.inv(x::Ball{T, Complex{T}}) where {T <: AbstractFloat}
    return conj(x) / (abs(x)^2)
end
