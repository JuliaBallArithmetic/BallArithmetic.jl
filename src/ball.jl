struct Ball{T<:AbstractFloat,CT<:Union{T,Complex{T}}} <: Number
    c::CT
    r::T
end
±(c, r) = Ball(c, r)
Ball(c, r) = Ball(float(c), float(r))

Base.show(io::IO,  ::MIME"text/plain", x::Ball) = print(io, x.c, " ± ", x.r)

mid(x::Ball) = x.c
rad(x::Ball) = x.r
mid(x::Number) = x
rad(::T) where {T<:Number} = zero(float(real(T)))

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
    r = @up (η +  ϵp * abs(c)) + ((abs(mid(x)) + rad(x)) * rad(y) + rad(x) * abs(mid(y))) 
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

Base.abs(x::Ball) = Ball(max(0, sub_down(abs(mid(x)), rad(x))), add_up(abs(mid(x)), rad(x)))
