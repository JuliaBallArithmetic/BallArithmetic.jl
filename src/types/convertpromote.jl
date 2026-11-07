"""
Promotion rules ensuring that `Ball` values interact smoothly with
integers, reals, and complex numbers. These follow Julia's promotion
conventions by preferring complex enclosures whenever one of the
arguments is complex-valued.
"""
#promote_type(Type{Ball{T, T}}, Type{Ball{T, Complex{T}}) where{T} = Ball{T, Complex{T}}

# Integers of every width. The rule names `Integer` rather than `Int64`, which left an `Int32`
# argument with no rule at all: on a 32-bit platform a machine integer literal is an `Int32`, so
# even `x + 1` failed there, and `Int8`, `Int16`, `Int128`, the unsigned widths and `BigInt` failed
# on every platform (issue #71). `convert(Ball{T, CT}, ::Number)` serves all of them and charges
# the rounding of the conversion to the radius, so an integer that `T` cannot hold exactly is
# still enclosed rather than silently rounded.
Base.promote_rule(::Type{Ball{T, T}}, ::Type{<:Integer}) where {T} = Ball{T, T}
function Base.promote_rule(::Type{Ball{T, Complex{T}}}, ::Type{<:Integer}) where {T}
    Ball{T, Complex{T}}
end
Base.promote_rule(::Type{Ball{T, T}}, ::Type{T}) where {T} = Ball{T, T}
function Base.promote_rule(::Type{Ball{T, Complex{T}}}, ::Type{T}) where {T}
    Ball{T, Complex{T}}
end
function Base.promote_rule(::Type{Ball{T, Complex{T}}}, ::Type{Ball{T, T}}) where {T}
    Ball{T, Complex{T}}
end
Base.promote_rule(::Type{Ball{T, T}}, ::Type{Complex{T}}) where {T} = Ball{T, Complex{T}}
function Base.promote_rule(::Type{Ball{T, Complex{T}}}, ::Type{Complex{T}}) where {T}
    Ball{T, Complex{T}}
end
Base.promote_rule(::Type{Ball{T, T}}, ::Type{Complex{Bool}}) where {T} = Ball{T, Complex{T}}
function Base.promote_rule(::Type{Ball{T, Complex{T}}}, ::Type{Complex{Bool}}) where {T}
    Ball{T, Complex{T}}
end

# BigFloat promotion rules - promote Float64 balls to BigFloat when mixed
Base.promote_rule(::Type{Ball{Float64, Float64}}, ::Type{Ball{BigFloat, BigFloat}}) = Ball{BigFloat, BigFloat}
Base.promote_rule(::Type{Ball{Float64, Complex{Float64}}}, ::Type{Ball{BigFloat, BigFloat}}) = Ball{BigFloat, Complex{BigFloat}}
Base.promote_rule(::Type{Ball{Float64, Float64}}, ::Type{Ball{BigFloat, Complex{BigFloat}}}) = Ball{BigFloat, Complex{BigFloat}}
Base.promote_rule(::Type{Ball{Float64, Complex{Float64}}}, ::Type{Ball{BigFloat, Complex{BigFloat}}}) = Ball{BigFloat, Complex{BigFloat}}

# BigFloat with scalar types
Base.promote_rule(::Type{Ball{BigFloat, BigFloat}}, ::Type{BigFloat}) = Ball{BigFloat, BigFloat}
Base.promote_rule(::Type{Ball{BigFloat, Complex{BigFloat}}}, ::Type{BigFloat}) = Ball{BigFloat, Complex{BigFloat}}
Base.promote_rule(::Type{Ball{BigFloat, BigFloat}}, ::Type{Complex{BigFloat}}) = Ball{BigFloat, Complex{BigFloat}}
Base.promote_rule(::Type{Ball{BigFloat, Complex{BigFloat}}}, ::Type{Complex{BigFloat}}) = Ball{BigFloat, Complex{BigFloat}}
Base.promote_rule(::Type{Ball{BigFloat, BigFloat}}, ::Type{Float64}) = Ball{BigFloat, BigFloat}
Base.promote_rule(::Type{Ball{BigFloat, Complex{BigFloat}}}, ::Type{Float64}) = Ball{BigFloat, Complex{BigFloat}}
Base.promote_rule(::Type{Ball{BigFloat, BigFloat}}, ::Type{Complex{Float64}}) = Ball{BigFloat, Complex{BigFloat}}
Base.promote_rule(::Type{Ball{BigFloat, Complex{BigFloat}}}, ::Type{Complex{Float64}}) = Ball{BigFloat, Complex{BigFloat}}

# `Int64` and `BigInt` against a BigFloat ball had their own rules here; the two `Integer` rules
# above cover them and return the same types.
