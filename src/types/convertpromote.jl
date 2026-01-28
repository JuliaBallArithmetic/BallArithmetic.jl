"""
Promotion rules ensuring that `Ball` values interact smoothly with
integers, reals, and complex numbers. These follow Julia's promotion
conventions by preferring complex enclosures whenever one of the
arguments is complex-valued.
"""
#promote_type(Type{Ball{T, T}}, Type{Ball{T, Complex{T}}) where{T} = Ball{T, Complex{T}}

Base.promote_rule(::Type{Ball{T, T}}, ::Type{Int64}) where {T} = Ball{T, T}
function Base.promote_rule(::Type{Ball{T, Complex{T}}}, ::Type{Int64}) where {T}
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
Base.promote_rule(::Type{Ball{BigFloat, BigFloat}}, ::Type{Int64}) = Ball{BigFloat, BigFloat}
Base.promote_rule(::Type{Ball{BigFloat, Complex{BigFloat}}}, ::Type{Int64}) = Ball{BigFloat, Complex{BigFloat}}
Base.promote_rule(::Type{Ball{BigFloat, BigFloat}}, ::Type{Float64}) = Ball{BigFloat, BigFloat}
Base.promote_rule(::Type{Ball{BigFloat, Complex{BigFloat}}}, ::Type{Float64}) = Ball{BigFloat, Complex{BigFloat}}
Base.promote_rule(::Type{Ball{BigFloat, BigFloat}}, ::Type{Complex{Float64}}) = Ball{BigFloat, Complex{BigFloat}}
Base.promote_rule(::Type{Ball{BigFloat, Complex{BigFloat}}}, ::Type{Complex{Float64}}) = Ball{BigFloat, Complex{BigFloat}}

# BigInt promotion (convert to BigFloat ball)
Base.promote_rule(::Type{Ball{BigFloat, BigFloat}}, ::Type{BigInt}) = Ball{BigFloat, BigFloat}
Base.promote_rule(::Type{Ball{BigFloat, Complex{BigFloat}}}, ::Type{BigInt}) = Ball{BigFloat, Complex{BigFloat}}
