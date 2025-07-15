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

# we should implement conversion rules also for BigInt and BigFloat...
