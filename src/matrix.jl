struct BallMatrix{T<:AbstractFloat,NT<:Union{T,Complex{T}},BT<:Ball{T,NT},CM<:AbstractMatrix{NT},RM<:AbstractMatrix{T}} <: AbstractMatrix{BT}
    c::CM
    r::RM
    function BallMatrix(c::AbstractMatrix{T}, r::AbstractMatrix{T}) where {T<:AbstractFloat}
        new{T,T,Ball{T,T},typeof(c),typeof(r)}(c, r)
    end
    function BallMatrix(c::AbstractMatrix{Complex{T}}, r::AbstractMatrix{T}) where {T<:AbstractFloat}
        new{T,Complex{T},Ball{T,Complex{T}},typeof(c),typeof(r)}(c, r)
    end
end

mid(A::BallMatrix) = map(mid, A)
rad(A::BallMatrix) = map(rad, A)

# Array interface
Base.eltype(::BallMatrix{T,NT,BT}) where {T,NT,BT} = BT
Base.IndexStyle(::Type{<:BallMatrix}) = IndexLinear()
Base.size(M::BallMatrix, i...) = size(M.c, i...)
Base.getindex(M::BallMatrix, i::Int) = Ball(getindex(M.c, i), getindex(M.r, i))
function Base.setindex!(M::BallMatrix, x, inds...)
  setindex!(M.c, mid(x), inds...)
  setindex!(M.r, rad(x), inds...) 
end
Base.copy(M::BallMatrix) = BallMatrix(copy(M.c), copy(M.r))

# Operations
function Base.:*(A::BallMatrix{T}, B::BallMatrix{T}) where {T<:AbstractFloat}
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)
    C = mA * mB
    R = setrounding(T, RoundUp) do
        R = abs.(mA) * rB + rA * (abs.(mB) + rB)
    end
    BallMatrix(C, R)
end