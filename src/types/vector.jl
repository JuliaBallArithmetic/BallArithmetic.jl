struct BallVector{T <: AbstractFloat, NT <: Union{T, Complex{T}}, BT <: Ball{T, NT},
    CV <: AbstractVector{NT}, RV <: AbstractVector{T}} <: AbstractVector{BT}
    c::CV
    r::RV
    function BallVector(c::AbstractVector{T},
            r::AbstractVector{T}) where {T <: AbstractFloat}
        new{T, T, Ball{T, T}, typeof(c), typeof(r)}(c, r)
    end
    function BallVector(c::AbstractVector{Complex{T}},
            r::AbstractVector{T}) where {T <: AbstractFloat}
        new{T, Complex{T}, Ball{T, Complex{T}}, typeof(c), typeof(r)}(c, r)
    end
end

BallMatrix(M::AbstractVector) = BallVector(mid.(M), rad.(M))
mid(A::AbstractVector) = A
rad(A::AbstractVector) = zeros(eltype(A), size(A))

# mid(A::BallMatrix) = map(mid, A)
# rad(A::BallMatrix) = map(rad, A)
mid(A::BallVector) = A.c
rad(A::BallVector) = A.r

# Array interface
Base.eltype(::BallVector{T, NT, BT}) where {T, NT, BT} = BT
Base.IndexStyle(::Type{<:BallVector}) = IndexLinear()
Base.size(M::BallVector, i...) = size(M.c, i...)
function Base.getindex(M::BallVector, inds...)
    return BallVector(getindex(M.c, inds...), getindex(M.r, inds...))
end

function Base.display(v::BallVector{
        T, NT, Ball{T, NT}, Vector{NT},
        Vector{T}}) where {T <: AbstractFloat, NT <: Union{T, Complex{T}}}
    #@info "test"
    m = length(v)
    V = [Ball(v.c[i], v.r[i]) for i in 1:m]
    display(V)
end

function Base.setindex!(M::BallVector, x, inds...)
    setindex!(M.c, mid(x), inds...)
    setindex!(M.r, rad(x), inds...)
end
Base.copy(M::BallVector) = BallVector(copy(M.c), copy(M.r))

function Base.zeros(::Type{B}, n::Integer) where {B <: Ball}
    BallVector(zeros(midtype(B), n), zeros(radtype(B), n))
end

function Base.ones(::Type{B}, n::Integer) where {B <: Ball}
    BallVector(ones(midtype(B), n), zeros(radtype(B), n))
end

# # LinearAlgebra functions
# function LinearAlgebra.adjoint(M::BallMatrix)
#     return BallMatrix(mid(M)', rad(M)')
# end

# # Operations
# for op in (:+, :-)
#     @eval function Base.$op(A::BallMatrix{T}, B::BallMatrix{T}) where {T <: AbstractFloat}
#         mA, rA = mid(A), rad(A)
#         mB, rB = mid(B), rad(B)

#         C = $op(mA, mB)
#         R = setrounding(T, RoundUp) do
#             R = (ϵp * abs.(C) + rA) + rB
#         end
#         BallMatrix(C, R)
#     end
# end

# function Base.:*(lam::Number, A::BallMatrix{T}) where {T}
#     B = LinearAlgebra.copymutable_oftype(A.c,
#         Base._return_type(+,
#             Tuple{eltype(A.c), typeof(lam)}))

#     B = lam * A.c

#     R = setrounding(T, RoundUp) do
#         return (η .+ ϵp * abs.(B)) + (A.r * abs(mid(lam)))
#     end

#     return BallMatrix(B, R)
# end

# function Base.:*(lam::Ball{T, NT}, A::BallMatrix{T}) where {T, NT <: Union{T, Complex{T}}}
#     B = LinearAlgebra.copymutable_oftype(A.c,
#         Base._return_type(+,
#             Tuple{eltype(A.c),
#                 typeof(mid(lam))}))

#     B = mid(lam) * A.c

#     R = setrounding(T, RoundUp) do
#         return (η .+ ϵp * abs.(B)) + ((abs.(A.c) + A.r) * rad(lam) + A.r * abs(mid(lam)))
#     end

#     return BallMatrix(B, R)
# end
