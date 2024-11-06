struct BallArray{T <: AbstractFloat, N, NT <: Union{T, Complex{T}},
    BT <: Ball{T, NT}, CA <: AbstractArray{NT, N}, RA <: AbstractArray{T, N}} <:
       AbstractArray{BT, N}
    c::CA
    r::RA
    function BallArray(c::AbstractArray{T, N},
            r::Array{T, N}) where {T <: AbstractFloat, N}
        new{T, N, T, Ball{T, T}, typeof(c), typeof(r)}(c, r)
    end
    function BallArray(c::AbstractArray{Complex{T}, N},
            r::Array{T, N}) where {T <: AbstractFloat, N}
        new{T, N, Complex{T}, Ball{T, Complex{T}}, typeof(c), typeof(r)}(c, r)
    end
end

BallArray(M::AbstractArray) = BallArray(mid(M), rad(M))
mid(A::AbstractArray) = A
rad(A::AbstractArray) = zeros(eltype(A), Base.size(A))

Base.size(A::BallArray) = Base.size(A.c)
Base.length(A::BallArray) = Base.length(A.c)

mid(A::BallArray) = A.c
rad(A::BallArray) = A.r

Base.getindex(A::BallArray, i::Int) = Ball(A.c[i], A.r[i])
function Base.getindex(A::BallArray, I::Vararg{Int, N}) where {N}
    Ball(Base.getindex(A.c, I...), Base.getindex(A.r, I...))
end

function Base.getindex(
        A::BallArray{T, N}, I::CartesianIndex{N}) where {T <: AbstractFloat, N}
    return Ball(Base.getindex(A.c, I), Base.getindex(A.r, I))
end

function Base.getindex(M::BallArray, inds...)
    return BallArray(Base.getindex(M.c, inds...), Base.getindex(M.r, inds...))
end

function Base.setindex!(M::BallArray, x, inds...)
    Base.setindex!(M.c, mid(x), inds...)
    Base.setindex!(M.r, rad(x), inds...)
end
Base.copy(M::BallArray) = BallArray(Base.copy(M.c), Base.copy(M.r))

function Base.real(A::BallArray{T, N, T}) where {T <: AbstractFloat, N}
    return A
end
function Base.imag(A::BallArray{T, N, T}) where {T <: AbstractFloat, N}
    BallArray(zeros(size(A)), zeros(size(A)))
end

function Base.real(A::BallArray{T, N, Complex{T}}) where {T <: AbstractFloat, N}
    BallArray(real.(A.c), A.r)
end
function Base.imag(A::BallArray{T, N, Complex{T}}) where {T <: AbstractFloat, N}
    BallArray(imag.(A.c), A.r)
end

# function Base.zeros(::Type{B}, dims::NTuple{N, Integer}) where {B <: Ball, N}
#     BallArray(zeros(midtype(B), dims), zeros(radtype(B), dims))
# end

# function Base.ones(::Type{B}, dims::NTuple{N, Integer}) where {B <: Ball, N}
#     BallArray(ones(midtype(B), dims), zeros(radtype(B), dims))
# end

# function Base.fill(x::Ball, I::Vararg{Int, N}) where {N}
#     BallArray(fill(mid(x), I...), fill(rad(x), I...))
# end
