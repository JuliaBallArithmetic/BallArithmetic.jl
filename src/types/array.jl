"""
    BallArray{T, N, NT, BT, CA, RA}

Multi-dimensional array whose entries are [`Ball`](@ref) values. The type
stores midpoint data `c::CA` and radius data `r::RA` separately while
presenting an `AbstractArray{BT, N}` interface that behaves like an array
of balls. The parameters mirror the element type and storage layout and
are inferred automatically from the provided midpoint and radius
containers.
"""
struct BallArray{T <: AbstractFloat, N, NT <: Union{T, Complex{T}},
    BT <: Ball{T, NT}, CA <: AbstractArray{NT, N}, RA <: AbstractArray{T, N}} <:
       AbstractArray{BT, N}
    c::CA
    r::RA
    function BallArray(c::AbstractArray{T, N},
            r::AbstractArray{T, N}) where {T <: AbstractFloat, N}
        new{T, N, T, Ball{T, T}, typeof(c), typeof(r)}(c, r)
    end
    function BallArray(c::AbstractArray{Complex{T}, N},
            r::AbstractArray{T, N}) where {T <: AbstractFloat, N}
        new{T, N, Complex{T}, Ball{T, Complex{T}}, typeof(c), typeof(r)}(c, r)
    end
end

"""
    BallArray(A::AbstractArray)

Wrap an array of midpoints `A` into a `BallArray` with zero radii. This is
equivalent to calling `BallArray(mid(A), rad(A))` and is particularly
useful when upgrading an existing numeric array to a rigorous enclosure.
"""
BallArray(M::AbstractArray) = BallArray(mid(M), rad(M))

"""
    mid(A::AbstractArray)

Fallback definition that treats ordinary arrays as their own midpoint
representation. Specialisations for `BallArray` overload this method to
return the stored midpoint data.
"""
mid(A::AbstractArray) = A

"""
    rad(A::AbstractArray)

Return a zero array of matching size that serves as the default radius
for non-ball arrays.
"""
rad(A::AbstractArray{T}) where {T <: AbstractFloat} = zeros(T, Base.size(A))
rad(A::AbstractArray{Complex{T}}) where {T <: AbstractFloat} = zeros(T, Base.size(A))

"""
    size(A::BallArray)

Forward the size of the underlying midpoint storage.
"""
Base.size(A::BallArray) = Base.size(A.c)

"""
    length(A::BallArray)

Total number of elements stored in the array, matching the midpoint
container.
"""
Base.length(A::BallArray) = Base.length(A.c)

"""
    mid(A::BallArray)

Return the stored midpoint array.
"""
mid(A::BallArray) = A.c

"""
    rad(A::BallArray)

Return the stored radius array.
"""
rad(A::BallArray) = A.r

"""
    getindex(A::BallArray, inds...)

Indexing a `BallArray` returns either a single `Ball` or another
`BallArray` depending on the provided indices. Midpoints and radii are
looked up independently so that the enclosure remains rigorous.
"""
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

"""
    setindex!(A::BallArray, x, inds...)

Assign a value `x` to the given indices by storing its midpoint and
radius separately.
"""
function Base.setindex!(M::BallArray, x, inds...)
    Base.setindex!(M.c, mid(x), inds...)
    Base.setindex!(M.r, rad(x), inds...)
end

"""
    copy(A::BallArray)

Create a fresh `BallArray` with copies of both midpoint and radius
storage.
"""
Base.copy(M::BallArray) = BallArray(Base.copy(M.c), Base.copy(M.r))

"""
    real(A::BallArray)

Extract the real part of a `BallArray`. For purely real storage the
result is returned unchanged, while complex arrays drop the imaginary
part of the midpoint.
"""
function Base.real(A::BallArray{T, N, T}) where {T <: AbstractFloat, N}
    return A
end
function Base.real(A::BallArray{T, N, Complex{T}}) where {T <: AbstractFloat, N}
    BallArray(real.(A.c), A.r)
end

"""
    imag(A::BallArray)

Return the imaginary part of a `BallArray`. Real arrays produce a zero
enclosure, while complex arrays keep the stored radii and extract the
imaginary midpoints.
"""
function Base.imag(A::BallArray{T, N, T}) where {T <: AbstractFloat, N}
    BallArray(zeros(size(A)), zeros(size(A)))
end
function Base.imag(A::BallArray{T, N, Complex{T}}) where {T <: AbstractFloat, N}
    BallArray(imag.(A.c), A.r)
end

"""
    zeros(::Type{Ball}, dims)

Allocate a zero `BallArray` of the requested dimensions, using the
element type's midpoint and radius types to choose the storage format.
"""
function Base.zeros(::Type{B}, dims::NTuple{N, Integer}) where {B <: Ball, N}
    BallArray(zeros(midtype(B), dims), zeros(radtype(B), dims))
end

"""
    ones(::Type{Ball}, dims)

Return a `BallArray` whose midpoints are filled with ones and whose radii
are identically zero.
"""
function Base.ones(::Type{B}, dims::NTuple{N, Integer}) where {B <: Ball, N}
    BallArray(ones(midtype(B), dims), zeros(radtype(B), dims))
end

"""
    fill(x::Ball, dims...)

Create a `BallArray` where every element equals the ball `x`.
"""
function Base.fill(x::Ball, I::Vararg{Int, N}) where {N}
    BallArray(fill(mid(x), I...), fill(rad(x), I...))
end
