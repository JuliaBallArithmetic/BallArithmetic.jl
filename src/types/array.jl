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
        _check_axes(c, r)
        new{T, N, T, Ball{T, T}, typeof(c), typeof(r)}(c, r)
    end
    function BallArray(c::AbstractArray{Complex{T}, N},
            r::AbstractArray{T, N}) where {T <: AbstractFloat, N}
        _check_axes(c, r)
        new{T, N, Complex{T}, Ball{T, Complex{T}}, typeof(c), typeof(r)}(c, r)
    end
end

"""
    _check_axes(c, r)

Reject midpoint and radius containers that do not line up. Mismatched shapes
used to be accepted silently: the resulting object reported `size(c)` while
indexing or norm bounds later failed with a confusing `BoundsError` or
`DimensionMismatch` far from the construction site.

Comparing `axes` rather than `size` keeps offset-indexed arrays working.
The check is `O(1)`, so it costs nothing on the hot paths.
"""
function _check_axes(c::AbstractArray, r::AbstractArray)
    if axes(c) != axes(r)
        throw(DimensionMismatch("midpoint array has axes $(axes(c)) but radius array has axes $(axes(r))"))
    end
    return nothing
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
    _zero_radius(A, ::Type{T})

Return an all-zero, `T`-valued array laid out like `A`. Structured and sparse
midpoints keep their storage type, so `BallMatrix(Diagonal(...))` gets a
`Diagonal` radius rather than a dense one — for a sparse midpoint the dense
fallback costs orders of magnitude more memory than the midpoints themselves.

This is sound because every entry a structured type stores implicitly is an
*exact* zero (or, for the unit triangular types, an exact one), so the
corresponding radius is exactly zero and needs no storage.

`fill!(similar(A, T), zero(T))` is used rather than `zero(A)`, which would keep
a complex element type, or `zero(similar(A, T))`, which reads the undefined
references `similar` leaves behind for `BigFloat`.

For `Adjoint`, `Transpose`, `SubArray` and ranges `similar` already yields a
plain dense array, matching the previous behaviour.
"""
_zero_radius(A::AbstractArray, ::Type{T}) where {T} = fill!(similar(A, T), zero(T))

# The unit triangular types cannot represent a zero diagonal, so store the
# radius in the corresponding non-unit type: the unit diagonal is exactly one
# and therefore carries radius zero, which `UpperTriangular` can hold.
function _zero_radius(A::LinearAlgebra.UnitUpperTriangular, ::Type{T}) where {T}
    return LinearAlgebra.UpperTriangular(_zero_radius(parent(A), T))
end
function _zero_radius(A::LinearAlgebra.UnitLowerTriangular, ::Type{T}) where {T}
    return LinearAlgebra.LowerTriangular(_zero_radius(parent(A), T))
end

"""
    rad(A::AbstractArray)

Return a zero array of matching size that serves as the default radius
for non-ball arrays. The storage layout of `A` is preserved; see
[`_zero_radius`](@ref).
"""
rad(A::AbstractArray{T}) where {T <: AbstractFloat} = _zero_radius(A, T)
rad(A::AbstractArray{Complex{T}}) where {T <: AbstractFloat} = _zero_radius(A, T)

"""
    isvalid_enclosure(A::BallArray) -> Bool

Report whether every radius of `A` is a genuine enclosure radius, that is
non-negative and not `NaN`. A negative or `NaN` radius describes no set at all,
so an array failing this test carries no rigorous meaning.

**Validating the radii is the caller's responsibility.** The package
deliberately does not check them on construction: the radii it produces
internally are non-negative by construction (accumulated under `RoundUp` from
non-negative quantities), and scanning every entry would cost up to 76% of a
`BallMatrix` addition at `n = 100`. When radii come from outside — read from a
file, supplied by a user, converted from another package — call this (or
[`check_enclosure`](@ref)) yourself before relying on the enclosure.

The axes of `c` and `r` *are* checked on construction, since that test is
`O(1)`.
"""
isvalid_enclosure(A::BallArray) = all(x -> x >= zero(x), A.r)

"""
    check_enclosure(A::BallArray) -> A

Throw an `ArgumentError` unless [`isvalid_enclosure`](@ref) holds, otherwise
return `A` unchanged so the call can be chained.
"""
function check_enclosure(A::BallArray)
    isvalid_enclosure(A) ||
        throw(ArgumentError("radius array contains a negative or NaN entry; such a ball encloses nothing"))
    return A
end

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
    c = Base.getindex(M.c, inds...)
    r = Base.getindex(M.r, inds...)
    # Not every index pattern that reaches this fallback selects a sub-array:
    # mixing scalars with `CartesianIndex{0}` yields a single element, and Base
    # generates exactly that internally (`permutedims!` does). Wrapping such a
    # result in a `BallArray` used to throw a `MethodError`.
    return c isa AbstractArray ? BallArray(c, r) : Ball(c, r)
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
    # `zeros(size(A))` would hand back `Float64` storage whatever `T` is, so a
    # `Float32` or `BigFloat` array silently changed precision here.
    BallArray(_zero_radius(A.c, T), _zero_radius(A.r, T))
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
