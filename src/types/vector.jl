"""
    BallVector{T, NT, BT, CM, RM}

Alias for the one-dimensional [`BallArray`](@ref), representing vectors
of balls.
"""
const BallVector{T, NT, BT, CM, RM} = BallArray{T, 1, NT, BT, CM, RM}

"""
    BallVector(v::AbstractVector)

Wrap a vector of midpoints into a `BallVector` with zero radii.
"""
BallVector(M::AbstractVector) = BallArray(mid(M), rad(M))

"""
    BallVector(c::AbstractVector, r::AbstractVector)

Construct a `BallVector` from matching midpoint and radius arrays.
"""
BallVector(c::AbstractVector, r::AbstractVector) = BallArray(c, r)

"""
    BallVector(v::AbstractVector{<:Ball})

Convert a vector of `Ball` elements to a `BallVector` by extracting
midpoints and radii into separate arrays.
"""
function BallVector(v::AbstractVector{<:Ball})
    c = [x.c for x in v]
    r = [x.r for x in v]
    BallArray(c, r)
end

# Conversion methods
Base.convert(::Type{BallVector}, v::AbstractVector{<:Ball}) = BallVector(v)
Base.convert(::Type{<:BallVector}, v::AbstractVector{<:Ball}) = BallVector(v)

"""
    mid(v::AbstractVector)

Treat ordinary vectors as their own midpoints.
"""
mid(A::AbstractVector) = A

"""
    rad(v::AbstractVector)

Default radius for non-ball vectors: a zero vector of the appropriate
floating-point type. The storage layout of `A` is preserved, so a sparse
vector yields a sparse radius; see [`_zero_radius`](@ref).
"""
rad(A::AbstractVector{T}) where {T <: AbstractFloat} = _zero_radius(A, T)
rad(A::AbstractVector{Complex{T}}) where {T <: AbstractFloat} = _zero_radius(A, T)

# # Operations
"""
    Base.:+(A::BallVector, B::BallVector)

Combine two ball vectors elementwise using addition, enlarging the radius
to include roundoff and the uncertainties of both operands.
"""
function Base.:+(A::BallVector{T}, B::BallVector{T}) where {T <: AbstractFloat}
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)

    C = mA + mB
    ϵ = machine_epsilon(T)
    R = setrounding(T, RoundUp) do
        (ϵ * abs.(C) + rA) + rB
    end
    BallVector(C, R)
end

# the opposite of a ball vector: exact
Base.:-(A::BallVector) = BallVector(-A.c, copy(A.r))

"""
    Base.:-(A::BallVector, B::BallVector)

Combine two ball vectors elementwise using subtraction, enlarging the
radius to include roundoff and the uncertainties of both operands.
"""
function Base.:-(A::BallVector{T}, B::BallVector{T}) where {T <: AbstractFloat}
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)

    C = mA - mB
    ϵ = machine_epsilon(T)
    R = setrounding(T, RoundUp) do
        (ϵ * abs.(C) + rA) + rB
    end
    BallVector(C, R)
end

"""
    *(λ::Number, v::BallVector)

Scale a ball vector by a scalar. The midpoint is scaled directly while
the radius accounts for propagated uncertainty and roundoff.
"""
function Base.:*(lam::Number, A::BallVector{T}) where {T}
    m, r = _scalar_factor(T, lam)
    B, R = _scale_ball_array(m, r, A.c, A.r)
    return BallVector(B, R)
end

"""
    *(λ::Ball, v::BallVector)

Scale a ball vector by a ball-valued scalar, combining the uncertainty in
both arguments.
"""
function Base.:*(lam::Ball{T, NT}, A::BallVector{T}) where {T, NT <: Union{T, Complex{T}}}
    B, R = _scale_ball_array(mid(lam), rad(lam), A.c, A.r)
    return BallVector(B, R)
end

"""
    *(A::BallMatrix, v::AbstractVector)

Multiply a ball matrix with a plain vector by promoting the vector to a
column `BallMatrix` and reusing the matrix-matrix multiplication kernel.

Typed on `AbstractVector` rather than `Vector` so that views, ranges and
sparse vectors take the rigorous kernel too; previously they fell through to
generic element-wise `Ball` multiplication, which is far slower and returns a
`Vector{Ball}` instead of a `BallVector`.
"""
function Base.:*(A::BallMatrix, v::AbstractVector)
    n = length(v)
    # `reshape` refuses some lazy vectors, so materialise when it does. The
    # radius is carried through explicitly: `rad` is zero for a plain vector,
    # but this keeps the method correct for any vector-like input.
    vc = _as_column(mid(v), n)
    vr = _as_column(rad(v), n)
    bV = BallMatrix(vc, vr)

    w = A * bV
    wc = vec(mid(w))
    wr = vec(rad(w))

    return BallVector(wc, wr)
end

function _as_column(v::AbstractVector, n::Integer)
    try
        return reshape(v, (n, 1))
    catch
        return reshape(collect(v), (n, 1))
    end
end

"""
    *(A::BallMatrix, v::BallVector)

Multiply a ball matrix with a ball vector. The vector is reshaped into a
column matrix so that the existing `BallMatrix` multiplication handles
the enclosure bookkeeping.
"""
function Base.:*(A::BallMatrix, v::BallVector)
    n = length(v)
    vc = reshape(mid(v), (n, 1))
    vr = reshape(rad(v), (n, 1))
    B = BallMatrix(vc, vr)
    w = A * B

    wc = vec(mid(w))
    wr = vec(rad(w))

    return BallVector(wc, wr)
end

"""
    *(A::AbstractMatrix, v::BallVector)

Promote a plain matrix to a `BallMatrix` before multiplying it with a
ball vector.
"""
Base.:*(A::AbstractMatrix, v::BallVector) = BallMatrix(A) * v
