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
    mid(v::AbstractVector)

Treat ordinary vectors as their own midpoints.
"""
mid(A::AbstractVector) = A

"""
    rad(v::AbstractVector)

Default radius for non-ball vectors: a zero vector of the appropriate
floating-point type.
"""
rad(A::AbstractVector{T}) where {T <: AbstractFloat} = zeros(T, size(A))
rad(A::AbstractVector{Complex{T}}) where {T <: AbstractFloat} = zeros(T, size(A))

# # Operations
for op in (:+, :-)
    @eval begin
        """
            Base.$(op)(A::BallVector, B::BallVector)

        Combine two ball vectors elementwise, enlarging the radius to
        include roundoff and the uncertainties of both operands.
        """
        function Base.$op(A::BallVector{T}, B::BallVector{T}) where {T <: AbstractFloat}
            mA, rA = mid(A), rad(A)
            mB, rB = mid(B), rad(B)

            C = $op(mA, mB)
            R = setrounding(T, RoundUp) do
                R = (ϵp * abs.(C) + rA) + rB
            end
            BallVector(C, R)
        end
    end
end

"""
    *(λ::Number, v::BallVector)

Scale a ball vector by a scalar. The midpoint is scaled directly while
the radius accounts for propagated uncertainty and roundoff.
"""
function Base.:*(lam::Number, A::BallVector{T}) where {T}
    B = LinearAlgebra.copymutable_oftype(A.c,
        Base._return_type(+,
            Tuple{eltype(A.c), typeof(lam)}))

    B = lam * A.c

    R = setrounding(T, RoundUp) do
        return (η .+ ϵp * abs.(B)) + (A.r * abs(mid(lam)))
    end

    return BallVector(B, R)
end

"""
    *(λ::Ball, v::BallVector)

Scale a ball vector by a ball-valued scalar, combining the uncertainty in
both arguments.
"""
function Base.:*(lam::Ball{T, NT}, A::BallVector{T}) where {T, NT <: Union{T, Complex{T}}}
    B = LinearAlgebra.copymutable_oftype(A.c,
        Base._return_type(+,
            Tuple{eltype(A.c),
                typeof(mid(lam))}))

    B = mid(lam) * A.c

    R = setrounding(T, RoundUp) do
        return (η .+ ϵp * abs.(B)) + ((abs.(A.c) + A.r) * rad(lam) + A.r * abs(mid(lam)))
    end

    return BallVector(B, R)
end

"""
    *(A::BallMatrix, v::AbstractVector)

Multiply a ball matrix with a plain vector by promoting the vector to a
column `BallMatrix` and reusing the matrix-matrix multiplication kernel.
"""
function Base.:*(A::BallMatrix, v::Vector)
    n = length(v)
    bV = BallMatrix(reshape(mid(v), (n, 1)))

    w = A * bV
    wc = vec(mid(w))
    wr = vec(rad(w))

    return BallVector(wc, wr)
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
