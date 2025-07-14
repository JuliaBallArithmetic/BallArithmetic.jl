const BallVector{T, NT, BT, CM, RM} = BallArray{T, 1, NT, BT, CM, RM}

# unclear why this need to be specified
BallVector(M::AbstractVector) = BallArray(mid(M), rad(M))
BallVector(c::AbstractVector, r::AbstractVector) = BallArray(c, r)

mid(A::AbstractVector) = A
rad(A::AbstractVector{T}) where {T <: AbstractVector} = zeros(T, size(A))
rad(A::AbstractVector{Complex{T}}) where {T <: AbstractFloat} = zeros(T, size(A))

# # Operations
for op in (:+, :-)
    @eval function Base.$op(A::BallVector{T}, B::BallVector{T}) where {T <: AbstractFloat}
        mA, rA = mid(A), rad(A)
        mB, rB = mid(B), rad(B)

        C = $op(mA, mB)
        R = setrounding(T, RoundUp) do
            R = (ϵp * abs.(C) + rA) + rB
        end
        BallVector(C, R)
    end
end

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

function Base.:*(A::BallMatrix, v::Vector)
    n = length(v)
    bV = reshape(mid(v), (n, 1))

    w = A * bV
    wc = vec(mid(w))
    wr = vec(rad(w))

    return BallVector(wc, wr)
end

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

Base.:*(A::AbstractMatrix, v::BallVector) = BallMatrix(A) * v
