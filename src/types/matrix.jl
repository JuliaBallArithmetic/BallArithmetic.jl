# struct BallMatrix{T <: AbstractFloat, NT <: Union{T, Complex{T}}, BT <: Ball{T, NT},
#     CM <: AbstractMatrix{NT}, RM <: AbstractMatrix{T}} <: AbstractMatrix{BT}
#     c::CM
#     r::RM
#     function BallMatrix(c::AbstractMatrix{T},
#             r::AbstractMatrix{T}) where {T <: AbstractFloat}
#         new{T, T, Ball{T, T}, typeof(c), typeof(r)}(c, r)
#     end
#     function BallMatrix(c::AbstractMatrix{Complex{T}},
#             r::AbstractMatrix{T}) where {T <: AbstractFloat}
#         new{T, Complex{T}, Ball{T, Complex{T}}, typeof(c), typeof(r)}(c, r)
#     end
# end

const BallMatrix{T, NT, BT, CM, RM} = BallArray{T, 2, NT, BT, CM, RM}

# unclear why this need to be specified
BallMatrix(M::AbstractMatrix) = BallArray(mid(M), rad(M))
BallMatrix(c::AbstractMatrix, r::AbstractMatrix) = BallArray(c, r)

mid(A::AbstractMatrix) = A
rad(A::AbstractMatrix{T}) where {T <: AbstractFloat} = zeros(T, size(A))
rad(A::AbstractMatrix{Complex{T}}) where {T <: AbstractFloat} = zeros(T, size(A))

# LinearAlgebra functions
function LinearAlgebra.adjoint(M::BallMatrix)
    return BallMatrix(mid(M)', rad(M)')
end

# Operations
for op in (:+, :-)
    @eval function Base.$op(A::BallMatrix{T}, B::BallMatrix{T}) where {T <: AbstractFloat}
        mA, rA = mid(A), rad(A)
        mB, rB = mid(B), rad(B)

        C = $op(mA, mB)
        R = setrounding(T, RoundUp) do
            R = (ϵp * abs.(C) + rA) + rB
        end
        BallMatrix(C, R)
    end
end

function Base.:*(lam::Number, A::BallMatrix{T}) where {T}
    B = LinearAlgebra.copymutable_oftype(A.c,
        Base._return_type(+,
            Tuple{eltype(A.c), typeof(lam)}))

    B = lam * A.c

    R = setrounding(T, RoundUp) do
        return (η .+ ϵp * abs.(B)) + (A.r * abs(mid(lam)))
    end

    return BallMatrix(B, R)
end

function Base.:*(lam::Ball{T, NT}, A::BallMatrix{T}) where {T, NT <: Union{T, Complex{T}}}
    B = LinearAlgebra.copymutable_oftype(A.c,
        Base._return_type(+,
            Tuple{eltype(A.c),
                typeof(mid(lam))}))

    B = mid(lam) * A.c

    R = setrounding(T, RoundUp) do
        return (η .+ ϵp * abs.(B)) + ((abs.(A.c) + A.r) * rad(lam) + A.r * abs(mid(lam)))
    end

    return BallMatrix(B, R)
end

# function Base.:*(lam::NT, A::BallMatrix{T}) where {T, NT<:Union{T,Complex{T}}}
#     B = LinearAlgebra.copymutable_oftype(A.c, Base._return_type(+, Tuple{eltype(A.c),typeof(mid(lam))}))

#     B = lam * A.c

#     R = setrounding(T, RoundUp) do
#         return (η .+ ϵp * abs.(B)) + (A.r * abs(mid(lam)))
#     end

#     return BallMatrix(B, R)
# end

for op in (:+, :-)
    @eval function Base.$op(A::BallMatrix{T}, B::Matrix{T}) where {T <: AbstractFloat}
        mA, rA = mid(A), rad(A)

        C = $op(mA, B)

        R = setrounding(T, RoundUp) do
            R = (ϵp * abs.(C) + rA)
        end
        BallMatrix(C, R)
    end
    # + and - are commutative
    @eval function Base.$op(B::Matrix{T}, A::BallMatrix{T}) where {T <: AbstractFloat}
        $op(A, B)
    end
end

function Base.:+(A::BallMatrix{T}, J::UniformScaling) where {T}
    LinearAlgebra.checksquare(A)
    B = LinearAlgebra.copymutable_oftype(A.c,
        Base._return_type(+,
            Tuple{eltype(A.c), typeof(J)}))
    R = copy(A.r)
    @inbounds for i in axes(A, 1)
        B[i, i] += J
    end

    R = setrounding(T, RoundUp) do
        @inbounds for i in axes(A, 1)
            R[i, i] += ϵp * abs(B[i, i])
        end
        return R
    end
    return BallMatrix(B, R)
end

function Base.:+(A::BallMatrix{T},
        J::UniformScaling{Ball{T, NT}}) where {T, NT <: Union{T, Complex{T}}}
    LinearAlgebra.checksquare(A)
    B = LinearAlgebra.copymutable_oftype(A.c, Base._return_type(+, Tuple{eltype(A.c), NT}))
    R = copy(A.r)
    @inbounds for i in axes(A, 1)
        B[i, i] += J.λ.c
    end

    R = setrounding(T, RoundUp) do
        @inbounds for i in axes(A, 1)
            R[i, i] += ϵp * abs(B[i, i]) + J.λ.r
        end
        return R
    end
    return BallMatrix(B, R)
end

function Base.:+(J::UniformScaling, A::BallMatrix)
    return A + J
end

function Base.:-(A::BallMatrix{T}, J::UniformScaling) where {T}
    LinearAlgebra.checksquare(A)
    B = LinearAlgebra.copymutable_oftype(A.c,
        Base._return_type(-,
            Tuple{eltype(A.c), typeof(J)}))
    R = copy(A.r)
    @inbounds for i in axes(A, 1)
        B[i, i] -= J
    end

    R = setrounding(T, RoundUp) do
        @inbounds for i in axes(A, 1)
            R[i, i] += ϵp * abs(B[i, i])
        end
        return R
    end
    return BallMatrix(B, R)
end

function Base.:-(A::BallMatrix{T},
        J::UniformScaling{Ball{T, NT}}) where {T, NT <: Union{T, Complex{T}}}
    LinearAlgebra.checksquare(A)
    B = LinearAlgebra.copymutable_oftype(A.c, Base._return_type(+, Tuple{eltype(A.c), NT}))
    R = copy(A.r)
    @inbounds for i in axes(A, 1)
        B[i, i] -= J.λ.c
    end

    R = setrounding(T, RoundUp) do
        @inbounds for i in axes(A, 1)
            R[i, i] += ϵp * abs(B[i, i]) + J.λ.r
        end
        return R
    end
    return BallMatrix(B, R)
end

function Base.:-(J::UniformScaling, A::BallMatrix{T}) where {T}
    LinearAlgebra.checksquare(A)
    B = LinearAlgebra.copymutable_oftype(A.c,
        Base._return_type(-,
            Tuple{eltype(A.c), typeof(J)}))
    R = copy(A.r)
    @inbounds for i in axes(A, 1)
        B[i, i] = J - B[i, i]
    end

    R = setrounding(T, RoundUp) do
        @inbounds for i in axes(A, 1)
            R[i, i] += ϵp * abs(B[i, i])
        end
        return R
    end
    return BallMatrix(B, R)
end

function Base.:-(J::UniformScaling{Ball{T, NT}},
        A::BallMatrix{T}
) where {T, NT <: Union{T, Complex{T}}}
    LinearAlgebra.checksquare(A)
    B = LinearAlgebra.copymutable_oftype(A.c, Base._return_type(+, Tuple{eltype(A.c), NT}))
    R = copy(A.r)
    @inbounds for i in axes(A, 1)
        B[i, i] = J.λ.c - B[i, i]
    end

    R = setrounding(T, RoundUp) do
        @inbounds for i in axes(A, 1)
            R[i, i] += ϵp * abs(B[i, i]) + J.λ.r
        end
        return R
    end
    return BallMatrix(B, R)
end

include("MMul/MMul2.jl")
include("MMul/MMul3.jl")
include("MMul/MMul4.jl")
include("MMul/MMul5.jl")

function Base.:*(A::BallMatrix{T, S}, B::BallMatrix{T, S}) where {S, T <: AbstractFloat}
    return MMul4(A, B)
end

function Base.:*(A::BallMatrix{T, S}, B::Matrix{S}) where {S, T <: AbstractFloat}
    return MMul4(A, B)
end

function Base.:*(A::Matrix{S}, B::BallMatrix{T, S}) where {S, T <: AbstractFloat}
    return MMul4(A, B)
end

function Base.:*(
        A::BallMatrix{T, Complex{T}},
        B::BallMatrix{T, T}) where {T <: AbstractFloat}
    return real(A) * B + im * (imag(A) * B)
end

function Base.:*(
        A::BallMatrix{T, T},
        B::BallMatrix{T, Complex{T}}) where {T <: AbstractFloat}
    return A * real(B) + im * (A * imag(B))
end

function Base.:*(
        A::BallMatrix{T, Complex{T}},
        B::BallMatrix{T, Complex{T}}) where {T <: AbstractFloat}
    return A * real(B) + im * (A * imag(B))
end
