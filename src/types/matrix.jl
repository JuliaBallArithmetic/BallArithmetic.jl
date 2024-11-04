struct BallMatrix{T <: AbstractFloat, NT <: Union{T, Complex{T}}, BT <: Ball{T, NT},
    CM <: AbstractMatrix{NT}, RM <: AbstractMatrix{T}} <: AbstractMatrix{BT}
    c::CM
    r::RM
    function BallMatrix(c::AbstractMatrix{T},
            r::AbstractMatrix{T}) where {T <: AbstractFloat}
        new{T, T, Ball{T, T}, typeof(c), typeof(r)}(c, r)
    end
    function BallMatrix(c::AbstractMatrix{Complex{T}},
            r::AbstractMatrix{T}) where {T <: AbstractFloat}
        new{T, Complex{T}, Ball{T, Complex{T}}, typeof(c), typeof(r)}(c, r)
    end
end

BallMatrix(M::AbstractMatrix) = BallMatrix(mid.(M), rad.(M))
mid(A::AbstractMatrix) = A
rad(A::AbstractMatrix) = zeros(eltype(A), size(A))

# mid(A::BallMatrix) = map(mid, A)
# rad(A::BallMatrix) = map(rad, A)
mid(A::BallMatrix) = A.c
rad(A::BallMatrix) = A.r

function Base.real(A::BallMatrix{T, T}) where {T <: AbstractFloat}
    return A
end
function Base.imag(A::BallMatrix{T, T}) where {T <: AbstractFloat}
    BallMatrix(zeros(size(A)), zeros(size(A)))
end

function Base.real(A::BallMatrix{T, Complex{T}}) where {T <: AbstractFloat}
    BallMatrix(real.(A.c), A.r)
end
function Base.imag(A::BallMatrix{T, Complex{T}}) where {T <: AbstractFloat}
    BallMatrix(imag.(A.c), A.r)
end

# Array interface
Base.eltype(::BallMatrix{T, NT, BT}) where {T, NT, BT} = BT
Base.IndexStyle(::Type{<:BallMatrix}) = IndexLinear()
Base.size(M::BallMatrix, i...) = size(M.c, i...)

function Base.getindex(M::BallMatrix, i::Int64)
    return Ball(getindex(M.c, i), getindex(M.r, i))
end

function Base.getindex(M::BallMatrix, I::CartesianIndex{1})
    return Ball(getindex(M.c, I), getindex(M.r, I))
end

function Base.getindex(M::BallMatrix, i::Int64, j::Int64)
    return Ball(getindex(M.c, i, j), getindex(M.r, i, j))
end

function Base.getindex(M::BallMatrix, I::CartesianIndex{2})
    return Ball(getindex(M.c, I), getindex(M.r, I))
end

function Base.getindex(M::BallMatrix, inds...)
    return BallMatrix(getindex(M.c, inds...), getindex(M.r, inds...))
end

function Base.display(X::BallMatrix{
        T, NT, Ball{T, NT}, Matrix{NT},
        Matrix{T}}) where {T <: AbstractFloat, NT <: Union{T, Complex{T}}}
    #@info "test"
    m, n = size(X)
    B = [Ball(X.c[i, j], X.r[i, j]) for i in 1:m, j in 1:n]
    display(B)
end

function Base.setindex!(M::BallMatrix, x, inds...)
    setindex!(M.c, mid(x), inds...)
    setindex!(M.r, rad(x), inds...)
end
Base.copy(M::BallMatrix) = BallMatrix(copy(M.c), copy(M.r))

function Base.zeros(::Type{B}, dims::NTuple{N, Integer}) where {B <: Ball, N}
    BallMatrix(zeros(midtype(B), dims), zeros(radtype(B), dims))
end

function Base.ones(::Type{B}, dims::NTuple{N, Integer}) where {B <: Ball, N}
    BallMatrix(ones(midtype(B), dims), zeros(radtype(B), dims))
end

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
