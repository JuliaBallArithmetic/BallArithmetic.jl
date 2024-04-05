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

# Array interface
Base.eltype(::BallMatrix{T, NT, BT}) where {T, NT, BT} = BT
Base.IndexStyle(::Type{<:BallMatrix}) = IndexLinear()
Base.size(M::BallMatrix, i...) = size(M.c, i...)

# function Base.getindex(M::BallMatrix, i1:Int64, i2:Int64)
#     return Ball(getindex(M.c, i1, i2), getindex(M.r, i1, i2))
# end

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

function Base.:*(A::BallMatrix{T}, B::BallMatrix{T}) where {T <: AbstractFloat}
    # mA, rA = mid(A), rad(A)
    # mB, rB = mid(B), rad(B)
    # C = mA * mB
    # R = setrounding(T, RoundUp) do
    #     R = abs.(mA) * rB + rA * (abs.(mB) + rB)
    # end
    # BallMatrix(C, R)
    return MMul3(A, B)
end

function Base.:*(A::BallMatrix{T}, B::Matrix{T}) where {T <: AbstractFloat}
    return MMul3(A, B)
end

function Base.:*(A::Matrix{T}, B::BallMatrix{T}) where {T <: AbstractFloat}
    return MMul3(A, B)
end

# TODO: Should we implement this?
# From Theveny https://theses.hal.science/tel-01126973/en
function MMul2(A::BallMatrix{T}, B::BallMatrix{T}) where {T <: AbstractFloat}
    @warn "Not Implemented"
end

# As in Revol-Theveny
# Parallel Implementation of Interval Matrix Multiplication
# pag. 4
# please check the values of u and η

function MMul3(A::BallMatrix{T}, B::BallMatrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)
    mC = mA * mB
    rC = setrounding(T, RoundUp) do
        rprimeB = ((k + 2) * ϵp * abs.(mB) + rB)
        rC = abs.(mA) * rprimeB + rA * (abs.(mB) + rB) .+ η / ϵp
    end
    BallMatrix(mC, rC)
end

function MMul3(A::BallMatrix{T}, B::Matrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    mA, rA = mid(A), rad(A)
    mC = mA * B
    rC = setrounding(T, RoundUp) do
        rprimeB = ((k + 2) * ϵp * abs.(B))
        rC = abs.(mA) * rprimeB + rA * (abs.(B)) .+ η / ϵp
    end
    BallMatrix(mC, rC)
end

function MMul3(A::Matrix{T}, B::BallMatrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    mB, rB = mid(B), rad(B)
    mC = A * mB
    rC = setrounding(T, RoundUp) do
        rprimeB = ((k + 2) * ϵp * abs.(mB) + rB)
        rC = abs.(A) * rprimeB .+ η / ϵp
    end
    BallMatrix(mC, rC)
end

function MMul3(A::Matrix{T}, B::Matrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    mC = A * B
    rC = setrounding(T, RoundUp) do
        rprimeB = ((k + 2) * ϵp * abs.(B))
        rC = abs.(A) * rprimeB .+ η / ϵp
    end
    BallMatrix(mC, rC)
end

# As in Revol-Theveny
# Parallel Implementation of Interval Matrix Multiplication
# pag. 4
# please check the values of u and η
function MMul5(A::BallMatrix{T}, B::BallMatrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)
    ρA = sign.(mA) .* min.(abs.(mA), rA)
    ρB = sign.(mB) .* min.(abs.(mB), rB)

    mC = mA * mB + ρA * ρB
    Γ = abs.(mA) * abs.(mB) + abs.(ρA) * abs.(ρB)
    rC = setrounding(T, RoundUp) do
        γ = (k + 1) * eps.(Γ) .+ 0.5 * η / ϵp
        rC = (abs.(mA) + rA) * (abs.(mB) + rB) - Γ + 2γ
    end
    BallMatrix(mC, rC)
end
