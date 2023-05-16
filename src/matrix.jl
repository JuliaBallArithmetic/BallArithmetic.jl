struct BallMatrix{T<:AbstractFloat,NT<:Union{T,Complex{T}},BT<:Ball{T,NT},CM<:AbstractMatrix{NT},RM<:AbstractMatrix{T}} <: AbstractMatrix{BT}
    c::CM
    r::RM
    function BallMatrix(c::AbstractMatrix{T}, r::AbstractMatrix{T}) where {T<:AbstractFloat}
        new{T,T,Ball{T,T},typeof(c),typeof(r)}(c, r)
    end
    function BallMatrix(c::AbstractMatrix{Complex{T}}, r::AbstractMatrix{T}) where {T<:AbstractFloat}
        new{T,Complex{T},Ball{T,Complex{T}},typeof(c),typeof(r)}(c, r)
    end
end

BallMatrix(M::AbstractMatrix{<:Ball}) = BallMatrix(mid.(M), rad.(M))

mid(A::BallMatrix) = map(mid, A)
rad(A::BallMatrix) = map(rad, A)

# Array interface
Base.eltype(::BallMatrix{T,NT,BT}) where {T,NT,BT} = BT
Base.IndexStyle(::Type{<:BallMatrix}) = IndexLinear()
Base.size(M::BallMatrix, i...) = size(M.c, i...)
Base.getindex(M::BallMatrix, i::Int) = Ball(getindex(M.c, i), getindex(M.r, i))
function Base.setindex!(M::BallMatrix, x, inds...)
  setindex!(M.c, mid(x), inds...)
  setindex!(M.r, rad(x), inds...) 
end
Base.copy(M::BallMatrix) = BallMatrix(copy(M.c), copy(M.r))

function Base.zeros(::Type{B}, dims::NTuple{N, Integer}) where {B<:Ball,N}
    BallMatrix(zeros(midtype(B), dims), zeros(radtype(B), dims))
end

function Base.ones(::Type{B}, dims::NTuple{N, Integer}) where {B<:Ball,N}
    BallMatrix(ones(midtype(B), dims), zeros(radtype(B), dims))
end

# LinearAlgebra functions
function LinearAlgebra.adjoint(M::BallMatrix)
    return BallMatrix(mid(M)', rad(M)')
end


# Operations
for op in (:+, :-)
    @eval function Base.$op(A::BallMatrix{T}, B::BallMatrix{T}) where {T<:AbstractFloat}
        mA, rA = mid(A), rad(A)
        mB, rB = mid(B), rad(B)
    
        C = $op(mA, mB) 
        R = setrounding(T, RoundUp) do
            R = (ϵp * abs.(C) + rA) + rB
        end
        BallMatrix(C, R)
    end
end

# TODO: maybe it is worth to define a convert function
for op in (:+, :-)
    @eval function Base.$op(A::BallMatrix{T}, B::Matrix{T}) where {T<:AbstractFloat}
        rB = zeros(size(B))
        $op(A, BallMatrix(B, rB)) 
    end
    # + and - are commutative
    @eval function Base.$op(B::Matrix{T}, A::BallMatrix{T}) where {T<:AbstractFloat}
        rB = zeros(size(B))
        $op(A, BallMatrix(B, rB)) 
    end
end

function Base.:*(A::BallMatrix{T}, B::BallMatrix{T}) where {T<:AbstractFloat}
    # mA, rA = mid(A), rad(A)
    # mB, rB = mid(B), rad(B)
    # C = mA * mB
    # R = setrounding(T, RoundUp) do
    #     R = abs.(mA) * rB + rA * (abs.(mB) + rB)
    # end
    # BallMatrix(C, R)
    MMul3(A, B)
end

function Base.:*(A::BallMatrix{T}, B::Matrix{T}) where {T<:AbstractFloat}
    rB = zeros(size(B))
    return A*BallMatrix(B, rB)
end

function Base.:*(B::Matrix{T}, A::BallMatrix{T}) where {T<:AbstractFloat}
    rB = zeros(size(B))
    return BallMatrix(B, rB)*A
end

# TODO: Should we implement this?
# From Theveny https://theses.hal.science/tel-01126973/en
function MMul2(A::BallMatrix{T}, B::BallMatrix{T}) where {T<:AbstractFloat}
    @warn "Not Implemented"
end

# As in Revol-Theveny
# Parallel Implementation of Interval Matrix Multiplication
# pag. 4
# please check the values of u and η
function MMul3(A::BallMatrix{T}, B::BallMatrix{T}) where {T<:AbstractFloat}
    m, k = size(A)
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)
    mC = mA * mB
    rC = setrounding(T, RoundUp) do
        rprimeB = ((k+2)*ϵp*abs.(mB)+rB)
        rC = abs.(mA) * rprimeB + rA * (abs.(mB) + rB).+η/ϵp
    end
    BallMatrix(mC, rC)
end

# As in Revol-Theveny
# Parallel Implementation of Interval Matrix Multiplication
# pag. 4
# please check the values of u and η
function MMul5(A::BallMatrix{T}, B::BallMatrix{T}) where {T<:AbstractFloat}
    m, k = size(A)
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)
    ρA = sign.(mA) .* min.(abs.(mA), rA)
    ρB = sign.(mB) .* min.(abs.(mB), rB)
    
    mC = mA * mB + ρA*ρB
    Γ = abs.(mA)*abs.(mB)+abs.(ρA)*abs.(ρB)
    rC = setrounding(T, RoundUp) do
        γ = (k + 1) * eps.(Γ) .+ 0.5 * η / ϵp
        rC = (abs.(mA)+rA) * (abs.(mB) + rB)-Γ+2γ
    end
    BallMatrix(mC, rC)
end
