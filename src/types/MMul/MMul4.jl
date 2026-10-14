
# The product of two floating-point matrices with the rounding mode downward and upward:
# `fl▽(AB)` and `fl△(AB)` in the notation of Rump's papers, entrywise below and above `AB`.
function _product_down_up(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: AbstractFloat}
    _blas_fallback(T) && return _product_down_up_prodk(A, B)
    up = setrounding(T, RoundUp) do
        A * B
    end
    down = setrounding(T, RoundDown) do
        A * B
    end
    return down, up
end

# `fl△(AB)` alone: an entrywise upper bound of the product, and of `AB` itself when `A` and `B`
# are non-negative. The products that are still formed under a rounding mode go through this
# function and `_product_down_up`, so that the dependence on the BLAS is in one place.
function _product_up(A::AbstractVecOrMat{T}, B::AbstractVecOrMat{T}) where {T <: AbstractFloat}
    _blas_fallback(T) && return _product_down_up_prodk(A, B)[2]
    return setrounding(T, RoundUp) do
        A * B
    end
end

# Where the BLAS failed the rounding test at load time, even on one thread
# (`NumericalTest.BLAS_ROUNDING_OK`), the three functions here do not use the rounding mode: the
# product is enclosed by `_accurate_product_sum`, the splitting scheme of Lange and Rump with the
# Dot2 accumulation where it declines, whose statements are in rounding to nearest and for a BLAS
# product that only adds and multiplies, and the ends of the enclosure are returned.
_blas_fallback(::Type{T}) where {T} =
    T <: Union{Float32, Float64} && !NumericalTest.BLAS_ROUNDING_OK[]

function _product_down_up_prodk(A::AbstractVecOrMat{T}, B::AbstractVecOrMat{T}) where {T}
    size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("product of sizes $(size(A)) and $(size(B))"))
    Am = A isa AbstractVector ? reshape(A, :, 1) : A
    Bm = B isa AbstractVector ? reshape(B, :, 1) : B
    C = _accurate_product_sum(((Am, Bm, 1),))
    down, up = sub_down.(mid(C), rad(C)), add_up.(mid(C), rad(C))
    return B isa AbstractVector ? (vec(down), vec(up)) : (down, up)
end

# `fl▽(AB)` alone.
function _product_down(A::AbstractVecOrMat{T}, B::AbstractVecOrMat{T}) where {T <: AbstractFloat}
    _blas_fallback(T) && return _product_down_up_prodk(A, B)[1]
    return setrounding(T, RoundDown) do
        A * B
    end
end

# Sums, differences and multiples of matrices rounded upward or downward. A product above is an
# OpenBLAS call and follows the rounding mode by the build of OpenBLAS that the package loads; a
# sum, a difference, a multiple by a scalar are Julia's own loops, and that they follow the mode is
# checked when the package is loaded (`NumericalTest.check_array_rounding`). Where that check
# failed, for `Float32` and `Float64`, they are formed entry by entry with the directed operations
# of RoundingEmulator. Every sum that the product routines form under a rounding mode goes through
# these functions.
_entrywise_rounding(::Type{T}) where {T} =
    T <: Union{Float32, Float64} && !NumericalTest.ARRAY_ROUNDING_OK[]

function _add_up(X::AbstractVecOrMat{T}, Y::Union{T, AbstractVecOrMat{T}}) where {T <: AbstractFloat}
    _entrywise_rounding(T) && return add_up.(X, Y)
    return setrounding(T, RoundUp) do
        X .+ Y
    end
end
function _add_down(X::AbstractVecOrMat{T}, Y::Union{T, AbstractVecOrMat{T}}) where {T <: AbstractFloat}
    _entrywise_rounding(T) && return add_down.(X, Y)
    return setrounding(T, RoundDown) do
        X .+ Y
    end
end
function _sub_up(X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) where {T <: AbstractFloat}
    _entrywise_rounding(T) && return sub_up.(X, Y)
    return setrounding(T, RoundUp) do
        X .- Y
    end
end
function _sub_down(X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) where {T <: AbstractFloat}
    _entrywise_rounding(T) && return sub_down.(X, Y)
    return setrounding(T, RoundDown) do
        X .- Y
    end
end
# `fl△(αX)` for a scalar `α ≥ 0` and `X ≥ 0`
function _scale_up(α::T, X::AbstractVecOrMat{T}) where {T <: AbstractFloat}
    _entrywise_rounding(T) && return mul_up.(α, X)
    return setrounding(T, RoundUp) do
        α .* X
    end
end

# The ball with the bounds `L ≤ U`: centre `M = fl△((L + U)/2)` and radius `fl△(M − L)`. The
# centre is rounded upward, so `U − M ≤ M − L` and the radius covers both ends.
function _centre_radius_up(L::AbstractVecOrMat{T}, U::AbstractVecOrMat{T}) where {T <: AbstractFloat}
    if _entrywise_rounding(T)
        Me = div_up.(add_up.(L, U), T(2))
        return Me, sub_up.(Me, L)
    end
    # a name of its own inside the closure: one shared with the branch above is boxed
    return setrounding(T, RoundUp) do
        Mr = (L .+ U) ./ 2
        Mr, Mr .- L
    end
end

# `fl△(X + AB)`: an entrywise upper bound of `X + AB` when `A` and `B` are non-negative, the sum
# under the rounding mode as well as the product.
function _sum_product_up(X::AbstractVecOrMat{T}, A::AbstractVecOrMat{T},
        B::AbstractVecOrMat{T}) where {T <: AbstractFloat}
    # the sum is Julia's own loop: entry by entry where the check at load time failed
    (_entrywise_rounding(T) || _blas_fallback(T)) && return add_up.(X, _product_up(A, B))
    return setrounding(T, RoundUp) do
        X + A * B
    end
end

# Rump, Numer. Algorithms 61 (2012), MMul4: with the radius `R` of the product,
# `[fl▽(mA mB) − R, fl△(mA mB) + R]` rounded outward, then the centre and the radius of that
# interval. The four methods differ only in `R`.
function _mmul4(mA::AbstractMatrix{T}, mB::AbstractMatrix{T}, R) where {T <: AbstractFloat}
    C1, C2 = _product_down_up(mA, mB)
    if R !== nothing
        C1, C2 = _sub_down(C1, R), _add_up(C2, R)
    end
    mC, rC = _centre_radius_up(C1, C2)
    return BallMatrix(mC, rC)
end

function MMul4(A::BallMatrix{T, T}, B::BallMatrix{T, T}) where {T <: AbstractFloat}
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)
    abs_mA = abs_preserving_structure(mA)
    abs_mB = abs_preserving_structure(mB)
    # R = fl△(|mA| rB + rA (|mB| + rB))
    R = _sum_product_up(_product_up(abs_mA, rB), rA, _add_up(abs_mB, rB))
    return _mmul4(mA, mB, R)
end

function MMul4(A::AbstractMatrix{T}, B::BallMatrix{T, T}) where {T <: AbstractFloat}
    mB, rB = mid(B), rad(B)
    return _mmul4(A, mB, _product_up(abs_preserving_structure(A), rB))
end

function MMul4(A::BallMatrix{T, T}, B::AbstractMatrix{T}) where {T <: AbstractFloat}
    mA, rA = mid(A), rad(A)
    return _mmul4(mA, B, _product_up(rA, abs_preserving_structure(B)))
end

MMul4(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: AbstractFloat} = _mmul4(A, B, nothing)
