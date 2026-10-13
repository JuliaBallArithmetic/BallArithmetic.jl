
function MMul4(A::BallMatrix{T, T}, B::BallMatrix{T, T}) where {T <: AbstractFloat}
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)

    C2, rC = setrounding(T, RoundUp) do
        abs_mA = abs_preserving_structure(mA)
        abs_mB = abs_preserving_structure(mB)
        rC = abs_mA * rB + rA * (abs_mB + rB)
        C2 = mA * mB + rC
        return C2, rC
    end

    C1 = setrounding(T, RoundDown) do
        C1 = mA * mB - rC
        return C1
    end

    mC, rC = setrounding(T, RoundUp) do
        mC = (C1 + C2) / 2
        rC = mC - C1
        return mC, rC
    end
    return BallMatrix(mC, rC)
end

function MMul4(A::AbstractMatrix{T}, B::BallMatrix{T, T}) where {T <: AbstractFloat}
    mA = mid(A)
    mB, rB = mid(B), rad(B)

    C2, rC = setrounding(T, RoundUp) do
        abs_mA = abs_preserving_structure(mA)
        rC = abs_mA * rB
        C2 = mA * mB + rC
        return C2, rC
    end

    C1 = setrounding(T, RoundDown) do
        C1 = mA * mB - rC
        return C1
    end

    mC, rC = setrounding(T, RoundUp) do
        mC = (C1 + C2) / 2
        rC = mC - C1
        return mC, rC
    end
    BallMatrix(mC, rC)
end

function MMul4(A::BallMatrix{T, T}, B::AbstractMatrix{T}) where {T <: AbstractFloat}
    mA, rA = mid(A), rad(A)
    mB = mid(B)

    C2, rC = setrounding(T, RoundUp) do
        abs_mB = abs_preserving_structure(mB)
        rC = rA * abs_mB
        C2 = mA * mB + rC
        return C2, rC
    end

    C1 = setrounding(T, RoundDown) do
        C1 = mA * mB - rC
        return C1
    end

    mC, rC = setrounding(T, RoundUp) do
        mC = (C1 + C2) / 2
        rC = mC - C1
        return mC, rC
    end
    return BallMatrix(mC, rC)
end

# The product of two floating-point matrices with the rounding mode downward and upward:
# `fl▽(AB)` and `fl△(AB)` in the notation of Rump's papers, entrywise below and above `AB`.
function _product_down_up(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: AbstractFloat}
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
_product_up(A::AbstractVecOrMat{T}, B::AbstractVecOrMat{T}) where {T <: AbstractFloat} =
    setrounding(T, RoundUp) do
        A * B
    end

# `fl△(X + AB)`: an entrywise upper bound of `X + AB` when `A` and `B` are non-negative, the sum
# under the rounding mode as well as the product.
function _sum_product_up(X::AbstractVecOrMat{T}, A::AbstractVecOrMat{T},
        B::AbstractVecOrMat{T}) where {T <: AbstractFloat}
    # the sum is Julia's own loop: entry by entry where the check at load time failed
    NumericalTest.ARRAY_ROUNDING_OK[] || return add_up.(X, _product_up(A, B))
    return setrounding(T, RoundUp) do
        X + A * B
    end
end

function MMul4(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: AbstractFloat}
    C1, C2 = _product_down_up(mid(A), mid(B))

    mC, rC = setrounding(T, RoundUp) do
        mC = (C1 + C2) / 2
        rC = mC - C1
        return mC, rC
    end
    return BallMatrix(mC, rC)
end
