
function MMul4(A::BallMatrix{T, T}, B::BallMatrix{T, T}) where {T <: AbstractFloat}
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)

    C2, rC = setrounding(T, RoundUp) do
        rC = abs.(mA) * rB + rA * (abs.(mB) + rB)
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

function MMul4(A::Matrix{T}, B::BallMatrix{T, T}) where {T <: AbstractFloat}
    mA = mid(A)
    mB, rB = mid(B), rad(B)

    C2, rC = setrounding(T, RoundUp) do
        rC = abs.(mA) * rB
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

function MMul4(A::BallMatrix{T, T}, B::Matrix{T}) where {T <: AbstractFloat}
    mA, rA = mid(A), rad(A)
    mB = mid(B)

    C2, rC = setrounding(T, RoundUp) do
        rC = rA * abs.(mB)
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

function MMul4(A::Matrix{T}, B::Matrix{T}) where {T <: AbstractFloat}
    mA = mid(A)
    mB = mid(B)

    C2 = setrounding(T, RoundUp) do
        C2 = mA * mB
        return C2
    end

    C1 = setrounding(T, RoundDown) do
        C1 = mA * mB
        return C1
    end

    mC, rC = setrounding(T, RoundUp) do
        mC = (C1 + C2) / 2
        rC = mC - C1
        return mC, rC
    end
    return BallMatrix(mC, rC)
end
