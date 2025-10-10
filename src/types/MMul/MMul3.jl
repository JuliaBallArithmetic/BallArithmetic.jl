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
        abs_mB = abs_preserving_structure(mB)
        abs_mA = abs_preserving_structure(mA)
        rprimeB = ((k + 2) * ϵp * abs_mB + rB)
        rC = abs_mA * rprimeB + rA * (abs_mB + rB) .+ η / ϵp
        return rC
    end
    BallMatrix(mC, rC)
end

function MMul3(A::BallMatrix{T}, B::AbstractMatrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    mA, rA = mid(A), rad(A)
    mC = mA * B
    rC = setrounding(T, RoundUp) do
        abs_B = abs_preserving_structure(B)
        abs_mA = abs_preserving_structure(mA)
        rprimeB = ((k + 2) * ϵp * abs_B)
        rC = abs_mA * rprimeB + rA * abs_B .+ η / ϵp
        return rC
    end
    BallMatrix(mC, rC)
end

function MMul3(A::AbstractMatrix{T}, B::BallMatrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    mB, rB = mid(B), rad(B)
    mC = A * mB
    rC = setrounding(T, RoundUp) do
        abs_mB = abs_preserving_structure(mB)
        abs_A = abs_preserving_structure(A)
        rprimeB = ((k + 2) * ϵp * abs_mB + rB)
        rC = abs_A * rprimeB .+ η / ϵp
        return rC
    end
    BallMatrix(mC, rC)
end

function MMul3(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    mC = A * B
    rC = setrounding(T, RoundUp) do
        abs_B = abs_preserving_structure(B)
        abs_A = abs_preserving_structure(A)
        rprimeB = ((k + 2) * ϵp * abs_B)
        rC = abs_A * rprimeB .+ η / ϵp
        return rC
    end
    BallMatrix(mC, rC)
end
