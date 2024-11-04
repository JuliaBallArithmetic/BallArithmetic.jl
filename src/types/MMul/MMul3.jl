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
