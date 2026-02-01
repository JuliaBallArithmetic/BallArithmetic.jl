# Revol-Théveny midpoint-radius interval matrix multiplication
# Reference: N. Revol & P. Théveny, "Parallel Implementation of Interval Matrix Multiplication",
#            Reliable Computing 19(1), pp. 91-106, 2013. https://hal.science/hal-00801890
#
# The midpoint is intentionally computed with RoundNearest (default) for efficiency,
# while the radius uses RoundUp. The error terms (k+2)*ε in the radius formula
# rigorously account for floating-point errors in the midpoint computation.

function MMul3(A::BallMatrix{T}, B::BallMatrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)
    mC = mA * mB
    ϵ = machine_epsilon(T)
    η_val = subnormal_min(T)
    rC = setrounding(T, RoundUp) do
        abs_mB = abs_preserving_structure(mB)
        abs_mA = abs_preserving_structure(mA)
        rprimeB = ((k + 2) * ϵ * abs_mB + rB)
        rC = abs_mA * rprimeB + rA * (abs_mB + rB) .+ η_val / ϵ
        return rC
    end
    BallMatrix(mC, rC)
end

function MMul3(A::BallMatrix{T}, B::AbstractMatrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    mA, rA = mid(A), rad(A)
    mC = mA * B
    ϵ = machine_epsilon(T)
    η_val = subnormal_min(T)
    rC = setrounding(T, RoundUp) do
        abs_B = abs_preserving_structure(B)
        abs_mA = abs_preserving_structure(mA)
        rprimeB = ((k + 2) * ϵ * abs_B)
        rC = abs_mA * rprimeB + rA * abs_B .+ η_val / ϵ
        return rC
    end
    BallMatrix(mC, rC)
end

function MMul3(A::AbstractMatrix{T}, B::BallMatrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    mB, rB = mid(B), rad(B)
    mC = A * mB
    ϵ = machine_epsilon(T)
    η_val = subnormal_min(T)
    rC = setrounding(T, RoundUp) do
        abs_mB = abs_preserving_structure(mB)
        abs_A = abs_preserving_structure(A)
        rprimeB = ((k + 2) * ϵ * abs_mB + rB)
        rC = abs_A * rprimeB .+ η_val / ϵ
        return rC
    end
    BallMatrix(mC, rC)
end

function MMul3(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    mC = A * B
    ϵ = machine_epsilon(T)
    η_val = subnormal_min(T)
    rC = setrounding(T, RoundUp) do
        abs_B = abs_preserving_structure(B)
        abs_A = abs_preserving_structure(A)
        rprimeB = ((k + 2) * ϵ * abs_B)
        rC = abs_A * rprimeB .+ η_val / ϵ
        return rC
    end
    BallMatrix(mC, rC)
end
