# Revol-Théveny midpoint-radius interval matrix multiplication
# Reference: N. Revol & P. Théveny, "Parallel Implementation of Interval Matrix Multiplication",
#            Reliable Computing 19(1), pp. 91-106, 2013. https://hal.science/hal-00801890
#
# The midpoint is intentionally computed with RoundNearest (default) for efficiency,
# while the radius uses RoundUp. The error terms (k+1)*eps(Γ) in the radius formula
# rigorously account for floating-point errors in the midpoint computation.
function MMul5(A::BallMatrix{T}, B::BallMatrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)
    ρA = sign.(mA) .* min.(abs.(mA), rA)
    ρB = sign.(mB) .* min.(abs.(mB), rB)

    mC = mA * mB + ρA * ρB
    abs_mA = abs_preserving_structure(mA)
    abs_mB = abs_preserving_structure(mB)
    abs_rhoA = abs_preserving_structure(ρA)
    abs_rhoB = abs_preserving_structure(ρB)

    Γ = abs_mA * abs_mB + abs_rhoA * abs_rhoB
    ϵ = machine_epsilon(T)
    η_val = subnormal_min(T)
    rC = setrounding(T, RoundUp) do
        γ = (k + 1) * eps.(Γ) .+ 0.5 * η_val / ϵ
        rC = (abs_mA + rA) * (abs_mB + rB) - Γ + 2γ
        return rC
    end
    BallMatrix(mC, rC)
end
