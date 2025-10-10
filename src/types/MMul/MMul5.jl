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
    abs_mA = abs_preserving_structure(mA)
    abs_mB = abs_preserving_structure(mB)
    abs_rhoA = abs_preserving_structure(ρA)
    abs_rhoB = abs_preserving_structure(ρB)

    Γ = abs_mA * abs_mB + abs_rhoA * abs_rhoB
    rC = setrounding(T, RoundUp) do
        γ = (k + 1) * eps.(Γ) .+ 0.5 * η / ϵp
        rC = (abs_mA + rA) * (abs_mB + rB) - Γ + 2γ
        return rC
    end
    BallMatrix(mC, rC)
end
