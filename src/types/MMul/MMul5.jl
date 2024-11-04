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
        return rC
    end
    BallMatrix(mC, rC)
end
