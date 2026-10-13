# Midpoint-radius interval matrix multiplication in five point matrix products.
#
#   S. M. Rump, Fast interval matrix multiplication, Numer. Algorithms 61 (2012) 1-34,
#   doi 10.1007/s11075-011-9524-z, Algorithm 4.9 (IImul5), without the conversions from and to
#   the infimum-supremum representation; it is Algorithm 2 (MMMul5) of
#   N. Revol and P. Théveny, Parallel implementation of interval matrix multiplication,
#   Reliable Computing 19(1) (2013) 91-106, https://hal.science/hal-00801890.
#
# With eps = 2^-53 the relative rounding error unit, realmin the smallest positive normalised
# number and n the inner dimension, the algorithm is
#
#     ρA = sign(mA) .* min(|mA|, rA),  ρB likewise
#     mC = fl(mA mB + ρA ρB)                      rounding to nearest
#     μ  = fl(|mA||mB| + |ρA||ρB|)                rounding to nearest
#     γ  = fl△((2n + 2) eps ufp(μ) + realmin)
#     rC = fl△((|mA| + rA)(|mB| + rB) − μ + 2γ)
#
# and is correct, also in the presence of underflow, provided 2(2n + 2) eps ≤ 1 and mC and μ are
# computed in the same order (Theorem 2.1 of Rump's paper). Below `eps.(Γ)` is the unit in the last
# place, 2 eps ufp(Γ), so `(k + 1) * eps.(Γ)` is the paper's (2n + 2) eps ufp(μ), and
# `η_val / ϵ` is realmin. The papers are for real matrices, and so is the method: a complex
# product goes through `*`, which forms (A + iB)(C + iD) = AC − BD + i(AD + BC) from four real
# products.
function MMul5(A::BallMatrix{T, T}, B::BallMatrix{T, T}) where {T <: AbstractFloat}
    m, k = size(A)
    2 * (2k + 2) * unit_roundoff(T) <= 1 ||
        throw(ArgumentError("MMul5: inner dimension $k too large, 2(2k + 2)eps ≤ 1 is required"))
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
    # all upward: γ = (k + 1) eps.(Γ) + η/ϵ, rC = (|mA| + rA)(|mB| + rB) − Γ + 2γ
    γ = _add_up(_scale_up(T(k + 1), eps.(Γ)), η_val / ϵ)
    rC = _add_up(_sub_up(_product_up(_add_up(abs_mA, rA), _add_up(abs_mB, rB)), Γ), 2 * γ)
    BallMatrix(mC, rC)
end
