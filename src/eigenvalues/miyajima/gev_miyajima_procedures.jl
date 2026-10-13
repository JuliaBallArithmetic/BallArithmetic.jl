"""
    _up_bound_Linf_opnorm(A::AbstractMatrix{T}) where T

An upper bound of the ∞-norm of the matrix `A`, of type `float(real(T))`: the largest row sum of
the moduli, each modulus and each sum rounded up.
"""
_up_bound_Linf_opnorm(A::AbstractMatrix{T}) where {T} =
    _colrow_sums_up(abs_up.(float.(A)))[2]

"""
    _up_bound_Linf_norm(v::AbstractVector{T}) where T

An upper bound of the ∞-norm of the vector `v`, of type `float(real(T))`: the largest modulus,
rounded up.
"""
_up_bound_Linf_norm(v::AbstractVector{T}) where {T} =
    maximum(abs_up, float.(v); init = zero(float(real(T))))

"""
    bdd_R2(Y::AbstractMatrix, Z::BallMatrix)

An upper bound of `‖Y Z̃ − I‖_∞` over the matrices `Z̃` of the ball `Z = ⟨Z_c, Z_r⟩`, for a
floating-point matrix `Y`: inequality (8) of Corollary 1 of

    S. Miyajima, Fast enclosure for all eigenvalues in generalized eigenvalue problems,
    J. Comput. Appl. Math. 233 (2010) 2994-3004, doi 10.1016/j.cam.2009.11.048,

    ‖R₂‖_∞ ≤ ‖fl(Y Z_c − I)‖_∞ + ‖(|Y_r| + |Y_i|) Z_r e‖_∞ + γ′ₙ₋₁ ‖|Y| |Z_c| e‖_∞
              + u (‖fl(Y Z_c)‖_∞ + 1) + 4 n² (1 + γₙ₋₁) u̲,

with `Y_r`, `Y_i` the real and imaginary parts of `Y`, `e` the vector of ones, `u` the unit
roundoff, `u̲` the underflow unit, `γₙ = n u/(1 − n u)` and `γ′ₙ₋₁ = √5 u + γₙ₋₁ (1 + √5 u)`. The
one product `Y Z_c` is formed in round-to-nearest, as the corollary asks, and its rounding is
what the last three terms pay for; it holds in the presence of underflow.

Differences from the paper, all on the side of a larger bound: `eps` is used for `u`, which is
twice the unit roundoff, and `γₙ` for `γₙ₋₁`. The two terms that the corollary states as exact
quantities, `‖(|Y_r| + |Y_i|) Z_r e‖_∞` and `‖|Y| |Z_c| e‖_∞`, are evaluated with every operation
rounded up.
"""
function bdd_R2(Y::AbstractMatrix{T}, Z::BallMatrix{RT, NT}) where {T, RT, NT}
    n = size(Y, 2)

    # Extract midpoint and radius from ball matrix Z
    Zc = mid(Z)
    Zr = rad(Z)

    # Working type for rigorous bounds
    WT = promote_type(float(real(T)), RT)

    # Compute Y * mid(Z) with nearest rounding
    YZc = Y * Zc

    # Bound 1: ‖Y * mid(Z) - I‖∞
    bd1 = _up_bound_Linf_opnorm(YZc - I)

    # an upper bound of ‖M w‖_∞ for a non-negative matrix M and a non-negative vector w
    matvec_norm_up(M, w) = maximum(
        foldl((acc, j) -> add_up(acc, mul_up(WT(M[i, j]), WT(w[j]))), axes(M, 2); init = zero(WT))
        for i in axes(M, 1); init = zero(WT))

    # Bound 2: ‖(|Y_r| + |Y_i|) Z_r e‖_∞, the contribution of the radius
    bd2 = matvec_norm_up(add_up.(abs.(real.(Y)), abs.(imag.(Y))), _rowsums_up(Zr))

    # Bound 3: ‖|Y| |Z_c| e‖_∞
    bd3 = matvec_norm_up(abs_up.(Y), _rowsums_up(abs_up.(Zc)))

    # Bound 4: Norm of YZc for error analysis
    bd4 = _up_bound_Linf_opnorm(YZc)

    # Final bound combining all error sources with rounding error analysis
    # every operation rounded up, and the denominator 1 − n u rounded down
    u = eps(WT)
    ui = nextfloat(zero(WT))
    nu = mul_up(WT(n), u)
    gamma = div_up(nu, sub_down(one(WT), nu))
    sqrt5u = mul_up(sqrt_up(WT(5)), u)
    gammaprime = add_up(sqrt5u, mul_up(gamma, add_up(one(WT), sqrt5u)))
    underflow = mul_up(mul_up(mul_up(WT(4), mul_up(WT(n), WT(n))), add_up(one(WT), gamma)), ui)
    bdd = add_up(add_up(add_up(add_up(bd1, bd2), mul_up(gammaprime, bd3)),
            mul_up(u, add_up(bd4, one(WT)))), underflow)
    return bdd
end

"""
    miyajima2010_algorithm1_steps1to3(A::BallMatrix, B::BallMatrix)

Steps 1 to 3 of Algorithm 1 of Miyajima (2010) for the pencil `A x = λ B x`, with the approximate
eigenvectors `X̃` and eigenvalues taken from `eigen(mid(A), mid(B))`:

- Step 1: `Z = ⟨Z_c, Z_r⟩` with `B X̃ ∈ Z`, by Algorithm 2 (`cprod`, [`_cprod`](@ref)) for the
  bounds of the real and of the imaginary part and Algorithm 3 (`ccr`, [`_ccr`](@ref)) for the
  ball;
- Step 2: `Y ≈ Z_c⁻¹`, in floating point;
- Step 3: `ε₂ ≥ ‖R₂‖_∞`, `R₂ = Y B X̃ − I`, by inequality (8) of Corollary 1 ([`bdd_R2`](@ref)).

Returns `(X, Y, Z, eigenvalues, coupling_bound)` with `coupling_bound = ε₂`. Step 4 of the
algorithm is the test `ε₂ < 1`, which proves that `B`, `X̃` and `Y` are nonsingular; Steps 5 and 6,
the bound `ε₁ ≥ ‖R₁‖_∞` by Procedure 1 or 2 of the paper and `ε = ε₁/(1 − ε₂)`, are not done
here. (This function was named `miyajima_algorithm_1_procedure_1`; Procedure 1 of the paper is
the enclosure of `R₁`, which it does not compute.)

The paper states the algorithm for point matrices. Here `B` may have a positive radius: the bounds
of Step 1 are then those of `⟨mid(B), rad(B)⟩ X̃`, by the kernel `ccrprod′` of Procedure 2
([`_ccrprod_prime`](@ref)), so that `B̃ X̃ ∈ Z` and `‖Y B̃ X̃ − I‖_∞ ≤ ε₂` for every `B̃` of the ball,
which is our use of the paper's kernel and not a statement of the paper. `A` enters through the
approximate eigenvectors only.

# Reference

S. Miyajima, *Fast enclosure for all eigenvalues in generalized eigenvalue problems*,
J. Comput. Appl. Math. **233**(11) (2010) 2994-3004, doi 10.1016/j.cam.2009.11.048.
"""
function miyajima2010_algorithm1_steps1to3(A::BallMatrix{T, NT},
        B::BallMatrix{T, NT}) where {T, NT}
    gev = eigen(mid(A), mid(B))
    CT = Complex{T}
    Xmid = CT.(gev.vectors)

    # Step 1: bounds of the real and the imaginary part of ⟨mid(B), rad(B)⟩ X̃, then the ball
    Zrl, Zru, Zil, Ziu, _ = _ccrprod_prime(CT.(mid(B)), rad(B), Xmid)
    Z, _ = _ccr(Zrl, Zru, Zil, Ziu)

    # Step 2
    Y = inv(mid(Z))

    # Step 3
    bdd_coupling = bdd_R2(Y, Z)

    return (X = Xmid, Y = Y, Z = Z, eigenvalues = gev.values,
        coupling_bound = bdd_coupling)
end
