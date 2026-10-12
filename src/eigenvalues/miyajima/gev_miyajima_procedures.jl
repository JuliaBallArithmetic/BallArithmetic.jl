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
    miyajima_algorithm_1_procedure_1(A::BallMatrix, B::BallMatrix)

Implement Procedure 1 of Algorithm 1 from Miyajima's generalized eigenvalue
verification method. Computes:
1. Approximate generalized eigenvectors X from mid(A), mid(B)
2. Ball enclosure Z for B * X using Miyajima's rigorous matrix multiplication
3. Left action Y = inv(mid(Z))
4. Bound on coupling defect R₂ = ‖Y * Z - I‖∞

Uses Miyajima's `_cprod` (complex product with directed rounding) and `_ccr`
(collapse rectangular bounds to ball form) for maximum rigor, particularly
important for BigFloat computations.

Returns a tuple (X, Y, Z, bdd_coupling) where:
- X: Approximate right eigenvectors (midpoint computation)
- Y: Left action matrix (inverse of mid(B*X))
- Z: Ball enclosure of B * X (computed via Miyajima procedures)
- eigenvalues: Approximate eigenvalues
- coupling_bound: Rigorous upper bound on ‖Y * Z - I‖∞

# References
* Miyajima (2010): Fast verified matrix multiplication
* Miyajima (2012): Numerical enclosure for generalized eigenvalue problems
"""
function miyajima_algorithm_1_procedure_1(A::BallMatrix{T, NT},
                                           B::BallMatrix{T, NT}) where {T, NT}
    # Compute approximate generalized eigenvalues/vectors from midpoints
    gev = eigen(mid(A), mid(B))
    Xmid = gev.vectors

    # Compute rigorous enclosure for B * X using Miyajima's procedures
    # This provides maximum rigor with directed rounding (essential for BigFloat)
    Z = _rigorous_matrix_product(mid(B), Xmid, T, NT)

    # Compute left action Y = inv(mid(B * X))
    Y = inv(mid(Z))

    # Compute rigorous bound on coupling defect ‖Y * Z - I‖∞
    # This uses Higham-style γ factors for a priori error bounds
    bdd_coupling = bdd_R2(Y, Z)

    return (X = Xmid, Y = Y, Z = Z, eigenvalues = gev.values,
            coupling_bound = bdd_coupling)
end

"""
    _rigorous_matrix_product(B::AbstractMatrix, X::AbstractMatrix, ::Type{T}, ::Type{NT})

Compute rigorous ball matrix enclosure of B * X using Miyajima's procedures.

For complex matrices: Uses `_cprod` (Algorithm from Oishi-Rump) followed by
`_ccr` (Algorithm 5 from Miyajima2010) to collapse rectangular bounds to
ball form with directed rounding.

For real matrices: Promotes to complex and uses the same procedure for consistency.

This ensures maximum rigor for all precision types (Float64, BigFloat, etc.).
"""
function _rigorous_matrix_product(B::AbstractMatrix{BT}, X::AbstractMatrix{XT},
                                    ::Type{T}, ::Type{NT}) where {BT, XT, T, NT}
    # Promote to complex to use Miyajima's _cprod procedure
    # This works for both real and complex inputs
    CT = Complex{T}
    B_complex = CT.(B)
    X_complex = CT.(X)

    # Use Miyajima's rigorous complex product with directed rounding
    # Returns rectangular bounds: Hrl ≤ Re(B*X) ≤ Hru, Hil ≤ Im(B*X) ≤ Hiu
    Hrl, Hru, Hil, Hiu, _ = _cprod(B_complex, X_complex)

    # Collapse rectangular bounds to ball form (midpoint + radius)
    # Algorithm 5 from Miyajima2010
    Z_ball, _ = _ccr(Hrl, Hru, Hil, Hiu)

    return Z_ball
end