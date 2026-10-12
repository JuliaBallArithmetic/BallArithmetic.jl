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

Compute rigorous bound for the residual R₂ = ‖Y * Z - I‖∞ where Y is a
floating-point matrix and Z is a ball matrix. This follows the error analysis
from Miyajima's algorithm for computing bounds on the coupling defect in
generalized eigenvalue problems.

The bound accounts for:
- The midpoint residual ‖Y * mid(Z) - I‖∞
- The radius contribution from interval arithmetic
- Floating-point rounding errors with Higham-style γ factors
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

    # Bound 2: Contribution from radius in ball arithmetic (real part)
    e = ones(WT, n)
    vr = abs.(real.(Y)) * (Zr * e)
    vi = abs.(imag.(Y)) * (Zr * e)
    bd2 = _up_bound_Linf_norm(vr + vi)

    # Bound 3: Mixed contribution from midpoint and radius
    vc = abs.(Y) * (abs.(Zc) * e)
    bd3 = _up_bound_Linf_norm(vc)

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