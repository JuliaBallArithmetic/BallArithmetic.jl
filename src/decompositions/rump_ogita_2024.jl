# Verified error bounds for matrix decompositions, after
#
#   S. M. Rump and T. Ogita, Verified Error Bounds for Matrix Decompositions,
#   SIAM J. Matrix Anal. Appl. 45(4) (2024), 2155-2183, doi 10.1137/24M165096X.
#
# Each routine is named for the algorithm or the section of the paper it implements.

"""
    _rumpogita2024_alg3_2(E::BallMatrix) -> (; LE, UE, LinvE, UinvE) or nothing

Algorithm 3.2 (`LU_E`) of Rump and Ogita (2024): inclusions of the factors of the LU
decomposition, without pivoting, of a perturbed identity matrix, and of their inverses.

For `E` of size `m × n`, `p = min(m, n)` and every `Ẽ` of the ball `E`, the matrix `I_{m,n} + Ẽ`
has a unique decomposition `LU` with `L` of size `m × p` unit lower triangular (trapezoidal) and
`U` of size `p × n` upper triangular (trapezoidal), and

    L ∈ I_{m,p} + LE,      U ∈ I_{p,n} + UE,

`LE` strictly lower and `UE` upper triangular. `LinvE` is returned when `L` is square
(`m ≤ n`), with `L⁻¹ ∈ I + LinvE`, and `UinvE` when `U` is square (`m ≥ n`), with
`U⁻¹ ∈ I + UinvE`; otherwise the field is `nothing`. The cost is `O(mn)` operations.

The bounds are (3.1)-(3.7) of the paper. With `Δ_L := sum(|E^{[ℓ]}|, 2) max(|E|) / (1 − ‖E‖_∞)`,
strictly lower part, `|L − I − E^{[ℓ]}| ≤ Δ_L` (3.1); with `G_L = |E^{[ℓ]}| + Δ_L`,
`|L⁻¹ − I + E^{[ℓ]}| ≤ Δ_L + sum(G_L, 2) max(G_L)/(1 − ‖G_L‖_∞)` (3.4); with `B` equal to `|E|`
with its strictly lower part replaced by `Δ_L`, `|U − I − E^{[u]}| ≤ Δ_U := sum(G_L, 2) max(B) /
(1 − ‖G_L‖_∞)`, upper part (3.5); and with `G_U = |E^{[u]}| + Δ_U`,
`|U⁻¹ − I + E^{[u]}| ≤ Δ_U + sum(G_U, 2) max(G_U)/(1 − ‖G_U‖_∞)` (3.7). Here `sum(·, 2)` is the
column of row sums, `max(·)` the row of column maxima, and the products are outer products.

Returns `nothing` when one of the three norms `‖E‖_∞`, `‖G_L‖_∞`, `‖G_U‖_∞` is not below one,
which the paper's listing assumes. Every bound is rounded up.

# Reference

S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 3.1 and Algorithm 3.2.
"""
function _rumpogita2024_alg3_2(E::BallMatrix{T}) where {T}
    m, n = size(E)
    p = min(m, n)
    Em, magE = mid(E), upper_abs(E)
    CT = eltype(Em)
    rowsum(M) = T[sum_up(view(M, i, :)) for i in axes(M, 1)]
    colmax(M) = T[maximum(view(M, :, j); init = zero(T)) for j in axes(M, 2)]
    # sum(M, 2) * max(N) / (1 − ν), keeping the entries (i, j) for which keep(i, j)
    function outer(σ, μ, ν, keep)
        d = sub_down(one(T), ν)
        return T[keep(i, j) ? div_up(mul_up(σ[i], μ[j]), d) : zero(T) for i in 1:m, j in 1:n]
    end
    lower(i, j) = i > j
    upper(i, j) = i <= j
    part(M, keep) = [keep(i, j) ? M[i, j] : zero(eltype(M)) for i in 1:m, j in 1:n]

    νE = maximum(rowsum(magE); init = zero(T))
    νE < 1 || return nothing
    # lines 4-6: L
    DeltaL = outer(rowsum(part(magE, lower)), colmax(magE), νE, lower)
    LEm = part(Em, lower)
    # line 7: GL = mag(LE)
    GL = add_up.(part(magE, lower), DeltaL)
    νL = maximum(rowsum(GL); init = zero(T))
    νL < 1 || return nothing
    # lines 8-10: L⁻¹
    deltaL = add_up.(DeltaL, outer(rowsum(GL), colmax(GL), νL, lower))
    # lines 11-14: U
    B = add_up.(part(magE, upper), DeltaL)
    DeltaU = outer(rowsum(GL), colmax(B), νL, upper)
    UEm = part(Em, upper)
    # line 15: GU = mag(UE)
    GU = add_up.(part(magE, upper), DeltaU)
    νU = maximum(rowsum(GU); init = zero(T))
    νU < 1 || return nothing
    # lines 16-18: U⁻¹
    deltaU = add_up.(DeltaU, outer(rowsum(GU), colmax(GU), νU, upper))

    # the radii of E itself belong to the inclusions: LE = tril(E, −1) + midrad(0, DeltaL)
    Er = rad(E)
    ball(M, R, rows, cols) = BallMatrix(Matrix{CT}(M[rows, cols]), R[rows, cols])
    LE = ball(LEm, add_up.(part(Er, lower), DeltaL), 1:m, 1:p)
    UE = ball(UEm, add_up.(part(Er, upper), DeltaU), 1:p, 1:n)
    LinvE = m <= n ? ball(-LEm, add_up.(part(Er, lower), deltaL), 1:m, 1:m) : nothing
    UinvE = m >= n ? ball(-UEm, add_up.(part(Er, upper), deltaU), 1:n, 1:n) : nothing
    return (; LE, UE, LinvE, UinvE)
end

"""
    _rumpogita2024_sec3_2(A::BallMatrix) -> (; L, U, p) or nothing

Section 3.2 of Rump and Ogita (2024): inclusions of the factors of the LU decomposition of a
square matrix. For every matrix `Ã` of the ball `A`, the row permutation `Ã[p, :]` has a unique
decomposition `LU`, `L` unit lower triangular and `U` upper triangular, with `L` in the ball
matrix `L` and `U` in the ball matrix `U` returned. `p` is the permutation of a floating-point
decomposition with partial pivoting of the midpoint. Returns `nothing` when the verification
fails; nothing is then proved.

# The method

`L̃`, `Ũ` are the floating-point factors of `mid(A)[p, :]`, `X_L ≈ L̃⁻¹` unit lower triangular and
`X_U ≈ Ũ⁻¹` upper triangular (a right inverse, `I/Ũ`). The matrix

    I_E := X_L Ã[p, :] X_U

is a perturbed identity; it is enclosed with both products in two-fold precision, the first one
kept as an unevaluated sum of two terms. Algorithm 3.2 gives its factors, `I_E = L_E U_E`, and by
uniqueness of the decomposition, (3.8) of the paper,

    L = Ã[p, :] X_U U_E⁻¹,        U = U_E X_U⁻¹.

`L` is computed from the two-term product `Ã[p, :] X_U` and the inclusion of `U_E⁻¹`. For
`X_U⁻¹`, which the paper leaves to the implementation, `X_U Ũ = I + F` is enclosed in two-fold
precision, `F` upper triangular, so that `X_U⁻¹ = Ũ(I + F)⁻¹` with `(I + F)⁻¹` enclosed by
Algorithm 3.2 again; this also proves `X_U` nonsingular. The entries of `L` above the diagonal
and of `U` below it are exact zeros and the diagonal of `L` exact ones, as for the factors.

# Reference

S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 3.2.
"""
function _rumpogita2024_sec3_2(A::BallMatrix{T}) where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(ArgumentError("_rumpogita2024_sec3_2: A must be square"))
    Am = Matrix(mid(A))
    S = eltype(Am)
    F = try
        lu(Am)
    catch err
        err isa SingularException || rethrow()
        return nothing
    end
    p = Vector{Int}(F.p)
    Lt, Ut = Matrix{S}(F.L), Matrix{S}(F.U)
    all(!iszero, diag(Ut)) || return nothing
    XL = Matrix{S}(inv(UnitLowerTriangular(Lt)))
    XU = Matrix{S}(I / UpperTriangular(Ut))
    (all(isfinite, XL) && all(isfinite, XU)) || return nothing
    Id = Matrix{T}(I, n, n)
    absXL, absXU = _modulus_up(XL), _modulus_up(XU)

    # P1 + P2 ± RP ∋ Ã[p, :] X_U
    P1, P2, RP = _two_term_product_sum(((Am[p, :], XU, one(T)),))
    RP = setrounding(T, RoundUp) do
        RP .+ rad(A)[p, :] * absXU
    end
    # E ∋ X_L (Ã[p, :] X_U) − I
    Eb = _accurate_product_sum(((XL, P1, one(T)), (XL, P2, one(T)), (Id, Id, -one(T))))
    E = BallMatrix(mid(Eb), setrounding(T, RoundUp) do
        rad(Eb) .+ absXL * RP
    end)
    fE = _rumpogita2024_alg3_2(E)
    fE === nothing && return nothing

    # L = (Ã[p, :] X_U)(I + UinvE)
    Pb = BallMatrix(P1) + BallMatrix(P2, RP)
    Lb = Pb + Pb * fE.UinvE
    Lm, Lr = copy(mid(Lb)), copy(rad(Lb))
    for j in 1:n, i in 1:j
        Lm[i, j] = i == j ? one(S) : zero(S)
        Lr[i, j] = zero(T)
    end

    # U = (I + UE) Ũ (I + F)⁻¹, with X_U Ũ = I + F
    Fb = _accurate_product_sum(((XU, Ut, one(T)), (Id, Id, -one(T))))
    Fm, Fr = copy(mid(Fb)), copy(rad(Fb))
    for j in 1:n, i in (j + 1):n            # a product of upper triangular matrices
        Fm[i, j] = zero(S)
        Fr[i, j] = zero(T)
    end
    fF = _rumpogita2024_alg3_2(BallMatrix(Fm, Fr))
    fF === nothing && return nothing
    Utb = BallMatrix(Ut)
    W = Utb + fE.UE * Utb
    Ub = W + W * fF.UinvE
    Um, Ur = copy(mid(Ub)), copy(rad(Ub))
    for j in 1:n, i in (j + 1):n
        Um[i, j] = zero(S)
        Ur[i, j] = zero(T)
    end
    return (; L = BallMatrix(Lm, Lr), U = BallMatrix(Um, Ur), p)
end
