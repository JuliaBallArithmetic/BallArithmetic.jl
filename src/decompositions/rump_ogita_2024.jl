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

# I + F ∋ X Y − I restricted to the triangle the product of two triangular matrices has, enclosed
# in two-fold precision, and Algorithm 3.2 on it: for X ≈ Y⁻¹ both triangular of the same kind,
# X Y = I + F proves X and Y nonsingular and gives X⁻¹ = Y (I + F)⁻¹, Y⁻¹ = (I + F)⁻¹ X.
function _rumpogita2024_triangular_defect(X::Matrix{S}, Y::Matrix{S}, uplo::Symbol) where {S}
    T = real(S)
    n = size(X, 1)
    Id = Matrix{T}(I, n, n)
    Fb = _accurate_product_sum(((X, Y, one(T)), (Id, Id, -one(T))))
    Fm, Fr = copy(mid(Fb)), copy(rad(Fb))
    for j in 1:n, i in 1:n
        if (uplo === :U && i > j) || (uplo === :L && i < j)
            Fm[i, j] = zero(S)
            Fr[i, j] = zero(T)
        end
    end
    return _rumpogita2024_alg3_2(BallMatrix(Fm, Fr))
end

# exact zeros and ones where the factor has them
function _triangular_ball(B::BallMatrix{T}, uplo::Symbol; unit::Bool = false) where {T}
    Bm, Br = copy(mid(B)), copy(rad(B))
    for j in axes(Bm, 2), i in axes(Bm, 1)
        if (uplo === :U && i > j) || (uplo === :L && i < j)
            Bm[i, j] = zero(eltype(Bm))
            Br[i, j] = zero(T)
        elseif unit && i == j
            Bm[i, j] = one(eltype(Bm))
            Br[i, j] = zero(T)
        end
    end
    return BallMatrix(Bm, Br)
end

"""
    _rumpogita2024_lu(A::BallMatrix) -> (; L, U, p, q) or nothing

Section 3 of Rump and Ogita (2024): inclusions of the factors of the LU decomposition of an
`m × n` matrix. For every matrix `Ã` of the ball `A`, the permuted matrix `Ã[p, q]` has a unique
decomposition `LU` with `L` of size `m × min(m, n)` unit lower triangular (trapezoidal) and `U`
of size `min(m, n) × n` upper triangular (trapezoidal), `L` and `U` in the ball matrices
returned. Returns `nothing` when the verification fails; nothing is then proved.

`p` is the row permutation of a floating-point decomposition with partial pivoting. `q` is the
identity for `m ≥ n`; for `m < n` it is the column permutation of a decomposition with partial
pivoting of the transpose, as in Section 3.4, after which the left square block has a
decomposition.

# The method

With `p = min(m, n)`, `B` the leading `p × p` block of `Ã[p, q]`, `L̃`, `Ũ` its floating-point
factors, `X_L ≈ L̃⁻¹` unit lower and `X_U ≈ Ũ⁻¹` upper triangular,

    I_E := X_L B X_U = L_E U_E

is a perturbed identity, enclosed with the product `B X_U` kept as an unevaluated sum of two
terms and the second product in two-fold precision; Algorithm 3.2 encloses `L_E`, `U_E` and their
inverses. Then, by uniqueness of the decomposition,

- for `m ≥ n`, (3.8) and (3.9): `U = U_E X_U⁻¹` and `L = Ã[p, :] X_U U_E⁻¹`, the latter from the
  two-term product of the whole matrix with `X_U`;
- for `m < n`, (3.10): `L = X_L⁻¹ L_E` and `U = L_E⁻¹ X_L Ã[p, q]`, with `X_L Ã[p, q]` kept in
  two terms.

These are the methods the paper finds best in each case (the blue curves of its Figures 3, 5
and 7), at `O(max(m, n) min(m, n)²)` operations. The inverses `X_U⁻¹` and `X_L⁻¹`, which the
paper leaves to the implementation, are enclosed through `X_U Ũ = I + F`, respectively
`X_L L̃ = I + F`, in two-fold precision and Algorithm 3.2 for `(I + F)⁻¹`, which also proves
`X_U` and `X_L` nonsingular. The entries that are zero or one in the factors are exact.

# Reference

S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Sections 3.2, 3.3 and 3.4.
"""
function _rumpogita2024_lu(A::BallMatrix{T}) where {T}
    m, n = size(A)
    k = min(m, n)
    Am0 = Matrix(mid(A))
    S = eltype(Am0)
    # the permutations, from floating-point decompositions with partial pivoting
    q = collect(1:n)
    Fl = try
        if m < n
            q = Vector{Int}(lu(permutedims(Am0)).p)
        end
        lu(Am0[:, q])
    catch err
        err isa SingularException || rethrow()
        return nothing
    end
    p = Vector{Int}(Fl.p)
    Am, Ar = Am0[p, q], rad(A)[p, q]
    # the floating-point factors of the leading square block
    Fs = try
        lu(Am[1:k, 1:k], NoPivot())
    catch err
        err isa Union{SingularException, ZeroPivotException} || rethrow()
        return nothing
    end
    Lt, Ut = Matrix{S}(Fs.L), Matrix{S}(Fs.U)
    (all(isfinite, Lt) && all(isfinite, Ut) && all(!iszero, diag(Ut))) || return nothing
    XL = Matrix{S}(inv(UnitLowerTriangular(Lt)))
    XU = Matrix{S}(I / UpperTriangular(Ut))
    (all(isfinite, XL) && all(isfinite, XU)) || return nothing
    Id = Matrix{T}(I, k, k)
    absXL, absXU = _modulus_up(XL), _modulus_up(XU)
    up(f) = setrounding(f, T, RoundUp)

    if m >= n
        # P1 + P2 ± RP ∋ Ã[p, :] X_U, m × n; its leading block enters I_E
        P1, P2, RP = _two_term_product_sum(((Am, XU, one(T)),))
        RP = up(() -> RP .+ Ar * absXU)
        Eb = _accurate_product_sum(((XL, P1[1:k, :], one(T)), (XL, P2[1:k, :], one(T)),
            (Id, Id, -one(T))))
        E = BallMatrix(mid(Eb), up(() -> rad(Eb) .+ absXL * RP[1:k, :]))
        fE = _rumpogita2024_alg3_2(E)
        fE === nothing && return nothing
        # L = (Ã[p, :] X_U)(I + UinvE)
        Pb = BallMatrix(P1) + BallMatrix(P2, RP)
        L = _triangular_ball(Pb + Pb * fE.UinvE, :L; unit = true)
        # U = (I + UE) Ũ (I + F)⁻¹, with X_U Ũ = I + F
        fF = _rumpogita2024_triangular_defect(XU, Ut, :U)
        fF === nothing && return nothing
        Utb = BallMatrix(Ut)
        W = Utb + fE.UE * Utb
        U = _triangular_ball(W + W * fF.UinvE, :U)
        return (; L, U, p, q)
    else
        # Q1 + Q2 ± RQ ∋ X_L Ã[p, q], m × n; C1 + C2 ± RC ∋ B X_U for I_E
        C1, C2, RC = _two_term_product_sum(((Am[:, 1:k], XU, one(T)),))
        RC = up(() -> RC .+ Ar[:, 1:k] * absXU)
        Eb = _accurate_product_sum(((XL, C1, one(T)), (XL, C2, one(T)), (Id, Id, -one(T))))
        E = BallMatrix(mid(Eb), up(() -> rad(Eb) .+ absXL * RC))
        fE = _rumpogita2024_alg3_2(E)
        fE === nothing && return nothing
        # L = L̃ (I + F)⁻¹ (I + LE), with X_L L̃ = I + F
        fF = _rumpogita2024_triangular_defect(XL, Lt, :L)
        fF === nothing && return nothing
        Ltb = BallMatrix(Lt)
        W = Ltb + Ltb * fF.LinvE
        L = _triangular_ball(W + W * fE.LE, :L; unit = true)
        # U = (I + LinvE)(X_L Ã[p, q])
        Q1, Q2, RQ = _two_term_product_sum(((XL, Am, one(T)),))
        RQ = up(() -> RQ .+ absXL * Ar)
        Qb = BallMatrix(Q1) + BallMatrix(Q2, RQ)
        U = _triangular_ball(Qb + fE.LinvE * Qb, :U)
        return (; L, U, p, q)
    end
end

"""
    _rumpogita2024_cholesky(A::BallMatrix) -> (; G) or nothing

Section 4 of Rump and Ogita (2024): an inclusion of the Cholesky factor. For every Hermitian
matrix `Ã` of the ball `A`, `Ã` is proved positive definite and its Cholesky factor, the upper
triangular `G` with positive diagonal and `G*G = Ã`, lies in the ball matrix `G` returned.
Positive definiteness is a conclusion and not an assumption. Returns `nothing` when the
verification fails. The midpoint of `A` must be Hermitian and its radii symmetric.

# The method

`G̃` is a floating-point Cholesky factor of the midpoint and `X_G ≈ G̃⁻¹`, upper triangular.
`I_E := X_G* Ã X_G` is a perturbed identity, enclosed with `Ã X_G` kept as an unevaluated sum of
two terms; Algorithm 3.2 encloses its factors, `I_E = L_E U_E`. `I_E` being Hermitian,
`U_E = D L_E*` with `D` the diagonal of `U_E`, which is real; when `D > 0` is proved,
`G_E = D^{1/2} L_E*` is the Cholesky factor of `I_E`, and by uniqueness, (4.1) of the paper,

    G = G_E X_G⁻¹ = D^{1/2} L_E* X_G⁻¹.

`X_G⁻¹` is enclosed through `G̃ X_G = I + F` in two-fold precision and Algorithm 3.2 for
`(I + F)⁻¹`, as in [`_rumpogita2024_lu`](@ref).

# Reference

S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 4.
"""
function _rumpogita2024_cholesky(A::BallMatrix{T}) where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(ArgumentError("_rumpogita2024_cholesky: A must be square"))
    Am, Ar = Matrix(mid(A)), rad(A)
    (Am == Am' && Ar == transpose(Ar)) || throw(ArgumentError(
        "_rumpogita2024_cholesky: the midpoint must be Hermitian and the radii symmetric"))
    S = eltype(Am)
    C = cholesky(Hermitian(Am); check = false)
    issuccess(C) || return nothing
    Gt = Matrix{S}(C.U)
    XG = Matrix{S}(I / UpperTriangular(Gt))
    all(isfinite, XG) || return nothing
    Id = Matrix{T}(I, n, n)
    absXG = _modulus_up(XG)
    XGa = Matrix{S}(XG')
    # C1 + C2 ± RC ∋ Ã X_G, then E ∋ X_G* (Ã X_G) − I
    C1, C2, RC = _two_term_product_sum(((Am, XG, one(T)),))
    RC = setrounding(T, RoundUp) do
        RC .+ Ar * absXG
    end
    Eb = _accurate_product_sum(((XGa, C1, one(T)), (XGa, C2, one(T)), (Id, Id, -one(T))))
    E = BallMatrix(mid(Eb), setrounding(T, RoundUp) do
        rad(Eb) .+ transpose(absXG) * RC
    end)
    fE = _rumpogita2024_alg3_2(E)
    fE === nothing && return nothing
    # D = diag(U_E) = 1 + diag(UE), real; its square root as a ball, or failure
    dm, dr = Vector{S}(undef, n), Vector{T}(undef, n)
    for i in 1:n
        b = Ball(one(T), zero(T)) + Ball(real(mid(fE.UE)[i, i]), rad(fE.UE)[i, i])
        lo, hi = sub_down(mid(b), rad(b)), add_up(mid(b), rad(b))
        lo > 0 || return nothing
        slo, shi = sqrt_down(lo), sqrt_up(hi)
        c = (slo + shi) / 2
        dm[i] = c
        dr[i] = max(sub_up(c, slo), sub_up(shi, c))
    end
    Dh = BallMatrix(Matrix{S}(Diagonal(dm)), Matrix{T}(Diagonal(dr)))
    # G = D^{1/2} (I + LE)* (I + F)⁻¹ G̃, with G̃ X_G = I + F
    fF = _rumpogita2024_triangular_defect(Gt, XG, :U)
    fF === nothing && return nothing
    Gtb = BallMatrix(Gt)
    W = Gtb + fF.UinvE * Gtb
    LEa = BallMatrix(Matrix{S}(mid(fE.LE)'), Matrix{T}(transpose(rad(fE.LE))))
    G = _triangular_ball(Dh * (W + LEa * W), :U)
    return (; G)
end
