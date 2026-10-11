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

# D = diag(U_E) = 1 + diag(UE) is real for a Hermitian I + E. Returns ball matrices containing
# D^{1/2} and D^{-1/2}, diagonal, or `nothing` when D > 0 is not proved.
function _rumpogita2024_sqrt_diagonal(UE::BallMatrix{T}, ::Type{S}) where {T, S}
    n = size(UE, 1)
    interval_ball(lo, hi) = (c = (lo + hi) / 2; (c, max(sub_up(c, lo), sub_up(hi, c))))
    dm, dr, im_, ir = Vector{S}(undef, n), Vector{T}(undef, n), Vector{S}(undef, n), Vector{T}(undef, n)
    for i in 1:n
        b = Ball(one(T), zero(T)) + Ball(real(mid(UE)[i, i]), rad(UE)[i, i])
        lo, hi = sub_down(mid(b), rad(b)), add_up(mid(b), rad(b))
        lo > 0 || return nothing
        slo, shi = sqrt_down(lo), sqrt_up(hi)
        dm[i], dr[i] = interval_ball(slo, shi)
        im_[i], ir[i] = interval_ball(div_down(one(T), shi), div_up(one(T), slo))
    end
    return BallMatrix(Matrix{S}(Diagonal(dm)), Matrix{T}(Diagonal(dr))),
    BallMatrix(Matrix{S}(Diagonal(im_)), Matrix{T}(Diagonal(ir)))
end

_ball_adjoint(B::BallMatrix{T}) where {T} =
    BallMatrix(Matrix(adjoint(mid(B))), Matrix{T}(transpose(rad(B))))

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
    Dh = _rumpogita2024_sqrt_diagonal(fE.UE, S)
    Dh === nothing && return nothing
    Dh = Dh[1]
    # G = D^{1/2} (I + LE)* (I + F)⁻¹ G̃, with G̃ X_G = I + F
    fF = _rumpogita2024_triangular_defect(Gt, XG, :U)
    fF === nothing && return nothing
    Gtb = BallMatrix(Gt)
    W = Gtb + fF.UinvE * Gtb
    G = _triangular_ball(Dh * (W + _ball_adjoint(fE.LE) * W), :U)
    return (; G)
end

"""
    _rumpogita2024_qr(A::BallMatrix; full = false) -> (; Q, R) or nothing

Section 5 of Rump and Ogita (2024): inclusions of the factors of the QR decomposition. For every
matrix `Ã` of the `m × n` ball `A`, `Ã` is proved to have full rank and `Ã = QR` with `R` upper
triangular with positive diagonal entries, `Q` and `R` in the ball matrices returned. Returns
`nothing` when the verification fails.

For `m ≥ n` and `full = false` the decomposition is the economy-size one, `Q` of size `m × n`
with orthonormal columns and `R` of size `n × n`, which is unique. With `full = true`, `Q` is
`m × m` unitary and `R` is `m × n`; its last `m − n` columns are an orthonormal basis of the
orthogonal complement of the range, which is not unique, and the statement is that some such
basis lies in the ball. For `m < n` the decomposition is the full one of the leading square
block, `Q` of size `m × m`, and `R = Q*Ã` of size `m × n`.

# The method

`Q̃R̃` is a floating-point decomposition of the midpoint, `R̃` made to have a real positive
diagonal, and `X_R ≈ R̃⁻¹`. `C := Ã X_R`, kept as an unevaluated sum of two terms, is close to a
matrix with orthonormal columns, and `C*C = I + E` is enclosed in two-fold precision without
forming `Ã*Ã`. The Cholesky factor of `I + E` is `G_E = D^{1/2} L_E*` by Algorithm 3.2 as in
Section 4, and

    R = G_E X_R⁻¹,        Q₁ = Ã X_R G_E⁻¹,

the second being the possibility the paper finds better for `Q₁`. `R` is the Cholesky factor of
`Ã*Ã`, upper triangular with positive diagonal since those of `G_E` and of `X_R` are.

For the orthogonal complement, with `Q̃₂` its floating-point approximation, the square system
(5.1), `[Ã*; N] Y = [0; αI]` with `N = fl(α Q̃₂*)`, has a solution whose columns span the
orthogonal complement of the range of `Ã`. An inclusion `Q̃₂ + Δ` of it comes from a verified
solve for the correction, with the residual in two-fold precision, and Lemma 5.1 gives an
orthonormal basis `Q₂` of that complement with

    ‖Q₂ − Q̃₂‖₂ ≤ ‖I − Q̃₂*Q̃₂‖₂ + √2 ‖Δ‖₂,

which bounds every entry of `Q₂ − Q̃₂`.

# Reference

S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 5 and Lemma 5.1.
"""
function _rumpogita2024_qr(A::BallMatrix{T}; full::Bool = false) where {T}
    m, n = size(A)
    if m < n
        Am, Ar = mid(A), rad(A)
        r = _rumpogita2024_qr(BallMatrix(Am[:, 1:m], Ar[:, 1:m]); full = true)
        r === nothing && return nothing
        rest = _ball_adjoint(r.Q) * BallMatrix(Am[:, (m + 1):n], Ar[:, (m + 1):n])
        return (; Q = r.Q, R = BallMatrix(hcat(mid(r.R), mid(rest)), hcat(rad(r.R), rad(rest))))
    end
    Am, Ar = Matrix(mid(A)), rad(A)
    S = eltype(Am)
    Fq = qr(Am)
    Rt = Matrix{S}(Fq.R)
    for i in 1:n                      # a real positive diagonal for the approximate factor
        d = Rt[i, i]
        (isfinite(d) && !iszero(d)) || return nothing
        Rt[i, :] .*= conj(d / abs(d))
        Rt[i, i] = abs(d)
    end
    XR = Matrix{S}(I / UpperTriangular(Rt))
    all(isfinite, XR) || return nothing
    Id = Matrix{T}(I, n, n)
    absXR = _modulus_up(XR)
    up(f) = setrounding(f, T, RoundUp)
    # C1 + C2 ± RC ∋ Ã X_R
    C1, C2, RC = _two_term_product_sum(((Am, XR, one(T)),))
    RC = up(() -> RC .+ Ar * absXR)
    # E ∋ C*C − I
    C1a, C2a = Matrix{S}(C1'), Matrix{S}(C2')
    Eb = _accurate_product_sum(((C1a, C1, one(T)), (C1a, C2, one(T)), (C2a, C1, one(T)),
        (C2a, C2, one(T)), (Id, Id, -one(T))))
    absC = up(() -> _modulus_up(C1) .+ _modulus_up(C2))
    E = BallMatrix(mid(Eb), up(() -> rad(Eb) .+ transpose(RC) * absC .+ transpose(absC) * RC .+
                                     transpose(RC) * RC))
    fE = _rumpogita2024_alg3_2(E)
    fE === nothing && return nothing
    D = _rumpogita2024_sqrt_diagonal(fE.UE, S)
    D === nothing && return nothing
    Dh, Dhinv = D
    # R = D^{1/2} (I + LE)* (I + F)⁻¹ R̃, with R̃ X_R = I + F
    fF = _rumpogita2024_triangular_defect(Rt, XR, :U)
    fF === nothing && return nothing
    Rtb = BallMatrix(Rt)
    W = Rtb + fF.UinvE * Rtb
    R = _triangular_ball(Dh * (W + _ball_adjoint(fE.LE) * W), :U)
    # Q₁ = C (I + LinvE)* D^{-1/2}
    Cb = BallMatrix(C1) + BallMatrix(C2, RC)
    Q1 = (Cb + Cb * _ball_adjoint(fE.LinvE)) * Dhinv
    (full && m > n) || return (; Q = Q1, R)

    # the orthogonal complement: system (5.1) and Lemma 5.1
    k = m - n
    Q2t = Matrix{S}((Fq.Q * Matrix{S}(I, m, m))[:, (n + 1):m])
    sv = svdvals(Rt)
    α = T(sqrt(sv[1] * sv[end]))
    (isfinite(α) && α > 0) || return nothing
    N = Matrix{S}(α * Q2t')
    M = BallMatrix(vcat(Matrix{S}(Am'), N), vcat(Matrix{T}(transpose(Ar)), zeros(T, k, m)))
    # the residual of Q̃₂: [−Ã*Q̃₂; αI − N Q̃₂]
    top = _accurate_product_sum(((Matrix{S}(Am'), Q2t, -one(T)),))
    top = BallMatrix(mid(top), up(() -> rad(top) .+ transpose(Ar) * _modulus_up(Q2t)))
    Ik = Matrix{T}(I, k, k)
    bot = _accurate_product_sum(((N, Q2t, -one(T)), (α * Ik, Ik, one(T))))
    sol = verifylss(M, BallMatrix(vcat(mid(top), mid(bot)), vcat(rad(top), rad(bot))))
    sol.certified || return nothing
    δ = upper_bound_L2_opnorm(sol.solution)
    Q2b = BallMatrix(Q2t)
    αo = upper_bound_L2_opnorm(_ball_adjoint(Q2b) * Q2b - I)
    ρ = add_up(T(αo), mul_up(sqrt_up(T(2)), T(δ)))
    isfinite(ρ) || return nothing
    Q = BallMatrix(hcat(mid(Q1), Q2t), hcat(rad(Q1), fill(ρ, m, k)))
    Rfull = BallMatrix(vcat(mid(R), zeros(S, k, n)), vcat(rad(R), zeros(T, k, n)))
    return (; Q, R = Rfull)
end

"""
    _rumpogita2024_schur(A::BallMatrix; kwargs...) -> (; Q, T, X, D) or nothing

Section 8 of Rump and Ogita (2024): inclusions of the factors of a complex Schur decomposition
of a matrix with simple eigenvalues. For every matrix `Ã` of the ball `A` there are a unitary
`Q` and an upper triangular `T` with `Ã = QTQ*`, `Q` and `T` in the ball matrices returned; the
diagonal of `T` holds the eigenvalues in the order of `D`. Returns `nothing` when the
verification fails, which includes every matrix for which the eigenvalues are not proved
simple: the paper restricts the method to that case, the decomposition being discontinuous at a
multiple eigenvalue.

# The method

(8.1) of the paper: with `Ã = XDX⁻¹` an eigendecomposition and `X = QR` the QR decomposition of
`X`, `Ã = QTQ*` with `T = RDR⁻¹`. Inclusions of `X` and of the diagonal `D` come from
[`verifyeigall`](@ref) (Section 6 refers to Rump (2022) for a general matrix), each cluster
being required to be a single certified eigenvalue; `kwargs` are passed to it. The inclusions of
`Q` and `R` are those of [`_rumpogita2024_qr`](@ref) for the ball matrix `X`, valid for the true
eigenvector matrix in it; and `T` is the solution of the linear system `TR = RD` with `R` and
`D` replaced by their inclusions, enclosed by a verified solve. Its diagonal is `D` and its
entries below the diagonal are exact zeros. The paper notes that the replacement of `X`, `R` and
`D` by inclusions is a source of overestimation that the other decompositions do not have.

# Reference

S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 8.
"""
function _rumpogita2024_schur(A::BallMatrix{T}; kwargs...) where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(ArgumentError("_rumpogita2024_schur: A must be square"))
    CT = complex(T)
    e = verifyeigall(A; kwargs...)
    (e.spectrum_covered && all(e.certified) && all(c -> length(c) == 1, e.clusters) &&
     length(e.clusters) == n) || return nothing
    Xm, Xr = Matrix{CT}(undef, n, n), Matrix{T}(undef, n, n)
    dm, dr = Vector{CT}(undef, n), Vector{T}(undef, n)
    for j in 1:n
        Xm[:, j] .= vec(mid(e.subspaces[j]))
        Xr[:, j] .= vec(rad(e.subspaces[j]))
        dm[j], dr[j] = mid(e.blocks[j])[1, 1], rad(e.blocks[j])[1, 1]
    end
    (all(isfinite, Xr) && all(isfinite, dr)) || return nothing
    X = BallMatrix(Xm, Xr)
    D = BallMatrix(Matrix{CT}(Diagonal(dm)), Matrix{T}(Diagonal(dr)))
    f = _rumpogita2024_qr(X)
    f === nothing && return nothing
    # T R = R D, that is Rᵀ Tᵀ = (R D)ᵀ
    Rt = BallMatrix(Matrix(transpose(mid(f.R))), Matrix(transpose(rad(f.R))))
    RD = f.R * D
    sol = verifylss(Rt, BallMatrix(Matrix(transpose(mid(RD))), Matrix(transpose(rad(RD)))))
    sol.certified || return nothing
    Tm, Tr = Matrix(transpose(mid(sol.solution))), Matrix(transpose(rad(sol.solution)))
    for j in 1:n, i in 1:n
        if i > j
            Tm[i, j], Tr[i, j] = zero(CT), zero(T)
        elseif i == j
            Tm[i, j], Tr[i, j] = dm[i], dr[i]
        end
    end
    return (; Q = f.Q, T = BallMatrix(Tm, Tr), X, D)
end

"""
    _rumpogita2024_polar(A::BallMatrix; kappa = 0) -> (; Q, P, svd) or nothing

Section 7 of Rump and Ogita (2024): inclusions of the factors of the polar decomposition. For
every matrix `Ã` of the `m × n` ball `A`, `m ≥ n`, `Ã` is proved to have full rank and
`Ã = QP` with `Q` of size `m × n` with orthonormal columns and `P` Hermitian positive definite,
`Q` and `P` in the ball matrices returned. Returns `nothing` when a singular value is not
separated from zero. `svd` is the result of [`verifysvdall`](@ref) the inclusions come from, and
`kappa` its threshold.

# The method

The paper obtains the factors from the singular value decomposition `Ã = UΣV*` as `Q = UV*`
and `P = VΣV*`, with the inclusions of Rump and Lange (2023). Those are, for each cluster `μ`
of singular values, inclusions of orthonormal bases of the left and of the right singular
subspace, and the two bases of a cluster are not matched to each other (for one singular value,
the two vectors are determined up to a factor of modulus one each). So the factors are written
with the right bases alone: with `V_μ` any orthonormal basis of the right singular subspace of
the cluster and `H_μ := V_μ*Ã*ÃV_μ`, whose eigenvalues are the squares of the singular values of
the cluster,

    P = Σ_μ V_μ H_μ^{1/2} V_μ*,        Q = Ã Σ_μ V_μ H_μ^{-1/2} V_μ*.

The singular values of the cluster lie in an interval `[a, b]` with `a > 0`, so `H_μ^{1/2}` is
within `(b − a)/2` of `((a + b)/2) I` in the spectral norm, hence entrywise, and `H_μ^{-1/2}`
within half the width of `[1/b, 1/a]` of its midpoint times `I`. For a cluster of one singular
value these are `P = Σ σ_j v_j v_j*` and `Q = Σ Ãv_j v_j*/σ_j`.

# Reference

S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 7; and Rump and Lange (2023)
for the singular value decomposition, see [`verifysvdall`](@ref).
"""
function _rumpogita2024_polar(A::BallMatrix{T}; kappa::Real = 0) where {T}
    m, n = size(A)
    m >= n || throw(ArgumentError("_rumpogita2024_polar: A must have at least as many rows as columns"))
    r = verifysvdall(A; kappa)
    S = eltype(mid(r.V))
    lo = T[sub_down(mid(b), rad(b)) for b in r.values]
    hi = T[add_up(mid(b), rad(b)) for b in r.values]
    (all(>(0), lo) && all(isfinite, hi) && all(isfinite, rad(r.V))) || return nothing
    # an interval [a, b] as the ball matrix (c ± ρ) I of the size of the cluster, every entry ± ρ
    function scalar_block(a, b, k)
        c = (a + b) / 2
        ρ = max(sub_up(c, a), sub_up(b, c))
        return BallMatrix(Matrix{S}(c * I, k, k), fill(ρ, k, k))
    end
    P = BallMatrix(zeros(S, n, n))
    Pinv = BallMatrix(zeros(S, n, n))
    for v in r.clusters
        a, b = minimum(lo[v]), maximum(hi[v])
        V = r.V[:, v]
        Va = _ball_adjoint(V)
        P = P + V * scalar_block(a, b, length(v)) * Va
        Pinv = Pinv + V * scalar_block(div_down(one(T), b), div_up(one(T), a), length(v)) * Va
    end
    Ac = BallMatrix(Matrix{S}(mid(A)), rad(A))
    return (; Q = Ac * Pinv, P, svd = r)
end

"""
    _rumpogita2024_takagi(A::BallMatrix) -> (; U, sigma) or nothing

Section 9 of Rump and Ogita (2024): inclusions of the factors of the Takagi decomposition of a
complex symmetric matrix. For every symmetric matrix `Ã = Ãᵀ` of the ball `A`, `Ã` is proved
nonsingular with simple singular values and `Ã = UΣUᵀ` with `U` unitary and `Σ` diagonal with
positive entries in decreasing order, `U` in the ball matrix `U` and the diagonal of `Σ` in the
balls `sigma`. `U` is determined up to the sign of each column. Returns `nothing` when the
singular values are not proved positive and simple; the decomposition is discontinuous at a
singular matrix.

# The method

The third method of the paper, which it finds best. With `Ã = E + iF`, `E` and `F` real
symmetric, the real symmetric matrix `M = [E F; F −E]` has the eigenvalues `±σ_j`; if `[x; y]`
is a unit eigenvector for `σ_j > 0`, then `u = x + iy` is a unit vector with `Ã ū = σ_j u`, and
these `u` are the columns of `U`. The eigenvalues and eigenvectors of `M` are enclosed by
[`_rumplange2023_eig`](@ref); each positive eigenvalue is required to be a cluster of its own,
so that its unit eigenvector is determined up to sign. The bound on the eigenvector is a bound
of the norm of the error of `[x; y]`, hence of the modulus of each entry of `x + iy`.

# Reference

S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 9.
"""
function _rumpogita2024_takagi(A::BallMatrix{T}) where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(ArgumentError("_rumpogita2024_takagi: A must be square"))
    Am, Ar = Matrix(mid(A)), rad(A)
    (Am == transpose(Am) && Ar == transpose(Ar)) || throw(ArgumentError(
        "_rumpogita2024_takagi: the midpoint must be symmetric (not Hermitian) and the radii symmetric"))
    E, F = Matrix{T}(real.(Am)), Matrix{T}(imag.(Am))
    M = BallMatrix([E F; F -E], [Ar Ar; Ar Ar])
    r = _rumplange2023_eig(M)
    pos = [v[1] for v in r.clusters if length(v) == 1 && r.lo[v[1]] > 0]
    length(pos) == n || return nothing
    count(v -> length(v) == 1 && r.hi[v[1]] < 0, r.clusters) == n || return nothing
    sort!(pos; by = j -> -(r.lo[j] + r.hi[j]))
    W, Wr = mid(r.vectors), rad(r.vectors)
    all(isfinite, Wr[:, pos]) || return nothing
    Um = Matrix{Complex{T}}(undef, n, n)
    Ur = Matrix{T}(undef, n, n)
    sigma = Vector{Ball{T, T}}(undef, n)
    for (k, j) in enumerate(pos)
        Um[:, k] .= complex.(W[1:n, j], W[(n + 1):(2n), j])
        Ur[:, k] .= Wr[1, j]
        c = (r.lo[j] + r.hi[j]) / 2
        sigma[k] = Ball(c, max(sub_up(c, r.lo[j]), sub_up(r.hi[j], c)))
    end
    return (; U = BallMatrix(Um, Ur), sigma)
end
