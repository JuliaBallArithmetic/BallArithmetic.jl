# Enclosures for the solution of the Sylvester equation A X + X B = C.
#
# The scheme of every routine here: an approximate solution in floating point, its residual in
# ball arithmetic, and a bound of the error from the residual.

_real_type(::Type{T}) where {T <: Real} = float(T)
_real_type(::Type{Complex{T}}) where {T <: Real} = float(T)

# a matrix or a ball matrix as a ball matrix with midpoints of type CT
_ball_as(::Type{CT}, M::BallMatrix) where {CT} = BallMatrix(Matrix{CT}(mid(M)), Matrix(rad(M)))
_ball_as(::Type{CT}, M::AbstractMatrix) where {CT} = BallMatrix(Matrix{CT}(M))
_mid_eltype(M::BallMatrix) = eltype(mid(M))
_mid_eltype(M::AbstractMatrix) = eltype(M)

# approximate eigenvectors V, an approximate inverse W of V, and approximate eigenvalues, in
# floating point: nothing is assumed about their accuracy
function _approx_eigen(M::Matrix)
    RT = _real_type(eltype(M))
    if (istriu(M) || istril(M)) && _has_distinct_diagonal(M, eps(RT))
        return triangular_eigenvectors(M; tol = eps(RT))
    end
    F = eigen(M)
    V = Matrix(F.vectors)
    return V, inv(V), F.values
end

_rowsums_up(M::AbstractMatrix{T}) where {T} =
    [foldl(add_up, view(M, i, :); init = zero(T)) for i in axes(M, 1)]

"""
    sylvester_miyajima_enclosure(A, B, C, X̃)

An entrywise enclosure of the solution `X*` of `A X + X B = C`, as a `BallMatrix` with midpoint
`X̃` (an approximate solution, in floating point) and radius `X^ε` with `|X̃ − X*| ≤ X^ε`. `A`
(`m×m`), `B` (`n×n`) and `C` may be matrices or `BallMatrix`; with balls, the enclosure holds for
the solution of every equation with data in them.

Theorems 1 and 2 of

S. Miyajima, *Fast enclosure for solutions of Sylvester equations*, Linear Algebra Appl. 439
(2013), 856–878, doi:10.1016/j.laa.2012.07.001 ([MiyajimaSylvester2013](@cite)),

Section 2.1 (spectral decomposition). The block-diagonalisation variant of Section 2.2
(Theorems 3 and 4), the accelerations of Section 3 and the refinement of Section 4 are not
implemented.

# The statement

Let `Ṽ_A, W_A` (`m×m`), `Ṽ_B, W_B` (`n×n`) be any matrices and `D̃_A, D̃_B` diagonal; in practice
`A ≈ Ṽ_A D̃_A Ṽ_A⁻¹`, `Bᵀ ≈ Ṽ_B D̃_B Ṽ_B⁻¹`, `W_A ≈ Ṽ_A⁻¹`, `W_B ≈ Ṽ_B⁻¹`, all computed in floating
point. `E` is the `m×n` matrix of ones, `./` and `|·|` are entrywise, `‖·‖_∞` is the maximum row
sum and `‖·‖_M` the largest entry in modulus.

    R_A = W_A(Ṽ_A D̃_A − A Ṽ_A),     R_B = W_B(Ṽ_B D̃_B − Bᵀ Ṽ_B),
    S_A = I − W_A Ṽ_A,              S_B = I − W_B Ṽ_B,
    T_A = |R_A| + ‖R_A‖_∞/(1 − ‖S_A‖_∞) |S_A|,     T_B likewise,
    T = T_A E + E T_Bᵀ,     D̃ = D̃_A E + E D̃_B,     T_D = T ./ |D̃|.

*Theorem 1.* If `‖S_A‖_∞ < 1`, `‖S_B‖_∞ < 1`, `D̃` has no zero entry and `‖T_D‖_M < 1`, the
equation has a unique solution `X*`.

With `R = A X̃ + X̃ B − C`, `R_W = W_A R W_Bᵀ`, and `ρ_i(M)`, `κ_j(M)` the largest entry in
modulus of row `i` and of column `j`,

    R_W⁽¹⁾ = |R_W| + 1/(1 − ‖S_B‖_∞) · diag(ρ(R_W)) E |S_B|ᵀ,
    R_V⁽¹⁾ = R_W⁽¹⁾ + 1/(1 − ‖S_A‖_∞) · |S_A| E diag(κ(R_W⁽¹⁾)),
    R_W⁽²⁾ = |R_W| + 1/(1 − ‖S_A‖_∞) · |S_A| E diag(κ(R_W)),
    R_V⁽²⁾ = R_W⁽²⁾ + 1/(1 − ‖S_B‖_∞) · diag(ρ(R_W⁽²⁾)) E |S_B|ᵀ,
    R_V = min(R_V⁽¹⁾, R_V⁽²⁾),     R_D = R_V ./ |D̃|,
    U = R_D + ‖R_D‖_M/(1 − ‖T_D‖_M) · T_D.

*Theorem 2.* Under the hypotheses of Theorem 1, `|X̃ − X*| ≤ X^ε := |Ṽ_A| U |Ṽ_B|ᵀ`.

`R_V` bounds `|Ṽ_A⁻¹ R Ṽ_B⁻ᵀ|`, the residual in the exact inverse frames, from `R_W`, the one in
the computed `W_A`, `W_B`.

# How it is evaluated

The products of matrices are ball products, of which the entrywise upper bound is taken. The
products with `E` are row sums (`(M E)_ij` is the sum of row `i` of `M`), accumulated rounded up;
`|D̃_ij|` is bounded from below; every other operation is rounded up. Throws `ArgumentError` when a
hypothesis of Theorem 1 is not proved.
"""
function sylvester_miyajima_enclosure(A::Union{AbstractMatrix, BallMatrix},
        B::Union{AbstractMatrix, BallMatrix}, C::Union{AbstractMatrix, BallMatrix},
        X̃::AbstractMatrix)
    m, mA = size(A)
    m == mA || throw(DimensionMismatch("A must be square"))
    nB, n = size(B)
    nB == n || throw(DimensionMismatch("B must be square"))
    size(C) == (m, n) || throw(DimensionMismatch("C must be of size ($m, $n)"))
    size(X̃) == (m, n) || throw(DimensionMismatch("X̃ must be of size ($m, $n)"))

    CT0 = float(promote_type(_mid_eltype(A), _mid_eltype(B), _mid_eltype(C), eltype(X̃)))
    VA, WA, λA = _approx_eigen(Matrix{CT0}(A isa BallMatrix ? mid(A) : A))
    VB, WB, λB = _approx_eigen(Matrix{CT0}(transpose(B isa BallMatrix ? mid(B) : B)))
    CT = promote_type(CT0, eltype(VA), eltype(WA), eltype(λA), eltype(VB), eltype(WB), eltype(λB))
    RT = real(CT)
    one_ = one(RT)

    bA, bB, bC = _ball_as(CT, A), _ball_as(CT, B), _ball_as(CT, C)
    bBt = BallMatrix(Matrix(transpose(mid(bB))), Matrix(transpose(rad(bB))))
    bVA, bWA = BallMatrix(Matrix{CT}(VA)), BallMatrix(Matrix{CT}(WA))
    bVB, bWB = BallMatrix(Matrix{CT}(VB)), BallMatrix(Matrix{CT}(WB))
    λA, λB = Vector{CT}(λA), Vector{CT}(λB)

    # Theorem 1
    absSA = upper_abs(bWA * bVA - I)
    absSB = upper_abs(bWB * bVB - I)
    absRA = upper_abs(bWA * (bVA * BallMatrix(Matrix(Diagonal(λA))) - bA * bVA))
    absRB = upper_abs(bWB * (bVB * BallMatrix(Matrix(Diagonal(λB))) - bBt * bVB))
    sA, sB = _rowsums_up(absSA), _rowsums_up(absSB)
    rA, rB = _rowsums_up(absRA), _rowsums_up(absRB)
    norm_SA, norm_SB = maximum(sA), maximum(sB)
    norm_SA < 1 || throw(ArgumentError("‖S_A‖_∞ must be < 1"))
    norm_SB < 1 || throw(ArgumentError("‖S_B‖_∞ must be < 1"))
    cA = div_up(one_, sub_down(one_, norm_SA))
    cB = div_up(one_, sub_down(one_, norm_SB))
    # the row sums of T_A and T_B: (T_A E)_ij = tA[i], (E T_Bᵀ)_ij = tB[j]
    tA = [add_up(rA[i], mul_up(mul_up(maximum(rA), cA), sA[i])) for i in 1:m]
    tB = [add_up(rB[j], mul_up(mul_up(maximum(rB), cB), sB[j])) for j in 1:n]
    # lower bounds of |D̃_ij| = |λA[i] + λB[j]|
    D_lo = Matrix{RT}(undef, m, n)
    for j in 1:n, i in 1:m
        d = Ball(λA[i], zero(RT)) + Ball(λB[j], zero(RT))
        D_lo[i, j] = sub_down(_abs_lo(mid(d)), rad(d))
    end
    all(>(0), D_lo) || throw(ArgumentError("Encountered zero spectral gap"))
    T_D = [div_up(add_up(tA[i], tB[j]), D_lo[i, j]) for i in 1:m, j in 1:n]
    norm_TD = maximum(T_D)
    norm_TD < 1 || throw(ArgumentError("Entrywise max norm of T_D must be < 1"))

    # Theorem 2
    bX = BallMatrix(Matrix{CT}(X̃))
    R = bA * bX + bX * bB - bC
    RW = upper_abs(bWA * R * BallMatrix(Matrix(transpose(mid(bWB)))))
    rowmax(M) = [maximum(view(M, i, :)) for i in 1:m]
    colmax(M) = [maximum(view(M, :, j)) for j in 1:n]
    with_SB(M, d) = [add_up(M[i, j], mul_up(mul_up(cB, d[i]), sB[j])) for i in 1:m, j in 1:n]
    with_SA(M, d) = [add_up(M[i, j], mul_up(mul_up(cA, sA[i]), d[j])) for i in 1:m, j in 1:n]
    RW1 = with_SB(RW, rowmax(RW))
    RV1 = with_SA(RW1, colmax(RW1))
    RW2 = with_SA(RW, colmax(RW))
    RV2 = with_SB(RW2, rowmax(RW2))
    R_D = [div_up(min(RV1[i, j], RV2[i, j]), D_lo[i, j]) for i in 1:m, j in 1:n]
    factor = div_up(maximum(R_D), sub_down(one_, norm_TD))
    U = [add_up(R_D[i, j], mul_up(factor, T_D[i, j])) for i in 1:m, j in 1:n]

    Xε = upper_abs(BallMatrix(_abs_hi.(mid(bVA))) * BallMatrix(U) *
                   BallMatrix(Matrix(transpose(_abs_hi.(mid(bVB))))))
    return BallMatrix(Matrix(X̃), Matrix{_real_type(eltype(X̃))}(Xε))
end

"""
    triangular_sylvester_miyajima_enclosure(T, k; sylvester_fallback = :direct)

For the upper triangular `T = [T₁₁ T₁₂; 0 T₂₂]` with `T₁₁` of size `k×k`, an enclosure of the
solution `Y` of

    T₂₂* Y − Y T₁₁* = T₁₂*,

that is of `A Y + Y B = C` with `A = T₂₂*`, `B = −T₁₁*`, `C = T₁₂*`. `T` is a matrix or a
`BallMatrix`; with a ball the enclosure holds for every matrix in it, the radii entering the
residuals (no perturbation expansion is used).

An approximate solution is computed column by column, and enclosed by
[`sylvester_miyajima_enclosure`](@ref). When that does not apply (its hypotheses are not proved,
typically for large triangular matrices whose eigenvector matrix is ill conditioned), the
enclosure comes from the triangular structure, which requires the radii of `T` below the diagonal
to be zero:

- `sylvester_fallback = :direct`: the column recurrence solved by forward substitution in ball
  arithmetic;
- `sylvester_fallback = :residual`: the residual of the approximate solution in ball arithmetic
  and, column by column from the last, `(A + b_jj I) e_j = R_j − Σ_{l>j} e_l b_lj` for the error
  `e_j` of column `j`, so `‖e_j‖₂ ≤ ‖(A + b_jj I)⁻¹‖₂ (‖R_j‖₂ + Σ_{l>j} ‖e_l‖₂ |b_lj|)`, the norm
  of the inverse of the triangular matrix bounded as in
  [`triangular_inverse_two_norm_bound`](@ref). The radius is uniform in each column.
"""
function triangular_sylvester_miyajima_enclosure(T::Union{AbstractMatrix, BallMatrix},
        k::Integer; sylvester_fallback::Symbol = :direct)
    sylvester_fallback ∈ (:direct, :residual) ||
        throw(ArgumentError("sylvester_fallback must be :direct or :residual, got :$sylvester_fallback"))
    n, m = size(T)
    n == m || throw(DimensionMismatch("T must be square"))
    1 <= k < n || throw(ArgumentError("k must satisfy 1 ≤ k < $n"))

    bT = _ball_as(float(_mid_eltype(T)), T)
    Tm, Tr = mid(bT), rad(bT)
    istriu(Tm) || throw(ArgumentError("T must be upper triangular"))
    adjoint_block(I, J) = BallMatrix(Matrix(adjoint(Tm[I, J])), Matrix(transpose(Tr[I, J])))
    A = adjoint_block((k + 1):n, (k + 1):n)       # lower triangular
    B = -adjoint_block(1:k, 1:k)                  # lower triangular
    C = adjoint_block(1:k, (k + 1):n)

    Ỹ = _sylvester_triangular_columns(mid(A), mid(B), mid(C))
    try
        return sylvester_miyajima_enclosure(A, B, C, Ỹ)
    catch e
        e isa ArgumentError || rethrow()
    end

    all(iszero, tril(Tr, -1)) ||
        throw(ArgumentError("the triangular fallback needs zero radius below the diagonal of T"))
    return sylvester_fallback === :residual ? _sylvester_residual_ball(A, B, C, Ỹ) :
           _sylvester_triangular_direct_ball(A, B, C)
end

# ============================================================================
# Direct triangular Sylvester solver (no eigenvector decomposition)
# ============================================================================

"""
    _sylvester_triangular_columns(A, B, C)

Solve `A * X + X * B = C` column-by-column when A is lower triangular and B is
lower triangular. Uses forward substitution on triangular systems — O(m²k)
where m = size(A,1) and k = size(B,1).

For B lower triangular, column j (sweeping j = k, k-1, ..., 1):
    (A + B[j,j]*I) * x_j = c_j - Σ_{l>j} x_l * B[l,j]
Each (A + B[j,j]*I) is lower triangular.
"""
function _sylvester_triangular_columns(A::AbstractMatrix{T},
                                        B::AbstractMatrix{T},
                                        C::AbstractMatrix{T}) where T
    m = size(A, 1)
    k = size(B, 1)
    X = zeros(T, m, k)

    for j in k:-1:1
        rhs = C[:, j]
        for l in (j+1):k
            rhs = rhs .- X[:, l] .* B[l, j]
        end
        # Solve (A + B[j,j]*I) * x_j = rhs — lower triangular forward substitution
        L = A + B[j, j] * I
        X[:, j] = _forward_substitution(L, rhs)
    end
    return X
end

"""
    _forward_substitution(L, b)

Solve `L * x = b` where L is lower triangular. Standard forward substitution.
"""
function _forward_substitution(L::AbstractMatrix{T}, b::AbstractVector{T}) where T
    n = length(b)
    x = zeros(T, n)
    for i in 1:n
        s = b[i]
        for j in 1:(i-1)
            s -= L[i, j] * x[j]
        end
        x[i] = s / L[i, i]
    end
    return x
end

"""
    _sylvester_triangular_direct_ball(A, B, C)

Solve `A * X + X * B = C` rigorously in ball arithmetic when A is lower
triangular and B is lower triangular.  Returns a `BallMatrix` enclosure.

Each column solve uses [`forward_substitution`](@ref) on a lower-triangular
`BallMatrix`, guaranteeing rigorous componentwise bounds. This method never
requires eigenvector decomposition and works for arbitrarily ill-conditioned
triangular matrices (provided the diagonal entries of `A + B[j,j]*I` are nonzero).
"""
function _sylvester_triangular_direct_ball(A::Union{AbstractMatrix, BallMatrix},
        B::Union{AbstractMatrix, BallMatrix}, C::Union{AbstractMatrix, BallMatrix})
    m = size(A, 1)
    k = size(B, 1)
    CT = float(promote_type(_mid_eltype(A), _mid_eltype(B), _mid_eltype(C)))
    RT = _real_type(CT)

    A_ball, B_ball, C_ball = _ball_as(CT, A), _ball_as(CT, B), _ball_as(CT, C)

    X_mid = zeros(CT, m, k)
    X_rad = zeros(RT, m, k)

    for j in k:-1:1
        # Build RHS: c_j - Σ_{l>j} x_l * B[l,j]
        rhs = [C_ball[i, j] for i in 1:m]
        for l in (j+1):k
            b_lj = B_ball[l, j]
            for i in 1:m
                rhs[i] = rhs[i] - Ball(X_mid[i, l], X_rad[i, l]) * b_lj
            end
        end

        # Coefficient matrix (A + B[j,j]*I) is lower triangular; the diagonal is a sum of balls,
        # so its rounding is in the radius
        shift_c = copy(A_ball.c)
        shift_r = copy(A_ball.r)
        b_jj = B_ball[j, j]
        for i in 1:m
            d = A_ball[i, i] + b_jj
            shift_c[i, i] = mid(d)
            shift_r[i, i] = rad(d)
        end
        L_ball = BallMatrix(shift_c, shift_r)

        sol = forward_substitution(L_ball, BallVector(mid.(rhs), rad.(rhs)))
        for i in 1:m
            X_mid[i, j] = mid(sol[i])
            X_rad[i, j] = rad(sol[i])
        end
    end

    return BallMatrix(X_mid, X_rad)
end

# The enclosure by the residual and the column recurrence described in
# `triangular_sylvester_miyajima_enclosure`, for lower triangular ball matrices A and B.
function _sylvester_residual_ball(A::BallMatrix, B::BallMatrix, C::BallMatrix,
        X̃::AbstractMatrix)
    m = size(A, 1)
    k = size(B, 1)
    RT = _real_type(eltype(mid(A)))
    bX = BallMatrix(Matrix{eltype(mid(A))}(X̃))
    R = C - A * bX - bX * B
    absB = upper_abs(B)
    absAt = Matrix(transpose(upper_abs(A)))       # entrywise bound of the upper triangular Aᵀ

    ε = zeros(RT, k)
    for j in k:-1:1
        # lower bounds of the diagonal of A + b_jj I
        dlo = Vector{RT}(undef, m)
        for i in 1:m
            d = A[i, i] + B[j, j]
            dlo[i] = sub_down(_abs_lo(mid(d)), rad(d))
        end
        L_inv = _two_norm_from_one_inf(_triangular_inverse_bounds(dlo, absAt)...)
        isfinite(L_inv) || return _sylvester_triangular_direct_ball(A, B, C)
        s = upper_bound_norm(BallVector(R.c[:, j], R.r[:, j]), 2)
        for l in (j + 1):k
            s = add_up(s, mul_up(ε[l], absB[l, j]))
        end
        ε[j] = mul_up(L_inv, s)
    end
    return BallMatrix(Matrix(X̃), repeat(reshape(ε, 1, k), m, 1))
end

"""
    schur_sylvester_midpoint(A, B, C; prefer_complex_schur = true)

An approximate solution of `A X + X B = C` in floating point, by Schur forms of `A` and `B` and
back substitution. No bound comes with it; it is the default approximation of
[`verified_sylvester_enclosure`](@ref).
"""
function schur_sylvester_midpoint(A, B, C; prefer_complex_schur::Bool = true)
    SA = prefer_complex_schur ? schur(complex.(A)) : schur(A)
    SB = prefer_complex_schur ? schur(complex.(B)) : schur(B)
    Y = sylvester(SA.T, SB.T, -(SA.Z' * C * SB.Z))      # T_A Y + Y T_B = Q_A* C Q_B
    return SA.Z * Y * SB.Z'
end

"""
    verified_sylvester_enclosure(A, B, C; X̃ = nothing, prefer_complex_schur = true)

[`sylvester_miyajima_enclosure`](@ref) with the approximate solution computed by
[`schur_sylvester_midpoint`](@ref) when `X̃` is not given. It throws `ArgumentError` when the
hypotheses of that enclosure are not proved, as happens for a defective `A` or `B`.
"""
function verified_sylvester_enclosure(A, B, C; X̃ = nothing,
        prefer_complex_schur::Bool = true)
    X̃ === nothing && (X̃ = schur_sylvester_midpoint(A, B, C; prefer_complex_schur))
    return sylvester_miyajima_enclosure(A, B, C, X̃)
end
