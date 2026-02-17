"""
    sylvester_miyajima_enclosure(A, B, C, X̃)

Compute a Miyajima-style verified enclosure for the solution of the Sylvester
problem `A * X + X * B = C`.  The method follows the componentwise certificate
from Ref. [MiyajimaSylvester2013](@cite) and returns a `BallMatrix` whose
midpoint is the supplied approximation `X̃` and whose radii enclose the exact
solution entrywise.

The routine raises an error when the spectral gaps `λ_i(A) + λ_j(B)` vanish or
when the contraction bound is not satisfied.
"""
function sylvester_miyajima_enclosure(A::AbstractMatrix, B::AbstractMatrix,
        C::AbstractMatrix, X̃::AbstractMatrix)
    m, mA = size(A)
    m == mA || throw(DimensionMismatch("A must be square"))
    nB, n = size(B)
    nB == n || throw(DimensionMismatch("B must be square"))
    size(C) == (m, n) || throw(DimensionMismatch("C must be of size ($m, $n)"))
    size(X̃) == (m, n) || throw(DimensionMismatch("X̃ must be of size ($m, $n)"))

    realtype = promote_type(_real_type(eltype(A)), _real_type(eltype(B)),
        _real_type(eltype(C)), _real_type(eltype(X̃)))

    AMat = Matrix(A)
    if (istriu(AMat) || istril(AMat)) && _has_distinct_diagonal(AMat, eps(realtype))
        VA, WA, λA = triangular_eigenvectors(AMat; tol=eps(realtype))
    else
        eigA = eigen(AMat)
        VA = Matrix(eigA.vectors)
        λA = eigA.values
        WA = inv(VA)
    end

    BT = Matrix(transpose(B))
    if (istriu(BT) || istril(BT)) && _has_distinct_diagonal(BT, eps(realtype))
        VB, WB, λB = triangular_eigenvectors(BT; tol=eps(realtype))
    else
        eigBT = eigen(BT)
        VB = Matrix(eigBT.vectors)
        λB = eigBT.values
        WB = inv(VB)
    end

    VA_ball = BallMatrix(VA)
    WA_ball = BallMatrix(WA)
    VB_ball = BallMatrix(VB)
    WB_ball = BallMatrix(WB)
    AMat_ball = BallMatrix(AMat)
    BT_ball = BallMatrix(BT)
    diagA_ball = BallMatrix(Diagonal(λA))
    diagB_ball = BallMatrix(Diagonal(λB))

    I_m = Matrix{eltype(WA)}(I, m, m)
    I_n = Matrix{eltype(WB)}(I, n, n)

    SA = BallMatrix(I_m) - WA_ball * VA_ball
    SB = BallMatrix(I_n) - WB_ball * VB_ball

    RA = WA_ball * (VA_ball * diagA_ball - AMat_ball * VA_ball)
    RB = WB_ball * (VB_ball * diagB_ball - BT_ball * VB_ball)

    abs_RA = _abs_sup_matrix(RA, realtype)
    abs_SA = _abs_sup_matrix(SA, realtype)
    abs_RB = _abs_sup_matrix(RB, realtype)
    abs_SB = _abs_sup_matrix(SB, realtype)

    norm_RA = setrounding(realtype, RoundUp) do
        _matrix_norm_inf(abs_RA)
    end
    norm_SA = setrounding(realtype, RoundUp) do
        _matrix_norm_inf(abs_SA)
    end
    norm_RB = setrounding(realtype, RoundUp) do
        _matrix_norm_inf(abs_RB)
    end
    norm_SB = setrounding(realtype, RoundUp) do
        _matrix_norm_inf(abs_SB)
    end

    norm_SA < 1 || throw(ArgumentError("\u2225S_A\u2225_\u221E must be < 1"))
    norm_SB < 1 || throw(ArgumentError("\u2225S_B\u2225_\u221E must be < 1"))

    TA = setrounding(realtype, RoundUp) do
        abs_RA .+ (norm_RA / (1 - norm_SA)) .* abs_SA
    end
    TB = setrounding(realtype, RoundUp) do
        abs_RB .+ (norm_RB / (1 - norm_SB)) .* abs_SB
    end

    E = ones(realtype, m, n)
    T_ball = BallMatrix(TA) * BallMatrix(E) + BallMatrix(E) * BallMatrix(transpose(TB))
    T = _abs_sup_matrix(T_ball, realtype)

    λA_mat = reshape(λA, m, 1)
    λB_row = reshape(λB, 1, n)
    D̃ = λA_mat .+ λB_row
    abs_D̃ = abs.(D̃)
    minimum(abs_D̃) > eps(realtype) || throw(ArgumentError("Encountered zero spectral gap"))

    T_D = setrounding(realtype, RoundUp) do
        T ./ abs_D̃
    end

    X̃Mat = Matrix(X̃)
    BMat = Matrix(B)
    CMat = Matrix(C)

    X̃_ball = BallMatrix(X̃Mat)
    BMat_ball = BallMatrix(BMat)
    CMat_ball = BallMatrix(CMat)

    R = AMat_ball * X̃_ball + X̃_ball * BMat_ball - CMat_ball
    WB_T_ball = BallMatrix(Matrix(transpose(WB)))
    R_W = WA_ball * R * WB_T_ball

    R_W_abs = _abs_sup_matrix(R_W, realtype)

    R_D = setrounding(realtype, RoundUp) do
        R_W_abs ./ abs_D̃
    end

    norm_TD = setrounding(realtype, RoundUp) do
        _entrywise_max_norm(T_D)
    end
    norm_RD = setrounding(realtype, RoundUp) do
        _entrywise_max_norm(R_D)
    end

    norm_TD < 1 || throw(ArgumentError("Entrywise max norm of T_D must be < 1"))

    U = setrounding(realtype, RoundUp) do
        R_D .+ (norm_RD / (1 - norm_TD)) .* T_D
    end

    abs_VA = abs.(VA)
    abs_VB = abs.(VB)
    Xε = setrounding(realtype, RoundUp) do
        abs_VA * U * transpose(abs_VB)
    end

    Xε = max.(Xε, zero(realtype))
    return BallMatrix(X̃Mat, Xε)
end

function _real_type(::Type{T}) where {T <: Real}
    return float(T)
end

function _real_type(::Type{Complex{T}}) where {T <: Real}
    return float(T)
end

function _abs_sup_matrix(M::BallMatrix, ::Type{T}) where {T <: AbstractFloat}
    setrounding(T, RoundUp) do
        result = Matrix{T}(undef, size(M))
        for i in axes(M, 1), j in axes(M, 2)
            result[i, j] = sup(abs(M[i, j]))
        end
        result
    end
end

function _entrywise_max_norm(M)
    T = _real_type(eltype(M))
    max_val = zero(T)
    for v in M
        max_val = max(max_val, abs(v))
    end
    return max_val
end

function _matrix_norm_inf(M)
    T = _real_type(eltype(M))
    max_sum = zero(T)
    for i in axes(M, 1)
        row_sum = zero(T)
        for j in axes(M, 2)
            row_sum += abs(M[i, j])
        end
        max_sum = max(max_sum, row_sum)
    end
    return max_sum
end

"""
    triangular_sylvester_miyajima_enclosure(T, k)

Construct a verified enclosure for the Sylvester system associated with the
upper-triangular matrix `T` partitioned as

```
T = [T₁₁  T₁₂;
     0    T₂₂],
```

where `T₁₁` is `k × k`.  The enclosure is computed for the solution `Y₂` of the
transformed Sylvester equation `T₂₂' * Y₂ - Y₂ * T₁₁' = T₁₂'`.  Forming the
standard Sylvester data `A = T₂₂'`, `B = -T₁₁'`, and `C = T₁₂'`, the routine
first attempts a Miyajima-style eigenvector-based enclosure, and falls back to
a direct column-by-column triangular solve in ball arithmetic when the
eigenvector approach fails (e.g. for large ill-conditioned triangular matrices
where the eigenvector condition number is too large).

The matrix `T` must be square and upper triangular, and the block size `k`
must satisfy `1 ≤ k < size(T, 1)`.
"""
function triangular_sylvester_miyajima_enclosure(T::AbstractMatrix, k::Integer)
    n, m = size(T)
    n == m || throw(DimensionMismatch("T must be square"))
    1 <= k < n || throw(ArgumentError("k must satisfy 1 ≤ k < $n"))

    Ttype = promote_type(eltype(T), Float64)
    Tmat = Matrix{Ttype}(T)
    istriu(Tmat) || throw(ArgumentError("T must be upper triangular"))

    T11 = @view Tmat[1:k, 1:k]
    T22 = @view Tmat[(k + 1):n, (k + 1):n]
    T12 = @view Tmat[1:k, (k + 1):n]

    A = Matrix{Ttype}(adjoint(T22))   # lower triangular
    B = -Matrix{Ttype}(adjoint(T11))  # lower triangular
    C = Matrix{Ttype}(adjoint(T12))

    # Approximate solution via column-by-column triangular solve (O(n²k))
    Ỹ = _sylvester_triangular_columns(A, B, C)

    # Try Miyajima eigenvector-based enclosure first (tighter bounds)
    try
        return sylvester_miyajima_enclosure(A, B, C, Ỹ)
    catch e
        if !(e isa ArgumentError)
            rethrow()
        end
    end

    # Fallback: direct column-by-column solve in ball arithmetic
    return _sylvester_triangular_direct_ball(A, B, C)
end

"""
    triangular_sylvester_miyajima_enclosure(T_ball::BallMatrix, k::Integer)

Miyajima enclosure for the Sylvester system when the triangular matrix `T` is
given as a `BallMatrix` (midpoint + radii).

The midpoint `mid(T_ball)` is used to solve the Sylvester equation via the
scalar method. The radii of `T_ball` produce a first-order perturbation
bound on the solution `Y`, which inflates the returned enclosure.

# Algorithm
1. Solve on midpoint: `Y_mid = triangular_sylvester_miyajima_enclosure(mid(T_ball), k)`
2. Compute separation: `sep = min_{i,j} |T₁₁[i,i] - T₂₂[j,j]|` (lower-bounded rigorously)
3. First-order perturbation bound on Y from T_ball radii:
   `δY ≤ sep⁻¹ · (‖ΔT₂₂‖·‖Y‖ + ‖ΔT₁₁‖·‖Y‖ + ‖ΔT₁₂‖)`
   where `ΔT_ij` are the radius sub-blocks
4. Inflate: `BallMatrix(mid(Y_mid), rad(Y_mid) .+ δY)`

The matrix `T_ball` must be square, and `mid(T_ball)` must be upper triangular.
"""
function triangular_sylvester_miyajima_enclosure(T_ball::BallMatrix, k::Integer)
    n = size(T_ball, 1)
    n == size(T_ball, 2) || throw(DimensionMismatch("T_ball must be square"))
    1 <= k < n || throw(ArgumentError("k must satisfy 1 ≤ k < $n"))

    T_mid = mid(T_ball)
    T_rad = rad(T_ball)

    # Step 1: Solve on midpoint
    Y_mid_ball = triangular_sylvester_miyajima_enclosure(T_mid, k)

    # Step 2: Compute separation from diagonal entries (upper triangular → eigenvalues on diagonal)
    RT = real(eltype(T_mid))
    T11_diag = diag(T_mid[1:k, 1:k])
    T22_diag = diag(T_mid[(k+1):n, (k+1):n])

    # Rigorous lower bound on sep: min|λ_i(T11) - λ_j(T22)| - radii of diagonals
    sep = convert(RT, Inf)
    for i in 1:k
        for j in 1:(n-k)
            diff = abs(T11_diag[i] - T22_diag[j])
            # Subtract the radii contribution for rigorous lower bound
            diff_lower = diff - T_rad[i, i] - T_rad[k+j, k+j]
            sep = min(sep, diff_lower)
        end
    end

    if sep <= zero(RT)
        @warn "triangular_sylvester_miyajima_enclosure(BallMatrix): " *
              "separation ≤ 0 after accounting for radii. Perturbation bound is infinite."
        return Y_mid_ball
    end

    # Step 3: First-order perturbation bound
    # For the Sylvester equation T11 Y - Y T22 = T12,
    # the first-order perturbation gives:
    # ‖δY‖ ≤ sep⁻¹ · (‖ΔT11‖·‖Y‖ + ‖Y‖·‖ΔT22‖ + ‖ΔT12‖)
    dT11 = BallMatrix(T_rad[1:k, 1:k])
    dT22 = BallMatrix(T_rad[(k+1):n, (k+1):n])
    dT12 = BallMatrix(T_rad[1:k, (k+1):n])

    norm_dT11 = upper_bound_L2_opnorm(dT11)
    norm_dT22 = upper_bound_L2_opnorm(dT22)
    norm_dT12 = upper_bound_L2_opnorm(dT12)
    norm_Y = upper_bound_L2_opnorm(Y_mid_ball)

    delta_Y_norm = (norm_dT11 * norm_Y + norm_Y * norm_dT22 + norm_dT12) / sep

    if norm_Y > zero(RT) && delta_Y_norm / norm_Y > RT(0.1)
        @warn "triangular_sylvester_miyajima_enclosure(BallMatrix): " *
              "large relative perturbation δY/Y = $(Float64(delta_Y_norm / norm_Y)). " *
              "Bound may be loose."
    end

    # Step 4: Inflate radii
    m_Y = size(Y_mid_ball, 1)
    n_Y = size(Y_mid_ball, 2)
    inflated_rad = rad(Y_mid_ball) .+ fill(delta_Y_norm, m_Y, n_Y)

    return BallMatrix(mid(Y_mid_ball), inflated_rad)
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
function _sylvester_triangular_direct_ball(A::AbstractMatrix, B::AbstractMatrix,
                                            C::AbstractMatrix)
    m = size(A, 1)
    k = size(B, 1)

    CT = promote_type(eltype(A), eltype(B), eltype(C))
    RT = _real_type(CT)

    A_ball = BallMatrix(Matrix{CT}(A))
    B_ball = BallMatrix(Matrix{CT}(B))
    C_ball = BallMatrix(Matrix{CT}(C))

    X_mid = zeros(CT, m, k)
    X_rad = zeros(RT, m, k)

    for j in k:-1:1
        # Build RHS: c_j - Σ_{l>j} x_l * B[l,j]
        rhs = BallVector(C_ball.c[:, j], C_ball.r[:, j])
        for l in (j+1):k
            x_l = BallVector(X_mid[:, l], X_rad[:, l])
            rhs = rhs - x_l * B_ball[l, j]
        end

        # Coefficient matrix (A + B[j,j]*I) is lower triangular
        shift_c = copy(A_ball.c)
        shift_r = copy(A_ball.r)
        b_jj = B_ball[j, j]
        for i in 1:m
            shift_c[i, i] += mid(b_jj)
            shift_r[i, i] += rad(b_jj)
        end
        L_ball = BallMatrix(shift_c, shift_r)

        sol = forward_substitution(L_ball, rhs)
        for i in 1:m
            X_mid[i, j] = mid(sol[i])
            X_rad[i, j] = rad(sol[i])
        end
    end

    return BallMatrix(X_mid, X_rad)
end

# Input: A,B,C
# 1. [Schur] A = Q TA Q*, B = Z TB Z*          // real Schur if real
# 2. [Approx solve] Solve TA Yhat + Yhat TB = Q* C Z  (back/forward substitution)
# 3. [Residual] R = (Q* C Z) - (TA Yhat + Yhat TB)
# 4. [Preconditioner M] define M(·) as: solve TA Δ + Δ TB = (·)
# 5. [Interval radius] pick initial Δ0 (e.g. scaled ||R|| bound)
# 6. loop:
#       // Krawczyk interval evaluation, outward rounding
#       Kmid = Yhat - M( (TA Yhat + Yhat TB) - (Q* C Z) )
#       E = I - M∘L   // realized by two triangular solves inside interval arithmetic
#       Kset = Kmid + E([-Δ, +Δ])
#       if Kset ⊆ (Yhat + (-Δ, +Δ)) then
#           return verified enclosure for Y*, hence X* = Q Y* Z*
#       else
#           shrink Δ or recompute using refined Yhat; repeat

function sylvester_krawczyk_enclosure(A::AbstractMatrix,
        B::AbstractMatrix, C::AbstractMatrix, X̃::AbstractMatrix;
        maxiter::Int = 10, tol::Real = 1e-12)
    TA, QA, _ = schur(Matrix(A))
    TB, QB, _ = schur(Matrix(B))

    tildeC = adjoint(QA) * C * QB
    Yhat = sylvester(TA, TB, tildeC)
    R = tildeC - (TA * Yhat + Yhat * TB)

    throw(ErrorException("sylvester_krawczyk_enclosure is not yet implemented"))
end
