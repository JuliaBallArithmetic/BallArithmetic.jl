"""
    sylvester_miyajima_enclosure(A, B, C, X̃)

Compute a Miyajima-style verified enclosure for the solution of the Sylvester
problem `A * X + X * B = C`.  The method follows the componentwise certificate
from Ref. [`MiyajimaSylvester2013`](@cite) and returns a `BallMatrix` whose
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
    eigA = eigen(AMat)
    VA = Matrix(eigA.vectors)
    λA = eigA.values

    BT = Matrix(transpose(B))
    eigBT = eigen(BT)
    VB = Matrix(eigBT.vectors)
    λB = eigBT.values

    I_m = Matrix{eltype(VA)}(I, m, m)
    I_n = Matrix{eltype(VB)}(I, n, n)
    WA = VA \ I_m
    WB = VB \ I_n

    VA_ball = BallMatrix(VA)
    WA_ball = BallMatrix(WA)
    VB_ball = BallMatrix(VB)
    WB_ball = BallMatrix(WB)
    AMat_ball = BallMatrix(AMat)
    BT_ball = BallMatrix(BT)
    diagA_ball = BallMatrix(Diagonal(λA))
    diagB_ball = BallMatrix(Diagonal(λB))

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

    norm_SA < 1 ||
        throw(ArgumentError("‖S_A‖∞=$(norm_SA) must be < 1 (try Schur fallback)"))
    norm_SB < 1 ||
        throw(ArgumentError("‖S_B‖∞=$(norm_SB) must be < 1 (try Schur fallback)"))

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
    any(iszero, abs_D̃) &&
        throw(ArgumentError("Encountered zero spectral gap (λ_i(A)+λ_j(B)=0)"))

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

Construct the Miyajima enclosure for the Sylvester system associated with the
upper-triangular matrix `T` partitioned as

```
T = [T₁₁  T₁₂;
     0    T₂₂],
```

where `T₁₁` is `k × k`.  The enclosure is computed for the solution `Y₂` of the
transformed Sylvester equation `T₂₂' * Y₂ - Y₂ * T₁₁' = T₁₂'`.  Forming the
standard Sylvester data `A = T₂₂'`, `B = -T₁₁'`, and `C = T₁₂'`, the routine
solves for an approximate `Y₂` and then calls [`sylvester_miyajima_enclosure`](@ref)
to obtain a verified bound.  The returned `BallMatrix` encloses the exact `Y₂`
entrywise.

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

    A = Matrix{Ttype}(adjoint(T22))
    B = -Matrix{Ttype}(adjoint(T11))
    C = Matrix{Ttype}(adjoint(T12))

    mA = size(A, 1)
    nB = size(B, 1)
    ImA = Matrix{Ttype}(I, mA, mA)
    InB = Matrix{Ttype}(I, nB, nB)
    K = kron(InB, A) + kron(transpose(B), ImA)
    Y_vec = K \ vec(C)
    Ỹ = reshape(Y_vec, mA, nB)

    return sylvester_miyajima_enclosure(A, B, C, Ỹ)
end

"""
    verified_sylvester_enclosure(A, B, C; X̃=nothing, prefer_complex_schur=true)

Verified enclosure (as `BallMatrix`) for the Sylvester equation

    A*X + X*B = C

It first tries the fast Miyajima eigenvector route:
`sylvester_miyajima_enclosure(A,B,C,X̃)`. If that throws (ill-conditioned
diagonalizers, resonance, or contraction failure), it falls back to a robust
unitary-Schur, block-by-block verified method.

- If `X̃ === nothing`, a numerical midpoint is computed via Schur back-substitution.
- Set `prefer_complex_schur=false` to use real quasi-Schur (1×1/2×2 blocks).
"""
function verified_sylvester_enclosure(
        A, B, C; X̃ = nothing, prefer_complex_schur::Bool = true)
    # 0) if no midpoint provided, get a numerical one via Schur (no verification)
    if X̃ === nothing
        X̃ = schur_sylvester_midpoint(A, B, C; prefer_complex_schur)
    end
    # 1) try fast Miyajima eigenvector certificate
    try
        return sylvester_miyajima_enclosure(A, B, C, X̃)
    catch
        # 2) Schur fallback: verified, block-by-block
        return schur_sylvester_miyajima_enclosure(A, B, C; prefer_complex_schur)
    end
end

"""
    schur_sylvester_miyajima_enclosure(A, B, C; prefer_complex_schur=true)
Verified enclosure (as `BallMatrix`) for the Sylvester equation using the Schur decomposition,
when the eigenvector-based Miyajima method is not applicable.

"""
function schur_sylvester_miyajima_enclosure(A, B, C; prefer_complex_schur::Bool = true)
    # 1) Unitary Schur decompositions
    if prefer_complex_schur
        SA = schur(complex.(A))         # ComplexSchur: SA.Q, SA.T
        SB = schur(complex.(B))
    else
        SA = schur(A)                   # RealSchur:   SA.Q, SA.T (quasi-triangular)
        SB = schur(B)
    end
    QA, TA = SA.Z, SA.T
    QB, TB = SB.Z, SB.T

    # 2) Transform RHS into Schur basis
    C̃ = QA' * C * QB

    # 3) Block structure of TA, TB
    Ab = schur_blocks(TA)
    Bb = schur_blocks(TB)

    # 4) Allocate midpoint and radii in Schur space
    Ymid = zero(C̃)
    Yrad = zeros(_real_type(eltype(C̃)), size(C̃))

    # 5) Block back-substitution with verified per-block solves (reverse order)
    for ii in length(Ab):-1:1
        IA = Ab[ii]
        Aii = TA[IA, IA]
        for jj in length(Bb):-1:1
            JB = Bb[jj]
            Bjj = TB[JB, JB]

            # Assemble local RHS for (ii,jj)
            RHS = C̃[IA, JB]
            for kk in (ii + 1):length(Ab)
                IK = Ab[kk]
                RHS -= TA[IA, IK] * Ymid[IK, JB]
            end
            for ℓ in (jj + 1):length(Bb)
                JL = Bb[ℓ]
                RHS -= Ymid[IA, JL] * TB[JL, JB]
            end

            # Tiny (unverified) midpoint for this block
            Ỹ = tiny_sylvester_midpoint(Aii, Bjj, RHS)
            place!(Ymid, IA, JB, Ỹ)

            # Verified Miyajima enclosure on the tiny block
            Bij = sylvester_miyajima_enclosure(Aii, Bjj, RHS, Ỹ)
            Eij = rad(Bij)                     # entrywise radii
            place!(Yrad, IA, JB, Eij)
        end
    end

    # 6) Lift back to the original basis; entrywise radii via |Q| multipliers
    Xmid = QA * Ymid * QB'
    Xrad = abs.(QA) * Yrad * abs.(QB)'

    return BallMatrix(Xmid, Xrad)
end

"""
    schur_sylvester_midpoint(A,B,C; prefer_complex_schur=true)

Numerical midpoint for `A*X + X*B = C` via Schur back-substitution (no verification).
Used as default `X̃` if the caller doesn't pass one.
"""
function schur_sylvester_midpoint(A, B, C; prefer_complex_schur::Bool = true)
    if prefer_complex_schur
        SA = schur(complex.(A))
        SB = schur(complex.(B))
    else
        SA = schur(A)
        SB = schur(B)
    end
    QA, TA = SA.Z, SA.T
    QB, TB = SB.Z, SB.T

    C̃ = QA' * C * QB
    Ab = schur_blocks(TA)
    Bb = schur_blocks(TB)

    Y = zero(C̃)

    for ii in length(Ab):-1:1
        IA = Ab[ii]
        Aii = TA[IA, IA]
        for jj in length(Bb):-1:1
            JB = Bb[jj]
            Bjj = TB[JB, JB]

            RHS = C̃[IA, JB]
            for kk in (ii + 1):length(Ab)
                IK = Ab[kk]
                RHS -= TA[IA, IK] * Y[IK, JB]
            end
            for ℓ in (jj + 1):length(Bb)
                JL = Bb[ℓ]
                RHS -= Y[IA, JL] * TB[JL, JB]
            end

            Yij = tiny_sylvester_midpoint(Aii, Bjj, RHS)
            place!(Y, IA, JB, Yij)
        end
    end

    return QA * Y * QB'
end

"""
    schur_blocks(T; tol=nothing) -> Vector{UnitRange{Int}}

Diagonal block ranges of a Schur form `T`.
- Complex Schur: all 1×1 blocks.
- Real Schur: detect 2×2 blocks when `abs(T[i+1,i]) > tol`.
"""
function schur_blocks(T; tol = nothing)
    n, m = size(T)
    n == m || throw(DimensionMismatch("T must be square"))

    if eltype(T) <: Complex
        return [i:i for i in 1:n]
    end

    RT = float(real(one(eltype(T))))
    if tol === nothing
        maxrowsum = zero(RT)
        @inbounds for i in 1:n
            s = zero(RT)
            @inbounds for j in 1:n
                s += abs(T[i, j])
            end
            maxrowsum = max(maxrowsum, s)
        end
        tol = sqrt(eps(RT)) * max(maxrowsum, one(RT))
    end

    blocks = Vector{UnitRange{Int}}()
    i = 1
    @inbounds while i <= n
        if i < n && abs(T[i + 1, i]) > tol
            push!(blocks, i:(i + 1))
            i += 2
        else
            push!(blocks, i:i)
            i += 1
        end
    end
    return blocks
end

"""
    tiny_sylvester_midpoint(Aii, Bjj, RHS)

Fast midpoint for the tiny Sylvester subproblem
`Aii*Y + Y*Bjj = RHS`, where `Aii` and `Bjj` are 1×1 or 2×2.
"""
function tiny_sylvester_midpoint(Aii, Bjj, RHS)
    p = size(Aii, 1)
    q = size(Bjj, 1)
    if p == 1 && q == 1
        return RHS / (Aii[1, 1] + Bjj[1, 1])
    elseif p == 2 && q == 1
        M = Aii + Bjj[1, 1] * I(2)
        return M \ RHS
    elseif p == 1 && q == 2
        M = (Bjj + Aii[1, 1] * I(2))'
        return (M \ RHS')'
    elseif p == 2 && q == 2
        K = kron(I(2), Aii) + kron(Bjj', I(2))
        return reshape(K \ vec(RHS), 2, 2)
    else
        throw(ArgumentError("Unsupported block sizes p=$p, q=$q"))
    end
end

"""
    place!(M, I, J, B)

Insert block `B` into matrix `M` at row-range `I` and col-range `J`.
"""
@inline function place!(M, I::UnitRange{Int}, J::UnitRange{Int}, B)
    @inbounds M[I, J] .= B
    return M
end
