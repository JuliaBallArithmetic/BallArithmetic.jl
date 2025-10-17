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
    WA = inv(VA)

    BT = Matrix(transpose(B))
    eigBT = eigen(BT)
    VB = Matrix(eigBT.vectors)
    λB = eigBT.values
    WB = inv(VB)

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
    R_W = WA_ball * R * transpose(WB)

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
    solve_leading_triangular_sylvester(A, B, C, k = 1)

Solve the Sylvester equation `A₁₁ * X + X * B₁₁ = C₁₁` associated with the
leading diagonal block of size `k × k` of the upper-triangular matrices `A` and
`B`. The right-hand side is given by the matching leading block of `C`.

The matrices `A` and `B` must be square and upper triangular, and `C` must have
dimensions compatible with the Sylvester equation `A * X + X * B = C`.
The returned matrix corresponds to the `k × k` leading block of the solution
`X` for this subsystem. The function throws an `ArgumentError` if any spectral
gap `Aᵢᵢ + Bⱼⱼ` for the selected block vanishes, as the subsystem would have no
unique solution.
"""
function solve_leading_triangular_sylvester(A::AbstractMatrix,
        B::AbstractMatrix, C::AbstractMatrix, k::Integer = 1)
    mA, nA = size(A)
    mA == nA || throw(DimensionMismatch("A must be square"))
    mB, nB = size(B)
    mB == nB || throw(DimensionMismatch("B must be square"))
    size(C) == (mA, nB) || throw(DimensionMismatch("C must be of size ($mA, $nB)"))

    0 < k <= min(mA, nB) || throw(ArgumentError("k must satisfy 1 ≤ k ≤ $(min(mA, nB))"))

    istriu(Matrix(A)) || throw(ArgumentError("A must be upper triangular"))
    istriu(Matrix(B)) || throw(ArgumentError("B must be upper triangular"))

    block_type = promote_type(eltype(A), eltype(B), eltype(C))

    A11 = Matrix{block_type}(A[1:k, 1:k])
    B11 = Matrix{block_type}(B[1:k, 1:k])
    C11 = Matrix{block_type}(C[1:k, 1:k])

    for i in 1:k, j in 1:k
        gap = A11[i, i] + B11[j, j]
        iszero(gap) && throw(ArgumentError("Encountered zero spectral gap at ($(i), $(j))"))
    end

    if k == 1
        return C11 ./ (A11[1, 1] + B11[1, 1])
    end

    Ik = Matrix{block_type}(I, k, k)
    K = kron(Ik, A11) + kron(transpose(B11), Ik)
    X_vec = K \ vec(C11)
    return reshape(X_vec, k, k)
end
