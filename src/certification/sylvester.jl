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
    T22 = @view Tmat[k+1:n, k+1:n]
    T12 = @view Tmat[1:k, k+1:n]

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
