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

    I_m = Matrix{eltype(WA)}(I, m, m)
    I_n = Matrix{eltype(WB)}(I, n, n)

    SA = I_m - WA * VA
    SB = I_n - WB * VB

    RA = WA * (VA * Diagonal(λA) - AMat * VA)
    RB = WB * (VB * Diagonal(λB) - BT * VB)

    norm_RA = _matrix_norm_inf(RA)
    norm_SA = _matrix_norm_inf(SA)
    norm_RB = _matrix_norm_inf(RB)
    norm_SB = _matrix_norm_inf(SB)

    norm_SA < 1 || throw(ArgumentError("\u2225S_A\u2225_\u221E must be < 1"))
    norm_SB < 1 || throw(ArgumentError("\u2225S_B\u2225_\u221E must be < 1"))

    abs_RA = abs.(RA)
    abs_SA = abs.(SA)
    abs_RB = abs.(RB)
    abs_SB = abs.(SB)

    TA = setrounding(realtype, RoundUp) do
        abs_RA .+ (norm_RA / (1 - norm_SA)) .* abs_SA
    end
    TB = setrounding(realtype, RoundUp) do
        abs_RB .+ (norm_RB / (1 - norm_SB)) .* abs_SB
    end

    E = ones(realtype, m, n)
    T = setrounding(realtype, RoundUp) do
        TA * E .+ E * transpose(TB)
    end

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

    R = setrounding(realtype, RoundUp) do
        AMat * X̃Mat .+ X̃Mat * BMat .- CMat
    end
    R_W = WA * R * transpose(WB)

    R_D = setrounding(realtype, RoundUp) do
        abs.(R_W) ./ abs_D̃
    end

    norm_TD = _entrywise_max_norm(T_D)
    norm_RD = _entrywise_max_norm(R_D)

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
