# The exported Takagi decomposition with error bounds. The method is in `rump_ogita_2024.jl`; the
# working arithmetic and the residual are shared with `verified_lu.jl`.

"""
    VerifiedTakagiResult{UM, SV, RT}

Result of [`verified_takagi`](@ref).

# Fields
- `U::UM`: ball matrix containing the unitary factor
- `Σ::SV`: balls containing the diagonal of `Σ`, the singular values of `A`, in decreasing order
- `success::Bool`: whether the verification succeeded; when `false` the radii are infinite and
  nothing is proved
- `residual_norm::RT`: an upper bound of `‖ŨΣ̃Ũᵀ − A‖_∞ / ‖A‖_∞` over all `Ũ ∈ U`, `Σ̃ ∈ Σ`

# Statement
When `success` is `true`, the symmetric part `(A + Aᵀ)/2` of the input, which is the input when
it is symmetric (`Aᵀ = A`, not `A* = A`), is nonsingular with simple singular values and equals
`UΣUᵀ` with `U` unitary and `Σ` diagonal with positive entries, `U` and the diagonal of `Σ` in
the balls returned. `U` is determined up to the sign of each column.

# Reference
S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 9.
"""
struct VerifiedTakagiResult{UM<:BallMatrix, SV<:AbstractVector, RT<:Real}
    U::UM
    Σ::SV
    success::Bool
    residual_norm::RT
end

"""
    verified_takagi(A::AbstractMatrix{<:Complex}; precision_bits = 256, use_bigfloat = true)

Inclusions of the factors of the Takagi decomposition `A = UΣUᵀ` of a complex symmetric matrix,
by the third method of Section 9 of Rump and Ogita (2024), the one through the real symmetric
matrix `[E F; F −E]` for `A = E + iF`; the method is described at
[`_rumpogita2024_takagi`](@ref). Returns a [`VerifiedTakagiResult`](@ref).

`success` is `false` when the singular values are not proved positive and simple, which
includes every matrix with a multiple singular value, the identity for one: the decomposition is
not unique there and the method does not apply.

When `A` is not symmetric a warning is issued (above a rounding-level threshold) and the
statement is for its symmetric part `(A + Aᵀ)/2`, enclosed in ball arithmetic.

# Arguments
- `use_bigfloat`: with `true`, the default, the computation runs in BigFloat at `precision_bits`;
  with `false` it runs in Float64. A BigFloat input is always treated in BigFloat.
- `precision_bits`: the BigFloat precision. An input with more digits is rounded to it and the
  rounding is enclosed.

# Example
```julia
B = randn(ComplexF64, 20, 20); A = B + transpose(B)
r = verified_takagi(A; use_bigfloat = false)
r.success      # A = U Σ Uᵀ with U in r.U and the diagonal of Σ in r.Σ
```

# Reference
S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 9.
"""
function verified_takagi(A::AbstractMatrix{Complex{S}};
                         precision_bits::Int=256,
                         use_bigfloat::Bool=true) where S<:Union{Float64, BigFloat}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square"))
    sym_error = maximum(abs.(A - transpose(A)); init = zero(S))
    if sym_error > 100 * eps(S) * maximum(abs.(A); init = zero(S))
        @warn "Matrix A is not complex symmetric (error = $sym_error)"
    end
    bigfloat = use_bigfloat || S === BigFloat
    return _decomposition_arithmetic(bigfloat, precision_bits) do
        Sb = _symmetric_part_ball(_decomposition_input(A, bigfloat), identity)
        T = eltype(rad(Sb))
        r = _rumpogita2024_takagi(Sb)
        r === nothing && return VerifiedTakagiResult(_unverified_ball(Sb, n, n),
            fill(Ball(T(NaN), T(Inf)), n), false, T(Inf))
        D = BallMatrix(Matrix{Complex{T}}(Diagonal(mid.(r.sigma))), Matrix{T}(Diagonal(rad.(r.sigma))))
        Ut = BallMatrix(Matrix(transpose(mid(r.U))), Matrix(transpose(rad(r.U))))
        residual_norm = _relative_residual_bound(r.U * D * Ut - Sb, Sb)
        return VerifiedTakagiResult(r.U, r.sigma, true, residual_norm)
    end
end

# Stub for Double64 extension
"""
    verified_takagi_double64(A; precision_bits=256, method=:real_compound)

Fast verified Takagi decomposition using Double64 oracle. Requires DoubleFloats.jl.
"""
function verified_takagi_double64 end

# Stub for MultiFloat extension
"""
    verified_takagi_multifloat(A; precision_bits=256, method=:real_compound, float_type=Float64x4)

Fast verified Takagi decomposition using MultiFloat oracle. Requires MultiFloats.jl.
"""
function verified_takagi_multifloat end

export VerifiedTakagiResult, verified_takagi, verified_takagi_double64, verified_takagi_multifloat
