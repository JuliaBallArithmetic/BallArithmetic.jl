# The exported QR decomposition with error bounds. The method is in `rump_ogita_2024.jl`; the
# working arithmetic and the residual are shared with `verified_lu.jl`.

"""
    VerifiedQRResult{QM, RM, RT}

Result of [`verified_qr`](@ref).

# Fields
- `Q::QM`: ball matrix containing the factor with orthonormal columns
- `R::RM`: ball matrix containing the upper triangular factor
- `success::Bool`: whether the verification succeeded; when `false` the radii are infinite and
  nothing is proved
- `residual_norm::RT`: an upper bound of `‖Q̃R̃ − A‖_∞ / ‖A‖_∞` over all `Q̃ ∈ Q`, `R̃ ∈ R`
- `orthogonality_defect::RT`: an upper bound of `‖Q̃*Q̃ − I‖₂` over all `Q̃ ∈ Q`; the factor
  itself has orthonormal columns, and this measures the width of the enclosure

# Statement
When `success` is `true`, `A` has full rank and `A = QR` with `Q` having orthonormal columns and
`R` upper triangular with positive diagonal entries, `Q` and `R` in the ball matrices returned.

# Reference
S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 5.
"""
struct VerifiedQRResult{QM<:BallMatrix, RM<:BallMatrix, RT<:Real}
    Q::QM
    R::RM
    success::Bool
    residual_norm::RT
    orthogonality_defect::RT
end

"""
    verified_qr(A::AbstractMatrix; compute_full_Q = false, precision_bits = 256,
                use_bigfloat = true)

Inclusions of the factors of the QR decomposition of the `m × n` matrix `A`, real or complex, by
Section 5 of Rump and Ogita (2024); the method is described at [`_rumpogita2024_qr`](@ref).
Returns a [`VerifiedQRResult`](@ref): when `success` is `true`, `A` has full rank and `A = QR`
with `R` upper triangular with positive diagonal entries.

For `m ≥ n` the decomposition is the economy-size one, `Q` of size `m × n` and `R` of size
`n × n`, which is unique. With `compute_full_Q = true`, `Q` is `m × m` unitary and `R` is
`m × n`; the last `m − n` columns of `Q` are an orthonormal basis of the orthogonal complement of
the range, which is not unique, and the statement is that some such basis lies in the ball. For
`m < n`, `Q` is `m × m`, from the leading square block, and `R = Q*A` is `m × n`.

# Arguments
- `use_bigfloat`: with `true`, the default, the computation runs in BigFloat at `precision_bits`
  and the factors are enclosed to about that precision; with `false` it runs in Float64. A
  BigFloat input is always treated in BigFloat.
- `precision_bits`: the BigFloat precision. A BigFloat input with more digits is rounded to it
  and the rounding is enclosed.
- `use_double_precision`: kept for compatibility; it has no effect.

# Example
```julia
A = randn(100, 50)
r = verified_qr(A; use_bigfloat = false)
r.success      # A = Q R with Q (100 × 50) in r.Q and R (50 × 50) in r.R
```

# Reference
S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 5 and Lemma 5.1.
"""
function verified_qr(A::AbstractMatrix{S};
                     precision_bits::Int=256,
                     use_double_precision::Bool=true,
                     compute_full_Q::Bool=false,
                     use_bigfloat::Bool=true) where S<:Union{Float64, ComplexF64, BigFloat, Complex{BigFloat}}
    bigfloat = use_bigfloat || real(S) === BigFloat
    return _decomposition_arithmetic(bigfloat, precision_bits) do
        Ab = _decomposition_input(A, bigfloat)
        T = eltype(rad(Ab))
        m, n = size(Ab)
        r = _rumpogita2024_qr(Ab; full = compute_full_Q)
        if r === nothing
            k = (compute_full_Q || m < n) ? m : n
            return VerifiedQRResult(_unverified_ball(Ab, m, k), _unverified_ball(Ab, k, n),
                false, T(Inf), T(Inf))
        end
        residual_norm = _relative_residual_bound(r.Q * r.R - Ab, Ab)
        orthogonality_defect = T(upper_bound_L2_opnorm(_ball_adjoint(r.Q) * r.Q - I))
        return VerifiedQRResult(r.Q, r.R, true, residual_norm, orthogonality_defect)
    end
end

# Stub for Double64 extension
"""
    verified_qr_double64(A; precision_bits=256)

Fast verified QR using Double64 oracle. Requires DoubleFloats.jl.
"""
function verified_qr_double64 end

# Stub for MultiFloat extension
"""
    verified_qr_multifloat(A; precision_bits=256, float_type=Float64x4)

Fast verified QR using MultiFloat oracle. Requires MultiFloats.jl.
"""
function verified_qr_multifloat end

export VerifiedQRResult, verified_qr, verified_qr_double64, verified_qr_multifloat
