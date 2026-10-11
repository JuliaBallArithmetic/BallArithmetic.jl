# The exported polar decomposition with error bounds. The method is in `rump_ogita_2024.jl`; the
# working arithmetic and the residual are shared with `verified_lu.jl`.

"""
    VerifiedPolarResult{QM, PM, RT}

Result of [`verified_polar`](@ref).

# Fields
- `Q::QM`: ball matrix containing the factor with orthonormal columns (rows, for the left
  decomposition)
- `P::PM`: ball matrix containing the Hermitian positive definite factor
- `is_right::Bool`: `true` for `A = QP`, `false` for `A = PQ`
- `success::Bool`: whether the verification succeeded; when `false` the radii are infinite and
  nothing is proved
- `residual_norm::RT`: an upper bound of `‖Q̃P̃ − A‖_∞ / ‖A‖_∞` (of `‖P̃Q̃ − A‖_∞ / ‖A‖_∞` for
  the left decomposition) over all `Q̃ ∈ Q`, `P̃ ∈ P`

# Statement
When `success` is `true`, `A` has full rank and `A = QP` (or `A = PQ`) with `P` Hermitian
positive definite and `Q` with orthonormal columns (rows), `Q` and `P` in the ball matrices
returned.

# Reference
S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 7.
"""
struct VerifiedPolarResult{QM<:BallMatrix, PM<:BallMatrix, RT<:Real}
    Q::QM
    P::PM
    is_right::Bool
    success::Bool
    residual_norm::RT
end

"""
    verified_polar(A::AbstractMatrix; right = true, kappa = 0, precision_bits = 256,
                   use_bigfloat = true)

Inclusions of the factors of the polar decomposition of the `m × n` matrix `A`, by Section 7 of
Rump and Ogita (2024), from the inclusions of the singular value decomposition of Rump and Lange
(2023); the method is described at [`_rumpogita2024_polar`](@ref). Returns a
[`VerifiedPolarResult`](@ref).

With `right = true`, which needs `m ≥ n`, the decomposition is `A = QP` with `Q` of size `m × n`
with orthonormal columns and `P` of size `n × n` Hermitian positive definite. With
`right = false`, which needs `m ≤ n`, it is `A = PQ` with `P` of size `m × m` and `Q` of size
`m × n` with orthonormal rows, obtained from the right decomposition of `A*`. `success` is
`false` when a singular value is not separated from zero.

# Arguments
- `kappa`: the clustering threshold of [`verifysvdall`](@ref).
- `use_bigfloat`: with `true`, the default, the computation runs in BigFloat at `precision_bits`;
  with `false` it runs in Float64. A BigFloat input is always treated in BigFloat.
- `precision_bits`: the BigFloat precision. A BigFloat input with more digits is rounded to it
  and the rounding is enclosed.

# Example
```julia
A = randn(ComplexF64, 50, 50)
r = verified_polar(A; use_bigfloat = false)
r.success      # A = Q P with Q in r.Q and P in r.P
```

# Reference
S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 7; S. M. Rump and M. Lange,
*Fast computation of error bounds for all eigenpairs of a Hermitian and all singular pairs of a
rectangular matrix with emphasis on eigen- and singular value clusters*, J. Comput. Appl. Math.
**434** (2023), 115332, doi 10.1016/j.cam.2023.115332.
"""
function verified_polar(A::AbstractMatrix{S};
                        precision_bits::Int=256,
                        right::Bool=true,
                        kappa::Real=0,
                        use_bigfloat::Bool=true) where S<:Union{Float64, ComplexF64, BigFloat, Complex{BigFloat}}
    m, n = size(A)
    (right ? m >= n : m <= n) || throw(DimensionMismatch(
        "verified_polar: A = QP needs at least as many rows as columns, A = PQ at most as many"))
    bigfloat = use_bigfloat || real(S) === BigFloat
    return _decomposition_arithmetic(bigfloat, precision_bits) do
        Ab = _decomposition_input(A, bigfloat)
        T = eltype(rad(Ab))
        k = right ? n : m
        r = _rumpogita2024_polar(right ? Ab : _ball_adjoint(Ab); kappa)
        r === nothing && return VerifiedPolarResult(_unverified_ball(Ab, m, n),
            _unverified_ball(Ab, k, k), right, false, T(Inf))
        Q = right ? r.Q : _ball_adjoint(r.Q)
        Ac = BallMatrix(Matrix{eltype(mid(Q))}(mid(Ab)), rad(Ab))
        residual_norm = _relative_residual_bound((right ? Q * r.P : r.P * Q) - Ac, Ac)
        return VerifiedPolarResult(Q, r.P, right, true, residual_norm)
    end
end

# Stub for Double64 extension
"""
    verified_polar_double64(A; precision_bits=256, right=true)

Fast verified polar decomposition using Double64 oracle. Requires DoubleFloats.jl.
"""
function verified_polar_double64 end

# Stub for MultiFloat extension
"""
    verified_polar_multifloat(A; precision_bits=256, float_type=Float64x4, right=true)

Fast verified polar decomposition using MultiFloat oracle. Requires MultiFloats.jl.
"""
function verified_polar_multifloat end

export VerifiedPolarResult, verified_polar, verified_polar_double64, verified_polar_multifloat
