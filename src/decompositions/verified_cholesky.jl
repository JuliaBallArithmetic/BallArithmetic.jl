# The exported Cholesky decomposition with error bounds. The method is in `rump_ogita_2024.jl`;
# the working arithmetic and the residual are shared with `verified_lu.jl`.

"""
    VerifiedCholeskyResult{GM, RT}

Result of [`verified_cholesky`](@ref).

# Fields
- `G::GM`: ball matrix containing the upper triangular Cholesky factor, `A = G*G`
- `success::Bool`: whether the verification succeeded; when `false` the radii are infinite and
  nothing is proved, in particular not that `A` fails to be positive definite
- `residual_norm::RT`: an upper bound of `‖G̃*G̃ − A‖_∞ / ‖A‖_∞` over all `G̃ ∈ G`

# Statement
When `success` is `true`, the Hermitian part `(A + A*)/2` of the input, which is the input when
it is Hermitian, is proved positive definite, and its Cholesky factor, upper triangular with
positive diagonal, lies in the ball matrix `G`.

# Reference
S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 4.
"""
struct VerifiedCholeskyResult{GM <: BallMatrix, RT <: Real}
    G::GM
    success::Bool
    residual_norm::RT
end

# A ball matrix with Hermitian (`f = conj`) or symmetric (`f = identity`) midpoint and symmetric
# radii that contains (Ã + f(Ã)ᵀ)/2 for every Ã of the ball A. The half sum is formed in ball
# arithmetic and its upper triangle is mirrored: the matrix enclosed has the symmetry, so its
# entry (j, i) is `f` of its entry (i, j), which the ball (i, j) contains.
function _symmetric_part_ball(A::BallMatrix{T}, f) where {T}
    Am, Ar = mid(A), rad(A)
    Ft = Matrix(transpose(f.(Am)))
    (Am == Ft && Ar == transpose(Ar)) && return BallMatrix(Matrix(Am), Matrix(Ar))
    H = (A + BallMatrix(Ft, Matrix(transpose(Ar)))) * Ball(one(T) / 2, zero(T))
    Hm, Hr = copy(mid(H)), copy(rad(H))
    for j in axes(Hm, 2)
        f === conj && (Hm[j, j] = real(Hm[j, j]))
        for i in 1:(j - 1)
            Hm[j, i] = f(Hm[i, j])
            Hr[j, i] = Hr[i, j]
        end
    end
    return BallMatrix(Hm, Hr)
end

"""
    verified_cholesky(A::AbstractMatrix; precision_bits = 256, use_bigfloat = true)

An inclusion of the Cholesky factor of `A`, by Section 4 of Rump and Ogita (2024); the method is
described at [`_rumpogita2024_cholesky`](@ref). Returns a [`VerifiedCholeskyResult`](@ref): when
`success` is `true`, `A` is proved positive definite and `A = G*G` with `G` upper triangular
with positive diagonal, in the ball matrix returned. Positive definiteness is a conclusion and
not an assumption.

When `A` is not Hermitian a warning is issued (above a rounding-level threshold) and the
statement is for its Hermitian part `(A + A*)/2`, enclosed in ball arithmetic.

# Arguments
- `use_bigfloat`: with `true`, the default, the computation runs in BigFloat at `precision_bits`
  and the factor is enclosed to about that precision; with `false` it runs in Float64. A
  BigFloat input is always treated in BigFloat.
- `precision_bits`: the BigFloat precision. A BigFloat input with more digits is rounded to it
  and the rounding is enclosed.
- `use_double_precision`: kept for compatibility; it has no effect.

# Example
```julia
B = randn(100, 100); A = B' * B + I
r = verified_cholesky(A; use_bigfloat = false)
r.success      # A is positive definite and its Cholesky factor lies in r.G
```

# Reference
S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 4.
"""
function verified_cholesky(A::AbstractMatrix{S};
        precision_bits::Int = 256,
        use_double_precision::Bool = true,
        use_bigfloat::Bool = true) where {S <: Union{
        Float64, ComplexF64, BigFloat, Complex{BigFloat}}}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square"))
    sym_error = maximum(abs.(A - A'); init = zero(real(S)))
    if sym_error > 100 * eps(real(S)) * maximum(abs.(A); init = zero(real(S)))
        @warn "Matrix A is not symmetric (error = $sym_error)"
    end
    bigfloat = use_bigfloat || real(S) === BigFloat
    return _decomposition_arithmetic(bigfloat, precision_bits) do
        Hb = _symmetric_part_ball(_decomposition_input(A, bigfloat), conj)
        T = eltype(rad(Hb))
        r = _rumpogita2024_cholesky(Hb)
        r === nothing && return VerifiedCholeskyResult(_unverified_ball(Hb, n, n), false, T(Inf))
        residual_norm = _relative_residual_bound(_ball_adjoint(r.G) * r.G - Hb, Hb)
        return VerifiedCholeskyResult(r.G, true, residual_norm)
    end
end

# Stub for Double64 extension
"""
    verified_cholesky_double64(A; precision_bits=256)

Fast verified Cholesky using Double64 oracle. Requires DoubleFloats.jl.
"""
function verified_cholesky_double64 end

# Stub for MultiFloat extension
"""
    verified_cholesky_multifloat(A; precision_bits=256, float_type=Float64x4)

Fast verified Cholesky using MultiFloat oracle. Requires MultiFloats.jl.
"""
function verified_cholesky_multifloat end

export VerifiedCholeskyResult, verified_cholesky, verified_cholesky_double64,
       verified_cholesky_multifloat
