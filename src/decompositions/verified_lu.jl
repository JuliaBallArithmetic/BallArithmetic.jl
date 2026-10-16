# The exported LU decomposition with error bounds, and what the exported decompositions share:
# the working arithmetic selected by the leading type argument and `precision_bits`, the input as a ball
# matrix in it, and the relative residual of an enclosure. The method is in `rump_ogita_2024.jl`.

# Helper to get appropriate BigFloat type (handles Complex)
_bigfloat_type(::Type{T}) where {T <: Real} = BigFloat
_bigfloat_type(::Type{Complex{T}}) where {T <: Real} = Complex{BigFloat}
_to_bigfloat(M::AbstractMatrix{T}) where {T} = convert.(_bigfloat_type(T), M)
_to_bigfloat(v::AbstractVector{T}) where {T} = convert.(_bigfloat_type(T), v)

# The working arithmetic of the enclosure, as a type, from the leading type argument
# `arithmetic` of the `verified_*` routines and the element type `S` of the input: the arithmetic
# asked for. Float64 for a BigFloat input raises, since it would mean either rounding the input to
# Float64 or running in an arithmetic other than the one asked for (Isaia, 2026-10-16; it used to
# run in BigFloat). The enclosure is computed in ball arithmetic, which this package has for
# Float64 and for BigFloat; another floating-point type can only be the `oracle`. It is a type,
# chosen by dispatch, so that the type of the result is inferred from the types of the arguments.
_working_arithmetic(::Type{BigFloat}, ::Type) = BigFloat
function _working_arithmetic(::Type{Float64}, ::Type{S}) where {S}
    real(S) === BigFloat &&
        throw(ArgumentError("Float64 arithmetic was asked for a BigFloat input; pass BigFloat, " *
                            "or convert the input to Float64 explicitly"))
    return Float64
end
_working_arithmetic(arithmetic::Type, ::Type) =
    throw(ArgumentError("$arithmetic: the enclosure is computed in Float64 or in " *
                        "BigFloat; a Double64 or a MultiFloat type can be given as `oracle`"))

# Runs `f` in the working arithmetic of the exported decompositions: Float64, or BigFloat at
# `precision_bits`.
_decomposition_arithmetic(f, ::Type{BigFloat}, precision_bits::Integer) =
    setprecision(f, BigFloat, precision_bits)
_decomposition_arithmetic(f, ::Type{Float64}, precision_bits::Integer) = f()

# The input as a ball matrix in the working arithmetic; called inside
# `_decomposition_arithmetic`. An entry with more digits than the working precision is rounded
# to nearest and the rounding error, which is computed exactly at the precision of the entry and
# then rounded up, goes into the radius, so that the ball contains the input.
_decomposition_input(A::AbstractMatrix{S}, ::Type{Float64}) where {S} =
    BallMatrix(Matrix{S <: Complex ? ComplexF64 : Float64}(A))
function _decomposition_input(A::AbstractMatrix{S}, ::Type{BigFloat}) where {S}
    p = precision(BigFloat)
    nearest(x::Real) = BigFloat(x; precision = p)
    function excess(x::Real, c::BigFloat)
        px = precision(x)
        px <= p && return BigFloat(0; precision = p)
        d = setprecision(() -> abs(BigFloat(x) - c), BigFloat, px)
        return BigFloat(d, RoundUp; precision = p)
    end
    c = Matrix{S <: Complex ? Complex{BigFloat} : BigFloat}(undef, size(A))
    r = Matrix{BigFloat}(undef, size(A))
    for i in eachindex(A)
        x = A[i]
        if S <: Complex
            cr, ci = nearest(real(x)), nearest(imag(x))
            c[i] = complex(cr, ci)
            r[i] = add_up(excess(real(x), cr), excess(imag(x), ci))
        else
            c[i] = nearest(x)
            r[i] = excess(x, c[i])
        end
    end
    return BallMatrix(c, r)
end

# What a failed verification returns: nothing is enclosed.
_unverified_ball(::BallMatrix{T, CT}, m::Integer, n::Integer) where {T, CT} =
    BallMatrix(fill(CT(NaN), m, n), fill(T(Inf), m, n))

# An upper bound of ‖R̃‖_∞ / ‖Ã‖_∞ over the matrices R̃ of the ball R and Ã of the ball A.
function _relative_residual_bound(R::BallMatrix{T}, A::BallMatrix{T}) where {T}
    num = T(upper_bound_L_inf_opnorm(R))
    Am, Ar = mid(A), rad(A)
    den = zero(T)
    for i in axes(Am, 1)
        s = zero(T)
        for j in axes(Am, 2)
            s = add_down(s, max(sub_down(abs_down(Am[i, j]), Ar[i, j]), zero(T)))
        end
        den = max(den, s)
    end
    return den > 0 ? div_up(num, den) : T(Inf)
end

"""
    VerifiedLUResult{LM, UM, RT}

Result of [`verified_lu`](@ref).

# Fields
- `L::LM`: ball matrix containing the unit lower triangular (trapezoidal) factor
- `U::UM`: ball matrix containing the upper triangular (trapezoidal) factor
- `p::Vector{Int}`: row permutation
- `certified::Bool`: whether the verification succeeded; when `false` the radii are infinite and
  nothing is proved
- `residual_norm::RT`: an upper bound of `‖L̃Ũ − A[p, q]‖_∞ / ‖A‖_∞` over all `L̃ ∈ L`, `Ũ ∈ U`
- `q::Vector{Int}`: column permutation, the identity unless the matrix has more columns than rows

# Statement
When `certified` is `true`, `A[p, q]` has a unique decomposition `LU` with `L` unit lower and `U`
upper triangular, and these factors lie in the ball matrices `L` and `U`.

# Reference
S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 3.
"""
struct VerifiedLUResult{LM <: BallMatrix, UM <: BallMatrix, RT <: Real}
    L::LM
    U::UM
    p::Vector{Int}
    certified::Bool
    residual_norm::RT
    q::Vector{Int}
end
VerifiedLUResult(L, U, p, success, residual_norm) =
    VerifiedLUResult(L, U, p, success, residual_norm, collect(1:size(U, 2)))

"""
    verified_lu([T = BigFloat,] A::AbstractMatrix; precision_bits = 256)

Inclusions of the factors of the LU decomposition of the `m × n` matrix `A`, by Section 3 of
Rump and Ogita (2024); the method is described at [`_rumpogita2024_lu`](@ref). Returns a
[`VerifiedLUResult`](@ref): when `certified` is `true`, `A[p, q] = LU` with `L` of size
`m × min(m, n)` unit lower triangular and `U` of size `min(m, n) × n` upper triangular, both in
the ball matrices returned. `p` comes from partial pivoting; `q` is the identity for `m ≥ n`, and
for `m < n` the column permutation of Section 3.4 of the paper.

# Arguments
- `T`: the arithmetic of the enclosure, `BigFloat` (the default) or `Float64`, as a leading
  argument. In
  BigFloat the computation runs at `precision_bits` and the factors are enclosed to about that
  precision; in Float64 the radii are of the order of the rounding unit times the size of the
  factors. `Float64` with a BigFloat input raises.
- `precision_bits`: the BigFloat precision. A BigFloat input with more digits is rounded to it
  and the rounding is enclosed.
- `oracle`: a floating-point type (`Double64`, a `MultiFloat`) in which the approximate
  factorisation the method starts from is computed, in place of the working arithmetic;
  `nothing`, the default, means the working arithmetic. It has no part in the verification:
  the factors are enclosed from residuals in the working arithmetic whatever their accuracy,
  which only decides how small the radii come out. The same keyword is taken by
  [`verified_cholesky`](@ref), [`verified_qr`](@ref), [`verified_polar`](@ref) and
  [`verified_takagi`](@ref).

# Example
```julia
A = randn(100, 100)
r = verified_lu(Float64, A)
r.certified      # A[r.p, r.q] = L U with L in r.L and U in r.U
```

# Reference
S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 3.
"""
function verified_lu(::Type{AT}, A::AbstractMatrix{S};
        precision_bits::Int = 256,
        oracle::Union{Nothing, Type} = nothing) where {AT, S <: Union{
        Float64, ComplexF64, BigFloat, Complex{BigFloat}}}
    W = _working_arithmetic(AT, S)
    return _decomposition_arithmetic(W, precision_bits) do
        Ab = _decomposition_input(A, W)
        T = eltype(rad(Ab))
        m, n = size(Ab)
        k = min(m, n)
        r = _rumpogita2024_lu(Ab; oracle)
        r === nothing && return VerifiedLUResult(_unverified_ball(Ab, m, k),
            _unverified_ball(Ab, k, n), collect(1:m), false, T(Inf), collect(1:n))
        Apq = BallMatrix(mid(Ab)[r.p, r.q], rad(Ab)[r.p, r.q])
        residual_norm = _relative_residual_bound(r.L * r.U - Apq, Apq)
        return VerifiedLUResult(r.L, r.U, r.p, true, residual_norm, r.q)
    end
end

verified_lu(A::AbstractMatrix; kwargs...) = verified_lu(BigFloat, A; kwargs...)

export VerifiedLUResult, verified_lu
