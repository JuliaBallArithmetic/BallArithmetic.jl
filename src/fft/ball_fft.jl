# The discrete Fourier transform of a vector or a matrix of balls,
# y_k = Σ_j x_j exp(−2πi jk/N), with the a priori error bound of
#
#   N. Brisebarre, J.-M. Muller and J. Picot, "Testing the sharpness of known error bounds on
#   the fast Fourier transform", 30th IEEE Symposium on Computer Arithmetic (ARITH 2023),
#   hal-04092770.
#
# The bound is proved for one algorithm, the radix-2 transform of the paper's Fig. 1 with the
# roots of unity rounded to nearest, and that algorithm is implemented here. The extension for
# FFTW adds the method `:brisebarreetal2023fftw`, in which the midpoint comes from FFTW and the
# same radius is used; the paper proves nothing for FFTW's algorithms, and the name says so.

# The unit roundoff u = 2^(−p) of the paper, half of `eps`.
_brisebarreetal2023_u(::Type{T}) where {T <: AbstractFloat} = eps(T) / 2

# Equation (4) of the paper: with ‖Z‖∞⊥ := max_k max(|Re z_k|, |Im z_k|),
#
#     ‖Ŷ − Y‖∞⊥ ≤ b_n ‖X‖∞⊥,      b_n = √2 · 2ⁿ · ((1 + u)ⁿ ∏_{j=1}^{n} (1 + g_j) − 1),
#
# g₁ = g₂ = 0 and g_j = δ_j + ρ(1 + δ_j) for j ≥ 3. δ_j is the largest absolute error of the
# rounded roots of unity of order 2^j: at most u when both parts are rounded to nearest, each
# being within u/2 of a number of modulus at most one. ρ is the relative error of the complex
# product: √5 u for the product formed without a fused multiply-add, as it is here.
#
# The product is not formed factor by factor: 1 + u is not a floating-point number, and rounding
# it upwards to 1 + 2u doubles the bound. With S := n u + (n − 2) g, the sum of the logarithms'
# upper bounds, ∏(1 + aᵢ) − 1 ≤ e^S − 1 ≤ S / (1 − S/2) for 0 ≤ S < 2, by comparison of the two
# series; every operation below is rounded in the direction that keeps the bound.
function _brisebarreetal2023_bn(n::Integer, ::Type{T}) where {T <: AbstractFloat}
    u = _brisebarreetal2023_u(T)
    ρ = mul_up(sqrt_up(T(5)), u)
    g = add_up(u, add_up(ρ, mul_up(ρ, u)))
    S = add_up(mul_up(T(n), u), mul_up(T(max(n - 2, 0)), g))
    S < 1 || return T(Inf)
    bracket = div_up(S, sub_down(one(T), div_up(S, T(2))))
    return mul_up(sqrt_up(T(2)), mul_up(T(2)^n, bracket))
end

# The roots of unity exp(−2πi j/N), j = 0, …, N/2 − 1, each part rounded to nearest. The bound
# is proved for that rounding, so the table has to be the correctly rounded one and not merely
# a close one; the values are computed in BigFloat with 64 guard bits and rounded once.
#
# Only the first octant is computed that way. For j ≤ N/8 the angle is θ = 2j/N ≤ 1/4, and the
# other three quarters of the half-circle come from
#
#     cos(π/2 − x) = sin x ,      cos(π − x) = −cos x ,
#
# a swap of the two parts and a change of sign, both exact in floating point: the entry a
# symmetry produces is the same float the direct evaluation would round to, since the two real
# numbers are equal. The loop over the octant is threaded, which is what makes the large sizes
# affordable: 2²⁴ takes 0.5 s against 20 s for the straightforward loop, and 2²⁸ about 8 s.
function _brisebarreetal2023_compute_roots(::Type{T}, N::Int) where {T <: AbstractFloat}
    n2 = max(N ÷ 2, 1)
    w = Vector{Complex{T}}(undef, n2)
    p = precision(T)
    oct = N ÷ 8
    if oct < 8                       # too small for the symmetry to be worth its bookkeeping
        setprecision(BigFloat, p + 64) do
            for j in 0:(n2 - 1)
                θ = BigFloat(2 * j) / N          # exact: N is a power of two
                w[j + 1] = Complex{T}(_nearest(T, cospi(θ), p), -_nearest(T, sinpi(θ), p))
            end
        end
        return w
    end
    c = Vector{T}(undef, oct + 1)
    sn = Vector{T}(undef, oct + 1)
    nt = max(Threads.nthreads(), 1)
    chunk = cld(oct + 1, nt)
    Threads.@threads for t in 1:nt
        lo = (t - 1) * chunk
        hi = min(t * chunk - 1, oct)
        lo > hi && continue
        setprecision(BigFloat, p + 64) do         # the precision is task local
            for j in lo:hi
                θ = BigFloat(2 * j) / N
                c[j + 1] = _nearest(T, cospi(θ), p)
                sn[j + 1] = _nearest(T, sinpi(θ), p)
            end
        end
    end
    @inbounds for j in 0:(n2 - 1)
        q, r = divrem(j, oct)
        cj, sj = if q == 0
            (c[r + 1], sn[r + 1])
        elseif q == 1
            (sn[oct - r + 1], c[oct - r + 1])
        elseif q == 2
            # at j = N/4 the angle is 1/2 and the cosine is exactly zero, where the direct
            # evaluation gives +0.0 and the negation would give −0.0
            (iszero(sn[r + 1]) ? zero(T) : -sn[r + 1], c[r + 1])
        else
            (-c[oct - r + 1], sn[oct - r + 1])
        end
        w[j + 1] = Complex{T}(cj, -sj)
    end
    return w
end

# the rounding to the working precision; for BigFloat that is the precision outside the block
_nearest(::Type{T}, x, p::Integer) where {T <: AbstractFloat} =
    T === BigFloat ? BigFloat(x; precision = p) : T(x)

# The sizes are powers of two, and the roots of order N are among those of order 2N, so one
# table serves every size up to its own. The tables for Float64 and Float32 up to this size are
# computed when the package is precompiled and stored with it; larger sizes, and BigFloat, are
# computed at the first use and kept.
const _BRISEBARREETAL2023_TABLE_SIZE = 1 << 16
const _BRISEBARREETAL2023_ROOTS_F64 = _brisebarreetal2023_compute_roots(Float64,
    _BRISEBARREETAL2023_TABLE_SIZE)
const _BRISEBARREETAL2023_ROOTS_F32 = _brisebarreetal2023_compute_roots(Float32,
    _BRISEBARREETAL2023_TABLE_SIZE)
const _BRISEBARREETAL2023_ROOTS = Dict{Tuple{DataType, Int, Int}, Any}()
const _BRISEBARREETAL2023_ROOTS_LOCK = ReentrantLock()

# A table `w` and a stride `s` with exp(−2πi j/N) rounded to nearest at `w[j s + 1]`.
function _brisebarreetal2023_roots(::Type{T}, N::Int) where {T <: AbstractFloat}
    if N <= _BRISEBARREETAL2023_TABLE_SIZE
        T === Float64 && return _BRISEBARREETAL2023_ROOTS_F64, _BRISEBARREETAL2023_TABLE_SIZE ÷ N
        T === Float32 && return _BRISEBARREETAL2023_ROOTS_F32, _BRISEBARREETAL2023_TABLE_SIZE ÷ N
    end
    key = (T, N, T === BigFloat ? precision(BigFloat) : 0)
    w = lock(_BRISEBARREETAL2023_ROOTS_LOCK) do
        get!(() -> _brisebarreetal2023_compute_roots(T, N), _BRISEBARREETAL2023_ROOTS, key)
    end::Vector{Complex{T}}
    return w, 1
end

# The radix-2 transform of Fig. 1 of the paper: the permutation by bit reversal, then the stages
# k = 1, …, n of butterflies y[j1] ← y[j1] + ω y[j2], y[j2] ← y[j1] − ω y[j2] on blocks of size
# 2^k, ω the rounded root of unity of order 2^k. The complex product is formed without a fused
# multiply-add and the complex sum part by part, in rounding to nearest.
function _brisebarreetal2023_radix2(x::AbstractVector{Complex{T}}) where {T <: AbstractFloat}
    N = length(x)
    w, stride = _brisebarreetal2023_roots(T, N)
    return _brisebarreetal2023_radix2(x, w, stride)
end

function _brisebarreetal2023_radix2(x::AbstractVector{Complex{T}},
                                    w::AbstractVector{Complex{T}},
                                    stride::Integer) where {T <: AbstractFloat}
    N = length(x)
    n = trailing_zeros(N)
    y = Vector{Complex{T}}(undef, N)
    for j in 0:(N - 1)
        r = 0
        for b in 0:(n - 1)
            r |= ((j >> b) & 1) << (n - 1 - b)
        end
        y[j + 1] = x[r + 1]
    end
    for k in 1:n
        bs = 1 << k
        half = bs >> 1
        step = (N >> k) * stride
        @inbounds for first in 0:bs:(N - 1), j in 0:(half - 1)
            j1 = first + j + 1
            j2 = j1 + half
            ω = w[j * step + 1]
            a, b = y[j1], y[j2]
            tr = real(ω) * real(b) - imag(ω) * imag(b)
            ti = real(ω) * imag(b) + imag(ω) * real(b)
            y[j1] = Complex(real(a) + tr, imag(a) + ti)
            y[j2] = Complex(real(a) - tr, imag(a) - ti)
        end
    end
    return y
end

"""
    _fftw_transform(x::Vector{<:Complex})

The discrete Fourier transform of `x` in floating point by FFTW. A function without methods in
the package; the extension for FFTW adds it. It is the midpoint of the method
`:brisebarreetal2023fftw` of [`ball_fft`](@ref).
"""
function _fftw_transform end

# the floating-point transform of a method
function _ball_fft_transform(method::Symbol)
    method === :brisebarreetal2023 && return _brisebarreetal2023_radix2
    if method === :brisebarreetal2023fftw
        isempty(methods(_fftw_transform)) && throw(ArgumentError(
            "the method :brisebarreetal2023fftw needs FFTW: load it with `using FFTW`"))
        return _fftw_transform
    end
    throw(ArgumentError("unknown method $(repr(method)); the implemented methods are " *
                        ":brisebarreetal2023 and :brisebarreetal2023fftw"))
end

function _ball_fft_power_of_two(N::Integer)
    (N >= 1 && ispow2(N)) || throw(ArgumentError(
        "the error bound of the transform is proved for sizes that are powers of two; got $N"))
    return trailing_zeros(N)
end

# ‖slice‖∞⊥ of the midpoints, which is exact, and an upper bound of the sum of the radii
function _ball_fft_norms(c_slice, r_slice, ::Type{T}) where {T <: AbstractFloat}
    x_inf = zero(T)
    @inbounds for z in c_slice
        x_inf = max(x_inf, T(abs(real(z))), T(abs(imag(z))))
    end
    r_1 = zero(T)
    @inbounds for r in r_slice
        r_1 = add_up(r_1, T(r))
    end
    return x_inf, r_1
end

# √2 · b_n · x_inf + r_1, rounded upwards: the modulus of the error of an output from the bound
# on its two parts, plus the effect of the radii of the input
_ball_fft_radius(bn::T, x_inf::T, r_1::T) where {T <: AbstractFloat} =
    add_up(mul_up(sqrt_up(T(2)), mul_up(bn, x_inf)), r_1)

"""
    ball_fft(v::BallVector; method = :brisebarreetal2023)

The discrete Fourier transform `y_k = Σ_j x_j exp(−2πi jk/N)` of every vector in the ball `v`,
of length `N = 2ⁿ`: a ball vector whose midpoint is the transform of the midpoint computed in
floating point and whose radii are all

    √2 · b_n · ‖X‖∞⊥ + ‖r‖₁.

Here `‖X‖∞⊥` is the largest absolute value of a real or imaginary part of the midpoints, `r`
the radii of `v`, and `b_n` the constant of equation (4) of [BrisebarreEtAl2023](@cite),
which bounds each part of the rounding error of an output by `b_n ‖X‖∞⊥`; the factor `√2`
passes from the two parts to the modulus. The term `‖r‖₁` is the effect of the radii: an output
is a sum of the inputs times numbers of modulus one.

Each algorithm is named for the paper it implements, a deviation being visible in the name:

| `method` | what it is |
|---|---|
| `:brisebarreetal2023` | the midpoint by the radix-2 algorithm of the paper's Fig. 1, with the roots of unity rounded to nearest and the complex product formed without a fused multiply-add: the algorithm the bound is proved for |
| `:brisebarreetal2023fftw` | the midpoint by FFTW, with the same radius: a hypothesis, not a theorem (see below). Needs `using FFTW` |

Why the bound is not known to hold for FFTW. The paper's proof is for the radix-2 step with
the roots of unity rounded to nearest, and FFTW chooses among other algorithms and computes its
roots in its own way. An audit of the FFTW source, to find which bound holds for what it
computes, is in progress and not published; until it is, the radius of this method is a
hypothesis.

The paper's analysis excludes underflow and overflow; an output that is not finite is an
`ArgumentError`, and underflow is not detected. A length that is not a power of two is an
`ArgumentError`. With FFTW loaded, `fft(v)` is `ball_fft(v)`.

# Reference

[BrisebarreEtAl2023](@cite), equation (4) for `b_n` and Fig. 1 for the algorithm the
bound is proved for.
"""
function ball_fft(v::BallVector{T}; method::Symbol = :brisebarreetal2023) where {T}
    transform = _ball_fft_transform(method)
    N = length(v)
    n = _ball_fft_power_of_two(N)
    Ŷ = Vector{Complex{T}}(transform(Vector{Complex{T}}(v.c)))
    all(isfinite, Ŷ) || throw(ArgumentError("the transform overflowed"))
    x_inf, r_1 = _ball_fft_norms(v.c, v.r, T)
    return BallVector(Ŷ, fill(_ball_fft_radius(_brisebarreetal2023_bn(n, T), x_inf, r_1), N))
end

"""
    ball_fft(A::BallMatrix, dims = (1, 2); method = :brisebarreetal2023)

The discrete Fourier transform of the ball matrix `A` along the dimensions `dims` (an integer,
or a tuple), each of size a power of two. Along one dimension it is [`ball_fft`](@ref) of each
column or row, with the bound of that fiber. Along both, the columns are transformed and then
the rows, and every output has the radius

    √2 · b_n · ‖X‖∞⊥ + ‖r‖₁,        n = log₂(number of entries),

with the norms taken over the whole matrix.

The bound along both dimensions is not in the paper; it follows from the bound it quotes for
the relative error in the Euclidean norm of the one-dimensional algorithm, `ε_m = (1 + u)ᵐ
∏_{j ≤ m} (1 + g_j) − 1` for size `2ᵐ`. Applied to the columns and summed over them, and then to
the rows, it gives `‖Ŷ − Y‖_F ≤ ((1 + ε_{n₁})(1 + ε_{n₂}) − 1) ‖Y‖_F`, the exact transform
multiplying the Frobenius norm by the square root of the size of a fiber; and
`(1 + ε_{n₁})(1 + ε_{n₂}) ≤ 1 + ε_n` for `n = n₁ + n₂`. An entry is at most the Frobenius norm,
and `‖Y‖_F = √N ‖X‖_F ≤ √2 N ‖X‖∞⊥` with `N = 2ⁿ`, which is `b_n ‖X‖∞⊥`.

`method` is as for a vector.
"""
function ball_fft(A::BallMatrix{T}, dims = (1, 2);
        method::Symbol = :brisebarreetal2023) where {T}
    transform = _ball_fft_transform(method)
    dims_t = dims isa Integer ? (Int(dims),) : Tuple(Int.(dims))
    all(d -> d in (1, 2), dims_t) && allunique(dims_t) ||
        throw(ArgumentError("dims must be 1, 2 or both"))
    n = sum(_ball_fft_power_of_two(size(A.c, d)) for d in dims_t)
    bn = _brisebarreetal2023_bn(n, T)

    Ŷ = Matrix{Complex{T}}(A.c)
    if 1 in dims_t
        for j in axes(Ŷ, 2)
            Ŷ[:, j] = transform(Ŷ[:, j])
        end
    end
    if 2 in dims_t
        for i in axes(Ŷ, 1)
            Ŷ[i, :] = transform(Ŷ[i, :])
        end
    end
    all(isfinite, Ŷ) || throw(ArgumentError("the transform overflowed"))

    err = zeros(T, size(A.c))
    if length(dims_t) == 2
        x_inf, r_1 = _ball_fft_norms(A.c, A.r, T)
        fill!(err, _ball_fft_radius(bn, x_inf, r_1))
    elseif dims_t[1] == 1
        for j in axes(err, 2)
            x_inf, r_1 = _ball_fft_norms(view(A.c, :, j), view(A.r, :, j), T)
            err[:, j] .= _ball_fft_radius(bn, x_inf, r_1)
        end
    else
        for i in axes(err, 1)
            x_inf, r_1 = _ball_fft_norms(view(A.c, i, :), view(A.r, i, :), T)
            err[i, :] .= _ball_fft_radius(bn, x_inf, r_1)
        end
    end
    return BallMatrix(Ŷ, err)
end

"""
    BallFFTPlan{T}

A size and the table of roots of unity it needs, so that many transforms of that size share one
table.  The table of order `N` holds the roots of every smaller power of two as well, which is
why a plan made for `N` transforms any size dividing it.

Build one with [`plan_ball_fft`](@ref) and pass it to [`ball_fft`](@ref).  Keeping it is what
makes a large size affordable twice: at `N = 2²⁸` the table is `2²⁷` complex numbers, 2.1 GB
for `Float64`, and some seconds of `BigFloat` evaluation to build.  Dropping the plan frees it.
"""
struct BallFFTPlan{T <: AbstractFloat}
    n::Int                        # log2 of the size the plan was made for
    method::Symbol
    w::Vector{Complex{T}}         # the roots of unity, rounded to nearest
    stride::Int                   # the step through `w` for the size being transformed
    bn::T                         # the rounding bound of the transform at this size
end

Base.length(p::BallFFTPlan) = 1 << p.n
function Base.show(io::IO, p::BallFFTPlan{T}) where {T}
    print(io, "BallFFTPlan{", T, "}(N = 2^", p.n, ", method = ", repr(p.method), ")")
end

"""
    plan_ball_fft(N; T = Float64, method = :brisebarreetal2023)

The [`BallFFTPlan`](@ref) for transforms of length `N`, a power of two: it computes the table
of roots once, which is the cost a repeated transform should not pay twice.  `T` is the float
type of the midpoints.
"""
function plan_ball_fft(N::Integer; T::Type{<:AbstractFloat} = Float64,
                       method::Symbol = :brisebarreetal2023)
    n = _ball_fft_power_of_two(N)
    _ball_fft_transform(method)            # rejects an unknown method, and FFTW if absent
    w, stride = _brisebarreetal2023_roots(T, Int(N))
    return BallFFTPlan{T}(n, method, w, stride, _brisebarreetal2023_bn(n, T))
end

"""
    ball_fft(v::BallVector, plan::BallFFTPlan)

[`ball_fft`](@ref) with the table of `plan`, which must have been made for the length of `v`.
"""
function ball_fft(v::BallVector{T}, plan::BallFFTPlan{T}) where {T}
    N = length(v)
    n = _ball_fft_power_of_two(N)
    n == plan.n || throw(ArgumentError(
        "the plan is for length 2^$(plan.n) and the vector has length $N"))
    c = Vector{Complex{T}}(v.c)
    Ŷ = if plan.method === :brisebarreetal2023
        _brisebarreetal2023_radix2(c, plan.w, plan.stride)
    else
        Vector{Complex{T}}(_ball_fft_transform(plan.method)(c))
    end
    all(isfinite, Ŷ) || throw(ArgumentError("the transform overflowed"))
    x_inf, r_1 = _ball_fft_norms(v.c, v.r, T)
    return BallVector(Ŷ, fill(_ball_fft_radius(plan.bn, x_inf, r_1), N))
end

export ball_fft, plan_ball_fft, BallFFTPlan
