module FFTExt

using BallArithmetic, LinearAlgebra
import BallArithmetic: add_up, sub_up, mul_up, sqrt_up
import FFTW

# The discrete Fourier transform of a vector of balls, y_k = Σ_j x_j exp(−2πi jk/N), with the
# a priori error bound of
#
#   N. Brisebarre, J.-M. Muller and J. Picot, "Testing the sharpness of known error bounds on
#   the fast Fourier transform", 30th IEEE Symposium on Computer Arithmetic (ARITH 2023),
#   hal-04092770.
#
# The bound is proved for one algorithm, the radix-2 transform of the paper's Fig. 1 with the
# roots of unity rounded to nearest. It says nothing of another implementation, so the transform
# is computed here by that algorithm and not by FFTW, whose codelets use other factorisations;
# FFTW is only the package that owns the name `fft`.

# The unit roundoff u = 2^(−p) of the paper, half of `eps`.
_bmp_u(::Type{T}) where {T <: AbstractFloat} = eps(T) / 2

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
function _bmp_bn(n::Integer, ::Type{T}) where {T <: AbstractFloat}
    u = _bmp_u(T)
    ρ = mul_up(sqrt_up(T(5)), u)
    g = add_up(u, add_up(ρ, mul_up(ρ, u)))
    S = add_up(mul_up(T(n), u), mul_up(T(max(n - 2, 0)), g))
    S < 1 || return T(Inf)
    bracket = BallArithmetic.div_up(S, BallArithmetic.sub_down(one(T), BallArithmetic.div_up(S, T(2))))
    return mul_up(sqrt_up(T(2)), mul_up(T(2)^n, bracket))
end

# exp(−2πi j/N) for j = 0, …, N/2 − 1, each part rounded to nearest: computed with 64 guard bits
# and rounded once. The table for a size is kept, the roots being the costly part.
const _ROOTS = Dict{Tuple{DataType, Int, Int}, Any}()
const _ROOTS_LOCK = ReentrantLock()
function _bmp_roots(::Type{T}, N::Int) where {T <: AbstractFloat}
    key = (T, N, T === BigFloat ? precision(BigFloat) : 0)
    return lock(_ROOTS_LOCK) do
        get!(_ROOTS, key) do
            w = Vector{Complex{T}}(undef, max(N ÷ 2, 1))
            setprecision(BigFloat, precision(T) + 64) do
                for j in 0:(length(w) - 1)
                    θ = BigFloat(2 * j) / N                  # exact: N is a power of two
                    w[j + 1] = Complex{T}(T(cospi(θ)), -T(sinpi(θ)))
                end
            end
            w
        end
    end::Vector{Complex{T}}
end

# The radix-2 transform of Fig. 1 of the paper: the permutation by bit reversal, then the stages
# k = 1, …, n of butterflies y[j1] ← y[j1] + ω y[j2], y[j2] ← y[j1] − ω y[j2] on blocks of size
# 2^k, ω the rounded root of unity of order 2^k. The complex product is formed without a fused
# multiply-add and the complex sum part by part, in rounding to nearest.
function _bmp_radix2_fft(x::AbstractVector{Complex{T}}) where {T <: AbstractFloat}
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
    w = _bmp_roots(T, N)
    for k in 1:n
        bs = 1 << k
        half = bs >> 1
        step = N >> k
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

function _require_power_of_two(N::Integer)
    (N >= 1 && ispow2(N)) || throw(ArgumentError(
        "the error bound of the transform is proved for sizes that are powers of two; got $N"))
    return trailing_zeros(N)
end

# ‖slice‖∞⊥ of the midpoints, which is exact, and an upper bound of the sum of the radii
function _fiber_inf_and_l1(c_slice, r_slice, ::Type{T}) where {T <: AbstractFloat}
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
_fft_err(bn::T, x_inf::T, r_1::T) where {T <: AbstractFloat} =
    add_up(mul_up(sqrt_up(T(2)), mul_up(bn, x_inf)), r_1)

"""
    fft(v::BallVector)

The discrete Fourier transform `y_k = Σ_j x_j exp(−2πi jk/N)` of every vector in the ball `v`,
of length `N = 2ⁿ`: a ball vector whose midpoint is the transform of the midpoint computed in
floating point and whose radii are all

    √2 · b_n · ‖X‖∞⊥ + ‖r‖₁.

Here `‖X‖∞⊥` is the largest absolute value of a real or imaginary part of the midpoints, `r`
the radii of `v`, and `b_n` the constant of equation (4) of Brisebarre, Muller and Picot (2023),
which bounds each part of the rounding error of an output by `b_n ‖X‖∞⊥`; the factor `√2`
passes from the two parts to the modulus. The term `‖r‖₁` is the effect of the radii: an output
is a sum of the inputs times numbers of modulus one.

The bound is proved for the radix-2 algorithm of the paper's Fig. 1 with the roots of unity
rounded to nearest and the complex product formed without a fused multiply-add. The midpoint is
computed by that algorithm, not by FFTW. The paper's analysis excludes underflow and overflow;
an output that is not finite is an `ArgumentError`, and underflow is not detected.

A length that is not a power of two is an `ArgumentError`.

# Reference

N. Brisebarre, J.-M. Muller and J. Picot, "Testing the sharpness of known error bounds on the
fast Fourier transform", 30th IEEE Symposium on Computer Arithmetic (ARITH 2023), Portland,
2023, hal-04092770.
"""
function FFTW.fft(v::BallVector{T}) where {T}
    N = length(v)
    n = _require_power_of_two(N)
    Ŷ = _bmp_radix2_fft(Vector{Complex{T}}(v.c))
    all(isfinite, Ŷ) || throw(ArgumentError("the transform overflowed"))
    x_inf, r_1 = _fiber_inf_and_l1(v.c, v.r, T)
    return BallVector(Ŷ, fill(_fft_err(_bmp_bn(n, T), x_inf, r_1), N))
end

"""
    fft(A::BallMatrix, dims = (1, 2))

The discrete Fourier transform of the ball matrix `A` along the dimensions `dims` (an integer,
or a tuple), each of size a power of two. Along one dimension it is [`fft`](@ref) of each column
or row, with the bound of that fiber. Along both, the columns are transformed and then the rows,
and every output has the radius

    √2 · b_n · ‖X‖∞⊥ + ‖r‖₁,        n = log₂(number of entries),

with the norms taken over the whole matrix.

The bound along both dimensions is not in the paper; it follows from the bound it quotes for
the relative error in the Euclidean norm of the one-dimensional algorithm, `ε_m = (1 + u)ᵐ
∏_{j ≤ m} (1 + g_j) − 1` for size `2ᵐ`. Applied to the columns and summed over them, and then to
the rows, it gives `‖Ŷ − Y‖_F ≤ ((1 + ε_{n₁})(1 + ε_{n₂}) − 1) ‖Y‖_F`, the exact transform
multiplying the Frobenius norm by the square root of the size of a fiber; and
`(1 + ε_{n₁})(1 + ε_{n₂}) ≤ 1 + ε_n` for `n = n₁ + n₂`. An entry is at most the Frobenius norm,
and `‖Y‖_F = √N ‖X‖_F ≤ √2 N ‖X‖∞⊥` with `N = 2ⁿ`, which is `b_n ‖X‖∞⊥`.
"""
function FFTW.fft(A::BallMatrix{T}, dims = (1, 2)) where {T}
    dims_t = dims isa Integer ? (Int(dims),) : Tuple(Int.(dims))
    all(d -> d in (1, 2), dims_t) && allunique(dims_t) ||
        throw(ArgumentError("dims must be 1, 2 or both"))
    n = sum(_require_power_of_two(size(A.c, d)) for d in dims_t)
    bn = _bmp_bn(n, T)

    Ŷ = Matrix{Complex{T}}(A.c)
    if 1 in dims_t
        for j in axes(Ŷ, 2)
            Ŷ[:, j] = _bmp_radix2_fft(Ŷ[:, j])
        end
    end
    if 2 in dims_t
        for i in axes(Ŷ, 1)
            Ŷ[i, :] = _bmp_radix2_fft(Ŷ[i, :])
        end
    end
    all(isfinite, Ŷ) || throw(ArgumentError("the transform overflowed"))

    err = zeros(T, size(A.c))
    if length(dims_t) == 2
        x_inf, r_1 = _fiber_inf_and_l1(A.c, A.r, T)
        fill!(err, _fft_err(bn, x_inf, r_1))
    elseif dims_t[1] == 1
        for j in axes(err, 2)
            x_inf, r_1 = _fiber_inf_and_l1(view(A.c, :, j), view(A.r, :, j), T)
            err[:, j] .= _fft_err(bn, x_inf, r_1)
        end
    else
        for i in axes(err, 1)
            x_inf, r_1 = _fiber_inf_and_l1(view(A.c, i, :), view(A.r, i, :), T)
            err[i, :] .= _fft_err(bn, x_inf, r_1)
        end
    end
    return BallMatrix(Ŷ, err)
end

end
