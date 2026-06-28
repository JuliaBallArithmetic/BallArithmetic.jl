module FFTExt

using BallArithmetic, LinearAlgebra
import BallArithmetic: machine_epsilon, add_up, sub_up, mul_up, sqrt_up
import FFTW

# Per-component absolute error constant from
# Brisebarre, Muller, Picot, "Testing the Sharpness of Known Error
# Bounds on the Fast Fourier Transform", ARITH 2023, Eq. (4):
#
#     b_n = √2 · 2^n · ((1+u) ∏_{j=3}^{n} (1+g_j) − 1)
#
# with g_j = δ_j + ρ(1+δ_j); δ_j ≤ u (correctly-rounded roots of
# unity) and ρ ≤ √5·u (relative error of complex multiplication
# without FMA — conservative). For 1+g_j = (1+δ)(1+ρ) the product
# collapses to ((1+u)(1+ρ))^(n−2). Each Re/Im component of the FP
# error then satisfies max(|Re|, |Im|) ≤ b_n · ‖X‖∞⊥, where
# ‖X‖∞⊥ := max_k max(|Re(x_k)|, |Im(x_k)|).
function _bmp_bn(n::Integer, ::Type{T}) where {T <: AbstractFloat}
    u = machine_epsilon(T)
    sqrt2 = sqrt_up(T(2))
    sqrt5 = sqrt_up(T(5))
    ρ = mul_up(sqrt5, u)
    one_plus_g = mul_up(add_up(T(1), u), add_up(T(1), ρ))
    prod_g = T(1)
    for _ in 3:n
        prod_g = mul_up(prod_g, one_plus_g)
    end
    bracket = sub_up(mul_up(add_up(T(1), u), prod_g), T(1))
    return mul_up(sqrt2, mul_up(T(2)^n, bracket))
end

# Compute ‖slice‖∞⊥ (component-wise sup of |Re|,|Im| of midpoints)
# and an upper bound on ‖slice_radii‖_1. abs/real/imag/max are exact
# on FP, so x_inf needs no directed rounding; the radius sum uses
# add_up to remain a rigorous upper bound.
function _fiber_inf_and_l1(c_slice, r_slice, ::Type{T}) where {T <: AbstractFloat}
    x_inf = T(0)
    @inbounds for z in c_slice
        a = max(T(abs(real(z))), T(abs(imag(z))))
        if a > x_inf
            x_inf = a
        end
    end
    r_1 = T(0)
    @inbounds for r in r_slice
        r_1 = add_up(r_1, T(r))
    end
    return x_inf, r_1
end

# √2 · b_n · x_inf + r_1, all rounded toward +∞.
function _fft_err(bn::T, sqrt2::T, x_inf::T, r_1::T) where {T <: AbstractFloat}
    return add_up(mul_up(sqrt2, mul_up(bn, x_inf)), r_1)
end

"""
    fft(v::BallVector{T})

Compute the FFT of a `BallVector` with a rigorous a-priori error
bound from [BrisebarreMullerPicot2023](@cite), Eq. (4). Each output
coefficient is enclosed by

    Ŷ_k ± (√2 · b_n · ‖X‖∞⊥ + ‖r‖₁),

where `b_n` is the BMP constant for a radix-2 FFT of length
`N = 2ⁿ`, `‖X‖∞⊥` is the component-wise ∞-norm of Re/Im parts of
the input midpoints, and `‖r‖₁` is the 1-norm of the input radii.
The latter is exact: each output is a sum of inputs weighted by
unit-modulus roots of unity, so the radius contribution per output
is bounded by the sum of input radii.

# References

* [BrisebarreMullerPicot2023](@cite) Brisebarre, Muller, Picot,
  ARITH 2023.
* [KnightKaiser1979](@cite) Knight, Kaiser, IEEE TASSP 27(6), 1979.
"""
function FFTW.fft(v::BallVector{T}) where {T}
    N = length(v)
    if !ispow2(N)
        @warn "The rigorous error estimate works for power of two sizes"
    end
    Ŷ = FFTW.fft(v.c)
    n = round(Int, log2(N))
    bn = _bmp_bn(n, T)
    sqrt2 = sqrt_up(T(2))
    x_inf, r_1 = _fiber_inf_and_l1(v.c, v.r, T)
    err_val = _fft_err(bn, sqrt2, x_inf, r_1)
    return BallVector(Ŷ, fill(err_val, N))
end

"""
    fft(A::BallMatrix{T}, dims = (1, 2))

Compute the FFT of a `BallMatrix` along the dimensions `dims`. The
[BrisebarreMullerPicot2023](@cite) bound is applied per fiber: for
each slice of `A` along `dims`, the per-output bound

    √2 · b_n · ‖X_fiber‖∞⊥ + ‖r_fiber‖₁

is computed independently with `n = log₂(prod(size(A,d) for d in
dims))`. Outputs in the same fiber share the same radius.

`dims` may be a single integer, a tuple, or any iterable of axis
indices.
"""
function FFTW.fft(A::BallMatrix{T}, dims = (1, 2)) where {T}
    dims_t = dims isa Integer ? (Int(dims),) : Tuple(Int.(dims))
    if any(!ispow2(size(A.c, d)) for d in dims_t)
        @warn "The rigorous error estimate works for power of two sizes"
    end

    Ŷ = FFTW.fft(A.c, dims_t)
    Ntot = prod(size(A.c, d) for d in dims_t)
    n = round(Int, log2(Ntot))
    bn = _bmp_bn(n, T)
    sqrt2 = sqrt_up(T(2))

    iter_dims = Tuple(setdiff(1:2, dims_t))
    err = zeros(T, size(A.c))

    if isempty(iter_dims)
        x_inf, r_1 = _fiber_inf_and_l1(A.c, A.r, T)
        fill!(err, _fft_err(bn, sqrt2, x_inf, r_1))
    else
        iter_sz = ntuple(i -> size(A.c, iter_dims[i]), length(iter_dims))
        for I in CartesianIndices(iter_sz)
            slice = ntuple(2) do d
                if d in dims_t
                    return Colon()
                else
                    pos = findfirst(==(d), iter_dims)
                    return I[pos]
                end
            end
            x_inf, r_1 = _fiber_inf_and_l1(view(A.c, slice...),
                                           view(A.r, slice...),
                                           T)
            view(err, slice...) .= _fft_err(bn, sqrt2, x_inf, r_1)
        end
    end

    return BallMatrix(Ŷ, err)
end

end
