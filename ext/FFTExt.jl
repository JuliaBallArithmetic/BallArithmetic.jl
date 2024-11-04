module FFTExt

using BallArithmetic, LinearAlgebra, RoundingEmulator
import BallArithmetic: @up, div_up, add_up, sub_up, mul_up
import FFTW

"""
    fft

Computes the FFT of a BallMatrix using the a priori error bound in
Ref. [Higham1996](@cite)

# References

* [Higham1996](@cite) Higham, Siam (1996)
"""

function FFTW.fft(A::BallMatrix{T}, dims = (1, 2)) where {T}
    if all([!ispow2(size(A.c)[i]) for i in dims])
        @warn "The rigorous error estimate works for power of two sizes"
    end

    FFTAc = FFTW.fft(A.c, dims)
    N = prod([size(A.c)[i] for i in dims])

    #norms = [upper_bound_norm(x, r) for (x, r) in zip(eachslice(A.c; dims), eachslice(A.r; dims))]

    norms_c = setrounding(T, RoundUp) do
        return [LinearAlgebra.norm(v) for v in eachslice(A.c; dims)]
    end

    norms_r = setrounding(T, RoundUp) do
        return [LinearAlgebra.norm(v) for v in eachslice(A.r; dims)]
    end

    μ = eps(eltype(A.c))
    u = eps(eltype(A.c))
    γ4 = @up 4.0 * u / (1.0 - 4.0 * u)
    η = @up μ + γ4 * (sqrt_up(2.0) + μ)

    l = @up log2(N) / sqrt_down(T(N))

    err = @up l .* (η / (1.0 - η) .* norms_c .+ norms_r)

    err_M = repeat(err, outer = size(A.c))
    #err_M = vcat([err for _ in 1:N]...)

    #@info err_M
    #reshape(err_M, size(A.r))

    return BallMatrix(FFTAc, err_M)
end

"""
    fft

Computes the FFT of a BallVector using the a priori error bound in
Ref. [Higham1996](@cite)

# References

* [Higham1996](@cite) Higham, Siam (1996)
"""
function FFTW.fft(v::BallVector{T}) where {T}
    if !ispow2(length(v))
        @warn "The rigorous error estimate works for power of two sizes"
    end

    FFTAc = FFTW.fft(v.c)
    N = length(v)

    #norms = [upper_bound_norm(x, r) for (x, r) in zip(eachslice(A.c; dims), eachslice(A.r; dims))]

    norms_c = setrounding(T, RoundUp) do
        return norm(v.c)
    end

    norms_r = setrounding(T, RoundUp) do
        return norm(v.r)
    end

    μ = eps(eltype(v.c))
    u = eps(eltype(v.c))
    γ4 = @up 4.0 * u / (1.0 - 4.0 * u)
    η = @up μ + γ4 * (sqrt_up(2.0) + μ)

    l = @up log2(N) / sqrt_down(T(N))

    err = @up l * (η / (1.0 - η) * norms_c + norms_r)
    err_M = fill(err, N)

    return BallVector(FFTAc, err_M)
end

end
