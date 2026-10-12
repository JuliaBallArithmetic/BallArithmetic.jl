module FFTExt

using BallArithmetic
import FFTW

# The transform of balls is `ball_fft`, in the package. This extension gives it FFTW's name,
# and the floating-point transform by FFTW for the method `:brisebarreetal2023fftw`.

BallArithmetic._fftw_transform(x::Vector{<:Complex}) = FFTW.fft(x)

"""
    fft(v::BallVector; method = :brisebarreetal2023)

[`ball_fft`](@ref)`(v; method)`.
"""
FFTW.fft(v::BallVector; method::Symbol = :brisebarreetal2023) = ball_fft(v; method)

"""
    fft(A::BallMatrix, dims = (1, 2); method = :brisebarreetal2023)

[`ball_fft`](@ref)`(A, dims; method)`.
"""
FFTW.fft(A::BallMatrix, dims = (1, 2); method::Symbol = :brisebarreetal2023) =
    ball_fft(A, dims; method)

end
