# Tests of ball_fft, the discrete Fourier transform of balls with the bound of Brisebarre, Muller
# and Picot (2023): the roots of unity, the constant, and containment against a direct sum in
# BigFloat.

using Test
using BallArithmetic
using LinearAlgebra
using Random

@testset "ball_fft" begin
    function _bigfloat_dft(c::AbstractVector{<:Complex})
        N = length(c)
        cb = Complex{BigFloat}.(c)
        out = zeros(Complex{BigFloat}, N)
        for k in 0:(N - 1), j in 0:(N - 1)
            out[k + 1] += cb[j + 1] * exp(-2 * BigFloat(π) * im * j * k / N)
        end
        return out
    end

    # the constant vector: every output but the first is zero
    v = BallVector(ones(1024), fill(2.0^-20, 1024))
    y = ball_fft(v)
    @test 1024 in y[1]
    @test all(0 in y[k] for k in 2:1024)
    @test_throws ArgumentError ball_fft(BallVector(ones(1023)))
    @test_throws ArgumentError ball_fft(BallMatrix(ones(8, 15)))
    @test_throws ArgumentError ball_fft(v; method = :other)

    # The algorithm under the bound: the radix-2 transform of the paper's Fig. 1.
    setprecision(BigFloat, 512) do
        # the roots of unity are the nearest floats to the exact ones
        # from the table stored with the package (N up to 2^16) and computed at the first use
        for T in (Float64, Float32), N in (2, 8, 64, 2^16, 2^17)
            w, stride = BallArithmetic._brisebarreetal2023_roots(T, N)
            @test stride == (N <= 2^16 ? 2^16 ÷ N : 1)
            for j in unique(vcat(0:min(N ÷ 2 - 1, 40), N ÷ 4, N ÷ 2 - 1))
                θ = BigFloat(2 * j) / N
                @test real(w[j * stride + 1]) == T(cospi(θ))
                @test imag(w[j * stride + 1]) == -T(sinpi(θ))
            end
        end
        wb, sb = setprecision(() -> BallArithmetic._brisebarreetal2023_roots(BigFloat, 8),
            BigFloat, 128)
        @test sb == 1 && precision(real(wb[2])) == 128
        # a size above the stored table
        c = randn(ComplexF64, 2^17)
        yb = ball_fft(BallVector(c))
        @test abs(sum(big.(c)) - big(mid(yb)[1])) <= big(rad(yb)[1])
        # the constant of equation (4), against its formula in high precision
        for T in (Float64, Float32), n in (1, 2, 3, 10, 20)
            u = big(eps(T)) / 2
            g = u + sqrt(big(5)) * u * (1 + u)
            exact = sqrt(big(2)) * big(2)^n * ((1 + u)^n * (1 + g)^max(n - 2, 0) - 1)
            b = BallArithmetic._brisebarreetal2023_bn(n, T)
            @test big(b) >= exact
            @test big(b) <= exact * (1 + 1e-3)
        end
        # the transform itself, and the bound without any radius: the error of the algorithm
        for T in (Float64, Float32), N in (1, 2, 4, 16, 256), trial in 1:3
            c = randn(Complex{T}, N)
            ŷ = ball_fft(BallVector(c))
            @test eltype(mid(ŷ)) == Complex{T}
            y_true = _bigfloat_dft(c)
            @test all(abs(y_true[k] - big(mid(ŷ)[k])) <= big(rad(ŷ)[k]) for k in 1:N)
            # the bound is not far above the error: within the factor the paper reports
            N >= 16 && T === Float64 && @test maximum(rad(ŷ)) <= 1e-9 * maximum(abs, c)
        end
        # real input, and a BigFloat one
        r = randn(32)
        ŷ = ball_fft(BallVector(r, fill(1e-13, 32)))
        y_true = _bigfloat_dft(complex(r))
        @test all(abs(y_true[k] - big(mid(ŷ)[k])) <= big(rad(ŷ)[k]) for k in 1:32)
        setprecision(BigFloat, 256) do
            cb = Complex{BigFloat}.(randn(ComplexF64, 16))
            ŷb = ball_fft(BallVector(cb))
            yb = setprecision(() -> _bigfloat_dft(cb), BigFloat, 512)
            @test all(abs(yb[k] - mid(ŷb)[k]) <= rad(ŷb)[k] for k in 1:16)
            @test maximum(rad(ŷb)) < 1e-70
        end
        # both dimensions: against the transform of the columns then of the rows
        for (m, n) in ((4, 8), (16, 16)), trial in 1:2
            C = randn(ComplexF64, m, n)
            B = BallMatrix(C, fill(1e-12, m, n))
            F2 = ball_fft(B)
            Y = Matrix{Complex{BigFloat}}(undef, m, n)
            for j in 1:n
                Y[:, j] = _bigfloat_dft(C[:, j])
            end
            for i in 1:m
                Y[i, :] = _bigfloat_dft(Y[i, :])
            end
            @test all(abs(Y[i, j] - big(mid(F2)[i, j])) <= big(rad(F2)[i, j])
            for i in 1:m, j in 1:n)
            @test ball_fft(B, 1).c == ball_fft(B, (1,)).c
        end
        @test_throws ArgumentError ball_fft(BallMatrix(ones(4, 4)), (1, 3))
    end

end
