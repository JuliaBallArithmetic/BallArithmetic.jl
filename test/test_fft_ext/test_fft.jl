@testset "Test FFT external module" begin
    using FFTW

    #ext = Base.get_extension(@__MODULE__, :FFTExt)

    dims = 1024

    v = ones(dims)

    v_rad = ones(dims) / 2^20

    v = BallVector(v, v_rad)

    fft_v = fft(v)

    @test 1024 in fft_v[1]

    A = BallMatrix(zeros(1024, 2))

    A[:, 1] = v
    fftA = fft(A, (1,))
    @test 1024 in fftA[1, 1]

    dims = 1023

    v = ones(dims)
    v_rad = ones(dims) / 2^20
    v = BallVector(v, v_rad)

    # the bound is for powers of two: another length is refused
    @test_throws ArgumentError fft(v)

    # 2D FFT: DC of constant matrix equals m·n.
    A2 = BallMatrix(ones(8, 16))
    fftA2 = fft(A2)
    @test 8 * 16 in fftA2[1, 1]
    @test size(fftA2.c) == size(A2.c)

    # 2D FFT: warning fires if any transformed dim is non-pow2.
    A_np2 = BallMatrix(ones(8, 15))
    @test_throws ArgumentError fft(A_np2)

    # Rigorous enclosure: random Float64 inputs vs naive BigFloat DFT.
    # Asserts the BMP 2023 bound encloses the true FFT for every output.
    function _bigfloat_dft(c::AbstractVector{<:Complex})
        N = length(c)
        cb = Complex{BigFloat}.(c)
        out = zeros(Complex{BigFloat}, N)
        for k in 0:(N - 1), j in 0:(N - 1)
            out[k + 1] += cb[j + 1] * exp(-2 * BigFloat(π) * im * j * k / N)
        end
        return out
    end

    setprecision(BigFloat, 256) do
        for N in (8, 32, 128), trial in 1:5
            c = randn(ComplexF64, N)
            r = abs.(randn(N)) .* 1e-10
            v = BallVector(c, r)
            ŷ = fft(v)
            y_true = _bigfloat_dft(c)
            for k in 1:N
                @test y_true[k] in ŷ[k]
            end
        end

        # Matrix path along columns (dims = (1,)) and rows (dims = (2,)).
        for (m, n) in ((16, 4), (8, 8)), trial in 1:3
            C = randn(ComplexF64, m, n)
            R = abs.(randn(m, n)) .* 1e-12
            B = BallMatrix(C, R)

            # Per-column FFT.
            FB = fft(B, (1,))
            for j in 1:n
                yref = _bigfloat_dft(C[:, j])
                for i in 1:m
                    @test yref[i] in FB[i, j]
                end
            end

            # Per-row FFT.
            FB2 = fft(B, (2,))
            for i in 1:m
                yref = _bigfloat_dft(C[i, :])
                for j in 1:n
                    @test yref[j] in FB2[i, j]
                end
            end
        end
    end
    # The algorithm under the bound: the radix-2 transform of the paper's Fig. 1.
    ext = Base.get_extension(BallArithmetic, :FFTExt)
    setprecision(BigFloat, 512) do
        # the roots of unity are the nearest floats to the exact ones
        for T in (Float64, Float32), N in (2, 8, 64)
            w = ext._bmp_roots(T, N)
            @test length(w) == max(N ÷ 2, 1)
            for j in 0:(length(w) - 1)
                θ = BigFloat(2 * j) / N
                @test real(w[j + 1]) == T(cospi(θ))
                @test imag(w[j + 1]) == -T(sinpi(θ))
            end
        end
        # the constant of equation (4), against its formula in high precision
        for T in (Float64, Float32), n in (1, 2, 3, 10, 20)
            u = big(eps(T)) / 2
            g = u + sqrt(big(5)) * u * (1 + u)
            exact = sqrt(big(2)) * big(2)^n * ((1 + u)^n * (1 + g)^max(n - 2, 0) - 1)
            b = ext._bmp_bn(n, T)
            @test big(b) >= exact
            @test big(b) <= exact * (1 + 1e-3)
        end
        # the transform itself, and the bound without any radius: the error of the algorithm
        for T in (Float64, Float32), N in (1, 2, 4, 16, 256), trial in 1:3
            c = randn(Complex{T}, N)
            ŷ = fft(BallVector(c))
            @test eltype(mid(ŷ)) == Complex{T}
            y_true = _bigfloat_dft(c)
            @test all(abs(y_true[k] - big(mid(ŷ)[k])) <= big(rad(ŷ)[k]) for k in 1:N)
            # the bound is not far above the error: within the factor the paper reports
            N >= 16 && T === Float64 && @test maximum(rad(ŷ)) <= 1e-9 * maximum(abs, c)
        end
        # real input, and a BigFloat one
        r = randn(32)
        ŷ = fft(BallVector(r, fill(1e-13, 32)))
        y_true = _bigfloat_dft(complex(r))
        @test all(abs(y_true[k] - big(mid(ŷ)[k])) <= big(rad(ŷ)[k]) for k in 1:32)
        setprecision(BigFloat, 256) do
            cb = Complex{BigFloat}.(randn(ComplexF64, 16))
            ŷb = fft(BallVector(cb))
            yb = setprecision(() -> _bigfloat_dft(cb), BigFloat, 512)
            @test all(abs(yb[k] - mid(ŷb)[k]) <= rad(ŷb)[k] for k in 1:16)
            @test maximum(rad(ŷb)) < 1e-70
        end
        # both dimensions: against the transform of the columns then of the rows
        for (m, n) in ((4, 8), (16, 16)), trial in 1:2
            C = randn(ComplexF64, m, n)
            B = BallMatrix(C, fill(1e-12, m, n))
            F2 = fft(B)
            Y = Matrix{Complex{BigFloat}}(undef, m, n)
            for j in 1:n
                Y[:, j] = _bigfloat_dft(C[:, j])
            end
            for i in 1:m
                Y[i, :] = _bigfloat_dft(Y[i, :])
            end
            @test all(abs(Y[i, j] - big(mid(F2)[i, j])) <= big(rad(F2)[i, j])
            for i in 1:m, j in 1:n)
            @test fft(B, 1).c == fft(B, (1,)).c
        end
        @test_throws ArgumentError fft(BallMatrix(ones(4, 4)), (1, 3))
    end

end
