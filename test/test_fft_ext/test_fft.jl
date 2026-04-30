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

    @test_logs (:warn, "The rigorous error estimate works for power of two sizes") fft(v)

    # 2D FFT: DC of constant matrix equals m·n.
    A2 = BallMatrix(ones(8, 16))
    fftA2 = fft(A2)
    @test 8 * 16 in fftA2[1, 1]
    @test size(fftA2.c) == size(A2.c)

    # 2D FFT: warning fires if any transformed dim is non-pow2.
    A_np2 = BallMatrix(ones(8, 15))
    @test_logs (:warn, "The rigorous error estimate works for power of two sizes") fft(A_np2)

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
end
