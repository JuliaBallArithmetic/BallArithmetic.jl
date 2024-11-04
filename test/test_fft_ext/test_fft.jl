@testset "Test FFT external module" begin
    using FFTW

    dims = 1024

    v = ones(dims)

    v_rad = ones(dims) / 2^20

    v = BallVector(v, v_rad)

    fft_v = BallArithmetic.fft(v)

    @test 1024 in fft_v[1]

    A = BallMatrix(zeros(1024, 2))

    A[:, 1] = v
    fftA = BallArithmetic.fft(A, (1,))
    @test 1024 in fftA[1, 1]

    dims = 1023

    v = ones(dims)
    v_rad = ones(dims) / 2^20
    v = BallVector(v, v_rad)

    @test_logs (:warn, "The rigorous error estimate works for power of two sizes") BallArithmetic.fft(v)

    # A = BallMatrix(zeros(1023, 2))
    # @test_logs (:warn, "The rigorous error estimate works for power of two sizes") BallArithmetic.fft(A)

end
