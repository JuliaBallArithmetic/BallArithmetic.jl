using Test
using LinearAlgebra
using BallArithmetic

@testset "CertifScripts serial" begin
    circle = BallArithmetic.CertifScripts.CertificationCircle(0.0, 0.25; samples = 8)
    A = BallArithmetic.BallMatrix(Matrix{ComplexF64}(I, 2, 2))
    result = BallArithmetic.CertifScripts.run_certification(A, circle; η = 0.9, check_interval = 4, log_io = IOBuffer())
    @test !isempty(result.certification_log)
    @test result.minimum_singular_value > 0
    @test result.resolvent_original >= result.resolvent_schur
end

@testset "CertifScripts serial with schur_data" begin
    A = BallArithmetic.BallMatrix(Matrix{ComplexF64}(I, 2, 2))
    # Pre-compute Schur data once
    sd = BallArithmetic.CertifScripts.compute_schur_and_error(A)
    S, errF, errT, norm_Z, norm_Z_inv = sd

    # Reuse the same Schur data for two different circles (avoids recomputation)
    for center in [0.0, 0.5]
        circle = BallArithmetic.CertifScripts.CertificationCircle(center, 0.25; samples = 8)
        result = BallArithmetic.CertifScripts.run_certification(A, circle;
            schur_data = sd, η = 0.9, check_interval = 4, log_io = IOBuffer())
        @test !isempty(result.certification_log)
        @test result.minimum_singular_value > 0
        @test result.errF == errF
        @test result.errT == errT
    end
end

@testset "CertifScripts distributed" begin
    using Distributed
    circle = BallArithmetic.CertifScripts.CertificationCircle(0.0, 0.25; samples = 8)
    A = BallArithmetic.BallMatrix(Matrix{ComplexF64}(I, 2, 2))
    result = BallArithmetic.CertifScripts.run_certification(A, circle, 1; η = 0.9, check_interval = 4, log_io = IOBuffer(), channel_capacity = 4)
    @test result.circle == circle
    @test !isempty(result.certification_log)

    pids = addprocs(1)
    pool = WorkerPool(pids)
    try
        pooled_result = BallArithmetic.CertifScripts.run_certification(A, circle, pool; η = 0.9, check_interval = 4, log_io = IOBuffer(), channel_capacity = 4)
        @test pooled_result.circle == circle
        @test !isempty(pooled_result.certification_log)

        reuse_result = BallArithmetic.CertifScripts.run_certification(A, circle, pool; η = 0.9, check_interval = 4, log_io = IOBuffer(), channel_capacity = 4)
        @test reuse_result.circle == circle
        @test !isempty(reuse_result.certification_log)
    finally
        rmprocs(pids)
    end
end

@testset "Polynomial helpers" begin
    coeffs = BallArithmetic.CertifScripts.poly_from_roots([1, 2, 3])
    @test coeffs ≈ [-6.0, 11.0, -6.0, 1.0]
end

@testset "Sylvester Miyajima enclosure" begin
    A = [3.0 1.0 0.0; 0.0 2.5 0.3; 0.0 0.0 4.0]
    B = [-1.5 0.2; 0.0 -0.75]
    C = [1.0 0.5; -0.2 0.8; 0.3 -0.4]

    n = size(B, 1)
    K = kron(Matrix{Float64}(I, n, n), A) + kron(transpose(B), Matrix{Float64}(I, size(A, 1), size(A, 1)))
    X_exact = reshape(K \ vec(C), size(A, 1), size(B, 1))

    X̃ = X_exact .+ 1e-12 .* ones(size(X_exact))
    enclosure = BallArithmetic.sylvester_miyajima_enclosure(A, B, C, X̃)

    @test all(rad(enclosure) .>= 0)
    diff = abs.(X_exact .- mid(enclosure))
    @test all(diff .<= rad(enclosure))
end

@testset "Triangular Miyajima Sylvester block" begin
    T = UpperTriangular([2.0 + 0.2im  0.3 - 0.1im   0.5 + 0.4im  -0.2 + 0.3im;
                         0.0          1.5 + 0.6im  -0.1 - 0.2im  0.4 + 0.1im;
                         0.0          0.0           2.7 - 0.5im  0.6 - 0.3im;
                         0.0          0.0           0.0           3.4 + 0.2im])
    k = 2

    enclosure = BallArithmetic.triangular_sylvester_miyajima_enclosure(T, k)
    @test size(enclosure) == (size(T, 1) - k, k)

    Tmat = Matrix(T)
    T11 = Matrix(Tmat[1:k, 1:k])
    T22 = Matrix(Tmat[k+1:end, k+1:end])
    T12 = Matrix(Tmat[1:k, k+1:end])

    A = Matrix(adjoint(T22))
    B = -Matrix(adjoint(T11))
    C = Matrix(adjoint(T12))
    In = Matrix{eltype(A)}(I, size(B, 1), size(B, 1))
    Im = Matrix{eltype(A)}(I, size(A, 1), size(A, 1))
    K = kron(In, A) + kron(transpose(B), Im)
    Y_exact = reshape(K \ vec(C), size(A, 1), size(B, 1))

    diff = abs.(Y_exact .- mid(enclosure))
    @test all(diff .<= rad(enclosure))

    @test_throws ArgumentError BallArithmetic.triangular_sylvester_miyajima_enclosure(T, size(T, 1))

    nontriangular = Matrix(T)
    nontriangular[end, 1] = 1.0
    @test_throws ArgumentError BallArithmetic.triangular_sylvester_miyajima_enclosure(nontriangular, k)
end
