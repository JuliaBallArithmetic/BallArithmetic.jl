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

@testset "Leading triangular Sylvester block" begin
    A = UpperTriangular([2.0 1.0 0.5; 0.0 3.0 -0.2; 0.0 0.0 4.0])
    B = UpperTriangular([1.5 -0.3 0.4; 0.0 0.75 0.6; 0.0 0.0 -1.2])
    C = [0.4 0.1 -0.2; -0.3 0.25 0.0; 0.2 -0.1 0.5]

    X11 = BallArithmetic.solve_leading_triangular_sylvester(A, B, C)
    @test size(X11) == (1, 1)
    @test isapprox(A[1, 1] * X11[1, 1] + X11[1, 1] * B[1, 1], C[1, 1]; atol = 1e-12)

    k = 2
    X_block = BallArithmetic.solve_leading_triangular_sylvester(A, B, C, k)
    A_block = Matrix(A[1:k, 1:k])
    B_block = Matrix(B[1:k, 1:k])
    C_block = Matrix(C[1:k, 1:k])
    @test A_block * X_block + X_block * B_block ≈ C_block atol = 1e-12

    singular_A = UpperTriangular([1.0 0.0; 0.0 -1.0])
    singular_B = UpperTriangular([-1.0 0.0; 0.0 2.0])
    singular_C = zeros(2, 2)
    @test_throws ArgumentError BallArithmetic.solve_leading_triangular_sylvester(singular_A, singular_B, singular_C)

    nontriangular = [1.0 1.0 0.0; 0.5 2.0 0.1; 0.0 0.0 3.0]
    @test_throws ArgumentError BallArithmetic.solve_leading_triangular_sylvester(nontriangular, B, C)
end
