using Test
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
