using Test
using LinearAlgebra
using BallArithmetic

@testset "Newton-Kantorovich Eigenpair Certifier" begin

    @testset "2×2 symmetric [2 1; 1 2]" begin
        A = BallMatrix([2.0 1.0; 1.0 2.0])
        eig = eigen(Hermitian(mid(A)))

        for i in 1:2
            result = certify_eigenpair(A, eig.values[i], eig.vectors[:, i])
            @test result.verified
            @test eig.values[i] ∈ result.eigenvalue
            @test result.enclosure_radius < 1e-13
            @test result.defect_q < 1.0
            @test result.discriminant > 0.0
        end
    end

    @testset "Diagonal matrix diag(1, 5, 10)" begin
        A = BallMatrix(Diagonal([1.0, 5.0, 10.0]))
        eig = eigen(mid(A))

        for i in 1:3
            result = certify_eigenpair(A, eig.values[i], eig.vectors[:, i])
            @test result.verified
            @test eig.values[i] ∈ result.eigenvalue
            @test result.enclosure_radius < 1e-14
        end
    end

    @testset "3×3 non-symmetric with real eigenvalues" begin
        A_mid = [3.0 1.0 0.0;
                 0.0 2.0 1.0;
                 0.0 0.0 1.0]
        A = BallMatrix(A_mid)
        eig = eigen(A_mid)

        for i in 1:3
            result = certify_eigenpair(A, eig.values[i], eig.vectors[:, i])
            @test result.verified
            @test real(eig.values[i]) ∈ result.eigenvalue
            @test result.enclosure_radius < 1e-12
        end
    end

    @testset "Complex eigenvalues (rotation-like matrix)" begin
        θ = π / 4
        A_mid = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        A = BallMatrix(A_mid)
        eig = eigen(A_mid)

        for i in 1:2
            result = certify_eigenpair(A, eig.values[i], eig.vectors[:, i])
            @test result.verified
            @test result.enclosure_radius < 1e-13
        end
    end

    @testset "Batch certify_eigenpairs" begin
        A = BallMatrix([4.0 1.0; 1.0 3.0])
        result = certify_eigenpairs(A; hermitian=true)

        @test result.n_total == 2
        @test result.n_verified == 2
        @test length(result) == 2

        for r in result
            @test r.verified
            @test r.enclosure_radius < 1e-13
        end
    end

    @testset "Selective indices certify_eigenpairs" begin
        A = BallMatrix(Diagonal([1.0, 5.0, 10.0]))
        result = certify_eigenpairs(A; indices=[1, 3])

        @test result.n_total == 2
        @test result.n_verified == 2
        @test length(result) == 2

        # First result corresponds to eigenvalue 1.0
        @test 1.0 ∈ result[1].eigenvalue
        # Second result corresponds to eigenvalue 10.0
        @test 10.0 ∈ result[2].eigenvalue
    end

    @testset "Near-defective matrix returns verified=false" begin
        # Jordan-like matrix with repeated eigenvalue
        ε = 1e-15
        A_mid = [1.0 1.0; ε 1.0]
        A = BallMatrix(A_mid)
        eig = eigen(A_mid)

        # At least one eigenpair should fail to certify (near-defective)
        results = [certify_eigenpair(A, eig.values[i], eig.vectors[:, i]) for i in 1:2]
        # The pair is nearly defective so at least one should fail or have large radius
        any_failed = any(r -> !r.verified || r.enclosure_radius > 1e-3, results)
        @test any_failed
    end

    @testset "Result struct fields" begin
        A = BallMatrix([2.0 0.0; 0.0 3.0])
        result = certify_eigenpair(A, 2.0, [1.0, 0.0])

        @test result isa NKEigenpairResult{Float64, Float64}
        @test result.verified == true
        @test result.eigenvalue isa Ball{Float64, Float64}
        @test length(result.eigenvector) == 2
        @test result.enclosure_radius isa Float64
        @test result.defect_q isa Float64
        @test result.C_norm isa Float64
        @test result.residual_y isa Float64
        @test result.discriminant isa Float64
    end

    @testset "NKEigenpairsResult iteration" begin
        A = BallMatrix(Diagonal([1.0, 2.0, 3.0]))
        result = certify_eigenpairs(A)

        @test length(result) == 3
        @test firstindex(result) == 1
        @test lastindex(result) == 3

        # Test iteration
        count = 0
        for r in result
            count += 1
            @test r isa NKEigenpairResult
        end
        @test count == 3
    end

    @testset "Non-hermitian batch" begin
        A_mid = [2.0 1.0; 0.5 3.0]
        A = BallMatrix(A_mid)
        result = certify_eigenpairs(A; hermitian=false)

        @test result.n_total == 2
        @test result.n_verified == 2
    end

    @testset "norm_method=:svd option" begin
        A = BallMatrix([2.0 1.0; 1.0 2.0])
        eig = eigen(Hermitian(mid(A)))
        result = certify_eigenpair(A, eig.values[1], eig.vectors[:, 1]; norm_method=:svd)
        @test result.verified
    end
end
