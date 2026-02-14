using Test
using LinearAlgebra
using BallArithmetic

@testset "Rump2022a Eigenvalue Bounds" begin
    @testset "Basic 2×2 Hermitian matrix" begin
        A = BallMatrix([2.0 1.0; 1.0 2.0])
        result = rump_2022a_eigenvalue_bounds(A; hermitian=true)

        @test length(result) == 2
        @test result.verified

        # True eigenvalues are 1.0 and 3.0
        λ_true = [1.0, 3.0]
        for i in 1:2
            @test λ_true[i] ∈ result.eigenvalues[i]
        end

        # Check that condition numbers are computed
        @test all(κ -> κ >= 1.0, result.condition_numbers)
        @test all(κ -> isfinite(κ), result.condition_numbers)

        # Check residual norms
        @test all(r -> r >= 0, result.residual_norms)
        @test all(r -> r < 1e-10, result.residual_norms)  # Should be small
    end

    @testset "Diagonal matrix (well-separated eigenvalues)" begin
        A = BallMatrix(Diagonal([1.0, 5.0, 10.0]))
        result = rump_2022a_eigenvalue_bounds(A; hermitian=true)

        @test length(result) == 3
        @test result.verified

        # Eigenvalues should be very accurate
        for i in 1:3
            λ_true = [1.0, 5.0, 10.0][i]
            @test λ_true ∈ result.eigenvalues[i]
            @test rad(result.eigenvalues[i]) < 1e-14
        end

        # Separation gaps should be large
        @test result.separation_gaps[1] ≥ 4.0  # min(5-1, 10-1)
        @test result.separation_gaps[2] ≥ 4.0  # min(5-1, 10-5)
        @test result.separation_gaps[3] ≥ 5.0  # min(10-5, 10-1)
    end

    @testset "Matrix with close eigenvalues" begin
        # Matrix with clustered eigenvalues
        A_mid = Matrix(Diagonal([1.0, 1.1, 5.0]))
        A_rad = zeros(size(A_mid))
        A_rad[1, 2] = A_rad[2, 1] = 0.01  # Small coupling
        A = BallMatrix(A_mid, A_rad)

        result = rump_2022a_eigenvalue_bounds(A; hermitian=true)

        @test length(result) == 3

        # First two eigenvalues are close
        @test result.separation_gaps[1] < 0.2  # Close to second eigenvalue
        @test result.separation_gaps[2] < 0.2  # Close to first eigenvalue
        @test result.separation_gaps[3] > 3.8  # Far from others
    end

    @testset "Method comparison: standard vs refined vs krawczyk" begin
        A = BallMatrix([2.0 1.0; 1.0 3.0])

        result_std = rump_2022a_eigenvalue_bounds(A; method=:standard, hermitian=true)
        result_ref = rump_2022a_eigenvalue_bounds(A; method=:refined, hermitian=true)
        result_kra = rump_2022a_eigenvalue_bounds(A; method=:krawczyk, hermitian=true)

        # All should verify
        @test result_std.verified
        @test result_ref.verified
        @test result_kra.verified

        # Refined should give equal or better bounds than standard
        for i in 1:2
            @test rad(result_ref.eigenvalues[i]) <= rad(result_std.eigenvalues[i]) + 1e-14
        end

        # All methods should contain true eigenvalues
        λ_true = eigvals(Hermitian(mid(A)))
        for i in 1:2
            @test λ_true[i] ∈ result_std.eigenvalues[i]
            @test λ_true[i] ∈ result_ref.eigenvalues[i]
            @test λ_true[i] ∈ result_kra.eigenvalues[i]
        end
    end

    @testset "Non-Hermitian matrix" begin
        A = BallMatrix([2.0 1.0; 0.5 3.0])
        result = rump_2022a_eigenvalue_bounds(A; hermitian=false)

        @test length(result) == 2

        # Check eigenvalues are contained (may have imaginary parts)
        λ_true = eigvals(mid(A))
        for i in 1:2
            # For complex eigenvalues, check containment
            ball_i = result.eigenvalues[i]
            @test abs(real(λ_true[i]) - mid(ball_i)) <= rad(ball_i)
        end
    end

    @testset "Coupling defect verification" begin
        A = BallMatrix(Diagonal([1.0, 2.0, 3.0]))
        result = rump_2022a_eigenvalue_bounds(A; hermitian=true)

        # Coupling defect should be small for diagonal matrix
        @test result.coupling_defect < 1e-14
    end

    @testset "Eigenvector error bounds" begin
        A = BallMatrix([2.0 1.0; 1.0 2.0])
        result = rump_2022a_eigenvalue_bounds(A; hermitian=true)

        # Eigenvector errors should be finite and relatively small
        @test all(e -> isfinite(e), result.eigenvector_errors)
        @test all(e -> e >= 0, result.eigenvector_errors)
        @test all(e -> e < 1e-10, result.eigenvector_errors)
    end

    @testset "Large separation improves conditioning" begin
        # Well-separated eigenvalues should have better conditioning
        A_good = BallMatrix(Diagonal([1.0, 10.0]))
        A_bad = BallMatrix(Diagonal([1.0, 1.01]))

        result_good = rump_2022a_eigenvalue_bounds(A_good; hermitian=true)
        result_bad = rump_2022a_eigenvalue_bounds(A_bad; hermitian=true)

        # Better separation should give better conditioning
        @test result_good.separation_gaps[1] > result_bad.separation_gaps[1]
        # Note: For diagonal matrices, condition numbers might be similar,
        # but eigenvector errors should be better for well-separated case
    end
end
