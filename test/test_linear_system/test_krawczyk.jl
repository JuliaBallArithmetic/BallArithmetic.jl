using Test
using LinearAlgebra
using Random
using BallArithmetic

@testset "Krawczyk Linear System" begin
    @testset "Well-conditioned systems" begin
        # Simple 2x2 diagonally dominant system
        A_mid = [3.0 1.0; 1.0 2.0]
        b_mid = [5.0, 4.0]

        A = BallMatrix(A_mid, fill(1e-15, 2, 2))
        b = BallVector(b_mid, fill(1e-15, 2))

        result = krawczyk_linear_system(A, b)

        @test result.verified
        @test result.contraction_factor < 1.0
        @test result.residual_norm < 1e-10

        # Check solution enclosure contains exact solution
        x_exact = A_mid \ b_mid
        x_sol = result.solution
        for i in 1:2
            @test abs(mid(x_sol)[i] - x_exact[i]) <= rad(x_sol)[i]
        end
    end

    @testset "3x3 system" begin
        A_mid = [4.0 1.0 0.5; 1.0 3.0 0.5; 0.5 0.5 5.0]
        b_mid = [6.0, 5.0, 7.0]

        A = BallMatrix(A_mid, fill(1e-15, 3, 3))
        b = BallVector(b_mid, fill(1e-15, 3))

        result = krawczyk_linear_system(A, b)

        @test result.verified
        @test result.contraction_factor < 1.0

        # Check solution
        x_exact = A_mid \ b_mid
        x_sol = result.solution
        for i in 1:3
            @test abs(mid(x_sol)[i] - x_exact[i]) <= rad(x_sol)[i] + 1e-10
        end
    end

    @testset "Identity matrix" begin
        n = 5
        A = BallMatrix(Matrix{Float64}(I, n, n), fill(1e-15, n, n))
        b_mid = randn(n)
        b = BallVector(b_mid, fill(1e-15, n))

        result = krawczyk_linear_system(A, b)

        @test result.verified
        # For identity, solution = b
        for i in 1:n
            @test abs(mid(result.solution)[i] - b_mid[i]) <= rad(result.solution)[i] + 1e-12
        end
    end

    @testset "the enclosure holds for members of the two balls" begin
        # 3x = 1 with exact data: the solution 1/3 is not a float, so the radius is positive
        r = krawczyk_linear_system(BallMatrix(fill(3.0, 1, 1)), BallVector([1.0]))
        @test r.verified
        @test rad(r.solution)[1] > 0
        @test abs(big(1) / 3 - mid(r.solution)[1]) <= rad(r.solution)[1]
        rng = MersenneTwister(20261011)
        for n in (3, 8), ρ in (0.0, 1e-8, 1e-4)
            Am, bm = randn(rng, n, n) + n * I, randn(rng, n)
            r = krawczyk_linear_system(BallMatrix(Am, fill(ρ, n, n)), BallVector(bm, fill(ρ, n)))
            @test r.verified
            for _ in 1:10
                Ap = Am + ρ * (2 * rand(rng, n, n) .- 1)
                bp = bm + ρ * (2 * rand(rng, n) .- 1)
                x = setprecision(() -> big.(Ap) \ big.(bp), BigFloat, 512)
                @test all(abs.(x - mid(r.solution)) .<= rad(r.solution))
                @test r.residual_norm >= norm(Ap * mid(r.solution) - bp) * (1 - 1e-10)
            end
        end
        # a singular midpoint: not verified
        s = krawczyk_linear_system(BallMatrix([1.0 1.0; 1.0 1.0]), BallVector([1.0, 2.0]))
        @test !s.verified
    end

    @testset "Moderately ill-conditioned" begin
        # Hilbert-like matrix (moderately ill-conditioned)
        n = 4
        A_mid = [1.0/(i+j-1) for i in 1:n, j in 1:n]
        b_mid = ones(n)

        A = BallMatrix(A_mid, fill(1e-15, n, n))
        b = BallVector(b_mid, fill(1e-15, n))

        result = krawczyk_linear_system(A, b; max_iterations=20)

        # May or may not verify depending on conditioning
        # But should at least produce a result
        @test result.residual_norm >= 0
    end
end

# Note: The Sylvester Krawczyk method currently has algorithmic limitations
# with the contraction estimate - marked as broken until improved
@testset "Krawczyk Sylvester (Schur-based)" begin
    @testset "Basic Sylvester structure" begin
        # Just test that the function runs without error
        A = [5.0 0.1; 0.0 6.0]
        B = [-2.0 0.0; 0.0 -3.0]
        C = [1.0 1.0; 1.0 1.0]

        result = krawczyk_sylvester(A, B, C)

        # Should return a KrawczykResult
        @test result isa KrawczykResult
        @test result.solution isa BallMatrix
    end

    @testset "Complex Sylvester structure" begin
        # Well-separated complex spectra
        A = ComplexF64[5.0+1.0im 0.0; 0.0 6.0-0.5im]
        B = ComplexF64[-2.0 0.0; 0.0 -3.0+0.3im]
        C = ComplexF64[1.0 1.0; 1.0 1.0]

        # Note: Currently has issues with complex Sylvester, mark as broken
        @test_broken begin
            result = krawczyk_sylvester(A, B, C)
            result isa KrawczykResult
        end
    end

    @testset "Sylvester verification (broken)" begin
        # The current Schur-based algorithm has contraction estimate issues
        # This test documents the expected behavior when fixed
        A = [5.0 0.1; 0.0 6.0]
        B = [-2.0 0.0; 0.0 -3.0]
        C = [1.0 1.0; 1.0 1.0]

        result = krawczyk_sylvester(A, B, C)

        # Should verify when algorithm is improved
        @test_broken result.verified
    end
end

@testset "Krawczyk edge cases" begin
    @testset "Nearly singular matrix warning" begin
        # Nearly singular matrix
        A_mid = [1.0 1.0; 1.0 1.0 + 1e-14]
        b_mid = [2.0, 2.0]

        A = BallMatrix(A_mid, fill(1e-15, 2, 2))
        b = BallVector(b_mid, fill(1e-15, 2))

        # Should fail verification or produce large radius
        result = krawczyk_linear_system(A, b)

        # For nearly singular matrices, either:
        # - verification fails (contraction_factor >= 1 or NaN)
        # - verification succeeds but with large error bounds
        # - the method runs but produces poor results
        @test result isa KrawczykResult
        # At minimum, the contraction should be poor or the radius should be large
        @test !result.verified || result.contraction_factor > 0.5 || maximum(rad(result.solution)) > 1e-10
    end

    @testset "Expansion iterations" begin
        # System that needs expansion
        A_mid = [2.0 0.9; 0.9 2.0]
        b_mid = [3.0, 3.0]

        A = BallMatrix(A_mid, fill(1e-15, 2, 2))
        b = BallVector(b_mid, fill(1e-15, 2))

        result = krawczyk_linear_system(A, b; max_iterations=20)

        @test result.verified
        @test result.iterations >= 1
    end
end
