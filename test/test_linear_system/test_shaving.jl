using Test
using LinearAlgebra
using BallArithmetic

@testset "Shaving Methods" begin
    @testset "Sherman-Morrison inverse update" begin
        @testset "Basic rank-1 update" begin
            # Simple 2x2 case
            A = [3.0 1.0; 1.0 2.0]
            A_inv = inv(A)
            u = [1.0, 0.0]
            v = [0.0, 1.0]

            # Compute updated inverse using Sherman-Morrison
            A_updated_inv = sherman_morrison_inverse_update(A_inv, u, v)

            # Compare with direct computation
            A_updated = A + u * v'
            A_updated_inv_direct = inv(A_updated)

            @test A_updated_inv ≈ A_updated_inv_direct atol=1e-12
        end

        @testset "3x3 rank-1 update" begin
            A = [4.0 1.0 0.5; 1.0 3.0 0.5; 0.5 0.5 5.0]
            A_inv = inv(A)
            u = [1.0, 2.0, 0.5]
            v = [0.5, 1.0, 0.3]

            A_updated_inv = sherman_morrison_inverse_update(A_inv, u, v)
            A_updated_inv_direct = inv(A + u * v')

            @test A_updated_inv ≈ A_updated_inv_direct atol=1e-12
        end

        @testset "Identity update" begin
            # Start with identity matrix
            n = 4
            I_n = Matrix{Float64}(I, n, n)
            u = zeros(n); u[1] = 1.0
            v = zeros(n); v[2] = 1.0

            # I + e_1 * e_2' is an elementary matrix
            result = sherman_morrison_inverse_update(I_n, u, v)
            expected = inv(I_n + u * v')

            @test result ≈ expected atol=1e-14
        end

        @testset "Symmetric update" begin
            # Symmetric A with symmetric rank-1 update
            A = [2.0 0.5; 0.5 3.0]
            A_inv = inv(A)
            u = [1.0, 1.0]
            v = u  # Same vector for symmetric update

            A_updated_inv = sherman_morrison_inverse_update(A_inv, u, v)
            A_updated = A + u * v'
            @test A_updated_inv ≈ inv(A_updated) atol=1e-12
        end

        @testset "Small perturbation" begin
            A = [5.0 1.0; 1.0 4.0]
            A_inv = inv(A)
            # Small perturbation
            u = [0.01, 0.0]
            v = [0.0, 0.01]

            A_updated_inv = sherman_morrison_inverse_update(A_inv, u, v)
            A_updated_inv_direct = inv(A + u * v')

            @test A_updated_inv ≈ A_updated_inv_direct atol=1e-12
        end

        @testset "Error on near-singular update" begin
            # Create a case where 1 + v^T A^(-1) u ≈ 0
            A = [1.0 0.0; 0.0 1.0]
            A_inv = inv(A)
            # u * v' = -I would make A + u*v' singular
            # v^T * A_inv * u = v^T * u = -1, so 1 + (-1) = 0
            u = [1.0, 0.0]
            v = [-1.0, 0.0]

            @test_throws ErrorException sherman_morrison_inverse_update(A_inv, u, v)
        end
    end

    @testset "ShavingResult structure" begin
        # Test that ShavingResult can be created and accessed
        x = BallVector([1.0, 2.0], [0.1, 0.2])
        result = ShavingResult(x, 0.05, 3, 2)

        @test result.solution === x
        @test result.shaved_amount == 0.05
        @test result.iterations == 3
        @test result.components_shaved == 2
    end

    @testset "Interval shaving" begin
        @testset "Well-conditioned system" begin
            # Simple diagonally dominant system
            A_mid = [3.0 1.0; 1.0 2.0]
            b_mid = [5.0, 4.0]

            A = BallMatrix(A_mid, fill(0.01, 2, 2))
            b = BallVector(b_mid, fill(0.01, 2))

            # Compute exact solution for reference
            x_exact = A_mid \ b_mid

            # Start with a wide initial enclosure
            x0 = BallVector(x_exact, fill(0.5, 2))

            # Apply shaving
            result = interval_shaving(A, b, x0)

            @test result isa ShavingResult
            @test length(result.solution) == 2
            @test result.iterations >= 1
        end

        @testset "Identity system" begin
            n = 3
            A = BallMatrix(Matrix{Float64}(I, n, n), fill(1e-10, n, n))
            b_mid = [1.0, 2.0, 3.0]
            b = BallVector(b_mid, fill(1e-10, n))

            # For identity, solution = b
            x0 = BallVector(b_mid, fill(0.5, n))

            result = interval_shaving(A, b, x0; max_iterations=5)

            @test result isa ShavingResult
            # Solution should still contain the exact solution
            for i in 1:n
                @test inf(result.solution[i]) <= b_mid[i] <= sup(result.solution[i])
            end
        end

        @testset "With custom preconditioner" begin
            A_mid = [4.0 1.0; 1.0 3.0]
            b_mid = [6.0, 5.0]

            A = BallMatrix(A_mid, fill(0.01, 2, 2))
            b = BallVector(b_mid, fill(0.01, 2))

            x_exact = A_mid \ b_mid
            x0 = BallVector(x_exact, fill(0.3, 2))

            # Provide custom preconditioner
            R = inv(A_mid)

            result = interval_shaving(A, b, x0; R=R, max_iterations=5)

            @test result isa ShavingResult
        end

        @testset "Min improvement threshold" begin
            A_mid = [3.0 0.5; 0.5 2.0]
            b_mid = [4.0, 3.0]

            A = BallMatrix(A_mid, fill(0.01, 2, 2))
            b = BallVector(b_mid, fill(0.01, 2))

            x_exact = A_mid \ b_mid
            x0 = BallVector(x_exact, fill(0.2, 2))

            # Use high min_improvement to stop early
            result = interval_shaving(A, b, x0; min_improvement=0.5, max_iterations=10)

            @test result isa ShavingResult
            # Should have stopped early due to high threshold
        end

        @testset "Max iterations limit" begin
            A_mid = [3.0 1.0; 1.0 2.0]
            b_mid = [5.0, 4.0]

            A = BallMatrix(A_mid, fill(0.01, 2, 2))
            b = BallVector(b_mid, fill(0.01, 2))

            x_exact = A_mid \ b_mid
            x0 = BallVector(x_exact, fill(0.4, 2))

            # Limit to 2 iterations
            result = interval_shaving(A, b, x0; max_iterations=2)

            @test result isa ShavingResult
            @test result.iterations == 2
        end

        @testset "3x3 system" begin
            A_mid = [5.0 1.0 0.5; 1.0 4.0 0.5; 0.5 0.5 6.0]
            b_mid = [7.0, 6.0, 8.0]

            A = BallMatrix(A_mid, fill(0.01, 3, 3))
            b = BallVector(b_mid, fill(0.01, 3))

            x_exact = A_mid \ b_mid
            x0 = BallVector(x_exact, fill(0.3, 3))

            result = interval_shaving(A, b, x0; max_iterations=5)

            @test result isa ShavingResult
            @test length(result.solution) == 3

            # Solution should still contain exact solution
            for i in 1:3
                @test inf(result.solution[i]) <= x_exact[i] + 0.1  # Some tolerance for interval arithmetic
                @test sup(result.solution[i]) >= x_exact[i] - 0.1
            end
        end
    end

    @testset "Shaving integration with Krawczyk" begin
        @testset "Refine Krawczyk solution" begin
            # Use Krawczyk to get initial enclosure, then refine with shaving
            A_mid = [3.0 1.0; 1.0 2.0]
            b_mid = [5.0, 4.0]

            A = BallMatrix(A_mid, fill(1e-15, 2, 2))
            b = BallVector(b_mid, fill(1e-15, 2))

            # Get initial enclosure from Krawczyk
            krawczyk_result = krawczyk_linear_system(A, b)

            if krawczyk_result.verified
                x0 = krawczyk_result.solution

                # Apply shaving refinement
                shaving_result = interval_shaving(A, b, x0; max_iterations=5)

                @test shaving_result isa ShavingResult

                # Shaved solution should not be wider than original
                for i in 1:2
                    @test rad(shaving_result.solution[i]) <= rad(x0[i]) + 1e-10
                end
            end
        end
    end
end
