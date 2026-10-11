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

    @testset "the solver must be available" begin
        A = BallMatrix([3.0 1.0; 1.0 2.0], fill(0.01, 2, 2))
        b = BallVector([5.0, 4.0], fill(0.01, 2))
        x0 = BallVector([1.2, 1.4], [0.5, 0.5])
        # a solver that gives up: nothing is cut
        r = interval_shaving(A, b, x0; lp_solver = (c, Aeq, beq, lb, ub) -> nothing)
        @test r.components_shaved == 0 && !r.empty && r.shaved_amount == 0
        @test all(rad(r.solution) .>= rad(x0))
        # a solver that returns nonsense: nothing is cut either, the certificate is verified
        r = interval_shaving(A, b, x0; lp_solver = (c, Aeq, beq, lb, ub) -> randn(length(c)))
        @test !r.empty
        sols = [[Rational{BigInt}(5) + s1 // 100, Rational{BigInt}(4) + s2 // 100]
                for s1 in (-1, 1), s2 in (-1, 1)]
        @test_throws ArgumentError interval_shaving(BallMatrix(randn(ComplexF64, 2, 2)),
            BallVector(randn(ComplexF64, 2)), BallVector(randn(ComplexF64, 2));
            lp_solver = (a...) -> nothing)
    end
end
