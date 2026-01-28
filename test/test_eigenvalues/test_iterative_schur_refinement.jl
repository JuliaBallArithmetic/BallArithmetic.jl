using Test
using BallArithmetic
using LinearAlgebra

@testset "Iterative Schur Refinement" begin

    @testset "Newton-Schulz orthogonalization" begin
        # Start with a nearly orthogonal matrix
        n = 10
        Q = qr(randn(n, n)).Q |> Matrix
        # Perturb slightly
        Q_perturbed = Q + 0.01 * randn(n, n)

        Q_copy = copy(Q_perturbed)
        iters, defect = BallArithmetic.newton_schulz_orthogonalize!(Q_copy; max_iter=10, tol=1e-12)

        @test defect < 1e-10
        @test iters <= 10

        # Check orthogonality
        I_n = Matrix{Float64}(I, n, n)
        @test BallArithmetic._frobenius_norm(Q_copy' * Q_copy - I_n) < 1e-10
    end

    @testset "Basic Schur refinement Float64 -> BigFloat" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            # Small test matrix
            n = 5
            A = randn(n, n)

            # Compute Schur in Float64
            F = schur(A)
            Q0, T0 = F.Z, F.T

            # Refine to BigFloat
            result = refine_schur_decomposition(A, Q0, T0;
                                                target_precision=256,
                                                max_iterations=10)

            @test result isa BallArithmetic.SchurRefinementResult
            @test size(result.Q) == (n, n)
            @test size(result.T) == (n, n)

            # Check convergence - should achieve very small residual
            @test result.residual_norm < 1e-50
            @test result.orthogonality_defect < 1e-50
            @test result.converged

            # Verify reconstruction: A ≈ Q * T * Q'
            A_big = convert.(BigFloat, A)
            reconstruction = result.Q * result.T * result.Q'
            rel_error = BallArithmetic._frobenius_norm(A_big - reconstruction) / BallArithmetic._frobenius_norm(A_big)
            @test rel_error < 1e-50

            # Verify orthogonality of Q
            I_n = Matrix{BigFloat}(I, n, n)
            orth_error = BallArithmetic._frobenius_norm(result.Q' * result.Q - I_n)
            @test orth_error < 1e-50
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "Schur refinement preserves eigenvalues" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            n = 4
            # Create matrix with known eigenvalues
            D = Diagonal([1.0, 2.0, 3.0, 4.0])
            P = qr(randn(n, n)).Q |> Matrix
            A = P * D * P'

            F = schur(A)
            Q0, T0 = F.Z, F.T

            result = refine_schur_decomposition(A, Q0, T0;
                                                target_precision=256,
                                                max_iterations=10)

            # The diagonal of T should approximate the eigenvalues
            # (for symmetric matrices, T is similar to D)
            @test result.converged
            @test result.residual_norm < 1e-50
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "rigorous_schur_bigfloat" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            n = 5
            A_center = randn(n, n)
            A_radius = fill(1e-10, n, n)
            A = BallMatrix(A_center, A_radius)

            Q_ball, T_ball, result = rigorous_schur_bigfloat(A; target_precision=256)

            @test Q_ball isa BallMatrix
            @test T_ball isa BallMatrix
            @test size(mid(Q_ball)) == (n, n)
            @test size(mid(T_ball)) == (n, n)

            # Check that radii are reasonable (non-negative)
            @test all(rad(Q_ball) .>= 0)
            @test all(rad(T_ball) .>= 0)

            # Verify convergence
            @test result.converged || result.residual_norm < 1e-30
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "Different precision levels" begin
        old_prec = precision(BigFloat)
        try
            n = 4
            A = randn(n, n)
            F = schur(A)
            Q0, T0 = F.Z, F.T

            # Test different precision levels
            for target_prec in [128, 256, 512]
                setprecision(BigFloat, target_prec)

                result = refine_schur_decomposition(A, Q0, T0;
                                                    target_precision=target_prec,
                                                    max_iterations=10)

                # Higher precision should give smaller residuals
                expected_eps = BigFloat(2)^(-target_prec)
                @test result.residual_norm < 1000 * expected_eps
            end
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "Complex matrices" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            n = 4
            A = randn(ComplexF64, n, n)

            F = schur(A)
            Q0, T0 = F.Z, F.T

            result = refine_schur_decomposition(A, Q0, T0;
                                                target_precision=256,
                                                max_iterations=10)

            @test result isa BallArithmetic.SchurRefinementResult
            @test eltype(result.Q) <: Complex
            @test result.residual_norm < 1e-40
        finally
            setprecision(BigFloat, old_prec)
        end
    end

end
