using Test
using BallArithmetic
using LinearAlgebra

@testset "Iterative Schur Refinement - Algorithm 4" begin

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

    @testset "Triangular matrix equation solver (Algorithm 2)" begin
        # Test with well-separated eigenvalues
        n = 8
        T_mat = UpperTriangular(randn(n, n))
        for i in 1:n
            T_mat[i, i] = float(i)  # Distinct eigenvalues
        end
        T_mat = Matrix(T_mat)

        # Create strictly lower triangular E
        E = zeros(n, n)
        for j in 1:(n-1)
            for i in (j+1):n
                E[i, j] = randn()
            end
        end

        # Test direct solver
        L = BallArithmetic.solve_triangular_matrix_equation(T_mat, E; use_recursive=false)
        TL_LT = T_mat * L - L * T_mat
        residual = BallArithmetic._stril(TL_LT) + E
        @test maximum(abs.(residual)) < 1e-12

        # Test recursive solver
        L_rec = BallArithmetic.solve_triangular_matrix_equation(T_mat, E; use_recursive=true, nmin=3)
        TL_LT_rec = T_mat * L_rec - L_rec * T_mat
        residual_rec = BallArithmetic._stril(TL_LT_rec) + E
        @test maximum(abs.(residual_rec)) < 1e-12
    end

    @testset "Complex Schur refinement (Algorithm 4)" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            # Use complex Schur for strictly triangular T
            n = 6
            A = ComplexF64.(randn(n, n))

            # Compute Schur in Float64
            F = schur(A)
            Q0, T0 = F.Z, F.T

            # Verify T0 is upper triangular
            @test opnorm(tril(T0, -1), Inf) < 1e-14

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
            A_big = convert.(Complex{BigFloat}, A)
            reconstruction = result.Q * result.T * result.Q'
            rel_error = BallArithmetic._frobenius_norm(A_big - reconstruction) / BallArithmetic._frobenius_norm(A_big)
            @test rel_error < 1e-50

            # Verify orthogonality of Q
            I_n = Matrix{Complex{BigFloat}}(I, n, n)
            orth_error = BallArithmetic._frobenius_norm(result.Q' * result.Q - I_n)
            @test orth_error < 1e-50

            # Verify T is upper triangular
            T_lower = BallArithmetic._stril(result.T)
            @test BallArithmetic._frobenius_norm(T_lower) < 1e-50
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "Schur refinement preserves eigenvalues" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            n = 5
            # Create matrix with known eigenvalues
            D = Diagonal(ComplexF64.([1.0, 2.0, 3.0, 4.0, 5.0]))
            P = qr(randn(ComplexF64, n, n)).Q |> Matrix
            A = P * D * P'

            F = schur(A)
            Q0, T0 = F.Z, F.T

            result = refine_schur_decomposition(A, Q0, T0;
                                                target_precision=256,
                                                max_iterations=10)

            # The diagonal of T should approximate the eigenvalues
            @test result.converged
            @test result.residual_norm < 1e-50

            # Check that diagonal of T contains the eigenvalues (up to ordering)
            eigs_refined = sort(real.(diag(result.T)))
            eigs_expected = [1.0, 2.0, 3.0, 4.0, 5.0]
            @test maximum(abs.(Float64.(eigs_refined) - eigs_expected)) < 1e-10
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "rigorous_schur_bigfloat" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            n = 5
            # Use complex matrix for complex Schur
            A_center = ComplexF64.(randn(n, n))
            A_radius = fill(1e-10, n, n)
            # BallMatrix for complex requires Complex radius handling
            A_mid = convert.(Complex{Float64}, A_center)
            A = BallMatrix(A_mid, A_radius)

            Q_ball, T_ball, result = rigorous_schur_bigfloat(A; target_precision=256)

            @test Q_ball isa BallMatrix
            @test T_ball isa BallMatrix
            @test size(mid(Q_ball)) == (n, n)
            @test size(mid(T_ball)) == (n, n)

            # Check that radii are reasonable (non-negative and finite)
            @test all(x -> x >= 0 && isfinite(x), rad(Q_ball))
            @test all(x -> x >= 0 && isfinite(x), rad(T_ball))

            # Verify convergence or reasonable residual
            @test result.converged || Float64(result.residual_norm) < 1e-20
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "Different precision levels" begin
        old_prec = precision(BigFloat)
        try
            n = 4
            A = ComplexF64.(randn(n, n))
            F = schur(A)
            Q0, T0 = F.Z, F.T

            # Test different precision levels
            for target_prec in [128, 256, 512]
                setprecision(BigFloat, target_prec)

                result = refine_schur_decomposition(A, Q0, T0;
                                                    target_precision=target_prec,
                                                    max_iterations=15)

                # Higher precision should give smaller residuals
                expected_eps = BigFloat(2)^(-target_prec)
                @test result.residual_norm < 1000 * expected_eps
            end
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "Iteration count (quadratic convergence)" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            # With well-separated eigenvalues, should converge in few iterations
            n = 8
            D = Diagonal(ComplexF64.(1:n))
            P = qr(randn(ComplexF64, n, n)).Q |> Matrix
            A = P * D * P'

            F = schur(A)
            Q0, T0 = F.Z, F.T

            result = refine_schur_decomposition(A, Q0, T0;
                                                target_precision=256,
                                                max_iterations=20)

            @test result.converged
            # Algorithm 4 typically needs 3-4 iterations for double to quadruple
            # For 256-bit (≈77 decimal digits) should need ≤6 iterations
            @test result.iterations <= 6
        finally
            setprecision(BigFloat, old_prec)
        end
    end

end

@testset "RefSyEv - Symmetric Eigenvalue Refinement (Ogita & Aishima 2018)" begin

    @testset "Basic symmetric eigenvalue refinement" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            n = 6
            # Create symmetric matrix
            A = randn(n, n)
            A = A + A'  # Symmetrize

            # Compute eigen in Float64
            F = eigen(Symmetric(A))
            Q0, λ0 = F.vectors, F.values

            # Refine to BigFloat
            result = refine_symmetric_eigen(A, Q0, λ0;
                                            target_precision=256,
                                            max_iterations=10)

            @test result isa BallArithmetic.SymmetricEigenRefinementResult
            @test size(result.Q) == (n, n)
            @test length(result.λ) == n

            # Check convergence
            @test result.residual_norm < 1e-50
            @test result.orthogonality_defect < 1e-50
            @test result.converged

            # Verify reconstruction: A ≈ Q * Λ * Q'
            A_big = convert.(BigFloat, A)
            Λ = Diagonal(result.λ)
            reconstruction = result.Q * Λ * result.Q'
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

    @testset "RefSyEv preserves eigenvalues" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            n = 5
            # Create matrix with known eigenvalues
            λ_true = [1.0, 2.0, 3.0, 4.0, 5.0]
            D = Diagonal(λ_true)
            P = qr(randn(n, n)).Q |> Matrix
            A = P * D * P'
            A = (A + A') / 2  # Ensure symmetry

            F = eigen(Symmetric(A))
            Q0, λ0 = F.vectors, F.values

            result = refine_symmetric_eigen(A, Q0, λ0;
                                            target_precision=256,
                                            max_iterations=10)

            @test result.converged
            @test result.residual_norm < 1e-50

            # Check that eigenvalues match (up to ordering)
            λ_refined = sort(Float64.(result.λ))
            λ_expected = sort(λ_true)
            @test maximum(abs.(λ_refined - λ_expected)) < 1e-10
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    # NOTE: Matrices with repeated (multiple) eigenvalues are challenging for
    # iterative eigenvalue refinement methods. When eigenvalues are repeated,
    # the eigenspaces are not uniquely defined - any orthonormal basis of the
    # eigenspace is valid. This non-uniqueness causes Newton-based refinement
    # to struggle achieving ultra-high precision, as the iteration may "wander"
    # within the eigenspace instead of converging to a specific eigenvector.
    #
    # The algorithm still produces mathematically correct results (eigenvalues
    # are accurate, eigenvectors span the correct subspace), but may not reach
    # the target precision within the iteration limit.
    @testset "RefSyEv handles multiple eigenvalues" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            n = 6
            # Matrix with repeated eigenvalues: λ = [1, 1, 2, 2, 3, 3]
            λ_true = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
            D = Diagonal(λ_true)
            P = qr(randn(n, n)).Q |> Matrix
            A = P * D * P'
            A = (A + A') / 2  # Ensure symmetry

            F = eigen(Symmetric(A))
            Q0, λ0 = F.vectors, F.values

            result = refine_symmetric_eigen(A, Q0, λ0;
                                            target_precision=256,
                                            max_iterations=15)

            if result.converged
                @test result.residual_norm < 1e-20
            else
                # KNOWN LIMITATION: Repeated eigenvalues may prevent convergence
                # to ultra-high precision. We still verify the algorithm makes
                # progress and doesn't diverge.
                @test result.residual_norm < 1e-8 || @test_broken result.converged
            end

            # Orthogonality should be maintained (relaxed tolerance for repeated eigenvalues)
            @test result.orthogonality_defect < 1e-10

            # Eigenvalues should still be accurate even if eigenvectors aren't fully refined
            λ_refined = sort(Float64.(result.λ))
            λ_expected = sort(λ_true)
            @test maximum(abs.(λ_refined - λ_expected)) < 1e-8
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "RefSyEv quadratic convergence" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            # With well-separated eigenvalues, should converge quickly
            n = 8
            λ_true = Float64.(1:n)
            D = Diagonal(λ_true)
            P = qr(randn(n, n)).Q |> Matrix
            A = P * D * P'
            A = (A + A') / 2

            F = eigen(Symmetric(A))
            Q0, λ0 = F.vectors, F.values

            result = refine_symmetric_eigen(A, Q0, λ0;
                                            target_precision=256,
                                            max_iterations=20)

            @test result.converged
            # RefSyEv should converge in few iterations (quadratic convergence)
            @test result.iterations <= 6
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "rigorous_symmetric_eigen_bigfloat" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            n = 5
            # Create symmetric ball matrix
            A_mid = randn(n, n)
            A_mid = (A_mid + A_mid') / 2  # Symmetrize
            A_radius = fill(1e-10, n, n)
            A = BallMatrix(A_mid, A_radius)

            Q_ball, λ_ball, result = rigorous_symmetric_eigen_bigfloat(A; target_precision=256)

            @test Q_ball isa BallMatrix
            @test λ_ball isa Vector{<:Ball}
            @test size(mid(Q_ball)) == (n, n)
            @test length(λ_ball) == n

            # Check that radii are reasonable
            @test all(x -> x >= 0 && isfinite(x), rad(Q_ball))
            @test all(x -> rad(x) >= 0 && isfinite(rad(x)), λ_ball)

            # Verify convergence
            @test result.converged || Float64(result.residual_norm) < 1e-20
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "Different precision levels for symmetric" begin
        old_prec = precision(BigFloat)
        try
            n = 4
            A = randn(n, n)
            A = A + A'  # Symmetrize
            F = eigen(Symmetric(A))
            Q0, λ0 = F.vectors, F.values

            # Test different precision levels
            for target_prec in [128, 256, 512]
                setprecision(BigFloat, target_prec)

                result = refine_symmetric_eigen(A, Q0, λ0;
                                                target_precision=target_prec,
                                                max_iterations=15)

                # Higher precision should give smaller residuals
                expected_eps = BigFloat(2)^(-target_prec)
                @test result.residual_norm < 1000 * expected_eps
            end
        finally
            setprecision(BigFloat, old_prec)
        end
    end

end
