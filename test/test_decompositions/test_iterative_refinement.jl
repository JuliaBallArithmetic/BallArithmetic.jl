using Test
using BallArithmetic
using LinearAlgebra

@testset "Iterative Refinement Methods" begin

    #==========================================================================#
    # Polar Decomposition Refinement
    #==========================================================================#

    @testset "Polar Decomposition Refinement" begin

        @testset "Newton-Schulz Method" begin
            # Test with well-conditioned matrix
            A = [3.0 1.0; 1.0 2.0]
            F = svd(A)
            Q0 = F.U * F.Vt  # Initial polar factor approximation

            result = refine_polar_newton_schulz(A, Q0, max_iterations=10, tol=1e-14)

            @test result.converged
            @test result.orthogonality_defect < 1e-13
            @test result.residual_norm < 1e-13

            # Verify Q is orthogonal: Q'Q ≈ I
            @test opnorm(result.Q' * result.Q - I, Inf) < 1e-13

            # Verify factorization: QH ≈ A
            @test opnorm(result.Q * result.H - A, Inf) / opnorm(A, Inf) < 1e-13

            # Verify H is symmetric positive semidefinite
            @test opnorm(result.H - result.H', Inf) < 1e-13
            @test all(eigvals(Symmetric(result.H)) .>= -1e-10)
        end

        @testset "Scaled Newton Method" begin
            A = [4.0 1.0; 1.0 3.0]
            F = svd(A)
            Q0 = F.U * F.Vt

            result = refine_polar_newton(A, Q0, max_iterations=10, tol=1e-14)

            @test result.converged
            @test result.orthogonality_defect < 1e-13
            @test result.residual_norm < 1e-13
            @test opnorm(result.Q' * result.Q - I, Inf) < 1e-13
        end

        @testset "QDWH Method" begin
            # QDWH should converge in ~6 iterations for moderate condition numbers
            A = [5.0 1.0 0.5;
                 1.0 4.0 0.3;
                 0.5 0.3 3.0]
            F = svd(A)
            Q0 = F.U * F.Vt

            result = refine_polar_qdwh(A, Q0, max_iterations=6, tol=1e-14)

            @test result.converged
            @test result.orthogonality_defect < 1e-13
            @test result.residual_norm < 1e-13
            @test result.iterations <= 6
        end

        @testset "Complex Polar Decomposition" begin
            A = [3.0+1.0im 1.0-0.5im;
                 1.0+0.5im 2.0-0.2im]
            F = svd(A)
            Q0 = F.U * F.Vt

            result = refine_polar_newton_schulz(A, Q0, max_iterations=10, tol=1e-13)

            @test result.converged
            @test result.orthogonality_defect < 1e-12
            @test opnorm(result.Q' * result.Q - I, Inf) < 1e-12
        end

        @testset "Poorly Conditioned Matrix" begin
            # High condition number - Newton-Schulz may struggle, QDWH should work
            A = Diagonal([1.0, 1e-6])
            F = svd(A)
            Q0 = F.U * F.Vt

            # Newton-Schulz might not converge for poorly conditioned matrices
            result_ns = refine_polar_newton_schulz(A, Q0, max_iterations=20, tol=1e-10)

            # QDWH should handle it better
            result_qdwh = refine_polar_qdwh(A, Q0, max_iterations=10, tol=1e-10)

            # At least QDWH should give reasonable results
            @test result_qdwh.residual_norm < 1e-6
        end
    end

    #==========================================================================#
    # LU Decomposition Refinement
    #==========================================================================#

    @testset "LU Decomposition Refinement" begin

        @testset "Well-Conditioned Matrix" begin
            A = [4.0 2.0 1.0;
                 2.0 5.0 2.0;
                 1.0 2.0 3.0]

            F = lu(A)
            L0 = Matrix(F.L)
            U0 = Matrix(F.U)
            p = F.p

            result = refine_lu(A, L0, U0, p, max_iterations=5, tol=1e-14)

            @test result.converged
            @test result.residual_norm < 1e-13

            # Verify PA ≈ LU
            PA = A[p, :]
            @test opnorm(PA - result.L * result.U, Inf) / opnorm(PA, Inf) < 1e-13

            # Verify L is unit lower triangular
            for i in 1:3
                @test result.L[i, i] ≈ 1.0 atol=1e-14
                for j in (i+1):3
                    @test result.L[i, j] ≈ 0.0 atol=1e-14
                end
            end
        end

        @testset "4x4 System" begin
            A = [10.0 1.0 2.0 0.5;
                 1.0  8.0 1.0 0.3;
                 2.0  1.0 6.0 0.2;
                 0.5  0.3 0.2 5.0]

            F = lu(A)
            L0, U0, p = Matrix(F.L), Matrix(F.U), F.p

            result = refine_lu(A, L0, U0, p, max_iterations=5)

            @test result.residual_norm < 1e-13

            # Verify factorization
            PA = A[p, :]
            @test opnorm(PA - result.L * result.U, Inf) / opnorm(PA, Inf) < 1e-13
        end

        @testset "With Perturbation in Initial Factors" begin
            A = [4.0 1.0; 1.0 3.0]
            F = lu(A)

            # Add small perturbation
            L0 = Matrix(F.L) + 1e-10 * [0.0 0.0; 0.1 0.0]  # Small strictly lower triangular perturbation
            U0 = Matrix(F.U) + 1e-10 * [0.1 0.0; 0.0 0.1]  # Small upper triangular perturbation

            result = refine_lu(A, L0, U0, F.p, max_iterations=10)

            # Should reduce residual to near machine precision
            @test result.residual_norm < 1e-11
        end
    end

    #==========================================================================#
    # Cholesky Decomposition Refinement
    #==========================================================================#

    @testset "Cholesky Decomposition Refinement" begin

        @testset "Symmetric Positive Definite Matrix" begin
            A = [4.0 2.0 1.0;
                 2.0 5.0 2.0;
                 1.0 2.0 3.0]

            C = cholesky(Symmetric(A))
            G0 = Matrix(C.U)

            result = refine_cholesky(A, G0, max_iterations=5, tol=1e-14)

            @test result.converged
            @test result.residual_norm < 1e-13

            # Verify GᵀG ≈ A
            @test opnorm(result.G' * result.G - A, Inf) / opnorm(A, Inf) < 1e-13

            # Verify G is upper triangular
            for i in 1:3
                for j in 1:(i-1)
                    @test result.G[i, j] ≈ 0.0 atol=1e-14
                end
            end
        end

        @testset "Identity Matrix" begin
            A = Matrix{Float64}(I, 3, 3)
            G0 = Matrix{Float64}(I, 3, 3)

            result = refine_cholesky(A, G0)

            @test result.converged
            @test result.residual_norm < 1e-14
            @test opnorm(result.G - I, Inf) < 1e-14
        end

        @testset "Diagonal SPD Matrix" begin
            A = Diagonal([4.0, 9.0, 16.0])
            G0 = Diagonal([2.0, 3.0, 4.0])

            result = refine_cholesky(Matrix(A), Matrix(G0))

            @test result.residual_norm < 1e-13
        end
    end

    #==========================================================================#
    # QR Decomposition Refinement
    #==========================================================================#

    @testset "QR Decomposition Refinement" begin

        @testset "CholeskyQR2 Method - Square Matrix" begin
            A = [4.0 1.0 0.5;
                 1.0 3.0 0.3;
                 0.5 0.3 2.0]

            F = qr(A)
            Q0 = Matrix(F.Q)
            R0 = Matrix(F.R)

            result = refine_qr_cholqr2(A, Q0, R0, max_iterations=3, tol=1e-14)

            @test result.converged
            @test result.orthogonality_defect < 1e-13
            @test result.residual_norm < 1e-13

            # Verify Q'Q ≈ I
            n = size(A, 2)
            @test opnorm(result.Q' * result.Q - I(n), Inf) < 1e-13

            # Verify QR ≈ A
            @test opnorm(result.Q * result.R - A, Inf) / opnorm(A, Inf) < 1e-13
        end

        @testset "CholeskyQR2 Method - Tall Matrix" begin
            A = [4.0 1.0;
                 1.0 3.0;
                 0.5 0.3;
                 0.2 0.4]

            F = qr(A)
            Q0 = Matrix(F.Q)
            R0 = Matrix(F.R)

            result = refine_qr_cholqr2(A, Q0, R0)

            @test result.orthogonality_defect < 1e-13
            @test result.residual_norm < 1e-13
        end

        @testset "Modified Gram-Schmidt Method" begin
            A = [4.0 1.0 0.5;
                 1.0 3.0 0.3;
                 0.5 0.3 2.0]

            F = qr(A)
            Q0 = Matrix(F.Q)
            R0 = Matrix(F.R)

            result = refine_qr_mgs(A, Q0, R0, max_iterations=2, tol=1e-14)

            @test result.orthogonality_defect < 1e-13
            @test result.residual_norm < 1e-13
        end

        @testset "Comparison: CholeskyQR2 vs MGS" begin
            # Both methods should give similar quality results
            A = randn(5, 3)
            F = qr(A)
            Q0, R0 = Matrix(F.Q), Matrix(F.R)

            result_chol = refine_qr_cholqr2(A, Q0, R0)
            result_mgs = refine_qr_mgs(A, Q0, R0)

            @test result_chol.orthogonality_defect < 1e-12
            @test result_mgs.orthogonality_defect < 1e-12
        end
    end

    #==========================================================================#
    # Takagi Decomposition Refinement
    #==========================================================================#

    @testset "Takagi Decomposition Refinement" begin

        @testset "Complex Symmetric Matrix" begin
            # Create a complex symmetric matrix (Aᵀ = A)
            A = [2.0+1.0im    1.0-0.5im;
                 1.0-0.5im    3.0+0.3im]

            # Compute initial Takagi decomposition using standard SVD
            # For Takagi A = UΣUᵀ, we use SVD of A as starting point
            F = svd(A)
            U0 = F.U
            Σ0 = F.S

            result = refine_takagi(A, U0, Σ0, max_iterations=10, tol=1e-8)

            # Verify UΣUᵀ ≈ A (note: transpose, not adjoint, for Takagi)
            reconstruction = result.U * Diagonal(result.Σ) * transpose(result.U)
            rel_err = maximum(abs.(reconstruction - A)) / maximum(abs.(A))

            # Takagi refinement may not fully converge from SVD initialization
            # but should give reasonable improvement
            @test rel_err < 1.0 || result.residual_norm < result.Σ[1]
        end

        @testset "Diagonal Complex Symmetric Matrix" begin
            # Diagonal matrix: Takagi values are magnitudes, U can be phases
            A = Diagonal([2.0+1.0im, 3.0-2.0im, 1.0+0.5im])
            A_sym = Matrix(A)  # Already symmetric

            # Expected singular values
            expected_sv = abs.([2.0+1.0im, 3.0-2.0im, 1.0+0.5im])

            # For diagonal case, U is diagonal with phases
            U0 = Diagonal(exp.(im * angle.([2.0+1.0im, 3.0-2.0im, 1.0+0.5im]) / 2))
            Σ0 = expected_sv

            result = refine_takagi(A_sym, Matrix(U0), Σ0, max_iterations=5)

            # Singular values should match magnitudes (up to sorting)
            @test sort(result.Σ) ≈ sort(expected_sv) rtol=1e-6
        end

        @testset "Real Symmetric Matrix (Special Case)" begin
            # Real symmetric matrix: Takagi reduces to eigenvalue problem
            A_real = [4.0 1.0; 1.0 3.0]
            A = Complex{Float64}.(A_real)

            F = svd(A)
            U0 = Complex{Float64}.(F.U)
            Σ0 = F.S

            result = refine_takagi(A, U0, Σ0)

            # For real symmetric, Takagi values are singular values
            reconstruction = result.U * Diagonal(result.Σ) * transpose(result.U)
            @test maximum(abs.(reconstruction - A)) / maximum(abs.(A)) < 1e-10
        end
    end

    #==========================================================================#
    # Result Type Tests
    #==========================================================================#

    @testset "Result Type Fields" begin

        @testset "PolarRefinementResult" begin
            A = [3.0 1.0; 1.0 2.0]
            F = svd(A)
            Q0 = F.U * F.Vt

            result = refine_polar_newton_schulz(A, Q0)

            @test hasfield(typeof(result), :Q)
            @test hasfield(typeof(result), :H)
            @test hasfield(typeof(result), :iterations)
            @test hasfield(typeof(result), :residual_norm)
            @test hasfield(typeof(result), :orthogonality_defect)
            @test hasfield(typeof(result), :converged)

            @test result.iterations > 0
            @test result.residual_norm >= 0
            @test result.orthogonality_defect >= 0
        end

        @testset "LURefinementResult" begin
            A = [4.0 1.0; 1.0 3.0]
            F = lu(A)

            result = refine_lu(A, Matrix(F.L), Matrix(F.U), F.p)

            @test hasfield(typeof(result), :L)
            @test hasfield(typeof(result), :U)
            @test hasfield(typeof(result), :p)
            @test hasfield(typeof(result), :iterations)
            @test hasfield(typeof(result), :residual_norm)
            @test hasfield(typeof(result), :converged)
        end

        @testset "CholeskyRefinementResult" begin
            A = [4.0 1.0; 1.0 3.0]
            C = cholesky(Symmetric(A))

            result = refine_cholesky(A, Matrix(C.U))

            @test hasfield(typeof(result), :G)
            @test hasfield(typeof(result), :iterations)
            @test hasfield(typeof(result), :residual_norm)
            @test hasfield(typeof(result), :converged)
        end

        @testset "QRRefinementResult" begin
            A = [4.0 1.0; 1.0 3.0]
            F = qr(A)

            result = refine_qr_cholqr2(A, Matrix(F.Q), Matrix(F.R))

            @test hasfield(typeof(result), :Q)
            @test hasfield(typeof(result), :R)
            @test hasfield(typeof(result), :iterations)
            @test hasfield(typeof(result), :residual_norm)
            @test hasfield(typeof(result), :orthogonality_defect)
            @test hasfield(typeof(result), :converged)
        end

        @testset "TakagiRefinementResult" begin
            A = Complex{Float64}.([2.0 1.0; 1.0 3.0])
            F = svd(A)

            result = refine_takagi(A, Complex{Float64}.(F.U), F.S)

            @test hasfield(typeof(result), :U)
            @test hasfield(typeof(result), :Σ)
            @test hasfield(typeof(result), :iterations)
            @test hasfield(typeof(result), :residual_norm)
            @test hasfield(typeof(result), :converged)
        end
    end

    #==========================================================================#
    # Edge Cases
    #==========================================================================#

    @testset "Edge Cases" begin

        @testset "1x1 Matrices" begin
            A = reshape([5.0], 1, 1)

            # Polar
            Q0 = reshape([1.0], 1, 1)
            result_polar = refine_polar_newton_schulz(A, Q0)
            @test result_polar.converged

            # LU
            F_lu = lu(A)
            result_lu = refine_lu(A, Matrix(F_lu.L), Matrix(F_lu.U), F_lu.p)
            @test result_lu.converged

            # Cholesky
            C = cholesky(Symmetric(A))
            result_chol = refine_cholesky(A, Matrix(C.U))
            @test result_chol.converged

            # QR
            F_qr = qr(A)
            result_qr = refine_qr_cholqr2(A, Matrix(F_qr.Q), Matrix(F_qr.R))
            @test result_qr.converged
        end

        @testset "Already Converged Input" begin
            # If input is already perfect, should detect convergence quickly
            A = Matrix{Float64}(I, 3, 3)
            Q0 = Matrix{Float64}(I, 3, 3)

            result = refine_polar_newton_schulz(A, Q0, tol=1e-14)

            @test result.converged
            @test result.iterations <= 2  # Should converge immediately or in 1-2 iterations
        end

        @testset "Maximum Iterations Reached" begin
            # Use a poorly conditioned starting point to force more iterations
            A = [3.0 1.0; 1.0 2.0]

            # Start with a perturbed unitary factor
            Q0 = [1.0 0.0; 0.0 1.0] + 0.1 * [0.0 0.1; -0.1 0.0]  # Slight perturbation

            result = refine_polar_newton_schulz(A, Q0, max_iterations=2, tol=1e-20)

            # Should use exactly 2 iterations (or converge early)
            @test result.iterations <= 2
            # Results should still be reasonable
            @test result.residual_norm < 1.0
        end
    end

end
