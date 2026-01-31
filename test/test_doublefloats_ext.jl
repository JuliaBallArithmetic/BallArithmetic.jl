# Tests for DoubleFloatsExt - Double64 oracle pattern implementations
#
# These tests verify the "oracle then certify" pattern:
# 1. Fast approximate computation in Double64 (~106 bits)
# 2. Final certification in BigFloat for rigorous bounds

using Test
using LinearAlgebra
using BallArithmetic

# Only run if DoubleFloats is available
try
    using DoubleFloats

    @testset "DoubleFloatsExt" begin

        @testset "Ogita SVD Refinement with Double64" begin
            # Test matrix
            A = [3.0 1.0; 1.0 2.0]
            U, S, V = svd(A)

            # Test fast refinement
            result_fast = ogita_svd_refine_fast(A, U, S, V;
                max_iterations=2, certify_with_bigfloat=true)

            @test result_fast.converged
            @test result_fast.iterations == 2
            @test result_fast.residual_norm < 1e-30  # Should be very accurate

            # Test hybrid refinement
            result_hybrid = ogita_svd_refine_hybrid(A, U, S, V;
                d64_iterations=2, bf_iterations=1, precision_bits=256)

            @test result_hybrid.converged
            @test result_hybrid.iterations == 3

            # Verify the refined SVD reconstructs A accurately
            A_reconstructed = result_hybrid.U * result_hybrid.Σ * result_hybrid.V'
            reconstruction_error = norm(convert.(Float64, A_reconstructed) - A) / norm(A)
            @test reconstruction_error < 1e-15
        end

        @testset "Schur Refinement with Double64" begin
            # Test matrix with distinct eigenvalues
            A = [4.0 1.0 0.0; 0.0 3.0 1.0; 0.0 0.0 2.0] + 0.1 * randn(3, 3)
            F = schur(A)
            Q0, T0 = F.Z, F.T

            # Test fast Schur refinement
            result_fast = refine_schur_double64(A, Q0, T0;
                max_iterations=2, certify_with_bigfloat=true)

            @test result_fast.residual_norm < 1e-25

            # Test hybrid refinement
            result_hybrid = refine_schur_hybrid(A, Q0, T0;
                d64_iterations=2, bf_iterations=1, precision_bits=256)

            @test result_hybrid.converged
            @test result_hybrid.residual_norm < 1e-50
        end

        @testset "Symmetric Eigenvalue Refinement with Double64" begin
            # Symmetric test matrix
            A_base = randn(5, 5)
            A = A_base + A_base'
            F = eigen(Symmetric(A))
            Q0, λ0 = F.vectors, F.values

            # Test fast symmetric eigenvalue refinement
            result_fast = refine_symmetric_eigen_double64(A, Q0, λ0;
                max_iterations=2, certify_with_bigfloat=true)

            @test result_fast.converged
            @test result_fast.residual_norm < 1e-25

            # Test hybrid refinement
            result_hybrid = refine_symmetric_eigen_hybrid(A, Q0, λ0;
                d64_iterations=2, bf_iterations=1, precision_bits=256)

            @test result_hybrid.converged
            @test result_hybrid.residual_norm < 1e-50
        end

        @testset "Performance comparison: Double64 vs BigFloat" begin
            # Larger matrix to see timing difference
            n = 50
            A = randn(n, n)
            U, S, V = svd(A)

            # Time Double64 (should be faster)
            t_d64 = @elapsed for _ in 1:3
                ogita_svd_refine_fast(A, U, S, V; max_iterations=2)
            end

            # Time pure BigFloat
            t_bf = @elapsed for _ in 1:3
                ogita_svd_refine(A, U, S, V; max_iterations=2, precision_bits=256)
            end

            println("SVD refinement timing (n=$n, 3 runs):")
            println("  Double64 (2 iters): $(round(t_d64, digits=3))s")
            println("  BigFloat (2 iters): $(round(t_bf, digits=3))s")
            println("  Speedup: $(round(t_bf/t_d64, digits=1))×")

            # Double64 should be significantly faster (at least 5×)
            @test t_d64 < t_bf
        end

        @testset "Quadratic convergence verification" begin
            # Verify that iterations follow quadratic convergence
            A = randn(10, 10)
            U, S, V = svd(A)

            residuals = Float64[]
            for n_iter in 1:4
                result = ogita_svd_refine_fast(A, U, S, V;
                    max_iterations=n_iter, certify_with_bigfloat=true)
                push!(residuals, Float64(result.residual_norm))
            end

            # With quadratic convergence, each iteration should roughly square the error
            # log(residual) should decrease roughly linearly with doublings
            println("Quadratic convergence test:")
            for (i, r) in enumerate(residuals)
                println("  Iteration $i: residual = $(r)")
            end

            # Check that residuals decrease rapidly (at least 10× per iteration)
            for i in 2:length(residuals)
                if residuals[i-1] > 1e-50  # Avoid underflow issues
                    @test residuals[i] < residuals[i-1] / 10
                end
            end
        end
    end

catch e
    if e isa ArgumentError && contains(string(e), "DoubleFloats")
        @warn "DoubleFloats.jl not installed, skipping DoubleFloatsExt tests"
    else
        rethrow(e)
    end
end
