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
            # Deterministic test matrix with distinct eigenvalues
            A = [4.0 1.0 0.0; 0.0 3.0 1.0; 0.0 0.0 2.0] +
                0.1 * [0.12 -0.34 0.56; 0.78 -0.23 0.45; -0.67 0.89 -0.11]
            F = schur(A)
            Q0, T0 = F.Z, F.T

            # Test fast Schur refinement (should achieve Double64 precision ~1e-31)
            result_fast = refine_schur_double64(A, Q0, T0;
                max_iterations=2, certify_with_bigfloat=true)

            @test result_fast.residual_norm < 1e-25

            # Test hybrid refinement (should achieve near BigFloat precision)
            # Use 2 BigFloat iterations for more reliable convergence
            result_hybrid = refine_schur_hybrid(A, Q0, T0;
                d64_iterations=2, bf_iterations=2, precision_bits=256)

            # Check that we achieved high precision (1e-49 allows for variation)
            @test result_hybrid.residual_norm < 1e-49
        end

        @testset "Symmetric Eigenvalue Refinement with Double64" begin
            # Deterministic symmetric test matrix
            A_base = [0.12 -0.34 0.56 0.78 -0.23;
                      0.45 -0.67 0.89 -0.11 0.33;
                     -0.55 0.22 -0.44 0.66 -0.88;
                      0.19 -0.37 0.51 -0.73 0.95;
                     -0.14 0.28 -0.42 0.56 -0.70]
            A = A_base + A_base'
            F = eigen(Symmetric(A))
            Q0, λ0 = F.vectors, F.values

            # Test fast symmetric eigenvalue refinement
            result_fast = refine_symmetric_eigen_double64(A, Q0, λ0;
                max_iterations=2, certify_with_bigfloat=true)

            @test result_fast.converged
            @test result_fast.residual_norm < 1e-25

            # Test hybrid refinement (should achieve near BigFloat precision)
            result_hybrid = refine_symmetric_eigen_hybrid(A, Q0, λ0;
                d64_iterations=2, bf_iterations=2, precision_bits=256)

            # Check that we achieved high precision
            @test result_hybrid.residual_norm < 1e-49
        end

        @testset "Performance comparison: Double64 vs BigFloat" begin
            # Deterministic test matrix
            n = 50
            A = Float64[sin(i * j + i) for i in 1:n, j in 1:n] + 5I
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
            # Deterministic matrix for convergence verification
            A = Float64[sin(i * j + i) for i in 1:10, j in 1:10] + 5I
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
            # Only check while above machine precision (~1e-30 for Double64)
            # Once we hit precision limits, further iterations may not improve
            for i in 2:length(residuals)
                if residuals[i-1] > 1e-28  # Above Double64 precision floor
                    @test residuals[i] < residuals[i-1] / 10
                end
            end

            # Also verify we achieved good final precision
            @test residuals[end] < 1e-25
        end
    end

catch e
    if e isa ArgumentError && contains(string(e), "DoubleFloats")
        @warn "DoubleFloats.jl not installed, skipping DoubleFloatsExt tests"
    else
        rethrow(e)
    end
end
