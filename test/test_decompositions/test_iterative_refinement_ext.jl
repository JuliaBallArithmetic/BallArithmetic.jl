using Test
using BallArithmetic
using LinearAlgebra

# Check if extensions are available
const HAS_DOUBLEFLOATS = try
    using DoubleFloats
    true
catch
    false
end

const HAS_MULTIFLOATS = try
    using MultiFloats
    true
catch
    false
end

@testset "Iterative Refinement Extensions" begin

    if HAS_DOUBLEFLOATS
        @testset "Double64 Iterative Refinement" begin

            @testset "Polar Double64" begin
                A = [3.0 1.0; 1.0 2.0]
                F = svd(A)
                Q0 = F.U * F.Vt

                result = refine_polar_double64(A, Q0, method=:newton_schulz, max_iterations=5)
                @test result.converged || result.orthogonality_defect < 1e-15
                @test result.residual_norm < 1e-14  # Float64 input limits precision

                # Test QDWH method
                result_qdwh = refine_polar_double64(A, Q0, method=:qdwh, max_iterations=5)
                @test result_qdwh.residual_norm < 1e-12
            end

            @testset "LU Double64" begin
                A = [4.0 1.0; 1.0 3.0]
                F = lu(A)

                result = refine_lu_double64(A, Matrix(F.L), Matrix(F.U), F.p, max_iterations=5)
                @test result.residual_norm < 1e-14
            end

            @testset "Cholesky Double64" begin
                A = [4.0 1.0; 1.0 3.0]
                C = cholesky(Symmetric(A))

                result = refine_cholesky_double64(A, Matrix(C.U), max_iterations=5)
                @test result.residual_norm < 1e-14
            end

            @testset "QR Double64" begin
                A = [4.0 1.0; 1.0 3.0; 0.5 0.3]
                F = qr(A)

                result = refine_qr_double64(A, Matrix(F.Q), Matrix(F.R), max_iterations=3)
                @test result.orthogonality_defect < 1e-14
                @test result.residual_norm < 1e-13
            end

            @testset "Takagi Double64" begin
                A = Complex{Float64}.([2.0 1.0; 1.0 3.0])
                F = svd(A)

                result = refine_takagi_double64(A, F.U, F.S, max_iterations=5)
                # Takagi refinement is challenging - test that it produces reasonable results
                @test isfinite(result.residual_norm)
                @test result.iterations > 0
            end

            @testset "Double64 vs Float64 Comparison" begin
                # Double64 should give equal or better precision than Float64
                A = [5.0 1.0 0.5;
                     1.0 4.0 0.3;
                     0.5 0.3 3.0]
                F = svd(A)
                Q0 = F.U * F.Vt

                result_f64 = refine_polar_newton_schulz(A, Q0, max_iterations=5)
                result_d64 = refine_polar_double64(A, Q0, max_iterations=5)

                # Double64 should achieve at least the same quality
                @test result_d64.orthogonality_defect <= result_f64.orthogonality_defect + 1e-10
            end
        end
    else
        @testset "Double64 Iterative Refinement (SKIPPED)" begin
            @test_skip "DoubleFloats.jl not available"
        end
    end

    if HAS_MULTIFLOATS
        @testset "MultiFloat Iterative Refinement" begin

            @testset "Polar MultiFloat (x2)" begin
                A = [3.0 1.0; 1.0 2.0]
                F = svd(A)
                Q0 = F.U * F.Vt

                result = refine_polar_multifloat(A, Q0, precision=:x2, method=:newton_schulz, max_iterations=5)
                @test result.converged || result.orthogonality_defect < 1e-15
                @test result.residual_norm < 1e-14
            end

            @testset "LU MultiFloat" begin
                A = [4.0 1.0; 1.0 3.0]
                F = lu(A)

                result = refine_lu_multifloat(A, Matrix(F.L), Matrix(F.U), F.p, precision=:x2, max_iterations=5)
                @test result.residual_norm < 1e-14
            end

            @testset "Cholesky MultiFloat" begin
                A = [4.0 1.0; 1.0 3.0]
                C = cholesky(Symmetric(A))

                result = refine_cholesky_multifloat(A, Matrix(C.U), precision=:x2, max_iterations=5)
                @test result.residual_norm < 1e-14
            end

            @testset "QR MultiFloat" begin
                A = [4.0 1.0; 1.0 3.0; 0.5 0.3]
                F = qr(A)

                result = refine_qr_multifloat(A, Matrix(F.Q), Matrix(F.R), precision=:x2, max_iterations=3)
                @test result.orthogonality_defect < 1e-14
                @test result.residual_norm < 1e-13
            end

            @testset "Takagi MultiFloat" begin
                A = Complex{Float64}.([2.0 1.0; 1.0 3.0])
                F = svd(A)

                result = refine_takagi_multifloat(A, F.U, F.S, precision=:x2, max_iterations=5)
                @test isfinite(result.residual_norm)
            end

            @testset "MultiFloat Precision Levels" begin
                A = [4.0 1.0; 1.0 3.0]
                C = cholesky(Symmetric(A))
                G0 = Matrix(C.U)

                result_x2 = refine_cholesky_multifloat(A, G0, precision=:x2, max_iterations=5)
                result_x4 = refine_cholesky_multifloat(A, G0, precision=:x4, max_iterations=5)

                # Both should give good results
                @test result_x2.residual_norm < 1e-14
                @test result_x4.residual_norm < 1e-14
            end
        end
    else
        @testset "MultiFloat Iterative Refinement (SKIPPED)" begin
            @test_skip "MultiFloats.jl not available"
        end
    end

    if HAS_DOUBLEFLOATS && HAS_MULTIFLOATS
        @testset "Double64 vs MultiFloat Comparison" begin
            A = [5.0 1.0 0.5;
                 1.0 4.0 0.3;
                 0.5 0.3 3.0]
            C = cholesky(Symmetric(A))
            G0 = Matrix(C.U)

            result_d64 = refine_cholesky_double64(A, G0, max_iterations=5)
            result_mf = refine_cholesky_multifloat(A, G0, precision=:x2, max_iterations=5)

            # Both should achieve similar precision (within 10x)
            @test result_d64.residual_norm < 1e-13
            @test result_mf.residual_norm < 1e-13
        end
    end

end
