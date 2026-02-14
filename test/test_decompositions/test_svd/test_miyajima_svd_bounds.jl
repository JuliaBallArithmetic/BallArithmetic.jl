using Test
using LinearAlgebra
using BallArithmetic

@testset "Miyajima 2014 SVD Bounds" begin

    @testset "MiyajimaM1 vs RumpOriginal - M1 gives tighter bounds" begin
        # Create a simple test matrix
        A_mid = [3.0 1.0 0.5; 0.0 2.0 0.3; 0.0 0.0 1.0]
        A_rad = fill(1e-10, size(A_mid))
        A = BallMatrix(A_mid, A_rad)

        result_m1 = rigorous_svd(A; method=MiyajimaM1(), apply_vbd=false)
        result_rump = rigorous_svd(A; method=RumpOriginal(), apply_vbd=false)

        # Both should give valid bounds containing the true singular values
        true_svd = svd(A_mid)

        for i in 1:length(true_svd.S)
            σ_true = true_svd.S[i]

            # M1 bounds should contain true value
            m1_lower = mid(result_m1.singular_values[i]) - rad(result_m1.singular_values[i])
            m1_upper = mid(result_m1.singular_values[i]) + rad(result_m1.singular_values[i])
            @test m1_lower <= σ_true <= m1_upper

            # Rump bounds should contain true value
            rump_lower = mid(result_rump.singular_values[i]) - rad(result_rump.singular_values[i])
            rump_upper = mid(result_rump.singular_values[i]) + rad(result_rump.singular_values[i])
            @test rump_lower <= σ_true <= rump_upper

            # M1 should give equal or tighter bounds than Rump
            m1_width = 2 * rad(result_m1.singular_values[i])
            rump_width = 2 * rad(result_rump.singular_values[i])
            @test m1_width <= rump_width + 1e-14  # Small tolerance for numerical noise
        end
    end

    @testset "MiyajimaM1 bound formula verification" begin
        # Test the formula: lower = σ * sqrt((1-F)(1-G)) - E
        #                   upper = σ * sqrt((1+F)(1+G)) + E
        A_mid = Diagonal([5.0, 3.0, 1.0])
        A = BallMatrix(Matrix(A_mid))

        result = rigorous_svd(A; method=MiyajimaM1(), apply_vbd=false)

        # For diagonal matrix, F and G should be very small, E should be very small
        # so bounds should be very tight around the true singular values
        for i in 1:3
            σ_true = A_mid[i, i]
            σ_ball = result.singular_values[i]
            @test abs(mid(σ_ball) - σ_true) < 1e-10
            @test rad(σ_ball) < 1e-10
        end
    end

    @testset "Comparison: M1 vs Rump on ill-conditioned matrix" begin
        # Create an ill-conditioned matrix where bounds differ more
        n = 5
        A_mid = zeros(n, n)
        for i in 1:n
            A_mid[i, i] = 10.0^(2 - i)  # Singular values: 100, 10, 1, 0.1, 0.01
        end
        # Add small off-diagonal perturbation
        A_mid[1, 2] = 0.1
        A_mid[2, 3] = 0.05

        A = BallMatrix(A_mid, fill(1e-12, size(A_mid)))

        result_m1 = rigorous_svd(A; method=MiyajimaM1(), apply_vbd=false)
        result_rump = rigorous_svd(A; method=RumpOriginal(), apply_vbd=false)

        true_svd = svd(A_mid)

        total_m1_width = sum(rad.(result_m1.singular_values))
        total_rump_width = sum(rad.(result_rump.singular_values))

        # M1 should be at least as good or better
        @test total_m1_width <= total_rump_width + 1e-10

        # Both should contain true values
        for i in 1:n
            @test true_svd.S[i] ∈ result_m1.singular_values[i]
            @test true_svd.S[i] ∈ result_rump.singular_values[i]
        end
    end

    @testset "MiyajimaM4 eigendecomposition method" begin
        # Test the M4 method
        A_mid = Diagonal([4.0, 2.0, 1.0])
        A = BallMatrix(Matrix(A_mid))

        result_m4 = rigorous_svd_m4(A; apply_vbd=false)

        # Should give correct bounds
        for i in 1:3
            σ_true = A_mid[i, i]
            σ_ball = result_m4.singular_values[i]
            @test mid(σ_ball) - rad(σ_ball) <= σ_true <= mid(σ_ball) + rad(σ_ball)
        end
    end

    @testset "VBD refinement for isolated singular values" begin
        # Create matrix with well-separated singular values
        A_mid = Diagonal([10.0, 5.0, 1.0])
        A = BallMatrix(Matrix(A_mid))

        result = rigorous_svd(A; method=MiyajimaM1(), apply_vbd=true)

        # All singular values should be isolated (each in its own cluster)
        @test result.block_diagonalisation !== nothing
        @test length(result.block_diagonalisation.clusters) == 3

        # Try VBD refinement
        refined = refine_svd_bounds_with_vbd(result)

        # Refined should still be valid
        for i in 1:3
            σ_true = A_mid[i, i]
            @test σ_true ∈ refined.singular_values[i]
        end
    end

    @testset "Method type exports and selection" begin
        A = BallMatrix(Diagonal([3.0, 2.0, 1.0]))

        # All methods should work
        @test rigorous_svd(A; method=MiyajimaM1()) isa RigorousSVDResult
        @test rigorous_svd(A; method=MiyajimaM4()) isa RigorousSVDResult
        @test rigorous_svd(A; method=RumpOriginal()) isa RigorousSVDResult

        # Default should be MiyajimaM1
        result_default = rigorous_svd(A)
        result_m1 = rigorous_svd(A; method=MiyajimaM1())
        @test result_default.singular_values == result_m1.singular_values
    end

    @testset "svdbox with method parameter" begin
        A = BallMatrix(Diagonal([5.0, 3.0, 1.0]))

        sv_default = svdbox(A)
        sv_m1 = svdbox(A; method=MiyajimaM1())
        sv_rump = svdbox(A; method=RumpOriginal())

        # Default should match M1
        @test sv_default == sv_m1

        # All should be valid
        @test length(sv_default) == 3
        @test length(sv_rump) == 3
    end

    @testset "Complex matrix support" begin
        A_mid = [1.0+im 0.5; 0.0 2.0-im]
        A = BallMatrix(A_mid)

        result_m1 = rigorous_svd(A; method=MiyajimaM1(), apply_vbd=false)
        result_rump = rigorous_svd(A; method=RumpOriginal(), apply_vbd=false)

        true_svd = svd(A_mid)

        for i in 1:2
            @test true_svd.S[i] ∈ result_m1.singular_values[i]
            @test true_svd.S[i] ∈ result_rump.singular_values[i]
        end
    end

    @testset "Rectangular matrix support" begin
        # Tall matrix
        A_mid = [3.0 1.0; 0.5 2.0; 0.1 0.3]
        A = BallMatrix(A_mid)

        result = rigorous_svd(A; method=MiyajimaM1())
        true_svd = svd(A_mid)

        @test length(result.singular_values) == 2
        for i in 1:2
            @test true_svd.S[i] ∈ result.singular_values[i]
        end

        # Wide matrix
        B_mid = A_mid'
        B = BallMatrix(B_mid)

        result_b = rigorous_svd(B; method=MiyajimaM1())
        true_svd_b = svd(B_mid)

        @test length(result_b.singular_values) == 2
        for i in 1:2
            @test true_svd_b.S[i] ∈ result_b.singular_values[i]
        end
    end

    @testset "Numerical improvement demonstration" begin
        # This test demonstrates the improvement from M1 over Rump
        # on a concrete example

        A_mid = [5.0 0.1 0.0;
                 0.0 3.0 0.05;
                 0.0 0.0 1.0]
        A = BallMatrix(A_mid, fill(1e-14, size(A_mid)))

        result_m1 = rigorous_svd(A; method=MiyajimaM1(), apply_vbd=false)
        result_rump = rigorous_svd(A; method=RumpOriginal(), apply_vbd=false)

        println("\n=== Miyajima M1 vs Rump Original Comparison ===")
        for i in 1:3
            m1_rad = rad(result_m1.singular_values[i])
            rump_rad = rad(result_rump.singular_values[i])
            improvement = (rump_rad - m1_rad) / rump_rad * 100

            println("σ$i: M1 radius = $(m1_rad), Rump radius = $(rump_rad)")
            println("    Improvement: $(round(improvement, digits=1))%")
        end

        # M1 should be strictly better (or equal)
        for i in 1:3
            @test rad(result_m1.singular_values[i]) <= rad(result_rump.singular_values[i]) + eps()
        end
    end

end
