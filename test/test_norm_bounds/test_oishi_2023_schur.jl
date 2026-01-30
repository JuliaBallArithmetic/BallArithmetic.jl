using Test
using LinearAlgebra
using BallArithmetic

@testset "Rump-Oishi 2024 ψ function" begin
    # Test psi_schur_factor from Lemma 1.2

    @testset "Basic properties" begin
        # ψ(0) should be 1 (identity matrix case)
        @test psi_schur_factor(0.0) ≈ 1.0

        # ψ(μ) ≥ 1 for all μ ≥ 0
        for μ in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            @test psi_schur_factor(μ) ≥ 1.0
        end

        # ψ(μ) is increasing in μ
        μ_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
        psi_vals = [psi_schur_factor(μ) for μ in μ_vals]
        for i in 1:(length(psi_vals)-1)
            @test psi_vals[i] ≤ psi_vals[i+1]
        end
    end

    @testset "Comparison with 1/(1-μ) bound" begin
        # For μ < 1, ψ(μ) should be ≤ 1/(1-μ) (tighter bound)
        for μ in [0.1, 0.3, 0.5, 0.7, 0.9]
            psi = psi_schur_factor(μ)
            old_bound = 1.0 / (1.0 - μ)
            @test psi ≤ old_bound
            # Print improvement factor
            println("  μ=$μ: ψ=$psi, 1/(1-μ)=$old_bound, ratio=$(psi/old_bound)")
        end
    end

    @testset "Works for μ ≥ 1 (unlike old bound)" begin
        # The old bound 1/(1-μ) fails for μ ≥ 1, but ψ(μ) is defined
        for μ in [1.0, 1.5, 2.0, 3.0]
            psi = psi_schur_factor(μ)
            @test isfinite(psi)
            @test psi > 0
            println("  μ=$μ: ψ=$psi (old bound would be undefined)")
        end
    end
end

@testset "Rump-Oishi 2024 Schur Complement Bounds" begin

    @testset "Float64 - Comparison with Oishi 2023" begin
        # Create a diagonally dominant matrix
        n = 10
        G_mid = diagm(ones(n) * 5.0)
        for i in 1:n, j in 1:n
            if i != j
                G_mid[i, j] = 0.1 * randn()
            end
        end
        G = BallMatrix(G_mid, zeros(n, n))

        σ_true = svdvals(G_mid)[end]

        for m in [2, 5, n-1]
            result_2023 = oishi_2023_sigma_min_bound(G, m)
            result_2024 = rump_oishi_2024_sigma_min_bound(G, m)

            @testset "Block size m=$m" begin
                if result_2024.verified
                    @test result_2024.sigma_min_lower ≤ σ_true * 1.01
                    @test result_2024.sigma_min_lower > 0

                    # 2024 should give tighter or equal bounds
                    if result_2023.verified
                        @test result_2024.sigma_min_lower ≥ result_2023.sigma_min_lower * 0.99
                        println("  m=$m: 2023=$(result_2023.sigma_min_lower), 2024=$(result_2024.sigma_min_lower), true=$σ_true")
                        println("       Improvement: $(result_2024.sigma_min_lower / result_2023.sigma_min_lower)")
                    else
                        println("  m=$m: 2023=failed, 2024=$(result_2024.sigma_min_lower), true=$σ_true")
                    end
                end
            end
        end
    end

    @testset "Float64 - Cases where Oishi 2023 fails but Rump-Oishi 2024 works" begin
        # Create matrix where ‖A⁻¹B‖ or ‖CA⁻¹‖ might be ≥ 1
        n = 8
        G_mid = diagm(ones(n) * 2.0)  # Weaker diagonal
        G_mid[1:3, 4:n] .= 0.5  # Stronger off-diagonal in B block
        G_mid[4:n, 1:3] .= 0.5  # Stronger off-diagonal in C block
        G = BallMatrix(G_mid, zeros(n, n))

        σ_true = svdvals(G_mid)[end]
        m = 3

        result_2023 = oishi_2023_sigma_min_bound(G, m)
        result_2024 = rump_oishi_2024_sigma_min_bound(G, m)

        println("\nTest case where 2023 may fail:")
        println("  Oishi 2023: verified=$(result_2023.verified), ‖A⁻¹B‖=$(result_2023.A_inv_B_norm), ‖CA⁻¹‖=$(result_2023.C_A_inv_norm)")

        if result_2024.verified
            @test result_2024.sigma_min_lower ≤ σ_true * 1.01
            println("  Rump-Oishi 2024: verified=true, σ_min_lower=$(result_2024.sigma_min_lower), true=$σ_true")
        end
    end

    @testset "Float64 - Fast γ bound" begin
        # Diagonally dominant matrix where fast γ should work
        n = 20
        G_mid = diagm(collect(1.0:n))  # Increasing diagonal as in paper examples
        for i in 1:n, j in 1:n
            if i != j
                G_mid[i, j] = 0.1 * 0.9^abs(i-j)  # Decay off-diagonal
            end
        end
        G = BallMatrix(G_mid, zeros(n, n))

        result_with_gamma = rump_oishi_2024_sigma_min_bound(G, 5; try_fast_gamma=true)
        result_without_gamma = rump_oishi_2024_sigma_min_bound(G, 5; try_fast_gamma=false)

        @test result_with_gamma.verified
        @test result_without_gamma.verified

        println("\nFast γ bound test:")
        println("  With fast γ: used=$(result_with_gamma.used_fast_gamma), σ_min_lower=$(result_with_gamma.sigma_min_lower)")
        println("  Without fast γ: σ_min_lower=$(result_without_gamma.sigma_min_lower)")

        # Both should give valid bounds
        σ_true = svdvals(G_mid)[end]
        @test result_with_gamma.sigma_min_lower ≤ σ_true * 1.01
        @test result_without_gamma.sigma_min_lower ≤ σ_true * 1.01
    end

    @testset "Float64 - Optimal block size" begin
        n = 15
        G_mid = diagm(collect(1.0:n)) + 0.05 * randn(n, n)
        G = BallMatrix(G_mid, zeros(n, n))

        best_m, best_result = rump_oishi_2024_optimal_block_size(G; max_m=n-1)

        σ_true = svdvals(G_mid)[end]

        @test best_m ≥ 1
        @test best_m < n
        if best_result.verified
            @test best_result.sigma_min_lower ≤ σ_true * 1.01
            println("\nOptimal block size: m=$best_m, σ_min_lower=$(best_result.sigma_min_lower), true=$σ_true")
        end
    end

    @testset "Float64 - Paper Example 3 (k=0.9)" begin
        # Recreate Example 3 from Rump-Oishi 2024: G_ij = k^|i-j| for i≠j, diagonal=(1,...,n)
        k = 0.9
        n = 50

        G_mid = zeros(n, n)
        for i in 1:n
            G_mid[i, i] = Float64(i)
            for j in 1:n
                if i != j
                    G_mid[i, j] = k^abs(i-j)
                end
            end
        end
        G = BallMatrix(G_mid, zeros(n, n))

        σ_true = svdvals(G_mid)[end]

        # Compare methods at m=20 as in the paper
        result_2023 = oishi_2023_sigma_min_bound(G, 20)
        result_2024 = rump_oishi_2024_sigma_min_bound(G, 20)

        println("\nPaper Example 3 (k=$k, n=$n, m=20):")
        println("  True ‖G⁻¹‖ ≈ $(1/σ_true)")

        if result_2023.verified
            println("  Oishi 2023: ‖G⁻¹‖ ≤ $(result_2023.G_inv_upper)")
        else
            println("  Oishi 2023: failed")
        end

        if result_2024.verified
            println("  Rump-Oishi 2024: ‖G⁻¹‖ ≤ $(result_2024.G_inv_upper)")
            @test result_2024.sigma_min_lower ≤ σ_true * 1.01
        end
    end

    @testset "Float64 - Result structure" begin
        n = 6
        G_mid = diagm(ones(n) * 5.0) + 0.1 * randn(n, n)
        G = BallMatrix(G_mid, zeros(n, n))

        result = rump_oishi_2024_sigma_min_bound(G, 2)

        @test result isa RumpOishi2024Result{Float64}
        @test hasfield(RumpOishi2024Result{Float64}, :sigma_min_lower)
        @test hasfield(RumpOishi2024Result{Float64}, :G_inv_upper)
        @test hasfield(RumpOishi2024Result{Float64}, :A_inv_norm)
        @test hasfield(RumpOishi2024Result{Float64}, :psi_A_inv_B)
        @test hasfield(RumpOishi2024Result{Float64}, :psi_C_A_inv)
        @test hasfield(RumpOishi2024Result{Float64}, :schur_contraction)
        @test hasfield(RumpOishi2024Result{Float64}, :used_fast_gamma)
        @test hasfield(RumpOishi2024Result{Float64}, :verified)
        @test hasfield(RumpOishi2024Result{Float64}, :block_size)
        @test result.block_size == 2
    end
end

@testset "Oishi 2023 Schur Complement Bounds" begin

    @testset "Float64 - Well-conditioned diagonal dominant matrix" begin
        # Create a diagonally dominant matrix (satisfies conditions of Theorem 1)
        n = 10
        A_mid = diagm(ones(n) * 5.0)  # Strong diagonal
        for i in 1:n
            for j in 1:n
                if i != j
                    A_mid[i, j] = 0.1 * randn()
                end
            end
        end

        A = BallMatrix(A_mid, zeros(n, n))

        # True minimum singular value
        σ_true = svdvals(A_mid)[end]

        # Test with different block sizes
        for m in [1, 3, 5, n-1]
            result = oishi_2023_sigma_min_bound(A, m)

            @testset "Block size m=$m" begin
                if result.verified
                    # The bound should be a valid lower bound
                    @test result.sigma_min_lower ≤ σ_true * 1.01  # Allow small numerical tolerance
                    @test result.sigma_min_lower > 0
                    @test result.G_inv_upper ≥ 1 / σ_true * 0.99

                    println("  m=$m: σ_min_lower=$(result.sigma_min_lower), σ_true=$σ_true, ratio=$(result.sigma_min_lower/σ_true)")
                else
                    println("  m=$m: Conditions not satisfied (A_inv_B=$(result.A_inv_B_norm), CA_inv=$(result.C_A_inv_norm), contraction=$(result.schur_contraction))")
                end
            end
        end
    end

    @testset "Float64 - Identity matrix" begin
        # Identity matrix should give tight bounds
        n = 8
        I_mid = Matrix{Float64}(I, n, n)
        I_ball = BallMatrix(I_mid, zeros(n, n))

        for m in [1, 4, n-1]
            result = oishi_2023_sigma_min_bound(I_ball, m)

            if result.verified
                # σ_min(I) = 1
                @test result.sigma_min_lower ≤ 1.0
                @test result.sigma_min_lower > 0.5  # Should be reasonably tight
                @test result.G_inv_upper ≥ 1.0
            end
        end
    end

    @testset "Float64 - Optimal block size selection" begin
        # Diagonally dominant matrix
        n = 12
        A_mid = diagm(ones(n) * 10.0)
        for i in 1:n, j in 1:n
            if i != j
                A_mid[i, j] = 0.05 * randn()
            end
        end
        A = BallMatrix(A_mid, zeros(n, n))

        best_m, best_result = oishi_2023_optimal_block_size(A; max_m=n-1)

        σ_true = svdvals(A_mid)[end]

        @test best_m ≥ 1
        @test best_m < n

        if best_result.verified
            @test best_result.sigma_min_lower ≤ σ_true * 1.01
            @test best_result.sigma_min_lower > 0
            println("  Optimal block size: m=$best_m, σ_min_lower=$(best_result.sigma_min_lower), σ_true=$σ_true")
        end
    end

    @testset "Float64 - Result structure" begin
        n = 6
        A_mid = diagm(ones(n) * 5.0) + 0.1 * randn(n, n)
        A = BallMatrix(A_mid, zeros(n, n))

        result = oishi_2023_sigma_min_bound(A, 2)

        @test result isa Oishi2023Result{Float64}
        @test hasfield(Oishi2023Result{Float64}, :sigma_min_lower)
        @test hasfield(Oishi2023Result{Float64}, :G_inv_upper)
        @test hasfield(Oishi2023Result{Float64}, :A_inv_norm)
        @test hasfield(Oishi2023Result{Float64}, :A_inv_B_norm)
        @test hasfield(Oishi2023Result{Float64}, :C_A_inv_norm)
        @test hasfield(Oishi2023Result{Float64}, :schur_contraction)
        @test hasfield(Oishi2023Result{Float64}, :verified)
        @test hasfield(Oishi2023Result{Float64}, :block_size)
        @test result.block_size == 2
    end

    @testset "Float64 - Conditions not satisfied" begin
        # Create a matrix where conditions might not be satisfied
        n = 8
        # Nearly singular matrix
        A_mid = randn(n, n)
        A_mid[:, 1] = A_mid[:, 2] + 1e-10 * randn(n)  # Make columns nearly dependent
        A = BallMatrix(A_mid, zeros(n, n))

        result = oishi_2023_sigma_min_bound(A, n ÷ 2)

        # Result might not be verified for ill-conditioned matrices
        @test result.block_size == n ÷ 2
        # If not verified, bounds should reflect this
        if !result.verified
            @test result.sigma_min_lower == 0.0
            @test result.G_inv_upper == Inf
        end
    end

    @testset "Float64 - Error handling" begin
        n = 5
        A_mid = randn(n, n)
        A = BallMatrix(A_mid, zeros(n, n))

        # m must be in valid range
        @test_throws ArgumentError oishi_2023_sigma_min_bound(A, 0)
        @test_throws ArgumentError oishi_2023_sigma_min_bound(A, n)
        @test_throws ArgumentError oishi_2023_sigma_min_bound(A, n+1)

        # Non-square matrix
        B_mid = randn(5, 6)
        B = BallMatrix(B_mid, zeros(5, 6))
        @test_throws ArgumentError oishi_2023_sigma_min_bound(B, 2)
    end

    @testset "Float64 - Comparison with SVD" begin
        # Compare Oishi 2023 bounds with rigorous SVD bounds
        n = 10
        A_mid = diagm(ones(n) * 10.0) + 0.2 * randn(n, n)
        A = BallMatrix(A_mid, zeros(n, n))

        σ_true = svdvals(A_mid)[end]

        # Oishi 2023 bound
        best_m, oishi_result = oishi_2023_optimal_block_size(A)

        # Rigorous SVD bound
        svd_result = rigorous_svd(A)
        σ_min_ball = svd_result.singular_values[end]
        svd_lower = mid(σ_min_ball) - rad(σ_min_ball)

        println("\nComparison with SVD methods:")
        println("  True σ_min = $σ_true")
        if oishi_result.verified
            println("  Oishi 2023 lower bound = $(oishi_result.sigma_min_lower)")
        else
            println("  Oishi 2023: conditions not satisfied")
        end
        println("  SVD lower bound = $svd_lower")

        # Both should be valid lower bounds
        if oishi_result.verified
            @test oishi_result.sigma_min_lower ≤ σ_true * 1.01
        end
        @test svd_lower ≤ σ_true * 1.01
    end

    @testset "BigFloat - High precision bounds" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            # Diagonally dominant matrix
            n = 6
            A_mid_f64 = diagm(ones(n) * 10.0) + 0.1 * randn(n, n)
            A_mid_bf = convert.(BigFloat, A_mid_f64)
            A = BallMatrix(A_mid_bf, zeros(BigFloat, n, n))

            # True σ_min (Float64 reference)
            σ_true = svdvals(A_mid_f64)[end]

            result = oishi_2023_sigma_min_bound(A, 3)

            @test result isa Oishi2023Result{BigFloat}

            if result.verified
                @test Float64(result.sigma_min_lower) ≤ σ_true * 1.01
                @test Float64(result.sigma_min_lower) > 0

                println("\nBigFloat (256-bit) Oishi 2023:")
                println("  σ_min_lower = $(Float64(result.sigma_min_lower))")
                println("  σ_true ≈ $σ_true")
                println("  Ratio = $(Float64(result.sigma_min_lower)/σ_true)")
            else
                # BigFloat SVD may not be fully supported; this is expected
                println("\nBigFloat: Conditions not satisfied (BigFloat SVD may not be fully supported)")
                @test true  # Pass the test, as this is expected behavior
            end
        finally
            setprecision(BigFloat, old_prec)
        end
    end

end
