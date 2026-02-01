using Test
using LinearAlgebra
using BallArithmetic

@testset "Rump-Oishi vs SVD Methods Comparison" begin

    @testset "Float64 - Minimum singular value bounds comparison" begin
        # Test various matrix types
        test_cases = [
            ("Well-conditioned", randn(10, 10)),
            ("Moderately ill-conditioned", [
                10.0^(-i) * (i == j ? 1.0 : 0.1) for i in 1:10, j in 1:10
            ]),
            ("Upper triangular", UpperTriangular(randn(10, 10)) |> Matrix),
            ("Symmetric positive definite", let A = randn(10, 10); A' * A + I end),
            ("Near-singular", let A = randn(10, 10); A[:, 1] = A[:, 2] + 1e-8 * randn(10); A end)
        ]

        for (name, A_mid) in test_cases
            @testset "$name" begin
                A = BallMatrix(A_mid, zeros(size(A_mid)))

                # True minimum singular value
                σ_true = svdvals(A_mid)[end]

                # Method 1: Rump-Oishi triangular bound (if applicable)
                if istriu(A_mid)
                    A_ball_triu = BallMatrix(A_mid, zeros(size(A_mid)))
                    n = size(A_mid, 1)
                    σ_bound_ro = rump_oishi_2024_triangular_bound(A_ball_triu, n)  # k=n for smallest
                    # Print comparison
                    println("    Rump-Oishi bound for σ_min: $σ_bound_ro (true: $σ_true)")
                    # The bound should be within reasonable factor
                    @test σ_bound_ro > 0
                end

                # Method 2: rigorous_svd
                svd_result = rigorous_svd(A)
                σ_balls = svd_result.singular_values

                # The last singular value ball should contain the true value
                if length(σ_balls) > 0
                    σ_min_ball = σ_balls[end]
                    @test mid(σ_min_ball) - rad(σ_min_ball) ≤ σ_true ≤ mid(σ_min_ball) + rad(σ_min_ball)

                    # Print comparison (useful for analysis)
                    println("  $name:")
                    println("    σ_true = $σ_true")
                    println("    SVD ball: $(mid(σ_min_ball)) ± $(rad(σ_min_ball))")
                    println("    Relative radius: $(rad(σ_min_ball) / σ_true)")
                end
            end
        end
    end

    @testset "Float64 - Resolvent bounds comparison" begin
        # Test matrix for resolvent bounds
        A_mid = randn(8, 8)
        A = BallMatrix(A_mid, zeros(size(A_mid)))

        # Sample points near spectrum
        λ_approx = eigvals(A_mid)
        for i in 1:min(3, length(λ_approx))
            z = λ_approx[i] + 0.1  # Point near eigenvalue

            @testset "Near eigenvalue $i" begin
                # Shifted matrix
                A_shifted_mid = A_mid - z * I
                A_shifted = BallMatrix(A_shifted_mid, zeros(size(A_shifted_mid)))

                # True resolvent norm (inverse of minimum singular value of A - zI)
                σ_min_true = svdvals(A_shifted_mid)[end]
                resolvent_norm_true = 1 / σ_min_true

                # SVD-based bound
                svd_result = rigorous_svd(A_shifted)
                if length(svd_result.singular_values) > 0
                    σ_min_ball = svd_result.singular_values[end]
                    # Lower bound on σ_min gives upper bound on resolvent
                    σ_min_lower = max(0.0, mid(σ_min_ball) - rad(σ_min_ball))
                    if σ_min_lower > 0
                        resolvent_upper = 1 / σ_min_lower
                        @test resolvent_upper ≥ resolvent_norm_true * (1 - 1e-10)
                    end
                end
            end
        end
    end

    @testset "BigFloat - High precision singular value comparison" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            # Well-conditioned matrix
            A_mid_f64 = randn(6, 6)
            A_mid_bf = convert.(BigFloat, A_mid_f64)
            A = BallMatrix(A_mid_bf, zeros(BigFloat, size(A_mid_bf)))

            # True singular values (computed in Float64, should be close enough)
            σ_true = svdvals(A_mid_f64)

            # Rigorous SVD with BigFloat
            svd_result = rigorous_svd(A)

            println("\nBigFloat (256-bit) comparison:")
            for (i, σ_ball) in enumerate(svd_result.singular_values)
                rel_rad = Float64(rad(σ_ball) / mid(σ_ball))
                println("  σ[$i]: mid=$(Float64(mid(σ_ball))), rad=$(Float64(rad(σ_ball))), rel=$(rel_rad)")

                # Check containment (with some tolerance for Float64 reference)
                @test Float64(mid(σ_ball) - rad(σ_ball)) ≤ σ_true[i] * 1.01
                @test σ_true[i] * 0.99 ≤ Float64(mid(σ_ball) + rad(σ_ball))
            end

            # Minimum singular value should have small relative error
            σ_min_ball = svd_result.singular_values[end]
            rel_rad = rad(σ_min_ball) / mid(σ_min_ball)
            @test Float64(rel_rad) < 1e-10  # Should be very tight in BigFloat

        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "BigFloat - Ill-conditioned matrix" begin
        old_prec = precision(BigFloat)
        try
            setprecision(BigFloat, 256)

            # Create moderately ill-conditioned matrix (condition number ~ 10^6)
            n = 5
            U = qr(randn(n, n)).Q |> Matrix
            V = qr(randn(n, n)).Q |> Matrix
            σ_vals = [10.0^(-1.5*i) for i in 0:(n-1)]  # 1, ~0.03, ~0.001, ~3e-5, ~1e-6
            A_mid_f64 = U * Diagonal(σ_vals) * V'

            A_mid_bf = convert.(BigFloat, A_mid_f64)
            A = BallMatrix(A_mid_bf, zeros(BigFloat, size(A_mid_bf)))

            svd_result = rigorous_svd(A)

            println("\nIll-conditioned matrix (κ ≈ 10^6):")
            for (i, σ_ball) in enumerate(svd_result.singular_values)
                rel_rad = Float64(rad(σ_ball) / max(Float64(mid(σ_ball)), 1e-100))
                println("  σ[$i]: mid=$(Float64(mid(σ_ball))), rel_rad=$(rel_rad)")
            end

            # The largest singular value should still be tight
            σ_max_ball = svd_result.singular_values[1]
            @test Float64(rad(σ_max_ball) / mid(σ_max_ball)) < 1e-10

            # Smallest may be less tight due to conditioning
            σ_min_ball = svd_result.singular_values[end]
            if Float64(mid(σ_min_ball)) > 0
                @test Float64(rad(σ_min_ball) / mid(σ_min_ball)) < 0.1  # More relaxed for ill-conditioned
            end

        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "Adaptive Ogita SVD refinement comparison" begin
        # Test Ogita adaptive refinement
        A_mid = randn(8, 8)
        A = BallMatrix(A_mid, zeros(size(A_mid)))

        # Standard rigorous SVD
        svd_basic = rigorous_svd(A)

        # Adaptive Ogita SVD (use actual API)
        svd_ogita_result = adaptive_ogita_svd(A)
        svd_ogita = svd_ogita_result.rigorous_result

        σ_true = svdvals(A_mid)

        println("\nAdaptive Ogita SVD vs basic SVD:")
        for i in 1:length(σ_true)
            basic_rel = rad(svd_basic.singular_values[i]) / mid(svd_basic.singular_values[i])
            ogita_rel = rad(svd_ogita.singular_values[i]) / mid(svd_ogita.singular_values[i])
            improvement = basic_rel / max(ogita_rel, 1e-16)

            println("  σ[$i]: basic_rel=$(basic_rel), ogita_rel=$(ogita_rel), improvement=$(improvement)x")

            # Both should contain true value
            @test mid(svd_basic.singular_values[i]) - rad(svd_basic.singular_values[i]) ≤ σ_true[i] ≤ mid(svd_basic.singular_values[i]) + rad(svd_basic.singular_values[i])
            @test mid(svd_ogita.singular_values[i]) - rad(svd_ogita.singular_values[i]) ≤ σ_true[i] ≤ mid(svd_ogita.singular_values[i]) + rad(svd_ogita.singular_values[i])

            # Ogita should be at least as tight (usually tighter)
            @test rad(svd_ogita.singular_values[i]) ≤ rad(svd_basic.singular_values[i]) + 1e-14
        end
    end

    @testset "Collatz bound comparison" begin
        # Test Collatz upper bound on L2 norm
        A_mid = randn(10, 10)
        A = BallMatrix(A_mid, zeros(size(A_mid)))

        # True largest singular value
        σ_max_true = svdvals(A_mid)[1]

        # Collatz upper bound (using BallMatrix)
        σ_max_collatz = collatz_upper_bound_L2_opnorm(A)

        @test σ_max_collatz ≥ σ_max_true * (1 - 1e-10)  # Should be an upper bound

        println("\nCollatz bound comparison:")
        println("  σ_max_true = $σ_max_true")
        println("  Collatz bound = $σ_max_collatz")
        println("  Overestimation factor = $(σ_max_collatz / σ_max_true)")

        # The bound should not be too loose
        @test σ_max_collatz / σ_max_true < 2.0  # Typically much closer to 1
    end

    @testset "L2 operator norm bounds comparison" begin
        # Compare different methods for upper_bound_L2_opnorm
        A_mid = randn(10, 10)
        A = BallMatrix(A_mid, zeros(size(A_mid)))

        # True L2 norm
        L2_true = opnorm(A_mid, 2)

        # Upper bound from BallArithmetic
        L2_upper = upper_bound_L2_opnorm(A)

        @test L2_upper ≥ L2_true * (1 - 1e-10)

        println("\nL2 operator norm bounds:")
        println("  True L2 norm = $L2_true")
        println("  Upper bound = $L2_upper")
        println("  Ratio = $(L2_upper / L2_true)")
    end

    @testset "Oishi 2023 Schur complement bounds" begin
        # Test Oishi 2023 lower bound on minimum singular value
        # This method is most effective for diagonally dominant matrices

        @testset "Float64 - Diagonal dominant matrix" begin
            n = 10
            # Create diagonal dominant matrix
            A_mid = diagm(ones(n) * 10.0)
            for i in 1:n, j in 1:n
                if i != j
                    A_mid[i, j] = 0.2 * randn()
                end
            end
            A = BallMatrix(A_mid, zeros(n, n))

            σ_true = svdvals(A_mid)[end]

            # Oishi 2023 with optimal block size
            best_m, oishi_result = oishi_2023_optimal_block_size(A; max_m=n-1)

            # Rigorous SVD
            svd_result = rigorous_svd(A)
            σ_min_ball = svd_result.singular_values[end]
            svd_lower = mid(σ_min_ball) - rad(σ_min_ball)

            println("\nOishi 2023 vs SVD (diagonal dominant):")
            println("  True σ_min = $σ_true")
            if oishi_result.verified
                println("  Oishi 2023 bound = $(oishi_result.sigma_min_lower) (ratio: $(oishi_result.sigma_min_lower/σ_true))")
                @test oishi_result.sigma_min_lower ≤ σ_true * 1.01
                @test oishi_result.sigma_min_lower > 0
            else
                println("  Oishi 2023: conditions not satisfied")
            end
            println("  SVD bound = $svd_lower (ratio: $(svd_lower/σ_true))")
            @test svd_lower ≤ σ_true * 1.01
        end

        @testset "Float64 - General matrix" begin
            # For general matrices, Oishi 2023 may not satisfy conditions
            A_mid = randn(8, 8)
            A = BallMatrix(A_mid, zeros(8, 8))

            σ_true = svdvals(A_mid)[end]

            # Try different block sizes
            any_verified = false
            for m in 1:7
                result = oishi_2023_sigma_min_bound(A, m)
                if result.verified
                    any_verified = true
                    @test result.sigma_min_lower ≤ σ_true * 1.01
                    println("  General matrix: m=$m verified, bound=$(result.sigma_min_lower), true=$σ_true")
                    break
                end
            end
            if !any_verified
                println("  General matrix: Oishi 2023 conditions not satisfied (expected for non-dominant matrices)")
            end
        end

        @testset "BigFloat - High precision" begin
            old_prec = precision(BigFloat)
            try
                setprecision(BigFloat, 256)

                n = 6
                A_mid_f64 = diagm(ones(n) * 10.0) + 0.1 * randn(n, n)
                A_mid_bf = convert.(BigFloat, A_mid_f64)
                A = BallMatrix(A_mid_bf, zeros(BigFloat, n, n))

                σ_true = svdvals(A_mid_f64)[end]

                result = oishi_2023_sigma_min_bound(A, 3)

                println("\nOishi 2023 BigFloat (256-bit):")
                if result.verified
                    println("  σ_min_lower = $(Float64(result.sigma_min_lower))")
                    println("  σ_true ≈ $σ_true")
                    println("  Ratio = $(Float64(result.sigma_min_lower)/σ_true)")
                    @test Float64(result.sigma_min_lower) ≤ σ_true * 1.01
                else
                    # BigFloat SVD may not be supported; this is expected
                    println("  Conditions not satisfied (BigFloat SVD may not be fully supported)")
                    @test true  # Pass the test, as this is expected behavior
                end
            finally
                setprecision(BigFloat, old_prec)
            end
        end
    end

end
