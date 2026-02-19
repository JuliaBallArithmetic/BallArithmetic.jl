using Test
using LinearAlgebra
using BallArithmetic

@testset "Sylvester Resolvent Bound" begin

    @testset "Triangular inverse bounds" begin
        # Test with a simple upper triangular matrix
        n = 5
        U = UpperTriangular(diagm(0 => 1.0:n) + 0.1 * UpperTriangular(randn(n, n)))

        # Compute bounds
        norm_inf = triangular_inverse_inf_norm_bound(U)
        norm_one = triangular_inverse_one_norm_bound(U)
        norm_two = triangular_inverse_two_norm_bound(U)

        # True values
        U_inv = inv(Matrix(U))
        true_inf = opnorm(U_inv, Inf)
        true_one = opnorm(U_inv, 1)
        true_two = opnorm(U_inv, 2)

        @test norm_inf ≥ true_inf * 0.999
        @test norm_one ≥ true_one * 0.999
        @test norm_two ≥ true_two * 0.999

        println("Triangular inverse bounds:")
        println("  ‖U⁻¹‖_∞: bound=$norm_inf, true=$true_inf, ratio=$(norm_inf/true_inf)")
        println("  ‖U⁻¹‖_1: bound=$norm_one, true=$true_one, ratio=$(norm_one/true_one)")
        println("  ‖U⁻¹‖_2: bound=$norm_two, true=$true_two, ratio=$(norm_two/true_two)")
    end

    @testset "Triangular inverse with complex matrix" begin
        n = 5
        U = UpperTriangular(diagm(0 => complex.(1.0:n, 0.1:0.1:0.5)) +
                            0.1 * UpperTriangular(randn(ComplexF64, n, n)))

        norm_two = triangular_inverse_two_norm_bound(U)
        true_two = opnorm(inv(Matrix(U)), 2)

        @test norm_two ≥ true_two * 0.999
        @test isfinite(norm_two)
    end

    @testset "Similarity condition number" begin
        # Test psi_squared function
        @test psi_squared(0.0) ≈ 1.0

        # For small μ, κ₂(S) should be close to 1
        # psi²(0.1) = 1 + 0.01/2 + 0.05*√4.01 ≈ 1.105
        @test psi_squared(0.1) < 1.2

        # For larger μ, it grows
        @test psi_squared(1.0) > 1.5
        @test psi_squared(2.0) > 3.0

        # Test with matrix
        X = 0.1 * randn(5, 3)
        K_S = similarity_condition_number(X)
        @test K_S ≥ 1.0
        @test isfinite(K_S)
    end

    @testset "Sylvester oracle" begin
        # Create a simple Schur matrix
        n = 10
        T = UpperTriangular(diagm(0 => complex.(1.0:n, 0.1:0.1:1.0)) +
                            0.1 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        k = 3
        T11 = T[1:k, 1:k]
        T12 = T[1:k, (k+1):n]
        T22 = T[(k+1):n, (k+1):n]

        # Solve Sylvester equation
        X = solve_sylvester_oracle(T11, T12, T22)

        # Check residual is small
        R = T12 + T11 * X - X * T22
        @test norm(R) < 1e-10 * norm(T12)

        println("Sylvester oracle residual: $(norm(R))")
    end

    @testset "V1 precomputation" begin
        n = 20
        T = UpperTriangular(diagm(0 => complex.(1.0:n, 0.5:0.5:10.0)) +
                            0.1 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        k = 5
        precomp = sylvester_resolvent_precompute(T, k)

        @test precomp.precomputation_success
        @test precomp.k == k
        @test precomp.n == n
        @test isfinite(precomp.residual_norm)
        @test isfinite(precomp.similarity_cond)
        @test precomp.similarity_cond ≥ 1.0

        println("\nPrecomputation results:")
        print_sylvester_diagnostics(precomp)
    end

    @testset "V1 resolvent bound" begin
        n = 15
        # Create Schur matrix with known eigenvalues
        λ = complex.(1.0:n, 0.5:0.5:7.5)
        T = UpperTriangular(diagm(0 => λ) + 0.1 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        k = 4
        z = 3.5 + 1.0im  # Away from eigenvalues

        precomp, result = sylvester_resolvent_bound(T, k, z)

        @test result.success
        @test isfinite(result.resolvent_bound)
        @test result.resolvent_bound > 0

        # Compare with true resolvent norm
        true_resolvent = opnorm(inv(z * I - T), 2)
        @test result.resolvent_bound ≥ true_resolvent * 0.99

        println("\nV1 resolvent bound at z=$z:")
        println("  Certified bound: $(result.resolvent_bound)")
        println("  True value: $true_resolvent")
        println("  Overestimation: $(result.resolvent_bound / true_resolvent)x")
        print_point_result(result)
    end

    @testset "V2 resolvent bound" begin
        n = 15
        λ = complex.(1.0:n, 0.5:0.5:7.5)
        T = UpperTriangular(diagm(0 => λ) + 0.1 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        k = 4
        z = 3.5 + 1.0im

        precomp, R, result = sylvester_resolvent_bound_v2(T, k, z)

        @test result.success
        @test isfinite(result.resolvent_bound)
        @test result.resolvent_bound > 0

        # V2 should be at least as tight as V1
        @test result.resolvent_bound ≤ result.resolvent_bound_v1 * 1.001

        # Compare with true resolvent norm
        true_resolvent = opnorm(inv(z * I - T), 2)
        @test result.resolvent_bound ≥ true_resolvent * 0.99

        println("\nV2 resolvent bound at z=$z:")
        print_point_result_v2(result)
        println("  True value: $true_resolvent")
    end

    @testset "V1 vs V2 comparison" begin
        n = 20
        λ = complex.(1.0:n, 0.2:0.2:4.0)
        T = UpperTriangular(diagm(0 => λ) + 0.05 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        k = 6
        z_list = [5.0 + 1.0im, 10.0 + 2.0im, 15.0 + 0.5im]

        println("\nV1 vs V2 comparison:")
        for z in z_list
            _, result_v1 = sylvester_resolvent_bound(T, k, z)
            _, _, result_v2 = sylvester_resolvent_bound_v2(T, k, z)

            true_resolvent = opnorm(inv(z * I - T), 2)

            @test result_v1.success
            @test result_v2.success

            improvement = (result_v1.resolvent_bound - result_v2.resolvent_bound) /
                          result_v1.resolvent_bound * 100
            @test result_v2.resolvent_bound ≤ result_v1.resolvent_bound * 1.001

            println("  z=$z:")
            println("    V1: $(result_v1.resolvent_bound) ($(result_v1.resolvent_bound/true_resolvent)x)")
            println("    V2: $(result_v2.resolvent_bound) ($(result_v2.resolvent_bound/true_resolvent)x)")
            println("    Improvement: $(round(improvement, digits=1))%")
            println("    Tightening ratio: $(result_v2.tightening_ratio)")
        end
    end

    @testset "Multiple points" begin
        n = 12
        λ = complex.(1.0:n, 0.3:0.3:3.6)
        T = UpperTriangular(diagm(0 => λ) + 0.1 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        k = 4
        z_list = [2.0 + 0.5im, 6.0 + 1.0im, 9.0 + 1.5im]

        # V1
        precomp, results_v1 = sylvester_resolvent_bound(T, k, z_list)

        @test length(results_v1) == length(z_list)
        @test all(r -> r.success, results_v1)

        # V2
        precomp, R, results_v2 = sylvester_resolvent_bound_v2(T, k, z_list)

        @test length(results_v2) == length(z_list)
        @test all(r -> r.success, results_v2)
    end

    @testset "Failure modes" begin
        # Create a matrix where we know exact eigenvalues
        n = 10
        # Use pure diagonal matrix so eigenvalues are exact
        λ = complex.(1.0:n, 0.0)
        T = diagm(0 => λ)  # Diagonal matrix - eigenvalues are exactly λ

        k = 3

        # z exactly at an eigenvalue of T11 should fail (σ_min = 0)
        z_at_eigenvalue = λ[1]  # First eigenvalue is in T11
        _, result = sylvester_resolvent_bound(T, k, z_at_eigenvalue)

        # Should fail (σ_min = 0 means matrix is singular)
        @test !result.success
        println("z at T11 eigenvalue: success=$(result.success), reason=$(result.failure_reason)")

        # z exactly at an eigenvalue of T22 should fail (triangular inverse bound blows up)
        z_at_eigenvalue_T22 = λ[k+1]  # First eigenvalue of T22
        _, result2 = sylvester_resolvent_bound(T, k, z_at_eigenvalue_T22)

        # Should fail or give infinite bound
        @test !result2.success || !isfinite(result2.resolvent_bound)
        println("z at T22 eigenvalue: success=$(result2.success), bound=$(result2.resolvent_bound)")
    end

    @testset "Optimal split selection" begin
        n = 20
        λ = complex.(1.0:n, 0.2:0.2:4.0)
        T = UpperTriangular(diagm(0 => λ) + 0.05 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        z = 10.0 + 1.5im

        # V1
        best_k_v1, best_precomp_v1, best_result_v1 = find_optimal_split(T, z; k_range=2:10, version=:V1)

        @test best_result_v1.success
        @test 2 ≤ best_k_v1 ≤ 10

        # V2
        best_k_v2, best_precomp_v2, best_R_v2, best_result_v2 = find_optimal_split(T, z; k_range=2:10, version=:V2)

        @test best_result_v2.success
        @test 2 ≤ best_k_v2 ≤ 10

        println("\nOptimal split selection:")
        println("  V1: best_k=$best_k_v1, bound=$(best_result_v1.resolvent_bound)")
        println("  V2: best_k=$best_k_v2, bound=$(best_result_v2.resolvent_bound)")
    end

    @testset "V3 Collatz-Neumann bound" begin
        n = 15
        λ = complex.(1.0:n, 0.5:0.5:7.5)
        T = UpperTriangular(diagm(0 => λ) + 0.1 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        k = 4
        z = 3.5 + 1.0im

        precomp, R, result = sylvester_resolvent_bound_v3(T, k, z)

        @test result.success
        @test isfinite(result.resolvent_bound)
        @test result.resolvent_bound > 0

        # Note: V3 Neumann bound is not always tighter than V1 triangular bound
        # It depends on the matrix structure. Both are valid upper bounds.

        # Compare with true resolvent norm
        true_resolvent = opnorm(inv(z * I - T), 2)
        @test result.resolvent_bound ≥ true_resolvent * 0.99

        println("\nV3 resolvent bound at z=$z:")
        print_point_result_v3(result)
        println("  True value: $true_resolvent")

        # Check if Neumann was certified
        if result.neumann_success
            println("  Neumann certified with α=$(result.alpha)")
        else
            println("  Neumann failed, using triangular fallback")
        end
    end

    @testset "V3 Collatz-Neumann internals" begin
        # Test Collatz bound directly with diagonally dominant matrix
        # (small off-diagonal ensures Neumann always succeeds here)
        n = 10
        T22 = UpperTriangular(diagm(0 => complex.(5.0:14.0, 0.5:0.5:5.0)) +
                              0.1 * UpperTriangular(randn(ComplexF64, n, n)))
        T22 = Matrix(T22)

        z = 9.0 + 2.0im

        alpha, Dd_inv_norm = collatz_norm_N_bound(T22, z; power_iterations=5)

        @test isfinite(alpha)
        @test isfinite(Dd_inv_norm)
        @test alpha ≥ 0
        @test Dd_inv_norm ≥ 0

        # Test Neumann bound — should succeed for diagonally dominant T22
        neumann_result = neumann_inverse_bound(T22, z; power_iterations=5)

        println("\nCollatz-Neumann internals:")
        println("  α = ‖N_z‖₂ ≤ $alpha")
        println("  ‖(D_z)_d⁻¹‖₂ = $Dd_inv_norm")
        println("  Neumann success: $(neumann_result.success)")

        @test neumann_result.success

        # Compare with true inverse norm
        D_z = z * I - T22
        true_D_inv = opnorm(inv(D_z), 2)
        println("  M_D = $(neumann_result.M_D)")
        println("  gap = $(neumann_result.neumann_gap)")
        println("  True ‖D_z⁻¹‖₂ = $true_D_inv")
        @test neumann_result.M_D ≥ true_D_inv * 0.99
    end

    @testset "V1 vs V2 vs V3 comparison" begin
        n = 20
        λ = complex.(1.0:n, 0.2:0.2:4.0)
        T = UpperTriangular(diagm(0 => λ) + 0.05 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        k = 6
        z = 10.0 + 2.0im

        _, result_v1 = sylvester_resolvent_bound(T, k, z)
        _, _, result_v2 = sylvester_resolvent_bound_v2(T, k, z)
        _, _, result_v3 = sylvester_resolvent_bound_v3(T, k, z)

        true_resolvent = opnorm(inv(z * I - T), 2)

        @test result_v1.success
        @test result_v2.success
        @test result_v3.success

        println("\nV1 vs V2 vs V3 comparison at z=$z:")
        println("  True: $true_resolvent")
        println("  V1: $(result_v1.resolvent_bound) ($(result_v1.resolvent_bound/true_resolvent)x)")
        println("  V2: $(result_v2.resolvent_bound) ($(result_v2.resolvent_bound/true_resolvent)x)")
        println("  V3: $(result_v3.resolvent_bound) ($(result_v3.resolvent_bound/true_resolvent)x)")
        println("  V3 M_D: $(result_v3.M_D) vs V1 M_D: $(result_v3.M_D_v1)")
        println("  V3 α: $(result_v3.alpha), gap: $(result_v3.neumann_gap)")

        # All should be valid upper bounds
        @test result_v1.resolvent_bound ≥ true_resolvent * 0.99
        @test result_v2.resolvent_bound ≥ true_resolvent * 0.99
        @test result_v3.resolvent_bound ≥ true_resolvent * 0.99
    end

    @testset "V3 Neumann failure mode" begin
        # Deterministic test: large off-diagonal entries cause α ≥ 1
        # (Neumann series diverges), so V3 must fall back to the triangular bound.
        n = 8
        λ = complex.(1.0:n, 0.0)
        T22 = diagm(0 => λ) + 5.0 * UpperTriangular(ones(n, n) - I)
        T22 = Matrix(T22)

        z = 4.5 + 0.0im  # Between eigenvalues, but off-diagonal dominates

        neumann_result = neumann_inverse_bound(T22, z; power_iterations=5)

        println("\nNeumann failure test:")
        println("  α = $(neumann_result.alpha)")
        println("  success = $(neumann_result.success)")

        @test neumann_result.alpha ≥ 1
        @test !neumann_result.success

        # Build a full Schur matrix with T11 eigenvalues well-separated
        # from T22's (eigenvalue separation ≈ 92), avoiding the singular
        # Sylvester equation that caused sporadic LAPACK failures.
        k = 2
        T11 = diagm(complex.([100.0, 200.0], 0.0))
        T12 = 0.1 * ones(ComplexF64, k, n)
        T_full = zeros(ComplexF64, k + n, k + n)
        T_full[1:k, 1:k] .= T11
        T_full[1:k, (k+1):end] .= T12
        T_full[(k+1):end, (k+1):end] .= T22

        precomp, R, result = sylvester_resolvent_bound_v3(T_full, k, z)

        @test result.success
        @test !result.neumann_success   # Neumann failed, used triangular fallback
        @test isfinite(result.resolvent_bound)

        true_resolvent = opnorm(inv(z * I - T_full), 2)
        @test result.resolvent_bound ≥ true_resolvent * 0.99

        println("  V3 overall success: $(result.success)")
        println("  Neumann certified: $(result.neumann_success)")
        println("  Bound: $(result.resolvent_bound), true: $true_resolvent")
    end

    @testset "BigFloat support" begin
        n = 8
        setprecision(256) do
            λ = Complex{BigFloat}.(1.0:n, 0.2:0.2:1.6)
            T_bf = Matrix(UpperTriangular(diagm(0 => λ) +
                          BigFloat(0.1) * UpperTriangular(randn(Complex{BigFloat}, n, n))))

            k = 3
            z = Complex{BigFloat}(3.5, 1.0)

            # Should work with BigFloat
            precomp = sylvester_resolvent_precompute(T_bf, k)

            @test precomp.precomputation_success
            @test typeof(precomp.residual_norm) == BigFloat
        end
    end

    # =====================================================
    # Extended Parametric Framework Tests
    # =====================================================

    @testset "Norm estimators" begin
        M = randn(ComplexF64, 10, 10)
        true_norm = opnorm(M, 2)

        # All estimators should give upper bounds
        norm_oneinf = estimate_2norm(M, OneInfNorm)
        norm_frob = estimate_2norm(M, FrobeniusNorm)
        norm_rowcol = estimate_2norm(M, RowCol2Norm)

        @test norm_oneinf ≥ true_norm * 0.99  # Upper bound
        @test norm_frob ≥ true_norm * 0.99
        @test norm_rowcol ≥ 0.0  # Non-negative

        # Frobenius is always ≥ spectral norm
        @test norm_frob ≥ true_norm * 0.99

        println("\nNorm estimators test:")
        println("  True ‖M‖₂:    $true_norm")
        println("  OneInfNorm:   $norm_oneinf ($(norm_oneinf/true_norm)x)")
        println("  FrobeniusNorm: $norm_frob ($(norm_frob/true_norm)x)")
        println("  RowCol2Norm:  $norm_rowcol")
    end

    @testset "Neumann 1/∞ bound" begin
        n = 10
        # Diagonal dominant matrix - Neumann should succeed
        T22 = diagm(0 => complex.(5.0:14.0, 0.5:0.5:5.0)) +
              0.1 * UpperTriangular(randn(ComplexF64, n, n))
        T22 = Matrix(T22)

        z = 9.0 + 2.0im

        result = neumann_one_inf_bound(T22, z)

        @test isfinite(result.alpha_inf)
        @test isfinite(result.alpha_one)
        @test result.alpha_inf ≥ 0
        @test result.alpha_one ≥ 0
        @test result.success

        # Should be valid upper bound
        D_z = z * I - T22
        true_D_inv = opnorm(inv(D_z), 2)
        @test result.M_D ≥ true_D_inv * 0.99

        println("\nNeumann 1/∞ bound test:")
        println("  α∞ = $(result.alpha_inf)")
        println("  α₁ = $(result.alpha_one)")
        println("  Success: $(result.success)")
        println("  M_D = $(result.M_D) (true: $true_D_inv)")
    end

    @testset "Config presets" begin
        # Test that all config presets are valid
        cfg_v1 = config_v1()
        cfg_v2 = config_v2()
        cfg_v2p5 = config_v2p5()
        cfg_v3 = config_v3()

        @test cfg_v1.d_inv_estimator == TriBacksub
        @test cfg_v1.coupling_estimator == CouplingNone
        @test cfg_v1.combiner == CombinerV1

        @test cfg_v2.coupling_estimator == CouplingARSolve
        @test cfg_v2.combiner == CombinerV2

        @test cfg_v2p5.coupling_estimator == CouplingOffDirect
        @test cfg_v2p5.combiner == CombinerV2p5

        @test cfg_v3.d_inv_estimator == NeumannCollatz2
    end

    @testset "Parametric resolvent bound - all configs" begin
        n = 20
        λ = complex.(1.0:n, 0.2:0.2:4.0)
        T = UpperTriangular(diagm(0 => λ) + 0.05 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        k = 6
        z = 10.0 + 2.0im

        true_resolvent = opnorm(inv(z * I - T), 2)

        # Test all configs
        precomp, R, result_v1 = parametric_resolvent_bound(T, k, z, config_v1())
        result_v2 = parametric_resolvent_bound(precomp, T, z, config_v2(); R=R)
        result_v2p5 = parametric_resolvent_bound(precomp, T, z, config_v2p5(); R=R)
        result_v3 = parametric_resolvent_bound(precomp, T, z, config_v3(); R=R)

        println("\nParametric framework comparison at z=$z:")
        println("  True: $true_resolvent")

        @test result_v1.success
        @test result_v1.resolvent_bound ≥ true_resolvent * 0.99
        println("  V1: $(result_v1.resolvent_bound) ($(result_v1.resolvent_bound/true_resolvent)x)")

        @test result_v2.success
        @test result_v2.resolvent_bound ≥ true_resolvent * 0.99
        println("  V2: $(result_v2.resolvent_bound) ($(result_v2.resolvent_bound/true_resolvent)x)")

        @test result_v2p5.success
        @test result_v2p5.resolvent_bound ≥ true_resolvent * 0.99
        println("  V2.5: $(result_v2p5.resolvent_bound) ($(result_v2p5.resolvent_bound/true_resolvent)x)")

        @test result_v3.success
        @test result_v3.resolvent_bound ≥ true_resolvent * 0.99
        println("  V3: $(result_v3.resolvent_bound) ($(result_v3.resolvent_bound/true_resolvent)x)")
    end

    @testset "compare_all_configs" begin
        n = 15
        λ = complex.(1.0:n, 0.2:0.2:3.0)
        T = UpperTriangular(diagm(0 => λ) + 0.05 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        k = 5
        z = 8.0 + 1.5im

        cmp = compare_all_configs(T, k, z)

        @test haskey(cmp.bounds, "V1")
        @test haskey(cmp.bounds, "V2")
        @test haskey(cmp.bounds, "V2.5")
        @test haskey(cmp.bounds, "V3")

        @test cmp.best ∈ ["V1", "V2", "V2.5", "V3"]

        # All results should be valid
        for (name, result) in cmp.results
            @test result.success
        end

        println("\ncompare_all_configs test:")
        println("  Best method: $(cmp.best)")
        for (name, bound) in sort(collect(cmp.bounds), by=x->x[2])
            println("  $name: $bound")
        end
    end

    @testset "SVDWarmStart" begin
        n = 20
        λ = complex.(1.0:n, 0.2:0.2:4.0)
        T = UpperTriangular(diagm(0 => λ) + 0.05 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        k = 5
        z1 = 10.0 + 2.0im
        z2 = 10.05 + 2.02im  # Close to z1

        precomp, R, result1 = parametric_resolvent_bound(T, k, z1, config_v2())

        # Create warm start from z1
        T11 = T[1:k, 1:k]
        A_z1 = z1 * I - T11
        svd_Az1 = svd(A_z1)
        warm_start = SVDWarmStart(svd_Az1.U, svd_Az1.S, svd_Az1.V)

        # Compute at z2 with and without warm start
        result2_cold = parametric_resolvent_bound(precomp, T, z2, config_v2(); R=R)
        result2_warm = parametric_resolvent_bound(precomp, T, z2, config_v2(); R=R, svd_warm_start=warm_start)

        true_resolvent = opnorm(inv(z2 * I - T), 2)

        @test result2_cold.success
        @test result2_warm.success

        # Both should be valid upper bounds
        @test result2_cold.resolvent_bound ≥ true_resolvent * 0.99
        @test result2_warm.resolvent_bound ≥ true_resolvent * 0.99

        # Should give similar results (both valid)
        rel_diff = abs(result2_cold.resolvent_bound - result2_warm.resolvent_bound) / result2_cold.resolvent_bound
        @test rel_diff < 0.01  # Less than 1% difference

        println("\nSVDWarmStart test:")
        println("  z1 = $z1, z2 = $z2")
        println("  Cold start: $(result2_cold.resolvent_bound)")
        println("  Warm start: $(result2_warm.resolvent_bound)")
        println("  Relative difference: $rel_diff")
    end

    # ==========================================================
    # Dedicated solve_sylvester_oracle tests
    # ==========================================================

    @testset "solve_sylvester_oracle — real Float64" begin
        n = 10
        T = UpperTriangular(diagm(0 => collect(1.0:n)) +
                            0.1 * UpperTriangular(randn(n, n)))
        T = Matrix(T)

        k = 3
        T11 = T[1:k, 1:k]; T12 = T[1:k, (k+1):n]; T22 = T[(k+1):n, (k+1):n]

        X = solve_sylvester_oracle(T11, T12, T22)

        # Verify: T11*X - X*T22 = -T12
        R = T12 + T11 * X - X * T22
        @test norm(R) < 1e-10 * norm(T12)
        @test eltype(X) <: Real
    end

    @testset "solve_sylvester_oracle — BigFloat complex (downcast)" begin
        setprecision(256) do
            n = 8
            T_bf = Matrix(UpperTriangular(
                diagm(0 => Complex{BigFloat}.(1:n, BigFloat(0.2):BigFloat(0.2):BigFloat(1.6))) +
                BigFloat(0.1) * UpperTriangular(randn(Complex{BigFloat}, n, n))))

            k = 3
            T11 = T_bf[1:k, 1:k]; T12 = T_bf[1:k, (k+1):n]; T22 = T_bf[(k+1):n, (k+1):n]

            X = solve_sylvester_oracle(T11, T12, T22)

            R = T12 + T11 * X - X * T22
            @test norm(R) < BigFloat(1e-10) * norm(T12)
            @test eltype(X) <: Complex{BigFloat}
        end
    end

    @testset "solve_sylvester_oracle — BigFloat real (downcast)" begin
        setprecision(256) do
            n = 8
            T_bf = Matrix(UpperTriangular(
                diagm(0 => BigFloat.(1:n)) +
                BigFloat(0.1) * UpperTriangular(randn(n, n) .|> BigFloat)))

            k = 3
            T11 = T_bf[1:k, 1:k]; T12 = T_bf[1:k, (k+1):n]; T22 = T_bf[(k+1):n, (k+1):n]

            X = solve_sylvester_oracle(T11, T12, T22)

            R = T12 + T11 * X - X * T22
            @test norm(R) < BigFloat(1e-10) * norm(T12)
            @test eltype(X) <: BigFloat
        end
    end

    @testset "solve_sylvester_oracle — k = 1 scalar block" begin
        n = 6
        T = UpperTriangular(diagm(0 => complex.(1.0:n, 0.1:0.1:0.6)) +
                            0.1 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        T11 = T[1:1, 1:1]; T12 = T[1:1, 2:n]; T22 = T[2:n, 2:n]
        X = solve_sylvester_oracle(T11, T12, T22)

        R = T12 + T11 * X - X * T22
        @test norm(R) < 1e-12 * norm(T12)
        @test size(X) == (1, n - 1)
    end

    @testset "solve_sylvester_oracle — k = n-1 scalar complement" begin
        n = 6
        T = UpperTriangular(diagm(0 => complex.(1.0:n, 0.1:0.1:0.6)) +
                            0.1 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        k = n - 1
        T11 = T[1:k, 1:k]; T12 = T[1:k, k+1:n]; T22 = T[k+1:n, k+1:n]
        X = solve_sylvester_oracle(T11, T12, T22)

        R = T12 + T11 * X - X * T22
        @test norm(R) < 1e-12 * norm(T12)
        @test size(X) == (k, 1)
    end

    # ==========================================================
    # Dedicated sylvester_resolvent_bound_v3 tests
    # ==========================================================

    @testset "sylvester_resolvent_bound_v3 — precomp + T + R + z" begin
        n = 15
        λ = complex.(1.0:n, 0.5:0.5:7.5)
        T = Matrix(UpperTriangular(diagm(0 => λ) +
                   0.1 * UpperTriangular(randn(ComplexF64, n, n))))

        k = 4
        z = 3.5 + 1.0im

        # Manually precompute and pass R
        precomp = sylvester_resolvent_precompute(T, k)
        T11 = T[1:k, 1:k]; T12 = T[1:k, (k+1):n]; T22 = T[(k+1):n, (k+1):n]
        X = solve_sylvester_oracle(T11, T12, T22)
        R = T12 + T11 * X - X * T22

        result = sylvester_resolvent_bound_v3(precomp, T, R, z)

        @test result.success
        @test isfinite(result.resolvent_bound)

        true_resolvent = opnorm(inv(z * I - T), 2)
        @test result.resolvent_bound >= true_resolvent * 0.99
    end

    @testset "sylvester_resolvent_bound_v3 — multi-point (vector of z)" begin
        n = 12
        λ = complex.(1.0:n, 0.3:0.3:3.6)
        T = Matrix(UpperTriangular(diagm(0 => λ) +
                   0.1 * UpperTriangular(randn(ComplexF64, n, n))))
        k = 4
        z_list = [2.0 + 0.5im, 6.0 + 1.0im, 9.0 + 1.5im]

        # Convenience multi-point
        precomp, R, results = sylvester_resolvent_bound_v3(T, k, z_list)

        @test length(results) == 3
        for (i, result) in enumerate(results)
            @test result.success
            @test isfinite(result.resolvent_bound)
            true_res = opnorm(inv(z_list[i] * I - T), 2)
            @test result.resolvent_bound >= true_res * 0.99
        end

        # Also test precomp + R + z_list path
        results2 = sylvester_resolvent_bound_v3(precomp, T, R, z_list)
        @test length(results2) == 3
        for (r1, r2) in zip(results, results2)
            @test r1.resolvent_bound ≈ r2.resolvent_bound rtol = 1e-10
        end
    end

    @testset "sylvester_resolvent_bound_v3 — use_v2_coupling=false" begin
        n = 15
        λ = complex.(1.0:n, 0.5:0.5:7.5)
        T = Matrix(UpperTriangular(diagm(0 => λ) +
                   0.1 * UpperTriangular(randn(ComplexF64, n, n))))
        k = 4
        z = 3.5 + 1.0im

        precomp, R, result_v2on = sylvester_resolvent_bound_v3(T, k, z;
                                       use_v2_coupling=true)
        result_v2off = sylvester_resolvent_bound_v3(precomp, T, R, z;
                                       use_v2_coupling=false)

        @test result_v2on.success
        @test result_v2off.success

        true_resolvent = opnorm(inv(z * I - T), 2)
        @test result_v2on.resolvent_bound  >= true_resolvent * 0.99
        @test result_v2off.resolvent_bound >= true_resolvent * 0.99

        # V2 coupling should be at least as tight as the product bound fallback
        @test result_v2on.resolvent_bound <= result_v2off.resolvent_bound * 1.001
    end

    @testset "sylvester_resolvent_bound_v3 — real matrix" begin
        n = 15
        λ = collect(1.0:n)
        T = Matrix(UpperTriangular(diagm(0 => λ) +
                   0.05 * UpperTriangular(randn(n, n))))
        k = 4
        z = 3.5 + 1.0im  # z must be complex

        precomp, R, result = sylvester_resolvent_bound_v3(T, k, z)

        @test result.success
        @test isfinite(result.resolvent_bound)

        true_resolvent = opnorm(inv(z * I - T), 2)
        @test result.resolvent_bound >= true_resolvent * 0.99
    end

    @testset "sylvester_resolvent_bound_v3 — X_oracle kwarg" begin
        n = 12
        λ = complex.(1.0:n, 0.3:0.3:3.6)
        T = Matrix(UpperTriangular(diagm(0 => λ) +
                   0.1 * UpperTriangular(randn(ComplexF64, n, n))))
        k = 4
        z = 6.0 + 1.0im

        T11 = T[1:k, 1:k]; T12 = T[1:k, (k+1):n]; T22 = T[(k+1):n, (k+1):n]
        X = solve_sylvester_oracle(T11, T12, T22)

        # With and without oracle should give same result
        _, _, result_auto   = sylvester_resolvent_bound_v3(T, k, z)
        _, _, result_oracle = sylvester_resolvent_bound_v3(T, k, z; X_oracle=X)

        @test result_auto.success
        @test result_oracle.success
        @test result_auto.resolvent_bound ≈ result_oracle.resolvent_bound rtol = 1e-10
    end

    @testset "sylvester_resolvent_bound_v3 — V3 vs V1 comparison" begin
        # V3 and V1 are both valid upper bounds; V3 may or may not be tighter
        # depending on whether Neumann improves M_D vs triangular backsubstitution.
        n = 15
        λ = complex.(1.0:n, 0.5:0.5:7.5)
        T = Matrix(UpperTriangular(diagm(0 => λ) +
                   0.1 * UpperTriangular(randn(ComplexF64, n, n))))
        k = 4
        z = 8.0 + 2.0im

        precomp, R, result = sylvester_resolvent_bound_v3(T, k, z)

        @test result.success

        true_resolvent = opnorm(inv(z * I - T), 2)
        @test result.resolvent_bound >= true_resolvent * 0.99
        @test result.resolvent_bound_v1 >= true_resolvent * 0.99

        if result.neumann_success
            # When Neumann succeeds, M_D should be finite
            @test isfinite(result.M_D)
            @test result.M_D > 0
        end
    end

    @testset "sylvester_resolvent_bound_v3 — miyajima_method :M4" begin
        n = 12
        λ = complex.(1.0:n, 0.3:0.3:3.6)
        T = Matrix(UpperTriangular(diagm(0 => λ) +
                   0.1 * UpperTriangular(randn(ComplexF64, n, n))))
        k = 4
        z = 6.0 + 1.0im

        precomp, R, result_m1 = sylvester_resolvent_bound_v3(T, k, z;
                                     miyajima_method=:M1)
        result_m4 = sylvester_resolvent_bound_v3(precomp, T, R, z;
                                     miyajima_method=:M4)

        @test result_m1.success
        @test result_m4.success

        true_resolvent = opnorm(inv(z * I - T), 2)
        @test result_m1.resolvent_bound >= true_resolvent * 0.99
        @test result_m4.resolvent_bound >= true_resolvent * 0.99
    end

    @testset "Off-diagonal direct bound" begin
        n = 10
        k = 4
        m = n - k

        λ = complex.(1.0:n, 0.2:0.2:2.0)
        T = UpperTriangular(diagm(0 => λ) + 0.05 * UpperTriangular(randn(ComplexF64, n, n)))
        T = Matrix(T)

        T11 = T[1:k, 1:k]
        T12 = T[1:k, (k+1):n]
        T22 = T[(k+1):n, (k+1):n]

        z = 5.0 + 1.0im
        A_z = z * I - T11
        D_z = z * I - T22

        # Compute precomp to get R
        X = solve_sylvester_oracle(T11, T12, T22)
        R = T12 + T11 * X - X * T22

        # Get M_A and M_D from rigorous SVD
        A_z_ball = BallMatrix(A_z, zeros(Float64, k, k))
        svd_result = rigorous_svd(A_z_ball)
        σ_min = mid(svd_result.singular_values[end]) - rad(svd_result.singular_values[end])
        M_A = 1.0 / σ_min
        M_D = triangular_inverse_two_norm_bound(D_z)

        # Test off-diagonal bound
        result = offdiag_direct_bound(A_z, D_z, R, M_A, M_D)

        @test result.success
        @test isfinite(result.M_off)
        @test result.M_off ≥ 0

        # Compare with product bound
        r = sqrt(opnorm(R, 1) * opnorm(R, Inf))
        product_bound = M_A * r * M_D

        println("\nOff-diagonal direct bound test:")
        println("  Product bound (M_A·r·M_D): $product_bound")
        println("  Direct bound (M_off):       $(result.M_off)")
        println("  Tightening ratio:           $(result.M_off / product_bound)")

        # Direct should be ≤ product (V2.5 is tighter or equal)
        @test result.M_off ≤ product_bound * 1.001  # Allow small numerical tolerance
    end

    # ==========================================================
    # Residual-based Sylvester fallback tests
    # ==========================================================

    @testset "Sylvester residual fallback — small well-conditioned" begin
        n = 8
        T = Matrix(UpperTriangular(diagm(0 => complex.(1.0:n, 0.5:0.5:4.0)) +
                    0.1 * UpperTriangular(randn(ComplexF64, n, n))))

        k = 3
        result_direct = triangular_sylvester_miyajima_enclosure(T, k;
                            sylvester_fallback=:direct)
        result_residual = triangular_sylvester_miyajima_enclosure(T, k;
                            sylvester_fallback=:residual)

        # Both should produce finite enclosures
        @test all(isfinite, mid(result_direct))
        @test all(isfinite, rad(result_direct))
        @test all(isfinite, mid(result_residual))
        @test all(isfinite, rad(result_residual))

        # Midpoints should match (same approximate solver)
        @test mid(result_direct) ≈ mid(result_residual) atol=1e-10

        println("\nResidual fallback — small well-conditioned:")
        println("  Direct max radius:   $(maximum(rad(result_direct)))")
        println("  Residual max radius: $(maximum(rad(result_residual)))")
    end

    @testset "Sylvester residual fallback — k > 1 coupling" begin
        n = 10
        T = Matrix(UpperTriangular(diagm(0 => complex.(1.0:n, 0.2:0.2:2.0)) +
                    0.05 * UpperTriangular(randn(ComplexF64, n, n))))

        k = 4
        result_direct = triangular_sylvester_miyajima_enclosure(T, k;
                            sylvester_fallback=:direct)
        result_residual = triangular_sylvester_miyajima_enclosure(T, k;
                            sylvester_fallback=:residual)

        @test all(isfinite, rad(result_residual))
        @test mid(result_direct) ≈ mid(result_residual) atol=1e-10

        println("\nResidual fallback — k=$k coupling:")
        println("  Direct max radius:   $(maximum(rad(result_direct)))")
        println("  Residual max radius: $(maximum(rad(result_residual)))")
    end

    @testset "Sylvester residual fallback — complex matrices" begin
        n = 6
        λ = complex.(1.0:n, -3.0:1.0:2.0)
        T = Matrix(UpperTriangular(diagm(0 => λ) +
                    0.2 * UpperTriangular(randn(ComplexF64, n, n))))

        k = 2
        result = triangular_sylvester_miyajima_enclosure(T, k;
                    sylvester_fallback=:residual)

        @test all(isfinite, mid(result))
        @test all(isfinite, rad(result))
        @test size(result) == (n - k, k)
    end

    @testset "Sylvester residual fallback — BallMatrix overload threads kwarg" begin
        n = 8
        T_mid = Matrix(UpperTriangular(diagm(0 => complex.(1.0:n, 0.5:0.5:4.0)) +
                    0.1 * UpperTriangular(randn(ComplexF64, n, n))))
        T_ball = BallMatrix(T_mid, fill(1e-12, n, n))

        k = 3
        result_direct = triangular_sylvester_miyajima_enclosure(T_ball, k;
                            sylvester_fallback=:direct)
        result_residual = triangular_sylvester_miyajima_enclosure(T_ball, k;
                            sylvester_fallback=:residual)

        @test all(isfinite, mid(result_direct))
        @test all(isfinite, mid(result_residual))
        @test all(isfinite, rad(result_direct))
        @test all(isfinite, rad(result_residual))
    end

    @testset "Sylvester residual fallback — compute_spectral_coefficient" begin
        n = 8
        A_mid = Matrix(UpperTriangular(diagm(0 => complex.(1.0:n, 0.5:0.5:4.0)) +
                    0.1 * UpperTriangular(randn(ComplexF64, n, n))))
        A = BallMatrix(A_mid)
        v = randn(ComplexF64, n)

        result_direct = compute_spectral_coefficient(A, v, 1:3;
                            sylvester_fallback=:direct)
        result_residual = compute_spectral_coefficient(A, v, 1:3;
                            sylvester_fallback=:residual)

        # Coefficients should have similar midpoints
        @test mid(result_direct.coefficients) ≈ mid(result_residual.coefficients) atol=1e-8
        @test all(isfinite, rad(result_residual.coefficients))
    end

    @testset "Sylvester residual fallback — invalid symbol" begin
        n = 6
        T = Matrix(UpperTriangular(diagm(0 => complex.(1.0:n, 0.5:0.5:3.0)) +
                    0.1 * UpperTriangular(randn(ComplexF64, n, n))))

        @test_throws ArgumentError triangular_sylvester_miyajima_enclosure(T, 2;
                                        sylvester_fallback=:invalid)
    end

end
