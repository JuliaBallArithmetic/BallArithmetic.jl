using Test
using LinearAlgebra
using Random
using BallArithmetic

@testset "Sylvester Resolvent Bound" begin
    Random.seed!(20261010)
    truth(T, z) = 1 / minimum(svdvals(ComplexF64.(z * I - T)))
    tol = 1 - 1e-10

    @testset "Triangular inverse bounds" begin
        for CT in (Float64, ComplexF64), n in (1, 2, 7)
            U = Matrix(UpperTriangular(randn(CT, n, n))) + 3I
            Ui = inv(U)
            @test triangular_inverse_inf_norm_bound(U) ≥ opnorm(Ui, Inf) * tol
            @test triangular_inverse_one_norm_bound(U) ≥ opnorm(Ui, 1) * tol
            @test triangular_inverse_two_norm_bound(U) ≥ opnorm(Ui, 2) * tol
        end
        # a diagonal matrix: the three bounds are 1/min|u_ii| up to rounding
        D = Matrix(Diagonal([2.0, -4.0, 0.5]))
        @test 2.0 ≤ triangular_inverse_two_norm_bound(D) ≤ 2.0 * (1 + 1e-14)
        @test triangular_inverse_two_norm_bound([1.0 2.0; 0.0 0.0]) == Inf
        @test_throws ArgumentError triangular_inverse_inf_norm_bound([1.0 0.0; 1e-300 1.0])
        @test_throws DimensionMismatch triangular_inverse_inf_norm_bound(ones(2, 3))
        # BigFloat
        Ub = BigFloat.([2 1 -1; 0 3 1; 0 0 1]) ./ 3
        @test triangular_inverse_two_norm_bound(Ub) ≥ opnorm(Float64.(inv(Ub)), 2) * tol
    end

    @testset "The similarity S(X)" begin
        @test psi_squared(0.0) == 1.0
        for (k, m) in ((1, 1), (2, 5), (4, 3))
            X = randn(ComplexF64, k, m)
            S = [Matrix{ComplexF64}(I, k, k) X; zeros(ComplexF64, m, k) Matrix{ComplexF64}(I, m, m)]
            κ = cond(S)
            @test psi_squared(opnorm(X) * (1 + 1e-12)) ≥ κ * tol
            @test psi_squared(opnorm(X) * (1 + 1e-12)) ≤ κ * (1 + 1e-9)
            @test similarity_condition_number(X) ≥ κ * tol
            @test similarity_condition_number(BallMatrix(X)) == similarity_condition_number(X)
        end
        # increasing, and rounded up: ψ(1)² = (3 + √5)/2
        @test psi_squared(1.0) ≥ (3 + sqrt(big(5))) / 2
        @test psi_squared(1.0) ≤ Float64((3 + sqrt(big(5))) / 2) * (1 + 1e-15)
        @test psi_squared(big(1.0)) ≥ (3 + sqrt(big(5))) / 2
    end

    @testset "Sylvester oracle" begin
        for CT in (Float64, ComplexF64, BigFloat, Complex{BigFloat}), (n, k) in ((6, 2), (5, 1), (5, 4))
            T = Matrix(UpperTriangular(CT.(0.2 * randn(real(CT) === BigFloat ? Float64 : real(CT), n, n)))) +
                Diagonal(CT.(1:n))
            T11, T12, T22 = T[1:k, 1:k], T[1:k, (k + 1):n], T[(k + 1):n, (k + 1):n]
            X = solve_sylvester_oracle(T11, T12, T22)
            @test eltype(X) == CT
            @test size(X) == (k, n - k)
            @test opnorm(Float64.(abs.(T11 * X - X * T22 + T12)), 1) ≤ 1e-10
        end
    end

    @testset "Precomputation" begin
        n, k = 8, 3
        T = Matrix(UpperTriangular(0.3 * randn(ComplexF64, n, n))) + Diagonal(ComplexF64.(1:n))
        p = sylvester_resolvent_precompute(T, k)
        @test p.precomputation_success && p.k == k && p.n == n
        @test p.residual_norm < 1e-12
        @test p.similarity_cond ≥ 1
        # R encloses T12 + T11 X − X T22 for the X stored, evaluated at 512 bits
        Rexact = setprecision(BigFloat, 512) do
            Tb, Xb = Complex{BigFloat}.(T), Complex{BigFloat}.(p.X)
            Tb[1:k, (k + 1):n] + Tb[1:k, 1:k] * Xb - Xb * Tb[(k + 1):n, (k + 1):n]
        end
        @test all(abs.(Rexact - p.R.c) .≤ p.R.r)
        @test p.residual_norm ≥ opnorm(ComplexF64.(Rexact)) * tol
        @test p.coupling_norm ≥ opnorm(T[1:k, (k + 1):n]) * tol
        @test p.similarity_cond ≥ psi_squared(opnorm(p.X)) * tol

        # any X is admissible: with X = 0 the residual is T12 and the similarity is the identity
        p0 = sylvester_resolvent_precompute(T, k; X_oracle = zeros(k, n - k))
        @test p0.precomputation_success && p0.similarity_cond == 1
        @test p0.residual_norm ≥ opnorm(T[1:k, (k + 1):n]) * tol

        # structure that the identity needs
        Tbad = copy(T); Tbad[n, 1] = 1e-300
        @test !sylvester_resolvent_precompute(Tbad, k).precomputation_success
        Tbad = copy(T); Tbad[n, n - 1] = 1e-300
        pbad = sylvester_resolvent_precompute(Tbad, k)
        @test !pbad.precomputation_success && occursin("upper triangular", pbad.failure_reason)
        @test !parametric_resolvent_bound(pbad, Tbad, 0.5 + 0im).success
        @test !sylvester_resolvent_precompute(T, k; X_oracle = fill(NaN, k, n - k)).precomputation_success
        @test_throws ArgumentError sylvester_resolvent_precompute(T, 0)
        @test_throws ArgumentError sylvester_resolvent_precompute(T, n)
        @test_throws ArgumentError sylvester_resolvent_precompute(ones(2, 3), 1)
        @test_throws DimensionMismatch sylvester_resolvent_precompute(T, k; X_oracle = zeros(k, k))
    end

    configs = (("V1", config_v1()), ("V2", config_v2()), ("V2.5", config_v2p5()), ("V3", config_v3()))

    @testset "The bound is above the resolvent norm: $name" for (name, cfg) in configs
        for CT in (Float64, ComplexF64), (n, k) in ((6, 2), (10, 3), (9, 1), (7, 6))
            T = Matrix(UpperTriangular(0.3 * randn(CT, n, n))) + Diagonal(CT.(1:n))
            p = sylvester_resolvent_precompute(T, k)
            for z in (0.2 + 0.3im, 2.5 + 0.4im, -1.0 + 0im, n + 0.5 + 0.1im, 3.5 - 2im)
                r = parametric_resolvent_bound(p, T, z, cfg)
                @test r.success
                @test r.resolvent_bound ≥ truth(T, z) * tol
                @test r.M_A ≥ truth(T[1:k, 1:k], z) * tol
                @test r.M_D ≥ truth(T[(k + 1):n, (k + 1):n], z) * tol
                @test r.K_S == p.similarity_cond && r.r == p.residual_norm
                @test r.z == ComplexF64(z)
            end
        end
    end

    @testset "A similarity that does not solve the equation" begin
        n, k = 8, 3
        T = Matrix(UpperTriangular(0.3 * randn(ComplexF64, n, n))) + Diagonal(ComplexF64.(1:n))
        for X in (zeros(ComplexF64, k, n - k), randn(ComplexF64, k, n - k))
            p = sylvester_resolvent_precompute(T, k; X_oracle = X)
            for (_, cfg) in configs, z in (0.3 + 0.2im, 4.5 + 1im)
                r = parametric_resolvent_bound(p, T, z, cfg)
                @test r.success && r.resolvent_bound ≥ truth(T, z) * tol
            end
        end
    end

    @testset "The bound against the one without similarity" begin
        # a well separated leading block: the bound stays within the factor κ₂(S) and a constant
        n, k = 12, 2
        T = Matrix(UpperTriangular(0.05 * randn(ComplexF64, n, n))) +
            Diagonal(ComplexF64.([0.0, 0.1, (5:(n + 2))...]))
        p = sylvester_resolvent_precompute(T, k)
        z = 0.05 + 0.3im
        r = parametric_resolvent_bound(p, T, z, config_v2())
        @test truth(T, z) * tol ≤ r.resolvent_bound ≤ 3 * p.similarity_cond * truth(T, z)
    end

    @testset "Points where no bound exists" begin
        n, k = 6, 2
        T = Matrix(UpperTriangular(0.3 * randn(ComplexF64, n, n))) + Diagonal(ComplexF64.(1:n))
        p = sylvester_resolvent_precompute(T, k)
        for (_, cfg) in configs
            r1 = parametric_resolvent_bound(p, T, T[1, 1], cfg)      # an eigenvalue of T11
            @test !r1.success && r1.resolvent_bound == Inf
            r2 = parametric_resolvent_bound(p, T, T[n, n], cfg)      # an eigenvalue of T22
            @test !r2.success && r2.resolvent_bound == Inf
        end
        @test_throws DimensionMismatch parametric_resolvent_bound(p, T[1:5, 1:5], 0.5im)
    end

    @testset "The estimators of the large block" begin
        n, k = 9, 2
        # strictly diagonally dominant T22: both Neumann bounds apply without the fallback
        T = Matrix(UpperTriangular(0.02 * randn(ComplexF64, n, n))) + Diagonal(ComplexF64.(1:n))
        p = sylvester_resolvent_precompute(T, k)
        z = -1.0 + 0.5im
        for est in (TriBacksub, NeumannOneInf, NeumannCollatz2)
            cfg = ResolventBoundConfig(OneInfNorm, est, CouplingNone, :M1, 3, false)
            r = parametric_resolvent_bound(p, T, z, cfg)
            @test r.success
            @test r.M_D ≥ truth(T[(k + 1):n, (k + 1):n], z) * tol
            @test r.resolvent_bound ≥ truth(T, z) * tol
        end
        # far from dominant: the Neumann series does not apply; with the fallback the recursion does
        T2 = Matrix(UpperTriangular(5 * ones(ComplexF64, n, n))) + Diagonal(ComplexF64.(1:n))
        p2 = sylvester_resolvent_precompute(T2, k)
        for est in (NeumannOneInf, NeumannCollatz2)
            off = ResolventBoundConfig(OneInfNorm, est, CouplingNone, :M1, 3, false)
            on = ResolventBoundConfig(OneInfNorm, est, CouplingNone, :M1, 3, true)
            @test !parametric_resolvent_bound(p2, T2, z, off).success
            r = parametric_resolvent_bound(p2, T2, z, on)
            @test r.success && r.resolvent_bound ≥ truth(T2, z) * tol
        end
    end

    @testset "Norm estimators" begin
        for M in (randn(4, 6), randn(ComplexF64, 5, 5), ones(2, 2))
            for est in (OneInfNorm, FrobeniusNorm)
                @test estimate_2norm(M, est) ≥ opnorm(M) * tol
                @test estimate_2norm(BallMatrix(M), est) ≥ opnorm(M) * tol
            end
        end
        # the larger of the row and column Euclidean norms is a LOWER bound of the spectral norm
        # (√2 against 2 for ones(2, 2)); it was offered as an estimator and is gone
        @test !isdefined(BallArithmetic, :RowCol2Norm)
        cfg = ResolventBoundConfig(FrobeniusNorm, TriBacksub, CouplingARSolve, :M4, 3, true)
        T = Matrix(UpperTriangular(0.3 * randn(ComplexF64, 7, 7))) + Diagonal(ComplexF64.(1:7))
        r = parametric_resolvent_bound(sylvester_resolvent_precompute(T, 3), T, 0.4im, cfg)
        @test r.success && r.resolvent_bound ≥ truth(T, 0.4im) * tol
    end

    @testset "BigFloat" begin
        setprecision(BigFloat, 256) do
            n, k = 6, 2
            T = Complex{BigFloat}.(Matrix(UpperTriangular(0.3 * randn(ComplexF64, n, n))) +
                                   Diagonal(ComplexF64.(1:n)))
            p = sylvester_resolvent_precompute(T, k)
            @test p.precomputation_success && p.residual_norm isa BigFloat
            for (_, cfg) in configs
                r = parametric_resolvent_bound(p, T, 0.3 + 0.4im, cfg)
                @test r.success && r.resolvent_bound isa BigFloat
                @test r.resolvent_bound ≥ truth(T, 0.3 + 0.4im) * tol
            end
        end
    end

    @testset "Several points, the convenience form, the warm start" begin
        n, k = 8, 3
        T = Matrix(UpperTriangular(0.3 * randn(ComplexF64, n, n))) + Diagonal(ComplexF64.(1:n))
        zs = [0.5im, 2.5 + 0.2im, -1.0 + 0im]
        p, rs = parametric_resolvent_bound(T, k, zs, config_v2())
        @test length(rs) == 3 && all(r -> r.success, rs)
        @test all(rs[i].resolvent_bound ≥ truth(T, zs[i]) * tol for i in 1:3)
        p1, r1 = parametric_resolvent_bound(T, k, zs[1])
        @test r1.config == config_v1() && r1.resolvent_bound ≥ truth(T, zs[1]) * tol

        F = svd(zs[1] * I - T[1:k, 1:k])
        z_near = zs[1] + 1e-5
        rw = parametric_resolvent_bound(p, T, z_near, config_v2();
            svd_warm_start = SVDWarmStart(F.U, F.S, F.V))
        @test rw.success && rw.resolvent_bound ≥ truth(T, z_near) * tol
        # a useless warm start may fail to certify, and must not produce a wrong bound
        Q = Matrix(qr(randn(ComplexF64, k, k)).Q)
        rb = parametric_resolvent_bound(p, T, z_near, config_v2();
            svd_warm_start = SVDWarmStart(Q, ones(k), Q))
        @test !rb.success || rb.resolvent_bound ≥ truth(T, z_near) * tol
    end

    @testset "Split selection and comparison of the configurations" begin
        n = 10
        T = Matrix(UpperTriangular(0.3 * randn(ComplexF64, n, n))) + Diagonal(ComplexF64.(1:n))
        z = 0.5 + 0.5im
        best = find_optimal_split(T, z; k_range = 2:5)
        @test best !== nothing
        kbest, pbest, rbest = best
        @test kbest in 2:5 && pbest.k == kbest
        @test rbest.resolvent_bound ≥ truth(T, z) * tol
        @test all(rbest.resolvent_bound ≤ parametric_resolvent_bound(T, k, z)[2].resolvent_bound
                  for k in 2:5)
        @test find_optimal_split(T, T[1, 1]; k_range = 2:3) === nothing

        cmp = compare_all_configs(T, 3, z)
        @test Set(keys(cmp.bounds)) == Set(["V1", "V2", "V2.5", "V3"])
        @test all(b ≥ truth(T, z) * tol for b in values(cmp.bounds))
        @test cmp.bounds[cmp.best] == minimum(values(cmp.bounds))
        # the solves can only improve on the product bound for the coupling
        @test cmp.results["V2"].coupling_term ≤ cmp.results["V1"].coupling_term
        @test cmp.results["V2.5"].coupling_term ≤ cmp.results["V1"].coupling_term
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
