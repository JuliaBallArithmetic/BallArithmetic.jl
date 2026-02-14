using BallArithmetic
using LinearAlgebra
using Test

@testset "ordschur_ball" begin

    @testset "ordschur_bigfloat — basic reordering" begin
        # 3×3 diagonal: move eigenvalue at position 3 to position 1
        T = diagm(0 => [1.0, 2.0, 5.0])
        Q = Matrix{Float64}(I, 3, 3)
        select = [false, false, true]

        Q_ord, T_ord, vals = ordschur_bigfloat(T, Q, select)

        # Selected eigenvalue (5.0) should be at top-left
        @test abs(T_ord[1, 1] - 5.0) < 1e-10
        # Remaining eigenvalues should be in the bottom-right block
        remaining = sort(real.([T_ord[2, 2], T_ord[3, 3]]))
        @test remaining ≈ [1.0, 2.0] atol = 1e-10
    end

    @testset "ordschur_bigfloat — eigenvalue preservation" begin
        n = 5
        A = randn(n, n)
        F = schur(A)
        eigs_before = sort(real.(F.values))

        select = [true, true, false, false, false]
        Q_ord, T_ord, vals = ordschur_bigfloat(F.T, F.Z, select)
        eigs_after = sort(real.(diag(T_ord)))

        @test eigs_before ≈ eigs_after atol = 1e-10
    end

    @testset "ordschur_bigfloat — orthogonality" begin
        n = 4
        A = randn(n, n)
        F = schur(A)
        select = [true, false, true, false]

        Q_ord, T_ord, _ = ordschur_bigfloat(F.T, F.Z, select)

        @test Q_ord' * Q_ord ≈ I atol = 1e-12
    end

    @testset "ordschur_bigfloat — reconstruction" begin
        n = 4
        A = randn(n, n)
        F = schur(A)
        select = [true, false, false, true]

        Q_ord, T_ord, _ = ordschur_bigfloat(F.T, F.Z, select)

        # Q_ord * T_ord * Q_ord' should reconstruct A
        @test Q_ord * T_ord * Q_ord' ≈ A atol = 1e-10
    end

    @testset "ordschur_bigfloat — complex Schur" begin
        n = 4
        # Non-symmetric matrix (complex eigenvalues)
        A = [0.0 1.0 0.0 0.0;
             -2.0 0.0 1.0 0.0;
             0.0 0.0 0.0 1.0;
             0.0 0.0 -3.0 0.0]
        Ac = complex(A)
        F = schur(Ac)
        select = [true, false, true, false]

        Q_ord, T_ord, _ = ordschur_bigfloat(F.T, F.Z, select)

        @test Q_ord' * Q_ord ≈ I atol = 1e-12
        @test Q_ord * T_ord * Q_ord' ≈ Ac atol = 1e-10
    end

    @testset "ordschur_ball — BallMatrix with zero radii" begin
        n = 4
        A = randn(n, n)
        Ac = complex(A)
        F = schur(Ac)

        Q_ball = BallMatrix(F.Z)
        T_ball = BallMatrix(F.T)
        select = [true, true, false, false]

        result = ordschur_ball(Q_ball, T_ball, select)

        # Radii should be finite
        @test all(isfinite, rad(result.T))
        @test all(isfinite, rad(result.Q))

        # Orthogonality defect should be small
        @test result.orth_defect < 1e-12

        # Factorization defect should be small
        @test result.fact_defect < 1e-10

        # Midpoint should reconstruct A
        @test mid(result.Q) * mid(result.T) * mid(result.Q)' ≈ Ac atol = 1e-10

        # T_ord midpoint should be upper triangular (regression: ordschur_ball
        # used to produce non-triangular midpoints from G'*T*G arithmetic)
        @test istriu(mid(result.T))
    end

    @testset "ordschur_ball — BallMatrix with non-zero radii" begin
        n = 3
        A_mid = [1.0+0im 2.0+0im 0.5+0im;
                 0.0+0im 3.0+0im 1.0+0im;
                 0.0+0im 0.0+0im 5.0+0im]
        A_rad = fill(1e-10, n, n)
        A_ball = BallMatrix(A_mid, A_rad)

        F = schur(A_mid)
        Q_ball = BallMatrix(F.Z, fill(1e-12, n, n))
        T_ball = BallMatrix(F.T, fill(1e-12, n, n))
        select = [false, false, true]  # move eigenvalue 5.0 to top

        result = ordschur_ball(Q_ball, T_ball, select)

        # Radii should be finite and positive
        @test all(isfinite, rad(result.T))
        @test all(r -> r >= 0, rad(result.T))
        @test all(isfinite, rad(result.Q))

        # Selected eigenvalue (≈5.0) should be at top-left
        @test abs(mid(result.T)[1, 1] - 5.0) < 1e-8
    end

    @testset "ordschur_ball — BigFloat 256-bit" begin
        old_prec = precision(BigFloat)
        setprecision(BigFloat, 256)
        try
            n = 3
            A_mid = Complex{BigFloat}[1 2 0; 0 3 1; 0 0 5]
            F = schur(Matrix(A_mid))
            Q_ball = BallMatrix(F.Z)
            T_ball = BallMatrix(F.T)
            select = [false, true, true]

            result = ordschur_ball(Q_ball, T_ball, select)

            # Orthogonality defect should be ≈ machine epsilon for BigFloat
            @test result.orth_defect < BigFloat(10)^(-60)
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "ordschur_ball — rigorous_schur_bigfloat pipeline" begin
        n = 4
        A_mid = randn(n, n)
        A_ball = BallMatrix(A_mid, fill(1e-10, n, n))

        Q_ball, T_ball, result = rigorous_schur_bigfloat(A_ball; target_precision=256)
        @test result.converged

        select = [true, false, false, true]
        ord_result = ordschur_ball(Q_ball, T_ball, select)

        # Check all fields are finite
        @test all(isfinite, rad(ord_result.T))
        @test all(isfinite, rad(ord_result.Q))
        @test isfinite(ord_result.orth_defect)
        @test isfinite(ord_result.fact_defect)

        # T midpoint must be upper triangular
        @test istriu(mid(ord_result.T))
    end

    @testset "ordschur_ball → Sylvester pipeline (regression)" begin
        n = 4
        A_mid = randn(n, n)
        A_ball = BallMatrix(A_mid, fill(1e-10, n, n))

        Q_ball, T_ball, result = rigorous_schur_bigfloat(A_ball; target_precision=256)
        @test result.converged

        select = [true, false, false, false]
        ord = ordschur_ball(Q_ball, T_ball, select)

        # This used to throw ArgumentError("T must be upper triangular")
        Y = triangular_sylvester_miyajima_enclosure(ord.T, 1)
        @test all(isfinite, mid(Y))
        @test all(isfinite, rad(Y))
    end
end

@testset "compute_spectral_projector_schur — arbitrary indices" begin

    @testset "Single eigenvalue at arbitrary position" begin
        # Upper triangular with known eigenvalues
        A = BallMatrix([1.0 2.0 0.5; 0.0 3.0 1.0; 0.0 0.0 5.0], fill(1e-10, 3, 3))

        # Project onto eigenvalue at position 3 (eigenvalue ≈ 5.0)
        result = compute_spectral_projector_schur(A, 3:3)

        # Should be rank 1 projector
        @test result.idempotency_defect < 1e-4
        @test isfinite(result.projector_norm)
    end

    @testset "Consistency: [1,2] via Vector vs 1:2 via UnitRange" begin
        A = BallMatrix([4.0 1.0 0.0; 0.0 3.0 0.5; 0.0 0.0 1.0], fill(1e-10, 3, 3))

        result_range = compute_spectral_projector_schur(A, 1:2)
        result_vec = compute_spectral_projector_schur(A, [1, 2])

        # Projectors should be similar (same eigenspace)
        P_range = mid(result_range.projector)
        P_vec = mid(result_vec.projector)
        @test norm(P_range - P_vec) < 1e-6
    end

    @testset "Idempotency for reordered projector" begin
        n = 4
        A_mid = triu(randn(n, n)) .+ Diagonal([1.0, 2.0, 5.0, 6.0])
        A = BallMatrix(A_mid, fill(1e-10, n, n))

        result = compute_spectral_projector_schur(A, 3:4)
        @test result.idempotency_defect < 1e-4
    end

    @testset "schur_data kwarg bypasses Schur" begin
        A_mid = [4.0 1.0 0.0; 0.0 3.0 0.5; 0.0 0.0 1.0]
        A = BallMatrix(A_mid, fill(1e-10, 3, 3))

        F = schur(A_mid)

        result = compute_spectral_projector_schur(A, 1:2; schur_data=(F.Z, F.T))
        @test result.idempotency_defect < 1e-4
        @test isfinite(result.projector_norm)
    end

    @testset "AbstractVector{Int} method" begin
        A_mid = triu(randn(5, 5)) .+ Diagonal([1.0, 2.0, 5.0, 6.0, 10.0])
        A = BallMatrix(A_mid, fill(1e-10, 5, 5))

        result = compute_spectral_projector_schur(A, [2, 4])
        @test result.idempotency_defect < 1e-3
        @test isfinite(result.projector_norm)
    end
end

@testset "spectral_projector_error_bound" begin

    @testset "Tiny residuals give tiny bound" begin
        # Typical BigFloat scenario: defects ≈ 10⁻⁷⁷
        bound = spectral_projector_error_bound(
            resolvent_bound_A = 10.0,
            contour_radius = 0.5,
            orth_defect = 1e-70,
            fact_defect = 1e-70
        )
        @test bound < 1e-60
        @test bound > 0
    end

    @testset "Returns Inf for δ ≥ 1" begin
        bound = spectral_projector_error_bound(
            resolvent_bound_A = 1.0,
            contour_radius = 1.0,
            orth_defect = 1.5,
            fact_defect = 1e-10
        )
        @test isinf(bound)
    end

    @testset "Returns Inf when γ ≥ 1" begin
        # Large resolvent * large factorization defect → γ ≥ 1
        bound = spectral_projector_error_bound(
            resolvent_bound_A = 1e10,
            contour_radius = 1.0,
            orth_defect = 0.0,
            fact_defect = 1.0     # M_A * ε / (1-δ) = 1e10 ≥ 1
        )
        @test isinf(bound)
    end

    @testset "Monotone in fact_defect" begin
        b1 = spectral_projector_error_bound(
            resolvent_bound_A = 5.0, contour_radius = 1.0,
            orth_defect = 1e-15, fact_defect = 1e-15)
        b2 = spectral_projector_error_bound(
            resolvent_bound_A = 5.0, contour_radius = 1.0,
            orth_defect = 1e-15, fact_defect = 1e-10)
        @test b2 > b1
    end

    @testset "Monotone in orth_defect" begin
        b1 = spectral_projector_error_bound(
            resolvent_bound_A = 5.0, contour_radius = 1.0,
            orth_defect = 1e-15, fact_defect = 1e-15)
        b2 = spectral_projector_error_bound(
            resolvent_bound_A = 5.0, contour_radius = 1.0,
            orth_defect = 1e-10, fact_defect = 1e-15)
        @test b2 > b1
    end

    @testset "Scales linearly with contour_radius" begin
        b1 = spectral_projector_error_bound(
            resolvent_bound_A = 5.0, contour_radius = 1.0,
            orth_defect = 1e-15, fact_defect = 1e-15)
        b2 = spectral_projector_error_bound(
            resolvent_bound_A = 5.0, contour_radius = 2.0,
            orth_defect = 1e-15, fact_defect = 1e-15)
        @test b2 ≈ 2 * b1 rtol = 1e-10
    end

    @testset "Works with BigFloat" begin
        bound = spectral_projector_error_bound(
            resolvent_bound_A = BigFloat(10),
            contour_radius = BigFloat("0.5"),
            orth_defect = BigFloat(10)^(-70),
            fact_defect = BigFloat(10)^(-70)
        )
        @test bound isa BigFloat
        @test bound < BigFloat(10)^(-60)
    end

    @testset "End-to-end: ordschur_ball residuals → bound" begin
        n = 4
        A_mid = randn(n, n)
        A_ball = BallMatrix(A_mid, fill(1e-10, n, n))

        Q_ball, T_ball, result = rigorous_schur_bigfloat(A_ball; target_precision=256)
        @test result.converged

        select = [true, true, false, false]
        ord = ordschur_ball(Q_ball, T_ball, select)

        # Use a hypothetical resolvent bound (just check it runs)
        bound = spectral_projector_error_bound(
            resolvent_bound_A = BigFloat(100),
            contour_radius = BigFloat("0.5"),
            orth_defect = ord.orth_defect,
            fact_defect = ord.fact_defect
        )
        @test isfinite(bound)
        @test bound > 0
        # Bound is dominated by r · M_A² · fact_defect; fact_defect includes
        # the input BallMatrix radii (1e-10) propagated through matrix products
        @test bound < BigFloat(1)   # sanity: much smaller than O(1)
    end
end

@testset "triangular_sylvester_miyajima_enclosure — BallMatrix" begin

    @testset "Zero radii matches plain matrix" begin
        T_mid = [1.0 0.5 0.2; 0.0 3.0 0.7; 0.0 0.0 5.0]
        k = 1

        Y_plain = triangular_sylvester_miyajima_enclosure(T_mid, k)
        Y_ball = triangular_sylvester_miyajima_enclosure(BallMatrix(T_mid), k)

        @test mid(Y_plain) ≈ mid(Y_ball) atol = 1e-14
        # Ball version radii should be close to plain version radii
        @test all(rad(Y_ball) .>= rad(Y_plain) .- 1e-15)
    end

    @testset "Non-zero radii inflate Y" begin
        T_mid = [1.0 0.5 0.2; 0.0 3.0 0.7; 0.0 0.0 5.0]
        T_rad = fill(1e-8, 3, 3)
        T_ball = BallMatrix(T_mid, T_rad)
        k = 1

        Y_plain = triangular_sylvester_miyajima_enclosure(T_mid, k)
        Y_ball = triangular_sylvester_miyajima_enclosure(T_ball, k)

        # Ball radii should be strictly larger (due to perturbation inflation)
        @test all(rad(Y_ball) .> rad(Y_plain))
    end

    @testset "Large radii produce warning" begin
        T_mid = [1.0 0.5; 0.0 1.001]  # small separation
        T_rad = fill(0.01, 2, 2)       # radii comparable to separation
        T_ball = BallMatrix(T_mid, T_rad)

        # Should warn about separation or large perturbation
        # (may not warn if separation holds; just check it doesn't error)
        Y = triangular_sylvester_miyajima_enclosure(T_ball, 1)
        @test all(isfinite, mid(Y))
    end
end
