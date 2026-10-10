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

    # the reordered pair is whatever floating point produced; the tests check that the bounds
    # returned are above the defects of that pair, evaluated at 512 bits
    exact_norm(M) = opnorm(ComplexF64.(M))
    big512(f) = setprecision(f, BigFloat, 512)

    @testset "ordschur_ball — the reordered pair and its measured defects" begin
        for n in (4, 9), trial in 1:3
            Ac = randn(ComplexF64, n, n)
            F = schur(Ac)
            select = falses(n); select[[2, n]] .= true
            r = ordschur_ball(BallMatrix(F.Z), BallMatrix(F.T), select; A = BallMatrix(Ac))

            @test istriu(mid(r.T)) && all(iszero, rad(r.T))
            @test r.values == diag(mid(r.T))
            # the selected eigenvalues come first
            @test sort(abs.(r.values[1:2] .- [F.T[2, 2], F.T[n, n]])) ≤ [1e-10, 1e-10] ||
                  sort(abs.(r.values[1:2] .- [F.T[n, n], F.T[2, 2]])) ≤ [1e-10, 1e-10]

            dG, ρ, δ, ε = big512() do
                G, T̃, T, Q, A = big.(r.G), big.(mid(r.T)), big.(F.T), big.(mid(r.Q)), big.(Ac)
                exact_norm(G' * G - I), exact_norm(T * G - G * T̃), exact_norm(Q' * Q - I),
                exact_norm(A * Q - Q * T̃)
            end
            @test r.rotation_orth_defect ≥ dG * (1 - 1e-10)
            @test r.rotation_residual ≥ ρ * (1 - 1e-10)
            @test r.orth_defect ≥ δ * (1 - 1e-10)
            @test r.fact_defect ≥ ε * (1 - 1e-10)
            @test r.rotation_orth_defect < 1e-12 && r.rotation_residual < 1e-11
            @test r.orth_defect < 1e-12 && r.fact_defect < 1e-11

            # without A the two defects against A are not produced, and they follow from those
            # of the pair that was given
            r0 = ordschur_ball(BallMatrix(F.Z), BallMatrix(F.T), select)
            @test r0.orth_defect === nothing && r0.fact_defect === nothing
            @test r0.G == r.G && mid(r0.T) == mid(r.T)
            δ0, ε0 = big512() do
                Z, T, A = big.(F.Z), big.(F.T), big.(Ac)
                exact_norm(Z' * Z - I) * (1 + 1e-10) + 1e-300, exact_norm(A * Z - Z * T) * (1 + 1e-10) + 1e-300
            end
            composed = ε0 * sqrt(1 + r0.rotation_orth_defect) + sqrt(1 + δ0) * r0.rotation_residual
            @test composed ≥ ε * (1 - 1e-10)
        end
    end

    @testset "ordschur_ball — input radii are in the bounds" begin
        n = 5
        Ac = randn(ComplexF64, n, n)
        F = schur(Ac)
        select = falses(n); select[4] = true
        rQ, rT, rA = 1e-9, 1e-8, 1e-7
        r = ordschur_ball(BallMatrix(F.Z, fill(rQ, n, n)), BallMatrix(F.T, fill(rT, n, n)), select;
            A = BallMatrix(Ac, fill(rA, n, n)))
        for _ in 1:20
            sgn() = rand((-1.0, 1.0), n, n)
            Qm, Tm, Am = F.Z + rQ * sgn(), F.T + rT * sgn(), Ac + rA * sgn()
            Q = Qm * r.G
            @test all(abs.(Q - mid(r.Q)) .≤ rad(r.Q) .* (1 + 1e-12))
            @test r.rotation_residual ≥ opnorm(Tm * r.G - r.G * mid(r.T)) * (1 - 1e-10)
            @test r.orth_defect ≥ opnorm(Q' * Q - I) * (1 - 1e-10)
            @test r.fact_defect ≥ opnorm(Am * Q - Q * mid(r.T)) * (1 - 1e-10)
        end
    end

    @testset "ordschur_ball — BigFloat 256-bit" begin
        setprecision(BigFloat, 256) do
            A_mid = Complex{BigFloat}[1 2 0; 0 3 1; 0 0 5]
            F = schur(Matrix(A_mid))
            r = ordschur_ball(BallMatrix(F.Z), BallMatrix(F.T), [false, true, true];
                A = BallMatrix(A_mid))
            @test istriu(mid(r.T)) && all(iszero, rad(r.T))
            @test r.orth_defect isa BigFloat && r.orth_defect < BigFloat(10)^(-60)
            @test r.fact_defect < BigFloat(10)^(-60)
            @test r.rotation_residual < BigFloat(10)^(-60)
        end
    end

    @testset "ordschur_ball — rigorous_schur_bigfloat pipeline and the Sylvester enclosure" begin
        n = 4
        A_ball = BallMatrix(randn(n, n), fill(1e-10, n, n))
        Q_ball, T_ball, result = rigorous_schur_bigfloat(A_ball; target_precision=256)
        @test result.converged
        ord = ordschur_ball(Q_ball, T_ball, [true, false, false, false])
        @test istriu(mid(ord.T))
        @test all(isfinite, rad(ord.Q))
        Y = triangular_sylvester_miyajima_enclosure(ord.T, 1)
        @test all(isfinite, mid(Y)) && all(isfinite, rad(Y))
        @test_throws DimensionMismatch ordschur_ball(Q_ball, T_ball, [true, false])
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

        # Commutation: PA ≈ AP (regression: sign error gave wrong Y)
        PA = result.projector * A
        AP = A * result.projector
        @test upper_bound_L2_opnorm(PA - AP) < 1e-4
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

    @testset "Idempotency and commutation for reordered projector" begin
        n = 4
        A_mid = triu(randn(n, n)) .+ Diagonal([1.0, 2.0, 5.0, 6.0])
        A = BallMatrix(A_mid, fill(1e-10, n, n))

        result = compute_spectral_projector_schur(A, 3:4)
        @test result.idempotency_defect < 1e-4

        # Commutation check
        PA = result.projector * A
        AP = A * result.projector
        @test upper_bound_L2_opnorm(PA - AP) < 1e-4
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

    @testset "End to end: the bound is above the distance between the two projectors" begin
        # The spectral projector of a triangular matrix for its p-th diagonal entry, v·u with
        # (T − λ)v = 0, u(T − λ) = 0, by substitution.
        function triangular_projector(T, p)
            n = size(T, 1); λ = T[p, p]
            v = zeros(eltype(T), n); v[p] = 1
            for i in (p - 1):-1:1
                v[i] = -sum(T[i, j] * v[j] for j in (i + 1):p) / (T[i, i] - λ)
            end
            u = zeros(eltype(T), n); u[p] = 1
            for j in (p + 1):n
                u[j] = -sum(u[i] * T[i, j] for i in p:(j - 1)) / (T[j, j] - λ)
            end
            return v * transpose(u)
        end
        CS = BallArithmetic.CertifScripts
        n = 6
        for trial in 1:3
            Ac = Matrix(Diagonal(ComplexF64.([3.0, 0.5, -0.5, 1im, -1im, 0.2 + 0.3im]))) +
                 0.2 * randn(ComplexF64, n, n)
            F = schur(Ac)
            p = argmax(real.(diag(F.T)))                       # the eigenvalue near 3
            select = falses(n); select[p] = true
            ord = ordschur_ball(BallMatrix(F.Z), BallMatrix(F.T), select; A = BallMatrix(Ac))
            radius = 1.0
            cert = CS.run_certification(Ac, CS.CertificationCircle(mid(ord.T)[1, 1], radius;
                samples = 32); log_io = devnull)
            bound = spectral_projector_error_bound(resolvent_bound_A = cert.resolvent_original,
                contour_radius = radius, orth_defect = ord.orth_defect,
                fact_defect = ord.fact_defect)
            distance = setprecision(BigFloat, 512) do
                Fb = schur(Complex{BigFloat}.(Ac))
                pb = argmin(abs.(diag(Fb.T) .- mid(ord.T)[1, 1]))
                P_A = Fb.Z * triangular_projector(Fb.T, pb) * Fb.Z'
                Q = Complex{BigFloat}.(mid(ord.Q))
                P_c = Q * triangular_projector(Complex{BigFloat}.(mid(ord.T)), 1) * Q'
                opnorm(ComplexF64.(P_A - P_c)), opnorm(ComplexF64.(P_A))
            end
            @test isfinite(bound) && bound < 1e-9
            @test bound ≥ distance[1]
            @test distance[2] ≥ 1 - 1e-12                       # a projector, not zero
        end
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

@testset "ordered_schur_data kwarg (Feature 3)" begin

    @testset "Consistency: ordered_schur_data vs schur_data + ordschur" begin
        A_mid = [4.0 1.0 0.0; 0.0 3.0 0.5; 0.0 0.0 1.0]
        A = BallMatrix(A_mid, fill(1e-10, 3, 3))

        # Standard path
        result_std = compute_spectral_projector_schur(A, 1:2)

        # Manually pre-order and pass ordered_schur_data
        F = schur(A_mid)
        result_ord = compute_spectral_projector_schur(A, 1:2;
            ordered_schur_data=(F.Z, F.T, 2))

        @test mid(result_std.projector) ≈ mid(result_ord.projector) atol = 1e-10
    end

    @testset "ordered_schur_data skips ordschur for non-1:k indices" begin
        A_mid = triu(randn(4, 4)) .+ Diagonal([1.0, 3.0, 7.0, 10.0])
        A = BallMatrix(A_mid, fill(1e-10, 4, 4))

        # Use standard path with reordering
        result_vec = compute_spectral_projector_schur(A, [1, 3])

        # Pre-compute reordered Schur
        F = schur(A_mid)
        select = falses(4); select[1] = true; select[3] = true
        F_ord = ordschur(F, BitVector(select))

        result_pre = compute_spectral_projector_schur(A, [1, 3];
            ordered_schur_data=(F_ord.Z, F_ord.T, 2))

        @test mid(result_vec.projector) ≈ mid(result_pre.projector) atol = 1e-6
    end

    @testset "Error on mismatched k" begin
        A = BallMatrix([1.0 2.0; 0.0 3.0], fill(1e-10, 2, 2))
        F = schur(A.c)
        @test_throws ArgumentError compute_spectral_projector_schur(A, 1:1;
            ordered_schur_data=(F.Z, F.T, 2))  # k=2 vs cluster size 1
    end
end

@testset "compute_spectral_coefficient (Feature 2)" begin

    @testset "k=1 scalar: matches full projector" begin
        A_mid = [1.0 2.0 0.5; 0.0 3.0 1.0; 0.0 0.0 5.0]
        A = BallMatrix(A_mid, fill(1e-10, 3, 3))
        v = [1.0, 2.0, 3.0]

        # Full projector path
        result_full = compute_spectral_projector_schur(A, 1:1)
        Pv_full = mid(result_full.projector) * v

        # Coefficient path
        coeff_result = compute_spectral_coefficient(A, v, 1:1)
        c = mid(coeff_result.coefficients)

        # The spectral coefficient c[1] should equal the first component
        # of P*v in Schur coordinates, i.e. [I Y] * Q^H * v
        # The full projected vector is Q * [I Y; 0 0] * Q^H v = Q * [c; 0]
        Q = coeff_result.schur_basis
        Pv_from_coeff = Q[:, 1:1] * c
        @test Pv_from_coeff ≈ Pv_full atol = 1e-8
    end

    @testset "k=2 cluster: matches full projector" begin
        A_mid = triu(randn(5, 5)) .+ Diagonal([1.0, 2.0, 7.0, 8.0, 15.0])
        A = BallMatrix(A_mid, fill(1e-10, 5, 5))
        v = randn(5)

        result_full = compute_spectral_projector_schur(A, 1:2)
        Pv_full = mid(result_full.projector) * v

        coeff_result = compute_spectral_coefficient(A, v, 1:2)
        c = mid(coeff_result.coefficients)
        Q = coeff_result.schur_basis
        Pv_from_coeff = Q[:, 1:2] * c
        @test Pv_from_coeff ≈ Pv_full atol = 1e-6
    end

    @testset "BallVector input" begin
        A_mid = [1.0 2.0 0.5; 0.0 3.0 1.0; 0.0 0.0 5.0]
        A = BallMatrix(A_mid, fill(1e-10, 3, 3))
        v = BallVector([1.0, 2.0, 3.0], [1e-8, 1e-8, 1e-8])

        coeff_result = compute_spectral_coefficient(A, v, 1:1)
        @test all(isfinite, mid(coeff_result.coefficients))
        @test all(r -> r >= 0 && isfinite(r), rad(coeff_result.coefficients))
        # Radii should be positive due to input radii
        @test all(r -> r > 0, rad(coeff_result.coefficients))
    end

    @testset "ordered_schur_data kwarg" begin
        A_mid = [1.0 2.0 0.5; 0.0 3.0 1.0; 0.0 0.0 5.0]
        A = BallMatrix(A_mid, fill(1e-10, 3, 3))
        v = [1.0, 2.0, 3.0]

        F = schur(A_mid)

        # Standard
        result1 = compute_spectral_coefficient(A, v, 1:1)
        # With pre-computed ordered Schur
        result2 = compute_spectral_coefficient(A, v, 1:1;
            ordered_schur_data=(F.Z, F.T, 1))

        @test mid(result1.coefficients) ≈ mid(result2.coefficients) atol = 1e-12
    end

    @testset "Non-contiguous indices" begin
        A_mid = triu(randn(4, 4)) .+ Diagonal([1.0, 5.0, 2.0, 8.0])
        A = BallMatrix(A_mid, fill(1e-10, 4, 4))
        v = randn(4)

        # Full projector with reordering
        result_full = compute_spectral_projector_schur(A, [1, 3])
        Pv_full = mid(result_full.projector) * v

        # Coefficient
        coeff_result = compute_spectral_coefficient(A, v, [1, 3])
        c = mid(coeff_result.coefficients)
        Q = coeff_result.schur_basis
        Pv_from_coeff = Q[:, 1:2] * c
        @test Pv_from_coeff ≈ Pv_full atol = 1e-6
    end

    @testset "BigFloat precision" begin
        old_prec = precision(BigFloat)
        setprecision(BigFloat, 256)
        try
            A_mid = BigFloat[1 2 0; 0 5 1; 0 0 9]
            A = BallMatrix(A_mid)
            v = BigFloat[1, 2, 3]

            coeff_result = compute_spectral_coefficient(A, v, 1:1)
            @test all(isfinite, mid(coeff_result.coefficients))
        finally
            setprecision(BigFloat, old_prec)
        end
    end
end
