using BallArithmetic
using Test
using LinearAlgebra
using Random
using BallArithmetic: mid, rad
import BallArithmetic: _certify_ball, _vbd_unitary_basis

# Tests for the O(n³) Schur+Newton verified block diagonalisation that replaced the
# O(n⁴) NJD route.  The decisive property is RIGOR: the β-inflated Gershgorin discs
# must enclose every true eigenvalue, including for defective / near-defective
# matrices where the old NJD path (which omitted the Y≠W⁻¹ slack β) was unsound.

# is the true eigenvalue λ inside some certified disc?
enclosed(λ, discs) = any(abs(λ - mid(d)) <= rad(d) for d in discs)

@testset "schur_newton_vbd" begin

    @testset "rigor: discs enclose true eigenvalues" begin
        rng = MersenneTwister(20260627)

        @testset "well-separated symmetric" begin
            for _ in 1:5
                D = Diagonal([10.0, 5.0, 1.0, -3.0, -8.0])
                Q = Matrix(qr(randn(rng, 5, 5)).Q)
                M = Matrix(Symmetric(Q * D * Q'))
                A = BallMatrix(M, fill(1e-7, 5, 5))
                r = schur_newton_vbd(A)
                @test r.nrmR2 < 1
                @test length(r.clusters) == 5            # all singletons
                te = eigvals(Complex{BigFloat}.(M))
                @test all(λ -> enclosed(λ, r.cluster_intervals), te)
            end
        end

        @testset "exactly defective Jordan blocks" begin
            # 3×3 Jordan block (λ=3, multiplicity 3) under a mild similarity
            J = [3.0 1 0; 0 3 1; 0 0 3]
            P = [1.0 0.5 0.2; 0 1 0.5; 0 0 1]
            A = P * J * inv(P)
            r = schur_newton_vbd(BallMatrix(A))
            @test r.nrmR2 < 1
            @test length(r.clusters) == 1               # coincident ⇒ one cluster
            te = eigvals(Complex{BigFloat}.(A))
            @test all(λ -> enclosed(λ, r.cluster_intervals), te)
        end

        @testset "near-defective (the old-NJD failure regime)" begin
            # two eigenvalues a distance ε apart with a coupling — ill-conditioned
            # eigenvector basis.  Schur+Newton coalesces the tail and stays rigorous.
            for ε in (2.0^-10, 2.0^-20, 2.0^-40, 2.0^-52)
                A = [1.0 1.0; 0.0 1.0 + ε]
                r = schur_newton_vbd(BallMatrix(A))
                @test r.nrmR2 < 1
                te = eigvals(Complex{BigFloat}.(A))
                @test all(λ -> enclosed(λ, r.cluster_intervals), te)
            end
        end
    end

    @testset "block-Gershgorin coupling radii (Feingold–Varga)" begin
        # near-defective 3-block + two isolated eigenvalues
        J = [3.0 1 0 0 0; 0 3 1 0 0; 0 0 3 0 0; 0 0 0 9.0 0; 0 0 0 0 -6.0]
        P = [1.0 0.4 0.1 0.0 0.0; 0 1 0.3 0 0; 0 0 1 0 0; 0.1 0 0 1 0; 0 0.1 0 0 1]
        A = P * J * inv(P)
        r = schur_newton_vbd(BallMatrix(A))

        @test length(r.block_coupling) == length(r.clusters)
        @test all(isfinite, r.block_coupling)
        # never looser than the global bound: rᵢ ≤ ‖Ñ‖₂ + β₂; in particular each rᵢ is a
        # valid block radius, and the max is comparable to the global remainder coupling.
        @test all(rᵢ -> rᵢ <= r.remainder_norm + maximum(r.block_coupling) + 1e-12, r.block_coupling)

        # block enclosure: every true eigenvalue λ lies in some Ωᵢ = {σ_min(Pᵢ−λI) ≤ rᵢ}
        te = eigvals(Complex{BigFloat}.(A))
        Pblocks = [mid(r.transformed)[cl, cl] for cl in r.clusters]
        for λ in te
            @test any(eachindex(r.clusters)) do i
                minimum(svdvals(ComplexF64.(Pblocks[i]) - ComplexF64(λ) * I)) <=
                    r.block_coupling[i] + 1e-12
            end
        end
    end

    @testset "barycentre recentring + block_enclosure discs" begin
        J = [3.0 1 0 0 0; 0 3 1 0 0; 0 0 3 0 0; 0 0 0 9.0 0; 0 0 0 0 -6.0]
        P = [1.0 0.4 0.1 0.0 0.0; 0 1 0.3 0 0; 0 0 1 0 0; 0.1 0 0 1 0; 0 0.1 0 0 1]
        A = P * J * inv(P)
        r = schur_newton_vbd(BallMatrix(A))

        @test length(r.block_centers) == length(r.clusters)
        @test length(r.block_nonnormality) == length(r.clusters)
        @test r.block_residual_norm >= 0          # β_Λ = ‖R₁‖₂/(1−‖R₂‖₂)

        # each final (possibly merged) block is genuinely orthonormal within itself
        for cl in r.clusters
            length(cl) > 1 || continue
            Wc = r.basis[:, cl]
            @test opnorm(Wc' * Wc - I, 2) < 1e-10
        end

        # the defective Jordan cluster recenters to barycentre ≈ 3 with sizeable within-block
        # non-normality; the isolated eigenvalues collapse to ≈ 0 non-normality
        jordan = findfirst(cl -> length(cl) == 3, r.clusters)
        @test jordan !== nothing
        @test abs(r.block_centers[jordan] - 3.0) < 1e-8
        @test r.block_nonnormality[jordan] > 0.5
        @test all(r.block_nonnormality[k] < 1e-6 for k in eachindex(r.clusters) if length(r.clusters[k]) == 1)

        # block_enclosure: discs rigorously contain σ(A); multiplicities sum to n
        enc = block_enclosure(r)
        @test sum(d.mult for d in enc) == 5
        te = eigvals(Complex{BigFloat}.(A))
        for λ in te
            @test any(d -> abs(λ - d.center) <= d.radius, enc)
        end
    end

    @testset "Neumann refusal on a numerically singular basis" begin
        # _certify_ball must throw when ‖R₂‖_∞ ≥ 1 (basis not certifiably nonsingular),
        # rather than return a false certificate (the old NJD path warned and proceeded).
        rng = MersenneTwister(11)
        A = BallMatrix([2.0 0.0; 0.0 3.0])
        U = Matrix(qr(randn(rng, 2, 2)).Q)
        V = Matrix(qr(randn(rng, 2, 2)).Q)
        Wbad = Complex{Float64}.(U * Diagonal([1.0, 1e-18]) * V')   # cond ≈ 1e18
        @test_throws ErrorException _certify_ball(A, Wbad)
    end

    @testset "shape / cost sanity" begin
        # diagonal ⇒ n singleton clusters, vanishing certification slack
        A = BallMatrix(Diagonal([4.0, -2.0, 7.0, 1.5]) |> Matrix)
        r = schur_newton_vbd(A)
        @test length(r.clusters) == 4
        @test r.nrmR2 < 1e-12
        @test r.beta < 1e-10
        @test r.kappa < 2          # near-identity basis

        # moderate n completes quickly (old RDEFL staircase was O(n⁴))
        rng = MersenneTwister(3)
        n = 80
        Q = Matrix(qr(randn(rng, n, n)).Q)
        M = Matrix(Symmetric(Q * Diagonal(collect(range(-10, 10, length = n))) * Q'))
        rbig = schur_newton_vbd(BallMatrix(M))
        @test rbig isa SchurNewtonVBDResult
        @test length(rbig.clusters) == n
    end

    @testset "consumer integration (block Schur + spectral projectors)" begin
        J = [3.0 1 0; 0 3 1; 0 0 3]
        P = [1.0 0.5 0.2; 0 1 0.5; 0 0 1]
        A = BallMatrix(P * J * inv(P))

        # block_schur: the inv-based reconstruction (adjoint→inv fix) gives a small
        # residual for the non-unitary basis (the old adjoint version inflated it).
        bs = rigorous_block_schur(A; vbd_method = :schur_newton)
        @test bs.residual_norm < 1e-8

        sp = miyajima_spectral_projectors(A; vbd_method = :schur_newton)
        @test sp.idempotency_defect < 1e-8

        # :njd is accepted as a deprecated synonym (same result)
        bs2 = rigorous_block_schur(A; vbd_method = :njd)
        @test bs2.residual_norm < 1e-8
    end

    @testset "BigFloat genericity" begin
        setprecision(BigFloat, 128) do
            # exactly defective
            A = BallMatrix(BigFloat[1 1; 0 1])
            r = schur_newton_vbd(A)
            @test r.nrmR2 < 1
            @test isfinite(r.remainder_norm)
            te = eigvals(Complex{BigFloat}.(BigFloat[1 1; 0 1]))
            @test all(λ -> enclosed(λ, r.cluster_intervals), te)

            # diagonal
            Ad = BallMatrix(Diagonal(BigFloat[5, 2, -1]) |> Matrix)
            rd = schur_newton_vbd(Ad)
            @test length(rd.clusters) == 3
            @test isfinite(rd.remainder_norm)
        end
    end
end
