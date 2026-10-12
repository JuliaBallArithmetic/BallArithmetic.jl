using Test
using LinearAlgebra
using Random
using BallArithmetic

@testset "Miyajima GEV Procedures (Fixed)" begin
    # Test with simple 3x3 GEV problem
    A = BallMatrix([2.0 0.1 0.0; 0.1 3.0 0.05; 0.0 0.05 1.0])
    B = BallMatrix(Matrix{Float64}(I, 3, 3))

    result = BallArithmetic.miyajima_algorithm_1_procedure_1(A, B)

    @test hasfield(typeof(result), :X)
    @test hasfield(typeof(result), :Y)
    @test hasfield(typeof(result), :Z)
    @test hasfield(typeof(result), :eigenvalues)
    @test hasfield(typeof(result), :coupling_bound)

    # Verify dimensions
    @test size(result.X) == (3, 3)
    @test size(result.Y) == (3, 3)
    @test size(result.Z.c) == (3, 3)

    # Verify coupling bound is reasonable
    @test result.coupling_bound >= 0
    @test result.coupling_bound < 1  # Should be small for well-conditioned problem

    # Verify Y * Z ≈ I
    YZ_mid = result.Y * mid(result.Z)
    YZ_error = opnorm(YZ_mid - I, Inf)
    @test YZ_error < result.coupling_bound
end

@testset "bdd_R2: inequality (8) of Miyajima (2010) is above ‖Y Z̃ − I‖_∞ on the ball" begin
    rng = MersenneTwister(20261012)
    setprecision(BigFloat, 512) do
        for S in (Float64, ComplexF64), n in (3, 12), ε in (0.0, 1e-13, 1e-6)
            Zc = randn(rng, S, n, n) + 3I
            Zr = fill(ε, n, n) .* rand(rng, n, n)
            Y = inv(Zc)
            b = BallArithmetic.bdd_R2(Y, BallMatrix(Zc, Zr))
            @test b isa Float64 && isfinite(b)
            worst = big(0.0)
            for _ in 1:20
                E = Zr .* (S <: Complex ? cis.(2π .* rand(rng, n, n)) : rand(rng, (-1, 1), n, n))
                R = Complex{BigFloat}.(Y) * (Complex{BigFloat}.(Zc) + Complex{BigFloat}.(E)) - I
                worst = max(worst, opnorm(R, Inf))
            end
            @test worst <= b
            # the two terms the corollary states exactly: evaluated above their exact values
            ε > 0 && @test b >= opnorm((abs.(real.(big.(Y))) + abs.(imag.(big.(Y)))) * big.(Zr), Inf)
        end
    end
end

@testset "Block Schur - Basic Decomposition" begin
    A = BallMatrix([2.0 0.1; 0.1 5.0])

    result = BallArithmetic._rigorous_block_schur(A, true; block_structure=:diagonal)

    # Verify basic properties
    @test size(result.Q) == (2, 2)
    @test size(result.T) == (2, 2)
    @test length(result.clusters) == 2

    # Verify orthogonality
    @test result.inverse_defect_norm < 1e-10

    # Verify residual
    @test result.projected_residual_norm < 1e-10

    # Test verify function
    @test verify_block_schur_properties(result; tol=1e-9)
end

@testset "Block Schur - Quasi-Triangular Structure" begin
    # 4x4 with 2 clusters - use radii to create clustering
    A_mid = Matrix(Diagonal([2.0, 10.0, 2.1, 5.0, 15.0, 5.1]))
    A_rad = zeros(size(A_mid))
    A_rad[1, 3] = A_rad[3, 1] = 0.3  # Cluster 2.0 and 2.1
    A_rad[4, 6] = A_rad[6, 4] = 0.3  # Cluster 5.0 and 5.1
    A = BallMatrix(A_mid, A_rad)

    result = BallArithmetic._rigorous_block_schur(A, true; block_structure=:quasi_triangular)

    # Should have 4 clusters total
    @test length(result.clusters) >= 2
    @test length(result.diagonal_blocks) >= 2

    # Verify basic properties (residual scales with radii for interval matrices)
    @test result.projected_residual_norm < 2.0
    @test result.inverse_defect_norm < 1e-9

    # If we have at least 2 clusters, test block extraction
    if length(result.clusters) >= 2
        T_11 = result.diagonal_blocks[1]
        T_22 = result.diagonal_blocks[2]
        T_12 = extract_cluster_block(result, 1, 2)

        @test size(T_11, 1) == length(result.clusters[1])
        @test size(T_22, 1) == length(result.clusters[2])
        @test size(T_12) == (length(result.clusters[1]), length(result.clusters[2]))
    end
end

@testset "Block Schur - Block Structure Options" begin
    A = BallMatrix(Diagonal([1.0, 1.1, 5.0]))

    # Test all three structures
    result_diag = BallArithmetic._rigorous_block_schur(A, true; block_structure=:diagonal)
    result_quasi = BallArithmetic._rigorous_block_schur(A, true; block_structure=:quasi_triangular)
    result_full = BallArithmetic._rigorous_block_schur(A, true; block_structure=:full)

    # All should have same Q and clusters
    @test result_diag.clusters == result_quasi.clusters == result_full.clusters

    # Residuals should all be small
    @test result_diag.projected_residual_norm < 1e-10
    @test result_quasi.projected_residual_norm < 1e-10
    @test result_full.projected_residual_norm < 1e-10

    # Diagonal structure should have minimal off-diagonal norm
    @test result_diag.off_diagonal_norm < 1e-15

    # Quasi-triangular should have some off-diagonal entries
    # (but in this diagonal case, they're still zero)
    @test result_quasi.off_diagonal_norm < 1e-15
end

@testset "Block Schur - Separation Estimation" begin
    A = BallMatrix([2.0 0.1; 0.1 5.0])

    result = BallArithmetic._rigorous_block_schur(A, true)

    @test length(result.clusters) == 2

    # Estimate separation between clusters
    sep = estimate_block_separation(result, 1, 2)

    # Should be approximately 3.0 (5.0 - 2.0)
    @test 2.5 < sep < 3.5
end

@testset "sep_clusters is a lower bound of the distance" begin
    rng = MersenneTwister(20261011)
    exact_gap(a, b) = setprecision(BigFloat, 512) do
        abs(Complex{BigFloat}(mid(a)) - Complex{BigFloat}(mid(b))) - BigFloat(rad(a)) -
        BigFloat(rad(b))
    end
    for S in (Float64, ComplexF64), _ in 1:5000
        ints = [Ball(randn(rng, S), abs(randn(rng)) * 10.0^rand(rng, -16:-1)) for _ in 1:4]
        sep = BallArithmetic.sep_clusters(ints, 1:1, 2:2)
        @test sep == 0 || sep <= exact_gap(ints[1], ints[2])
        # clusters of two balls each: no pair of members is closer than the bound
        sep2 = BallArithmetic.sep_clusters(ints, 1:2, 3:4)
        @test sep2 == 0 || all(sep2 <= exact_gap(ints[i], ints[j]) for i in 1:2, j in 3:4)
    end
end

@testset "Block Schur - the similarity it certifies" begin
    # Q⁻¹AQ = T + S with ‖S‖ ≤ perturbation_norm: checked at 512 bits for the midpoints
    for (A, herm) in (([4.0 1.0 0.2 0.1; 0.3 3.0 0.5 0.2; 0.1 0.2 -1.0 0.4; 0.05 0.1 0.3 -2.0], false),
        ([2.0 0.1 0.05; 0.1 5.0 0.2; 0.05 0.2 9.0], true)),
        structure in (:full, :quasi_triangular, :diagonal)

        r = rigorous_block_schur(BallMatrix(A); hermitian = herm, block_structure = structure)
        @test r.inverse_defect_norm < 1e-10
        @test isfinite(r.perturbation_norm)
        @test r.perturbation_norm ≥ r.projected_residual_norm
        S_norm, R1_norm, R2_norm = setprecision(BigFloat, 512) do
            Ab, Qb, Yb, Tb = big.(A), big.(mid(r.Q)), big.(mid(r.Q_inv)), big.(mid(r.T))
            opnorm(ComplexF64.(inv(Qb) * Ab * Qb - Tb)), opnorm(ComplexF64.(Yb * (Ab * Qb - Qb * Tb))),
            opnorm(ComplexF64.(Yb * Qb - I))
        end
        @test r.perturbation_norm ≥ S_norm * (1 - 1e-10)
        @test r.projected_residual_norm ≥ R1_norm * (1 - 1e-10)
        @test r.inverse_defect_norm ≥ R2_norm * (1 - 1e-10)
        # the ball matrix `similar` contains Q⁻¹AQ, whatever blocks T keeps
        M = setprecision(() -> (Qb = big.(mid(r.Q)); inv(Qb) * big.(A) * Qb), BigFloat, 512)
        @test r.perturbation_error < 1e-12
        @test all(abs.(M - mid(r.similar)) .≤ rad(r.similar) .* (1 + 1e-12))
        @test size(r.perturbation_approx) == size(A)
        # the blocks T drops are in S̃
        structure === :diagonal && length(r.clusters) > 1 &&
            @test opnorm(r.perturbation_approx) ≥ 0.5 * S_norm
    end
    # a ball of matrices: Q⁻¹BQ is in `similar` for members B of the ball
    rng = MersenneTwister(20261011)
    A = [4.0 1.0 0.2 0.1; 0.3 3.0 0.5 0.2; 0.1 0.2 -1.0 0.4; 0.05 0.1 0.3 -2.0]
    r = rigorous_block_schur(BallMatrix(A, fill(1e-6, 4, 4)))
    @test isfinite(r.perturbation_error)
    for _ in 1:20
        B = A + 1e-6 * (2 * rand(rng, 4, 4) .- 1)
        M = setprecision(() -> (Qb = big.(mid(r.Q)); inv(Qb) * big.(B) * Qb), BigFloat, 512)
        @test all(abs.(M - mid(r.similar)) .≤ rad(r.similar) .* (1 + 1e-12))
    end
    @test !isdefined(BallArithmetic, :refine_off_diagonal_block)
end

@testset "Integration: VBD → Block Schur" begin
    # Test complete workflow with matrix designed to create 2 clusters
    A_mid = Matrix(Diagonal([1.0, 10.0, 1.2, 3.0, 15.0, 3.3]))
    A_rad = zeros(size(A_mid))
    A_rad[1, 3] = A_rad[3, 1] = 0.25  # Cluster 1.0 and 1.2
    A_rad[4, 6] = A_rad[6, 4] = 0.35  # Cluster 3.0 and 3.3
    A = BallMatrix(A_mid, A_rad)

    # Step 1: VBD
    vbd = BallArithmetic._schur_gershgorin_hermitian_members(A)
    @test length(vbd.clusters) >= 2

    # Step 2: Block Schur
    schur = BallArithmetic._rigorous_block_schur(A, true)
    @test length(schur.clusters) >= 2

    # both identify the same clusters
    @test vbd.clusters == schur.clusters
end

@testset "BigFloat Support" begin
    # Test that all methods work with BigFloat
    setprecision(BigFloat, 128) do
        A_big = BallMatrix(Diagonal(BigFloat[1.0, 1.1, 5.0]))

        # VBD
        vbd = BallArithmetic._schur_gershgorin_hermitian_members(A_big)
        @test vbd.remainder_norm < BigFloat(1e-30)

        # Block Schur
        schur = BallArithmetic._rigorous_block_schur(A_big, true)
        @test schur.projected_residual_norm < BigFloat(1e-30)
    end
end

@testset "Complex Matrices" begin
    # Test with complex Hermitian matrix
    A_complex = BallMatrix([1.0+0.0im   0.1-0.05im;
                            0.1+0.05im  5.0+0.0im])

    # VBD
    vbd = BallArithmetic._schur_gershgorin_hermitian_members(A_complex)
    @test length(vbd.clusters) == 2

    # Block Schur
    schur = BallArithmetic._rigorous_block_schur(A_complex, true)
    @test verify_block_schur_properties(schur; tol=1e-9)
end

@testset "Non-Hermitian Matrices" begin
    # Test with general (non-Hermitian) matrix
    A = BallMatrix([2.0 1.0; 0.5 3.0])

    # VBD (using Schur form)
    vbd = schur_gershgorin_enclosure(A)
    @test vbd.remainder_norm >= 0

    # Block Schur
    schur = rigorous_block_schur(A)
    @test schur.projected_residual_norm < 1e-9
end

@testset "a ball matrix is not Hermitian: the kind of enclosure follows the type" begin
    # the midpoint is symmetric, the ball contains matrices with complex eigenvalues
    A = BallMatrix([1.0 0.0; 0.0 1.0 + 1e-8], fill(1e-6, 2, 2))
    X = [1.0 1e-6; -1e-6 1.0 + 1e-8]
    λ = eigvals(X)
    @test maximum(abs ∘ imag, λ) > 1e-7
    for r in (schur_gershgorin_enclosure(A), rigorous_block_schur(A),
        rigorous_block_schur(A; vbd_method = :nsd))
        @test eltype(r.cluster_intervals) == Ball{Float64, ComplexF64}
        @test all(any(abs(l - mid(b)) <= rad(b) for b in r.cluster_intervals) for l in λ)
    end
    @test_throws MethodError schur_gershgorin_enclosure(A; hermitian = true)
    @test_throws MethodError rigorous_block_schur(A; hermitian = true)
    @test_throws MethodError schur_gershgorin_enclosure(Hermitian(A))
    @test_throws MethodError rigorous_block_schur(Hermitian(A))

    # a point matrix that is Hermitian by its type: real intervals
    for H in (Hermitian([2.0 0.1; 0.1 5.0]), Symmetric([2.0 0.1; 0.1 5.0]),
        Hermitian(ComplexF64[1 0.1-0.05im; 0.1+0.05im 5]))
        μ = eigvals(H)
        for r in (schur_gershgorin_enclosure(H), rigorous_block_schur(H))
            @test eltype(r.cluster_intervals) == Ball{Float64, Float64}
            @test all(any(abs(l - mid(b)) <= rad(b) + 1e-12 for b in r.cluster_intervals) for l in μ)
        end
    end
end
