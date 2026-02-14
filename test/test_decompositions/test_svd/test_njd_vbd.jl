using BallArithmetic
using Test
using LinearAlgebra

@testset "NJD-based VBD" begin

    @testset "Diagonal matrix — singleton clusters" begin
        D = BallMatrix(Diagonal([1.0, 3.0, 7.0]))
        result = miyajima_vbd_njd(D)

        @test length(result.clusters) == 3
        @test result.max_nilpotent_index == 1
        @test isfinite(result.remainder_norm)
        # All Jordan blocks should be size 1
        for ji in result.jordan_info
            @test ji.nilpotent_index == 1
            @test all(==(1), ji.block_sizes)
        end
    end

    @testset "Exactly defective 2×2 — single Jordan block" begin
        # A = [1 1; 0 1] has a single 2×2 Jordan block at λ=1
        A = BallMatrix([1.0 1.0; 0.0 1.0])
        result = miyajima_vbd_njd(A)

        @test length(result.clusters) == 1
        @test result.max_nilpotent_index == 2
        @test isfinite(result.remainder_norm)
        # Should find one cluster with a 2×2 Jordan block
        @test result.jordan_info[1].nilpotent_index == 2
        @test result.jordan_info[1].block_sizes == [2]
    end

    @testset "Near-defective 2×2 (Paper Example 2 transition)" begin
        # A = [1 1; 0 1+ε] — transitions from diagonalisable to defective
        for k in [0, 10, 20, 52]
            ε = 2.0^(-k)
            A = BallMatrix([1.0 1.0; 0.0 1.0 + ε])
            result = miyajima_vbd_njd(A)

            @test isfinite(result.remainder_norm)
            @test length(result.clusters) >= 1

            if k >= 40
                # Very near-defective: should cluster into one group
                @test length(result.clusters) == 1
            end
        end
    end

    @testset "Explicitly defective 4×4" begin
        # A₀ = [2 2 1 0; 0 1 1 1; -1 -1 0 0; 1 1 1 1]
        A = BallMatrix([2.0 2.0 1.0 0.0;
                        0.0 1.0 1.0 1.0;
                       -1.0 -1.0 0.0 0.0;
                        1.0 1.0 1.0 1.0])
        result = miyajima_vbd_njd(A)

        @test isfinite(result.remainder_norm)
        @test result.max_nilpotent_index >= 1
        @test length(result.clusters) >= 1
    end

    @testset "Block Jordan structure — 6×6 known structure" begin
        # Construct a 6×6 matrix with known Jordan structure:
        # Two 2×2 Jordan blocks at λ=2 and two 1×1 blocks at λ=5
        J = diagm(0 => [2.0, 2.0, 2.0, 2.0, 5.0, 5.0],
                  1 => [1.0, 0.0, 1.0, 0.0, 0.0])
        # Random similarity transform
        rng_state = [0.3 0.1 0.2 0.5 0.1 0.3;
                     0.1 0.4 0.3 0.1 0.2 0.1;
                     0.2 0.3 0.5 0.2 0.1 0.2;
                     0.1 0.2 0.1 0.6 0.3 0.1;
                     0.3 0.1 0.1 0.1 0.4 0.2;
                     0.1 0.2 0.2 0.3 0.2 0.5]
        P = rng_state + 2I
        A = BallMatrix(P * J / P)

        result = miyajima_vbd_njd(A)

        @test isfinite(result.remainder_norm)
        @test length(result.clusters) >= 1  # at least some clustering
        @test result.max_nilpotent_index >= 2  # should detect the 2×2 blocks
    end

    @testset "Well-conditioned symmetric — NJD agrees with NSD" begin
        M = [3.0 0.1 0.05; 0.1 5.0 0.1; 0.05 0.1 8.0]
        A = BallMatrix(Symmetric(M))

        r_nsd = miyajima_vbd(A; hermitian=true)
        r_njd = miyajima_vbd_njd(A)

        # Both should find the same number of clusters
        @test length(r_njd.clusters) == length(r_nsd.clusters)
        # Both should have finite remainder norms
        @test isfinite(r_nsd.remainder_norm)
        @test isfinite(r_njd.remainder_norm)
        # NJD should have nilpotent index 1 for diagonalisable matrices
        @test r_njd.max_nilpotent_index == 1
    end

    @testset "Integration with rigorous_block_schur via vbd_method=:njd" begin
        A = BallMatrix([2.0 2.0 1.0 0.0;
                        0.0 1.0 1.0 1.0;
                       -1.0 -1.0 0.0 0.0;
                        1.0 1.0 1.0 1.0])
        bschur = rigorous_block_schur(A; vbd_method=:njd)

        @test isfinite(bschur.residual_norm)
        @test length(bschur.clusters) >= 1
    end

    @testset "Integration with miyajima_spectral_projectors via vbd_method=:njd" begin
        A = BallMatrix(Diagonal([1.0, 1.1, 5.0, 5.1]))
        proj = miyajima_spectral_projectors(A; vbd_method=:njd)

        @test length(proj.projectors) >= 1
        @test isfinite(proj.idempotency_defect)
    end

    @testset "NJDVBDResult fields are well-typed" begin
        A = BallMatrix([1.0 1.0; 0.0 1.0])
        result = miyajima_vbd_njd(A)

        @test result isa NJDVBDResult
        @test result.basis isa AbstractMatrix
        @test result.transformed isa BallMatrix
        @test result.block_diagonal isa BallMatrix
        @test result.remainder isa BallMatrix
        @test result.clusters isa Vector{UnitRange{Int}}
        @test result.jordan_info isa Vector{<:JordanBlockInfo}
        @test result.clustering_tolerance > 0
        @test result.max_nilpotent_index >= 1
    end

    @testset "Theorem 2 bound helper" begin
        # Simple 2×2 Jordan block
        ji = JordanBlockInfo{Float64}(
            [2], 2, complex(1.0),
            zeros(ComplexF64, 2, 2),
            Matrix{ComplexF64}(I, 2, 2)
        )
        R = 0.01 * ones(Float64, 2, 2)
        bound = BallArithmetic._theorem2_bound(ji, R, 0.5)
        @test isfinite(bound)
        @test bound >= 0
    end

    @testset "SVD staircase on known nilpotent" begin
        # 3×3 shift matrix: nilpotent index 3
        N = [0.0+0im 1.0 0.0; 0.0 0.0 1.0; 0.0 0.0 0.0]
        bs, ni = BallArithmetic._svd_staircase(N, 1e-10)
        @test ni == 3
        @test bs == [3]  # single block of size 3

        # Block diagonal: two 2×2 shift matrices — nilpotent index 2
        M = zeros(ComplexF64, 4, 4)
        M[1, 2] = 1.0
        M[3, 4] = 1.0
        bs2, ni2 = BallArithmetic._svd_staircase(M, 1e-10)
        @test ni2 == 2
        @test sort(bs2) == [2, 2]
    end

    @testset "Canonical nilpotent construction" begin
        M = BallArithmetic._build_canonical_nilpotent([3, 2, 1], 6)
        @test M[1, 2] ≈ 1.0
        @test M[2, 3] ≈ 1.0
        @test M[3, 4] ≈ 0.0  # block boundary
        @test M[4, 5] ≈ 1.0
        @test M[5, 6] ≈ 0.0  # block boundary
        @test count(!iszero, M) == 3  # exactly 3 nonzeros: (1,2), (2,3), (4,5)
    end

end

@testset "BigFloat NJD" begin
        A_bf = BallMatrix(BigFloat.([1.0 1.0; 0.0 1.0]))
        r = miyajima_vbd_njd(A_bf)
        @test isfinite(r.remainder_norm)
        @test length(r.clusters) == 1
        @test r.max_nilpotent_index == 2
        @test r.jordan_info[1].block_sizes == [2]

        # BigFloat diagonal
        D_bf = BallMatrix(BigFloat.(Diagonal([1.0, 3.0, 7.0])))
        r2 = miyajima_vbd_njd(D_bf)
        @test length(r2.clusters) == 3
        @test r2.max_nilpotent_index == 1
    end
