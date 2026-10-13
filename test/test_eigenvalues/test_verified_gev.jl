using Test
using BallArithmetic
using LinearAlgebra
using Random
using GenericLinearAlgebra

# the symmetry of the pencil is stated by the types of the arguments
_sym(X::BallMatrix) = HermitianBallMatrix(X)
_sym(X::AbstractMatrix) = Symmetric(X)

@testset "Verified Generalized Eigenvalue Problems" begin
    @testset "Beta Bound Computation" begin
        # Simple diagonal SPD matrix
        B = BallMatrix([2.0 0.0; 0.0 3.0], fill(1e-10, 2, 2))
        β = compute_beta_bound(B)

        @test β > 0
        @test isfinite(β)

        # β is a rigorous upper bound on ‖B^{-1/2}‖₂, computed via Cholesky
        # with error accounting. For diagonal B = diag(2,3), the ideal value
        # is 1/√λ_min = 1/√2 ≈ 0.707, but the rigorous bound is larger due to
        # rounding error accounting. The bound should be within a reasonable
        # factor of the ideal value.
        @test β >= 1/sqrt(2.0)  # Must be at least the ideal value
        @test β < 2.0  # Should not be excessively large

        # Non-diagonal SPD matrix
        B2 = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
        β2 = compute_beta_bound(B2)
        @test β2 > 0
        @test isfinite(β2)
    end

    @testset "Small 2×2 System" begin
        # Simple generalized eigenvalue problem with known solution
        A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
        B = BallMatrix([2.0 0.5; 0.5 2.0], fill(1e-10, 2, 2))

        # Get approximate solution
        F = eigen(Symmetric(A.c), Symmetric(B.c))
        X̃ = F.vectors
        λ̃ = F.values

        # Verify
        result = verify_generalized_eigenpairs(_sym(A), _sym(B), X̃, λ̃)

        @test result.certified
        @test length(result.eigenvalue_intervals) == 2
        @test length(result.eigenvector_radii) == 2
        @test all(isfinite.(result.eigenvector_radii))

        # Check that approximate eigenvalues are inside intervals
        for i in 1:2
            λ_lower, λ_upper = result.eigenvalue_intervals[i]
            @test λ_lower <= λ̃[i] <= λ_upper
        end

        # Check diagnostic information
        @test result.beta > 0
        @test result.global_bound > 0
        @test all(result.individual_bounds .> 0)
        @test all(result.separation_bounds .> 0)
    end

    @testset "Diagonal Matrices (Easy Case)" begin
        # When both A and B are diagonal, eigenvalues are A[i,i]/B[i,i]
        A = BallMatrix(Diagonal([4.0, 9.0, 16.0]), fill(1e-10, 3, 3))
        B = BallMatrix(Diagonal([2.0, 3.0, 4.0]), fill(1e-10, 3, 3))

        # Expected eigenvalues: [2.0, 3.0, 4.0]
        expected = [4.0/2.0, 9.0/3.0, 16.0/4.0]

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(_sym(A), _sym(B), F.vectors, F.values)

        @test result.certified

        # Check eigenvalues are in intervals
        for i in 1:3
            λ_lower, λ_upper = result.eigenvalue_intervals[i]
            @test λ_lower <= expected[i] <= λ_upper
            # Intervals should be tight for this easy problem
            @test λ_upper - λ_lower < 0.1
        end
    end

    @testset "3×3 System with Well-Separated Eigenvalues" begin
        A = BallMatrix([10.0 1.0 0.5;
                        1.0 5.0 0.2;
                        0.5 0.2 2.0], fill(1e-8, 3, 3))

        B = BallMatrix([2.0 0.0 0.0;
                        0.0 2.0 0.0;
                        0.0 0.0 2.0], fill(1e-8, 3, 3))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(_sym(A), _sym(B), F.vectors, F.values)

        @test result.certified
        @test length(result.eigenvalue_intervals) == 3

        # Well-separated eigenvalues should have good bounds
        for i in 1:3
            @test result.separation_bounds[i] > 0
            @test result.eigenvector_radii[i] < 1.0  # Should be small
        end

        # Intervals should be non-overlapping
        for i in 1:2
            λ_i_upper = result.eigenvalue_intervals[i][2]
            λ_ip1_lower = result.eigenvalue_intervals[i + 1][1]
            @test λ_i_upper < λ_ip1_lower
        end
    end

    @testset "Global vs Individual Bounds" begin
        A = BallMatrix([5.0 1.0; 1.0 4.0], fill(1e-10, 2, 2))
        B = BallMatrix([2.0 0.0; 0.0 2.0], fill(1e-10, 2, 2))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(_sym(A), _sym(B), F.vectors, F.values)

        @test result.certified

        # Individual bounds should be tighter than or equal to global bound
        for i in 1:2
            @test result.individual_bounds[i] <= result.global_bound + 1e-10
        end

        # Separation bounds should be at most the individual bounds
        for i in 1:2
            @test result.separation_bounds[i] <= result.individual_bounds[i] + 1e-10
        end
    end

    @testset "the radii of (39): Lemmas 2 and 3" begin
        radii = BallArithmetic._gev_eigenvalue_radii
        # isolated values, every gap above 2δ: Lemma 2, η = ε
        @test radii([1.0, 2.0, 3.0], 0.2, [0.1, 0.05, 0.15]) == [0.1, 0.05, 0.15]
        # ε above δ: the smaller of the two is used
        @test radii([1.0, 2.0, 3.0], 0.2, [0.3, 0.05, Inf]) == [0.2, 0.05, 0.2]
        # a run 1.0, 1.1, 1.2 (gaps 0.1 ≤ 2δ) whose intervals are disjoint: Lemma 3, η = ε
        @test radii([1.0, 1.1, 1.2, 2.0], 0.1, [0.01, 0.02, 0.03, 0.05]) ==
              [0.01, 0.02, 0.03, 0.05]
        # the same run with overlapping intervals: η = δ on the whole run, ε outside it
        @test radii([1.0, 1.1, 1.2, 2.0], 0.1, [0.06, 0.06, 0.03, 0.05]) ==
              [0.1, 0.1, 0.1, 0.05]
        # touching intervals are not disjoint
        @test radii([1.0, 1.5], 0.25, [0.25, 0.25]) == [0.25, 0.25]
        @test radii([1.0, 1.25], 0.25, [0.125, 0.125]) == [0.25, 0.25]
        # one eigenvalue
        @test radii([1.0], 0.5, [0.1]) == [0.1]
    end

    @testset "a cluster is bounded by δ, not shrunk" begin
        # Eigenvalues {1, 1 + 1e-9, 5} with input balls of radius 1e-8. An earlier version
        # shrank the radii of the pair to half their gap, 5e-10, and 63% of the eigenvalues of
        # sampled matrices fell outside. By (39) the radius of the pair is δ.
        rng = MersenneTwister(2)
        Q = Matrix(qr(randn(rng, 3, 3)).Q)
        Ac = Matrix(Symmetric(Q * Diagonal([1.0, 1.0 + 1e-9, 5.0]) * Q'))
        Bc = Matrix(1.0I, 3, 3)
        r = 1e-8
        A = BallMatrix(Ac, fill(r, 3, 3))
        B = BallMatrix(Bc, fill(r, 3, 3))

        F = eigen(Symmetric(Ac), Symmetric(Bc))
        res = verify_generalized_eigenpairs(_sym(A), _sym(B), Matrix(F.vectors), collect(F.values))

        @test res.certified
        @test res.separation_bounds[1] == res.global_bound
        @test res.separation_bounds[2] == res.global_bound
        @test res.separation_bounds[3] == min(res.global_bound, res.individual_bounds[3])
        @test all(isinf, res.eigenvector_radii[1:2])
        @test isfinite(res.eigenvector_radii[3])
        # symmetric matrices of the two balls: the i-th eigenvalue is in the i-th interval
        outside = 0
        for _ in 1:300
            E = (2rand(rng, 3, 3) .- 1) .* r
            E = (E + E') / 2
            G = (2rand(rng, 3, 3) .- 1) .* r
            G = (G + G') / 2
            ev = sort(eigvals(Symmetric(Ac + E), Symmetric(Bc + G)))
            for i in 1:3
                lo, hi = res.eigenvalue_intervals[i]
                (lo <= ev[i] <= hi) || (outside += 1)
            end
        end
        @test outside == 0
    end

    @testset "eigenvectors against a 256-bit reference (Theorem 7)" begin
        setprecision(BigFloat, 256) do
            rng = MersenneTwister(31)
            for n in (3, 6, 10)
                M = randn(rng, n, n)
                Ac = Matrix(Symmetric(M + M'))
                N = randn(rng, n, n)
                Bc = Matrix(Symmetric(N' * N + n * I))
                F = eigen(Symmetric(Ac), Symmetric(Bc))
                res = verify_generalized_eigenpairs(_sym(BallMatrix(Ac)), _sym(BallMatrix(Bc)),
                    Matrix(F.vectors), collect(F.values))
                @test res.certified
                @test all(isfinite, res.eigenvector_radii)
                # the reference, through the Cholesky factor of B: L⁻¹AL⁻ᵀ y = λ y, x = L⁻ᵀ y
                Lb = cholesky(Symmetric(BigFloat.(Bc))).L
                Cb = Lb \ BigFloat.(Ac) / Lb'
                Fb = eigen(Symmetric((Cb + Cb') / 2))
                Vb = Lb' \ Fb.vectors
                for i in 1:n
                    lo, hi = res.eigenvalue_intervals[i]
                    @test lo <= Fb.values[i] <= hi
                    # the distance of x̃ from the line spanned by the true eigenvector
                    v = Vb[:, i]
                    x = BigFloat.(F.vectors[:, i])
                    d = norm(x - v * (dot(v, x) / dot(v, v)))
                    @test d <= res.eigenvector_radii[i]
                end
            end
        end
    end

    @testset "BigFloat" begin
        setprecision(BigFloat, 256) do
            Ac = BigFloat[4 1 0; 1 3 1; 0 1 5]
            Bc = BigFloat[2 1 0; 1 3 0; 0 0 1]
            F = eigen(Symmetric(Float64.(Ac)), Symmetric(Float64.(Bc)))
            res = verify_generalized_eigenpairs(_sym(BallMatrix(Ac)), _sym(BallMatrix(Bc)), F.vectors,
                F.values)
            @test res isa GEVResult{BigFloat}
            @test res.certified
            @test all(res.separation_bounds .< 1e-12)
        end
    end

    @testset "the paper's version: Theorem 10 for β" begin
        setprecision(BigFloat, 256) do
            rng = MersenneTwister(5)
            for n in (2, 5, 12, 30), T in (Float64, Float32)
                N = randn(rng, T, n, n)
                Bc = Matrix(Symmetric(N' * N + n * I))
                β = BallArithmetic._gev_beta_theorem_10(Bc)
                @test β isa T
                @test isfinite(β)
                # √‖B⁻¹‖₂ = 1/√λ_min(B)
                λmin = minimum(eigvals(Symmetric(BigFloat.(Bc))))
                @test 1 / sqrt(λmin) <= β
                # the price of α₁α∞ ≥ ‖L̃⁻¹‖₂²: at most a factor √n here
                @test β <= sqrt(BigFloat(n)) / sqrt(λmin) * (1 + 1e-3)
            end
            # not positive definite: the factorisation does not run to completion
            @test BallArithmetic._gev_beta_theorem_10([1.0 2.0; 2.0 1.0]) == Inf
            @test BallArithmetic._gev_beta_theorem_10([1.0 0.0; 0.0 -1.0]) == Inf
        end
    end

    @testset "the paper's version against a 256-bit reference" begin
        setprecision(BigFloat, 256) do
            rng = MersenneTwister(77)
            for n in (3, 6, 10, 25), T in (Float64, Float32)
                M = randn(rng, T, n, n)
                Ac = Matrix(Symmetric(M + M'))
                N = randn(rng, T, n, n)
                Bc = Matrix(Symmetric(N' * N + n * I))
                F = eigen(Symmetric(Ac), Symmetric(Bc))
                # plain matrices: the paper's version is the default
                res = verify_generalized_eigenpairs(_sym(Ac), _sym(Bc), Matrix(F.vectors),
                    collect(F.values))
                @test res isa GEVResult{T}
                @test res.certified
                same = verify_generalized_eigenpairs(_sym(BallMatrix(Ac)), _sym(BallMatrix(Bc)),
                    Matrix(F.vectors), collect(F.values); method = :paper)
                @test same.separation_bounds == res.separation_bounds
                Lb = cholesky(Symmetric(BigFloat.(Bc))).L
                Cb = Lb \ BigFloat.(Ac) / Lb'
                Fb = eigen(Symmetric((Cb + Cb') / 2))
                Vb = Lb' \ Fb.vectors
                for i in 1:n
                    lo, hi = res.eigenvalue_intervals[i]
                    @test lo <= Fb.values[i] <= hi
                    @test abs(Fb.values[i] - BigFloat(F.values[i])) <= res.global_bound
                    isfinite(res.eigenvector_radii[i]) || continue
                    v = Vb[:, i]
                    x = BigFloat.(F.vectors[:, i])
                    @test norm(x - v * (dot(v, x) / dot(v, v))) <= res.eigenvector_radii[i]
                end
                # Theorem 6: εᵢ ≤ δ for the δ of (26)
                @test all(res.individual_bounds .<= res.global_bound)
            end
        end
    end

    @testset "the paper's version: a cluster, and what it refuses" begin
        A = [6.0 0.0 0.0; 0.0 9.0 0.0; 0.0 0.0 16.0]
        B = [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]
        F = eigen(Symmetric(A), Symmetric(B))
        res = verify_generalized_eigenpairs(_sym(A), _sym(B), F.vectors, F.values)
        @test res.certified
        @test res.separation_bounds[1] == res.global_bound
        @test res.separation_bounds[2] == res.global_bound
        @test all(isinf, res.eigenvector_radii[1:2])
        @test isfinite(res.eigenvector_radii[3])
        for (i, λ) in enumerate((3.0, 3.0, 4.0))
            lo, hi = res.eigenvalue_intervals[i]
            @test lo <= λ <= hi
        end
        # matrices with radii, and other number types, are for method = :ball
        @test_throws ArgumentError verify_generalized_eigenpairs(
            _sym(BallMatrix(A, fill(1e-12, 3, 3))), _sym(BallMatrix(B)), F.vectors, F.values;
            method = :paper)
        @test_throws ArgumentError verify_generalized_eigenpairs(
            _sym(BallMatrix(BigFloat.(A))), _sym(BallMatrix(BigFloat.(B))), F.vectors, F.values;
            method = :paper)
        @test_throws ArgumentError verify_generalized_eigenpairs(_sym(A), _sym(B), F.vectors, F.values;
            method = :other)
        # B indefinite
        res = verify_generalized_eigenpairs(_sym(A), _sym([1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 1.0]),
            F.vectors, F.values)
        @test !res.certified
        @test occursin("Theorem 10", res.message)
    end

    @testset "input that is not accepted" begin
        A = BallMatrix([4.0 1.0; 1.0 3.0])
        B = BallMatrix([2.0 0.5; 0.5 2.0])
        F = eigen(Symmetric(mid(A)), Symmetric(mid(B)))
        # the symmetry is stated by the types: a plain matrix or a plain ball matrix is refused,
        # whether its entries are symmetric or not
        @test_throws ArgumentError verify_generalized_eigenpairs(A, B, F.vectors, F.values)
        @test_throws ArgumentError verify_generalized_eigenpairs(mid(A), mid(B), F.vectors, F.values)
        @test_throws ArgumentError verify_generalized_eigenpairs(BallMatrix([4.0 1.0; 1.5 3.0]),
            B, F.vectors, F.values)
        # of a matrix whose two triangles differ one triangle is read, as `Symmetric` does
        res = verify_generalized_eigenpairs(_sym(BallMatrix([4.0 1.0; 1.5 3.0])), _sym(B), F.vectors,
            F.values)
        @test res.certified
        res = verify_generalized_eigenpairs(_sym(BallMatrix(mid(A), [0.0 1e-3; 0.0 0.0])), _sym(B),
            F.vectors, F.values)
        @test res.certified
        # not sorted
        res = verify_generalized_eigenpairs(_sym(A), _sym(B), F.vectors[:, [2, 1]], F.values[[2, 1]])
        @test !res.certified
        @test occursin("increasing", res.message)
        # B not positive definite: Theorem 4 does not apply
        Bi = BallMatrix([1.0 0.0; 0.0 -1.0])
        Fi = eigen(mid(A), mid(Bi))
        res = verify_generalized_eigenpairs(_sym(A), _sym(Bi), real.(Fi.vectors), sort(real.(Fi.values)))
        @test !res.certified
        # complex data
        @test_throws ArgumentError verify_generalized_eigenpairs(
            _sym(BallMatrix(complex(mid(A)))), _sym(BallMatrix(complex(mid(B)))), F.vectors, F.values)
    end

    @testset "successful results really contain the whole input ball" begin
        # Where verification does succeed, every matrix in the ball must have
        # its eigenvalues distributed one per reported interval.
        rng = MersenneTwister(9)
        for (gap, r) in [(1.0, 1e-10), (2.0, 1e-8), (3.0, 1e-6)]
            Q = Matrix(qr(randn(rng, 3, 3)).Q)
            Ac = Matrix(Symmetric(Q * Diagonal([1.0, 1.0 + gap, 5.0]) * Q'))
            Bc = Matrix(1.0I, 3, 3)
            A = BallMatrix(Ac, fill(r, 3, 3))
            B = BallMatrix(Bc, fill(r, 3, 3))

            F = eigen(Symmetric(Ac), Symmetric(Bc))
            p = sortperm(F.values)
            res = verify_generalized_eigenpairs(_sym(A), _sym(B), Matrix(F.vectors)[:, p],
                collect(F.values)[p])
            res.certified || continue

            # Aggregate: one assertion per configuration rather than one per
            # sampled eigenvalue, so the sampling strength is unchanged but the
            # suite does not gain thousands of @test records.
            outside = 0
            for _ in 1:300
                E = (2rand(rng, 3, 3) .- 1) .* r
                E = (E + E') / 2
                ev = sort(eigvals(Symmetric(Ac + E)))
                for i in 1:3
                    lo, hi = res.eigenvalue_intervals[i]
                    (lo <= ev[i] <= hi) || (outside += 1)
                end
            end
            @test outside == 0
        end
    end

    @testset "B = Identity (Standard Eigenvalue Problem)" begin
        # When B = I, this reduces to standard eigenvalue problem
        A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
        B = BallMatrix(Matrix{Float64}(I, 2, 2), fill(1e-10, 2, 2))

        F = eigen(Symmetric(A.c))  # Standard eigenvalue problem
        X̃ = F.vectors
        λ̃ = F.values

        result = verify_generalized_eigenpairs(_sym(A), _sym(B), X̃, λ̃)

        @test result.certified

        # Compare with true eigenvalues of A
        true_λ = eigvals(Symmetric(A.c))
        for i in 1:2
            λ_lower, λ_upper = result.eigenvalue_intervals[i]
            @test λ_lower <= true_λ[i] <= λ_upper
        end
    end

    @testset "Larger Matrices (4×4)" begin
        n = 4
        # Create random SPD matrices
        A_center = randn(n, n)
        A_center = A_center * A_center' + 5.0 * I  # Make SPD
        A = BallMatrix(A_center, fill(1e-8, n, n))

        B_center = randn(n, n)
        B_center = B_center * B_center' + 2.0 * I  # Make SPD
        B = BallMatrix(B_center, fill(1e-8, n, n))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(_sym(A), _sym(B), F.vectors, F.values)

        @test result.certified
        @test length(result.eigenvalue_intervals) == n
        @test length(result.eigenvector_radii) == n

        # All bounds should be finite
        @test all(isfinite.(result.separation_bounds))
        @test all(isfinite.(result.eigenvector_radii))
    end

    @testset "Interval Matrices with Larger Uncertainties" begin
        # Test with significant uncertainties
        A = BallMatrix([10.0 2.0; 2.0 8.0], fill(0.1, 2, 2))  # 10% uncertainty
        B = BallMatrix([3.0 0.5; 0.5 3.0], fill(0.05, 2, 2))  # 5% uncertainty

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(_sym(A), _sym(B), F.vectors, F.values)

        @test result.certified

        # Bounds should be larger due to uncertainties
        for i in 1:2
            λ_lower, λ_upper = result.eigenvalue_intervals[i]
            interval_width = λ_upper - λ_lower
            @test interval_width > 0.01  # Should have non-trivial width
        end
    end

    @testset "Error Handling: Non-Square Matrices" begin
        A = BallMatrix(randn(3, 2), fill(1e-10, 3, 2))
        B = BallMatrix(Matrix{Float64}(I, 3, 3), fill(1e-10, 3, 3))
        @test_throws DimensionMismatch verify_generalized_eigenpairs(_sym(A), _sym(B), randn(3, 3),
            randn(3))
    end

    @testset "Error Handling: Dimension Mismatch" begin
        A = BallMatrix(Matrix{Float64}(I, 3, 3), fill(1e-10, 3, 3))
        B = BallMatrix(Matrix{Float64}(I, 3, 3), fill(1e-10, 3, 3))
        @test_throws DimensionMismatch verify_generalized_eigenpairs(_sym(A), _sym(B), randn(3, 2),
            randn(3))
    end

    @testset "Poor Approximate Solution" begin
        # Use random vectors instead of eigenvectors.
        # Seeded: with unseeded `randn` a lucky draw occasionally lands close
        # enough to a true eigenpair that verification succeeds with tight
        # bounds, failing the assertion below roughly 0.8% of runs.
        Random.seed!(20260731)

        A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
        B = BallMatrix([2.0 0.5; 0.5 2.0], fill(1e-10, 2, 2))

        X̃ = randn(2, 2)
        λ̃ = randn(2)
        λ̃ = sort(λ̃)  # Sort for separation algorithm

        result = verify_generalized_eigenpairs(_sym(A), _sym(B), X̃, λ̃)

        # Should fail or give very large bounds
        if result.certified
            @test any(result.separation_bounds .> 10.0) ||
                  any(result.eigenvector_radii .> 10.0)
        else
            @test !result.certified
        end
    end

    @testset "Nearly Clustered Eigenvalues" begin
        # Create matrix with two close eigenvalues
        A = BallMatrix([5.0 0.0 0.0;
                        0.0 5.01 0.0;
                        0.0 0.0 10.0], fill(1e-10, 3, 3))
        B = BallMatrix(Matrix{Float64}(I, 3, 3), fill(1e-10, 3, 3))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(_sym(A), _sym(B), F.vectors, F.values)

        @test result.certified

        # First two eigenvalues are very close
        # Separation bounds should be small for them
        @test result.separation_bounds[1] < 0.01
        @test result.separation_bounds[2] < 0.01

        # Third eigenvalue is well-separated
        @test result.separation_bounds[3] > result.separation_bounds[1]
    end

    @testset "Verification Diagnostics" begin
        A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
        B = BallMatrix([2.0 0.5; 0.5 2.0], fill(1e-10, 2, 2))

        F = eigen(Symmetric(A.c), Symmetric(B.c))
        result = verify_generalized_eigenpairs(_sym(A), _sym(B), F.vectors, F.values)

        # Check all diagnostic fields are present and reasonable
        @test result.beta > 0
        @test isfinite(result.beta)

        @test result.global_bound > 0
        @test isfinite(result.global_bound)

        @test length(result.individual_bounds) == 2
        @test all(result.individual_bounds .> 0)

        @test length(result.separation_bounds) == 2
        @test all(result.separation_bounds .> 0)

        @test result.residual_norm >= 0
        @test isfinite(result.residual_norm)

        @test isempty(result.message)
    end
end
