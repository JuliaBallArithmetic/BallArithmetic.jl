using BallArithmetic
using Test
using LinearAlgebra
using Random

# Rump (2022), Theorem 2.2 and the algorithm `verifyeigall`.
#
# The accuracy attainable is governed by the Jordan structure: an eigenvalue whose largest Jordan
# block has size k cannot be enclosed better than about u^(1/k) in floating-point arithmetic
# (paper, lines 140-145). The assertions below are therefore about soundness and about the method
# declining when it cannot certify, not about a fixed accuracy.

# The construction of the paper's section 3: a diagonal matrix with one k x k Jordan block, then a
# similarity by a random matrix.
function _rump_cluster(n, k, rng)
    J = diagm(0 => randn(rng, n))
    λ = randn(rng)
    for i in 1:k
        J[i, i] = λ
        i < k && (J[i, i + 1] = 1.0)
    end
    V = randn(rng, n, n)
    return V \ J * V
end

_covered(r, λs) = all(any(r.certified[i] && abs(l - r.centers[i]) <= r.radii[i]
                          for i in eachindex(r.centers)) for l in λs)

@testset "verifyeigall: random matrices, every eigenvalue enclosed" begin
    rng = MersenneTwister(20260927)
    for n in (8, 20, 40)
        B = randn(rng, n, n) ./ sqrt(n)
        r = verifyeigall(BallMatrix(B))
        @test r.transform_defect < 1
        @test r.spectrum_covered
        @test length(r.clusters) == n          # a random matrix has simple eigenvalues
        @test _covered(r, eigvals(B))
        @test maximum(r.radii) < 1e-8          # simple eigenvalues are enclosed tightly
    end
end

@testset "verifyeigall: a cluster is found and enclosed" begin
    rng = MersenneTwister(11)
    for k in (2, 3)
        n = 30
        B = _rump_cluster(n, k, rng)
        r = verifyeigall(BallMatrix(B))
        @test r.transform_defect < 1
        # the k coincident eigenvalues are grouped, so there are fewer than n clusters
        @test length(r.clusters) <= n - k + 1
        @test any(length(c) >= k for c in r.clusters)
        if r.spectrum_covered
            @test _covered(r, eigvals(B))
            # and no tighter than Wilkinson's floor allows
            @test maximum(r.radii) >= eps(Float64)^(1 / k) / 1e4
        end
    end
end

@testset "verifyeigall: a defective matrix is enclosed, not mis-enclosed" begin
    n = 24
    J = diagm(0 => fill(0.5 + 0im, n), 1 => fill(1.0 + 0im, n - 1))
    U = Matrix(qr(randn(MersenneTwister(4), ComplexF64, n, n)).Q)
    r = verifyeigall(BallMatrix(U * J * U'))
    # the whole spectrum is the single point 0.5; whatever is certified must contain it
    for i in eachindex(r.clusters)
        r.certified[i] || continue
        @test abs(0.5 - r.centers[i]) <= r.radii[i] || length(r.clusters) > 1
    end
    if r.spectrum_covered
        @test any(abs(0.5 - r.centers[i]) <= r.radii[i] for i in eachindex(r.centers))
        # Wilkinson: nothing below u^(1/24) ≈ 0.22 is attainable here
        @test maximum(r.radii) >= 0.1
    end
end

@testset "verifyeigall: declines rather than asserting" begin
    # a large cluster is beyond what the method can certify; it must say so
    rng = MersenneTwister(5)
    B = _rump_cluster(40, 5, rng)
    r = verifyeigall(BallMatrix(B))
    @test r.spectrum_covered == all(r.certified)
    # every certified cluster carries a finite radius, every uncertified one carries Inf
    for i in eachindex(r.clusters)
        @test isfinite(r.radii[i]) == r.certified[i]
    end
end
