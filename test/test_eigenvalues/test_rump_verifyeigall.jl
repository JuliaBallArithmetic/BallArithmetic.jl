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

# A reference that is exact by construction. LAPACK's `eigvals` is not usable here: on a random
# 30 x 30 it is wrong by up to 4.3e-15 where the certified radius is 2.0e-16, so comparing against
# it would fail the sound enclosure. An upper triangular matrix is exactly representable and its
# eigenvalues are exactly its diagonal, so it tests the enclosure against the truth.
function _triangular_with_spectrum(λ::Vector{Float64}, rng)
    n = length(λ)
    A = triu(randn(rng, n, n), 1) ./ sqrt(n)
    for i in 1:n
        A[i, i] = λ[i]
    end
    return A
end

@testset "verifyeigall: every eigenvalue enclosed, exact reference" begin
    rng = MersenneTwister(20260927)
    for n in (8, 20, 40)
        λ = collect(range(-1.0, 1.0; length = n)) .+ 0.13 .* randn(rng, n)
        B = _triangular_with_spectrum(λ, rng)
        r = verifyeigall(BallMatrix(B))
        @test r.transform_defect < 1
        @test r.spectrum_covered
        @test length(r.clusters) == n          # well separated, so simple
        @test _covered(r, λ)                   # against the exact spectrum
        @test maximum(r.radii) < 1e-8
    end
end

@testset "verifyeigall: random matrices are certified and self-consistent" begin
    rng = MersenneTwister(7)
    for n in (8, 20, 40)
        B = randn(rng, n, n) ./ sqrt(n)
        r = verifyeigall(BallMatrix(B))
        @test r.transform_defect < 1
        @test r.spectrum_covered
        @test length(r.clusters) == n
        # each certified disc carries a finite radius, and the radii are tight enough to be
        # narrower than LAPACK's own accuracy on such a matrix
        @test all(isfinite, r.radii)
        @test maximum(r.radii) < 1e-10
    end
end

@testset "verifyeigall: a near cluster shows its sensitivity, then declines" begin
    # The paper (lines 315-320) notes that computing V^{-1} J V in floating point does NOT give a
    # multiple eigenvalue: the k coincident eigenvalues of J become a cluster of radius about
    # u^(1/k). With an accurate transformation those are resolved individually, so what the test
    # asserts is the sensitivity showing up in the radii, and the method declining rather than
    # asserting once the cluster is too tight to separate.
    rng = MersenneTwister(11)
    res = map((1, 2, 3)) do k
        verifyeigall(BallMatrix(_rump_cluster(30, k, rng)))
    end
    for r in res
        @test r.transform_defect < 1
        @test all(isfinite(r.radii[i]) == r.certified[i] for i in eachindex(r.clusters))
    end
    # simple eigenvalues: everything certified, at the rounding unit
    @test res[1].spectrum_covered
    @test maximum(res[1].radii) < 1e-13
    # a double eigenvalue: still certified, but the radius is at Wilkinson's u^(1/2)
    @test res[2].spectrum_covered
    @test maximum(res[2].radii) > 1e-10
    @test maximum(res[2].radii) < 1e-6
    # a triple eigenvalue: the members of the cluster can no longer be certified individually,
    # so the method declines on them and says so rather than returning a bound
    @test count(res[3].certified) < length(res[3].clusters)
    @test !res[3].spectrum_covered
end

@testset "verifyeigall: a defective matrix is declined, not mis-enclosed" begin
    # A matrix unitarily similar to one Jordan block of size 24: the whole spectrum is the single
    # point 0.5, and Wilkinson's bound puts the narrowest attainable inclusion at u^(1/24) = 0.22.
    # Rump writes of such a matrix (lines 491-493) that "verified inclusions can hardly be
    # computed - and they are not". What matters is that nothing false is asserted.
    n = 24
    J = diagm(0 => fill(0.5 + 0im, n), 1 => fill(1.0 + 0im, n - 1))
    U = Matrix(qr(randn(MersenneTwister(4), ComplexF64, n, n)).Q)
    r = verifyeigall(BallMatrix(U * J * U'))

    @test !r.spectrum_covered                     # the union is not claimed to be the spectrum
    # whatever is certified must contain the true eigenvalue, and nothing certified may be
    # narrower than Wilkinson's floor
    for i in eachindex(r.clusters)
        r.certified[i] || continue
        @test abs(0.5 - r.centers[i]) <= r.radii[i]
        @test r.radii[i] >= 0.1
    end
    @test all(isfinite(r.radii[i]) == r.certified[i] for i in eachindex(r.clusters))
    # the result is either a decline or an honest wide enclosure, never a narrow wrong one
    @test count(r.certified) == 0 ||
          all(r.radii[i] >= 0.1 for i in eachindex(r.clusters) if r.certified[i])
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
