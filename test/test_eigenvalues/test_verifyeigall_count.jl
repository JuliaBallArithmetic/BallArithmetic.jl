# Counting eigenvalues in a region from the inclusions of verifyeigall, against the eigenvalues of
# matrices built with a known spectrum, including a double eigenvalue.
using LinearAlgebra, Random, Printf

@testset "eigencount_outside and eigencount_in_disc from verifyeigall" begin
    rng = MersenneTwister(5)
    for (λ, R) in (([3.0, 2.0, 0.5, 0.2, 0.1], 1.0),
                   ([2.0 + 1.0im, 2.0 - 1.0im, 1.5, 0.3, -0.2], 1.0),
                   ([4.0, 4.0, 1.0, 0.5], 2.0))
        n = length(λ)
        V = randn(rng, ComplexF64, n, n)
        A = V * Diagonal(ComplexF64.(λ)) / V
        r = verifyeigall(BallMatrix(A))
        @test r.spectrum_covered
        cnt, ok = eigencount_outside(r, R)
        @test ok
        @test cnt == count(x -> abs(x) > R, λ)
        c = ComplexF64(λ[1])
        cd, okd = eigencount_in_disc(r, c, 0.25)
        @test okd
        @test cd == count(x -> abs(x - c) < 0.25, λ)
    end

    # a circle through an eigenvalue: no count
    A = Diagonal([2.0, 1.0, 0.5]) |> Matrix
    r = verifyeigall(BallMatrix(A))
    @test eigencount_outside(r, 1.0) == (0, false)
    @test eigencount_in_disc(r, 0.0, 2.0) == (0, false)
    @test eigencount_in_disc(r, 0, 1.5) == (2, true)
end

@testset "overlap_components on complex discs" begin
    b = [Ball(0.0 + 0.0im, 1.0), Ball(1.5 + 0.0im, 1.0), Ball(10.0 + 0.0im, 1.0),
         Ball(0.0 + 2.0im, 0.5)]
    comps = sort(sort.(overlap_components(b)))
    @test comps == [[1, 2], [3], [4]]
    # tangent discs count as overlapping (no false negatives)
    @test length(overlap_components([Ball(0.0 + 0.0im, 1.0), Ball(2.0 + 0.0im, 1.0)])) == 1
end

@testset "counts from the Gershgorin discs when Theorem 2.2 does not cover the spectrum" begin
    rng = MersenneTwister(17)
    # a nearly defective cluster near the origin, inside the circle, and two eigenvalues outside
    J = 0.01 .* diagm(1 => ones(5))        # an exact Jordan block at 0, which (2.10) cannot certify
    A0 = cat(J, Diagonal([3.0, 2.0]); dims = (1, 2))
    V = randn(rng, 8, 8)
    A = ComplexF64.(V * A0 / V)
    r = verifyeigall(BallMatrix(A))
    @test !r.spectrum_covered
    @test length(r.gershgorin_centers) == 8
    cnt, ok = eigencount_outside(r, 1.0)
    @test ok && cnt == 2
    cd, okd = eigencount_in_disc(r, 0.0, 1.0)
    @test okd && cd == 6
    # (no containment check against eigvals(A): for a defective block the floating-point
    # eigenvalues are themselves off by about u^(1/6) of its scale, so they are no reference)
    @printf("  nearly defective cluster: spectrum covered by Theorem 2.2 %s, count outside 1 from %s = %d\n",
        r.spectrum_covered, r.spectrum_covered ? "Theorem 2.2" : "Gershgorin", cnt)
end
