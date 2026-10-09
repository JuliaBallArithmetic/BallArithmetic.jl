# Counting eigenvalues in a region from the inclusions of verifyeigall, against the eigenvalues of
# matrices built with a known spectrum, including a double eigenvalue.
using LinearAlgebra, Random

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
