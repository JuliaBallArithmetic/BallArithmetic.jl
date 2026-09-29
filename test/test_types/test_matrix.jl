using Random

@testset "Matrix type" begin
    A = BallMatrix(rand(4, 4))

    B = copy(A)
    @test A.c == B.c && A.r == B.r
end

@testset "UniformScaling minus BallMatrix negates every entry" begin
    # `J - A` used to negate only the diagonal, so `I - A` returned `+A[i,j]` off the diagonal
    # and `z*I - A`, the resolvent expression its docstring advertises, was wrong with it.
    A = BallMatrix([1.0 2.0; 3.0 4.0])
    @test mid(I - A) == [0.0 -2.0; -3.0 -3.0]
    @test mid(2 * I - A) == [1.0 -2.0; -3.0 -2.0]
    @test mid(A - I) == [0.0 2.0; 3.0 3.0]

    rng = MersenneTwister(3)
    for n in (2, 5, 9), _ in 1:10
        M = randn(rng, n, n)
        z = randn(rng)
        @test mid(I - BallMatrix(M)) == I - M
        @test mid(z * I - BallMatrix(M)) ≈ z * I - M
        @test mid(BallMatrix(M) - I) == M - I
        # the enclosure still holds entrywise against exact arithmetic
        C = I - BallMatrix(M)
        Ctrue = I - BigFloat.(M)
        @test all(abs.(BigFloat.(mid(C)) .- Ctrue) .<= BigFloat.(rad(C)))
    end

    @testset "ball-valued scaling" begin
        Ab = BallMatrix(ComplexF64[1.0 2.0; 3.0 4.0])
        zb = Ball(1.0 + 2.0im, 1e-8)
        C = zb * I - Ab
        @test mid(C)[1, 2] == -2.0 + 0.0im
        @test mid(C)[2, 1] == -3.0 + 0.0im
        @test mid(C)[1, 1] == (1.0 + 2.0im) - 1.0
        # the radius of the scaling reaches the diagonal
        @test rad(C)[1, 1] >= 1e-8
    end

    @testset "non-square input is rejected" begin
        @test_throws DimensionMismatch I - BallMatrix(randn(2, 3))
    end
end
