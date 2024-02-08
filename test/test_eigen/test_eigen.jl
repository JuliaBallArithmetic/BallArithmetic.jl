@testset "Eigvals" begin
    using BallArithmetic

    A = BallMatrix(rand(256, 256))

    v = BallArithmetic.gevbox(A, A)

    @test all([abs(v[i].c - 1.0) < v[i].r for i in 1:256])

    bA = BallMatrix([125.0 0.0; 0.0 256.0])

    @test BallArithmetic.collatz_upper_bound(bA) >= 256.0

    v = BallArithmetic.evbox(bA)

    @test abs(v[1].c - 125.0) < v[1].r
end
