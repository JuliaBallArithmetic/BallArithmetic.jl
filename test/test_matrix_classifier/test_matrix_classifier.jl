@testset "Matrix classifier" begin
    using BallArithmetic

    bA = BallMatrix([1.0 0.0; 0.0 1.0])
    @test BallArithmetic.is_M_matrix(bA) == true

    bA = BallMatrix([1.0 0.1; 0.1 1.0])

    bB = BallArithmetic.off_diagonal_abs(bA)

    @test bB.c == [0.0 0.1; 0.1 0.0]

    v = BallArithmetic.diagonal_abs_lower_bound(bA)

    @test v == [1.0; 1.0]

    ρ = BallArithmetic.collatz_upper_bound(bB)

    @test ρ >= 0.1

    @test BallArithmetic.is_M_matrix(bA) == true
end
