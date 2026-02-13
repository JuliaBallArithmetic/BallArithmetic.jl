using Test
using BallArithmetic

@testset "Epsilon Inflation" begin
    A = [2.0 4.0; 0.0 4.0]
    bA = BallMatrix(A)
    b = BallVector(ones(2))

    @testset "Vector inflation" begin
        (v, cert) = BallArithmetic.epsilon_inflation(bA, BallVector(ones(2)))
        @test cert
        @test 0.0 ∈ v[1] && 0.25 in v[2]
    end

    @testset "Matrix inflation" begin
        (A_result, cert) = BallArithmetic.epsilon_inflation(bA, BallMatrix([1.0 0.0; 0.0 1.0]))
        @test cert
        @test -0.5 ∈ A_result[1, 2] && 0.0 in A_result[2, 1]
    end
end
