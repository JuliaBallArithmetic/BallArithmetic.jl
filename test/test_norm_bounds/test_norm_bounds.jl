@testset "Norm bounds" begin
    A = zeros(1024, 1024) + I

    bA = BallMatrix(A)

    @test BallArithmetic.upper_bound_L1_opnorm(bA) >= 1.0
    @test BallArithmetic.upper_bound_L_inf_opnorm(bA) >= 1.0
    @test BallArithmetic.upper_bound_L2_opnorm(bA) >= 1.0
    @test BallArithmetic.svd_bound_L2_opnorm(bA) >= 1.0
    @test BallArithmetic.svd_bound_L2_opnorm_inverse(bA) >= 1.0
    @test BallArithmetic.svd_bound_L2_resolvent(bA, Ball(0.5)) >= 2.0

    bA = bA + Ball(0.0, 1 / 16) * I

    @test BallArithmetic.upper_bound_L1_opnorm(bA) >= 1.0 + 1 / 16
    @test BallArithmetic.upper_bound_L_inf_opnorm(bA) >= 1.0 + 1 / 16
    @test BallArithmetic.upper_bound_L2_opnorm(bA) >= 1.0 + 1 / 16
    @test BallArithmetic.svd_bound_L2_opnorm(bA) >= 1.0 + 1 / 16
    @test BallArithmetic.svd_bound_L2_opnorm_inverse(bA) >= 1 / (1.0 - 1 / 16)
    @test BallArithmetic.svd_bound_L2_resolvent(bA, Ball(0.5)) >= 1 / (0.5 - 1 / 16)
end
