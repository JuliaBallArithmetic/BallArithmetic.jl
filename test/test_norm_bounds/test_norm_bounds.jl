using LinearAlgebra

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

    Ac = [1.0 0.0; 0.0 1.0]
    Ar = [0.125 0.0; 0.0 0.125]
    bA = BallMatrix(Ac, Ar)

    @test BallArithmetic._upper_bound_norm(Ac, Ar) >= 1 / 8

    @testset "Oishi bounds" begin
        diag_entries = [2.0, 3.0, 4.0]
        Tdiag = BallMatrix(Matrix(Diagonal(diag_entries)))
        expected = maximum(1.0 ./ abs.(diag_entries))

        @test isapprox(BallArithmetic.oishi_rump_bound(Tdiag, 1), expected; rtol = 1e-12, atol = 0.0)
        @test isapprox(BallArithmetic.rump_oishi_triangular(Tdiag, 1), expected; rtol = 1e-12, atol = 0.0)
        @test BallArithmetic.oishi_rump_bound(Tdiag, size(Tdiag, 1)) ==
              BallArithmetic.svd_bound_L2_opnorm_inverse(Tdiag)

        Tmid = [2.0 0.15 0.05;
                0.0 1.3 0.2;
                0.0 0.0 0.7]
        T = BallMatrix(Tmid)
        k = 2

        bound_general = BallArithmetic.oishi_rump_bound(T, k)
        bound_triangular = BallArithmetic.rump_oishi_triangular(T, k)
        true_opnorm = opnorm(inv(Tmid), 2)

        @test bound_general >= true_opnorm
        @test bound_triangular >= true_opnorm
        @test bound_triangular <= bound_general + eps(bound_general)
    end
end
