@testset "verify singular value perron frobenius " begin
    A = [0.5 0.5; 0.3 0.7]
    ρ = BallArithmetic.collatz_upper_bound_L2_opnorm(BallMatrix(A))
    @test 1 ≤ ρ
end

@testset "verified svd" begin
    mA = [1.0 0 0 0 2; 0 0 3 0 0; 0 0 0 0 0; 0 2 0 0 0; 0 0 0 0 0]
    rA = zeros(size(mA))
    rA[1, 1] = 0.1
    A = BallMatrix(mA, rA)

    Σ = BallArithmetic.svdbox(A)

    @test abs(3 - Σ[1].c) < Σ[1].r
    @test abs(sqrt(5) - (Σ[2].c)) < Σ[2].r
    @test abs(2 - Σ[3].c) < Σ[3].r
    @test abs(Σ[4].c) < Σ[4].r
    @test abs(Σ[5].c) < Σ[5].r

    A = im * A

    # Σ = IntervalLinearAlgebra.svdbox(A, IntervalLinearAlgebra.R1())
    # @test all([abs(3-Σ[1].c)< Σ[1].r;
    #             sqrt(abs(5-(Σ[2].c)^2)< Σ[2].r);
    #             abs(2-Σ[3].c)< Σ[3].r;
    #             abs(Σ[4].c)< Σ[4].r;
    #             abs(Σ[5].c)< Σ[5].r])

end
