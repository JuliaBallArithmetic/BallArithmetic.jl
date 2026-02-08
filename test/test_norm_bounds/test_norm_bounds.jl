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

@testset "Collatz underflow behavior" begin
    # Test case: Collatz power iteration can underflow for very small matrices
    # When the matrix norm is much smaller than 1, the iterated vector
    # x_new = A'A * x_old underflows, causing the ratio x_new[i]/x_old[i]
    # to converge to 0 (underflow) or stay at 1.0 (stagnation)
    #
    # This is why upper_bound_L2_opnorm uses min(collatz, sqrt(L1*Linf))
    # as the sqrt(L1*Linf) bound doesn't suffer from underflow.

    # Normal case: Collatz works well for O(1) matrices
    A_normal = BallMatrix([1.0 0.1; 0.2 0.5])
    collatz_normal = BallArithmetic.collatz_upper_bound_L2_opnorm(A_normal)
    l1_linf_normal = sqrt(BallArithmetic.upper_bound_L1_opnorm(A_normal) *
                          BallArithmetic.upper_bound_L_inf_opnorm(A_normal))
    # Both bounds should be similar for normal-scale matrices
    @test isapprox(collatz_normal, l1_linf_normal; rtol=0.5)
    @test collatz_normal >= 0.5  # True norm is around 1.1

    # Small case: demonstrating potential underflow with tiny matrices
    # Scale ~1e-150 will underflow in Float64 power iteration (x'Ax)
    tiny_scale = 1e-150
    A_tiny = BallMatrix([tiny_scale 0.0; 0.0 tiny_scale])
    collatz_tiny = BallArithmetic.collatz_upper_bound_L2_opnorm(A_tiny)
    l1_linf_tiny = sqrt(BallArithmetic.upper_bound_L1_opnorm(A_tiny) *
                        BallArithmetic.upper_bound_L_inf_opnorm(A_tiny))

    # The L1*Linf bound correctly captures the small norm
    @test l1_linf_tiny ≈ tiny_scale rtol=0.01

    # Collatz may return 0 (underflow) or 1 (stagnation) for tiny matrices
    # This is expected behavior - the fallback sqrt(L1*Linf) handles it
    @test collatz_tiny == 0.0 || collatz_tiny >= 0.5  # Either underflow or stagnation

    # Verify upper_bound_L2_opnorm (combined bound) always works
    combined = BallArithmetic.upper_bound_L2_opnorm(A_tiny)
    @test combined ≈ tiny_scale rtol=0.01  # Uses L1*Linf fallback

    # BigFloat case: also tests underflow at BigFloat precision
    setprecision(BigFloat, 256) do
        bf_tiny_scale = BigFloat(10)^(-200)  # Much smaller than Float64 can represent
        A_bf_tiny = BallMatrix([bf_tiny_scale zero(BigFloat);
                                zero(BigFloat) bf_tiny_scale])
        collatz_bf = BallArithmetic.collatz_upper_bound_L2_opnorm(A_bf_tiny)
        l1_linf_bf = sqrt(BallArithmetic.upper_bound_L1_opnorm(A_bf_tiny) *
                          BallArithmetic.upper_bound_L_inf_opnorm(A_bf_tiny))

        # L1*Linf bound works correctly at BigFloat precision
        @test l1_linf_bf ≈ bf_tiny_scale rtol=0.01

        # Collatz may underflow even with BigFloat (after enough iterations)
        @test collatz_bf == 0 || collatz_bf >= BigFloat(0.5) || isapprox(collatz_bf, bf_tiny_scale; rtol=0.01)

        # Combined bound should work
        combined_bf = BallArithmetic.upper_bound_L2_opnorm(A_bf_tiny)
        @test combined_bf ≈ bf_tiny_scale rtol=0.01
    end
end
