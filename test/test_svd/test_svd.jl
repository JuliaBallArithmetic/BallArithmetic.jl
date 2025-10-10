using LinearAlgebra

@testset "verify singular value perron frobenius " begin
    A = [0.5 0.5; 0.3 0.7]
    ρ = BallArithmetic.collatz_upper_bound_L2_opnorm(BallMatrix(A))
    @test 1 ≤ ρ
end

@testset "Qi intervals for diagonally dominant matrices" begin
    mid = [4.0 1.0; 0.5 3.0]
    A = BallMatrix(mid)
    intervals = BallArithmetic.qi_intervals(A)

    @test length(intervals) == size(mid, 1)

    for i in 1:length(intervals)
        expected_center = abs(mid[i, i])
        row_sum = sum(abs(mid[i, j]) for j in axes(mid, 2) if j != i)
        col_sum = sum(abs(mid[j, i]) for j in axes(mid, 1) if j != i)
        expected_radius = max(row_sum, col_sum)

        @test isapprox(intervals[i].c, expected_center; atol = 1e-12, rtol = 0)
        @test intervals[i].r ≥ expected_radius
        @test intervals[i].r ≤ expected_radius + 1e-12
    end
end

@testset "Qi intervals with dominant off-diagonal entries" begin
    mid = [0.1 10.0; 10.0 0.1]
    A = BallMatrix(mid)
    intervals = BallArithmetic.qi_intervals(A)

    @test length(intervals) == size(mid, 1)

    for i in 1:length(intervals)
        row_sum = sum(abs(mid[i, j]) for j in axes(mid, 2) if j != i)
        col_sum = sum(abs(mid[j, i]) for j in axes(mid, 1) if j != i)
        expected_radius = max(row_sum, col_sum)
        expected_center = (abs(mid[i, i]) + expected_radius) / 2

        @test isapprox(intervals[i].c, expected_center; atol = 1e-12, rtol = 0)
        @test isapprox(intervals[i].r, expected_center; atol = 1e-12, rtol = 0)
    end
end

@testset "Qi square-root intervals are sharp" begin
    mid = [1.0 0.3 0.2; 0.4 1.2 0.5; 0.1 0.2 0.9]
    rad = fill(0.05, size(mid))
    A = BallMatrix(mid, rad)

    qi = BallArithmetic.qi_intervals(A)
    qi_sqrt = BallArithmetic.qi_sqrt_intervals(A)
    σ = svd(mid).S

    @test length(qi_sqrt) == min(size(mid)...)

    tol = maximum(A.r)

    for i in 1:length(qi_sqrt)
        lower_sqrt = BallArithmetic.inf(qi_sqrt[i])
        upper_sqrt = BallArithmetic.sup(qi_sqrt[i])
        lower_outer = BallArithmetic.inf(qi[i])
        upper_outer = BallArithmetic.sup(qi[i])

        @test σ[i] + tol ≥ lower_sqrt
        @test σ[i] ≤ upper_sqrt + tol
        @test lower_sqrt ≥ lower_outer - tol
        @test upper_sqrt ≤ upper_outer + tol
    end
end

@testset "Rebalanced Qi intervals" begin
    mid = [1.0 2.0 0.1; 0.3 0.5 1.2; 0.4 0.1 0.8]
    rad = fill(0.02, size(mid))
    A = BallMatrix(mid, rad)

    reb = BallArithmetic.qi_intervals_rebalanced(A)

    row_norms = [norm(mid[i, :], 1) for i in axes(mid, 1)]
    col_norms = [norm(mid[:, j], 1) for j in axes(mid, 2)]
    k = col_norms ./ row_norms
    D = Diagonal(k)
    Dinv = Diagonal(1 ./ k)
    balanced = Dinv * (A * D)
    expected = BallArithmetic.qi_intervals(balanced)

    for i in 1:length(expected)
        @test isapprox(BallArithmetic.mid(reb[i]), BallArithmetic.mid(expected[i]); atol = 1e-12, rtol = 0)
        @test isapprox(BallArithmetic.rad(reb[i]), BallArithmetic.rad(expected[i]); atol = 1e-12, rtol = 0)
    end
end

@testset "Rebalanced Qi square-root intervals" begin
    mid = [1.0 1.1 0.5; 0.2 0.6 1.3; 0.3 0.4 0.9]
    rad = fill(0.015, size(mid))
    A = BallMatrix(mid, rad)

    reb = BallArithmetic.qi_sqrt_intervals_rebalanced(A)

    row_norms = [norm(mid[i, :], 1) for i in axes(mid, 1)]
    col_norms = [norm(mid[:, j], 1) for j in axes(mid, 2)]
    k = col_norms ./ row_norms
    D = Diagonal(k)
    Dinv = Diagonal(1 ./ k)
    balanced = Dinv * (A * D)
    expected = BallArithmetic.qi_sqrt_intervals(balanced)

    for i in 1:length(expected)
        @test isapprox(BallArithmetic.mid(reb[i]), BallArithmetic.mid(expected[i]); atol = 1e-12, rtol = 0)
        @test isapprox(BallArithmetic.rad(reb[i]), BallArithmetic.rad(expected[i]); atol = 1e-12, rtol = 0)
    end
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
