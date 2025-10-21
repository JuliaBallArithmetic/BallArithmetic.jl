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

@testset "rigorous_svd result object" begin
    mid = [3.0 0.1 0.0; 0.0 2.0 0.3; 0.0 0.0 0.5]
    rad = zeros(size(mid))
    rad[1, 1] = 1.0e-2
    A = BallMatrix(mid, rad)

    result = BallArithmetic.rigorous_svd(A)

    @test result.singular_values == BallArithmetic.svdbox(A)
    @test result.Σ isa BallMatrix
    @test result.residual isa BallMatrix
    @test result.block_diagonalisation isa BallArithmetic.MiyajimaVBDResult

    for i in 1:length(result.singular_values)
        @test result.Σ[i, i] == result.singular_values[i]
        for j in 1:length(result.singular_values)
            if i != j
                @test result.Σ.c[i, j] == 0
                @test result.Σ.r[i, j] == 0
            end
        end
    end

    @test result.residual_norm ≥ 0
    @test result.right_orthogonality_defect < 1
    @test result.left_orthogonality_defect < 1

    H = adjoint(result.Σ) * result.Σ
    recomputed = BallArithmetic.miyajima_vbd(H; hermitian = true)
    @test result.block_diagonalisation.clusters == recomputed.clusters
    @test result.block_diagonalisation.remainder_norm == recomputed.remainder_norm

    result_no_vbd = BallArithmetic.rigorous_svd(A; apply_vbd = false)

    @test result_no_vbd.block_diagonalisation === nothing
    @test result_no_vbd.singular_values == result.singular_values
    @test result_no_vbd.Σ == result.Σ
    @test result_no_vbd.residual == result.residual

    Σ_no_vbd = BallArithmetic.svdbox(A; apply_vbd = false)
    @test Σ_no_vbd == result.singular_values
end

@testset "Miyajima VBD block diagonalisation" begin
    Σmid = Matrix(Diagonal([5.0, 5.0, 2.0]))
    Σrad = zeros(size(Σmid))
    Σrad[1, 3] = Σrad[2, 3] = Σrad[3, 1] = Σrad[3, 2] = 1.0e-6
    Σ = BallMatrix(Σmid, Σrad)

    result = BallArithmetic.miyajima_vbd(Σ; hermitian = true)

    @test length(result.clusters) == 2
    @test result.clusters[1] == 1:1
    @test result.clusters[2] == 2:3

    @test result.block_diagonal.c[1, 2] == 0
    @test result.block_diagonal.c[1, 3] == 0
    @test result.block_diagonal.c[2, 1] == 0
    @test result.block_diagonal.c[3, 1] == 0
    @test result.block_diagonal.r[1, 2] == 0
    @test result.block_diagonal.r[1, 3] == 0
    @test result.block_diagonal.r[2, 1] == 0
    @test result.block_diagonal.r[3, 1] == 0

    @test result.remainder.c[1, 2] ≈ result.transformed.c[1, 2]
    @test result.remainder.c[1, 3] ≈ result.transformed.c[1, 3]
    @test result.remainder_norm ≥ 0
    collatz = BallArithmetic.collatz_upper_bound_L2_opnorm(result.remainder)
    @test result.remainder_norm <= collatz

    Q = result.basis
    @test isapprox(Q' * Q, Matrix{eltype(Q)}(I, size(Q, 1), size(Q, 1)); atol = 1e-10)

    for (interval, λ) in zip(result.cluster_intervals, result.eigenvalues)
        @test λ ∈ interval
    end
end

@testset "Miyajima VBD permutation grouping" begin
    Σmid = Matrix(Diagonal([1.0, 10.0, 1.1]))
    Σrad = zeros(size(Σmid))
    Σrad[1, 3] = Σrad[3, 1] = 0.3
    Σ = BallMatrix(Σmid, Σrad)

    result = BallArithmetic.miyajima_vbd(Σ; hermitian = true)

    @test length(result.clusters) == 2
    @test result.clusters[1] == 1:2
    @test result.clusters[2] == 3:3
    @test result.eigenvalues[result.clusters[1]] ≈ [1.0, 1.1]

    collatz = BallArithmetic.collatz_upper_bound_L2_opnorm(result.remainder)
    @test result.remainder_norm <= collatz
end

@testset "Miyajima VBD zero remainder" begin
    Σmid = Diagonal([4.0, 2.0, 1.0])
    Σ = BallMatrix(Matrix(Σmid))

    result = BallArithmetic.miyajima_vbd(Σ; hermitian = true)

    @test length(result.clusters) == 3
    @test all(result.clusters[i] == i:i for i in 1:3)
    @test all(iszero, result.remainder.c)
    @test all(==(0.0), result.remainder.r)
    @test result.remainder_norm == 0.0
end

@testset "Miyajima VBD complex input" begin
    Σmid = [1 + 2im  0.3 - 0.2im; -0.4 + 0.1im  1 - im]
    Σ = BallMatrix(Σmid)

    result = BallArithmetic.miyajima_vbd(Σ; hermitian = true)

    @test length(result.cluster_intervals) == size(Σmid, 1)
    @test all(isreal, mid.(result.cluster_intervals))
    covered = sort!(reduce(vcat, [collect(cluster) for cluster in result.clusters]))
    @test covered == collect(1:size(Σmid, 1))
end

@testset "Miyajima VBD general matrix" begin
    Amid = [1.0 2.0; -3.0 4.0]
    Arad = fill(1.0e-8, size(Amid))
    A = BallMatrix(Amid, Arad)

    result = BallArithmetic.miyajima_vbd(A)

    @test size(result.transformed) == size(A)
    @test length(result.clusters) >= 1
    @test length(result.cluster_intervals) == size(A, 1)
    @test eltype(result.cluster_intervals) <: BallArithmetic.Ball{Float64, ComplexF64}
    covered = sort!(reduce(vcat, [collect(cluster) for cluster in result.clusters]))
    @test covered == collect(1:size(A, 1))
end
