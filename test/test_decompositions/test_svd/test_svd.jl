using LinearAlgebra

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

@testset "rigorous_svd result object" begin
    mid = [3.0 0.1 0.0; 0.0 2.0 0.3; 0.0 0.0 0.5]
    rad = zeros(size(mid))
    rad[1, 1] = 1.0e-2
    A = BallMatrix(mid, rad)

    result = BallArithmetic.rigorous_svd(A)

    # `A` carries a radius, where the two entry points are documented to differ: `rigorous_svd`
    # certifies the whole ball by Theorem 7, since it also returns vector bounds and the residuals
    # E, F, G measured against that ball, while `svdbox` returns values alone and so certifies
    # mid(A) by Theorem 10 and widens by ‖rad(A)‖₂ (`_miyajima2014_thm10_weyl`), the narrower route.
    @test result.singular_values ==
          BallArithmetic.svdbox(A; method = :miyajima2014_thm7)
    # `rad` is shadowed by the local radius matrix above, hence the qualified calls
    @test maximum(BallArithmetic.rad.(BallArithmetic.svdbox(A))) <=
          maximum(BallArithmetic.rad.(result.singular_values))
    @test result.Σ isa BallMatrix
    @test result.residual isa BallMatrix
    @test result.block_diagonalisation isa BallArithmetic.SchurGershgorinResult

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
    recomputed = schur_gershgorin_enclosure(H)
    @test result.block_diagonalisation.clusters == recomputed.clusters
    @test result.block_diagonalisation.remainder_norm == recomputed.remainder_norm

    result_no_vbd = BallArithmetic.rigorous_svd(A; apply_vbd = false)

    @test result_no_vbd.block_diagonalisation === nothing
    @test result_no_vbd.singular_values == result.singular_values
    @test result_no_vbd.Σ == result.Σ
    @test result_no_vbd.residual == result.residual

    # svdbox returns the values alone and never forms a block diagonalisation, so it has no
    # apply_vbd to pass; asked for Theorem 7 by name it computes exactly what rigorous_svd's
    # MiyajimaM1 certifies with on this ball input, so the two agree there
    Σ_box = BallArithmetic.svdbox(A; method = :miyajima2014_thm7)
    @test Σ_box == result.singular_values
end

@testset "Miyajima VBD block diagonalisation" begin
    Σmid = Matrix(Diagonal([5.0, 5.0, 2.0]))
    Σrad = zeros(size(Σmid))
    Σrad[1, 3] = Σrad[2, 3] = Σrad[3, 1] = Σrad[3, 2] = 1.0e-6
    Σ = BallMatrix(Σmid, Σrad)

    result = schur_gershgorin_enclosure(Σ)

    # one cluster for the double value 5 and one for 2; their order follows the reduction
    @test length(result.clusters) == 2
    @test sort(length.(result.clusters)) == [1, 2]
    @test sort(reduce(vcat, collect.(result.clusters))) == [1, 2, 3]

    # entries between the two clusters: zero in the block diagonal part, and in the remainder
    c1, c2 = result.clusters
    for (i, j) in Iterators.flatten((Iterators.product(c1, c2), Iterators.product(c2, c1)))
        @test result.block_diagonal.c[i, j] == 0
        @test result.block_diagonal.r[i, j] == 0
        @test result.remainder.c[i, j] ≈ result.transformed.c[i, j]
    end
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

    result = schur_gershgorin_enclosure(Σ)

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

    result = schur_gershgorin_enclosure(Σ)

    @test length(result.clusters) == 3
    @test all(result.clusters[i] == i:i for i in 1:3)
    @test all(iszero, result.remainder.c)
    @test all(==(0.0), result.remainder.r)
    @test result.remainder_norm == 0.0
end

@testset "Miyajima VBD complex input" begin
    Σmid = [1 + 2im  0.3 - 0.2im; -0.4 + 0.1im  1 - im]
    Σ = BallMatrix(Σmid)

    result = schur_gershgorin_enclosure(Σ)

    @test length(result.cluster_intervals) == size(Σmid, 1)
    # the matrix is not Hermitian and its eigenvalues are not real: discs of the complex plane
    # (this test asked for real centres, with `hermitian = true`)
    @test eltype(result.cluster_intervals) == Ball{Float64, ComplexF64}
    @test all(any(abs(l - mid(b)) <= rad(b) for b in result.cluster_intervals)
    for l in eigvals(Σmid))
    covered = sort!(reduce(vcat, [collect(cluster) for cluster in result.clusters]))
    @test covered == collect(1:size(Σmid, 1))
end

@testset "Miyajima VBD general matrix" begin
    Amid = [1.0 2.0; -3.0 4.0]
    Arad = fill(1.0e-8, size(Amid))
    A = BallMatrix(Amid, Arad)

    result = BallArithmetic.schur_gershgorin_enclosure(A)

    @test size(result.transformed) == size(A)
    @test length(result.clusters) >= 1
    @test length(result.cluster_intervals) == size(A, 1)
    @test eltype(result.cluster_intervals) <: BallArithmetic.Ball{Float64, ComplexF64}
    covered = sort!(reduce(vcat, [collect(cluster) for cluster in result.clusters]))
    @test covered == collect(1:size(A, 1))
end
