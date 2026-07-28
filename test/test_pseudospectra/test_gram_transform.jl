using Test
using LinearAlgebra
using Random
using BallArithmetic

const CS = BallArithmetic.CertifScripts

# Reference factor of exactly the matrix `verified_cholesky` factors, namely the
# symmetrised (A + A*)/2, computed in high precision.
function _reference_factor(G)
    BT = eltype(G) <: Complex ? Complex{BigFloat} : BigFloat
    Gb = BT.(G)
    return cholesky(Hermitian((Gb + Gb') / 2)).U
end

_encloses(ball, exact) = all(abs.(convert.(eltype(exact), ball.c) .- exact) .<=
                             BigFloat.(ball.r))

@testset "Gram transform" begin
    old_prec = precision(BigFloat)
    setprecision(BigFloat, 256)
    try

    @testset "diagonal Gram: exact factor" begin
        G = collect(Diagonal([1.0, 4.0, 9.0]))
        gt = CS.gram_transform(G)

        @test gt.source === :diagonal
        # √1, √4, √9 are exact, so the enclosure is a point
        @test all(iszero, gt.factor.r)
        @test gt.factor.c ≈ Diagonal([1.0, 2.0, 3.0])
        @test gt.gram_residual == 0
        # κ₂(G) = 9/1
        @test gt.cond_gram >= 9
        @test gt.cond_gram < 9 + 1e-10

        A = [0.0 1.0 0.0; 0.0 0.0 1.0; 0.5 0.0 0.0]
        At = CS.apply_gram_transform(BallMatrix(A), gt)
        L = Diagonal([1.0, 2.0, 3.0])
        @test _encloses(At, BigFloat.(L * A * inv(L)))
    end

    @testset "general Gram: encloses the exact conjugation" begin
        Random.seed!(20260727)
        n = 10
        B = randn(n, n)
        G = B'B + n * I
        A = randn(ComplexF64, n, n) / sqrt(n)

        gt = CS.gram_transform(G)
        @test gt.source === :cholesky
        @test gt.gram_residual < 1e-25          # BigFloat Cholesky enclosure

        # the factor enclosure contains the exact Cholesky factor
        Ltrue = _reference_factor(G)
        @test _encloses(gt.factor, Ltrue)
        @test _encloses(gt.factor_inv, inv(Ltrue))

        At = CS.apply_gram_transform(BallMatrix(A), gt)
        @test eltype(At.r) === Float64          # rounded outward to A's type
        exact = Ltrue * Complex{BigFloat}.(A) * inv(Ltrue)
        @test _encloses(At, exact)

        # similarity ⇒ the spectrum is unchanged
        @test sort(abs.(eigvals(At.c))) ≈ sort(abs.(eigvals(A))) rtol = 1e-8
    end

    @testset "the norm identity ‖R‖_G = ‖(Ã - z)⁻¹‖₂" begin
        Random.seed!(11)
        n = 8
        B = randn(n, n)
        G = B'B + n * I
        A = randn(ComplexF64, n, n) / sqrt(n)

        gt = CS.gram_transform(G)
        Ltrue = _reference_factor(G)
        At_exact = Ltrue * Complex{BigFloat}.(A) * inv(Ltrue)

        for z in (1.3 + 0.7im, 0.2 - 0.9im, -1.1 + 0.0im)
            R = inv(A - z * I)
            g_norm = Float64(opnorm(Ltrue * Complex{BigFloat}.(R) * inv(Ltrue), 2))
            t_norm = Float64(opnorm(inv(At_exact - Complex{BigFloat}(z) * I), 2))
            @test g_norm ≈ t_norm rtol = 1e-12

            # and the change of variables is sharper than inflating by √κ₂(G)
            @test t_norm <= sqrt(cond(G, 2)) * opnorm(R, 2)
        end
    end

    @testset "complex Hermitian Gram" begin
        Random.seed!(7)
        n = 6
        C = randn(ComplexF64, n, n)
        G = C'C + n * I
        gt = CS.gram_transform(G)

        @test gt.source === :cholesky
        @test _encloses(gt.factor, _reference_factor(G))
        @test gt.gram_residual < 1e-25
    end

    @testset "inverse_method variants agree" begin
        Random.seed!(3)
        n = 7
        B = randn(n, n)
        G = B'B + n * I
        Ltrue = _reference_factor(G)

        for method in (:backsub, :verify)
            gt = CS.gram_transform(G; inverse_method = method)
            @test _encloses(gt.factor_inv, inv(Ltrue))
        end

        # the uniform-inflation route is the looser of the two
        tight = CS.gram_transform(G; inverse_method = :backsub)
        loose = CS.gram_transform(G; inverse_method = :verify)
        @test maximum(tight.factor_inv.r) <= maximum(loose.factor_inv.r)
    end

    @testset "user-supplied factor defines the norm" begin
        L = [2.0 1.0; 0.0 3.0]
        gt = CS.gram_transform(nothing; factor = L)

        @test gt.source === :user_factor
        @test gt.gram_residual === nothing
        @test gt.factor.c == L

        A = [0.0 1.0; -0.5 0.0]
        At = CS.apply_gram_transform(BallMatrix(A), gt)
        @test _encloses(At, BigFloat.(L) * BigFloat.(A) * inv(BigFloat.(L)))
    end

    @testset "supplied inverse is verified, not trusted" begin
        Random.seed!(5)
        n = 6
        B = randn(n, n)
        G = B'B + n * I

        chol = BallArithmetic.verified_cholesky(G)
        L = chol.G
        good_inv = inv(Matrix(L.c))
        gt = CS.gram_transform(G; factor = L, factor_inv = good_inv)
        @test _encloses(gt.factor_inv, inv(_reference_factor(G)))

        # an inverse that is not one must be rejected rather than believed
        @test_throws ArgumentError CS.gram_transform(G; factor = L,
            factor_inv = Matrix{Float64}(I, n, n))
    end

    @testset "argument validation" begin
        G = collect(Diagonal([1.0, 2.0]))

        @test_throws ArgumentError CS.gram_transform(nothing)
        # not Hermitian
        @test_throws ArgumentError CS.gram_transform([1.0 2.0; 0.0 1.0])
        # not positive definite
        @test_throws ArgumentError CS.gram_transform(collect(Diagonal([1.0, -1.0])))
        @test_throws ArgumentError CS.gram_transform(zeros(2, 2))
        # not square
        @test_throws DimensionMismatch CS.gram_transform(ones(2, 3))
        # factor/gram size clash
        @test_throws DimensionMismatch CS.gram_transform(G;
            factor = Matrix{Float64}(I, 3, 3))
        # unknown inverse method
        @test_throws ArgumentError CS.gram_transform(G; inverse_method = :nope)

        # transform applied to a matrix of the wrong size
        gt = CS.gram_transform(G)
        @test_throws DimensionMismatch CS.apply_gram_transform(
            BallMatrix(zeros(3, 3)), gt)
    end

    @testset "certification wiring" begin
        Random.seed!(31)
        n = 6
        B = randn(n, n)
        G = B'B + n * I
        A = BallMatrix(randn(ComplexF64, n, n) / sqrt(n))
        circle = CS.CertificationCircle(0.0, 0.6; samples = 8)

        plain = CS.run_certification(A, circle; η = 0.9, check_interval = 4,
            log_io = IOBuffer())
        @test plain.gram === nothing

        gram = CS.run_certification(A, circle; gram = G, η = 0.9,
            check_interval = 4, log_io = IOBuffer())
        @test gram.gram isa CS.GramTransform
        @test gram.gram.source === :cholesky
        @test !isempty(gram.certification_log)
        @test gram.minimum_singular_value > 0

        # certifying the transformed matrix by hand must give the same thing
        gt = CS.gram_transform(G)
        manual = CS.run_certification(CS.apply_gram_transform(A, gt), circle;
            η = 0.9, check_interval = 4, log_io = IOBuffer())
        @test gram.resolvent_original ≈ manual.resolvent_original rtol = 1e-8

        # schur_data cannot be combined with gram: it would belong to Ã, not A
        sd = CS.compute_schur_and_error(A)
        @test_throws ArgumentError CS.run_certification(A, circle; gram = G,
            schur_data = sd, log_io = IOBuffer())
        # an inverse without a factor is a usage error
        @test_throws ArgumentError CS.run_certification(A, circle;
            gram_factor_inv = Matrix{Float64}(I, n, n), log_io = IOBuffer())
    end

    finally
        setprecision(BigFloat, old_prec)
    end
end
