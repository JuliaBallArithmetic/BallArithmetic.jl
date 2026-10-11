# Tests for src/decompositions/verified_takagi.jl

using Test
using LinearAlgebra
using BallArithmetic

@testset "Verified Takagi Decomposition" begin

    @testset "VerifiedTakagiResult struct" begin
        n = 2
        setprecision(BigFloat, 256) do
            U_mid = zeros(Complex{BigFloat}, n, n)
            U_rad = ones(BigFloat, n, n) * 1e-10
            for i in 1:n
                U_mid[i, i] = one(Complex{BigFloat})
            end
            U = BallMatrix(U_mid, U_rad)
            Σ = [Ball(BigFloat(2.0), BigFloat(1e-10)), Ball(BigFloat(1.0), BigFloat(1e-10))]
            success = true
            residual_norm = BigFloat(1e-50)

            result = BallArithmetic.VerifiedTakagiResult(U, Σ, success, residual_norm)

            @test result.success
            @test result.residual_norm == residual_norm
            @test length(result.Σ) == n
        end
    end

    @testset "verified_takagi with real_compound method" begin
        n = 3
        # Create a complex symmetric matrix
        Q = Matrix(qr(randn(ComplexF64, n, n)).Q)
        Σ_true = [3.0, 2.0, 1.0]
        A = Q * Diagonal(Σ_true) * transpose(Q)

        # Verify it's complex symmetric (not Hermitian)
        @test norm(A - transpose(A)) < 1e-12

        result = verified_takagi(A)

        @test result.success
        @test result.residual_norm < 1e-8

        # Verify reconstruction
        U_mid = mid(result.U)
        Σ_mid = mid.(result.Σ)
        A_reconstructed = U_mid * Diagonal(Σ_mid) * transpose(U_mid)
        @test norm(A_reconstructed - A) / norm(A) < 1e-8
    end

    @testset "verified_takagi rejects non-symmetric" begin
        n = 3
        A = randn(ComplexF64, n, n)  # Not symmetric

        # Should warn about non-symmetry
        result = @test_logs (:warn, r"not complex symmetric") match_mode = :any verified_takagi(A)
        # Still returns a result (it symmetrizes)
        @test result isa BallArithmetic.VerifiedTakagiResult
    end

    @testset "verified_takagi has one method" begin
        # the SVD-based variants were removed; the keyword went with them
        A = [1.0+0im 0.5+0im; 0.5+0im 1.0+0im]
        @test_throws MethodError verified_takagi(A; method=:svd)
    end

    @testset "verified_takagi with real symmetric" begin
        # Real symmetric matrix treated as complex symmetric
        n = 4
        B = randn(n, n)
        A_real = B + B'
        A = Complex.(A_real)

        result = verified_takagi(A)

        @test result.success

        # For real symmetric, Takagi values are |eigenvalues|
        eigs = eigvals(Symmetric(A_real))
        Σ_mid = sort(mid.(result.Σ), rev=true)
        eigs_abs = sort(abs.(eigs), rev=true)
        @test maximum(abs.(Σ_mid - eigs_abs)) < 1e-8
    end

    @testset "verified_takagi with diagonal matrix" begin
        D = Diagonal(ComplexF64.([3.0, 2.0im, 1.0+1.0im]))
        A = Matrix(D)

        result = verified_takagi(A)

        @test result.success

        # Takagi values should be |diagonal entries|
        expected_Σ = sort(abs.(diag(D)), rev=true)
        Σ_mid = sort(mid.(result.Σ), rev=true)
        @test maximum(abs.(Σ_mid - expected_Σ)) < 1e-8
    end

    @testset "verified_takagi with identity matrix" begin
        n = 3
        A = Complex.(Matrix(1.0I, n, n))

        # a multiple singular value: the decomposition is not unique and nothing is certified
        result = verified_takagi(A)

        @test !result.success
        @test all(isinf, rad(result.U))
        @test all(b -> isinf(rad(b)), result.Σ)
    end

    @testset "verified_takagi precision setting" begin
        n = 2
        Q = Matrix(qr(randn(ComplexF64, n, n)).Q)
        Σ_true = [2.0, 1.0]
        A = Q * Diagonal(Σ_true) * transpose(Q)

        # Low precision
        result_low = verified_takagi(A; precision_bits=64)
        @test result_low.success

        # High precision
        result_high = verified_takagi(A; precision_bits=512)
        @test result_high.success

        # Both should give valid results
        @test result_low.residual_norm < 1e-6
        @test result_high.residual_norm < 1e-6
    end

    @testset "verified_takagi non-square matrix" begin
        # Should fail for non-square
        A = randn(ComplexF64, 3, 4)

        @test_throws DimensionMismatch verified_takagi(A)
    end

    @testset "the factors computed at higher precision lie in the enclosure" begin
        for n in (3, 8), (kw, tol) in (((; use_bigfloat = false), 1e-9), ((;), 1e-60))
            B = randn(ComplexF64, n, n)
            A = B + transpose(B)
            r = verified_takagi(A; kw...)
            @test r.success
            @test r.residual_norm < tol
            setprecision(BigFloat, 1024) do
                Ab = big.(A)
                # the singular values, and A ū = σ u for each column u of the midpoint, up to
                # the radius: |A conj(u) − σ u| ≤ (‖A‖ + σ) √n ρ for a column within ρ entrywise
                σ = sqrt.(sort(real.(eigvals(Hermitian(Ab' * Ab))); rev = true))
                @test all(abs.(σ .- big.(mid.(r.Σ))) .<= big.(rad.(r.Σ)))
                U, ρ = big.(mid(r.U)), big.(rad(r.U))
                for k in 1:n
                    defect = norm(Ab * conj.(U[:, k]) - σ[k] * U[:, k])
                    @test defect <= (opnorm(Ab, 2) + σ[k]) * sqrt(big(n)) * maximum(ρ[:, k]) * 2
                end
            end
        end
    end

    @testset "the symmetric part of a non-symmetric input" begin
        A = [2.0+1im 0.5; 0.25 1.0-1im]
        r = @test_logs (:warn,) verified_takagi(A; use_bigfloat = false)
        @test r.success
        S = (A + transpose(A)) / 2
        @test maximum(abs.(svdvals(S) .- mid.(r.Σ))) < 1e-12
    end

    @testset "verified_takagi unitarity of U" begin
        n = 4
        Q = Matrix(qr(randn(ComplexF64, n, n)).Q)
        Σ_true = [4.0, 3.0, 2.0, 1.0]
        A = Q * Diagonal(Σ_true) * transpose(Q)

        result = verified_takagi(A)

        @test result.success

        # U should be unitary: U' * U ≈ I
        U_mid = mid(result.U)
        @test norm(U_mid' * U_mid - I) < 1e-8
    end

    @testset "verified_takagi non-negative singular values" begin
        n = 3
        Q = Matrix(qr(randn(ComplexF64, n, n)).Q)
        Σ_true = [3.0, 2.0, 1.0]
        A = Q * Diagonal(Σ_true) * transpose(Q)

        result = verified_takagi(A)

        # All Σ should be non-negative
        Σ_mid = mid.(result.Σ)
        @test all(Σ_mid .>= -1e-10)
    end
end
