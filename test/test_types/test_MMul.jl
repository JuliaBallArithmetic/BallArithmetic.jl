using LinearAlgebra
using Random
using Test

import BallArithmetic: abs_preserving_structure

@testset "Test MMul structure support" begin

    @testset "abs_preserving_structure" begin
        # AbstractTriangular inputs preserve their type and zeroed structure
        upper_data = [1.0 -2.0 3.0; -4.0 5.0 -6.0; 7.0 -8.0 9.0]
        lower_data = copy(upper_data)
        UA = UpperTriangular(upper_data)
        LA = LowerTriangular(lower_data)

        abs_UA = abs_preserving_structure(UA)
        abs_LA = abs_preserving_structure(LA)

        @test abs_UA isa UpperTriangular
        @test abs_LA isa LowerTriangular
        @test Matrix(abs_UA) == abs.(Matrix(UA))
        @test Matrix(abs_LA) == abs.(Matrix(LA))
        @test istriu(Matrix(abs_UA))
        @test istril(Matrix(abs_LA))

        # Symmetric and Hermitian parents are preserved
        S = Symmetric([1.0 -2.0; -2.0 3.0], :U)
        H = Hermitian([1.0 2.0+3.0im; 2.0-3.0im 4.0], :U)

        abs_S = abs_preserving_structure(S)
        abs_H = abs_preserving_structure(H)

        @test abs_S isa Symmetric
        @test abs_H isa Hermitian
        @test abs_S.uplo == S.uplo
        @test abs_H.uplo == H.uplo
        @test Matrix(abs_S) == abs.(Matrix(S))
        @test Matrix(abs_H) == abs.(Matrix(H))

        # Diagonal, adjoint and transpose fall back to dense absolute values while
        # maintaining their wrappers
        D = Diagonal([-1.0, 2.0, -3.0])
        abs_D = abs_preserving_structure(D)
        @test abs_D isa Diagonal
        @test diag(abs_D) == abs.(diag(D))

        dense = [1.0 -2.0im; -3.0im 4.0]
        adj = adjoint(dense)
        trans = transpose(dense)

        abs_adj = abs_preserving_structure(adj)
        abs_trans = abs_preserving_structure(trans)

        @test abs_adj == adjoint(abs.(dense))
        @test abs_trans == transpose(abs.(dense))

        # Generic matrices use component-wise absolute values
        @test abs_preserving_structure(dense) == abs.(dense)
    end

    @testset "MMul structured operands" begin
        diag_vals_A = rand(4)
        diag_vals_B = rand(4)
        rad_vals_A = rand(4)
        rad_vals_B = rand(4)
        tol = 10 * eps(Float64)

        Ad = BallMatrix(Diagonal(diag_vals_A), Diagonal(rad_vals_A))
        Bd = BallMatrix(Diagonal(diag_vals_B), Diagonal(rad_vals_B))

        Cd_struct = BallArithmetic.MMul4(Ad, Bd)
        Cd_dense = BallArithmetic.MMul4(
            BallMatrix(Matrix(Diagonal(diag_vals_A)), Matrix(Diagonal(rad_vals_A))),
            BallMatrix(Matrix(Diagonal(diag_vals_B)), Matrix(Diagonal(rad_vals_B)))
        )

        @test isapprox(Matrix(Cd_struct.c), Matrix(Cd_dense.c); rtol = tol, atol = 0)
        @test isapprox(Matrix(Cd_struct.r), Matrix(Cd_dense.r); rtol = tol, atol = 0)
        Cd_mul = Ad * Bd
        @test isapprox(Matrix(Cd_mul.c), Matrix(Cd_struct.c); rtol = tol, atol = 0)
        @test isapprox(Matrix(Cd_mul.r), Matrix(Cd_struct.r); rtol = tol, atol = 0)

        UA = UpperTriangular(rand(4, 4))
        UB = UpperTriangular(rand(4, 4))
        rUA = UpperTriangular(rand(4, 4))
        rUB = UpperTriangular(rand(4, 4))

        Atri = BallMatrix(UA, rUA)
        Btri = BallMatrix(UB, rUB)

        Ctri_struct = BallArithmetic.MMul4(Atri, Btri)
        Ctri_dense = BallArithmetic.MMul4(
            BallMatrix(Matrix(UA), Matrix(rUA)),
            BallMatrix(Matrix(UB), Matrix(rUB))
        )

        @test isapprox(Matrix(Ctri_struct.c), Matrix(Ctri_dense.c); rtol = tol, atol = 0)
        @test isapprox(Matrix(Ctri_struct.r), Matrix(Ctri_dense.r); rtol = tol, atol = 0)
        Ctri_mul = Atri * Btri
        @test isapprox(Matrix(Ctri_mul.c), Matrix(Ctri_struct.c); rtol = tol, atol = 0)
        @test isapprox(Matrix(Ctri_mul.r), Matrix(Ctri_struct.r); rtol = tol, atol = 0)

        Cmix_left = BallArithmetic.MMul4(UA, Btri)
        Cmix_left_dense = BallArithmetic.MMul4(
            Matrix(UA),
            BallMatrix(Matrix(UB), Matrix(rUB))
        )
        @test isapprox(Matrix(Cmix_left.c), Matrix(Cmix_left_dense.c); rtol = tol, atol = 0)
        @test isapprox(Matrix(Cmix_left.r), Matrix(Cmix_left_dense.r); rtol = tol, atol = 0)

        Cmix_right = BallArithmetic.MMul4(Atri, UB)
        Cmix_right_dense = BallArithmetic.MMul4(
            BallMatrix(Matrix(UA), Matrix(rUA)),
            Matrix(UB)
        )
        @test isapprox(
            Matrix(Cmix_right.c), Matrix(Cmix_right_dense.c); rtol = tol, atol = 0)
        @test isapprox(
            Matrix(Cmix_right.r), Matrix(Cmix_right_dense.r); rtol = tol, atol = 0)
    end

    sampleR(T, m, n; scale = 1, shift = 0) = T[scale * (i - j) + shift / (i + j) for i in 1:m, j in 1:n]

    sampleC(T, m, n; re_scale = 1, im_scale = 2, shift = 0) = Complex{T}[
        T(re_scale * (i + 2j) + shift) / T(3) + im * T(im_scale * (2i - j) + 2shift) / T(5)
        for i in 1:m, j in 1:n
    ]

    @testset "oishi_MMul: mid/rad and containment" begin
        setprecision(BigFloat, 256) do
            for (m, k, n) in ((3, 3, 3), (4, 5, 2), (6, 3, 4))
                F = sampleC(BigFloat, m, k; re_scale = 1, im_scale = 2, shift = 1)
                G = sampleC(BigFloat, k, n; re_scale = 3, im_scale = 5, shift = 2)

                # low-level rectangular bounds + working type
                Hrl, Hru, Hil, Hiu, T = BallArithmetic._oishi_MMul_up_lo(F, G)

                # manual center/radius with directed rounding
                half = T(0.5)
                Rc, Ic = setrounding(T, RoundNearest) do
                    (Hru .+ Hrl) .* half, (Hiu .+ Hil) .* half
                end
                Hr = setrounding(T, RoundUp) do
                    Rr = (Hru .- Hrl) .* half
                    Ir = (Hiu .- Hil) .* half
                    sqrt.(Rr .^ 2 .+ Ir .^ 2)
                end
                Hc = complex.(Rc, Ic)

                # wrapper
                B = BallArithmetic.oishi_MMul(F, G)

                # 1) mid/rad match our manual construction
                @test mid(B) == Hc
                @test rad(B) == Hr

                # 2) containment of the exact product
                P = Complex{T}.(F) * Complex{T}.(G)
                @test all(abs.(P .- mid(B)) .<= rad(B))

                # 3) types and sizes
                @test size(mid(B)) == (m, n)
                @test size(rad(B)) == (m, n)
                @test eltype(mid(B)) <: Complex{T}
                @test eltype(rad(B)) <: T
            end
        end
    end

    @testset "Miyajima auxiliary products" begin
        setprecision(BigFloat, 256) do
            # Dimensions: J is (m, k), H is (k, n)
            # _ccrprod computes J * H which requires k (inner dims match)
            # _ccrprod_prime computes H * J which requires n == m
            # So we use test cases where n == m for the _prime functions to work
            for (m, k, n) in ((2, 3, 2), (3, 2, 3))
                J = sampleC(BigFloat, m, k; re_scale = 2, im_scale = 3, shift = 1)
                Hc = sampleC(BigFloat, k, n; re_scale = 1, im_scale = 2, shift = -1)
                Hr = BigFloat.(abs.(sampleR(Float64, k, n; scale = 1))) ./ BigFloat(4)

                Krl, Kru, Kil, Kiu, T = BallArithmetic._ccrprod(J, Hc, Hr)
                Brect, Tccr = BallArithmetic._ccr(Krl, Kru, Kil, Kiu)
                @test Tccr == T
                Brect_direct = BallArithmetic._ccr(Krl, Kru, Kil, Kiu, T)
                Jball = BallMatrix(J)
                Hball = BallMatrix(Hc, Hr)
                prod_ball = Jball * Hball

                # _ccr + _ccrprod uses rectangular bounds converted to balls,
                # while BallMatrix multiplication uses MMul4 with different radius formulas.
                # Both are valid enclosures but may differ in radius tightness.
                # Test that midpoints are approximately equal (rounding differences)
                # and that both enclosures are valid (contain sampled points).
                @test mid(Brect) ≈ mid(prod_ball) rtol=1e-10
                @test mid(Brect_direct) == mid(Brect)
                @test rad(Brect_direct) == rad(Brect)

                Krlp, Krup, Kilp, Kiup, Tp = BallArithmetic._ccrprod_prime(Hc, Hr, J)
                @test Tp == T
                Brect_prime, Tccr_prime = BallArithmetic._ccr(Krlp, Krup, Kilp, Kiup)
                @test Tccr_prime == Tp
                prod_ball_prime = Hball * Jball
                @test mid(Brect_prime) ≈ mid(prod_ball_prime) rtol=1e-10

                # Test that _ccrprod bounds contain sampled true products
                for _ in 1:5
                    perturb = [Hr[i, j] == 0 ? zero(BigFloat) : Hr[i, j] *
                        (rand(BigFloat) - BigFloat(0.5)) * 2 for i in 1:k, j in 1:n]
                    thetas = 2π .* rand(BigFloat, k, n)
                    phase = [Complex{BigFloat}(cos(thetas[i, j]), sin(thetas[i, j]))
                        for i in 1:k, j in 1:n]
                    Hsamp = Hc .+ perturb .* phase
                    JS = Complex{BigFloat}.(J) * Hsamp
                    @test all(Krl .<= real(JS) .<= Kru)
                    @test all(Kil .<= imag(JS) .<= Kiu)

                    SJ = Hsamp * Complex{BigFloat}.(J)
                    @test all(Krlp .<= real(SJ) .<= Krup)
                    @test all(Kilp .<= imag(SJ) .<= Kiup)
                end

                # Test _ciprod with rectangular input - verify it also contains samples
                Hrl = real.(Hc) .- Hr
                Hru = real.(Hc) .+ Hr
                Hil = imag.(Hc) .- Hr
                Hiu = imag.(Hc) .+ Hr

                Crl, Cru, Cil, Ciu, Tci = BallArithmetic._ciprod(J, Hrl, Hru, Hil, Hiu)
                @test Tci == T

                # Verify _ciprod also contains sampled products
                for _ in 1:5
                    Hsamp_real = Hrl .+ (Hru .- Hrl) .* rand(BigFloat, k, n)
                    Hsamp_imag = Hil .+ (Hiu .- Hil) .* rand(BigFloat, k, n)
                    Hsamp = Complex{BigFloat}.(Hsamp_real, Hsamp_imag)
                    JS = Complex{BigFloat}.(J) * Hsamp
                    @test all(Crl .<= real(JS) .<= Cru)
                    @test all(Cil .<= imag(JS) .<= Ciu)
                end

                Crlp, Crup, Cilp, Ciup, Tci_prime = BallArithmetic._ciprod_prime(Hrl, Hru, Hil, Hiu, J)
                @test Tci_prime == T

                # Verify _ciprod_prime also contains sampled products
                for _ in 1:5
                    Hsamp_real = Hrl .+ (Hru .- Hrl) .* rand(BigFloat, k, n)
                    Hsamp_imag = Hil .+ (Hiu .- Hil) .* rand(BigFloat, k, n)
                    Hsamp = Complex{BigFloat}.(Hsamp_real, Hsamp_imag)
                    SJ = Hsamp * Complex{BigFloat}.(J)
                    @test all(Crlp .<= real(SJ) .<= Crup)
                    @test all(Cilp .<= imag(SJ) .<= Ciup)
                end
            end
        end
    end
end
