@testset "Test MMul structure support" begin
    using LinearAlgebra

    import BallArithmetic: abs_preserving_structure

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

    using Test, Random

    randC(T, m, n) = rand(T, m, n) .+ im * rand(T, m, n)

    @testset "oishi_MMul: mid/rad and containment" begin
        Random.seed!(0xC0FFEE)
        setprecision(BigFloat, 256) do
            for (m, k, n) in ((3, 3, 3), (4, 5, 2), (6, 3, 4))
                F = randC(BigFloat, m, k)
                G = randC(BigFloat, k, n)

                # low-level rectangular bounds + working type
                Hrl, Hru, Hil, Hiu, T = _oishi_MMul_up_lo(F, G)

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
                B = oishi_MMul(F, G)

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
end
