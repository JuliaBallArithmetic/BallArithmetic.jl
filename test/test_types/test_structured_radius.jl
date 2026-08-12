using SparseArrays

@testset "Structure-preserving radius" begin
    # The one-argument constructors route through `rad`, which must return an
    # all-zero radius laid out like the midpoints. Before this was fixed `rad`
    # always produced a dense `Matrix`, so a sparse midpoint matrix carried a
    # dense radius costing orders of magnitude more memory than the midpoints.

    @testset "structured midpoints keep their storage type" begin
        for (M, R) in [(Diagonal(rand(4)), Diagonal),
            (UpperTriangular(rand(4, 4)), UpperTriangular),
            (LowerTriangular(rand(4, 4)), LowerTriangular),
            (Symmetric(rand(4, 4)), Symmetric),
            (Hermitian(rand(ComplexF64, 4, 4)), Hermitian),
            (Tridiagonal(rand(3), rand(4), rand(3)), Tridiagonal),
            (SymTridiagonal(rand(4), rand(3)), SymTridiagonal),
            (Bidiagonal(rand(4), rand(3), :U), Bidiagonal),
            (sprand(6, 6, 0.3), SparseMatrixCSC),
            (sprand(ComplexF64, 6, 6, 0.3), SparseMatrixCSC)]
            B = BallMatrix(M)
            @test B.r isa R
            @test all(iszero, B.r)
            @test size(B.r) == size(M)
        end

        # The radius is real even when the midpoints are complex.
        @test eltype(BallMatrix(sprand(ComplexF64, 5, 5, 0.4)).r) == Float64
        @test eltype(BallMatrix(Hermitian(rand(ComplexF64, 4, 4))).r) == Float64
        @test eltype(BallMatrix(Complex{BigFloat}.(rand(ComplexF64, 3, 3))).r) == BigFloat

        # Non-float radius types follow the midpoint precision.
        @test eltype(BallMatrix(Diagonal(rand(Float32, 3))).r) == Float32
        @test eltype(BallMatrix(Diagonal(BigFloat.(rand(3)))).r) == BigFloat
    end

    @testset "unit triangular stores the radius in the non-unit type" begin
        # UnitUpperTriangular cannot represent a zero diagonal, so the radius
        # uses UpperTriangular: the unit diagonal is exactly one, radius zero.
        BU = BallMatrix(UnitUpperTriangular(rand(5, 5)))
        @test BU.r isa UpperTriangular
        @test all(iszero, BU.r)
        @test all(BU.c[i, i] == 1 for i in 1:5)
        @test all(BU.r[i, i] == 0 for i in 1:5)

        BL = BallMatrix(UnitLowerTriangular(rand(5, 5)))
        @test BL.r isa LowerTriangular
        @test all(iszero, BL.r)
    end

    @testset "lazy wrappers and views fall back to dense" begin
        # Adjoint/Transpose/SubArray have no structure worth preserving.
        @test BallMatrix(adjoint(rand(3, 3))).r isa Matrix
        @test BallMatrix(transpose(rand(3, 3))).r isa Matrix
        @test BallMatrix(view(rand(5, 5), 1:3, 1:3)).r isa Matrix
        @test BallMatrix(rand(3, 3)).r isa Matrix
    end

    @testset "vectors" begin
        @test BallVector(sprand(8, 0.4)).r isa SparseVector
        @test all(iszero, BallVector(sprand(8, 0.4)).r)
        @test BallVector(rand(4)).r isa Vector
        # Ranges are immutable; `similar` gives a plain dense vector.
        @test BallVector(1.0:4.0).r isa Vector
        @test BallVector(view(rand(6), 1:3)).r isa Vector
    end

    @testset "sparse radius does not densify" begin
        n = 400
        S = sprand(n, n, 0.005)
        B = BallMatrix(S)
        # A dense radius would need n^2 * 8 bytes; the sparse one is far smaller.
        @test Base.summarysize(B.r) < n * n * 8 / 10
    end

    @testset "structured midpoints still give rigorous enclosures" begin
        # Storing fewer radius entries must not lose any inflation: every
        # implicitly stored entry of a structured type is an exact zero (or an
        # exact one on a unit diagonal), so its radius is exactly zero.
        cases = [Diagonal(rand(6)),
            UpperTriangular(rand(6, 6)),
            LowerTriangular(rand(6, 6)),
            UnitUpperTriangular(rand(6, 6)),
            Symmetric(rand(6, 6)),
            Tridiagonal(rand(5), rand(6), rand(5)),
            SymTridiagonal(rand(6), rand(5)),
            Bidiagonal(rand(6), rand(5), :U),
            Matrix(sprand(6, 6, 0.5))]

        for M in cases
            B = BallMatrix(M)
            ref = BigFloat.(Matrix(M))

            P = B * B
            ex = ref * ref
            @test all(abs(ex[i] - P.c[i]) <= P.r[i] for i in eachindex(ex))

            S = B + B
            exs = ref + ref
            @test all(abs(exs[i] - S.c[i]) <= S.r[i] for i in eachindex(exs))

            # A rigorous norm bound must dominate the true norm.
            @test BallArithmetic.upper_bound_L2_opnorm(B) >= opnorm(Matrix(M), 2)
        end
    end

    @testset "structured times unstructured still encloses" begin
        # A Symmetric/Hermitian radius wrapper reads only one triangle; check
        # that no inflation is silently mirrored away when the other operand
        # has no symmetry.
        A = Symmetric(rand(6, 6))
        G = rand(6, 6)
        BA, BG = BallMatrix(A), BallMatrix(G)

        P = BA * BG
        ex = BigFloat.(Matrix(A)) * BigFloat.(G)
        @test all(abs(ex[i] - P.c[i]) <= P.r[i] for i in eachindex(ex))

        P2 = BG * BA
        ex2 = BigFloat.(G) * BigFloat.(Matrix(A))
        @test all(abs(ex2[i] - P2.c[i]) <= P2.r[i] for i in eachindex(ex2))

        SS = sprand(8, 8, 0.3)
        BD = BallMatrix(rand(8, 8))
        P3 = BallMatrix(SS) * BD
        ex3 = BigFloat.(Matrix(SS)) * BigFloat.(mid(BD))
        @test all(abs(ex3[i] - P3.c[i]) <= P3.r[i] for i in eachindex(ex3))
    end
end
