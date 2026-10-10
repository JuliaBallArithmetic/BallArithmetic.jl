using BallArithmetic
using Test
using LinearAlgebra
using Random
using BallArithmetic: mid, rad

# The solution of A X + X B = C at 512 bits, through the Kronecker form (small sizes only), and the
# test that an enclosure contains it: entrywise, evaluated at 512 bits, with no tolerance.
function _exact_sylvester(A, B, C)
    setprecision(BigFloat, 512) do
        Ab, Bb, Cb = Complex{BigFloat}.(A), Complex{BigFloat}.(B), Complex{BigFloat}.(C)
        m, n = size(Ab, 1), size(Bb, 1)
        K = kron(Matrix{Complex{BigFloat}}(I, n, n), Ab) + kron(transpose(Bb), Matrix{Complex{BigFloat}}(I, m, m))
        reshape(K \ vec(Cb), m, n)
    end
end
_encloses(BM, X) = setprecision(BigFloat, 512) do
    all(abs.(X .- Complex{BigFloat}.(mid(BM))) .<= BigFloat.(rad(BM)))
end

@testset "Sylvester enclosures" begin
    Random.seed!(20261010)

    @testset "Theorem 2 of Miyajima (2013): random data" begin
        for CT in (Float64, ComplexF64), (m, n) in ((5, 3), (2, 6), (1, 1)), trial in 1:3
            A = randn(CT, m, m) + 3I
            B = randn(CT, n, n) + 3I
            C = randn(CT, m, n)
            X̃ = sylvester(A, B, -C)                       # A X + X B = C
            E = sylvester_miyajima_enclosure(A, B, C, X̃)
            @test mid(E) == X̃
            @test _encloses(E, _exact_sylvester(A, B, C))
            @test maximum(rad(E)) < 1e-9
            @test _encloses(verified_sylvester_enclosure(A, B, C), _exact_sylvester(A, B, C))
        end
    end

    @testset "A poor approximate solution is still enclosed" begin
        A = [3.0 1.0 0.2; 0.1 -2.0 0.5; 0.3 0.2 5.0]
        B = [5.0 0.0; 1.0 4.0]
        C = [1.0 2.0; 3.0 4.0; -1.0 0.5]
        X = _exact_sylvester(A, B, C)
        for shift in (1e-8, 1e-3, 0.5)
            X̃ = sylvester(A, B, -C) .+ shift
            E = sylvester_miyajima_enclosure(A, B, C, X̃)
            @test _encloses(E, X)
            @test maximum(rad(E)) ≥ shift * (1 - 1e-6)
        end
    end

    @testset "Real data with complex eigenvalues; ill-conditioned eigenvectors" begin
        A = [0.0 -1.0; 1.0 0.0]
        B = [5.0 0.0; 1.0 4.0]
        C = [1.0 2.0; 3.0 4.0]
        @test _encloses(verified_sylvester_enclosure(A, B, C), _exact_sylvester(A, B, C))
        A2 = [1.0 1e4 0.0; 0.0 1.5 1e4; 0.0 0.0 2.0]
        C2 = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        @test _encloses(verified_sylvester_enclosure(A2, B, C2), _exact_sylvester(A2, B, C2))
    end

    @testset "Ball data: every member's solution is enclosed" begin
        A = [3.0 1.0; 0.2 -2.0]; B = [5.0 0.0; 1.0 4.0]; C = [1.0 2.0; 3.0 4.0]
        r = 1e-6
        bA, bB, bC = BallMatrix(A, fill(r, 2, 2)), BallMatrix(B, fill(r, 2, 2)), BallMatrix(C, fill(r, 2, 2))
        E = sylvester_miyajima_enclosure(bA, bB, bC, sylvester(A, B, -C))
        for _ in 1:40
            s() = r .* rand((-1.0, 1.0), 2, 2)
            @test _encloses(E, _exact_sylvester(A + s(), B + s(), C + s()))
        end
        @test maximum(rad(E)) < 1e-4
    end

    @testset "A defective matrix is refused" begin
        A = [2.0 1.0; 0.0 2.0]
        B = [5.0 0.0; 1.0 4.0]; C = [1.0 2.0; 3.0 4.0]
        @test_throws ArgumentError verified_sylvester_enclosure(A, B, C)
        @test !isdefined(BallArithmetic, :schur_sylvester_miyajima_enclosure)
    end

    @testset "The triangular form and its two fallbacks" begin
        for CT in (Float64, ComplexF64), (n, k) in ((7, 3), (5, 1), (6, 5))
            T = Matrix(UpperTriangular(0.4 * randn(CT, n, n))) + Diagonal(CT.(1:n))
            T11, T12, T22 = T[1:k, 1:k], T[1:k, (k + 1):n], T[(k + 1):n, (k + 1):n]
            A, B, C = Matrix(T22'), -Matrix(T11'), Matrix(T12')
            Y = _exact_sylvester(A, B, C)
            @test _encloses(triangular_sylvester_miyajima_enclosure(T, k), Y)
            @test _encloses(triangular_sylvester_miyajima_enclosure(BallMatrix(T), k), Y)
            Ỹ = BallArithmetic._sylvester_triangular_columns(A, B, C)
            bA, bB, bC = BallMatrix(A), BallMatrix(B), BallMatrix(C)
            @test _encloses(BallArithmetic._sylvester_residual_ball(bA, bB, bC, Ỹ), Y)
            @test _encloses(BallArithmetic._sylvester_triangular_direct_ball(bA, bB, bC), Y)
            @test _encloses(BallArithmetic._sylvester_triangular_direct_ball(A, B, C), Y)
        end
    end

    @testset "The triangular form with a ball: members are enclosed, by the three routes" begin
        n, k, r = 5, 2, 1e-7
        T = Matrix(UpperTriangular(0.4 * randn(n, n))) + Diagonal(Float64.(1:n))
        Tr = Matrix(UpperTriangular(fill(r, n, n)))
        bT = BallMatrix(T, Tr)
        E = triangular_sylvester_miyajima_enclosure(bT, k)
        blocks(M) = (BallMatrix(Matrix(M[(k + 1):n, (k + 1):n]'), Matrix(Tr[(k + 1):n, (k + 1):n]')),
            -BallMatrix(Matrix(M[1:k, 1:k]'), Matrix(Tr[1:k, 1:k]')),
            BallMatrix(Matrix(M[1:k, (k + 1):n]'), Matrix(Tr[1:k, (k + 1):n]')))
        bA, bB, bC = blocks(T)
        Ỹ = BallArithmetic._sylvester_triangular_columns(mid(bA), mid(bB), mid(bC))
        E_res = BallArithmetic._sylvester_residual_ball(bA, bB, bC, Ỹ)
        E_dir = BallArithmetic._sylvester_triangular_direct_ball(bA, bB, bC)
        for _ in 1:40
            Tv = T + Tr .* rand((-1.0, 1.0), n, n)
            Y = _exact_sylvester(Matrix(Tv[(k + 1):n, (k + 1):n]'), -Matrix(Tv[1:k, 1:k]'),
                Matrix(Tv[1:k, (k + 1):n]'))
            @test _encloses(E, Y)
            @test _encloses(E_res, Y)
            @test _encloses(E_dir, Y)
        end
        @test_throws ArgumentError triangular_sylvester_miyajima_enclosure(T, 0)
        @test_throws ArgumentError triangular_sylvester_miyajima_enclosure([1.0 0.0; 1.0 2.0], 1)
        @test_throws ArgumentError triangular_sylvester_miyajima_enclosure(T, 2; sylvester_fallback = :other)
    end
end
