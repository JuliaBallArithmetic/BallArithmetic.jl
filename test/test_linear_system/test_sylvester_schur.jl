using BallArithmetic
using Test
using LinearAlgebra
using BallArithmetic: mid, rad

# Schur-fallback Sylvester enclosure (salvaged from the Sylvester-schur branch).
# Solves A*X + X*B = C with a verified, block-by-block Schur back-substitution that
# also works when A or B is defective (where the eigenvector method is inapplicable).

# exact solution via vectorization (small n only)
function _exact_sylvester(A, B, C)
    m, n = size(A, 1), size(B, 1)
    K = kron(Matrix(I, n, n), A) + kron(transpose(B), Matrix(I, m, m))
    return reshape(K \ vec(C), m, n)
end

_encloses(BM, X; atol = 1e-12) = maximum(abs.(X .- mid(BM)) .- (rad(BM) .+ atol)) <= 0

@testset "Schur-fallback Sylvester enclosure" begin
    B = [5.0 0.0; 1.0 4.0]
    C = [1.0 2.0; 3.0 4.0]

    @testset "diagonalizable A — encloses, tight" begin
        A = [3.0 1.0; 0.0 -2.0]
        X = _exact_sylvester(A, B, C)
        for r in (verified_sylvester_enclosure(A, B, C),
                  schur_sylvester_miyajima_enclosure(A, B, C))
            @test _encloses(r, X)
        end
        @test maximum(rad(verified_sylvester_enclosure(A, B, C))) < 1e-10
    end

    @testset "defective A (Jordan) — eigenvector method fails, Schur fallback encloses" begin
        A = [2.0 1.0; 0.0 2.0]            # single Jordan block, not diagonalizable
        X = _exact_sylvester(A, B, C)
        r = verified_sylvester_enclosure(A, B, C)
        @test _encloses(r, X)            # rigorous (loose: defective-block sensitivity)
        @test _encloses(schur_sylvester_miyajima_enclosure(A, B, C), X)
    end

    @testset "block-order regression (reverse-order back-substitution)" begin
        # a larger triangular A with several distinct eigenvalues; the reverse-order
        # block sweep must use already-solved blocks correctly
        A = [4.0 1.0 0.5 0.2; 0.0 3.0 0.7 0.1; 0.0 0.0 2.0 0.3; 0.0 0.0 0.0 1.0]
        Bm = [6.0 0.2; 0.0 5.0]
        Cm = reshape(collect(1.0:8.0), 4, 2)
        X = _exact_sylvester(A, Bm, Cm)
        @test _encloses(verified_sylvester_enclosure(A, Bm, Cm), X)
    end

    @testset "real-Schur 2×2 path refuses (not yet rigorous)" begin
        A = [0.0 -1.0; 1.0 0.0]          # complex eigenvalues ±i ⇒ a 2×2 real block
        @test_throws ArgumentError schur_sylvester_miyajima_enclosure(
            A, B, C; prefer_complex_schur = false)
        # but the default complex-Schur path handles it (all 1×1)
        @test _encloses(schur_sylvester_miyajima_enclosure(A, B, C),
            _exact_sylvester(A, B, C))
    end

    @testset "schur_blocks" begin
        @test BallArithmetic.schur_blocks(ComplexF64[1 2; 0 3]) == [1:1, 2:2]
        @test BallArithmetic.schur_blocks([0.0 -1.0; 1.0 0.0]) == [1:2]   # real 2×2
        @test BallArithmetic.schur_blocks([2.0 1.0; 0.0 3.0]) == [1:1, 2:2]
    end
end
