using Test
using LinearAlgebra
using Random

# ---------- helpers ----------

# exact solution via vectorization (OK for small n)
function exact_sylvester(A, B, C)
    m, n = size(A, 1), size(B, 1)
    K = kron(Matrix(I, n, n), A) + kron(transpose(B), Matrix(I, m, m))
    return reshape(K \ vec(C), m, n)
end

# try to extract mid/rad from a BallMatrix in a couple of common ways
# check entrywise enclosure: |X - mid| ≤ rad + atol
function assert_encloses(BM, X; atol = 1e-12)
    M = mid(BM)
    R = rad(BM)
    @test size(M) == size(X) == size(R)
    viol = maximum(abs.(X .- M) .- (R .+ atol))
    @test viol ≤ 0
end

# random well-conditioned normal matrix with separated spectrum
function rand_normal_with_spectrum(n; a0 = 1.0, step = 0.7)
    Q, _ = qr(randn(ComplexF64, n, n))  # unitary
    λ = a0 .+ step .* (0:(n - 1))
    return Matrix(Q) * diagm(0 => λ) * Matrix(Q')
end

# random “ill-conditioned diagonalizable” (to nudge eigen route to struggle occasionally)
function rand_bad_diag(n; cond = 1e8, gap = 0.4)
    # diagonalizable: A = S Λ S⁻¹ with S badly conditioned
    S = UpperTriangular(randn(ComplexF64, n, n))
    for i in 1:n
        S[i, i] = 1.0 + (cond)^(i / n) * 1e-16  # skew pivots
    end
    Λ = diagm(0 => (1.0 .+ gap .* (0:(n - 1))))
    A = Matrix(S) * Λ * inv(Matrix(S))
    return A
end

# ---------- tests ----------

@testset "Miyajima Sylvester enclosure – fast path, fallback, triangular" begin
    Random.seed!(0x5eed)

    # 1) Fast eigenvector-based certificate on a well-conditioned case
    @testset "fast eigenvector route (well-separated normal A,B)" begin
        n = 5
        A = rand_normal_with_spectrum(n; a0 = 1.0, step = 0.5)
        B = rand_normal_with_spectrum(n; a0 = 0.8, step = 0.6)
        C = randn(ComplexF64, n, n)
        Xtrue = exact_sylvester(A, B, C)

        # midpoint from Schur (cheap & OK)
        X̃ = schur_sylvester_midpoint(A, B, C)

        BM = sylvester_miyajima_enclosure(A, B, C, X̃)
        assert_encloses(BM, Xtrue; atol = 1e-12)
    end

    # 2) Schur fallback explicitly (works even if eigen route would pass)
    @testset "explicit Schur fallback enclosure" begin
        n = 7
        A = rand_bad_diag(n; cond = 1e6, gap = 0.35)
        B = rand_bad_diag(n; cond = 1e6, gap = 0.45)
        C = randn(ComplexF64, n, n)
        Xtrue = exact_sylvester(A, B, C)

        BM = schur_sylvester_miyajima_enclosure(A, B, C; prefer_complex_schur = true)
        assert_encloses(BM, Xtrue; atol = 1e-12)
    end

    @testset "Schur fallback respects block coupling order" begin
        A = Diagonal([2.0, 3.0, 5.0]) |> Matrix
        B = [1.0 1.0 0.0; 0.0 4.0 2.0; 0.0 0.0 6.0]
        C = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        Xtrue = exact_sylvester(A, B, C)

        Xmid_complex = schur_sylvester_midpoint(A, B, C; prefer_complex_schur = true)
        @test norm(Xmid_complex - Xtrue) ≤ 1e-13
        BM_complex = schur_sylvester_miyajima_enclosure(A, B, C; prefer_complex_schur = true)
        assert_encloses(BM_complex, Xtrue; atol = 1e-13)

        Xmid_real = schur_sylvester_midpoint(A, B, C; prefer_complex_schur = false)
        @test norm(Xmid_real - Xtrue) ≤ 1e-13
        BM_real = schur_sylvester_miyajima_enclosure(A, B, C; prefer_complex_schur = false)
        assert_encloses(BM_real, Xtrue; atol = 1e-13)
    end

    # 3) Wrapper: try fast, else fallback (we only assert correctness)
    @testset "wrapper: verified_sylvester_enclosure" begin
        n = 6
        A = rand_bad_diag(n; cond = 1e10, gap = 0.3)  # a bit nastier
        B = rand_normal_with_spectrum(n; a0 = 1.3, step = 0.55)
        C = randn(ComplexF64, n, n)
        Xtrue = exact_sylvester(A, B, C)

        BM = verified_sylvester_enclosure(
            A, B, C; X̃ = nothing, prefer_complex_schur = true)
        assert_encloses(BM, Xtrue; atol = 1e-12)
    end

    # 4) Near-resonance but solvable (small min |λ_i(A)+λ_j(B)|)
    @testset "near resonance, solvable – Schur path stays robust" begin
        n = 5
        Qa, Ta = schur(randn(ComplexF64, n, n))
        A = Qa * Ta * Qa'
        # choose B ≈ -A + δI in the same unitary basis ⇒ sums ≈ δ on-diagonal
        δ = 1e-1
        Qb, Tb = schur(randn(ComplexF64, n, n))
        B = Qb * (-Ta + δ * I) * Qb'
        C = randn(ComplexF64, n, n)
        Xtrue = exact_sylvester(A, B, C)

        BM = schur_sylvester_miyajima_enclosure(A, B, C; prefer_complex_schur = true)
        assert_encloses(BM, Xtrue; atol = 1e-11)  # slightly looser atol
    end

    # 5) Triangular helper – build T and check Y₂ enclosure
    @testset "triangular helper" begin
        n = 6
        k = 3
        T = triu(randn(ComplexF64, n, n))
        # define the sub Sylvester A=T22', B=-T11', C=T12'
        T11 = T[1:k, 1:k]
        T22 = T[(k + 1):end, (k + 1):end]
        T12 = T[1:k, (k + 1):end]
        A = T22'
        B = -T11'
        C = T12'
        Ytrue = exact_sylvester(A, B, C)

        BM = triangular_sylvester_miyajima_enclosure(T, k)
        assert_encloses(BM, Ytrue; atol = 1e-12)
    end

    # 6) Shift invariance check: A→A+σI, B→B−σI should have same solution
    @testset "shift invariance" begin
        n = 5
        A = rand_normal_with_spectrum(n; a0 = 0.5, step = 0.7)
        B = rand_normal_with_spectrum(n; a0 = 0.9, step = 0.6)
        C = randn(ComplexF64, n, n)
        σ = 0.12345
        Xtrue = exact_sylvester(A, B, C)
        Xtrue_shift = exact_sylvester(A + σ * I, B - σ * I, C)
        @test norm(Xtrue - Xtrue_shift) ≤ 1e-12

        BM = verified_sylvester_enclosure(A, B, C)
        BMσ = verified_sylvester_enclosure(A + σ * I, B - σ * I, C)
        assert_encloses(BM, Xtrue; atol = 1e-11)
        assert_encloses(BMσ, Xtrue; atol = 1e-11)
    end
end
