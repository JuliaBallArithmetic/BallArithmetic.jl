using BallArithmetic
using LinearAlgebra
using Random
using Test

# Rump and Ogita (2024), Verified Error Bounds for Matrix Decompositions.

# LU decomposition without pivoting, in the arithmetic of the input: L is m × p unit lower
# trapezoidal and U is p × n upper trapezoidal, p = min(m, n)
function _lu_nopivot(A::AbstractMatrix{S}) where {S}
    m, n = size(A)
    p = min(m, n)
    U = copy(A)
    L = zeros(S, m, p)
    for k in 1:p
        L[k, k] = one(S)
        for i in (k + 1):m
            L[i, k] = U[i, k] / U[k, k]
            for j in k:n
                U[i, j] -= L[i, k] * U[k, j]
            end
        end
    end
    return L, triu(U[1:p, :])
end

_inside(X, B) = all(abs.(X - mid(B)) .<= rad(B))
_eye(S, m, n) = Matrix{S}(I, m, n)

@testset "Rump-Ogita 2024, Algorithm 3.2: the LU factors of a perturbed identity" begin
    rng = MersenneTwister(20261011)
    for (m, n) in ((6, 6), (9, 5), (5, 9), (1, 1), (1, 4), (4, 1)),
        S in (Float64, ComplexF64), ε in (1e-10, 1e-4, 0.05), ρ in (0.0, 1e-6)

        Em = ε * randn(rng, S, m, n) / max(m, n)
        E = BallMatrix(Em, fill(ρ * ε, m, n))
        r = BallArithmetic._rumpogita2024_alg3_2(E)
        @test r !== nothing
        p = min(m, n)
        @test size(r.LE) == (m, p) && size(r.UE) == (p, n)
        @test (r.LinvE === nothing) == (m > n)
        @test (r.UinvE === nothing) == (m < n)
        @test istril(mid(r.LE), -1) && iszero(triu(rad(r.LE)))
        @test istriu(mid(r.UE)) && iszero(tril(rad(r.UE), -1))
        for trial in 1:(ρ == 0 ? 1 : 6)
            Ep = ρ == 0 ? Em : Em + ρ * ε * (2 * rand(rng, m, n) .- 1)
            L, U = setprecision(BigFloat, 512) do
                B = S <: Complex ? Complex{BigFloat} : BigFloat
                _lu_nopivot(_eye(B, m, n) + B.(Ep))
            end
            @test _inside(L - _eye(eltype(L), m, p), r.LE)
            @test _inside(U - _eye(eltype(U), p, n), r.UE)
            r.LinvE === nothing || @test _inside(inv(L) - I, r.LinvE)
            r.UinvE === nothing || @test _inside(inv(U) - I, r.UinvE)
        end
        # the radius beyond that of E is of second order in E, as (3.1) and (3.5) say
        if ρ == 0
            @test maximum(rad(r.LE); init = 0.0) <= 4 * max(m, n) * ε^2
            @test maximum(rad(r.UE); init = 0.0) <= 4 * max(m, n) * ε^2
        end
    end
    # a perturbation of norm one or more is declined
    @test BallArithmetic._rumpogita2024_alg3_2(BallMatrix(fill(0.6, 3, 3))) === nothing
    @test BallArithmetic._rumpogita2024_alg3_2(BallMatrix(zeros(3, 3), fill(0.4, 3, 3))) === nothing
    # the exact identity: the factors are the identity with radius zero
    r = BallArithmetic._rumpogita2024_alg3_2(BallMatrix(zeros(4, 4)))
    @test all(iszero, mid(r.LE)) && all(iszero, rad(r.LE)) && all(iszero, rad(r.UinvE))
end

# a matrix with singular values from 1 to 10^-k, as in the paper (gallery/randsvd)
function _randsvd(rng, n, k; S = Float64)
    σ = exp10.(range(0, -k; length = n))[randperm(rng, n)]
    return Matrix(qr(randn(rng, S, n, n)).Q) * Diagonal(σ) * Matrix(qr(randn(rng, S, n, n)).Q)
end

_relerr(B) = (m = abs.(mid(B)); r = rad(B); [r[i] / m[i] for i in eachindex(m) if m[i] > 0])

_ro_lu_square(A) = BallArithmetic._rumpogita2024_lu(A)

@testset "Rump-Ogita 2024, Section 3.2: the LU decomposition of a square matrix" begin
    rng = MersenneTwister(20261011)
    med(v) = sort(v)[(length(v) + 1) ÷ 2]
    @testset "n = $n, cond = 1e$k, $S" for (n, k) in ((8, 1), (30, 2), (30, 8), (30, 12)),
        S in (Float64, ComplexF64)

        A = _randsvd(rng, n, k; S)
        r = _ro_lu_square(BallMatrix(A))
        @test r !== nothing
        @test sort(r.p) == 1:n
        @test istril(mid(r.L)) && all(isone, diag(mid(r.L))) && iszero(triu(rad(r.L)))
        @test istriu(mid(r.U)) && iszero(tril(rad(r.U), -1))
        L, U = setprecision(BigFloat, 1024) do
            B = S <: Complex ? Complex{BigFloat} : BigFloat
            _lu_nopivot(B.(A[r.p, :]))
        end
        @test _inside(L, r.L)
        @test _inside(U, r.U)
        # the paper's Figure 3: with the products in two-fold precision the bounds are close to
        # the rounding unit for every condition number
        @test med(_relerr(r.U)) < 1e-13
        @test med(_relerr(r.L)) < 1e-12
    end
    @testset "a ball of matrices" begin
        A = _randsvd(rng, 10, 2)
        ρ = 1e-10
        r = _ro_lu_square(BallMatrix(A, fill(ρ, 10, 10)))
        @test r !== nothing
        for _ in 1:10
            Ap = A + ρ * (2 * rand(rng, 10, 10) .- 1)
            L, U = setprecision(() -> _lu_nopivot(big.(Ap[r.p, :])), BigFloat, 1024)
            @test _inside(L, r.L) && _inside(U, r.U)
        end
    end
    @testset "declined" begin
        @test _ro_lu_square(BallMatrix([1.0 2.0; 2.0 4.0])) === nothing
        @test _ro_lu_square(BallMatrix([1.0 0.0; 0.0 1.0], fill(0.6, 2, 2))) === nothing
    end
    @testset "the two-term product" begin
        for S in (Float64, ComplexF64)
            A, B = randn(rng, S, 7, 9) .* exp10.(8 .* randn(rng, 7, 9)), randn(rng, S, 9, 5)
            P1, P2, R = BallArithmetic._two_term_product_sum(((A, B, 1.0),))
            X = setprecision(BigFloat, 1024) do
                BT = S <: Complex ? Complex{BigFloat} : BigFloat
                BT.(A) * BT.(B) - BT.(P1) - BT.(P2)
            end
            @test all(abs.(X) .<= R)
            C = BallArithmetic._accurate_product_sum(((A, B, 1.0), (Matrix{S}(I, 7, 7), randn(rng, S, 7, 5), -1.0)))
            @test size(C) == (7, 5)
        end
    end
end

# an m × n matrix with singular values from 1 to 10^-k
function _randsvd(rng, m, n, k; S = Float64)
    r = min(m, n)
    σ = r == 1 ? [1.0] : exp10.(range(0, -k; length = r))[randperm(rng, r)]
    return Matrix(qr(randn(rng, S, m, m)).Q)[:, 1:r] * Diagonal(σ) * Matrix(qr(randn(rng, S, n, n)).Q)[1:r, :]
end

@testset "Rump-Ogita 2024, Sections 3.3 and 3.4: the LU decomposition of a rectangular matrix" begin
    rng = MersenneTwister(20261012)
    med(v) = sort(v)[(length(v) + 1) ÷ 2]
    @testset "$m × $n, cond = 1e$k, $S" for (m, n) in ((30, 15), (15, 30), (9, 8), (8, 9), (5, 1), (1, 5)),
        k in (2, 8, 12), S in (Float64, ComplexF64)

        A = _randsvd(rng, m, n, k; S)
        r = BallArithmetic._rumpogita2024_lu(BallMatrix(A))
        @test r !== nothing
        kk = min(m, n)
        @test size(r.L) == (m, kk) && size(r.U) == (kk, n)
        @test sort(r.p) == 1:m && sort(r.q) == 1:n
        m >= n && @test r.q == 1:n
        L, U = setprecision(BigFloat, 1024) do
            B = S <: Complex ? Complex{BigFloat} : BigFloat
            _lu_nopivot(B.(A[r.p, r.q]))
        end
        @test _inside(L, r.L)
        @test _inside(U, r.U)
        @test all(isone, diag(mid(r.L))) && iszero(triu(rad(r.L)))
        @test iszero(tril(rad(r.U), -1))
        # the paper's Figures 5 and 7: close to the rounding unit for every condition number
        if min(m, n) > 1
            @test med(_relerr(r.U)) < 1e-12
            @test med(_relerr(r.L)) < 1e-11
        end
    end
    A = _randsvd(rng, 7, 12, 3)
    ρ = 1e-11
    for B in (A, permutedims(A))
        r = BallArithmetic._rumpogita2024_lu(BallMatrix(B, fill(ρ, size(B)...)))
        @test r !== nothing
        for _ in 1:8
            Bp = B + ρ * (2 * rand(rng, size(B)...) .- 1)
            L, U = setprecision(() -> _lu_nopivot(big.(Bp[r.p, r.q])), BigFloat, 1024)
            @test _inside(L, r.L) && _inside(U, r.U)
        end
    end
    @test BallArithmetic._rumpogita2024_lu(BallMatrix([1.0 2.0 3.0; 2.0 4.0 6.0])) === nothing
end

@testset "Rump-Ogita 2024, Section 4: the Cholesky decomposition" begin
    rng = MersenneTwister(20261013)
    med(v) = sort(v)[(length(v) + 1) ÷ 2]
    function spd(n, k, S)
        Q = Matrix(qr(randn(rng, S, n, n)).Q)
        A = Q * Diagonal(exp10.(range(0, -k; length = n))) * Q'
        return (A + A') / 2
    end
    @testset "n = $n, cond = 1e$k, $S" for (n, k) in ((6, 1), (25, 2), (25, 8), (25, 13)),
        S in (Float64, ComplexF64)

        A = spd(n, k, S)
        r = BallArithmetic._rumpogita2024_cholesky(BallMatrix(A))
        @test r !== nothing
        G = setprecision(BigFloat, 1024) do
            B = S <: Complex ? Complex{BigFloat} : BigFloat
            Matrix(cholesky(Hermitian(B.(A))).U)
        end
        @test _inside(G, r.G)
        @test istriu(mid(r.G)) && iszero(tril(rad(r.G), -1))
        @test all(real(mid(r.G)[i, i]) - rad(r.G)[i, i] > 0 for i in 1:n)
        # the paper's Figure 8: close to the rounding unit for every condition number
        @test med(_relerr(r.G)) < 1e-12
    end
    @testset "a ball of matrices, and what is declined" begin
        A = spd(8, 2, Float64)
        ρ = 1e-10
        r = BallArithmetic._rumpogita2024_cholesky(BallMatrix(A, fill(ρ, 8, 8)))
        @test r !== nothing
        for _ in 1:10
            Ep = ρ * (2 * rand(rng, 8, 8) .- 1)
            Ap = A + (Ep + Ep') / 2
            G = setprecision(() -> Matrix(cholesky(Hermitian(big.(Ap))).U), BigFloat, 1024)
            @test _inside(G, r.G)
        end
        # not positive definite, and a ball that contains a singular matrix: not proved
        @test BallArithmetic._rumpogita2024_cholesky(BallMatrix([1.0 2.0; 2.0 1.0])) === nothing
        @test BallArithmetic._rumpogita2024_cholesky(BallMatrix([1.0 0.0; 0.0 1e-3], fill(1e-2, 2, 2))) === nothing
        @test_throws ArgumentError BallArithmetic._rumpogita2024_cholesky(BallMatrix([1.0 2.0; 0.0 1.0]))
    end
end

@testset "Rump-Ogita 2024, Section 5: the QR decomposition" begin
    rng = MersenneTwister(20261014)
    med(v) = sort(v)[(length(v) + 1) ÷ 2]
    # the factors with a positive diagonal of R, at 1024 bits, through the Cholesky factor of A*A
    function refqr(A)
        setprecision(BigFloat, 1024) do
            B = eltype(A) <: Complex ? Complex{BigFloat} : BigFloat
            Ab = B.(A)
            R = Matrix(cholesky(Hermitian(Ab' * Ab)).U)
            Ab / UpperTriangular(R), R
        end
    end
    @testset "$m × $n, cond = 1e$k, $S" for (m, n) in ((20, 20), (30, 12), (7, 1)), k in (2, 8, 12),
        S in (Float64, ComplexF64)

        A = m == n ? _randsvd(rng, n, k; S) : _randsvd(rng, m, n, k; S)
        Q1, R = refqr(A)
        r = BallArithmetic._rumpogita2024_qr(BallMatrix(A))
        @test r !== nothing
        @test size(r.Q) == (m, n) && size(r.R) == (n, n)
        @test _inside(Q1, r.Q) && _inside(R, r.R)
        @test iszero(tril(rad(r.R), -1)) && all(real(mid(r.R)[i, i]) - rad(r.R)[i, i] > 0 for i in 1:n)
        # the paper's Figures 9 and 10
        @test med(_relerr(r.R)) < 1e-12
        @test med(_relerr(r.Q)) < 1e-11
        # the full decomposition: an orthonormal basis of the complement lies in the last columns
        f = BallArithmetic._rumpogita2024_qr(BallMatrix(A); full = true)
        @test f !== nothing
        @test size(f.Q) == (m, m) && size(f.R) == (m, n)
        @test _inside(Q1, f.Q[:, 1:n]) && _inside(R, f.R[1:n, :])
        if m > n
            @test iszero(mid(f.R)[(n + 1):m, :]) && iszero(rad(f.R)[(n + 1):m, :])
            U = setprecision(BigFloat, 1024) do
                Z = (I - Q1 * Q1') * big.(mid(f.Q)[:, (n + 1):m])
                H = Hermitian(Z' * Z)
                E = eigen(H)
                Z * (E.vectors * Diagonal(1 ./ sqrt.(E.values)) * E.vectors')
            end
            @test _inside(U, f.Q[:, (n + 1):m])
        end
    end
    @testset "more columns than rows" begin
        for S in (Float64, ComplexF64)
            A = _randsvd(rng, 8, 13, 3; S)
            Q, R1 = refqr(A[:, 1:8])
            r = BallArithmetic._rumpogita2024_qr(BallMatrix(A))
            @test r !== nothing && size(r.Q) == (8, 8) && size(r.R) == (8, 13)
            @test _inside(Q, r.Q)
            # the left block is the R-factor itself, with exact zeros below the diagonal
            @test _inside(R1, r.R[:, 1:8])
            @test _inside(setprecision(() -> Q' * big.(A[:, 9:13]), BigFloat, 1024), r.R[:, 9:13])
        end
    end
    @testset "a ball of matrices, and what is declined" begin
        A = _randsvd(rng, 9, 5, 2)
        ρ = 1e-11
        r = BallArithmetic._rumpogita2024_qr(BallMatrix(A, fill(ρ, 9, 5)))
        @test r !== nothing
        for _ in 1:8
            Q1, R = refqr(A + ρ * (2 * rand(rng, 9, 5) .- 1))
            @test _inside(Q1, r.Q) && _inside(R, r.R)
        end
        @test BallArithmetic._rumpogita2024_qr(BallMatrix([1.0 2.0; 2.0 4.0; 3.0 6.0])) === nothing
        @test BallArithmetic._rumpogita2024_qr(BallMatrix([1.0 0.0; 0.0 1e-3; 0.0 0.0], fill(1e-2, 3, 2))) === nothing
    end
end

@testset "Rump-Ogita 2024, Section 8: the Schur decomposition" begin
    rng = MersenneTwister(20261015)
    med(v) = sort(v)[(length(v) + 1) ÷ 2]
    @testset "n = $n, cond = 1e$k, $S" for (n, k) in ((6, 1), (15, 2), (15, 6)), S in (Float64, ComplexF64)
        A = _randsvd(rng, n, k; S)
        r = BallArithmetic._rumpogita2024_schur(BallMatrix(A); fallback = nothing)
        @test r !== nothing
        @test istriu(mid(r.T)) && iszero(tril(rad(r.T), -1))
        # the factors of the eigenvector matrix that verifyeigall normalises, at 1024 bits
        e = verifyeigall(BallMatrix(A); fallback = nothing)
        @test maximum(rad(e.similarity)) == 0        # no recursion: the frame is the point matrix W
        Q, Tt, X, λ = setprecision(BigFloat, 1024) do
            Ab = Complex{BigFloat}.(A)
            F = eigen(Ab)
            W = Complex{BigFloat}.(mid(e.similarity))
            X = similar(Ab)
            λ = Vector{Complex{BigFloat}}(undef, n)
            for j in 1:n
                i = argmin(abs.(F.values .- e.centers[j]))
                v = F.vectors[:, i]
                X[:, j] = v / (W \ v)[e.clusters[j][1]]
                λ[j] = F.values[i]
            end
            R = Matrix(cholesky(Hermitian(X' * X)).U)
            Q = X / UpperTriangular(R)
            Q, R * Diagonal(λ) / UpperTriangular(R), X, λ
        end
        @test _inside(X, r.X)
        @test all(abs(λ[j] - mid(r.D)[j, j]) <= rad(r.D)[j, j] for j in 1:n)
        @test _inside(Q, r.Q)
        @test _inside(triu(Tt), r.T)
        @test Float64(opnorm(Q * triu(Tt) * Q' - A, Inf)) < 1e-60        # the reference is a Schur decomposition (product at 256 bits)
        k <= 2 && @test med(_relerr(r.Q)) < 1e-9
    end
    @testset "what is declined" begin
        # a double eigenvalue, a defective matrix, a non-square one
        @test BallArithmetic._rumpogita2024_schur(BallMatrix([2.0 0.0; 0.0 2.0])) === nothing
        @test BallArithmetic._rumpogita2024_schur(BallMatrix([1.0 1.0; 0.0 1.0])) === nothing
        @test_throws ArgumentError BallArithmetic._rumpogita2024_schur(BallMatrix(randn(rng, 2, 3)))
    end
    @testset "a ball of matrices" begin
        A = Matrix(Diagonal(collect(1.0:5.0))) + 0.1 * randn(rng, 5, 5)
        r = BallArithmetic._rumpogita2024_schur(BallMatrix(A, fill(1e-10, 5, 5)); fallback = nothing)
        @test r !== nothing
        # A Q − Q T contains zero, and so does Q*Q − I
        res = BallMatrix(ComplexF64.(A), fill(1e-10, 5, 5)) * r.Q - r.Q * r.T
        @test all(abs.(mid(res)) .<= rad(res))
        orth = BallArithmetic._ball_adjoint(r.Q) * r.Q - I
        @test all(abs.(mid(orth)) .<= rad(orth))
    end
end

@testset "Rump-Ogita 2024, Section 7: the polar decomposition" begin
    rng = MersenneTwister(20261018)
    function refpolar(A)
        setprecision(BigFloat, 1024) do
            B = eltype(A) <: Complex ? Complex{BigFloat} : BigFloat
            F = svd(B.(A))
            F.U * F.Vt, F.V * Diagonal(F.S) * F.Vt
        end
    end
    @testset "$m × $n, $S" for (m, n) in ((6, 6), (20, 9), (4, 1)), S in (Float64, ComplexF64)
        σ = n == 1 ? [2.0] : collect(range(3.0, 0.5; length = n))
        A = Matrix(qr(randn(rng, S, m, m)).Q)[:, 1:n] * Diagonal(σ) * Matrix(qr(randn(rng, S, n, n)).Q)'
        r = BallArithmetic._rumpogita2024_polar(BallMatrix(A))
        @test r !== nothing
        Q, P = refpolar(A)
        @test _inside(Q, r.Q) && _inside(P, r.P)
        @test maximum(rad(r.Q)) < 1e-8 && maximum(rad(r.P)) < 1e-8
    end
    @testset "a cluster of singular values, a ball, a singular matrix" begin
        Qm, Qn = Matrix(qr(randn(rng, 8, 8)).Q), Matrix(qr(randn(rng, 5, 5)).Q)
        A = Qm[:, 1:5] * Diagonal([3.0, 2.0, 2.0, 2.0 + 1e-12, 1.0]) * Qn'
        r = BallArithmetic._rumpogita2024_polar(BallMatrix(A); kappa = 1e-8)
        @test r !== nothing && maximum(length, r.svd.clusters) == 3
        Q, P = refpolar(A)
        @test _inside(Q, r.Q) && _inside(P, r.P)
        b = BallArithmetic._rumpogita2024_polar(BallMatrix(A, fill(1e-10, 8, 5)); kappa = 1e-6)
        @test b !== nothing
        for _ in 1:8
            Q, P = refpolar(A + 1e-10 * (2 * rand(rng, 8, 5) .- 1))
            @test _inside(Q, b.Q) && _inside(P, b.P)
        end
        @test BallArithmetic._rumpogita2024_polar(BallMatrix(Qm[:, 1:3] * Diagonal([1.0, 1.0, 0.0]) * Qn[1:3, 1:3])) === nothing
        @test_throws ArgumentError BallArithmetic._rumpogita2024_polar(BallMatrix(randn(rng, 2, 3)))
    end
end

@testset "Rump-Ogita 2024, Section 9: the Takagi decomposition" begin
    rng = MersenneTwister(20261019)
    # a complex symmetric matrix with singular values from 1 to 10^-k, as in the paper
    function symmetric(n, k)
        Q = Matrix(qr(randn(rng, ComplexF64, n, n)).Q)
        A = transpose(Q) * Diagonal(exp10.(range(0, -k; length = n))) * Q
        return (A + transpose(A)) / 2
    end
    @testset "n = $n, cond = 1e$k" for (n, k) in ((4, 1), (12, 2), (12, 6))
        A = symmetric(n, k)
        r = BallArithmetic._rumpogita2024_takagi(BallMatrix(A))
        @test r !== nothing
        # the factors at 1024 bits, by the same reduction, signs matched to the midpoint
        U, σ = setprecision(BigFloat, 1024) do
            E, F = big.(real.(A)), big.(imag.(A))
            G = eigen(Symmetric([E F; F -E]))
            idx = sortperm(G.values; rev = true)[1:n]
            U = complex.(G.vectors[1:n, idx], G.vectors[(n + 1):(2n), idx])
            for j in 1:n
                real(dot(U[:, j], mid(r.U)[:, j])) < 0 && (U[:, j] .*= -1)
            end
            U, G.values[idx]
        end
        @test _inside(U, r.U)
        @test all(abs(σ[j] - mid(r.sigma[j])) <= rad(r.sigma[j]) for j in 1:n)
        @test issorted([mid(b) for b in r.sigma]; rev = true)
        @test Float64(opnorm(U * Diagonal(σ) * transpose(U) - A, Inf)) < 1e-60     # the reference is a Takagi decomposition
        @test Float64(opnorm(U' * U - I, Inf)) < 1e-60
        @test maximum(rad(b) / mid(b) for b in r.sigma) < 1e-8
    end
    @testset "what is declined" begin
        # a double singular value, a singular matrix, a matrix that is not symmetric
        @test BallArithmetic._rumpogita2024_takagi(BallMatrix(Matrix{ComplexF64}(I, 3, 3))) === nothing
        @test BallArithmetic._rumpogita2024_takagi(BallMatrix(ComplexF64[1 0; 0 0])) === nothing
        @test_throws ArgumentError BallArithmetic._rumpogita2024_takagi(BallMatrix(ComplexF64[1 2; 3 4]))
    end
    @testset "a ball of matrices" begin
        A = symmetric(5, 1)
        r = BallArithmetic._rumpogita2024_takagi(BallMatrix(A, fill(1e-10, 5, 5)))
        @test r !== nothing
        Σ = BallMatrix(Matrix{ComplexF64}(Diagonal([mid(b) for b in r.sigma])), Matrix(Diagonal([rad(b) for b in r.sigma])))
        Ut = BallMatrix(Matrix(transpose(mid(r.U))), Matrix(transpose(rad(r.U))))
        res = r.U * Σ * Ut - BallMatrix(A, fill(1e-10, 5, 5))
        @test all(abs.(mid(res)) .<= rad(res))
    end
end
