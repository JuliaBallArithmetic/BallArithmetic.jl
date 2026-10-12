# The variants of the decompositions with error bounds that compute their starting factorisation
# in Double64 or in a MultiFloat, and the names kept for the BigFloat routines: every factor
# against a reference computed at 2048 bits, in the normalisation of the routine.

using Test
using BallArithmetic
using LinearAlgebra
using Random
using DoubleFloats
using MultiFloats
using GenericLinearAlgebra

# every entry of the reference within the radius of the enclosure
_ext_inside(B, ref) = all(abs.(big.(mid(B)) .- ref) .<= big.(rad(B)))
_ext_maxrad(B) = Float64(maximum(rad(B)))

@testset "decompositions: the Double64, MultiFloat and BigFloat variants" begin
    setprecision(BigFloat, 2048) do
        variants = (
            ("double64", verified_lu_double64, verified_cholesky_double64, verified_qr_double64,
                verified_polar_double64, verified_takagi_double64),
            ("multifloat x2", verified_lu_multifloat, verified_cholesky_multifloat,
                verified_qr_multifloat, verified_polar_multifloat, verified_takagi_multifloat),
            ("multifloat x4", (A; kw...) -> verified_lu_multifloat(A; precision = :x4, kw...),
                (A; kw...) -> verified_cholesky_multifloat(A; precision = :x4, kw...),
                (A; kw...) -> verified_qr_multifloat(A; precision = :x4, kw...),
                (A; kw...) -> verified_polar_multifloat(A; precision = :x4, kw...),
                (A; kw...) -> verified_takagi_multifloat(A; precision = :x4, kw...)),
            ("gla", verified_lu_gla, verified_cholesky_gla, verified_qr_gla, verified_polar_gla,
                nothing))
        rng = MersenneTwister(20261012)
        for T in (Float64, ComplexF64), n in (6, 15)
            A = randn(rng, T, n, n)
            H = A' * A + I
            H = (H + H') / 2
            Ab, Hb = big.(A), big.(H)
            # references: the Cholesky factor with positive diagonal; A = QR with the same
            # normalisation of R; A = QP with P = (A*A)^(1/2)
            Gref = Matrix(cholesky(Hermitian(Hb)).U)
            Rref = Matrix(cholesky(Hermitian(Ab' * Ab)).U)
            Qref = Ab / Rref
            E = eigen(Hermitian(Ab' * Ab))
            Pref = E.vectors * Diagonal(sqrt.(E.values)) * E.vectors'
            Qpref = Ab / Pref
            for (tag, flu, fch, fqr, fpo, fta) in variants
                @testset "$tag, $T, n = $n" begin
                    r = flu(A)
                    @test r.certified
                    F = lu(Ab[r.p, r.q], NoPivot())
                    @test _ext_inside(r.L, Matrix(F.L))
                    @test _ext_inside(r.U, Matrix(F.U))
                    @test _ext_maxrad(r.L) < 1e-25 && _ext_maxrad(r.U) < 1e-25

                    r = fch(H)
                    @test r.certified
                    @test _ext_inside(r.G, Gref)
                    @test _ext_maxrad(r.G) < 1e-25

                    r = fqr(A)
                    @test r.certified
                    @test _ext_inside(r.Q, Qref)
                    @test _ext_inside(r.R, Rref)
                    @test _ext_maxrad(r.Q) < 1e-25 && _ext_maxrad(r.R) < 1e-25

                    r = fpo(A)
                    @test r.certified
                    @test _ext_inside(r.Q, Qpref)
                    @test _ext_inside(r.P, Pref)
                    @test _ext_maxrad(r.Q) < 1e-20 && _ext_maxrad(r.P) < 1e-20
                end
            end
        end

        @testset "Takagi" begin
            for n in (4, 9)
                B = randn(rng, ComplexF64, n, n)
                S = B + transpose(B)
                Sb = big.(S)
                # the singular values, and the factor up to the sign of each column, from the
                # real symmetric matrix [E F; F −E], S = E + iF
                M = [real.(Sb) imag.(Sb); imag.(Sb) -real.(Sb)]
                Em = eigen(Hermitian(M))
                pos = sortperm(Em.values; rev = true)[1:n]
                σref = Em.values[pos]
                Uref = complex.(Em.vectors[1:n, pos], Em.vectors[(n + 1):(2n), pos])
                for (tag, _, _, _, _, fta) in variants
                    fta === nothing && continue
                    @testset "$tag, n = $n" begin
                        r = fta(S)
                        @test r.certified
                        @test all(abs(σref[k] - big(mid(r.Σ[k]))) <= big(rad(r.Σ[k])) for k in 1:n)
                        Um = big.(mid(r.U))
                        ok = true
                        for k in 1:n
                            u = Uref[:, k] / norm(Uref[:, k])
                            s = real(dot(u, Um[:, k])) >= 0 ? 1 : -1
                            ok &= all(abs.(s .* u .- Um[:, k]) .<= big.(rad(r.U)[:, k]))
                        end
                        @test ok
                        @test _ext_maxrad(r.U) < 1e-20
                    end
                end
            end
        end

        @testset "the oracle keyword, and verified_svd_gla" begin
            A = randn(rng, 7, 7)
            r1 = verified_lu(A; oracle = Double64)
            r2 = verified_lu_double64(A)
            @test mid(r1.L) == mid(r2.L) && rad(r1.U) == rad(r2.U)
            # the starting factorisation in Double64 gives wider enclosures than in BigFloat
            r0 = verified_lu(A)
            @test _ext_maxrad(r0.U) <= _ext_maxrad(r1.U)
            @test r1.L isa BallMatrix{BigFloat}
            # a rectangular matrix
            R = randn(rng, 9, 5)
            r = verified_lu_double64(R)
            @test r.certified
            F = lu(big.(R)[r.p, r.q], NoPivot())
            @test _ext_inside(r.L, Matrix(F.L)) && _ext_inside(r.U, Matrix(F.U))
            r = verified_qr_double64(R)
            @test r.certified
            Rref = Matrix(cholesky(Hermitian(big.(R)' * big.(R))).U)
            @test _ext_inside(r.R, Rref) && _ext_inside(r.Q, big.(R) / Rref)
            # a singular matrix is not certified
            @test !verified_lu_double64([1.0 2.0; 2.0 4.0]).certified
            @test !verified_cholesky_multifloat([1.0 2.0; 2.0 1.0]).certified
            @test_throws ArgumentError verified_lu_multifloat(A; precision = :x3)
            # all the singular values
            s = verified_svd_gla(A)
            σ = svdvals(big.(A))
            @test all(any(abs(σ[k] - big(mid(b))) <= big(rad(b)) for b in s.values) for k in 1:7)
            @test maximum(Float64.(rad.(s.values))) < 1e-40
        end
    end
end
