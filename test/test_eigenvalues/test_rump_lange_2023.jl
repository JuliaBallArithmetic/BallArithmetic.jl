using BallArithmetic
using LinearAlgebra
using Random
using Test

# Rump and Lange (2023): all eigenpairs of a Hermitian matrix, all singular pairs of a matrix.

_rl_inside(X, B) = all(abs.(X - mid(B)) .<= rad(B))

# the orthonormal basis of span(V) nearest to X: the unitary polar factor of P X, P the projector
function _nearest_basis(V, X)
    Z = V * (V' * X)
    E = eigen(Hermitian(Z' * Z))
    return Z * (E.vectors * Diagonal(1 ./ sqrt.(E.values)) * E.vectors')
end

@testset "Rump-Lange 2023: the Hermitian eigenproblem" begin
    rng = MersenneTwister(20261016)
    function hermitian(d, S)
        Q = Matrix(qr(randn(rng, S, length(d), length(d))).Q)
        A = Q * Diagonal(d) * Q'
        return (A + A') / 2
    end
    # Theorem 4.2 and Theorem 6.2 against an eigendecomposition at 1024 bits
    function check(A, r; ρ = 0.0)
        n = size(A, 1)
        B = eltype(A) <: Complex ? Complex{BigFloat} : BigFloat
        setprecision(BigFloat, 1024) do
            F = eigen(Hermitian(B.(A)))
            λ = F.values
            @test sort(reduce(vcat, r.clusters)) == 1:n
            used = falses(n)
            for v in r.clusters
                a, b = minimum(r.lo[v]), maximum(r.hi[v])
                idx = findall(l -> a <= l <= b, λ)
                @test length(idx) == length(v)                  # exactly |μ| eigenvalues
                used[idx] .= true
                # a numbering with λ_j ∈ L_j: the sorted eigenvalues of the cluster against the
                # intervals sorted by midpoint
                ord = sort(v; by = j -> r.lo[j] + r.hi[j])
                @test all(r.lo[ord[t]] <= λ[idx[t]] <= r.hi[ord[t]] for t in eachindex(idx))
                if ρ == 0 && length(idx) == length(v)
                    Q = _nearest_basis(F.vectors[:, idx], B.(mid(r.vectors)[:, v]))
                    @test _rl_inside(Q, r.vectors[:, v])
                end
            end
            @test all(used)
            for (s, v) in enumerate(r.clusters), w in r.clusters[(s + 1):end]
                @test maximum(r.hi[v]) < minimum(r.lo[w]) || maximum(r.hi[w]) < minimum(r.lo[v])
            end
        end
    end
    @testset "separated eigenvalues, n = $n, $S" for n in (5, 30), S in (Float64, ComplexF64)
        A = hermitian(randn(rng, n) .* 3, S)
        r = BallArithmetic._rumplange2023_eig(BallMatrix(A))
        @test all(length(v) == 1 for v in r.clusters)
        check(A, r)
        # Table 2 of the paper: refined inclusions of relative error about 1e-14
        @test maximum((r.hi .- r.lo) ./ max.(abs.(r.lo), abs.(r.hi))) < 1e-11
        @test maximum(rad(r.vectors)) < 1e-10
    end
    @testset "clusters, $S" for S in (Float64, ComplexF64)
        # two tenfold clusters of width 1e-11, as in the paper's Table 3, and separated ones
        d = vcat(0.1 .+ 1e-11 .* randn(rng, 10), 0.2 .+ 1e-11 .* randn(rng, 10),
            range(0.3, 1.0; length = 10), range(-1.0, -0.3; length = 10))
        A = hermitian(d, S)
        r = BallArithmetic._rumplange2023_eig(BallMatrix(A))
        check(A, r)
        @test all(isfinite, rad(r.vectors))
        # an exactly double eigenvalue
        A2 = Matrix{S}(Diagonal([1.0, 1.0, 2.0, 3.0]))
        r2 = BallArithmetic._rumplange2023_eig(BallMatrix(A2))
        check(A2, r2)
        # a multiple of the identity: one cluster of everything
        r3 = BallArithmetic._rumplange2023_eig(BallMatrix(Matrix{S}(2.0I, 4, 4)))
        @test r3.clusters == [[1, 2, 3, 4]]
        check(Matrix{S}(2.0I, 4, 4), r3)
    end
    @testset "the paper's interval matrix (4.5)" begin
        A = [16.0 7 0 3 7; 7 -4 -1 -2 1; 0 -1 -6 5 1; 3 -2 5 -6 3; 7 1 1 3 -2]
        r = BallArithmetic._rumplange2023_eig(BallMatrix(A, fill(0.5, 5, 5)))
        # Section 4: the first three eigenvalues form one cluster, the other two are alone
        @test sort(length.(r.clusters)) == [1, 1, 3]
        for _ in 1:20
            Ep = 0.5 * (2 * rand(rng, 5, 5) .- 1)
            check(A + (Ep + Ep') / 2, r; ρ = 0.5)
        end
        # Table 4: at radius 0.1 the five eigenvalues are separated
        @test length(BallArithmetic._rumplange2023_eig(BallMatrix(A, fill(0.1, 5, 5))).clusters) == 5
    end
    @testset "through verifyeigall" begin
        A = hermitian([-2.0, -1.0, 0.5, 0.5 + 1e-12, 3.0, 4.0], Float64)
        r = verifyeigall(BallMatrix(A); method = :rumplange2023)
        @test r isa VerifyEigAllResult
        @test r.spectrum_covered && all(r.certified)
        λ = setprecision(() -> eigvals(Hermitian(big.(A))), BigFloat, 1024)
        for (i, v) in enumerate(r.clusters)
            @test count(l -> abs(l - r.centers[i]) <= r.radii[i], λ) == length(v)
            # A Q = Q M on the ball: the residual contains zero
            res = BallMatrix(ComplexF64.(A)) * r.subspaces[i] - r.subspaces[i] * r.blocks[i]
            @test all(abs.(mid(res)) .<= rad(res))
        end
        @test all(l -> any(j -> abs(l - r.gershgorin_centers[j]) <= r.gershgorin_radii[j], 1:6), λ)
        # the block resolvent floor of a Hermitian matrix: the distance to the spectrum
        f = block_resolvent_floor(r)
        @test f.kappa < 1 + 1e-8
        for z in (0.7 + 0.3im, -5.0 + 0im, 3.5 + 2.0im)
            truth = minimum(svdvals(A - z * I))
            s = sigma_min_floor(f, z; near = true)
            @test 0 < s <= truth * (1 + 1e-10)
            @test s >= 0.99 * truth
        end
        @test_throws MethodError verifyeigall(BallMatrix(A); method = :rumplange2023, maxiter = 3)
        @test_throws ArgumentError BallArithmetic._rumplange2023_eig(BallMatrix(randn(rng, 2, 3)))
    end
end

@testset "Rump-Lange 2023: singular values and singular subspaces" begin
    rng = MersenneTwister(20261017)
    function withsv(m, n, σ, S)
        p = min(m, n)
        return Matrix(qr(randn(rng, S, m, m)).Q)[:, 1:p] * Diagonal(σ) * Matrix(qr(randn(rng, S, n, n)).Q)[:, 1:p]'
    end
    function check(A, r; vectors = true)
        m, n = size(A)
        p = min(m, n)
        B = eltype(A) <: Complex ? Complex{BigFloat} : BigFloat
        setprecision(BigFloat, 1024) do
            F = svd(B.(A))
            σ = F.S
            lo = [mid(b) - rad(b) for b in r.values]
            hi = [mid(b) + rad(b) for b in r.values]
            @test all(>=(0), lo)
            @test sort(reduce(vcat, r.clusters)) == 1:p
            used = falses(p)
            for v in r.clusters
                a, b = minimum(lo[v]), maximum(hi[v])
                idx = findall(s -> a <= s <= b, σ)
                @test length(idx) == length(v)
                used[idx] .= true
                ord = sort(v; by = j -> -(lo[j] + hi[j]))      # σ is in decreasing order
                @test all(lo[ord[t]] <= σ[idx[t]] <= hi[ord[t]] for t in eachindex(idx))
                if vectors && length(idx) == length(v)
                    if all(isfinite, rad(r.U)[:, v])
                        @test _rl_inside(_nearest_basis(F.U[:, idx], B.(mid(r.U)[:, v])), r.U[:, v])
                    end
                    if all(isfinite, rad(r.V)[:, v])
                        @test _rl_inside(_nearest_basis(F.V[:, idx], B.(mid(r.V)[:, v])), r.V[:, v])
                    end
                end
            end
            @test all(used)
        end
    end
    @testset "$m × $n, $S" for (m, n) in ((6, 6), (30, 12), (12, 30), (5, 1)), S in (Float64, ComplexF64)
        p = min(m, n)
        A = withsv(m, n, collect(range(3.0, 0.5; length = p)), S)
        r = verifysvdall(BallMatrix(A))
        @test r isa VerifySvdAllResult
        @test size(r.U) == (m, p) && size(r.V) == (n, p)
        @test all(length(v) == 1 for v in r.clusters)
        check(A, r)
        # Tables 11 and 12 of the paper: singular values to about 1e-14, vectors to the gap
        @test maximum(rad(b) / mid(b) for b in r.values) < 1e-11
        @test maximum(rad(r.U)) < 1e-9 && maximum(rad(r.V)) < 1e-9
    end
    @testset "clusters and the threshold" begin
        σ = vcat(0.1 .+ 1e-11 .* randn(rng, 5), 0.2 .+ 1e-11 .* randn(rng, 5), collect(range(0.3, 1.0; length = 8)))
        A = withsv(40, 18, σ, Float64)
        r = verifysvdall(BallMatrix(A))
        check(A, r)
        # with the threshold the two groups are two clusters with narrow subspaces (Table 14)
        k = verifysvdall(BallMatrix(A); kappa = 1e-8)
        check(A, k)
        @test sort(length.(k.clusters)) == vcat(fill(1, 8), [5, 5])
        @test maximum(rad(k.U)) < 1e-9 && maximum(rad(k.V)) < 1e-9
        @test_throws ArgumentError verifysvdall(BallMatrix(A); kappa = -1)
    end
    @testset "a singular matrix, and a ball of matrices" begin
        # rank deficient: the smallest interval reaches zero and its left subspace is not claimed
        A = withsv(6, 4, [3.0, 2.0, 1.0, 0.0], Float64)
        r = verifysvdall(BallMatrix(A))
        check(A, r; vectors = false)
        @test minimum(mid(b) - rad(b) for b in r.values) == 0
        A = withsv(7, 5, [5.0, 4.0, 3.0, 2.0, 1.0], Float64)
        r = verifysvdall(BallMatrix(A, fill(1e-9, 7, 5)))
        for _ in 1:10
            check(A + 1e-9 * (2 * rand(rng, 7, 5) .- 1), r; vectors = false)
        end
    end
end
