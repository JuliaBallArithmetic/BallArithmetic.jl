using BallArithmetic
using Test
using LinearAlgebra
using Random
using BallArithmetic: mid, rad

# Rump (2022), Theorem 2.2 and the algorithm `verifyeigall`.
#
# The accuracy attainable is governed by the Jordan structure: an eigenvalue whose largest Jordan
# block has size k cannot be enclosed better than about u^(1/k) in floating-point arithmetic
# (paper, lines 140-145). The assertions below are therefore about soundness and about the method
# declining when it cannot certify, not about a fixed accuracy.

# The construction of the paper's section 3: a diagonal matrix with one k x k Jordan block, then a
# similarity by a random matrix.
function _rump_cluster(n, k, rng)
    J = diagm(0 => randn(rng, n))
    λ = randn(rng)
    for i in 1:k
        J[i, i] = λ
        i < k && (J[i, i + 1] = 1.0)
    end
    V = randn(rng, n, n)
    return V \ J * V
end

_covered(r, λs) = all(any(r.certified[i] && abs(l - r.centers[i]) <= r.radii[i]
                          for i in eachindex(r.centers)) for l in λs)

# A reference that is exact by construction. LAPACK's `eigvals` is not usable here: on a random
# 30 x 30 it is wrong by up to 4.3e-15 where the certified radius is 2.0e-16, so comparing against
# it would fail the sound enclosure. An upper triangular matrix is exactly representable and its
# eigenvalues are exactly its diagonal, so it tests the enclosure against the truth.
function _triangular_with_spectrum(λ::Vector{Float64}, rng)
    n = length(λ)
    A = triu(randn(rng, n, n), 1) ./ sqrt(n)
    for i in 1:n
        A[i, i] = λ[i]
    end
    return A
end

@testset "verifyeigall: every eigenvalue enclosed, exact reference" begin
    rng = MersenneTwister(20260927)
    for n in (8, 20, 40)
        λ = collect(range(-1.0, 1.0; length = n)) .+ 0.13 .* randn(rng, n)
        B = _triangular_with_spectrum(λ, rng)
        r = verifyeigall(BallMatrix(B))
        @test r.transform_defect < 1
        @test r.spectrum_covered
        @test length(r.clusters) == n          # well separated, so simple
        @test _covered(r, λ)                   # against the exact spectrum
        @test maximum(r.radii) < 1e-8
    end
end

@testset "verifyeigall: random matrices are certified and self-consistent" begin
    rng = MersenneTwister(7)
    for n in (8, 20, 40)
        B = randn(rng, n, n) ./ sqrt(n)
        r = verifyeigall(BallMatrix(B))
        @test r.transform_defect < 1
        @test r.spectrum_covered
        @test length(r.clusters) == n
        # each certified disc carries a finite radius, and the radii are tight enough to be
        # narrower than LAPACK's own accuracy on such a matrix
        @test all(isfinite, r.radii)
        @test maximum(r.radii) < 1e-10
    end
end

# The eigenvalues of the floating-point input itself. A matrix built as V⁻¹JV or UJU' in floating
# point does not have the multiple eigenvalue of J: its eigenvalues are simple and spread by about
# u^(1/k) (paper, lines 315-320), and those, not the eigenvalue of J, are what an enclosure of the
# input must contain. They move by about eps^(1/k) under a perturbation eps, so the reference is
# computed at 2048 bits, where a 24-fold block is still resolved to about 1e-25.
_reference_eigvals(M; bits = 2048) = setprecision(bits) do
    eigvals(Complex{BigFloat}.(M))
end

# Soundness against the reference: every eigenvalue lies in some disc, and each certified disc that
# is disjoint from all the others holds exactly as many eigenvalues as its cluster has members.
function _sound(r, λ)
    c = Complex{BigFloat}.(r.centers)
    ρ = BigFloat.(r.radii)
    all(l -> any(i -> abs(l - c[i]) <= ρ[i], eachindex(c)), λ) || return false
    for i in eachindex(c)
        r.certified[i] || continue
        all(j -> j == i || abs(c[i] - c[j]) > ρ[i] + ρ[j], eachindex(c)) || continue
        count(l -> abs(l - c[i]) <= ρ[i], λ) == length(r.clusters[i]) || return false
    end
    return true
end

@testset "verifyeigall: Jordan clusters of size k, against the input's own eigenvalues" begin
    # Rump's Tables 2 and 4 report no failure for a cluster of size 1 or 2 at n = 100 and of size 3
    # up to n = 500, so the whole spectrum is expected to be covered here; the step 6 recursion
    # (a second transformation on the uncertified columns) is what certifies the near pair.
    rng = MersenneTwister(11)
    for k in (1, 2, 3)
        B = _rump_cluster(30, k, rng)
        r = verifyeigall(BallMatrix(B))
        @test r.transform_defect < 1
        @test all(isfinite, r.radii)
        @test all(all(isfinite, rad(r.subspaces[i])) == r.certified[i]
        for i in eachindex(r.clusters))
        @test _sound(r, _reference_eigvals(B))
        @test r.spectrum_covered
        k == 1 && @test maximum(r.radii) < 1e-13
    end
end

@testset "verifyeigall: a 24-fold Jordan block, against the input's own eigenvalues" begin
    # U J U' with J one Jordan block at 0.5. In floating point the input has 24 simple eigenvalues
    # between 0.211 and 0.214 away from 0.5 (2048 and 4096 bits agree), and an enclosure is judged
    # against those; comparing with 0.5 would test a matrix the routine was never given.
    n = 24
    J = diagm(0 => fill(0.5 + 0im, n), 1 => fill(1.0 + 0im, n - 1))
    U = Matrix(qr(randn(MersenneTwister(4), ComplexF64, n, n)).Q)
    M = U * J * U'
    λ = _reference_eigvals(M)
    for method in (:rump2022a, :rump2022aneumann, :rump2022adiscclusters)
        r = verifyeigall(BallMatrix(M); method)
        @test all(isfinite, r.radii)
        @test all(all(isfinite, rad(r.subspaces[i])) == r.certified[i]
        for i in eachindex(r.clusters))
        @test _sound(r, λ)
    end
end

@testset "verifyeigall: the invariant subspaces satisfy B Y = Y M" begin
    rng = MersenneTwister(20260927)
    for n in (12, 30)
        B = randn(rng, n, n) ./ sqrt(n)
        Bc = Matrix{ComplexF64}(B)
        r = verifyeigall(BallMatrix(B))
        @test r.spectrum_covered
        for i in eachindex(r.clusters)
            r.certified[i] || continue
            Y = mid(r.subspaces[i])
            M = mid(r.blocks[i])
            @test size(Y) == (n, length(r.clusters[i]))
            @test size(M) == (length(r.clusters[i]), length(r.clusters[i]))
            # Theorem 2.2: A Yhat = Yhat Mhat, carried back to B by W
            @test norm(Bc * Y - Y * M) / max(1.0, norm(Y)) < 1e-12
            @test all(isfinite, rad(r.subspaces[i]))
        end
    end
end

@testset "verifyeigall: uncertified clusters carry no subspace" begin
    # a 24-fold Jordan block under a real orthogonal similarity: the transformation is not
    # certified (Rump's Table 12 has no inclusion for his "Jordan" matrix either), nothing is
    # certified, and the result must say so in every field rather than return an unjustified basis
    n = 24
    rng = MersenneTwister(20260928)
    Q = Matrix(qr(randn(rng, n, n)).Q)
    B = Q * diagm(0 => fill(0.7, n), 1 => ones(n - 1)) * Q'
    # the paper's algorithm alone; the fallback certifies these clusters (testset below)
    r = verifyeigall(BallMatrix(B); fallback = nothing)
    @test count(r.certified) < length(r.clusters)
    for i in eachindex(r.clusters)
        if r.certified[i]
            @test isfinite(r.radii[i])
            @test all(isfinite, rad(r.subspaces[i]))
        else
            # a Gershgorin bound on the location, but no subspace and no Jordan block
            @test isfinite(r.radii[i])
            @test all(isinf, rad(r.subspaces[i]))
            @test all(isinf, rad(r.blocks[i]))
        end
    end
end

@testset "verifyeigall: a cluster of size 5, against the input's own eigenvalues" begin
    # Rump's Table 4 has failures from k = 5 on, so coverage is not asserted, only that whatever is
    # returned is true: a covered spectrum means every cluster certified, every radius is finite
    # (the theorem's or Gershgorin's), and the subspace is finite exactly where certified
    rng = MersenneTwister(5)
    B = _rump_cluster(40, 5, rng)
    r = verifyeigall(BallMatrix(B))
    @test !r.spectrum_covered || all(r.certified)
    @test all(isfinite, r.radii)
    @test all(all(isfinite, rad(r.subspaces[i])) == r.certified[i] for i in eachindex(r.clusters))
    @test _sound(r, _reference_eigvals(B))
end

@testset "verifyeigall: the caller selects an algorithm and rejects the rest" begin
    rng = MersenneTwister(11)
    B = BallMatrix(randn(rng, 12, 12))
    # the default is the faithful transformation
    a = verifyeigall(B)
    b = verifyeigall(B; method = :rump2022a)
    @test a.radii == b.radii
    @test a.spectrum_covered == b.spectrum_covered
    # both transformations are implemented, and both certify this matrix
    c = verifyeigall(B; method = :rump2022aneumann)
    @test c.spectrum_covered
    @test a.spectrum_covered
    # where both certify, the verified solve is at least as tight as the Neumann bound
    @test all(a.radii .<= c.radii)
    @test_throws ArgumentError verifyeigall(B; method = :nonsense)
    # keywords reach the algorithm through the caller
    @test verifyeigall(B; method = :rump2022aneumann, maxiter = 5) isa VerifyEigAllResult
    # the squareness check lives in the caller
    @test_throws ArgumentError verifyeigall(BallMatrix(randn(rng, 3, 4)))
end

@testset "Theorem 2.2 needs D constant on each cluster" begin
    # Theorem 2.2: "let mutually distinct lambda_i be given, and let D be a diagonal matrix with
    # D_jj = lambda_i for all j in mu_i". A cluster whose diagonal entries stay distinct violates
    # that hypothesis, and the disc lambda_i +- rho then need not contain the cluster's own
    # eigenvalues. Rump's step 2 groups only entries agreeing to 1e-14||A||, so the hypothesis held
    # to working precision and the defect was invisible; :rump2022adiscclusters groups wider and
    # made it visible, with discs of 2.8e-10 to 7.7e-10 holding one eigenvalue of each triple.
    D = ComplexF64[1.0, 1.0 + 3e-10, 1.0 - 2e-10, 5.0]
    clusters = [[1, 2, 3], [4]]
    Dc = BallArithmetic._rump2022a_collapse_clusters(D, clusters)
    @test Dc[1] == Dc[2] == Dc[3]
    @test Dc[1] ≈ sum(D[1:3]) / 3
    @test Dc[4] == D[4]                       # singletons are untouched
    # and the collapse moves D by no more than the spread it removes
    @test maximum(abs.(Dc[1:3] .- D[1:3])) <= 3e-10

    # ten triples spread by about 3e-10: the wider clustering must enclose all thirty
    rng = MersenneTwister(20260927)
    n = 30
    dd = Float64[]
    while length(dd) < n
        l = randn(rng)
        for _ in 1:3
            push!(dd, l)
        end
    end
    Q = Matrix(qr(randn(rng, n, n)).Q)
    M = Q * (triu(randn(rng, n, n) .* 1e-6, 1) + Diagonal(dd[1:n])) * Q'
    A = BallMatrix(M)
    setprecision(256) do
        lams = eigvals(Complex{BigFloat}.(M))
        r = verifyeigall(A; method = :rump2022adiscclusters)
        @test all(length(c) == 3 for c in r.clusters)      # the triples are grouped
        @test all(r.certified)                             # and every one passes (2.10)
        for l in lams
            @test any(i -> isfinite(r.radii[i]) &&
                          abs(l - Complex{BigFloat}(r.centers[i])) <= BigFloat(r.radii[i]),
                eachindex(r.clusters))
        end
    end
end

@testset ":rump2022adiscclusters leaves separated spectra alone" begin
    # widening the clustering must not disturb a spectrum the fixed rule already handles
    rng = MersenneTwister(77)
    for M in (randn(rng, 12, 12), (x -> (x + x') / 2)(randn(rng, 12, 12)))
        A = BallMatrix(M)
        a = verifyeigall(A; method = :rump2022a)
        b = verifyeigall(A; method = :rump2022adiscclusters)
        @test length(a.clusters) == length(b.clusters)
        @test a.radii == b.radii
        @test count(a.certified) == count(b.certified)
    end
    # and the caller knows the method
    A = BallMatrix(randn(MersenneTwister(3), 6, 6))
    @test verifyeigall(A; method = :rump2022adiscclusters) isa VerifyEigAllResult
    @test_throws ArgumentError verifyeigall(A; method = :rump2022adisc)
end

@testset "a declined cluster still carries a bound, from Gershgorin" begin
    # Theorem 2.2 proves nothing where (2.10) declines, but the transformed A encloses W^{-1}BW and
    # so has the eigenvalues of B, and its Gershgorin discs locate them in O(n^2). Rump's Table 1 is
    # "Eigenvalue bounds by Gershgorin circles and the new method verifyeigall", so that is the
    # baseline his paper measures against. `certified` still marks the theorem's claim, and the
    # subspace and block stay Inf, since Gershgorin proves no Jordan structure.
    n = 24
    rng = MersenneTwister(20260928)
    Q = Matrix(qr(randn(rng, n, n)).Q)
    # a single Jordan block: nothing can be certified, so every radius is a fallback
    M = Q * diagm(0 => fill(0.7, n), 1 => ones(n - 1)) * Q'
    A = BallMatrix(M)
    r = verifyeigall(A; method = :rump2022a)
    @test !r.spectrum_covered
    @test count(r.certified) == 0
    @test all(isfinite, r.radii)                       # a bound for every cluster
    setprecision(256) do
        for l in eigvals(Complex{BigFloat}.(M))
            @test any(i -> abs(l - Complex{BigFloat}(r.centers[i])) <= BigFloat(r.radii[i]),
                eachindex(r.clusters))
        end
    end
    # no subspace or block is claimed where the test declined
    for i in eachindex(r.clusters)
        r.certified[i] && continue
        @test all(!isfinite, rad(r.subspaces[i]))
        @test all(!isfinite, rad(r.blocks[i]))
    end

    # where the test does succeed, the radius is the theorem's and not the fallback
    B = BallMatrix(randn(rng, 12, 12))
    q = verifyeigall(B; method = :rump2022a)
    @test all(q.certified)
    @test q.spectrum_covered
    @test maximum(q.radii) < 1e-12                     # nowhere near a Gershgorin radius
end

@testset ":rump2022aschur: Theorem 2.2 in the Schur frame" begin
    rng = MersenneTwister(20261011)
    # a matrix with a non-normal part: every certified disc holds an eigenvalue of the input, the
    # similarity is the Schur factor, and the transformed matrix contains Q⁻¹BQ
    for n in (6, 12)
        B = Matrix(Diagonal(collect(1.0:n))) + 0.3 * triu(randn(rng, n, n), 1) + 1e-3 * randn(rng, n, n)
        r = verifyeigall(BallMatrix(B); method = :rump2022aschur)
        Q = mid(r.similarity)
        @test r.basis == Q
        @test opnorm(Q' * Q - I) < 1e-12
        M, λ = setprecision(BigFloat, 512) do
            Qb = Complex{BigFloat}.(Q)
            Qb \ (Complex{BigFloat}.(B) * Qb), eigvals(Complex{BigFloat}.(B))
        end
        @test all(abs.(M - mid(r.transformed)) .<= rad(r.transformed) .* (1 + 1e-12))
        @test any(r.certified)
        for i in eachindex(r.clusters)
            r.certified[i] || continue
            @test count(l -> abs(l - r.centers[i]) <= r.radii[i], λ) >= length(r.clusters[i])
        end
        if r.spectrum_covered
            @test all(l -> any(i -> abs(l - r.centers[i]) <= r.radii[i], eachindex(r.clusters)), λ)
        end
    end
    # a Jordan block is its own Schur form: the frame is kept where the eigenvector matrix is not
    J = diagm(0 => fill(0.7, 24), 1 => ones(23))
    r = verifyeigall(BallMatrix(J); method = :rump2022aschur)
    @test opnorm(mid(r.similarity)' * mid(r.similarity) - I) < 1e-12
    @test all(i -> abs(0.7 - r.centers[i]) <= r.radii[i], eachindex(r.clusters))
    @test_throws MethodError verifyeigall(BallMatrix(J); method = :rump2022aschur, maxlevels = 2)
end

@testset ":rump2022aschurstep6: the Schur frame, then the recursion of step 6" begin
    # randn(32, 32)/√32: the Schur frame alone leaves clusters uncertified; the recursion does not
    # certify fewer columns, and every certified disc holds an eigenvalue of the input
    rng = MersenneTwister(20261011)
    cols(r) = sum((length(c) for (c, ok) in zip(r.clusters, r.certified) if ok); init = 0)
    for n in (12, 32)
        B = randn(rng, n, n) / sqrt(n)
        a = verifyeigall(BallMatrix(B); method = :rump2022aschur)
        b = verifyeigall(BallMatrix(B); method = :rump2022aschurstep6)
        @test cols(b) >= cols(a)
        λ = setprecision(() -> eigvals(Complex{BigFloat}.(B)), BigFloat, 512)
        for i in eachindex(b.clusters)
            b.certified[i] || continue
            @test count(l -> abs(l - b.centers[i]) <= b.radii[i], λ) >= length(b.clusters[i])
        end
        if b.spectrum_covered
            @test all(l -> any(i -> abs(l - b.centers[i]) <= b.radii[i], eachindex(b.clusters)), λ)
        end
        # with the recursion switched off it is the Schur variant
        c = verifyeigall(BallMatrix(B); method = :rump2022aschurstep6, maxlevels = 0)
        @test c.certified == a.certified && c.radii == a.radii
        @test mid(c.similarity) == mid(a.similarity)
    end
end

@testset "the Schur variants, against the input's own eigenvalues" begin
    # the matrices of the testsets above and four with Jordan blocks under a similarity, judged as
    # there: every eigenvalue of the floating-point input in some disc, and an isolated certified
    # disc holding as many as its cluster has members
    rng = MersenneTwister(20261011)
    jord(n, λ) = diagm(0 => fill(λ, n), 1 => ones(n - 1))
    rot(A) = (Q = Matrix(qr(randn(rng, size(A)...)).Q); Q * A * Q')
    blk(args...) = cat(args...; dims = (1, 2))
    S = randn(rng, 12, 12)
    cases = [_rump_cluster(30, 1, MersenneTwister(11)), _rump_cluster(30, 2, MersenneTwister(12)),
        _rump_cluster(30, 3, MersenneTwister(13)), _rump_cluster(40, 5, MersenneTwister(5)),
        rot(jord(24, 0.7)), rot(jord(6, 0.7)), jord(6, 0.7),
        rot(blk(jord(3, 1.0), Diagonal([2.0, 3.0, 4.0, 5.0]))),
        rot(blk(jord(4, 1.0), jord(4, -1.0))),
        S * blk(jord(3, 1.0), Diagonal(collect(2.0:10.0))) / S]
    for B in cases
        λ = _reference_eigvals(B)
        for method in (:rump2022aschur, :rump2022aschurstep6)
            r = verifyeigall(BallMatrix(B); method)
            @test all(isfinite, r.radii)
            @test all(all(isfinite, rad(r.subspaces[i])) == r.certified[i]
            for i in eachindex(r.clusters))
            @test _sound(r, λ)
            @test !r.spectrum_covered || all(r.certified)
        end
    end
end

@testset "verifyeigall: the fallback runs where the method leaves the spectrum uncovered" begin
    cols(r) = sum((length(c) for (c, ok) in zip(r.clusters, r.certified) if ok); init = 0)
    same(a, b) = a.clusters == b.clusters && a.certified == b.certified && a.radii == b.radii &&
                 mid(a.similarity) == mid(b.similarity)
    rng = MersenneTwister(20261011)
    # a covered spectrum: the fallback is not run
    B = BallMatrix(randn(rng, 10, 10))
    @test same(verifyeigall(B), verifyeigall(B; fallback = nothing))
    # the 24-fold Jordan block under an orthogonal similarity: the paper's algorithm certifies
    # nothing, and the result is the one of whichever of the two does better
    n = 24
    Q = Matrix(qr(randn(MersenneTwister(20260928), n, n)).Q)
    J = BallMatrix(Q * diagm(0 => fill(0.7, n), 1 => ones(n - 1)) * Q')
    p = verifyeigall(J; fallback = nothing)
    q = verifyeigall(J; method = :rump2022aschurstep6)
    r = verifyeigall(J)
    @test !p.spectrum_covered
    @test same(r, (q.spectrum_covered || cols(q) > cols(p)) ? q : p)
    @test cols(r) >= cols(p)
    @test _sound(r, _reference_eigvals(mid(J)))
    # a method asked for by name is run alone
    @test same(verifyeigall(J; method = :rump2022a), p)
    # the fallback is a keyword, and is not run after itself or after Miyajima's method
    @test same(verifyeigall(J; fallback = :rump2022aschur),
        (s = verifyeigall(J; method = :rump2022aschur); (s.spectrum_covered || cols(s) > cols(p)) ? s : p))
    @test same(verifyeigall(J; method = :rump2022aschurstep6, fallback = :rump2022aschurstep6), q)
    @test verifyeigall(J; method = :miyajima2014a) isa VerifyEigAllResult
    @test_throws ArgumentError verifyeigall(J; fallback = :nonsense)
end

@testset "verifyeigall: an exactly singular eigenvector matrix declines the transformation" begin
    # for the 24-fold Jordan block in its own basis the computed eigenvector matrix is singular
    # and the floating-point solve of the Newton step throws; the result is then the one of a
    # transformation that is not certified, Gershgorin discs of the input and nothing certified
    J = diagm(0 => fill(0.7, 24), 1 => ones(23))
    for method in (:rump2022a, :rump2022aneumann, :rump2022adiscclusters)
        r = verifyeigall(BallMatrix(J); method)
        @test count(r.certified) == 0
        @test !r.spectrum_covered
        @test mid(r.similarity) == I
        @test mid(r.transformed) == J
        @test all(i -> abs(0.7 - r.centers[i]) <= r.radii[i], eachindex(r.clusters))
    end
end

@testset "orthonormal_invariant_basis: an orthonormal basis and its invariance defect" begin
    # Theorem 2.2's basis is the frozen-rows one, V' Y = I, so its columns are not orthonormal and
    # the enclosure can be badly conditioned. Orthonormalising does not change the eigenvalue
    # radius, which does not depend on the basis, but it gives a usable basis and a certified
    # measure of how far it is from invariant.
    rng = MersenneTwister(20260928)
    n = 12
    d = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    Q0 = Matrix(qr(randn(rng, n, n)).Q)
    M = Q0 * (triu(randn(rng, n, n) .* 1e-8, 1) + Diagonal(d)) * Q0'
    B = BallMatrix(M)
    r = verifyeigall(B; method = :rump2022adiscclusters)
    bs = orthonormal_invariant_basis(B, r)

    @test length(bs) == count(r.certified)
    @test any(length(b.cluster) == 2 for b in bs)        # the doubles are there as blocks of 2
    for b in bs
        k = length(b.cluster)
        @test size(b.basis) == (n, k)
        @test size(b.block) == (k, k)
        # orthonormal to the rounding unit, where the frozen-rows basis had cond 1e3 to 1e4
        @test cond(mid(b.basis)) < 1 + 1e-10
        @test b.orthogonality_defect < 1e-13
        # and almost invariant: the defect is what says so. prodK against a float Rayleigh block
        # puts it at the rounding unit for a simple eigenvalue, where bounding B*Q - Q*H by ball
        # products against a ball H overstated it by 22 to 78 times.
        @test isfinite(b.invariance_defect)
        @test b.invariance_defect < 1e-10
        k == 1 && @test b.invariance_defect < 1e-14
        # the block is an exact float candidate, not an enclosure
        @test all(iszero, rad(b.block))
    end
    # the frozen-rows basis really is the ill-conditioned one, so this is not a no-op
    @test maximum(cond(mid(r.subspaces[i]))
    for i in eachindex(r.clusters) if r.certified[i] && length(r.clusters[i]) > 1) > 1e2

    @testset "the invariance defect is an upper bound, and a tight one" begin
        setprecision(256) do
            for b in bs
                Q = Complex{BigFloat}.(mid(b.basis))
                Hn = Complex{BigFloat}.(mid(b.block))
                true_def = opnorm(Complex{BigFloat}.(M) * Q - Q * Hn, 2)
                @test true_def <= BigFloat(b.invariance_defect)
                # prodK makes it tight, not merely valid: within a factor of two of the truth
                @test BigFloat(b.invariance_defect) <= 2 * true_def
            end
        end
    end

    @testset "prodK takes a thin basis directly" begin
        # the residual of an n by k basis against a k by k block is n by k; the transform of
        # Theorem 2.2 uses the square case and this the thin one
        nn = 8
        Bq = BallMatrix(randn(MersenneTwister(2), nn, nn))
        W = Matrix(qr(randn(MersenneTwister(3), nn, 3)).Q)[:, 1:3]
        X = Matrix{ComplexF64}(mid(BallMatrix(W)' * Bq * BallMatrix(W)))
        R = BallArithmetic._rump2022a_prodK(Bq, Matrix{ComplexF64}(W), X)
        @test size(R) == (nn, 3)
        @test_throws DimensionMismatch BallArithmetic._rump2022a_prodK(Bq,
            Matrix{ComplexF64}(W), zeros(ComplexF64, 2, 2))
    end

    @testset "an exactly invariant coordinate subspace has a defect at rounding level" begin
        # block diagonal, so the first two coordinates span an invariant subspace exactly
        C = BallMatrix([2.0 1.0 0.0 0.0; 0.0 2.0 0.0 0.0; 0.0 0.0 7.0 1.0; 0.0 0.0 0.0 9.0])
        rc = verifyeigall(C; method = :rump2022adiscclusters)
        for b in orthonormal_invariant_basis(C, rc)
            @test b.invariance_defect < 1e-13
        end
    end

    @testset "declined clusters are skipped, since there is no subspace" begin
        # the 24-fold Jordan block whose transformation is not certified, so nothing is
        nn = 24
        Qj = Matrix(qr(randn(MersenneTwister(20260928), nn, nn)).Q)
        J = BallMatrix(Qj * diagm(0 => fill(0.7, nn), 1 => ones(nn - 1)) * Qj')
        rj = verifyeigall(J; method = :rump2022a)
        @test count(rj.certified) == 0
        @test isempty(orthonormal_invariant_basis(J, rj))
    end
end
