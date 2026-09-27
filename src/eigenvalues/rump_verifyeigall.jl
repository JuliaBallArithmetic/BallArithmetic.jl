# Verified inclusions of ALL eigenvalues and invariant subspaces: Theorem 2.2 and the algorithm
# `verifyeigall` of section 2 of
#
#   @Article{Rump2022a,
#     author  = {Rump, Siegfried M.},
#     title   = {Verified Error Bounds for All Eigenvalues and Eigenvectors of a Matrix},
#     journal = {SIAM Journal on Matrix Analysis and Applications},
#     year    = {2022},
#     volume  = {43},
#     number  = {4},
#     pages   = {1736--1754},
#     doi     = {10.1137/21M1451440},
#   }
#
# Two routines implement the theorem, distinguished by the transformation and named for it:
# `_rump2022a`, which encloses the correction by the paper's verified linear solve (Algorithm 10.7
# of Rump (2010), `_rump2010_verifylss`), and `_rump2022aneumann`, which bounds it through an
# explicit inverse and a uniform Neumann term. `verifyeigall` selects between them.
#
# The block diagonalisation this method avoids is the second algorithm of
#
#   @Article{Miyajima2014a,
#     author  = {Miyajima, Shinya},
#     title   = {Fast Enclosure for All Eigenvalues and Invariant Subspaces in Generalized
#                Eigenvalue Problems},
#     journal = {SIAM Journal on Matrix Analysis and Applications},
#     year    = {2014},
#     volume  = {35},
#     number  = {3},
#     pages   = {1205--1225},
#     doi     = {10.1137/140953150},
#   }
#
# The method is in the Krawczyk-Moore-Rump line: an inclusion of the ERROR with respect to an
# approximation, certified by a self-mapping into the interior rather than by a proof of
# nonsingularity, with epsilon-inflation driving the iteration. It deliberately does NOT use a
# numerical block diagonalisation: Rump records (lines 79-90 of the paper) that the numerical
# Jordan decomposition Miyajima's second method rests on, which is Bavely-Stewart's, "is known to
# be ill-posed, occasionally leading to computational problems".
#
# Notation of the paper. [n] = mu_1 u ... u mu_m is a partition into clusters, V_i = I(:,mu_i),
# U_i = I(:,[n]\mu_i). For C in M_n, C_D = sum_i V_i V_i' C V_i V_i' is the block diagonal part
# along the clusters and C_O = C - C_D the rest. With D diagonal carrying the approximate
# eigenvalues, E = A - D, and Rtilde from (2.8)-(2.9),
#
#     Y := X_O X_D - E - E X_O,      Z := Rtilde .* Y,                                   (2.3)
#
# and if Z V_i is contained in the interior of X V_i for a cluster i, then A has a Jordan block
# Mhat_i in lambda_i I + V_i' Z V_i with an invariant subspace enclosed by V_i + U_i U_i' Z V_i.
#
# Remark 2.3: every cluster is handled in ONE matrix Z, which is O(n^2); only the transformation
# at the start is O(n^3).
#
# The sharpness of anything computed here is governed by Wilkinson's bound, which the paper states
# at lines 140-145: the sensitivity of an eigenvalue is of order u^(1/k) for a largest Jordan block
# of size k, and that is the minimum width of an inclusion attainable in floating point. A 3-fold
# eigenvalue therefore cannot be enclosed better than about 1e-5, a 24-fold one better than 0.22.

using LinearAlgebra

export VerifyEigAllResult, verifyeigall

"""
    VerifyEigAllResult{T, CT}

Outcome of [`verifyeigall`](@ref).

# Fields
- `clusters::Vector{Vector{Int}}`: the partition `mu` of `1:n` the algorithm worked with, one entry
  per cluster of approximate eigenvalues; a cluster of length `> 1` is a multiple or nearly
  multiple eigenvalue whose individual eigenvectors are ill-posed.
- `certified::Vector{Bool}`: which clusters satisfied the self-mapping test (2.10).
- `centers::Vector{CT}`: the approximate eigenvalue `lambda_i` of each cluster.
- `radii::Vector{T}`: `rho(|V_i' Z V_i|)` for each certified cluster, so that every eigenvalue of
  `A` belonging to cluster `i` lies in the disc `centers[i] ± radii[i]`; `Inf` where not certified.
- `subspaces::Vector{BallMatrix}`: for each certified cluster, an enclosure of a basis of the
  corresponding invariant subspace of the INPUT matrix, `n` by `|mu_i|`, satisfying
  `B * subspaces[i] = subspaces[i] * blocks[i]`; radius `Inf` where not certified.
- `blocks::Vector{BallMatrix}`: the enclosed Jordan block `Mhat_i` of each certified cluster.
- `basis::Matrix{CT}`: the approximate eigenvector matrix `W` the transformation used.
- `spectrum_covered::Bool`: true when every cluster is certified, in which case the union of the
  discs contains the whole spectrum of the input.
- `iterations::Int`, `transform_defect::T`: the number of interval iterations, and the certified
  `‖I − RW‖` of the transformation, which must be below one.
"""
struct VerifyEigAllResult{T, CT}
    clusters::Vector{Vector{Int}}
    certified::Vector{Bool}
    centers::Vector{CT}
    radii::Vector{T}
    subspaces::Vector{BallMatrix{T, CT}}
    blocks::Vector{BallMatrix{T, CT}}
    basis::Matrix{CT}
    spectrum_covered::Bool
    iterations::Int
    transform_defect::T
end

function Base.show(io::IO, r::VerifyEigAllResult)
    nc = count(r.certified)
    println(io, "VerifyEigAllResult: ", length(r.clusters), " clusters, ", nc, " certified",
        r.spectrum_covered ? ", spectrum covered" : "")
    println(io, "  transformation defect ‖I − RW‖ = ", r.transform_defect,
        ", ", r.iterations, " iterations")
    for i in eachindex(r.clusters)
        println(io, "  cluster ", i, " (", length(r.clusters[i]), "): ", r.centers[i],
            r.certified[i] ? string(" ± ", r.radii[i]) : "  not certified")
    end
end

# ---------------------------------------------------------------------------------------------
# The accurate residual: Rump's `prodK`
# ---------------------------------------------------------------------------------------------
#
# Built on the error-free transformations of `src/error_free_transformations.jl`, which carry the
# reasoning for why a ball product will not serve here.

"""
Enclosure of `B*W - W*X` with `B` a ball matrix and `W`, `X` floating point, by compensated
accumulation with error-free transformations and the Ogita-Rump-Oishi bound. This is the role
`prodK` plays in Rump's `transform`.
"""
function _veig_residual(B::BallMatrix{T}, W::Matrix{CT}, X::Matrix{CT}) where {T, CT}
    n = size(W, 1)
    Bm, Br = mid(B), rad(B)
    # split into real components so the error-free transformations apply
    Bre, Bim = real.(Bm), imag.(Bm)
    Wre, Wim = real.(W), imag.(W)
    Xre, Xim = real.(X), imag.(X)
    M = Matrix{CT}(undef, n, n)
    for j in 1:n, i in 1:n
        # Re(BW - WX) = Bre Wre - Bim Wim - (Wre Xre - Wim Xim)
        re = compensated_terms(((Bre, Wre, 1.0), (Bim, Wim, -1.0),
                (Wre, Xre, -1.0), (Wim, Xim, 1.0)), i, j)
        # Im(BW - WX) = Bre Wim + Bim Wre - (Wre Xim + Wim Xre)
        im_ = compensated_terms(((Bre, Wim, 1.0), (Bim, Wre, 1.0),
                (Wre, Xim, -1.0), (Wim, Xre, -1.0)), i, j)
        M[i, j] = CT(re, im_)
    end
    u = eps(T) / 2
    γ = gamma_bound(4 * n, T)                  # 4n products summed per entry
    r = setrounding(T, RoundUp) do
        absprod = (abs.(Bm) * abs.(W)) .+ (abs.(W) * abs.(X))
        u .* abs.(M) .+ (γ * γ) .* absprod .+ Br * abs.(W)
    end
    return BallMatrix(M, r)
end

# ---------------------------------------------------------------------------------------------
# The transformation: an inclusion A of W^{-1} B W
# ---------------------------------------------------------------------------------------------

# With R an approximate inverse of W, W^{-1} B W = (RW)^{-1}(RBW), so an enclosure of R B W
# together with ‖I − RW‖ < 1 gives one of W^{-1} B W by a Neumann series, the extra radius being
# ‖I − RW‖/(1 − ‖I − RW‖) times the norm of the enclosure. This is the same pattern the verified
# block diagonalisations in this package use for their basis residual.
# Rump's `transform`: the point is NOT to enclose W^{-1} B W directly, whose radius would be that
# of the product, but to enclose the small correction. With X the approximate eigenvalue matrix,
#
#     W^{-1} B W  =  X + W^{-1}(B W - W X),
#
# and the residual B W - W X is tiny, so an enclosure of W^{-1} times it is tiny too. One Newton
# step on X makes the residual smaller still. The inner solve is by an approximate inverse R with
# the Neumann correction ‖I − RW‖/(1 − ‖I − RW‖) · ‖R·Res‖, which is now multiplied by the size of
# the residual rather than by the size of A, which is the whole difference.
function _rump2022aneumann_transform(B::BallMatrix{T}, W::Matrix{CT}, X0::Matrix{CT}) where {T, CT}
    R = inv(W)
    Rb, Wb = BallMatrix(R), BallMatrix(W)
    S = Rb * Wb - I                        # encloses R W − I
    defect = upper_bound_L2_opnorm(S)
    defect < 1 || return nothing, defect

    # one Newton step on the approximate eigenvalue matrix, in floating point: it only has to be
    # a better approximation, nothing here is certified
    X = X0 + R * mid(_veig_residual(B, W, X0))

    Res = _veig_residual(B, W, X)          # certified enclosure of B W − W X
    P = Rb * Res                           # encloses R (B W − W X)
    extra = setrounding(T, RoundUp) do
        defect * upper_bound_L2_opnorm(P) / (one(T) - defect)
    end
    Delta = BallMatrix(mid(P), setrounding(T, RoundUp) do
        rad(P) .+ extra
    end)
    return BallMatrix(X) + Delta, defect
end

# Rump's `transform`, as the paper writes it: the correction is enclosed by a verified linear
# solve rather than bounded through an explicit inverse.
#
#     W^{-1} B W  =  X + W^{-1}(B W - W X),   so   W * Delta = B W - W X,
#
# and `verifylss(W, BW - WX)` encloses Delta while proving W nonsingular, which is Theorem 10.8
# of Rump (2010) applied to the point matrix W. The residual on the right is the prodK
# enclosure, so the two terms that nearly cancel are handled by the error-free transformations
# and not by ball arithmetic.
function _rump2022a_transform(B::BallMatrix{T}, W::Matrix{CT}, X0::Matrix{CT}) where {T, CT}
    # one Newton step on the approximate eigenvalue matrix, in floating point: it only has to be
    # a better approximation, nothing here is certified
    X = X0 + (W \ mid(_veig_residual(B, W, X0)))

    Res = _veig_residual(B, W, X)          # certified enclosure of B W - W X
    sol = verifylss(BallMatrix(W), Res)
    # Theorem 10.8: success proves W nonsingular and encloses the solution of W*Delta = Res.
    # Without it there is no enclosure of W^{-1}BW, so the transformation declines.
    sol.certified || return nothing, sol.spectral_radius_bound
    return BallMatrix(X) + sol.solution, sol.spectral_radius_bound
end

# ---------------------------------------------------------------------------------------------
# Clustering: step 2 of the algorithm, connected components of "these two are indistinguishable"
# ---------------------------------------------------------------------------------------------

# mig of the ball difference d_i − d_j: the least |z| over the two enclosures, which is zero when
# they overlap. Rump's guess at the Jordan structure is the connected components of the graph on
# which that is below 1e-14 ‖A‖_inf.
function _veig_clusters(d_mid::Vector{CT}, d_rad::Vector{T}, normA::T) where {T, CT}
    n = length(d_mid)
    tol = setrounding(T, RoundUp) do
        T(1e-14) * normA
    end
    parent = collect(1:n)
    find(x) = (parent[x] == x ? x : (parent[x] = find(parent[x])))
    for i in 1:n, j in (i + 1):n
        sep = abs(d_mid[i] - d_mid[j]) - (d_rad[i] + d_rad[j])
        if sep <= tol
            pi_, pj = find(i), find(j)
            pi_ == pj || (parent[pi_] = pj)
        end
    end
    groups = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(groups, find(i), Int[]), i)
    end
    return sort(collect(values(groups)); by = first)
end

# ---------------------------------------------------------------------------------------------
# The iteration of Theorem 2.2
# ---------------------------------------------------------------------------------------------

# Elementwise product of two ball matrices, rounded outward. `Rtilde` is not exactly
# representable: (2.8) defines it by R_i(D - lambda_i I)U_i = U_i, whose entries are the exact
# reciprocals 1/(D_l - D_j), and a floating-point reciprocal satisfies (2.8) only approximately.
# It is therefore carried as an enclosure and multiplied in ball arithmetic, so that every
# quantity downstream of it remains an enclosure of the exact one.
#
#   (a +/- ra)(b +/- rb)  is contained in  ab +/- (|a| rb + |b| ra + ra rb + rounding of ab).
#
# The rounding term is 4 eps(|ab|), which covers the four multiplications and two additions of a
# complex product (relative error at most gamma_4 ~ 4u, against eps = 2u).
function _veig_hadamard(Rm::Matrix{CT}, Rr::Matrix{T}, Y::BallMatrix{T}) where {T, CT}
    m = Rm .* mid(Y)
    r = setrounding(T, RoundUp) do
        abs.(Rm) .* rad(Y) .+ Rr .* abs.(mid(Y)) .+ Rr .* rad(Y) .+ 4 .* eps.(abs.(m))
    end
    return BallMatrix(m, r)
end

# X_D, the block diagonal part along the clusters, and X_O the rest.
function _veig_split(X::BallMatrix{T}, clusters) where {T}
    mD = zero(mid(X))
    rD = zero(rad(X))
    for c in clusters
        mD[c, c] = mid(X)[c, c]
        rD[c, c] = rad(X)[c, c]
    end
    XD = BallMatrix(mD, rD)
    XO = BallMatrix(mid(X) .- mD, setrounding(T, RoundUp) do
        rad(X) .- rD
    end)
    return XD, XO
end

# Epsilon-inflation, the first of the three standard techniques.
function _veig_inflate(Y::BallMatrix{T}; factor = T(0.1), eta = T(1e-300)) where {T}
    r = setrounding(T, RoundUp) do
        rad(Y) .* (one(T) + factor) .+ eta
    end
    return BallMatrix(mid(Y), r)
end

# Z V_i ⊆ int(X V_i), column by column: (2.10). This is `in0` restricted to the columns of the
# cluster, so it uses the package's interior-containment predicate rather than repeating it.
_veig_contained(Z::BallMatrix{T}, X::BallMatrix{T}, cols) where {T} =
    in0(Z[:, cols], X[:, cols])

"""
    verifyeigall(B::BallMatrix; maxiter = 20, inflate = 0.1) -> VerifyEigAllResult

Verified inclusions of all eigenvalues and invariant subspaces of `B`, by Theorem 2.2 of
Rump (2022). Returns a [`VerifyEigAllResult`](@ref); `spectrum_covered` says whether the union of
the returned discs is proved to contain the whole spectrum.

Unlike the verified block diagonalisations of this package, no numerical Jordan decomposition is
formed. The accuracy attainable is governed by the Jordan structure: an eigenvalue whose largest
Jordan block has size `k` cannot be enclosed more tightly than about `u^(1/k)` in floating-point
arithmetic, so a triple eigenvalue is limited to about `1e-5` and the method will decline rather
than return a bound it cannot justify.

# Example
```julia
A = BallMatrix(randn(50, 50))
r = verifyeigall(A)
r.spectrum_covered && println("all ", length(r.clusters), " clusters certified")
```
"""
# One pass of steps 3 to 6 on a fixed set of approximate eigenvalues `D`.
# Returns (certified, radii, subspaces, blocks, covered, iters); `Z` is kept so the invariant
# subspaces can be read off the same iteration that certified them.
function _veig_pass(A::BallMatrix{T}, clusters, D::Vector{CT}, maxiter::Integer,
        inflate::Real) where {T, CT}
    n = size(A, 1)
    m = length(clusters)

    Em = copy(mid(A))
    for i in 1:n
        Em[i, i] -= D[i]
    end
    E = BallMatrix(Em, rad(A))

    # Rtilde of (2.8)-(2.9). The entries -1 are exact; the reciprocals are not, so they are
    # carried as enclosures: D[l] and D[j] are floats whose exact difference lies within one
    # rounding of the computed one.
    RRm = Matrix{CT}(undef, n, n)
    RRr = zeros(T, n, n)
    for j in 1:n, l in 1:n
        if l == j
            RRm[l, j] = -one(CT)
        else
            dif = D[l] - D[j]
            b = inv(Ball(dif, max(eps(abs(dif)), floatmin(T))))
            RRm[l, j] = mid(b)
            RRr[l, j] = rad(b)
        end
    end
    for c in clusters, j in c, l in c
        RRm[l, j] = -one(CT)
        RRr[l, j] = zero(T)
    end
    (all(isfinite, RRm) && all(isfinite, RRr)) ||
        return (falses(m), fill(T(Inf), m), Matrix{CT}[], BallMatrix[], false, 0)

    Y = _veig_hadamard(-RRm, RRr, E)

    # Theorem 2.2 is a statement about ONE Y: the set Phi satisfying (2.10), the rows and columns
    # J it occupies, and max{rho(Z) : Z in Z} < 1 on that submatrix. Certification is therefore
    # judged per iteration, and everything reported is read off the same Z.
    certified = falses(m)
    radii = fill(T(Inf), m)
    subspaces = Vector{BallMatrix{T, CT}}(undef, m)
    blocks = Vector{BallMatrix{T, CT}}(undef, m)
    covered = false
    iters = 0
    for it in 1:maxiter
        iters = it
        X = _veig_inflate(Y; factor = T(inflate))
        XD, XO = _veig_split(X, clusters)
        Z = _veig_hadamard(RRm, RRr, XO * XD - E - E * XO)
        ok = [_veig_contained(Z, X, c) for c in clusters]
        if count(ok) >= count(certified)
            certified = BitVector(ok)
            fill!(radii, T(Inf))
            for (i, c) in enumerate(clusters)
                ok[i] || continue
                # Remark 2.4's rho(mag(Z_ii)), by the package's Collatz-Wielandt bound;
                # collatz_upper_bound applies upper_abs internally, which is |mid| + rad
                radii[i] = collatz_upper_bound(BallMatrix(mid(Z)[c, c], rad(Z)[c, c]))
                blocks[i], subspaces[i] = _veig_block_and_subspace(Z, D, c)
            end
            # Remark 2.4: rho(mag(Z)) < 1 on the rows and columns of the certified clusters is
            # what upgrades "each M_i is a Jordan block" to "their union is the whole spectrum".
            if all(ok)
                J = reduce(vcat, clusters)
                covered = collatz_upper_bound(BallMatrix(mid(Z)[J, J], rad(Z)[J, J])) < 1
            end
        end
        Y = Z
        (all(certified) && covered) && break
        all(isfinite, mid(Z)) || break
    end
    return certified, radii, subspaces, blocks, covered, iters
end

# Theorem 2.2's Jordan block and invariant subspace for one cluster, read off Z:
#
#     Mhat_i in lambda_i I + V_i' Z_i,      Yhat_i in V_i + U_i U_i' Z_i,     A Yhat_i = Yhat_i Mhat_i,
#
# with V_i = I(:,mu_i) and U_i U_i' = I - V_i V_i' by (2.5). So the block is the cluster's own
# diagonal block of Z shifted by lambda_i, and the subspace is the identity on the cluster's rows
# together with the off-block entries of the cluster's columns: a graph representation, normalised
# so that Yhat_i[mu_i, :] is exactly the identity. That normalisation is why the subspace stays
# well posed where an individual eigenvector of a multiple eigenvalue does not.
function _veig_block_and_subspace(Z::BallMatrix{T, CT}, D::Vector{CT}, c) where {T, CT}
    n = size(Z, 1)
    k = length(c)
    Zm, Zr = mid(Z), rad(Z)

    Mm = Zm[c, c] + Diagonal(fill(D[c[1]], k))
    Mr = Zr[c, c]
    block = BallMatrix(Mm, Mr)

    Ym = Zm[:, c]
    Yr = Zr[:, c]
    Ym[c, :] .= zero(CT)                       # U_i U_i' zeroes the cluster's own rows
    Yr[c, :] .= zero(T)
    for (t, j) in enumerate(c)
        Ym[j, t] = one(CT)                     # plus V_i
    end
    return block, BallMatrix(Ym, Yr)
end

"""
    _rump2022aneumann(B::BallMatrix; maxiter = 20, inflate = 0.1, maxlevels = 3)
        -> VerifyEigAllResult

Theorem 2.2 of Rump (2022), with the transformation of §2.3 replaced by an explicit inverse
and a uniform Neumann bound. Not exported; reached through [`verifyeigall`](@ref) with
`method = :rump2022aneumann`.

Everything the theorem requires is here: the clustering of the diagonal by connected
components, `Rtilde` of (2.8)-(2.9) as an enclosure, the iteration
`Y = X_O X_D - E - E X_O` with `Z = Rtilde .* Y`, the self-mapping test (2.10) per cluster
with its left side rounded up, and the condition `rho(Z) < 1` that Theorem 2.2 needs in
addition to (2.10) before the union of the discs may be called the spectrum (Remark 2.4).
`spectrum_covered` reports the conjunction of the two.

**Where it departs from the paper.** Rump transforms by a verified linear solve,
`verifylss(W, B*W)`, which encloses `W^{-1}BW` and proves `W` nonsingular at the same time;
[`_rump2022aneumann_transform`](@ref) instead forms `R = inv(W)` in floating point, bounds
`R*W - I` in the spectral norm, and charges that defect to every entry of the residual
enclosure uniformly. The name records the substitution. Its cost is quantitative: tight
clusters of size three and above are declined where Rump's Table 4 reports no failures, at
27 of 30 clusters certified for `k = 3`, because for three eigenvalues separated by
`u^(1/3)` the entries of `Rtilde` reach `1.6e5` against transformed off-diagonals of order
`u*cond(W)`, and the quadratic term of the iteration then puts `Z` outside `X`. A faithful
`_rump2022a` would need the verified solve and does not exist yet.

No numerical Jordan decomposition is formed. The accuracy attainable is governed by the
Jordan structure: an eigenvalue whose largest Jordan block has size `k` cannot be enclosed
more tightly than about `u^(1/k)` in floating-point arithmetic, so a triple eigenvalue is
limited to about `1e-5`, and the method declines rather than return a bound it cannot
justify. Uncertified clusters carry `Inf` in radius, subspace and block.

When a pass leaves clusters uncertified, the approximate eigenvalues of those clusters are
recomputed from the corresponding submatrix and the pass is repeated, up to `maxlevels`
times. Soundness does not rest on that refinement: Theorem 2.1's assertions hold "for any
quality" of the approximation, and the certification is (2.10) alone.
"""
_rump2022aneumann(B::BallMatrix; kwargs...) =
    _verifyeigall_core(B, _rump2022aneumann_transform; kwargs...)

"""
    _rump2022a(B::BallMatrix; maxiter = 20, inflate = 0.1, maxlevels = 3)
        -> VerifyEigAllResult

Theorem 2.2 of Rump (2022) with the paper's own transformation, a verified linear solve.
Not exported; reached through [`verifyeigall`](@ref) with `method = :rump2022a`.

Identical to [`_rump2022aneumann`](@ref) in every step of the theorem, and differing only in
[`_rump2022a_transform`](@ref): the correction `Δ` to the approximate eigenvalue matrix is
enclosed by solving `WΔ = BW − WX` with `_rump2010_verifylss`, which is the Krawczyk iteration
Rump's `verifylss` performs, instead of being bounded through an explicit inverse. The solve
proves `W` nonsingular as a by-product, so no separate nonsingularity argument is needed.
"""
_rump2022a(B::BallMatrix; kwargs...) = _verifyeigall_core(B, _rump2022a_transform; kwargs...)

# The algorithm of Theorem 2.2, shared by both transformations: `transform(B, W, X0)` returns an
# enclosure of W^{-1} B W together with the diagnostic that certified it, or `nothing` when it
# cannot certify one. Everything after the transformation is the theorem itself and is identical
# for the two, so it lives here once.
function _verifyeigall_core(B::BallMatrix{T, NT}, transform;
        maxiter::Integer = 20, inflate::Real = 0.1,
        maxlevels::Integer = 3) where {T, NT}
    n = size(B, 1)
    CT = complex(T)

    F = eigen(Matrix{CT}(mid(B)))
    W = Matrix{CT}(F.vectors)
    X0 = Matrix{CT}(Diagonal(F.values))
    A, defect = transform(B, W, X0)
    A === nothing && return VerifyEigAllResult(Vector{Int}[], Bool[], CT[], T[],
        BallMatrix{T, CT}[], BallMatrix{T, CT}[], W, false, 0, defect)

    normA = upper_bound_L_inf_opnorm(A)
    dm = CT[mid(A)[i, i] for i in 1:n]
    dr = T[rad(A)[i, i] for i in 1:n]
    clusters = _veig_clusters(dm, dr, normA)
    D = copy(dm)

    certified, radii, subspaces, blocks, covered, iters =
        _veig_pass(A, clusters, D, maxiter, inflate)
    total = iters

    # The recursion of step 6: where a pass leaves clusters open, recompute their approximate
    # eigenvalues from the submatrix they occupy and try again. It stops as soon as a level fails
    # to certify more than the one before.
    level = 1
    while level <= maxlevels && !all(certified)
        J = reduce(vcat, [clusters[i] for i in eachindex(clusters) if !certified[i]];
            init = Int[])
        length(J) >= 2 || break
        sub = try
            eigen(Matrix{CT}(mid(A)[J, J])).values
        catch
            break
        end
        all(isfinite, sub) || break
        Dnew = copy(D)
        Dnew[J] .= sub
        c2, r2, s2, b2, cov2, it2 = _veig_pass(A, clusters, Dnew, maxiter, inflate)
        total += it2
        count(c2) > count(certified) || break
        certified, radii, subspaces, blocks, covered = c2, r2, s2, b2, cov2
        D = Dnew
        level += 1
    end

    centers = CT[D[c[1]] for c in clusters]
    # the invariant subspaces of B, not of the transformed A: they transform by W
    Wb = BallMatrix(W)
    subsB = BallMatrix{T, CT}[]
    blocksB = BallMatrix{T, CT}[]
    for i in eachindex(clusters)
        if certified[i] && isassigned(subspaces, i)
            push!(subsB, Wb * subspaces[i])
            push!(blocksB, blocks[i])
        else
            push!(subsB, BallMatrix(zeros(CT, n, length(clusters[i])),
                fill(T(Inf), n, length(clusters[i]))))
            push!(blocksB, BallMatrix(zeros(CT, length(clusters[i]), length(clusters[i])),
                fill(T(Inf), length(clusters[i]), length(clusters[i]))))
        end
    end
    return VerifyEigAllResult(clusters, collect(certified), centers, radii, subsB, blocksB, W,
        covered, total, defect)
end

"""
    verifyeigall(B::BallMatrix; method = :rump2022aneumann, maxiter = 20, inflate = 0.1,
                 maxlevels = 3) -> VerifyEigAllResult

Verified inclusions of all eigenvalues and invariant subspaces of `B`. Returns a
[`VerifyEigAllResult`](@ref), whose `spectrum_covered` says whether the union of the returned
discs is proved to contain the whole spectrum; where a cluster is not certified, its radius,
subspace and block are `Inf` rather than an unjustified bound.

This function selects an algorithm and does nothing else. Each algorithm is a separate
unexported function named for the paper it implements, so that a deviation from a paper is
visible in the name rather than buried in a docstring:

| `method` | routine | what it is |
|---|---|---|
| `:rump2022a` | [`_rump2022a`](@ref) | Theorem 2.2 of Rump (2022) with the paper's transformation, a verified linear solve |
| `:rump2022aneumann` | [`_rump2022aneumann`](@ref) | the same theorem with the transformation bounded through an explicit inverse and a uniform Neumann term |

The default is `:rump2022a`. The Neumann variant is kept because it does not need the solve to
succeed, so it still returns a result where the solve declines; where both succeed, the
faithful one is at least as tight.

# Example
```julia
r = verifyeigall(BallMatrix(randn(50, 50)))
if r.spectrum_covered
    for i in eachindex(r.clusters)
        println(r.centers[i], " ± ", r.radii[i])     # eigenvalue disc
        Y = r.subspaces[i]                            # basis of the invariant subspace of B
    end
end
```

# Reference

S. M. Rump, *Verified error bounds for all eigenvalues and eigenvectors of a matrix*,
SIAM J. Matrix Anal. Appl. **43**(4):1736-1754, 2022, doi 10.1137/21M1451440.
"""
function verifyeigall(B::BallMatrix; method::Symbol = :rump2022a, kwargs...)
    size(B, 1) == size(B, 2) ||
        throw(ArgumentError("verifyeigall expects a square matrix"))
    method === :rump2022a && return _rump2022a(B; kwargs...)
    method === :rump2022aneumann && return _rump2022aneumann(B; kwargs...)
    throw(ArgumentError("verifyeigall: unknown method $(repr(method)); " *
                        "the implemented methods are :rump2022a and :rump2022aneumann"))
end
