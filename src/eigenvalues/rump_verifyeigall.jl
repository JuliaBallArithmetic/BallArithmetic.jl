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
# of Rump (2010), `_rump2010_alg10_7`), and `_rump2022aneumann`, which bounds it through an
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

export inclusion_discs, eigencount_outside, eigencount_in_disc, VerifyEigAllResult, verifyeigall, AlmostInvariantBasis,
    orthonormal_invariant_basis

"""
    VerifyEigAllResult{T, CT}

Outcome of [`verifyeigall`](@ref).

# Fields
- `clusters::Vector{Vector{Int}}`: the partition `mu` of `1:n` the algorithm worked with, one entry
  per cluster of approximate eigenvalues; a cluster of length `> 1` is a multiple or nearly
  multiple eigenvalue whose individual eigenvectors are ill-posed.
- `certified::Vector{Bool}`: which clusters satisfied the self-mapping test (2.10).
- `centers::Vector{CT}`: the approximate eigenvalue `lambda_i` of each cluster.
- `radii::Vector{T}`: for a certified cluster `rho(|V_i' Z V_i|)` of Theorem 2.2; for one where the
  self-mapping test (2.10) declined, the smallest disc about `centers[i]` containing the Gershgorin
  connected component of the transformed matrix that the cluster's rows fall in, which is a valid
  enclosure but proves no Jordan structure. Either way every eigenvalue of cluster `i` lies in
  `centers[i] ± radii[i]`, and `certified[i]` says which of the two claims is being made.
- `subspaces::Vector{BallMatrix}`: `Inf` unless the cluster is certified, since no subspace follows
  from the Gershgorin fallback. For each certified cluster, an enclosure of a basis of the
  corresponding invariant subspace of the INPUT matrix, `n` by `|mu_i|`, satisfying
  `B * subspaces[i] = subspaces[i] * blocks[i]`; radius `Inf` where not certified.
- `blocks::Vector{BallMatrix}`: the enclosed Jordan block `Mhat_i` of each certified cluster.
- `basis::Matrix{CT}`: the approximate eigenvector matrix `W` the transformation used.
- `spectrum_covered::Bool`: true when every cluster is certified, in which case the union of the
  discs contains the whole spectrum of the input.
- `iterations::Int`, `transform_defect::T`: the number of interval iterations, and the certified
  `‖I − RW‖` of the transformation, which must be below one.
- `gershgorin_centers::Vector{CT}`, `gershgorin_radii::Vector{T}`: the Gershgorin row discs of the
  transformed ball matrix, which encloses `W⁻¹BW` and so carries the eigenvalues of `B`; those of
  `B` itself when the transformation failed; empty when the result comes from another method. Every eigenvalue lies in their
  union, and a union of discs disjoint from the others holds as many eigenvalues as it has discs.
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
    gershgorin_centers::Vector{CT}
    gershgorin_radii::Vector{T}
end

# the methods that have no Gershgorin discs to report
VerifyEigAllResult(clusters, certified, centers::Vector{CT}, radii::Vector{T}, subspaces, blocks,
    basis, covered, iterations, defect) where {T, CT} =
    VerifyEigAllResult(clusters, certified, centers, radii, subspaces, blocks, basis, covered,
        iterations, defect, CT[], T[])

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
accumulation with error-free transformations and the Ogita-Rump-Oishi bound.

This is Rump's `prodK`, called in his Algorithm `transform` as `prodK(B, W, -W, X)` and
described there as computing "an accurate approximation of the residual `BW - WX` using
error-free transformations", with a second output giving the radius. The paper attributes
`prodK` to no numbered reference; it is a routine of his in INTLAB. The parts have their own
provenance: `two_sum` is Knuth's, `two_product` and the splitting are Dekker's and Veltkamp's,
and the error bound is

    @Article{OgitaRumpOishi2005,
      author  = {Ogita, Takeshi and Rump, Siegfried M. and Oishi, Shin'ichi},
      title   = {Accurate Sum and Dot Product},
      journal = {SIAM Journal on Scientific Computing},
      year    = {2005},
      volume  = {26},
      number  = {6},
      pages   = {1955--1988},
      doi     = {10.1137/030601818},
    }
"""
function _rump2022a_prodK(B::BallMatrix{T}, W::Matrix{CT}, X::Matrix{CT}) where {T, CT}
    # W may be n by k and X k by k, the residual then n by k; the transform of Theorem 2.2 uses the
    # square case, and `orthonormal_invariant_basis` the thin one
    n, k = size(W)
    size(X) == (k, k) ||
        throw(DimensionMismatch("prodK: W is $(size(W)) so X must be ($k, $k), got $(size(X))"))
    size(B, 2) == n ||
        throw(DimensionMismatch("prodK: B has $(size(B, 2)) columns and W has $n rows"))
    Bm, Br = mid(B), rad(B)
    # split into real components so the error-free transformations apply
    Bre, Bim = real.(Bm), imag.(Bm)
    Wre, Wim = real.(W), imag.(W)
    Xre, Xim = real.(X), imag.(X)
    M = Matrix{CT}(undef, n, k)
    # the signs must carry the working type: Float64 literals here made the error-free
    # transformations promote and fail on a BigFloat input
    p1, m1 = one(T), -one(T)
    for j in 1:k, i in 1:n
        # Re(BW - WX) = Bre Wre - Bim Wim - (Wre Xre - Wim Xim)
        re = compensated_terms(((Bre, Wre, p1), (Bim, Wim, m1),
                (Wre, Xre, m1), (Wim, Xim, p1)), i, j)
        # Im(BW - WX) = Bre Wim + Bim Wre - (Wre Xim + Wim Xre)
        im_ = compensated_terms(((Bre, Wim, p1), (Bim, Wre, p1),
                (Wre, Xim, m1), (Wim, Xre, m1)), i, j)
        M[i, j] = CT(re, im_)
    end
    # The radius. Each of the real and imaginary parts of an entry is a compensated dot product
    # of N = 2n + 2k ≤ 4 max(n, k) terms, to which Proposition 5.5 of Ogita, Rump and Oishi (2005)
    # applies with its underflow term (the reference and the two steps that are ours, the passage
    # from |exact| to |computed| and the factor for a complex entry, are at `_accuracy_term` in
    # src/types/MMul/ogita_rump_oishi.jl):
    #     |M − (BW − WX)| ≤ (u|M| + (3/2)(γ_N² (|B||W| + |W||X|) + 5 N eta)) / (1 − u),
    # to which the interval part rad(B)|W| is added. The moduli are bounded above with `abs_up`.
    N = 4 * max(n, k)
    absW, absX, absB = abs_up.(W), abs_up.(X), abs_up.(Bm)
    absprod, interval_part = setrounding(T, RoundUp) do
        (absB * absW) .+ (absW * absX), Br * absW
    end
    r = add_up.(_accuracy_term(abs_up.(M), absprod, N, T, true), interval_part)
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
    X = X0 + R * mid(_rump2022a_prodK(B, W, X0))

    Res = _rump2022a_prodK(B, W, X)          # certified enclosure of B W − W X
    P = Rb * Res                           # encloses R (B W − W X)
    # defect ‖P‖ / (1 − defect), the numerator rounded up and the denominator down
    extra = div_up(mul_up(defect, upper_bound_L2_opnorm(P)), sub_down(one(T), defect))
    Delta = BallMatrix(mid(P), add_up.(rad(P), extra))
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
    X = X0 + (W \ mid(_rump2022a_prodK(B, W, X0)))

    Res = _rump2022a_prodK(B, W, X)          # certified enclosure of B W - W X
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
function _rump2022a_clusters(d_mid::Vector{CT}, d_rad::Vector{T}, normA::T) where {T, CT}
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
function _rump2022a_eq2_3(Rm::Matrix{CT}, Rr::Matrix{T}, Y::BallMatrix{T}) where {T, CT}
    m = Rm .* mid(Y)
    # the moduli bounded above (`abs` of a complex number is `hypot`, which does not honour the
    # rounding mode) and the rounding of each complex product by `_product_roundoff`
    r = add_up.(add_up.(mul_up.(abs_up.(Rm), rad(Y)), mul_up.(Rr, abs_up.(mid(Y)))),
        add_up.(mul_up.(Rr, rad(Y)), _product_roundoff.(m)))
    return BallMatrix(m, r)
end

# X_D, the block diagonal part along the clusters, and X_O the rest.
function _rump2022a_split(X::BallMatrix{T}, clusters) where {T}
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
function _rump2022a_inflate(Y::BallMatrix{T}; factor = T(0.1), eta = T(1e-300)) where {T}
    r = setrounding(T, RoundUp) do
        rad(Y) .* (one(T) + factor) .+ eta
    end
    return BallMatrix(mid(Y), r)
end

# Z V_i ⊆ int(X V_i), column by column: (2.10). This is `in0` restricted to the columns of the
# cluster, so it uses the package's interior-containment predicate rather than repeating it.
_rump2022a_eq2_10(Z::BallMatrix{T}, X::BallMatrix{T}, cols) where {T} =
    in0(Z[:, cols], X[:, cols])

# One pass of steps 3 to 6 on a fixed set of approximate eigenvalues `D`.
# Returns (certified, radii, subspaces, blocks, covered, iters); `Z` is kept so the invariant
# subspaces can be read off the same iteration that certified them.
# Theorem 2.2 requires "mutually distinct lambda_i" and "D_jj = lambda_i for all j in mu_i": the
# diagonal must be CONSTANT on each cluster. The printed algorithm sets D = d.mid, keeping the
# distinct computed eigenvalues, which satisfies the hypothesis only because step 2 groups j and p
# when mig(d_j - d_p) <= 1e-14 ||A||_inf, so the entries of a cluster agree to working precision.
# Collapsing them to their mean makes the hypothesis exactly true and moves D by at most that same
# threshold. Without it, a cluster of size > 1 gets a disc that does not contain its own
# eigenvalues: on a 30 by 30 with ten triples spread by 3e-10, the discs came out at 2.8e-10 to
# 7.7e-10 about the FIRST eigenvalue of each cluster and held one of the three.
# Theorem 2.2 needs the values of distinct clusters to differ; the collapse to means does not
# guarantee it when a cluster is wide, so it is checked rather than assumed.
_rump2022a_distinct(D, clusters) = allunique(D[c[1]] for c in clusters)

function _rump2022a_collapse_clusters(D::Vector{CT}, clusters) where {CT}
    D = copy(D)
    for c in clusters
        length(c) == 1 && continue
        lam = sum(D[j] for j in c) / length(c)
        for j in c
            D[j] = lam
        end
    end
    return D
end

# Where (2.10) declines, Theorem 2.2 asserts nothing, but a coarser bound is still available and
# costs O(n^2): the Gershgorin discs of the transformed matrix A, which encloses W^{-1}BW and so has
# the eigenvalues of B. Rump's own Table 1 is "Eigenvalue bounds by Gershgorin circles and the new
# method verifyeigall", so this is the baseline his paper measures against.
#
# Gershgorin gives: every eigenvalue lies in the union of the n discs, and a set of discs forming a
# connected component isolated from the rest contains exactly as many eigenvalues as it has discs.
# So for a declined cluster the honest report is the smallest disc about its lambda_i containing the
# whole connected component its rows fall in, together with that component's size as the count.
# `certified[i]` stays false: the subspace and the Jordan block are not proved, only the location.
# The Gershgorin row discs of a ball matrix, valid for every member: centre the midpoint of the
# diagonal entry, radius its own radius plus the moduli of the off-diagonal entries of the row,
# each rounded upward.
function _gershgorin_rows(A::BallMatrix{T}) where {T}
    n = size(A, 1)
    Am, Ar = mid(A), rad(A)
    ctr = [complex(Am[i, i]) for i in 1:n]
    rad_ = T[sum_up(add_up(abs_up(Am[i, j]), Ar[i, j]) for j in 1:n if j != i) for i in 1:n]
    rad_ = T[add_up(rad_[i], Ar[i, i]) for i in 1:n]
    return ctr, rad_
end

function _rump2022a_gershgorin_discs(A::BallMatrix{T}, D::Vector{CT}, clusters) where {T, CT}
    ctr, rad_ = _gershgorin_rows(A)
    comps = overlap_components([Ball(CT(ctr[i]), rad_[i]) for i in eachindex(ctr)])
    comp_of = Dict(i => c for c in comps for i in c)
    out_r = Vector{T}(undef, length(clusters))
    out_m = Vector{Int}(undef, length(clusters))
    for (k, c) in enumerate(clusters)
        rows = sort(unique(reduce(vcat, [comp_of[j] for j in c])))
        lam = D[c[1]]
        out_r[k] = maximum(add_up(dist_up(lam, ctr[p]), rad_[p]) for p in rows)
        out_m[k] = length(rows)
    end
    return out_r, out_m
end

function _rump2022a_thm2_2_pass(A::BallMatrix{T}, clusters, D::Vector{CT}, maxiter::Integer,
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
    # (2.9) reads Rtilde[:, j] = diag(R_i) for j in mu_i, and (2.8) makes R_i equal to -1 on mu_i
    # and 1/(D_pp - lambda_i) off it. So the whole block mu_i x mu_i is -1, the diagonal included,
    # and the reciprocal must not be formed there: D is constant on a cluster, so the difference is
    # exactly zero and `inv` of that ball throws.
    cid = zeros(Int, n)
    for (k, c) in enumerate(clusters), j in c
        cid[j] = k
    end
    RRm = Matrix{CT}(undef, n, n)
    RRr = zeros(T, n, n)
    for j in 1:n, l in 1:n
        if cid[l] == cid[j]
            RRm[l, j] = -one(CT)
        else
            # 1/(D_l − D_j) as an enclosure: the ball difference contains the exact difference
            # of the two floats, and the complex ball inverse the reciprocal of every member. A
            # difference not proved away from zero means this partition cannot be used.
            dif = Ball(D[l]) - Ball(D[j])
            b = try
                inv(dif)
            catch e
                e isa ArgumentError || rethrow(e)
                return (falses(m), fill(T(Inf), m), Matrix{CT}[], BallMatrix[], false, 0)
            end
            RRm[l, j] = mid(b)
            RRr[l, j] = rad(b)
        end
    end
    (all(isfinite, RRm) && all(isfinite, RRr)) ||
        return (falses(m), fill(T(Inf), m), Matrix{CT}[], BallMatrix[], false, 0)

    Y = _rump2022a_eq2_3(-RRm, RRr, E)

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
        X = _rump2022a_inflate(Y; factor = T(inflate))
        XD, XO = _rump2022a_split(X, clusters)
        Z = _rump2022a_eq2_3(RRm, RRr, XO * XD - E - E * XO)
        ok = [_rump2022a_eq2_10(Z, X, c) for c in clusters]
        if count(ok) >= count(certified)
            certified = BitVector(ok)
            fill!(radii, T(Inf))
            for (i, c) in enumerate(clusters)
                ok[i] || continue
                # Remark 2.4's rho(mag(Z_ii)), by the package's Collatz-Wielandt bound;
                # collatz_upper_bound applies upper_abs internally, which is |mid| + rad
                radii[i] = collatz_upper_bound(BallMatrix(mid(Z)[c, c], rad(Z)[c, c]))
                blocks[i], subspaces[i] = _rump2022a_thm2_2_block(Z, D, c)
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
function _rump2022a_thm2_2_block(Z::BallMatrix{T, CT}, D::Vector{CT}, c) where {T, CT}
    n = size(Z, 1)
    k = length(c)
    Zm, Zr = mid(Z), rad(Z)

    Mm = Zm[c, c] + Diagonal(fill(D[c[1]], k))
    Mr = Zr[c, c]
    # the diagonal entries are rounded sums λ_i + Z_tt: their rounding goes into the radius
    ϵ = machine_epsilon(T)
    for t in 1:k
        Mr[t, t] = add_up(Mr[t, t], mul_up(ϵ, abs_up(Mm[t, t])))
    end
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
`u*cond(W)`, and the quadratic term of the iteration then puts `Z` outside `X`. That figure was
measured before the step 6 recursion transformed the uncertified columns again.

No numerical Jordan decomposition is formed. The accuracy attainable is governed by the
Jordan structure: an eigenvalue whose largest Jordan block has size `k` cannot be enclosed
more tightly than about `u^(1/k)` in floating-point arithmetic, so a triple eigenvalue is
limited to about `1e-5`, and the method declines rather than return a bound it cannot
justify. Uncertified clusters carry a Gershgorin radius and `Inf` in subspace and block.

When a pass leaves clusters uncertified, those columns are transformed again and the pass is
repeated, up to `maxlevels` times, as for `_rump2022a`. Soundness does not rest on that refinement: Theorem 2.1's assertions hold "for any
quality" of the approximation, and the certification is (2.10) alone.
"""
_rump2022aneumann(B::BallMatrix; kwargs...) =
    _rump2022a_thm2_2_core(B, _rump2022aneumann_transform; kwargs...)

"""
    _rump2022a(B::BallMatrix; maxiter = 20, inflate = 0.1, maxlevels = 3)
        -> VerifyEigAllResult

Theorem 2.2 of Rump (2022) with the paper's own transformation, a verified linear solve.
Not exported; reached through [`verifyeigall`](@ref) with `method = :rump2022a`.

Identical to [`_rump2022aneumann`](@ref) in every step of the theorem, and differing only in
[`_rump2022a_transform`](@ref): the correction `Δ` to the approximate eigenvalue matrix is
enclosed by solving `WΔ = BW − WX` with `_rump2010_alg10_7`, which is the Krawczyk iteration
Rump's `verifylss` performs, instead of being bounded through an explicit inverse. The solve
proves `W` nonsingular as a by-product, so no separate nonsingularity argument is needed.
"""
_rump2022a(B::BallMatrix; kwargs...) = _rump2022a_thm2_2_core(B, _rump2022a_transform; kwargs...)

# The clustering of Rump's step 2 groups i and j when mig(d_i - d_j) <= 1e-14 ||A||_inf, a FIXED
# absolute threshold. A k-fold eigenvalue computed in floating point spreads by about u^(1/k), and
# the paper says so itself: "this attempt to construct a matrix with 3-fold eigenvalue generates a
# matrix with a cluster of radius 1e-5". That is eight orders above the threshold, so a triple is
# left as three singletons, and a singleton cannot satisfy (2.10) for a defective eigenvalue.
#
# This rule groups i and j when the discs of their diagonal entries (midpoint and radius) come
# within sqrt(eps)*||A||_inf of each other, the floating-point spread of a double eigenvalue. It is
# not Miyajima's (2014a) rule, which groups on overlap of the Gershgorin discs of the rows: the row
# radii are not available here. It is a DEVIATION from Rump's algorithm and carries its own name for
# that reason. Theorem 2.2 permits it: the partition mu is arbitrary there, and (2.10) is what proves
# the inclusion, so the change can only alter how often the test succeeds and never whether a
# success is valid. The one further hypothesis, that the values of distinct clusters differ, is
# checked by `_rump2022a_distinct` after the collapse.
function _rump2022a_discclusters_rule(d_mid::Vector{CT}, d_rad::Vector{T}, normA::T) where {T, CT}
    n = length(d_mid)
    # the Gershgorin radius of row i is not available here, so the disc is the diagonal enclosure
    # widened by the floating-point sensitivity of a multiple eigenvalue, sqrt(eps)*||A||, which is
    # the k = 2 floor and the smallest widening that groups anything the fixed rule does not
    tol = setrounding(T, RoundUp) do
        sqrt(eps(T)) * normA
    end
    parent = collect(1:n)
    find(x) = (parent[x] == x ? x : (parent[x] = find(parent[x])))
    for i in 1:n, j in (i + 1):n
        sep = setrounding(T, RoundDown) do
            abs(d_mid[i] - d_mid[j]) - (d_rad[i] + d_rad[j])
        end
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

"""
    _rump2022a_discclusters(B::BallMatrix; maxiter = 20, inflate = 0.1, maxlevels = 3)
        -> VerifyEigAllResult

Theorem 2.2 of Rump (2022) with the paper's transformation, but clustering the diagonal at
`√eps·‖A‖_∞` instead of the `1e-14·‖A‖_∞` of its step 2. **A deviation from Rump's algorithm**,
named for it; reached through [`verifyeigall`](@ref) with `method = :rump2022adiscclusters`.

The reason is that the fixed threshold cannot group a multiple eigenvalue. A `k`-fold eigenvalue
computed in floating point spreads by about `u^(1/k)`, and the paper states the case: "this attempt
to construct a matrix with 3-fold eigenvalue generates a matrix with a cluster of radius 1e-5".
Against a threshold of `1e-14‖A‖` those three eigenvalues stay three singletons, and a singleton
cannot satisfy (2.10) for a defective eigenvalue, so the algorithm declines on exactly the case
Theorem 2.2 was written to handle. Miyajima (2014a) groups on overlap of the Gershgorin discs of
the rows, whose radii follow the residual; this variant does not have the row radii and widens the
diagonal discs by the `k = 2` sensitivity `√eps·‖A‖` instead.

Soundness does not depend on the choice. The partition in Theorem 2.2 is arbitrary and (2.10) is
what proves the inclusion, so a different clustering changes how often the test succeeds and never
whether a success is valid. The cost is that well-separated eigenvalues closer than `√eps‖A‖` are
merged into one block, which widens their discs to the block's.
"""
_rump2022a_discclusters(B::BallMatrix; kwargs...) =
    _rump2022a_thm2_2_core(B, _rump2022a_transform, _rump2022a_discclusters_rule; kwargs...)

# The algorithm of Theorem 2.2, shared by both transformations: `transform(B, W, X0)` returns an
# enclosure of W^{-1} B W together with the diagnostic that certified it, or `nothing` when it
# cannot certify one. Everything after the transformation is the theorem itself and is identical
# for the two, so it lives here once.
function _rump2022a_thm2_2_core(B::BallMatrix{T, NT}, transform,
        cluster_rule = _rump2022a_clusters;
        maxiter::Integer = 20, inflate::Real = 0.1,
        maxlevels::Integer = 3) where {T, NT}
    n = size(B, 1)
    CT = complex(T)

    F = eigen(Matrix{CT}(mid(B)))
    W = Matrix{CT}(F.vectors)
    X0 = Matrix{CT}(Diagonal(F.values))
    A, defect = transform(B, W, X0)
    A === nothing && return _rump2022a_untransformed(B, W, defect)

    lv = _rump2022a_level(A, cluster_rule, maxiter, inflate)
    total = lv.iters
    # the similarity accumulated over the levels, as an enclosure: the invariant subspaces of B are
    # those of the final A multiplied by it
    S = BallMatrix(W)

    # The recursion of step 6, "if the number of successful columns does not increase, the function
    # verifyeigall is applied recursively to the columns with no inclusion". The paper gives no code
    # for it; it is implemented here as the algorithm's own first step applied to those columns: with
    # J the uncertified indices, T is the identity except T[J,J], the eigenvectors of mid(A)[J,J],
    # and `transform` encloses T⁻¹AT by the same verified solve, which also proves T nonsingular.
    # T⁻¹AT has the eigenvalues of A and its invariant subspaces transform by T, so Theorem 2.2
    # applied to it is as valid as on A. Recomputing only the eigenvalues of the block and keeping
    # A, which is what this did before, leaves the coupling of a near pair in E, where it is as
    # large as the pair's separation, and (2.10) declines again; at n = 30 with a double eigenvalue
    # (test_rump_verifyeigall.jl, the k = 2 case) that left two clusters uncertified.
    level = 1
    while level <= maxlevels && !all(lv.certified)
        J = reduce(vcat, [lv.clusters[i] for i in eachindex(lv.clusters) if !lv.certified[i]];
            init = Int[])
        length(J) >= 2 || break
        sub = try
            eigen(Matrix{CT}(mid(A)[J, J]))
        catch
            break
        end
        (all(isfinite, sub.values) && all(isfinite, sub.vectors)) || break
        Tm = Matrix{CT}(I, n, n)
        Tm[J, J] .= sub.vectors
        d = CT[mid(A)[i, i] for i in 1:n]
        d[J] .= sub.values
        A2, _ = transform(A, Tm, Matrix{CT}(Diagonal(d)))
        A2 === nothing && break
        lv2 = _rump2022a_level(A2, cluster_rule, maxiter, inflate)
        total += lv2.iters
        _rump2022a_columns(lv2) > _rump2022a_columns(lv) || break
        A, lv = A2, lv2
        S = S * BallMatrix(Tm)
        level += 1
    end

    (; clusters, D, certified, radii, subspaces, blocks, covered) = lv
    # D is constant on each cluster, so D[c[1]] IS the cluster's lambda_i of Theorem 2.2
    # Theorem 2.2 says nothing where (2.10) declined; report the Gershgorin location instead of
    # Inf, so every cluster carries SOME valid bound and `certified` marks which are the theorem's.
    if !all(certified)
        gr, _ = _rump2022a_gershgorin_discs(A, D, clusters)
        for i in eachindex(clusters)
            certified[i] || (radii[i] = min(radii[i], gr[i]))
        end
    end

    centers = CT[D[c[1]] for c in clusters]
    # the invariant subspaces of B, not of the transformed A: they transform by S
    subsB = BallMatrix{T, CT}[]
    blocksB = BallMatrix{T, CT}[]
    for i in eachindex(clusters)
        if certified[i] && isassigned(subspaces, i)
            push!(subsB, S * subspaces[i])
            push!(blocksB, blocks[i])
        else
            push!(subsB, BallMatrix(zeros(CT, n, length(clusters[i])),
                fill(T(Inf), n, length(clusters[i]))))
            push!(blocksB, BallMatrix(zeros(CT, length(clusters[i]), length(clusters[i])),
                fill(T(Inf), length(clusters[i]), length(clusters[i]))))
        end
    end
    gc, gr = _gershgorin_rows(A)
    return VerifyEigAllResult(clusters, collect(certified), centers, radii, subsB, blocksB, W,
        covered, total, defect, CT.(gc), gr)
end

# Steps 2 to 6 on one transformed matrix A: the clustering, D constant on each cluster, and the
# iteration of Theorem 2.2. Where two clusters collapse to the same value, which Theorem 2.2
# excludes, nothing is certified and the caller's Gershgorin fallback supplies the radii.
function _rump2022a_level(A::BallMatrix{T}, cluster_rule, maxiter, inflate) where {T}
    n = size(A, 1)
    CT = complex(T)
    normA = upper_bound_L_inf_opnorm(A)
    dm = CT[mid(A)[i, i] for i in 1:n]
    dr = T[rad(A)[i, i] for i in 1:n]
    clusters = cluster_rule(dm, dr, normA)
    # Theorem 2.2 needs D constant on each cluster; see _rump2022a_collapse_clusters
    D = _rump2022a_collapse_clusters(dm, clusters)
    m = length(clusters)
    if !_rump2022a_distinct(D, clusters)
        @warn "verifyeigall: two clusters collapsed to the same value, which Theorem 2.2 excludes; nothing is certified"
        return (; clusters, D, certified = falses(m), radii = fill(T(Inf), m),
            subspaces = Vector{BallMatrix{T, CT}}(undef, m),
            blocks = Vector{BallMatrix{T, CT}}(undef, m), covered = false, iters = 0)
    end
    certified, radii, subspaces, blocks, covered, iters =
        _rump2022a_thm2_2_pass(A, clusters, D, maxiter, inflate)
    return (; clusters, D, certified, radii, subspaces, blocks, covered, iters)
end

# the number of columns satisfying (2.10), which is what the paper's step 6 counts
_rump2022a_columns(lv) = sum((length(c) for (c, ok) in zip(lv.clusters, lv.certified) if ok); init = 0)

# When the transformation cannot be certified there is no enclosure of W⁻¹BW, and Theorem 2.2 has
# nothing to work on; Rump's algorithm returns no inclusion then (his Table 12, the "Jordan" matrix).
# The Gershgorin discs of B itself still hold every eigenvalue, a union of discs disjoint from the
# rest holding as many as it has discs (Varga, Gersgorin and His Circles, Theorem 1.6), so each
# eigenvalue index gets the smallest disc about the diagonal entry containing its component, with
# `certified` false, no subspace and no block, as for any declined cluster.
function _rump2022a_untransformed(B::BallMatrix{T}, W, defect) where {T}
    n = size(B, 1)
    CT = complex(T)
    Bc = BallMatrix(Matrix{CT}(mid(B)), rad(B))
    clusters = [[i] for i in 1:n]
    D = CT[mid(Bc)[i, i] for i in 1:n]
    radii, _ = _rump2022a_gershgorin_discs(Bc, D, clusters)
    gc, gr = _gershgorin_rows(Bc)
    return VerifyEigAllResult(clusters, falses(n) |> collect, D, radii,
        BallMatrix{T, CT}[BallMatrix(zeros(CT, n, 1), fill(T(Inf), n, 1)) for _ in 1:n],
        BallMatrix{T, CT}[BallMatrix(zeros(CT, 1, 1), fill(T(Inf), 1, 1)) for _ in 1:n],
        Matrix{CT}(W), false, 0, defect, CT.(gc), gr)
end

"""
    verifyeigall(B::BallMatrix; method = :rump2022a, maxiter = 20, inflate = 0.1,
                 maxlevels = 3) -> VerifyEigAllResult

Verified inclusions of all eigenvalues and invariant subspaces of `B`. Returns a
[`VerifyEigAllResult`](@ref), whose `spectrum_covered` says whether the union of the returned
discs is proved to contain the whole spectrum. Where a cluster is not certified, its radius is
a Gershgorin bound and its subspace and block are `Inf`; `certified` says which is which. When
the transformation itself cannot be certified, every eigenvalue index gets the Gershgorin disc
of `B` about its diagonal entry and nothing is certified.

`maxlevels` bounds the recursion of step 6: where a pass leaves clusters uncertified, those
columns are transformed again by the eigenvectors of their diagonal block and Theorem 2.2 is
applied to the result. The paper describes this step in one sentence and gives no code; the
implementation is the algorithm's own first step applied to those columns. `maxlevels = 0`
switches it off.

This function selects an algorithm and does nothing else. Each algorithm is a separate
unexported function named for the paper it implements, so that a deviation from a paper is
visible in the name rather than buried in a docstring:

| `method` | routine | what it is |
|---|---|---|
| `:rump2022a` | [`_rump2022a`](@ref) | Theorem 2.2 of Rump (2022) with the paper's transformation, a verified linear solve |
| `:rump2022aneumann` | [`_rump2022aneumann`](@ref) | the same theorem with the transformation bounded through an explicit inverse and a uniform Neumann term |
| `:rump2022adiscclusters` | [`_rump2022a_discclusters`](@ref) | Theorem 2.2 with the diagonal clustered at `√eps·‖A‖` rather than the `1e-14·‖A‖` of the paper's step 2, so that a multiple eigenvalue is grouped; a deviation, named for it |
| `:miyajima2014a` | [`_miyajima2014a_alg1`](@ref) | Algorithms 1 and 2 of Miyajima (2014): Gershgorin discs on the pencil transformed by an approximate generalised eigendecomposition, each cluster certified by Brouwer's theorem on a Newton operator |

The default is `:rump2022a`. The Neumann variant is kept because it does not need the solve to
succeed, so it still returns a result where the solve declines; where both succeed, the
faithful one is at least as tight. `:miyajima2014a` is the only method that takes a pencil, so
`verifyeigall(A, B)` accepts it alone; called with one matrix it solves `A x = λ x`.

Note that `spectrum_covered` is not the same condition in the two families. Rump's needs the
self-mapping test on every cluster together with `ρ(Z) < 1`; Miyajima's needs only `‖t‖_∞ < 1`,
after which the discs cover the spectrum whether or not the per-cluster tests succeed. In both,
`certified[i]` is the per-cluster claim about the invariant subspace.

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
    method === :rump2022adiscclusters && return _rump2022a_discclusters(B; kwargs...)
    method === :miyajima2014a &&
        return _miyajima2014a_alg1(B, BallMatrix(Matrix{eltype(mid(B))}(I, size(B)...)))
    throw(ArgumentError("verifyeigall: unknown method $(repr(method)); the implemented " *
                        "methods are :rump2022a, :rump2022aneumann, :rump2022adiscclusters " *
                        "and :miyajima2014a"))
end

"""
    verifyeigall(A::BallMatrix, B::BallMatrix; method = :miyajima2014a)
        -> VerifyEigAllResult

Verified inclusions of all eigenvalues and invariant subspaces of the pencil `A x = λ B x`.

Only [`_miyajima2014a_alg1`](@ref) solves a pencil, so `:miyajima2014a` is the only method accepted
here; Rump (2022) is stated for a single matrix and `:rump2022a` is refused rather than applied
to `B⁻¹A`, which would be a different algorithm from the one the name refers to. `B` is not
assumed nonsingular: its nonsingularity is proved, by `‖t‖_∞ < 1`.
"""
function verifyeigall(A::BallMatrix, B::BallMatrix; method::Symbol = :miyajima2014a)
    size(A) == size(B) ||
        throw(DimensionMismatch("verifyeigall: A is $(size(A)) and B is $(size(B))"))
    size(A, 1) == size(A, 2) ||
        throw(ArgumentError("verifyeigall expects square matrices"))
    method === :miyajima2014a && return _miyajima2014a_alg1(A, B)
    throw(ArgumentError("verifyeigall: method $(repr(method)) does not take a pencil; " *
                        "the implemented pencil method is :miyajima2014a"))
end

# ---------------------------------------------------------------------------------------------
# Counting eigenvalues in a region from the inclusions
#
# When every cluster is certified and max rho(Z) < 1, Theorem 2.2 gives for each cluster i a
# Jordan block of size |mu_i| whose eigenvalues lie in the disc centers[i] ± radii[i], and these
# blocks together carry the whole spectrum with algebraic multiplicity. When some cluster is not
# certified, the Gershgorin row discs of the transformed matrix still carry it: every eigenvalue is
# in their union, and a union of discs disjoint from the others holds as many eigenvalues as it has
# discs (Varga, Gersgorin and His Circles, Theorem 1.6, for every member of the ball). In both
# cases the number of eigenvalues in a region is the sum of the counts of the discs inside it,
# provided no disc meets its boundary.

"""
    inclusion_discs(r::VerifyEigAllResult) -> (centers, radii, counts, ok)

Discs holding all the eigenvalues, each with the number it holds when it is disjoint from the
others: when every cluster is certified (`r.spectrum_covered`), the discs of Theorem 2.2 with the
cluster sizes; otherwise the Gershgorin row discs of the transformed matrix, each counting one, so
that a union of discs disjoint from the rest holds as many eigenvalues as it has discs. `ok` is
false when neither is available.
"""
function inclusion_discs(r::VerifyEigAllResult)
    if r.spectrum_covered
        return r.centers, r.radii, [length(c) for c in r.clusters], true
    end
    isempty(r.gershgorin_centers) && return (r.centers[1:0], r.radii[1:0], Int[], false)
    return r.gershgorin_centers, r.gershgorin_radii, ones(Int, length(r.gershgorin_centers)), true
end

"""
    eigencount_outside(r::VerifyEigAllResult, R) -> (count, ok)

Number of eigenvalues of modulus greater than `R`, with algebraic multiplicity, from the
[`inclusion_discs`](@ref) of [`verifyeigall`](@ref): the sum of the counts of the discs outside the
circle, when no disc meets it. Each disc is then inside or outside, so the discs outside are a union
of components disjoint from the rest and hold exactly that many eigenvalues. `ok` is false, and
`count` zero, when no discs are available or some disc meets the circle `|z| = R`.
"""
function eigencount_outside(r::VerifyEigAllResult, R::Real)
    ctr, rad_, cnt_, ok = inclusion_discs(r)
    ok || return (0, false)
    cnt = 0
    for i in eachindex(ctr)
        if sub_down(abs_down(ctr[i]), rad_[i]) > R
            cnt += cnt_[i]
        elseif add_up(abs_up(ctr[i]), rad_[i]) < R
            continue
        else
            return (0, false)
        end
    end
    return (cnt, true)
end

"""
    eigencount_in_disc(r::VerifyEigAllResult, c, ρ) -> (count, ok)

Number of eigenvalues in the open disc `|z − c| < ρ`, with algebraic multiplicity, from the
[`inclusion_discs`](@ref) of [`verifyeigall`](@ref), as in [`eigencount_outside`](@ref); `ok` is
false when no discs are available or some disc meets the circle `|z − c| = ρ`.
"""
function eigencount_in_disc(r::VerifyEigAllResult, c::Number, ρ::Real)
    ctr, rad_, cnt_, ok = inclusion_discs(r)
    ok || return (0, false)
    c = convert(complex(eltype(r.centers)), c)
    cnt = 0
    for i in eachindex(ctr)
        dhi, dlo = dist_up(ctr[i], c), dist_down(ctr[i], c)
        if add_up(dhi, rad_[i]) < ρ
            cnt += cnt_[i]
        elseif sub_down(dlo, rad_[i]) > ρ
            continue
        else
            return (0, false)
        end
    end
    return (cnt, true)
end

# ---------------------------------------------------------------------------------------------
# An orthonormal basis for a certified cluster, with a certified measure of how far it is from
# being invariant.
# ---------------------------------------------------------------------------------------------
#
# Theorem 2.2 delivers the subspace basis in the FROZEN-ROWS normalisation, V_i' Yhat_i = I_k, so
# its columns are not orthonormal and the enclosure can be badly conditioned: on a 12 by 12 with
# three semisimple doubles, cond(mid Y) came out 2.6e3, 2.6e3 and 1.2e4. That conditioning does not
# affect the eigenvalue radius, which Theorem 2.2 takes from rho(|V_i' Z V_i|), but it does affect
# anything built on the basis afterwards: a spectral projector, a sigma_min of a block, a resolvent
# bound through the block all degrade with cond(Y).
#
# So after the cluster is certified, orthonormalise and report what the new basis satisfies. The
# orthonormal Q comes from a floating-point QR of mid(Y) and is a CANDIDATE: no claim is made that
# it spans the invariant subspace. What is computed rigorously, in ball arithmetic, is
#
#     ||Q*Q - I||_2                the orthogonality defect,
#     H := Q* B Q                  the Rayleigh block,
#     ||B Q - Q H||_2              the invariance defect,
#
# and those three say exactly how far Q is from an invariant subspace of B, with no theorem needed
# beyond the arithmetic. They are the standard "almost invariant subspace" data.

"""
    AlmostInvariantBasis{T, CT}

An orthonormal basis for one certified cluster of a [`VerifyEigAllResult`](@ref), with certified
defects. Produced by [`orthonormal_invariant_basis`](@ref).

# Fields
- `cluster::Vector{Int}`: the indices of the cluster this basis belongs to.
- `basis::BallMatrix`: `Q`, `n × k`, orthonormal to within `orthogonality_defect`.
- `block::BallMatrix`: the Rayleigh block `H`, `k × k`, as an exact floating-point matrix with zero
  radius. It is a *candidate*, not an enclosure of anything: the certified statement is
  `invariance_defect`, a bound on `‖BQ − QH‖₂` for this `H`.
- `orthogonality_defect::T`: a rigorous bound on `‖Q*Q − I‖₂`.
- `invariance_defect::T`: a rigorous bound on `‖BQ − QH‖₂`. This is the measure of how far `Q` is
  from spanning an invariant subspace of `B`; it is zero exactly when `Q` spans one.
"""
struct AlmostInvariantBasis{T, CT}
    cluster::Vector{Int}
    basis::BallMatrix{T, CT}
    block::BallMatrix{T, CT}
    orthogonality_defect::T
    invariance_defect::T
end

function Base.show(io::IO, b::AlmostInvariantBasis)
    print(io, "AlmostInvariantBasis(k = ", length(b.cluster),
        ", ‖Q*Q − I‖ = ", b.orthogonality_defect,
        ", ‖BQ − QH‖ = ", b.invariance_defect, ")")
end

"""
    orthonormal_invariant_basis(B::BallMatrix, r::VerifyEigAllResult)
        -> Vector{AlmostInvariantBasis}

For every certified cluster of `r`, an orthonormal basis of its subspace together with certified
bounds on how far that basis is from orthonormal and from invariant. One entry per certified
cluster, in cluster order; clusters where the self-mapping test declined are skipped, since there
is no subspace to orthonormalise.

Theorem 2.2 returns the basis in the frozen-rows normalisation `Vᵢᵀ Ŷᵢ = I_k`, whose columns are
not orthonormal; measured on three semisimple doubles, `cond(mid Y)` was `2.6e3`, `2.6e3` and
`1.2e4`. Consumers of the basis — a spectral projector, the smallest singular value of a block, a
resolvent bound through the block — degrade with that conditioning, while the eigenvalue radius
does not depend on it at all.

The orthonormal `Q` is obtained by a floating-point QR of `mid(Ŷᵢ)` and is a **candidate**: nothing
here claims it spans the invariant subspace. What is rigorous is the triple

    ‖Q*Q − I‖₂,        H = fl(Q*BQ),        ‖BQ − QH‖₂,

the last computed by `prodK`, the error-free-transformation residual, against the input `B`, and it
is zero exactly when `Q` spans an invariant subspace of `B`. Taking `H` as a floating-point
candidate rather than a ball matters: a ball `H` carries a radius of order `u‖B‖`, which enters the
defect multiplied by `‖Q‖ = 1` and dominated it, overstating the defect by 22 to 78 times for a
simple eigenvalue. With `prodK` and a float `H` the bound agrees with a 256-bit reference to three
digits. Turning `invariance_defect` into a distance to a true
invariant subspace needs a separation between the cluster and the rest of the spectrum as well, and
is not done here.

# Example
```julia
r = verifyeigall(B)
for b in orthonormal_invariant_basis(B, r)
    println(length(b.cluster), "  ", b.invariance_defect)
end
```
"""
function orthonormal_invariant_basis(B::BallMatrix{T, NT},
        r::VerifyEigAllResult{T, CT}) where {T, NT, CT}
    out = AlmostInvariantBasis{T, CT}[]
    for i in eachindex(r.clusters)
        r.certified[i] || continue
        Y = r.subspaces[i]
        all(isfinite, rad(Y)) || continue
        k = size(Y, 2)
        Qm = Matrix{CT}(qr(mid(Y)).Q)[:, 1:k]      # the candidate, needing no verification
        all(isfinite, Qm) || continue
        Q = BallMatrix(Qm)
        orth = upper_bound_L2_opnorm(Q' * Q - I)
        # H is taken as the FLOAT Rayleigh quotient, a candidate needing no verification. That
        # makes B*Q - Q*H exactly prodK's shape, so the residual is computed by error-free
        # transformations and the radius of a ball-arithmetic H never enters. Measured against a
        # 256-bit reference the defect then agrees to three digits, where forming H as a ball and
        # bounding B*Q - Q*H by ball products overstated it by 22 to 78 times for a simple
        # eigenvalue: the ball H carries a radius of order u||B||, which dominated everything.
        Hm = Matrix{CT}(mid(Q' * B * Q))
        inv_def = upper_bound_L2_opnorm(_rump2022a_prodK(B, Qm, Hm))
        push!(out,
            AlmostInvariantBasis{T, CT}(copy(r.clusters[i]), Q, BallMatrix(Hm), orth, inv_def))
    end
    return out
end
