# Verified enclosure of ALL eigenvalues and invariant subspaces of the pencil A x = lambda B x,
# by Algorithms 1 and 2 of
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
# The route is different from Rump's in `rump_verifyeigall.jl`: the discs come from Gershgorin
# applied to the pencil transformed by an approximate generalised eigendecomposition, and a
# cluster is certified by Brouwer's theorem on a Newton operator for the invariant subspace,
# not by a self-mapping test on a divided-difference iteration.
#
# NOTATION, all of it the paper's. Xt and Dt are the numerical GED of (A, B), lambda_i = Dt[i,i],
# and Y approximates (B Xt)^{-1}. Then
#
#     R := Y (A Xt - B Xt Dt),     S := I - Y B Xt,     t := |S| 1,     u := |R| 1,        (3.1)
#
# and Corollary 3.2 says: if ||t||_inf < 1 then B, Xt and Y are nonsingular and every eigenvalue
# lies in some disc <lambda_i, r_i> with
#
#     r := u + <u>_t t,           <f>_t := max_i |f_i| / (1 - t_i)                (Lemma 2.3)
#
# the functional being Lemma 2.3 of Minamihata, and Corollary 2.4 extending it to a matrix F by
# |(I-S)^{-1}||F| <= |F| + t w^T with w_p = <F[:,p]>_t. Remark 3.3: k discs forming a connected
# component isolated from the rest contain precisely k eigenvalues.
#
# An ISOLATED disc gets an eigenvector by Theorem 3.5, in O(n^2): with I_i and J_i the identity
# without its i-th row and column,
#
#     f := | dt - lambda_i 1 | - r_i 1,   v := |R| J_i 1,   w := |R| e_i,
#     g := (I_i (v + <v>_t t)) ./ (I_i f),   h := (I_i (w + <w>_t t)) ./ (I_i f),
#     q := h + <h>_g g,
#
# and then the disc holds exactly one eigenvalue, of geometric multiplicity one, with an
# eigenvector satisfying | x* - Xt[:,i] | <= |Xt| J_i q.
#
# A CLUSTER v = {i_1 < ... < i_k} with complement u gets Algorithm 2. V and U are the columns of
# I indexed by v and u; lambda is the mean of the cluster's approximate eigenvalues; Dt' agrees
# with Dt off the cluster and is lambda on it; R' := Y (A Xt - B Xt Dt'), which Remark 3.8
# obtains from R in O(n^2) as R' = R + Y B Xt (Dt - Dt'). Freezing the k rows of the unknown by
# V^T Phat = I_k turns the invariant-subspace equation into F(P) = 0 for
#
#     F(P) := ((Dt' + Q') U U^T - V V^T) P - U U^T P V^T P + (Dt' + Q') V,
#
# whose Newton operator at lambda*V has derivative Z_Q = (Dt' - lambda I + Q') U U^T - V V^T.
# Lemma 3.9 certifies Z_Q invertible through mu; Lemma 3.10 encloses the image of the Newton
# operator on <lambda V, P_r>; and Theorem 3.15 chooses P_r = eta * Pbar_eps so that the
# self-mapping into the INTERIOR holds. Theorem 3.11 then gives k eigenvalues in
# <lambda, rho(V^T Pbar_eps)> and an invariant subspace inside <Xt V, |Xt| U U^T Pbar_eps>, with
# the block itself in <lambda I_k, V^T Pbar_eps>.
#
# Every bound below is computed with the rounding mode set upward, which is the paper's fl_(.).
# Two exceptions use the package's emulated directed operations instead. The modulus of a complex
# difference goes through `dist_up`/`dist_down`, since `abs(::Complex)` is `hypot`, which does not
# honour the rounding mode: under RoundUp it fell below the exact modulus for 7.8% of 10^6 random
# arguments. And the scalar recurrences of Steps 4 to 6 use `add_up` and the like, because there
# the same expression is needed rounded both ways, and inside one function the compiler may
# evaluate it once for both blocks, the rounding mode being invisible to it: in a probe, `a*b`
# under RoundDown returned the RoundUp value in all of 200000 cases when both sat in one loop.

using LinearAlgebra

# <f>_t of Lemma 2.3: max_i |f_i| / (1 - t_i), rounded up. Requires t_i < 1, which the caller
# has checked through ||t||_inf < 1.
function _miyajima2014a_lem2_3(f::AbstractVector, t::AbstractVector{T}) where {T <: AbstractFloat}
    m = zero(T)
    for i in eachindex(f)
        den = sub_down(one(T), t[i])
        den > 0 || return T(Inf)
        m = max(m, div_up(abs(f[i]), den))
    end
    return m
end

# Corollary 2.4: w_p = <F[:,p]>_t for the columns of F
_miyajima2014a_cor2_4(F::AbstractMatrix, t::AbstractVector{T}) where {T} =
    T[_miyajima2014a_lem2_3(view(F, :, p), t) for p in axes(F, 2)]

# the discs of Corollary 3.2, and the quantities Algorithm 2 reuses
struct _Miyajima2014aDiscs{T, CT}
    lambdas::Vector{CT}          # lambda_i = Dt[i,i]
    r::Vector{T}                 # the Gershgorin radii of Corollary 3.2
    t::Vector{T}                 # |S| 1
    absR::Matrix{T}              # an entrywise upper bound for |R|
    absXt::Matrix{T}             # an entrywise upper bound for |Xt|
    Xt::Matrix{CT}               # the approximate eigenvector matrix
    YBXt::BallMatrix{T, CT}      # reused by Remark 3.8 to form R'
    R::BallMatrix{T, CT}         # the residual enclosure
    tinf::T                      # ||t||_inf, proved < 1
end

# Step 1 of Algorithm 1: "compute Dt and Xt by the numerical GED of A and B". The candidate is
# arbitrary, since nothing after this step assumes anything about how it was obtained: Steps 2
# onwards verify whatever they are given, and a poor candidate shows up as ||t||_inf >= 1 and a
# refusal, never as a wrong enclosure. So where LAPACK's generalised `eigen` exists we use it,
# and where it does not, which is every type other than Float32/Float64 and their complex forms,
# we take the eigendecomposition of mid(B) \ mid(A) instead. That fails outright for a singular
# mid(B), and is a worse candidate for an ill-conditioned one; in both cases the verification
# declines rather than returning something unproved.
function _miyajima2014a_alg1_step1(Am::Matrix{CT}, Bm::Matrix{CT}) where {CT}
    try
        F = eigen(Am, Bm)
        return Matrix{CT}(F.vectors), Vector{CT}(F.values)
    catch e
        e isa MethodError || e isa SingularException || rethrow(e)
    end
    try
        F = eigen(Bm \ Am)
        return Matrix{CT}(F.vectors), Vector{CT}(F.values)
    catch e
        (e isa SingularException || e isa LAPACKException) && return nothing, CT[]
        rethrow(e)
    end
end

# Steps 2 and 3 of Algorithm 1, on the candidate of Step 1.
function _miyajima2014a_cor3_2(A::BallMatrix{T}, B::BallMatrix{T}) where {T}
    n = size(A, 1)
    CT = complex(T)
    Xt, lambdas = _miyajima2014a_alg1_step1(Matrix{CT}(mid(A)), Matrix{CT}(mid(B)))
    Xt === nothing && return nothing
    all(isfinite, Xt) && all(isfinite, lambdas) || return nothing
    Dt = Diagonal(lambdas)

    Xtb = BallMatrix(Xt)
    BXt = B * Xtb
    Y = try
        inv(mid(BXt))
    catch e
        e isa SingularException ? (return nothing) : rethrow(e)
    end
    all(isfinite, Y) || return nothing
    Yb = BallMatrix(Y)

    # R = Y(A Xt - B Xt Dt) and S = I - Y B Xt
    R = Yb * (A * Xtb - BXt * BallMatrix(Matrix{CT}(Dt)))
    YBXt = Yb * BXt
    S = I - YBXt

    absS, absR = upper_abs(S), upper_abs(R)
    t = setrounding(T, RoundUp) do
        vec(sum(absS, dims = 2))
    end
    tinf = maximum(t)
    tinf < 1 || return nothing                       # Step 2: terminate with failure
    uvec = setrounding(T, RoundUp) do
        vec(sum(absR, dims = 2))
    end
    # Corollary 3.2: r = u + <u>_t t
    au = _miyajima2014a_lem2_3(uvec, t)
    r = setrounding(T, RoundUp) do
        uvec .+ au .* t
    end
    return _Miyajima2014aDiscs(lambdas, r, t, absR, upper_abs(Xtb), Xt, YBXt, R, tinf)
end

# Step 5 of Algorithm 1: the connected components of the disc graph. Two discs are joined when
# they overlap, the distance being bounded BELOW and the radii above, so a component that comes
# back isolated is proved isolated.
function _miyajima2014a_rem3_3(lambdas::Vector{CT}, r::Vector{T}) where {T, CT}
    n = length(lambdas)
    parent = collect(1:n)
    find(x) = (while parent[x] != x; x = parent[x]; end; x)
    for i in 1:n, j in (i + 1):n
        overlap = dist_down(lambdas[i], lambdas[j]) <= add_up(r[i], r[j])
        if overlap
            a, b = find(i), find(j)
            a != b && (parent[a] = b)
        end
    end
    groups = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(groups, find(i), Int[]), i)
    end
    return sort(collect(values(groups)); by = first)
end

# Theorem 3.5: the eigenvector of an isolated disc, |x* - Xt[:,i]| <= |Xt| J_i q.
function _miyajima2014a_thm3_5(G::_Miyajima2014aDiscs{T, CT}, i::Int) where {T, CT}
    n = length(G.lambdas)
    keep = [j for j in 1:n if j != i]          # the action of I_i
    # f = |dt - lambda_i 1| - r_i 1, rounded DOWN
    f = T[sub_down(dist_down(G.lambdas[j], G.lambdas[i]), G.r[i]) for j in 1:n]
    all(>(0), view(f, keep)) || return nothing # Remark 3.6: isolation gives I_i f > 0
    w = G.absR[:, i]                           # |R| e_i
    v = setrounding(T, RoundUp) do             # |R| J_i 1 = row sums of |R| off column i
        vec(sum(G.absR, dims = 2)) .- w
    end
    av, aw = _miyajima2014a_lem2_3(v, G.t), _miyajima2014a_lem2_3(w, G.t)
    g = setrounding(T, RoundUp) do
        T[(v[j] + av * G.t[j]) / f[j] for j in keep]
    end
    maximum(g) < 1 || return nothing           # Remark 3.6: ||g||_inf < 1
    h = setrounding(T, RoundUp) do
        T[(w[j] + aw * G.t[j]) / f[j] for j in keep]
    end
    ah = _miyajima2014a_lem2_3(h, g)
    q = setrounding(T, RoundUp) do
        h .+ ah .* g
    end
    # |Xt| J_i q, an n-vector: J_i drops column i, so this is |Xt|[:, keep] * q
    rad = setrounding(T, RoundUp) do
        view(G.absXt, :, keep) * q
    end
    return BallMatrix(reshape(G.Xt[:, i], n, 1), reshape(rad, n, 1))
end

# Algorithm 2: the k eigenvalues and invariant subspace of a connected component `cl`.
# Returns (radius, subspace, block) or nothing on any of the paper's failure exits.
function _miyajima2014a_alg2(G::_Miyajima2014aDiscs{T, CT}, A::BallMatrix{T}, B::BallMatrix{T},
        cl::Vector{Int}) where {T, CT}
    n, k = length(G.lambdas), length(cl)
    incl = falses(n)
    incl[cl] .= true
    comp = [j for j in 1:n if !incl[j]]        # the index set u, the complement of the cluster

    # Step 1: lambda as the mean of the cluster, and lambda != lambda_j for every j outside
    lam = sum(G.lambdas[p] for p in cl) / k
    for j in comp
        dist_down(G.lambdas[j], lam) > 0 || return nothing
    end

    # R' = R + Y B Xt (Dt - Dt') by Remark 3.8; Dt - Dt' is zero off the cluster, and on it the
    # difference of two floats, which is enclosed as a ball rather than rounded to the nearest
    dd = zeros(CT, n)
    ddr = zeros(T, n)
    for p in cl
        b = Ball(G.lambdas[p]) - Ball(lam)
        dd[p], ddr[p] = mid(b), rad(b)
    end
    Rp = G.R + G.YBXt * BallMatrix(Matrix{CT}(Diagonal(dd)), Matrix{T}(Diagonal(ddr)))
    absRp = upper_abs(Rp)

    # Step 2: phi, nu' and mu of Lemma 3.9
    absphi = T[incl[j] ? one(T) : dist_down(G.lambdas[j], lam) for j in 1:n]
    all(>(0), absphi) || return nothing
    nup = setrounding(T, RoundUp) do            # nu' = |R'| U U^T 1: columns outside the cluster
        isempty(comp) ? zeros(T, n) : vec(sum(view(absRp, :, comp), dims = 2))
    end
    anup = _miyajima2014a_lem2_3(nup, G.t)
    mu = setrounding(T, RoundUp) do
        (nup .+ anup .* G.t) ./ absphi
    end
    maximum(mu) < 1 || return nothing

    # Step 3: Rw = |R'| V + t w^T, floored at sqrt(realmin) so no entry underflows to zero
    wcols = _miyajima2014a_cor2_4(view(absRp, :, cl), G.t)
    sqrtrealmin = sqrt(floatmin(T))
    Rw = setrounding(T, RoundUp) do
        M = view(absRp, :, cl) .+ G.t * transpose(wcols)
        max.(M, sqrtrealmin)
    end

    # Step 4: Pbar_eps = P*_eps, which is P_eps of Lemma 3.10 at P_r = 0, and sigma
    Tm = setrounding(T, RoundUp) do
        Rw ./ absphi                            # T = (Rw + 0) ./ (|phi| 1^T)
    end
    z = _miyajima2014a_cor2_4(Tm, mu)
    Pe = setrounding(T, RoundUp) do
        Tm .+ mu * transpose(z)
    end
    # sigma >= || (U U^T Pbar_eps)(V^T Pbar_eps) ./ Rw ||_M, the max over entries
    VtPe = Pe[cl, :]                            # V^T Pbar_eps, k by k
    UUtPe = copy(Pe)                            # U U^T Pbar_eps zeroes the cluster rows
    UUtPe[cl, :] .= zero(T)
    eps_T = eps(T)
    sigma = setrounding(T, RoundUp) do
        maximum((UUtPe * VtPe) ./ Rw)
    end
    onepeps = add_up(one(T), eps_T)
    c6 = mul_up(sigma, pow_up(onepeps, 6))
    c6 < one(T) / 4 || return nothing

    # Step 5: eta, and the upper bound it must respect; disc is sqrt(1 - 4 sigma (1+eps)^6)
    # bounded below, which makes eta larger and its upper bound smaller
    disc = sqrt_down(sub_down(one(T), mul_up(T(4), c6)))
    eta = div_up(mul_up(T(2), pow_up(onepeps, 3)), add_down(one(T), disc))
    etamax = div_down(add_down(one(T), disc), mul_up(T(2), mul_up(sigma, pow_up(onepeps, 4))))
    eta < etamax || return nothing

    # Step 6: Pbarbar_eps = (1 + sigma eta^2) Pbar_eps, then the eigenvalue disc and the subspace
    fac = add_up(one(T), mul_up(sigma, mul_up(eta, eta)))
    Pee = setrounding(T, RoundUp) do
        fac .* Pe
    end
    VtPee = Pee[cl, :]
    rho = collatz_upper_bound(BallMatrix(VtPee))
    isfinite(rho) || return nothing
    # the disc must be disjoint from the discs of every eigenvalue outside the cluster
    for j in comp
        dist_down(G.lambdas[j], lam) > add_up(rho, G.r[j]) || return nothing
    end

    UUtPee = copy(Pee)
    UUtPee[cl, :] .= zero(T)
    subrad = setrounding(T, RoundUp) do
        G.absXt * UUtPee
    end
    subspace = BallMatrix(G.Xt[:, cl], subrad)
    # the block itself: Lambda in <lambda I_k, V^T Pbarbar_eps>
    block = BallMatrix(Matrix{CT}(lam * I, k, k), VtPee)
    return rho, subspace, block
end

# The perturbation of the two-residual device, enclosed through a computed solution. With
# S = (I + R₂)⁻¹R₁, S̃ a floating-point solution of (I + R₂)S̃ = R₁ and V = (I + R₂)S̃ − R₁,
#
#     S − S̃ = −(I + R₂)⁻¹V,        ‖S − S̃‖_p ≤ ‖V‖_p / (1 − ‖R₂‖_p)   when ‖R₂‖_p < 1,
#
# for p = 2 and p = ∞, and an entry of a matrix is at most its norm in either. Returns `(S̃, e)`
# with `e` an upper bound of every entry of |S − S̃|, for every R₁ and R₂ of the two balls; `e` is
# `Inf` when neither norm of R₂ is proved below one. This is the lemma "Residual control of S" of
# I. Nisoli, *Enclosure of eigenvalues via approximate diagonalization* (unpublished note, 2026).
function _residual_controlled_perturbation(R1::BallMatrix{T}, R2::BallMatrix{T}) where {T}
    St = (I + mid(R2)) \ mid(R1)
    V = (R2 + I) * BallMatrix(St) - R1
    e = T(Inf)
    for nrm in (upper_bound_L2_opnorm, upper_bound_L_inf_opnorm)
        d = nrm(R2)
        d < 1 && (e = min(e, div_up(T(nrm(V)), sub_down(one(T), T(d)))))
    end
    return St, e
end

"""
    _miyajima2014a_alg1(A::BallMatrix, B::BallMatrix) -> VerifyEigAllResult

Algorithms 1 and 2 of Miyajima (2014): verified inclusions of all eigenvalues and invariant
subspaces of the pencil `A x = λ B x`. Not exported; reached through [`verifyeigall`](@ref) with
`method = :miyajima2014a`.

`B` is not assumed nonsingular; its nonsingularity, and that of `X̃` and `Y`, is proved by
`‖t‖_∞ < 1` with `t = |I − YBX̃|𝟙` (Theorem 3.1, Corollary 3.2). Once that holds, every
eigenvalue of the pencil lies in one of the `n` discs `⟨λ̃ᵢ, rᵢ⟩` with `r = u + ⟨u⟩_t t`, so
`spectrum_covered` reports that single condition and not the per-cluster tests: a cluster of `k`
connected discs contains precisely `k` eigenvalues by Remark 3.3 whether or not Algorithm 2 then
succeeds on it.

`certified[i]` is the stronger, per-cluster claim. For a singleton it is Theorem 3.5: the disc
holds exactly one eigenvalue, of geometric multiplicity one, and `subspaces[i]` encloses an
eigenvector for it. For a cluster of size `k` it is Theorem 3.11 via Algorithm 2: `radii[i]` is
`ρ(V ᵀP̄̄_ε)`, `subspaces[i]` encloses a basis of the invariant subspace, and `blocks[i]` encloses
the block `Λ ∈ ⟨λ̃I_k, V ᵀP̄̄_ε⟩`.

Where Algorithm 2 declines, `radii[i]` falls back to the smallest disc about `λ̃` containing the
union of the cluster's Gershgorin discs, which is still a rigorous enclosure of those `k`
eigenvalues, while `subspaces[i]` and `blocks[i]` carry `Inf`, since no subspace was proved.

The result also carries `similarity`, the matrix `X̃`, and `transformed`, a ball matrix containing
`X̃⁻¹B⁻¹AX̃ = D̃ + S`. Here `S = (I + R₂)⁻¹R₁` with the paper's two residuals
`R₁ = Y(AX̃ − BX̃D̃)` and `R₂ = YBX̃ − I`, and it is enclosed through a floating-point solution of
`(I + R₂)S̃ = R₁` and the residual of that solve. This is an addition to the paper's algorithm,
which works with the norms of the rows of `R₁`; the discs and the clusters above do not use it.

# Reference

S. Miyajima, *Fast Enclosure for All Eigenvalues and Invariant Subspaces in Generalized
Eigenvalue Problems*, SIAM J. Matrix Anal. Appl. **35**(3):1205-1225, 2014,
doi 10.1137/140953150.
"""
function _miyajima2014a_alg1(A::BallMatrix{T}, B::BallMatrix{T}) where {T}
    n = size(A, 1)
    CT = complex(T)
    G = _miyajima2014a_cor3_2(A, B)
    G === nothing && return VerifyEigAllResult(Vector{Int}[], Bool[], CT[], T[],
        BallMatrix{T, CT}[], BallMatrix{T, CT}[], Matrix{CT}(I, n, n), false, 0, T(Inf))

    clusters = _miyajima2014a_rem3_3(G.lambdas, G.r)
    m = length(clusters)
    certified = falses(m)
    centers = Vector{CT}(undef, m)
    radii = fill(T(Inf), m)
    subspaces = Vector{BallMatrix{T, CT}}(undef, m)
    blocks = Vector{BallMatrix{T, CT}}(undef, m)

    for (idx, cl) in enumerate(clusters)
        k = length(cl)
        if k == 1
            i = cl[1]
            centers[idx] = G.lambdas[i]
            radii[idx] = G.r[i]
            v = _miyajima2014a_thm3_5(G, i)
            if v === nothing
                subspaces[idx] = BallMatrix(zeros(CT, n, 1), fill(T(Inf), n, 1))
                blocks[idx] = BallMatrix(zeros(CT, 1, 1), fill(T(Inf), 1, 1))
            else
                certified[idx] = true
                subspaces[idx] = v
                blocks[idx] = BallMatrix(reshape([G.lambdas[i]], 1, 1),
                    reshape([G.r[i]], 1, 1))
            end
        else
            lam = sum(G.lambdas[p] for p in cl) / k
            centers[idx] = lam
            out = _miyajima2014a_alg2(G, A, B, cl)
            if out === nothing
                # the union of the cluster's discs, as one disc about lambda: still rigorous
                radii[idx] = maximum(add_up(dist_up(G.lambdas[p], lam), G.r[p]) for p in cl)
                subspaces[idx] = BallMatrix(zeros(CT, n, k), fill(T(Inf), n, k))
                blocks[idx] = BallMatrix(zeros(CT, k, k), fill(T(Inf), k, k))
            else
                rho, subspace, block = out
                certified[idx] = true
                radii[idx] = rho
                subspaces[idx] = subspace
                blocks[idx] = block
            end
        end
    end

    St, e = _residual_controlled_perturbation(G.R, G.YBXt - I)
    transformed = isfinite(e) ?
                  BallMatrix(Matrix{CT}(Diagonal(G.lambdas))) +
                  BallMatrix(Matrix{CT}(St), fill(e, n, n)) : BallMatrix(zeros(CT, 0, 0))
    return VerifyEigAllResult(clusters, collect(certified), centers, radii, subspaces, blocks,
        G.Xt, true, 0, G.tinf, CT[], T[], BallMatrix(Matrix{CT}(G.Xt)), transformed)
end
