# Verified inclusions of ALL eigenvalues and invariant subspaces, following
#
#   S. M. Rump, "Verified error bounds for all eigenvalues and eigenvectors of a matrix",
#   SIAM J. Matrix Anal. Appl. 43(4):1736-1754, 2022.  Theorem 2.2 and the algorithm
#   `verifyeigall` of its section 2.
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
# The transformation: an inclusion A of W^{-1} B W
# ---------------------------------------------------------------------------------------------

# With R an approximate inverse of W, W^{-1} B W = (RW)^{-1}(RBW), so an enclosure of R B W
# together with ‖I − RW‖ < 1 gives one of W^{-1} B W by a Neumann series, the extra radius being
# ‖I − RW‖/(1 − ‖I − RW‖) times the norm of the enclosure. This is the same pattern the verified
# block diagonalisations in this package use for their basis residual.
function _veig_transform(B::BallMatrix{T}, W::Matrix{CT}) where {T, CT}
    R = inv(W)
    Rb = BallMatrix(R)
    Wb = BallMatrix(W)
    P = Rb * (B * Wb)                      # encloses R B W
    S = Rb * Wb - I                        # encloses R W − I
    defect = upper_bound_L2_opnorm(S)
    defect < 1 || return nothing, defect
    extra = setrounding(T, RoundUp) do
        defect * upper_bound_L2_opnorm(P) / (one(T) - defect)
    end
    A = BallMatrix(mid(P), setrounding(T, RoundUp) do
        rad(P) .+ extra
    end)
    return A, defect
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

# Z V_i ⊆ int(X V_i), column by column: (2.10). The left-hand side is rounded UP and compared
# against the stored radius, which is exact, so a true answer is a proof of the containment.
function _veig_contained(Z::BallMatrix{T}, X::BallMatrix{T}, cols) where {T}
    mZ, rZ, mX, rX = mid(Z), rad(Z), mid(X), rad(X)
    return setrounding(T, RoundUp) do
        for j in cols, i in axes(mZ, 1)
            abs(mZ[i, j] - mX[i, j]) + rZ[i, j] < rX[i, j] || return false
        end
        return true
    end
end

# rho(|M|) bounded from above by a few power iterations with Collatz's inclusion, as the paper
# does; the returned value is an upper bound on the spectral radius of the magnitude matrix.
function _veig_rho_mag(M::AbstractMatrix{T}; iters = 12) where {T}
    A = abs.(M)
    n = size(A, 1)
    n == 0 && return zero(T)
    x = ones(T, n)
    ρ = zero(T)
    for _ in 1:iters
        y = setrounding(T, RoundUp) do
            A * x
        end
        all(>(0), y) || return setrounding(T, RoundUp) do
            maximum(sum(A; dims = 2))       # fall back to the infinity norm
        end
        ρ = setrounding(T, RoundUp) do
            maximum(y ./ x)                  # Collatz: rho <= max_i (Ax)_i / x_i
        end
        x = y ./ maximum(y)
    end
    return ρ
end

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
function verifyeigall(B::BallMatrix{T, NT}; maxiter::Integer = 20,
        inflate::Real = 0.1) where {T, NT}
    n = size(B, 1)
    n == size(B, 2) || throw(ArgumentError("verifyeigall expects a square matrix"))

    F = eigen(Matrix{complex(T)}(mid(B)))
    W = Matrix{complex(T)}(F.vectors)
    CT = complex(T)

    A, defect = _veig_transform(B, W)
    A === nothing && return VerifyEigAllResult(Vector{Int}[], Bool[], CT[], T[], W, false, 0,
        defect)

    normA = upper_bound_L_inf_opnorm(A)
    dm = CT[mid(A)[i, i] for i in 1:n]
    dr = T[rad(A)[i, i] for i in 1:n]
    clusters = _veig_clusters(dm, dr, normA)

    # step 3: D and E = A − D
    D = dm
    Em = copy(mid(A))
    for i in 1:n
        Em[i, i] -= D[i]
    end
    E = BallMatrix(Em, rad(A))

    # step 4: Rtilde. Column j belongs to a cluster; entries in that cluster's rows are −1, the
    # rest are 1/(D_l − D_j).
    # The entries -1 are exact; the reciprocals are not. D[l] and D[j] are floats, so their exact
    # difference lies within one rounding of the computed one, and the reciprocal of that
    # enclosure is taken in ball arithmetic.
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
        return VerifyEigAllResult(clusters, fill(false, length(clusters)),
            [D[c[1]] for c in clusters], fill(T(Inf), length(clusters)), W, false, 0, defect)

    # step 5
    Y = _veig_hadamard(-RRm, RRr, E)

    # Theorem 2.2 is a statement about ONE Y: the set Phi of clusters satisfying (2.10), the rows
    # and columns J they occupy, and the requirement max{rho(Z) : Z in Z} < 1 on that submatrix.
    # Certification is therefore not accumulated across iterations; each iteration is judged on
    # its own Z, and the radii and the rho test are read off the same one.
    certified = falses(length(clusters))
    centers = CT[D[c[1]] for c in clusters]
    radii = fill(T(Inf), length(clusters))
    covered = false
    iters = 0
    for it in 1:maxiter
        iters = it
        X = _veig_inflate(Y; factor = T(inflate))
        XD, XO = _veig_split(X, clusters)
        Y2 = XO * XD - E - E * XO
        Z = _veig_hadamard(RRm, RRr, Y2)
        ok = [_veig_contained(Z, X, c) for c in clusters]
        if count(ok) >= count(certified)
            certified = BitVector(ok)
            fill!(radii, T(Inf))
            for (i, c) in enumerate(clusters)
                ok[i] || continue
                radii[i] = _veig_rho_mag(setrounding(T, RoundUp) do
                    abs.(mid(Z)[c, c]) .+ rad(Z)[c, c]
                end)
            end
            # Remark 2.4: rho(mag(Z)) < 1 on the rows and columns of the certified clusters is
            # what upgrades "each M_i is a Jordan block" to "their union is the whole spectrum".
            if all(ok)
                J = reduce(vcat, clusters)
                magZ = setrounding(T, RoundUp) do
                    abs.(mid(Z)[J, J]) .+ rad(Z)[J, J]
                end
                covered = _veig_rho_mag(magZ) < 1
            end
        end
        Y = Z
        (all(certified) && covered) && break
        all(isfinite, mid(Z)) || break      # the open clusters have diverged; stop
    end
    return VerifyEigAllResult(clusters, collect(certified), centers, radii, W,
        covered, iters, defect)
end
