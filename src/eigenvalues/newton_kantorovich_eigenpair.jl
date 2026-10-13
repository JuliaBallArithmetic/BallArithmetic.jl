# Enclosure of one eigenpair of a matrix by a contraction argument on the map
#
#     F(z, w) = (A w − z w, uᴴ w − 1),        (z, w) ∈ ℂ × ℂᴺ,
#
# with u a fixed vector. The statement and its proof are in the docstring of
# `certify_eigenpair`; nothing is taken from a reference.

using LinearAlgebra

"""
    NKEigenpairResult{T, CT}

Result of Newton-Kantorovich certification for a single eigenpair.

# Fields
- `certified::Bool`: whether the eigenpair was rigorously certified
- `eigenvalue::Ball{T,CT}`: certified eigenvalue enclosure
- `eigenvector::Vector{Ball{T,CT}}`: certified eigenvector enclosure
- `enclosure_radius::T`: NK ball radius in the product space
- `defect_q::T`: q₀ = ‖I - C·J‖₂
- `C_norm::T`: ‖C‖₂
- `residual_y::T`: y = ‖C·F(x₀)‖₂
- `discriminant::T`: (1-q₀)² - 4·‖C‖₂·y
"""
struct NKEigenpairResult{T<:AbstractFloat, CT<:Union{T, Complex{T}}}
    certified::Bool
    eigenvalue::Ball{T, CT}
    eigenvector::Vector{Ball{T, CT}}
    enclosure_radius::T
    defect_q::T
    C_norm::T
    residual_y::T
    discriminant::T
end

"""
    NKEigenpairsResult{T, CT}

Batch result from `certify_eigenpairs`.

# Fields
- `results::Vector{NKEigenpairResult{T,CT}}`: individual results
- `n_certified::Int`: number of verified eigenpairs
- `n_total::Int`: total number of eigenpairs attempted
"""
struct NKEigenpairsResult{T<:AbstractFloat, CT<:Union{T, Complex{T}}}
    results::Vector{NKEigenpairResult{T, CT}}
    n_certified::Int
    n_total::Int
end

# Forward iteration/indexing to results
Base.length(r::NKEigenpairsResult) = length(r.results)
Base.getindex(r::NKEigenpairsResult, i) = r.results[i]
Base.iterate(r::NKEigenpairsResult) = iterate(r.results)
Base.iterate(r::NKEigenpairsResult, state) = iterate(r.results, state)
Base.eltype(::Type{NKEigenpairsResult{T,CT}}) where {T,CT} = NKEigenpairResult{T,CT}
Base.firstindex(r::NKEigenpairsResult) = 1
Base.lastindex(r::NKEigenpairsResult) = length(r.results)

"""
    certify_eigenpair(A, λ_approx, v_approx; u_approx=nothing, norm_method=:fast)

An enclosure of one eigenpair of `A` near the approximate pair `(λ̃, ṽ)`, valid for every matrix
of `A` when it is a ball matrix.

# What is proved

Let `u` be a vector with `uᴴṽ = 1` up to rounding (an approximate left eigenvector), and
`F(z, w) := (A w − z w, uᴴ w − 1)` on `ℂ × ℂᴺ` with the Euclidean norm. With `x₀ := (λ̃, ṽ)`,
`J := F′(x₀)`, a matrix `C` (the floating-point inverse of the midpoint of `J`) and

    y ≥ ‖C F(x₀)‖₂,        q₀ ≥ ‖I − C J‖₂,        c ≥ ‖C‖₂,

if `q₀ < 1` and `(1 − q₀)² − 4 c y ≥ 0`, then with `r := 2y/(1 − q₀)` there is exactly one
`(z, w)` with `‖(z − λ̃, w − ṽ)‖₂ ≤ r`, `A w = z w` and `uᴴ w = 1`; and `z` is a simple
eigenvalue of `A`. When `certified`, the result has `eigenvalue = ⟨λ̃, r⟩` and each component of
`eigenvector` of radius `r` about that of `ṽ`.

Proof. `F` is quadratic: `F(x₀ + h) = F(x₀) + J h + Q(h)` with `Q(h) = (−h_z h_w, 0)`, so
`‖Q(h)‖₂ = |h_z| ‖h_w‖₂ ≤ ‖h‖₂²/2` and `‖Q′(h)‖₂ ≤ ‖h‖₂`. For `T(x) := x − C F(x)`,

    ‖T(x₀ + h) − x₀‖₂ ≤ y + q₀ ‖h‖₂ + c ‖h‖₂²/2,        ‖T′(x₀ + h)‖₂ ≤ q₀ + c ‖h‖₂.

From `4cy ≤ (1 − q₀)²`, `c r ≤ (1 − q₀)/2`: the first bound at `‖h‖₂ ≤ r` is at most
`y + q₀ r + (1 − q₀) r/4 ≤ r`, and the second at most `(1 + q₀)/2 < 1`. So `T` maps the ball
of radius `r` into itself and is a contraction on it; its fixed point is the only zero of
`C F` there, and `C` is invertible because `‖I − CJ‖₂ < 1`. At the zero the derivative of `F`
is invertible, by the second bound, which excludes a second independent eigenvector `w′`
(then `(0, w′ − (uᴴw′) w)` would be in its kernel) and a generalised eigenvector `w′` with
`(A − z) w′ = w` (then `(1, w′ − (uᴴw′) w)` would be).

`y`, `q₀` and `c` are computed in ball arithmetic over the ball `A`, so the conclusion holds
for each of its matrices, with an eigenpair that depends on the matrix.

# Arguments
- `A`: square matrix (plain or `BallMatrix`)
- `λ_approx::Number`: approximate eigenvalue
- `v_approx::AbstractVector`: approximate eigenvector
- `u_approx`: optional left eigenvector (computed automatically if `nothing`)
- `norm_method::Symbol`: `:fast` uses `upper_bound_L2_opnorm`, `:svd` uses
  `svd_bound_L2_opnorm`

The enclosed eigenvector is the one normalised by `uᴴ w = 1`, with `u` the vector given, or
computed, divided by the conjugate of `uᴴ ṽ`.

# Returns
[`NKEigenpairResult`](@ref) with certification status and enclosure.
"""
function certify_eigenpair(
    A::Union{AbstractMatrix, BallMatrix},
    λ_approx::Number,
    v_approx::AbstractVector;
    u_approx::Union{Nothing, AbstractVector} = nothing,
    norm_method::Symbol = :fast
)
    # --- Step 1: Input prep ---
    A_ball = A isa BallMatrix ? A : BallMatrix(A)
    A_mid = mid(A_ball)
    N = size(A_mid, 1)
    size(A_mid, 1) == size(A_mid, 2) || throw(ArgumentError("A must be square"))

    T = real(eltype(A_mid))
    # Promote element type to handle complex eigenvalues of real matrices
    NT = promote_type(eltype(A_mid), typeof(λ_approx), eltype(v_approx))

    λ = convert(NT, λ_approx)
    v = convert(Vector{NT}, collect(v_approx))

    # --- Step 2: Left eigenvector ---
    # Convert A_mid to NT if needed (e.g., real A with complex eigenvalues)
    A_mid_nt = NT == eltype(A_mid) ? A_mid : convert(Matrix{NT}, A_mid)
    if u_approx === nothing
        u = _compute_left_eigenvector(A_mid_nt, λ, v, NT)
    else
        u = convert(Vector{NT}, collect(u_approx))
    end
    # Normalise so that uᴴv = 1: uᴴv is antilinear in u, so u is divided by the conjugate
    uv = LinearAlgebra.dot(u, v)
    if abs(uv) < eps(T) * 100
        return _unverified_result(T, NT, λ, v, N)
    end
    u = u / conj(uv)

    # --- Step 3: Augmented Jacobian J_mid ---
    J_mid = zeros(NT, N + 1, N + 1)
    J_mid[1:N, 1] .= -v
    J_mid[1:N, 2:N+1] .= A_mid_nt - λ * I
    J_mid[N+1, 1] = zero(NT)
    J_mid[N+1, 2:N+1] .= conj.(u)

    # --- Step 4: Preconditioner C = inv(J_mid) ---
    C = try
        inv(J_mid)
    catch e
        if e isa SingularException || e isa LAPACKException
            return _unverified_result(T, NT, λ, v, N)
        end
        rethrow(e)
    end

    # --- Step 5: Residual F(λ, v) in ball arithmetic ---
    # If NT is complex but A_ball is real, promote A_ball to complex
    A_ball_nt = if NT <: Complex && !(eltype(mid(A_ball)) <: Complex)
        BallMatrix(convert(Matrix{NT}, mid(A_ball)), rad(A_ball))
    else
        A_ball
    end
    v_ball = BallVector(v)
    λ_ball = Ball(λ)
    Av = A_ball_nt * v_ball
    λv = λ_ball * v_ball
    F1 = Av - λv  # N-length BallVector

    F2_ball = dot(BallVector(u), v_ball) - Ball(one(NT))

    # Assemble F_vec as (N+1)-length BallVector
    F_mid = zeros(NT, N + 1)
    F_rad = zeros(T, N + 1)
    F_mid[1:N] .= mid(F1)
    F_rad[1:N] .= rad(F1)
    F_mid[N+1] = mid(F2_ball)
    F_rad[N+1] = rad(F2_ball)
    F_vec = BallVector(F_mid, F_rad)

    # --- Step 6: y = ‖C·F‖₂ ---
    C_ball = BallMatrix(C)
    CF = C_ball * F_vec
    y = upper_bound_norm(CF, 2)

    # --- Step 7: q₀ = ‖I - C·J‖₂ ---
    J_ball = _build_augmented_jacobian_ball(A_ball_nt, λ, v, u, N, T, NT)
    I_aug = BallMatrix(Matrix{NT}(I, N + 1, N + 1))
    defect = C_ball * J_ball - I_aug
    q0 = _opnorm_bound(defect, norm_method)

    # --- Step 8: ‖C‖₂ ---
    C_norm = _opnorm_bound(C_ball, norm_method)

    # --- Step 9: NK conditions ---
    if q0 >= one(T)
        return _unverified_result_with_diagnostics(T, NT, λ, v, N, q0, C_norm, y)
    end

    disc = _nk_discriminant(q0, C_norm, y)

    if disc < zero(T)
        return _unverified_result_with_diagnostics(T, NT, λ, v, N, q0, C_norm, y)
    end

    # --- Step 10: Enclosure radius r ≤ 2y/(1-q₀) ---
    r = div_up(mul_up(T(2), y), sub_down(one(T), q0))

    # --- Step 11: Build enclosure ---
    eigenvalue = Ball(λ, r)
    eigenvector = [Ball(v[i], r) for i in 1:N]

    return NKEigenpairResult{T, NT}(
        true, eigenvalue, eigenvector, r, q0, C_norm, y, disc
    )
end

"""
    certify_eigenpairs(A; indices=nothing, norm_method=:fast)

Certify all (or selected) eigenpairs of matrix `A`.

# Arguments
- `A`: square matrix (plain or `BallMatrix`)
- `indices`: optional vector of eigenvalue indices to certify
- `norm_method::Symbol`: `:fast` or `:svd`

The approximate eigenpairs come from the eigendecomposition of the midpoint, the Hermitian one
when the midpoint is Hermitian; each is then certified by [`certify_eigenpair`](@ref), for every
matrix of the ball.

# Returns
[`NKEigenpairsResult`](@ref) containing individual results plus summary counts.
"""
function certify_eigenpairs(
    A::Union{AbstractMatrix, BallMatrix};
    indices::Union{Nothing, AbstractVector{Int}} = nothing,
    norm_method::Symbol = :fast
)
    A_ball = A isa BallMatrix ? A : BallMatrix(A)
    A_mid = mid(A_ball)
    N = size(A_mid, 1)

    # The approximate eigenpairs that are then certified. They are an oracle: the certification
    # is the same whatever they are and is for every matrix of the ball, so the best one
    # available is taken, the Hermitian eigendecomposition when the midpoint is Hermitian.
    hermitian = ishermitian(A_mid)
    if hermitian
        eig = eigen(Hermitian(A_mid))
    else
        eig = eigen(A_mid)
    end

    V = eig.vectors
    λs = eig.values

    # Determine result element type from eigenvalues (may be complex for real A)
    RT = real(eltype(A_mid))
    # the eigenvalues of a Hermitian midpoint are real numbers while the certified pair of a
    # complex matrix is complex
    ET = promote_type(eltype(A_mid), eltype(λs))

    # Left eigenvectors
    if hermitian
        U = V  # For Hermitian, left = right (conjugate)
    else
        U_mat = inv(V)'
        U = U_mat
    end

    # Select indices
    idxs = indices === nothing ? (1:N) : indices

    results = NKEigenpairResult{RT, ET}[]
    for i in idxs
        u_i = U[:, i]
        result = certify_eigenpair(A_ball, λs[i], V[:, i];
                                   u_approx=u_i, norm_method=norm_method)
        push!(results, result)
    end

    n_certified = count(r -> r.certified, results)
    return NKEigenpairsResult{RT, ET}(results, n_certified, length(results))
end


# ─── Internal helpers ───────────────────────────────────────────────

"""
Compute left eigenvector closest to λ_approx from A'.
"""
function _compute_left_eigenvector(A_mid::AbstractMatrix{NT}, λ::NT,
                                    ::Vector{NT}, ::Type{NT}) where {NT}
    eig_adj = eigen(Matrix(A_mid'))
    # Find eigenvector of A' closest to conj(λ)
    λ_conj = conj(λ)
    dists = abs.(eig_adj.values .- λ_conj)
    idx = argmin(dists)
    return eig_adj.vectors[:, idx]
end

"""
Build the (N+1)×(N+1) augmented Jacobian as a BallMatrix.
The A-λI block inherits radii from A_ball; other blocks have zero radii.
"""
function _build_augmented_jacobian_ball(A_ball::BallMatrix, λ::NT, v::Vector{NT},
                                         u::Vector{NT}, N::Int,
                                         ::Type{T}, ::Type{NT}) where {T, NT}
    J_mid = zeros(NT, N + 1, N + 1)
    J_rad = zeros(T, N + 1, N + 1)

    A_mid_mat = mid(A_ball)
    A_rad_mat = rad(A_ball)

    # Block (1:N, 1): -v (exact, zero radii)
    J_mid[1:N, 1] .= -v

    # Block (1:N, 2:N+1): A - λI (inherits A's radii)
    J_mid[1:N, 2:N+1] .= A_mid_mat - λ * I
    J_rad[1:N, 2:N+1] .= A_rad_mat

    # Block (N+1, 1): 0 (exact)
    J_mid[N+1, 1] = zero(NT)

    # Block (N+1, 2:N+1): conj(u) (exact, zero radii)
    J_mid[N+1, 2:N+1] .= conj.(u)

    return BallMatrix(J_mid, J_rad)
end

# A lower bound of (1 − q₀)² − 4 c y for q₀ < 1: the square from below, the product from above.
# For q₀ ≥ 1 the value is only a diagnostic.
function _nk_discriminant(q0::T, c::T, y::T) where {T}
    a = sub_down(one(T), q0)
    sq = a >= 0 ? mul_down(a, a) : mul_up(a, a)
    return sub_down(sq, mul_up(mul_up(T(4), c), y))
end

"""
Compute ‖M‖₂ upper bound using the selected method.
"""
function _opnorm_bound(M::BallMatrix, method::Symbol)
    if method == :svd
        return svd_bound_L2_opnorm(M)
    else
        return upper_bound_L2_opnorm(M)
    end
end

"""
Return an unverified result with zero diagnostics.
"""
function _unverified_result(::Type{T}, ::Type{NT}, λ::NT, v::Vector{NT}, N::Int) where {T, NT}
    eigenvalue = Ball(λ, T(Inf))
    eigenvector = [Ball(v[i], T(Inf)) for i in 1:N]
    return NKEigenpairResult{T, NT}(
        false, eigenvalue, eigenvector,
        T(Inf), T(Inf), T(Inf), T(Inf), -one(T)
    )
end

"""
Return an unverified result with computed diagnostics.
"""
function _unverified_result_with_diagnostics(::Type{T}, ::Type{NT}, λ::NT, v::Vector{NT},
                                              N::Int, q0::T, C_norm::T, y::T) where {T, NT}
    disc = _nk_discriminant(q0, C_norm, y)
    eigenvalue = Ball(λ, T(Inf))
    eigenvector = [Ball(v[i], T(Inf)) for i in 1:N]
    return NKEigenpairResult{T, NT}(
        false, eigenvalue, eigenvector,
        T(Inf), q0, C_norm, y, disc
    )
end
