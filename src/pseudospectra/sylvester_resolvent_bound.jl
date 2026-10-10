# The resolvent of a block upper triangular matrix through a block-diagonalising similarity.
#
# Let T = [T₁₁ T₁₂; 0 T₂₂] with T₁₁ of size k×k and T₂₂ upper triangular, and let X be ANY k×(n−k)
# matrix. With S(X) = [I X; 0 I], whose inverse is S(−X),
#
#     S(X)⁻¹ T S(X) = [T₁₁ R; 0 T₂₂],        R := T₁₂ + T₁₁X − XT₂₂,
#
# an identity, exact for every X. For z outside the spectrum, with A_z := zI − T₁₁ and
# D_z := zI − T₂₂,
#
#     (zI − T)⁻¹ = S(X) [A_z⁻¹  A_z⁻¹ R D_z⁻¹; 0  D_z⁻¹] S(X)⁻¹,
#
# and bounding the three blocks separately gives
#
#     ‖(zI − T)⁻¹‖₂ ≤ κ₂(S(X)) ( ‖A_z⁻¹‖₂ + ‖D_z⁻¹‖₂ + ‖A_z⁻¹ R D_z⁻¹‖₂ ).               (*)
#
# This is the bound for a block triangular matrix (the inverse of [P Q; 0 W] is
# [P⁻¹ −P⁻¹QW⁻¹; 0 W⁻¹]) applied after the similarity S(X), which costs the factor
# κ₂(S(X)) = ψ(‖X‖₂)² (see `psi_squared`). X is chosen as an approximate solution of the Sylvester
# equation T₁₁X − XT₂₂ = −T₁₂, which makes R small; nothing is assumed about how good it is, since
# R is enclosed for the X actually used.
#
# The three terms of (*) are obtained by different means:
#   ‖A_z⁻¹‖₂         the reciprocal of a certified lower bound of σ_min(A_z), on the k×k block only;
#   ‖D_z⁻¹‖₂         from the triangular structure, with no singular value computation;
#   ‖A_z⁻¹RD_z⁻¹‖₂   by the product of the norms, or through an approximate solve with its residual.
#
# T is taken with exact entries (a point matrix). Every quantity that enters (*) is enclosed: the
# shifts z − t_ii, the residual R and the residuals of the approximate solves are computed in ball
# arithmetic, and the scalar operations are rounded outward.

"""
    SylvesterResolventResult{T}

The quantities of the bound of [`parametric_resolvent_bound`](@ref) that do not depend on `z`, for
the split `T = [T₁₁ T₁₂; 0 T₂₂]` at index `k` and an approximate solution `X` of
`T₁₁X − XT₂₂ = −T₁₂`.

# Fields
- `residual_norm`: upper bound of `‖R‖₂`, `R = T₁₂ + T₁₁X − XT₂₂`
- `coupling_norm`: upper bound of `‖T₁₂‖₂`
- `similarity_cond`: upper bound of `κ₂(S(X))`
- `reduction_factor`, `net_penalty`: the diagnostics `residual_norm/coupling_norm` and its product
  with `similarity_cond`, in plain floating point; they enter no bound
- `k`, `n`: the split index and the size of `T`
- `precomputation_success`, `failure_reason`
- `X`: the matrix `X` used
- `R`: a `BallMatrix` containing `R`
"""
struct SylvesterResolventResult{T}
    residual_norm::T
    coupling_norm::T
    similarity_cond::T
    reduction_factor::T
    net_penalty::T
    k::Int
    n::Int
    precomputation_success::Bool
    failure_reason::String
    X::Union{Nothing, Matrix}
    R::Union{Nothing, BallMatrix}
end

"""
    solve_sylvester_oracle(T11::AbstractMatrix, T12::AbstractMatrix, T22::AbstractMatrix)

An approximate solution `X` of the Sylvester equation `T₁₁X − XT₂₂ = −T₁₂`, in floating point: no
bound comes with it. It is computed by LAPACK in `Float64`; for other element types the data are
rounded to `Float64` and the solution converted back.
"""
function solve_sylvester_oracle(T11::AbstractMatrix{CT}, T12::AbstractMatrix{CT},
        T22::AbstractMatrix{CT}) where {CT <: Complex}
    # sylvester(A, B, C) solves AX + XB + C = 0, so this is T11*X − X*T22 = −T12
    real(CT) === Float64 && return sylvester(T11, -T22, T12)
    return CT.(sylvester(ComplexF64.(T11), -ComplexF64.(T22), ComplexF64.(T12)))
end

function solve_sylvester_oracle(T11::AbstractMatrix{T}, T12::AbstractMatrix{T},
        T22::AbstractMatrix{T}) where {T <: Real}
    T === Float64 && return sylvester(T11, -T22, T12)
    return T.(sylvester(Float64.(T11), -Float64.(T22), Float64.(T12)))
end

# the three blocks of T at the split k, as complex matrices (an exact conversion)
function _split_blocks(T::AbstractMatrix, k::Int)
    n = size(T, 1)
    CT = Complex{float(real(eltype(T)))}
    return CT.(T[1:k, 1:k]), CT.(T[1:k, (k + 1):n]), CT.(T[(k + 1):n, (k + 1):n])
end

# a ball matrix containing zI − M, for M with exact entries: the rounding of z − m_ii is in it
function _shifted_ball(z::Complex{RT}, M::AbstractMatrix{Complex{RT}}) where {RT}
    c = -Matrix(M)                       # exact
    r = zeros(RT, size(M))
    for i in axes(M, 1)
        d = Ball(z, zero(RT)) - Ball(M[i, i], zero(RT))
        c[i, i] = mid(d)
        r[i, i] = rad(d)
    end
    return BallMatrix(c, r)
end

"""
    sylvester_resolvent_precompute(T, k; X_oracle = nothing)

The part of the bound of [`parametric_resolvent_bound`](@ref) that does not depend on `z`, for the
split of `T` at index `k`: `T₁₁ = T[1:k, 1:k]`, `T₂₂ = T[k+1:end, k+1:end]`.

`T` must have exact entries, be zero in the block `[k+1:end, 1:k]`, and have `T₂₂` upper
triangular. `X_oracle` is the matrix `X` of the similarity; when it is not given,
[`solve_sylvester_oracle`](@ref) computes one. Any `X` gives a valid bound: the residual
`R = T₁₂ + T₁₁X − XT₂₂` is enclosed in ball arithmetic for the `X` used, and its norm and the
norm of `X` are bounded from above (`upper_bound_L2_opnorm`).

Returns a [`SylvesterResolventResult`](@ref); `precomputation_success` is `false`, with the
reason, when the structure is not the required one or `X` is not finite.
"""
function sylvester_resolvent_precompute(T::AbstractMatrix{ET}, k::Int;
        X_oracle::Union{Nothing, AbstractMatrix} = nothing) where {ET}
    n = size(T, 1)
    n == size(T, 2) || throw(ArgumentError("T must be square"))
    1 ≤ k < n || throw(ArgumentError("k must satisfy 1 ≤ k < n"))
    RT = float(real(ET))
    fail(reason) = SylvesterResolventResult{RT}(RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf),
        k, n, false, reason, nothing, nothing)

    # the identity behind the bound needs these zeros exactly: an entry below the diagonal, however
    # small, is not covered
    iszero(view(T, (k + 1):n, 1:k)) || return fail("the block T[k+1:n, 1:k] is not zero")
    istriu(view(T, (k + 1):n, (k + 1):n)) || return fail("T22 block is not upper triangular")

    T11, T12, T22 = _split_blocks(T, k)
    X = if X_oracle !== nothing
        size(X_oracle) == (k, n - k) || throw(DimensionMismatch("X_oracle must be k×(n−k)"))
        Matrix(Complex{RT}.(X_oracle))
    else
        try
            Matrix(solve_sylvester_oracle(T11, T12, T22))
        catch e
            return fail("Sylvester solve failed: $(e)")
        end
    end
    all(isfinite, X) || return fail("the approximate Sylvester solution is not finite")

    bX = BallMatrix(X)
    R = BallMatrix(T12) + BallMatrix(T11) * bX - bX * BallMatrix(T22)
    r = upper_bound_L2_opnorm(R)
    t12 = upper_bound_L2_opnorm(BallMatrix(T12))
    K_S = similarity_condition_number(bX)
    reduction = t12 > 0 ? r / t12 : RT(Inf)
    return SylvesterResolventResult{RT}(RT(r), RT(t12), RT(K_S), RT(reduction),
        RT(K_S * reduction), k, n, true, "", X, R)
end

#==============================================================================#
# The choices
#==============================================================================#

"""
    NormEstimator

The upper bound of the spectral norm used for the matrices of the approximate solves.

- `OneInfNorm`: `√(‖·‖₁‖·‖_∞)`
- `FrobeniusNorm`: `‖·‖_F`
"""
@enum NormEstimator begin
    OneInfNorm
    FrobeniusNorm
end

"""
    DInverseEstimator

How `‖D_z⁻¹‖₂` is bounded for the upper triangular `D_z = zI − T₂₂`.

- `TriBacksub`: the recursion of [`triangular_inverse_two_norm_bound`](@ref)
- `NeumannOneInf`: `D_z = Δ(I + N)` with `Δ` the diagonal of `D_z` and `N = Δ⁻¹(D_z − Δ)`, so that
  `‖D_z⁻¹‖_p ≤ ‖Δ⁻¹‖_p/(1 − ‖N‖_p)` when `‖N‖_p < 1`, for `p = 1` and `p = ∞`, combined by
  `‖·‖₂ ≤ √(‖·‖₁‖·‖_∞)`
- `NeumannCollatz2`: the same factorisation in the spectral norm, `‖N‖₂` bounded by
  [`collatz_upper_bound_L2_opnorm`](@ref) applied to an entrywise upper bound of `|N|`
"""
@enum DInverseEstimator begin
    TriBacksub
    NeumannOneInf
    NeumannCollatz2
end

"""
    CouplingEstimator

How the off-diagonal term `‖A_z⁻¹ R D_z⁻¹‖₂` is bounded.

- `CouplingNone`: `‖A_z⁻¹‖₂‖R‖₂‖D_z⁻¹‖₂`
- `CouplingARSolve`: `‖A_z⁻¹R‖₂‖D_z⁻¹‖₂`, with `A_z⁻¹R = Y + A_z⁻¹(R − A_zY)` for an approximate
  solution `Y` of `A_zY = R`, so `‖A_z⁻¹R‖₂ ≤ ‖Y‖₂ + ‖A_z⁻¹‖₂‖R − A_zY‖₂`
- `CouplingOffDirect`: `A_z⁻¹RD_z⁻¹ = W + A_z⁻¹(R − A_zWD_z)D_z⁻¹` for an approximate `W`, so the
  term is at most `‖W‖₂ + ‖A_z⁻¹‖₂‖R − A_zWD_z‖₂‖D_z⁻¹‖₂`

In the last two the residual is computed in ball arithmetic; when the approximate solve does not
return finite numbers the first bound is used.
"""
@enum CouplingEstimator begin
    CouplingNone
    CouplingARSolve
    CouplingOffDirect
end

"""
    ResolventBoundConfig(norm_estimator, d_inv_estimator, coupling_estimator, miyajima_method,
                         power_iterations, fallback_to_tri)

The choices of [`parametric_resolvent_bound`](@ref): a [`NormEstimator`](@ref), a
[`DInverseEstimator`](@ref), a [`CouplingEstimator`](@ref), the method for the certified singular
values of the small block (`:M1` or `:M4`), the number of iterations of the Collatz bound, and
whether `TriBacksub` replaces a Neumann bound that does not apply (`‖N‖ ≥ 1`).

`config_v1()`, `config_v2()`, `config_v2p5()` and `config_v3()` are the combinations in use.
"""
struct ResolventBoundConfig
    norm_estimator::NormEstimator
    d_inv_estimator::DInverseEstimator
    coupling_estimator::CouplingEstimator
    miyajima_method::Symbol
    power_iterations::Int
    fallback_to_tri::Bool
end

"`TriBacksub` and the product bound for the coupling."
config_v1() = ResolventBoundConfig(OneInfNorm, TriBacksub, CouplingNone, :M1, 3, true)
"`TriBacksub` and `CouplingARSolve`."
config_v2() = ResolventBoundConfig(OneInfNorm, TriBacksub, CouplingARSolve, :M1, 3, true)
"`TriBacksub` and `CouplingOffDirect`."
config_v2p5() = ResolventBoundConfig(OneInfNorm, TriBacksub, CouplingOffDirect, :M1, 3, true)
"`NeumannCollatz2` and `CouplingARSolve`."
config_v3() = ResolventBoundConfig(OneInfNorm, NeumannCollatz2, CouplingARSolve, :M1, 3, true)

"""
    estimate_2norm(M, method::NormEstimator)

An upper bound of `‖M‖₂` for a `BallMatrix` (or a matrix with exact entries) by the chosen
[`NormEstimator`](@ref).
"""
estimate_2norm(M::AbstractMatrix, method::NormEstimator) = estimate_2norm(BallMatrix(M), method)
function estimate_2norm(M::BallMatrix, method::NormEstimator)
    method == FrobeniusNorm && return upper_bound_norm(M, 2)
    return sqrt_up(mul_up(upper_bound_L1_opnorm(M), upper_bound_L_inf_opnorm(M)))
end

#==============================================================================#
# The large block
#==============================================================================#

# Upper bound of ‖(zI − T22)⁻¹‖₂ for the upper triangular T22 with exact entries. `dlo[i]` is a
# lower bound of |z − t_ii| and `absU` an entrywise upper bound of |T22|, of which only the strict
# upper triangle is read. Inf when the estimator does not apply.
function _large_block_inverse_bound(dlo::Vector{RT}, absU::Matrix{RT},
        estimator::DInverseEstimator, power_iterations::Int) where {RT}
    all(d -> isfinite(d) && d > 0, dlo) || return RT(Inf)
    estimator == TriBacksub &&
        return _two_norm_from_one_inf(_triangular_inverse_bounds(dlo, absU)...)

    # entrywise upper bound of |N|, N = Δ⁻¹(D_z − Δ), and of ‖Δ⁻¹‖ (the same in every p-norm)
    m = length(dlo)
    inv_d = [div_up(one(RT), d) for d in dlo]
    Δ_inv = maximum(inv_d; init = zero(RT))
    Nup = zeros(RT, m, m)
    for j in 2:m, i in 1:(j - 1)
        Nup[i, j] = mul_up(absU[i, j], inv_d[i])
    end
    bN = BallMatrix(Nup)
    neumann(α) = α < 1 ? div_up(Δ_inv, sub_down(one(RT), α)) : RT(Inf)
    if estimator == NeumannOneInf
        return _two_norm_from_one_inf(neumann(upper_bound_L1_opnorm(bN)),
            neumann(upper_bound_L_inf_opnorm(bN)))
    end
    # |N| ≤ Nup entrywise gives ‖N‖₂ ≤ ‖Nup‖₂
    return neumann(collatz_upper_bound_L2_opnorm(bN; iterates = power_iterations))
end

#==============================================================================#
# The bound at a point
#==============================================================================#

"""
    ParametricResolventResult{T}

The bound of [`parametric_resolvent_bound`](@ref) at a point `z` and its terms: `resolvent_bound`
(upper bound of `‖(zI − T)⁻¹‖₂`), `K_S` (of `κ₂(S(X))`), `M_A` (of `‖(zI − T₁₁)⁻¹‖₂`), `M_D` (of
`‖(zI − T₂₂)⁻¹‖₂`), `r` (of `‖R‖₂`) and `coupling_term` (of `‖(zI − T₁₁)⁻¹R(zI − T₂₂)⁻¹‖₂`), with
the `config` used. When `success` is `false` the numbers are `Inf` and `failure_reason` says which
step did not apply.
"""
struct ParametricResolventResult{T}
    z::Complex{T}
    resolvent_bound::T
    success::Bool
    failure_reason::String
    config::ResolventBoundConfig
    K_S::T
    M_A::T
    M_D::T
    r::T
    coupling_term::T
end

"""
    SVDWarmStart(U, Σ, V)

An approximate singular value decomposition of `zI − T₁₁` at a nearby point, refined (Ogita and
Aishima) and then certified in place of a fresh decomposition. It only changes the approximation
the certificate is computed with.
"""
struct SVDWarmStart{UT, ST, VT}
    U::UT
    Σ::ST
    V::VT
end

"""
    parametric_resolvent_bound(precomp, T, z, config = config_v1(); svd_warm_start = nothing)

An upper bound of `‖(zI − T)⁻¹‖₂` for the block upper triangular matrix `T` with exact entries,
through the similarity `S(X) = [I X; 0 I]` of `precomp = sylvester_resolvent_precompute(T, k)`.

# The bound

With `S(X)⁻¹TS(X) = [T₁₁ R; 0 T₂₂]`, `R = T₁₂ + T₁₁X − XT₂₂` (an identity for every `X`),
`A_z = zI − T₁₁` and `D_z = zI − T₂₂`,

    (zI − T)⁻¹ = S(X) [A_z⁻¹  A_z⁻¹RD_z⁻¹; 0  D_z⁻¹] S(X)⁻¹,

    ‖(zI − T)⁻¹‖₂ ≤ κ₂(S(X)) ( ‖A_z⁻¹‖₂ + ‖D_z⁻¹‖₂ + ‖A_z⁻¹RD_z⁻¹‖₂ ).

It is the bound for a block triangular matrix after a similarity, so it carries the condition
number `κ₂(S(X)) = ψ(‖X‖₂)²` ([`psi_squared`](@ref)) where the bound computed directly on `T`
carries none; in exchange the certified singular value computation is on the `k×k` block only.

# The terms

1. `‖A_z⁻¹‖₂ ≤ 1/σ`, `σ` a certified lower bound of the smallest singular value of the ball
   matrix `zI − T₁₁` ([`rigorous_svd`](@ref) with the method of `config`).
2. `‖D_z⁻¹‖₂` by the [`DInverseEstimator`](@ref) of `config`.
3. `‖A_z⁻¹RD_z⁻¹‖₂` by the [`CouplingEstimator`](@ref) of `config`.

The moduli `|z − t_ii|` are rounded down, `R` and the residuals of the approximate solves are
ball matrices, and the scalar operations are rounded up.

Returns a [`ParametricResolventResult`](@ref). `T` must be the matrix `precomp` was computed from.
"""
function parametric_resolvent_bound(precomp::SylvesterResolventResult{RT}, T::AbstractMatrix,
        z::Number, config::ResolventBoundConfig = config_v1();
        svd_warm_start::Union{Nothing, SVDWarmStart} = nothing) where {RT}
    zc = Complex{RT}(z)
    fail(reason) = ParametricResolventResult{RT}(zc, RT(Inf), false, reason, config,
        RT(Inf), RT(Inf), RT(Inf), RT(Inf), RT(Inf))
    precomp.precomputation_success ||
        return fail("Precomputation failed: $(precomp.failure_reason)")
    size(T) == (precomp.n, precomp.n) ||
        throw(DimensionMismatch("T is not the matrix of the precomputation"))

    k = precomp.k
    r = precomp.residual_norm
    K_S = precomp.similarity_cond
    T11, _, T22 = _split_blocks(T, k)
    m = precomp.n - k

    # 1. the small block: certified σ_min of the ball zI − T11
    A_ball = _shifted_ball(zc, T11)
    M_A = try
        method = config.miyajima_method == :M4 ? MiyajimaM4() : MiyajimaM1()
        svd_result = if svd_warm_start !== nothing
            refined = ogita_svd_refine(A_ball.c, svd_warm_start.U, svd_warm_start.Σ,
                svd_warm_start.V; max_iterations = 2, precision_bits = precision(RT),
                check_convergence = false)
            Σ_vec = refined.Σ isa Diagonal ? diag(refined.Σ) : refined.Σ
            _certify_svd(A_ball, SVD(Matrix(refined.U), Vector(Σ_vec), Matrix(refined.V')),
                method; apply_vbd = true)
        else
            rigorous_svd(A_ball; method)
        end
        σ = svd_result.singular_values[end]
        σ_lo = sub_down(RT(mid(σ)), RT(rad(σ)))
        (isfinite(σ_lo) && σ_lo > 0) || return fail("σ_min(A_z) is not proved positive")
        div_up(one(RT), σ_lo)
    catch e
        return fail("Miyajima SVD failed: $e")
    end

    # 2. the large block
    dlo = [dist_down(zc, T22[i, i]) for i in 1:m]
    absU = abs_up.(T22)
    M_D = _large_block_inverse_bound(dlo, absU, config.d_inv_estimator, config.power_iterations)
    if !isfinite(M_D) && config.fallback_to_tri && config.d_inv_estimator != TriBacksub
        M_D = _large_block_inverse_bound(dlo, absU, TriBacksub, config.power_iterations)
    end
    isfinite(M_D) || return fail("D_z inverse bound failed")

    # 3. the coupling
    product = mul_up(mul_up(M_A, r), M_D)
    coupling = product
    R = precomp.R
    norm_of(M) = RT(estimate_2norm(M, config.norm_estimator))
    if config.coupling_estimator == CouplingARSolve
        Y = try
            A_ball.c \ R.c
        catch
            nothing
        end
        if Y !== nothing && all(isfinite, Y)
            bY = BallMatrix(Y)
            M_AR = add_up(norm_of(bY), mul_up(M_A, norm_of(R - A_ball * bY)))
            coupling = min(product, mul_up(M_AR, M_D))
        end
    elseif config.coupling_estimator == CouplingOffDirect
        D_ball = _shifted_ball(zc, T22)
        W = try
            (A_ball.c \ R.c) / D_ball.c
        catch
            nothing
        end
        if W !== nothing && all(isfinite, W)
            bW = BallMatrix(W)
            Δ = norm_of(R - A_ball * bW * D_ball)
            coupling = min(product, add_up(norm_of(bW), mul_up(mul_up(M_A, Δ), M_D)))
        end
    end
    isfinite(coupling) || return fail("the coupling bound is not finite")

    bound = mul_up(K_S, add_up(add_up(M_A, M_D), coupling))
    return ParametricResolventResult{RT}(zc, bound, true, "", config, K_S, M_A, M_D, r, coupling)
end

"""
    parametric_resolvent_bound(precomp, T, z_list::AbstractVector, config = config_v1())

The bound at each point of `z_list`.
"""
parametric_resolvent_bound(precomp::SylvesterResolventResult, T::AbstractMatrix,
    z_list::AbstractVector, config::ResolventBoundConfig = config_v1()) =
    [parametric_resolvent_bound(precomp, T, z, config) for z in z_list]

"""
    parametric_resolvent_bound(T, k::Int, z, config = config_v1(); X_oracle = nothing)

Precompute for the split `k` and evaluate at `z` (a point or a vector of points). Returns
`(precomp, result)`.
"""
function parametric_resolvent_bound(T::AbstractMatrix, k::Int, z,
        config::ResolventBoundConfig = config_v1(); X_oracle = nothing)
    precomp = sylvester_resolvent_precompute(T, k; X_oracle)
    return precomp, parametric_resolvent_bound(precomp, T, z, config)
end

"""
    find_optimal_split(T, z; k_range = 2:min(size(T, 1) - 2, 50), config = config_v1())

The split index in `k_range` with the smallest bound at `z`, as `(k, precomp, result)`; `nothing`
when the bound applies for no index of the range.
"""
function find_optimal_split(T::AbstractMatrix, z::Number;
        k_range = 2:min(size(T, 1) - 2, 50), config::ResolventBoundConfig = config_v1())
    best = nothing
    for k in k_range
        precomp, result = parametric_resolvent_bound(T, k, z, config)
        result.success || continue
        if best === nothing || result.resolvent_bound < best[3].resolvent_bound
            best = (k, precomp, result)
        end
    end
    return best
end

"""
    compare_all_configs(T, k, z; miyajima_method = :M1, power_iterations = 3)

The bound at `z` for `config_v1()`, `config_v2()`, `config_v2p5()` and `config_v3()` on one
precomputation. Returns `(; precomp, results, bounds, best, z)`, the middle two being dictionaries
indexed by `"V1"`, `"V2"`, `"V2.5"`, `"V3"` (`Inf` where a configuration does not apply) and
`best` the name with the smallest bound.
"""
function compare_all_configs(T::AbstractMatrix, k::Int, z::Number;
        miyajima_method::Symbol = :M1, power_iterations::Int = 3)
    precomp = sylvester_resolvent_precompute(T, k)
    results = Dict{String, ParametricResolventResult}()
    bounds = Dict{String, Float64}()
    for (name, cfg) in (("V1", config_v1()), ("V2", config_v2()), ("V2.5", config_v2p5()),
        ("V3", config_v3()))
        adjusted = ResolventBoundConfig(cfg.norm_estimator, cfg.d_inv_estimator,
            cfg.coupling_estimator, miyajima_method, power_iterations, cfg.fallback_to_tri)
        result = parametric_resolvent_bound(precomp, T, z, adjusted)
        results[name] = result
        bounds[name] = result.success ? Float64(result.resolvent_bound) : Inf
    end
    return (; precomp, results, bounds, best = findmin(bounds)[2], z)
end
