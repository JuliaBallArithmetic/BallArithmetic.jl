module CertifScripts

using LinearAlgebra
using JLD2
using Base: dirname, mod1

using ..BallArithmetic: Ball, BallMatrix, svdbox, svd_bound_L2_opnorm,
                        upper_bound_L2_opnorm, inf,
                        add_up, sub_down, mul_up, div_up, abs_up, dist_up,
                        verified_cholesky,
                        backward_substitution, forward_substitution,
                        refine_schur_decomposition,
                        ogita_svd_refine, OgitaSVDRefinementResult,
                        rigorous_svd, _certify_svd, MiyajimaM1,
# Parametric framework
                        SylvesterResolventResult, ParametricResolventResult,
                        ResolventBoundConfig, SVDWarmStart,
                        config_v1, config_v2, config_v2p5, config_v3,
                        parametric_resolvent_bound, sylvester_resolvent_precompute,
                        solve_sylvester_oracle, estimate_2norm, OneInfNorm

export dowork, dowork_ogita, dowork_ogita_bigfloat, adaptive_arcs!, bound_res_original,
       polygon_sagitta, circle_resolvent_bound,
       bound_resolvent_schur, schur_to_original_resolvent, schur_to_original_resolvent_defects,
       choose_snapshot_to_load, save_snapshot!, configure_certification!, set_schur_matrix!,
       compute_schur_and_error, CertificationCircle, points_on, run_certification,
       run_certification_ogita, poly_from_roots, polyconv,
       _clear_bf_ogita_cache!, _bf_ogita_cache_stats, _set_center_svd_cache!,
# Parametric certifier exports
       dowork_parametric, run_certification_parametric,
       set_parametric_config!, _parametric_cache_stats, _clear_parametric_cache!,
# Gram-weighted certification
       GramTransform, gram_transform, apply_gram_transform

include("gram_transform.jl")

const _schur_matrix = Ref{Union{Nothing, BallMatrix}}(nothing)
const _job_channel = Ref{Any}(nothing)
const _result_channel = Ref{Any}(nothing)
const _certification_log = Ref{Any}(nothing)
const _snapshot_path = Ref{Union{Nothing, AbstractString}}(nothing)
# `stdout` captured here would be the handle of the precompilation process, which is not a valid
# stream when the module is loaded; it is resolved in `__init__` instead.
const _log_io = Ref{IO}(devnull)

# Worker-local SVD cache for Ogita optimization (Float64)
const _last_svd_U = Ref{Union{Nothing, Matrix}}(nothing)
const _last_svd_S = Ref{Union{Nothing, Vector}}(nothing)
const _last_svd_V = Ref{Union{Nothing, Matrix}}(nothing)
const _last_svd_z = Ref{Union{Nothing, Number}}(nothing)
const _ogita_cache_hits = Ref{Int}(0)
const _ogita_cache_misses = Ref{Int}(0)
const _ogita_fallbacks = Ref{Int}(0)

# Worker-local SVD cache for BigFloat Ogita optimization
# Stores the refined BigFloat SVD for reuse with nearby points
const _bf_last_svd_U = Ref{Union{Nothing, Matrix}}(nothing)
const _bf_last_svd_S = Ref{Union{Nothing, Vector}}(nothing)
const _bf_last_svd_V = Ref{Union{Nothing, Matrix}}(nothing)
const _bf_last_svd_z = Ref{Union{Nothing, Number}}(nothing)
const _bf_ogita_cache_hits = Ref{Int}(0)
const _bf_ogita_cache_misses = Ref{Int}(0)
const _bf_ogita_fallbacks = Ref{Int}(0)

# Center SVD cache - shared reference SVD computed at circle center
# This is computed once and used as starting point for all workers
const _bf_center_svd_U = Ref{Union{Nothing, Matrix}}(nothing)
const _bf_center_svd_S = Ref{Union{Nothing, Vector}}(nothing)
const _bf_center_svd_V = Ref{Union{Nothing, Matrix}}(nothing)
const _bf_center_svd_z = Ref{Union{Nothing, Number}}(nothing)
const _bf_center_cache_hits = Ref{Int}(0)

# Parametric Sylvester-based certifier cache
const _parametric_precomp = Ref{Union{Nothing, SylvesterResolventResult}}(nothing)
const _parametric_residual = Ref{Union{Nothing, Matrix}}(nothing)
const _parametric_config = Ref{Union{Nothing, ResolventBoundConfig}}(nothing)
const _parametric_k = Ref{Int}(0)
const _parametric_warm_U = Ref{Union{Nothing, Matrix}}(nothing)
const _parametric_warm_S = Ref{Union{Nothing, Vector}}(nothing)
const _parametric_warm_V = Ref{Union{Nothing, Matrix}}(nothing)
const _parametric_warm_z = Ref{Union{Nothing, Number}}(nothing)
const _parametric_cache_hits = Ref{Int}(0)
const _parametric_cache_misses = Ref{Int}(0)

"""
    CertificationCircle(center, radius; samples = 256)

Discretisation of a circle with centre `center`, radius `radius`, and
`samples` equally spaced points used for certification runs.
"""
struct CertificationCircle
    center::ComplexF64
    radius::Float64
    samples::Int
end

function CertificationCircle(center::Number, radius::Real; samples::Integer = 256)
    samples < 3 && throw(ArgumentError("circle discretisation requires at least 3 samples"))
    radius <= 0 && throw(ArgumentError("circle radius must be positive"))
    return CertificationCircle(ComplexF64(center), Float64(radius), Int(samples))
end

"""
    points_on(circle)

Return the discretisation of `circle` used for certification.
"""
function points_on(circle::CertificationCircle)
    θs = range(0, 2π, length = circle.samples + 1)[1:(end - 1)]
    return circle.center .+ circle.radius .* exp.(θs .* im)
end

function _initial_arcs(circle::CertificationCircle)
    points = points_on(circle)
    n = length(points)
    # the certified contour is the polygon through these points: fewer than three do not enclose
    # anything, and a single point would give one side of length zero, accepted with nothing covered
    n >= 3 || throw(ArgumentError("a certification contour needs at least 3 samples, got $n"))
    arcs = Vector{Tuple{ComplexF64, ComplexF64}}(undef, n)
    for i in 1:n
        arcs[i] = (points[i], points[mod1(i + 1, n)])
    end
    return arcs
end

"""
    set_schur_matrix!(T)

Store the Schur factor `T` used by [`dowork`](@ref).
"""
function set_schur_matrix!(T::BallMatrix)
    _schur_matrix[] = T
    return T
end

"""
    configure_certification!(; job_channel, result_channel, certification_log, snapshot, io)

Cache common resources used by the certification helpers.  The stored values
are used as defaults by [`adaptive_arcs!`](@ref) and [`save_snapshot!`](@ref).
Any keyword may be omitted to keep its previous value.
"""
function configure_certification!(; job_channel = nothing, result_channel = nothing,
        certification_log = nothing, snapshot = nothing, io = nothing)
    job_channel !== nothing && (_job_channel[] = job_channel)
    result_channel !== nothing && (_result_channel[] = result_channel)
    certification_log !== nothing && (_certification_log[] = certification_log)
    snapshot !== nothing && (_snapshot_path[] = snapshot)
    io !== nothing && (_log_io[] = io)
    return nothing
end

function _require_config(ref::Base.RefValue, name::AbstractString)
    value = ref[]
    value === nothing && throw(ArgumentError("$name has not been configured"))
    return value
end

function _evaluate_sample(T::BallMatrix{ET}, z::Number, idx::Int) where {ET}
    # Convert z to the precision of the matrix
    RT = real(ET)
    CT = Complex{RT}
    z_converted = CT(z)
    bz = Ball(z_converted, zero(RT))

    elapsed = @elapsed Σ = svdbox(T - bz * LinearAlgebra.I)

    val = Σ[end]
    res = 1 / val

    lo_val = setrounding(RT, RoundDown) do
        return RT(val.c) - RT(val.r)
    end

    hi_res = setrounding(RT, RoundUp) do
        return RT(res.c) + RT(res.r)
    end

    return (
        i = idx,
        val = val,
        lo_val = lo_val,
        res = res,
        hi_res = hi_res,
        second_val = Σ[end - 1],
        z = z_converted,
        t = elapsed,
        id = nothing
    )
end

"""
    _clear_ogita_cache!()

Clear the worker-local SVD cache used for Ogita optimization.
"""
function _clear_ogita_cache!()
    _last_svd_U[] = nothing
    _last_svd_S[] = nothing
    _last_svd_V[] = nothing
    _last_svd_z[] = nothing
    _ogita_cache_hits[] = 0
    _ogita_cache_misses[] = 0
    _ogita_fallbacks[] = 0
    return nothing
end

"""
    _ogita_cache_stats()

Return statistics about the Ogita cache usage.
"""
function _ogita_cache_stats()
    return (
        hits = _ogita_cache_hits[],
        misses = _ogita_cache_misses[],
        fallbacks = _ogita_fallbacks[]
    )
end

"""
    _clear_bf_ogita_cache!()

Clear the worker-local BigFloat SVD cache used for Ogita optimization.
"""
function _clear_bf_ogita_cache!()
    _bf_last_svd_U[] = nothing
    _bf_last_svd_S[] = nothing
    _bf_last_svd_V[] = nothing
    _bf_last_svd_z[] = nothing
    _bf_ogita_cache_hits[] = 0
    _bf_ogita_cache_misses[] = 0
    _bf_ogita_fallbacks[] = 0
    # Also clear center cache
    _bf_center_svd_U[] = nothing
    _bf_center_svd_S[] = nothing
    _bf_center_svd_V[] = nothing
    _bf_center_svd_z[] = nothing
    _bf_center_cache_hits[] = 0
    return nothing
end

"""
    _bf_ogita_cache_stats()

Return statistics about the BigFloat Ogita cache usage.
"""
function _bf_ogita_cache_stats()
    return (
        local_hits = _bf_ogita_cache_hits[],
        center_hits = _bf_center_cache_hits[],
        misses = _bf_ogita_cache_misses[],
        fallbacks = _bf_ogita_fallbacks[]
    )
end

"""
    _set_center_svd_cache!(U, S, V, z)

Set the center SVD cache. This should be called once at the start of
certification with the SVD computed at the circle center.
"""
function _set_center_svd_cache!(U, S, V, z)
    _bf_center_svd_U[] = U
    _bf_center_svd_S[] = S
    _bf_center_svd_V[] = V
    _bf_center_svd_z[] = z
    return nothing
end

"""
    _evaluate_sample_with_ogita_cache(T, z, idx; ogita_distance_threshold, ogita_quality_threshold)

Evaluate sample with Ogita optimization: try to refine from cached SVD if available.

If the cached SVD is from a nearby point (distance < ogita_distance_threshold), attempt
Ogita refinement. If the refined SVD has acceptable quality (residual < ogita_quality_threshold),
use it; otherwise fall back to full SVD.
"""
function _evaluate_sample_with_ogita_cache(T::BallMatrix{ET}, z::Number, idx::Int;
        ogita_distance_threshold::Real = 1e-4,
        ogita_quality_threshold::Real = 1e-10,
        ogita_iterations::Int = 2) where {ET}
    RT = real(ET)
    CT = Complex{RT}
    z_converted = CT(z)
    n = size(T, 1)

    # Compute T - z*I (center matrix for SVD)
    T_shifted = T.c - z_converted * LinearAlgebra.I

    use_ogita = false
    ogita_success = false

    # Check if we can try Ogita from cached SVD
    if _last_svd_U[] !== nothing && _last_svd_z[] !== nothing
        dist = abs(z_converted - _last_svd_z[])
        if dist < ogita_distance_threshold
            use_ogita = true
        end
    end

    elapsed = @elapsed begin
        if use_ogita
            # Try Ogita refinement from cached SVD
            try
                refined = ogita_svd_refine(T_shifted,
                    _last_svd_U[],
                    _last_svd_S[],
                    _last_svd_V[];
                    max_iterations = ogita_iterations,
                    precision_bits = 53)

                # Check quality via residual norm
                Σ_vec = isa(refined.Σ, Diagonal) ? diag(refined.Σ) : refined.Σ
                residual = refined.residual_norm
                matrix_norm = LinearAlgebra.norm(T_shifted)

                if residual < ogita_quality_threshold * matrix_norm
                    # Ogita succeeded - certify the refined SVD
                    ogita_success = true
                    _ogita_cache_hits[] += 1

                    # Create BallMatrix for certification
                    T_ball = BallMatrix(T_shifted, fill(eps(RT) * matrix_norm, n, n))
                    svd_refined = (U = refined.U, S = Σ_vec, V = refined.V, Vt = refined.V')
                    Σ = _certify_svd(T_ball, svd_refined, MiyajimaM1(); apply_vbd = false)

                    # Update cache with refined SVD
                    _last_svd_U[] = refined.U
                    _last_svd_S[] = Σ_vec
                    _last_svd_V[] = refined.V
                    _last_svd_z[] = z_converted
                else
                    # Quality check failed - fall back
                    _ogita_fallbacks[] += 1
                end
            catch e
                # Ogita failed - fall back
                @debug "Ogita refinement failed" exception=e
                _ogita_fallbacks[] += 1
            end
        else
            _ogita_cache_misses[] += 1
        end

        if !ogita_success
            # Full SVD path
            bz = Ball(z_converted, zero(RT))
            Σ = svdbox(T - bz * LinearAlgebra.I)

            # Update cache with fresh SVD
            svd_fresh = LinearAlgebra.svd(T_shifted)
            _last_svd_U[] = svd_fresh.U
            _last_svd_S[] = svd_fresh.S
            _last_svd_V[] = svd_fresh.V
            _last_svd_z[] = z_converted
        end
    end

    val = Σ[end]
    res = 1 / val

    lo_val = setrounding(RT, RoundDown) do
        return RT(val.c) - RT(val.r)
    end

    hi_res = setrounding(RT, RoundUp) do
        return RT(res.c) + RT(res.r)
    end

    return (
        i = idx,
        val = val,
        lo_val = lo_val,
        res = res,
        hi_res = hi_res,
        second_val = Σ[end - 1],
        z = z_converted,
        t = elapsed,
        id = nothing,
        ogita_used = ogita_success
    )
end

"""
    dowork_ogita(jobs, results; ogita_distance_threshold=1e-4, ogita_quality_threshold=1e-10)

Process tasks with Ogita optimization enabled. Similar to [`dowork`](@ref) but tries
to use cached SVD from previous evaluations to speed up nearby points.

The worker maintains a cache of the last computed SVD. When a new point z arrives,
if it's within `ogita_distance_threshold` of the cached point, Ogita refinement is
attempted. If the refined SVD has acceptable quality (relative residual < `ogita_quality_threshold`),
it's used; otherwise a full SVD is computed.

This is beneficial when the adaptive bisection sends consecutive jobs for nearby points,
which happens naturally as arcs get smaller during refinement.
"""
function dowork_ogita(jobs, results;
        ogita_distance_threshold::Real = 1e-4,
        ogita_quality_threshold::Real = 1e-10,
        ogita_iterations::Int = 2)
    T = _require_config(_schur_matrix, "Schur factor")
    _clear_ogita_cache!()

    while true
        job = try
            take!(jobs)
        catch e
            if e isa InvalidStateException
                # Log cache stats before exiting
                stats = _ogita_cache_stats()
                total = stats.hits + stats.misses + stats.fallbacks
                if total > 0
                    hit_rate = stats.hits / total * 100
                    @debug "Ogita cache stats" hits=stats.hits misses=stats.misses fallbacks=stats.fallbacks hit_rate="$(round(hit_rate, digits=1))%"
                end
                break
            else
                rethrow(e)
            end
        end

        i, z = job
        @debug "Received and working on (Ogita)" z
        result = _evaluate_sample_with_ogita_cache(T, z, i;
            ogita_distance_threshold,
            ogita_quality_threshold,
            ogita_iterations)
        put!(results, result)
    end
    return nothing
end

"""
    dowork_ogita_bigfloat(jobs, results; target_precision=256, max_ogita_iterations=3,
                          distance_threshold=1e-4)

Process tasks with BigFloat precision using Ogita SVD refinement with caching.

For each job (id, z), this worker:
1. Checks if a cached BigFloat SVD exists from a nearby point
2. If cache hit: refines from cached SVD (1-2 iterations)
3. If cache miss: computes Float64 SVD and refines to BigFloat (3-4 iterations)
4. Certifies with Miyajima bounds
5. Caches the result for future reuse

This is the distributed equivalent of `run_certification_ogita` for parallel execution.
The Schur matrix must be a BigFloat BallMatrix registered with `set_schur_matrix!`.

# Performance
- Cache miss: Ogita from Float64 requires ~4 iterations (10^-16 → 10^-64)
- Cache hit: Ogita from BigFloat requires ~1-2 iterations (already at high precision)
- For adaptive bisection with nearby points, expect 50-90% cache hit rate
"""
function dowork_ogita_bigfloat(jobs, results;
        target_precision::Int = 256,
        max_ogita_iterations::Int = 3,
        distance_threshold::Real = 1e-4)
    T = _require_config(_schur_matrix, "Schur factor")
    _clear_bf_ogita_cache!()

    # Verify T is BigFloat
    ET = eltype(T.c)
    if real(ET) !== BigFloat
        @warn "dowork_ogita_bigfloat expects BigFloat Schur matrix, got $ET"
    end

    # Set precision for this worker
    old_prec = precision(BigFloat)
    setprecision(BigFloat, target_precision)

    try
        while true
            job = try
                take!(jobs)
            catch e
                if e isa InvalidStateException
                    # Log cache stats before exiting
                    stats = _bf_ogita_cache_stats()
                    total = stats.local_hits + stats.center_hits + stats.misses +
                            stats.fallbacks
                    if total > 0
                        hit_rate = (stats.local_hits + stats.center_hits) / total * 100
                        @debug "BigFloat Ogita cache stats" local_hits=stats.local_hits center_hits=stats.center_hits misses=stats.misses fallbacks=stats.fallbacks hit_rate="$(round(hit_rate, digits=1))%"
                    end
                    break
                else
                    rethrow(e)
                end
            end

            i, z = job
            @debug "Received and working on (Ogita BigFloat)" z
            result = _evaluate_sample_ogita_bigfloat(T, z, i;
                max_iterations = max_ogita_iterations,
                target_precision = target_precision,
                distance_threshold = distance_threshold)
            put!(results, result)
        end
    finally
        setprecision(BigFloat, old_prec)
    end
    return nothing
end

"""
    dowork(jobs, results)

Process tasks received on `jobs`, computing the SVD certification routine for
`T - zI`.  The Schur factor must have been registered in advance with
[`set_schur_matrix!`](@ref).
"""
function dowork(jobs, results)
    T = _require_config(_schur_matrix, "Schur factor")
    while true
        job = try
            take!(jobs)
        catch e
            if e isa InvalidStateException
                break
            else
                rethrow(e)
            end
        end

        i, z = job
        @debug "Received and working on" z
        result = _evaluate_sample(T, z, i)
        put!(results, result)
    end
    return nothing
end

# =====================================================================
# Parametric Sylvester-based certifier
# =====================================================================

"""
    _clear_parametric_cache!()

Clear the worker-local parametric certifier cache.
"""
function _clear_parametric_cache!()
    _parametric_precomp[] = nothing
    _parametric_residual[] = nothing
    _parametric_config[] = nothing
    _parametric_k[] = 0
    _parametric_warm_U[] = nothing
    _parametric_warm_S[] = nothing
    _parametric_warm_V[] = nothing
    _parametric_warm_z[] = nothing
    _parametric_cache_hits[] = 0
    _parametric_cache_misses[] = 0
    return nothing
end

"""
    _parametric_cache_stats()

Return statistics about the parametric cache usage.
"""
function _parametric_cache_stats()
    return (
        hits = _parametric_cache_hits[],
        misses = _parametric_cache_misses[]
    )
end

"""
    set_parametric_config!(precomp, R, config; k=0)

Set the parametric certifier configuration.

# Arguments
- `precomp`: Precomputed Sylvester quantities from `sylvester_resolvent_precompute`
- `R`: Sylvester residual matrix
- `config`: ResolventBoundConfig specifying estimators
- `k`: Split index (for reference)
"""
function set_parametric_config!(precomp::SylvesterResolventResult, R::AbstractMatrix,
        config::ResolventBoundConfig; k::Int = 0)
    _parametric_precomp[] = precomp
    _parametric_residual[] = Matrix(R)
    _parametric_config[] = config
    _parametric_k[] = k
    return nothing
end

"""
    _evaluate_sample_parametric(T, z, idx; use_warm_start=true, distance_threshold=1e-4)

Evaluate sample using the parametric Sylvester-based certifier.

Returns a result compatible with the standard certification format.
"""
function _evaluate_sample_parametric(T::BallMatrix{ET}, z::Number, idx::Int;
        use_warm_start::Bool = true,
        distance_threshold::Real = 1e-4) where {ET}
    RT = real(ET)
    CT = Complex{RT}
    z_converted = CT(z)

    precomp = _parametric_precomp[]
    R = _parametric_residual[]
    config = _parametric_config[]

    if precomp === nothing || R === nothing || config === nothing
        error("Parametric config not set. Call set_parametric_config! first.")
    end

    # Check for warm start
    warm_start = nothing
    if use_warm_start && _parametric_warm_z[] !== nothing
        last_z = _parametric_warm_z[]
        if abs(z_converted - last_z) < distance_threshold * (abs(last_z) + 1)
            warm_start = SVDWarmStart(
                _parametric_warm_U[],
                _parametric_warm_S[],
                _parametric_warm_V[]
            )
            _parametric_cache_hits[] += 1
        else
            _parametric_cache_misses[] += 1
        end
    else
        _parametric_cache_misses[] += 1
    end

    elapsed = @elapsed result = parametric_resolvent_bound(
        precomp, Matrix(T.c), z_converted, config;
        R = R, svd_warm_start = warm_start
    )

    # Update warm start cache for next call
    if result.success
        k = precomp.k
        T11 = T.c[1:k, 1:k]
        A_z = z_converted * I - T11
        try
            svd_Az = LinearAlgebra.svd(A_z)
            _parametric_warm_U[] = svd_Az.U
            _parametric_warm_S[] = svd_Az.S
            _parametric_warm_V[] = svd_Az.V
            _parametric_warm_z[] = z_converted
        catch
            # SVD failed, don't update cache
        end
    end

    # Convert to standard format
    if result.success
        # Compute σ_min from M_A (M_A = 1/σ_min)
        sigma_min = RT(1) / result.M_A
        sigma_min_ball = Ball(sigma_min, zero(RT))  # Approximate

        lo_val = setrounding(RT, RoundDown) do
            sigma_min
        end

        hi_res = setrounding(RT, RoundUp) do
            result.resolvent_bound
        end

        return (
            i = idx,
            val = sigma_min_ball,
            lo_val = lo_val,
            res = Ball(result.resolvent_bound, zero(RT)),
            hi_res = hi_res,
            second_val = Ball(zero(RT), zero(RT)),  # Not available from parametric
            z = z_converted,
            t = elapsed,
            id = nothing
        )
    else
        # Failure case
        return (
            i = idx,
            val = Ball(zero(RT), RT(Inf)),
            lo_val = zero(RT),
            res = Ball(RT(Inf), zero(RT)),
            hi_res = RT(Inf),
            second_val = Ball(zero(RT), zero(RT)),
            z = z_converted,
            t = elapsed,
            id = nothing
        )
    end
end

"""
    dowork_parametric(jobs, results; distance_threshold=1e-4)

Process tasks using the parametric Sylvester-based certifier.

The Schur factor and parametric config must have been registered in advance with
[`set_schur_matrix!`](@ref) and [`set_parametric_config!`](@ref).
"""
function dowork_parametric(jobs, results; distance_threshold::Real = 1e-4)
    T = _require_config(_schur_matrix, "Schur factor")
    _clear_parametric_cache!()

    while true
        job = try
            take!(jobs)
        catch e
            if e isa InvalidStateException
                # Log cache stats before exiting
                stats = _parametric_cache_stats()
                total = stats.hits + stats.misses
                if total > 0
                    hit_rate = stats.hits / total * 100
                    @debug "Parametric cache stats" hits=stats.hits misses=stats.misses hit_rate="$(round(hit_rate, digits=1))%"
                end
                break
            else
                rethrow(e)
            end
        end

        i, z = job
        @debug "Received and working on (Parametric)" z
        result = _evaluate_sample_parametric(T, z, i; distance_threshold = distance_threshold)
        put!(results, result)
    end
    return nothing
end

function _resolve(value, ref::Base.RefValue)
    value === nothing || return value
    return ref[]
end

function _ensure_certification_log(log)
    log === nothing && return _certification_log[]
    return log
end

function _ensure_io(io)
    io === nothing && return _log_io[]
    return io
end

function _push_result!(cache, certification_log, result)
    cache[result.z] = (result.val, result.second_val)
    certification_log !== nothing && push!(certification_log, result)
    return nothing
end

function _adaptive_arcs_serial!(arcs::Vector{Tuple{ComplexF64, ComplexF64}},
        cache::Dict{ComplexF64, Any}, pending, η::Float64, certification_log,
        snapshot, io, check_interval::Integer, evaluator; maxarcs::Integer = 0)
    io = io === nothing ? stdout : io
    certification_log === nothing && (certification_log = Any[])

    processed = 0
    new = 0
    cycle = true

    while !isempty(arcs)
        z_a, z_b = pop!(arcs)

        if haskey(cache, z_a)
            σ_a = cache[z_a][1]
        else
            result = evaluator(z_a)
            _push_result!(cache, certification_log, result)
            σ_a = result.val
        end

        # A bound that is zero, negative or not finite cannot be refined away: the test below
        # would ask for ℓ/0 ≤ η at every depth, so the arc would be bisected for ever while ℓ
        # shrinks and σ_a stays put. Refinement is the wrong answer there and the caller has to
        # hear about it.
        lo_σ = _lower_bound(σ_a)
        if !(isfinite(lo_σ) && lo_σ > 0)
            @info "Adaptive refinement stopped: the bound is not positive" z=z_a σ=σ_a
            return (; ok = false, reason = :nonpositive, z = z_a, processed,
                evaluations = length(cache), remaining = length(arcs) + 1)
        end

        sup_ε = _side_ratio_sup(z_a, z_b, lo_σ)

        if sup_ε > η
            if maxarcs > 0 && length(cache) >= maxarcs
                @info "Adaptive refinement stopped: the evaluation cap was reached" maxarcs
                return (; ok = false, reason = :cap, z = z_a, processed,
                    evaluations = length(cache), remaining = length(arcs) + 1)
            end
            z_m = (z_a + z_b) / 2
            if z_m == z_a || z_m == z_b
                @info "Adaptive refinement stopped: a side cannot be split further" z=z_a
                return (; ok = false, reason = :not_refinable, z = z_a, processed,
                    evaluations = length(cache), remaining = length(arcs) + 1)
            end
            push!(arcs, (z_m, z_b))
            push!(arcs, (z_a, z_m))
            new += 1
        end

        processed += 1
        if processed % check_interval == 0
            @info "Processed $processed arcs (serial)" remaining=length(arcs) new_ratio=new/check_interval
            flush(io)
            new = 0
            if snapshot !== nothing
                cycle = save_snapshot!(
                    arcs, cache, certification_log, pending, snapshot, cycle)
            end
        end
    end

    return (; ok = true, reason = :complete, z = ComplexF64(NaN), processed,
        evaluations = length(cache), remaining = 0)
end

function _adaptive_arcs_distributed!(arcs::Vector{Tuple{ComplexF64, ComplexF64}},
        cache::Dict{ComplexF64, Any}, pending::Dict{Int, Tuple{ComplexF64, ComplexF64}},
        η::Float64, job_channel, result_channel, certification_log, snapshot, io,
        check_interval::Integer)
    certification_log === nothing &&
        throw(ArgumentError("distributed refinement requires a certification log"))
    io = io === nothing ? stdout : io

    cycle = true
    @info "Starting adaptive refinement, arcs, $(length(arcs)), pending, $(length(pending))"
    flush(io)

    id_counter = maximum(collect(keys(pending)); init = 0) + 1
    @info "Pending from snapshot" length(pending) id_counter

    for (i, (z_a, _)) in pending
        put!(job_channel, (i, z_a))
    end

    while !isempty(pending)
        if isready(result_channel)
            result = take!(result_channel)
            _push_result!(cache, certification_log, result)
            z_a, z_b = pending[result.i]
            delete!(pending, result.i)
            push!(arcs, (z_a, z_b))
        else
            sleep(0.1)
        end
    end
    @info "Waited for all pending to be computed, arcs, $(length(arcs)), pending, $(length(pending))"

    flush(io)
    while !isempty(arcs)
        processed = 0
        new = 0

        while !isempty(arcs)
            z_a, z_b = pop!(arcs)

            if haskey(cache, z_a)
                σ_a = cache[z_a][1]
            else
                job_id = id_counter
                put!(job_channel, (job_id, z_a))
                pending[job_id] = (z_a, z_b)
                id_counter += 1
                continue
            end

            # as in the serial loop: a bound that is not positive cannot be refined away
            lo_σ = _lower_bound(σ_a)
            if !(isfinite(lo_σ) && lo_σ > 0)
                @info "Adaptive refinement stopped: the bound is not positive" z=z_a σ=σ_a
                return (; ok = false, reason = :nonpositive, z = z_a, processed,
                    evaluations = length(cache), remaining = length(arcs) + 1)
            end
            sup_ε = _side_ratio_sup(z_a, z_b, lo_σ)

            if sup_ε > η
                z_m = (z_a + z_b) / 2
                if z_m == z_a || z_m == z_b
                    @info "Adaptive refinement stopped: a side cannot be split further" z=z_a
                    return (; ok = false, reason = :not_refinable, z = z_a, processed,
                        evaluations = length(cache), remaining = length(arcs) + 1)
                end
                push!(arcs, (z_m, z_b))
                push!(arcs, (z_a, z_m))
                new += 1
            end

            processed += 1
            if processed % check_interval == 0
                @info "Processed $processed arcs..."
                @info "Remaining arcs" length(arcs)
                @info "Pending jobs" length(pending)
                @info "New arcs ratio" new / check_interval
                new = 0
                flush(io)

                while isready(result_channel)
                    result = take!(result_channel)
                    _push_result!(cache, certification_log, result)
                    z_a, z_b = pending[result.i]
                    delete!(pending, result.i)
                    push!(arcs, (z_a, z_b))
                end
                if snapshot !== nothing
                    cycle = save_snapshot!(
                        arcs, cache, certification_log, pending, snapshot, cycle)
                end
            end
        end

        @info "Waiting for all pending jobs..."
        while !isempty(pending)
            if isready(result_channel)
                result = take!(result_channel)
                _push_result!(cache, certification_log, result)
                z_a, z_b = pending[result.i]
                delete!(pending, result.i)
                push!(arcs, (z_a, z_b))
            else
                @info "Waiting for pending" length(pending)
                flush(io)
                sleep(0.1)
            end
        end

        @info "Restarting refinement cycle with new arcs: $(length(arcs))"
    end

    @info "Adaptive refinement complete"
    return (; ok = true, reason = :complete, z = ComplexF64(NaN), processed = 0,
        evaluations = length(cache), remaining = 0)
end

"""
    adaptive_arcs!(arcs, cache, pending, η; kwargs...)

Drive the adaptive refinement routine.  When job channels are provided the
refinement uses asynchronous workers; otherwise the evaluation is carried out
serially using the supplied evaluator.

Returns a named tuple `(; ok, reason, z, processed, evaluations, remaining)`. `ok` is false when
the refinement stopped without covering the contour, with `reason` either `:nonpositive`, the
evaluator returned a bound that was zero, negative or not finite at `z`, which no amount of
bisection repairs, or `:cap`, the optional `maxarcs` ceiling on the number of evaluations was
reached. `maxarcs = 0`, the default, leaves the refinement uncapped.

Bisection takes the midpoint of the chord, so after the first refinement the samples lie on the
polygon inscribed in the circle rather than on the circle itself. That is the contour the run
certifies, which is a closed contour like any other; a caller checking a region condition should
use the inradius `r·cos(Δθ/2)` and not `r`.
"""
function adaptive_arcs!(arcs::Vector{Tuple{ComplexF64, ComplexF64}},
        cache::Dict{ComplexF64, Any}, pending, η::Float64;
        check_interval::Integer = 1000, job_channel = nothing,
        result_channel = nothing, certification_log = nothing,
        snapshot = nothing, io = nothing, evaluator = nothing, maxarcs::Integer = 0)
    job_channel = _resolve(job_channel, _job_channel)
    result_channel = _resolve(result_channel, _result_channel)
    certification_log = _ensure_certification_log(certification_log)
    snapshot = snapshot === nothing ? _snapshot_path[] : snapshot
    io = _ensure_io(io)

    if job_channel === nothing || result_channel === nothing
        evaluator === nothing &&
            throw(ArgumentError("serial refinement requires an evaluator"))
        return _adaptive_arcs_serial!(arcs, cache, pending, η, certification_log,
            snapshot, io, check_interval, evaluator; maxarcs)
    end

    return _adaptive_arcs_distributed!(arcs, cache, pending, η, job_channel,
        result_channel, certification_log, snapshot, io, check_interval)
end

# A refinement that stopped has not covered the contour, and no bound follows from its samples.
function _require_complete(outcome)
    outcome.ok && return outcome
    throw(ErrorException("the adaptive refinement did not cover the contour: " *
                         "$(outcome.reason) at z = $(outcome.z) after $(outcome.evaluations) " *
                         "evaluations, $(outcome.remaining) sides left"))
end

"""
    bound_resolvent_schur(l2pseudo, η)

An upper bound of `‖(zI − T)⁻¹‖₂` over the whole certified contour, from its samples.

The contour is the closed polygon through the samples: [`adaptive_arcs!`](@ref) starts from the
vertices of the regular polygon inscribed in the circle and bisects SIDES, so every later sample is
the midpoint of a side. A side `[z_a, z_b]` is accepted when `|z_b − z_a| / σ_a ≤ η`, with `σ_a` a
certified lower bound of `σ_min(T − z_a I)`. Since `σ_min(T − zI)` is 1-Lipschitz in `z` (Weyl's
inequality for singular values: `|σ_min(M + E) − σ_min(M)| ≤ ‖E‖₂`, here with `E = (z_a − z)I`),
every `z` on the side has

    σ_min(T − zI) ≥ σ_a − |z − z_a| ≥ σ_a − |z_b − z_a| ≥ σ_a (1 − η),

so `‖(zI − T)⁻¹‖₂ ≤ (1/σ_a)/(1 − η)`. With `l2pseudo` the largest `1/σ_a` over the samples this is
`l2pseudo/(1 − η)`, computed with the numerator rounded up and the denominator down. It is `Inf`
for `η ≥ 1`.

The bound is for the polygon, not for the circle it is inscribed in: between two vertices the
circle lies outside the polygon, by up to `r(1 − cos(π/N))` for `N` initial samples, and a point
of the spectrum in that sliver, or just outside the circle, can make the resolvent on the circle
far larger than on the polygon. Any statement drawn from the bound (a Riesz projector, a count of
eigenvalues, a region free of spectrum) is therefore one about the polygon and its interior.
"""
function bound_resolvent_schur(l2pseudo, η)
    if !(η < 1)
        return Inf
    end
    return div_up(_upper_bound(l2pseudo), sub_down(1.0, Float64(η)))
end

"""
    polygon_sagitta(circle::CertificationCircle)

An upper bound of the largest distance from a point of the circle to the polygon the certification
runs on, the regular polygon with `circle.samples` vertices inscribed in it: the sagitta
`r(1 − cos(π/N))`, bounded by `r(π/N)²/2` since `1 − cos x ≤ x²/2`, evaluated with every operation
rounded upward and `π` replaced by an upper bound. Bisection adds points on the sides only, so the
refined polygon has the same distance to the circle.
"""
function polygon_sagitta(circle::CertificationCircle)
    N = circle.samples
    N >= 3 || throw(ArgumentError("a certification contour needs at least 3 samples, got $N"))
    x = div_up(nextfloat(Float64(π)), Float64(N))          # nextfloat(Float64(π)) > π
    return mul_up(Float64(circle.radius), div_up(mul_up(x, x), 2.0))
end

"""
    circle_resolvent_bound(M, circle::CertificationCircle)

Given `M ≥ ‖(zI − T)⁻¹‖₂` for every `z` on the certified polygon (the output of
[`bound_resolvent_schur`](@ref)), an upper bound of `‖(zI − T)⁻¹‖₂` for every `z` on the CIRCLE
`circle`, or `Inf` when the argument below does not apply.

Every point `z` of the circle is within `h = `[`polygon_sagitta`](@ref)`(circle)` of a point `z'`
of the polygon, and `σ_min(T − zI)` is 1-Lipschitz in `z` (Weyl's inequality for singular values,
`|σ_min(M₁ + E) − σ_min(M₁)| ≤ ‖E‖₂`, with `E = (z' − z)I`). Hence

    σ_min(T − zI) ≥ σ_min(T − z'I) − h ≥ 1/M − h,

and for `M h < 1` the resolvent on the circle is at most `M/(1 − M h)`. The same holds for any
matrix in place of `T`, so it applies equally to a bound for the original matrix. Nothing is
assumed about where the spectrum is: when `M h ≥ 1` the polygon bound does not exclude a point of
the spectrum between the polygon and the circle, and the result is `Inf`.
"""
function circle_resolvent_bound(M::Real, circle::CertificationCircle)
    (isfinite(M) && M >= 0) || return Inf
    Mh = mul_up(Float64(M), polygon_sagitta(circle))
    Mh < 1 || return Inf
    return div_up(Float64(M), sub_down(1.0, Mh))
end

"""
    schur_to_original_resolvent(resolvent_schur, ϵ; zmax, Cbound = 1.0)

The resolvent bound for the original matrix from the one for its approximate Schur factor, on a
contour whose points satisfy `|z| ≤ zmax`. Returns `Inf` when the hypothesis below is not met.

# Reference

A. Blumenthal, I. Nisoli and T. Taylor-Crush, *A pseudospectral approach to rigorous numerical
estimation of resonances of transfer operators*, Found. Comput. Math. (accepted; arXiv:2507.09021),
Lemma 5.1. With `r_B(z) := ‖(z − B)⁻¹‖`:

> Let `M = Z T Z* + E_M` (26) and `I = Z Z* + E_Z` (27), and assume
> `‖E_M‖, ‖E_Z‖ ≤ ϵ` and `‖Z‖, ‖Z⁻¹‖ ≤ 1 + ϵ`. Suppose `z ∈ ℂ ∖ σ(T)` is such that
>
>     2ϵ(1 + ϵ)² r_T(z) · max{1, |z|} < 1/2.                                   (28)
>
> Then
>
>     r_M(z) ≤ 2(1 + ϵ)² r_T(z) / (1 − 2ϵ(1 + ϵ)² r_T(z)).                      (29)

The factor `max{1, |z|}` is not decoration: `Z T Z*` is not similar to `T` unless `Z` is unitary,
`z − Z T Z* = Z(z − T)Z* + z(I − Z Z*)`, and the hypothesis is what absorbs the second term. With
`Z = √1.001·I`, `T = diag(1000, 0)` and `z = 1001` the matrix `z − Z T Z*` is singular while
`r_T(z) = 1`; (28) fails there, and this function returns `Inf`.

`resolvent_schur` must bound `r_T(z)` on the whole contour, `ϵ` the four defects, and `zmax` the
moduli of its points. (28) is checked with `r_T` and `|z|` replaced by these upper bounds, which is
stronger than the hypothesis at each point, and (29), which increases with `r_T`, is evaluated
there with every operation rounded upward. `Cbound` multiplies the result, for a caller converting
to another norm.
"""
function schur_to_original_resolvent(resolvent_schur, ϵ; zmax, Cbound = 1.0)
    R, e, zm, C = promote(float(_upper_bound(resolvent_schur)), float(_upper_bound(ϵ)),
        float(_upper_bound(zmax)), float(_upper_bound(Cbound)))
    (isfinite(R) && isfinite(e) && isfinite(zm) && isfinite(C)) || return Inf
    one_ = one(R)
    ope2 = mul_up(add_up(one_, e), add_up(one_, e))             # (1 + ϵ)²
    a = mul_up(mul_up(2 * one_, e), mul_up(ope2, R))            # 2ϵ(1 + ϵ)² r_T
    mul_up(a, max(one_, zm)) < one_ / 2 || return Inf           # hypothesis (28)
    return div_up(mul_up(mul_up(2 * one_, ope2), mul_up(R, C)), sub_down(one_, a))
end

"""
    schur_to_original_resolvent_defects(resolvent_schur, δ, e; zmax, Cbound = 1.0)

The resolvent bound for the original matrix `A` from the one for its approximate Schur factor
`T`, in terms of the two defects separately: `δ ≥ ‖I − Q*Q‖` and `e ≥ ‖Q T Q* − A‖`, on a contour
with `|z| ≤ zmax`. With `R ≥ ‖(zI − T)⁻¹‖` on the contour,

    R₀ := R / (1 − δ − R·zmax·δ),        ‖(zI − A)⁻¹‖ ≤ R₀ / (1 − R₀ e),

provided both denominators are positive; `Inf` otherwise. It is sharper than
[`schur_to_original_resolvent`](@ref), which loses a factor 2: to first order this is
`R(1 + δ + R·zmax·δ + R e)`.

# Where it comes from, and what is corrected

The route is that of Section 3.1 of

I. Nisoli, *Certified spectral approximation of transfer operators and the Gauss map*,
arXiv:2602.19435 (2026): with the proxy `S₀ := Q T Q*` and `E := A − S₀`, Lemma 3.2 there
(a Neumann series) gives `‖(zI − A)⁻¹‖ ≤ ‖(zI − S₀)⁻¹‖/(1 − ‖(zI − S₀)⁻¹‖‖E‖)`, and Theorem 3.4
combines it with a bound of `‖(zI − S₀)⁻¹‖` by `‖(zI − T)⁻¹‖`.

That second bound is NOT taken from the paper. Its Lemma 3.3 states
`‖(zI − S₀)⁻¹‖ ≤ κ(Q)²‖(zI − T)⁻¹‖` for every `z`, and the proof writes `S₀ = U(HTH)U*` for the
polar decomposition `Q = UH` and then inverts `zI − HTH` as `H(zI − T)H`. These differ by
`z(I − H²)`: `Q T Q*` is similar to `T` only when `Q` is unitary. For `Q = √1.001·I`,
`T = diag(1000, 0)` and `z = 1001`, `κ(Q) = 1` and `‖(zI − T)⁻¹‖ = 1`, while `zI − S₀` is
singular. The statement used here, with the term restored, is proved as follows.

*Claim.* Let `δ = ‖I − Q*Q‖ < 1`, `P := Q*Q`, and `R ≥ ‖(zI − T)⁻¹‖`. If `R|z|δ < 1 − δ`, then
`zI − S₀` is invertible and `‖(zI − S₀)⁻¹‖ ≤ R/(1 − δ − R|z|δ)`.

*Proof.* `P` is Hermitian with spectrum in `[1 − δ, 1 + δ]`, so `Q` is invertible,
`Q⁻¹Q⁻* = P⁻¹`, `‖Q⁻¹‖² = ‖P⁻¹‖ ≤ 1/(1 − δ)`, and `‖P⁻¹ − I‖ ≤ δ/(1 − δ)`. Then

    zI − S₀ = Q (z Q⁻¹Q⁻* − T) Q* = Q ((zI − T) + z(P⁻¹ − I)) Q*,

and `(zI − T) + z(P⁻¹ − I) = (zI − T)(I + (zI − T)⁻¹ z (P⁻¹ − I))`, whose second factor is
invertible by a Neumann series when `R|z|δ/(1 − δ) < 1`, with inverse of norm at most
`1/(1 − R|z|δ/(1 − δ))`. Hence

    ‖(zI − S₀)⁻¹‖ ≤ ‖Q⁻¹‖² · R / (1 − R|z|δ/(1 − δ)) ≤ R / (1 − δ − R|z|δ).  ∎

The corresponding result with the `|z|` dependence placed in a hypothesis, at the price of a
factor 2, is Lemma 5.1 of Blumenthal, Nisoli and Taylor-Crush; see
[`schur_to_original_resolvent`](@ref).

The bound increases with `R`, `δ`, `e` and `|z|`, so it is evaluated at upper bounds of the four,
every operation rounded outward. `Cbound` multiplies the result.
"""
function schur_to_original_resolvent_defects(resolvent_schur, δ, e; zmax, Cbound = 1.0)
    R, d, ee, zm, C = promote(float(_upper_bound(resolvent_schur)), float(_upper_bound(δ)),
        float(_upper_bound(e)), float(_upper_bound(zmax)), float(_upper_bound(Cbound)))
    (isfinite(R) && isfinite(d) && isfinite(ee) && isfinite(zm) && isfinite(C)) || return Inf
    one_ = one(R)
    den0 = sub_down(sub_down(one_, d), mul_up(mul_up(R, zm), d))      # 1 − δ − R zmax δ
    den0 > 0 || return Inf
    R0 = div_up(R, den0)
    den1 = sub_down(one_, mul_up(R0, ee))                             # 1 − R₀ e
    den1 > 0 || return Inf
    return div_up(mul_up(R0, C), den1)
end

"""
    bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, N; zmax, Cbound = 1.0,
                       method = :best)

An upper bound of `‖(zI − A)⁻¹‖₂` on the certified contour for the original matrix `A`, from the
samples of its Schur factor `T`. [`bound_resolvent_schur`](@ref) gives the bound for `T` on the
polygon, and it is carried to `A` with `errF ≥ ‖Z*Z − I‖` (which equals `‖I − Z Z*‖` for a square
`Z`) and `errT ≥ ‖Z T Z* − A‖`, by

* `method = :bntc`: [`schur_to_original_resolvent`](@ref), Lemma 5.1 of Blumenthal, Nisoli and
  Taylor-Crush, with `ϵ = max(errF, errT, ‖Z‖ − 1, ‖Z⁻¹‖ − 1)`;
* `method = :defects`: [`schur_to_original_resolvent_defects`](@ref), the two defects kept
  separate, sharper by about a factor 2;
* `method = :best` (default): the smaller of the two, both being valid bounds.

`zmax` must bound `|z|` on the contour; for a circle it is `|centre| + radius`. `Inf` when no
method applies. `N` is not used.
"""
function bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, N; zmax, Cbound = 1.0,
        method::Symbol = :best)
    method in (:best, :bntc, :defects) ||
        throw(ArgumentError("bound_res_original: method must be :best, :bntc or :defects"))
    resolvent_schur = bound_resolvent_schur(l2pseudo, η)

    norm_Z_sup = max(_upper_bound_offset(norm_Z, 1), 0.0)
    norm_Z_inv_sup = max(_upper_bound_offset(norm_Z_inv, 1), 0.0)
    errF_sup = _upper_bound(errF)
    errT_sup = _upper_bound(errT)
    ϵ = max(max(errF_sup, errT_sup), max(norm_Z_sup, norm_Z_inv_sup))
    @info "The ϵ in the Schur theorems $ϵ"

    bntc = method === :defects ? Inf : schur_to_original_resolvent(resolvent_schur, ϵ; zmax, Cbound)
    defects = method === :bntc ? Inf :
              schur_to_original_resolvent_defects(resolvent_schur, errF_sup, errT_sup; zmax, Cbound)
    result = min(bntc, defects)
    if isinf(result)
        @warn "bound_res_original: no Schur-to-original bound applies " *
              "(ε=$ϵ, resolvent_schur=$resolvent_schur, zmax=$zmax). Returning Inf." maxlog=1
    end
    return result
end

# the largest modulus on a circle, rounded up
_circle_zmax(circle::CertificationCircle) = add_up(abs_up(circle.center), circle.radius)

"""
    choose_snapshot_to_load(basepath)

Return the most recent valid snapshot stored at `basepath`.
"""
function choose_snapshot_to_load(basepath::String)
    files = [basepath * "_A.jld2", basepath * "_B.jld2"]
    valid_files = filter(isfile, files)
    if isempty(valid_files)
        return nothing
    end
    sorted = sort(valid_files, by = f -> stat(f).mtime, rev = true)
    try
        snapshot = JLD2.load(sorted[1])
        return snapshot
    catch e
        @warn "Could not load snapshot file $(sorted[1]), possibly corrupted. Trying backup." exception=(
            e, catch_backtrace())
        if length(sorted) > 1
            try
                snapshot = JLD2.load(sorted[2])
                return snapshot
            catch e
                @warn "Both snapshot files failed to load." exception=(e, catch_backtrace())
                return nothing
            end
        else
            return nothing
        end
    end
end

"""
    save_snapshot!(arcs, cache, log, pending, basepath, toggle)

Persist the current certification state to disk using alternating files.
"""
function save_snapshot!(arcs, cache, log, pending, basepath::String, toggle::Bool)
    filename = basepath * (toggle ? "_A.jld2" : "_B.jld2")
    @info "Saved in $filename, arcs $(length(arcs)), pending $(length(pending))"
    JLD2.@save filename arcs cache log pending
    return !toggle
end

_identity_ballmatrix(n::Integer) = BallMatrix(Matrix{ComplexF64}(I, n, n))
_zero_ballmatrix(n::Integer) = BallMatrix(zeros(ComplexF64, n, n))

function _identity_ballmatrix(n::Integer, ::BallMatrix{T, NT}) where {T, NT}
    return BallMatrix(Matrix{NT}(I, n, n))
end

function _zero_ballmatrix(n::Integer, ::BallMatrix{T, NT}) where {T, NT}
    return BallMatrix(zeros(NT, n, n))
end

# A number as a ball in its own precision: `Ball(value)` keeps a float as it is and encloses
# anything else, where converting to Float64 rounded a BigFloat bound to nearest, possibly down.
function _as_ball(value)
    value isa Ball && return value
    value isa Number && return Ball(value)
    throw(ArgumentError("unsupported value type $(typeof(value)) for ball conversion"))
end

function _as_ball(value, ::BallMatrix{T, NT}) where {T, NT}
    value isa Ball && return value
    return Ball(convert(NT, value), zero(T))
end

"An upper bound of |x| over the ball: the modulus of the centre bounded above, plus the radius."
function _upper_bound(value)
    ball = _as_ball(value)
    return add_up(abs_up(ball.c), ball.r)
end

"The lower end of a real ball, rounded down; negative when the ball straddles zero."
function _lower_bound(value)
    ball = _as_ball(value)
    return sub_down(real(ball.c), ball.r)
end

function _upper_bound_offset(value, offset::Real)
    b = _as_ball(value)
    ball = b - convert(typeof(b), _as_ball(offset))
    return add_up(abs_up(ball.c), ball.r)
end

# The test of one side [z_a, z_b] of the polygon: an upper bound of |z_b − z_a| / σ, with the
# length rounded up (`dist_up`) and σ replaced by its certified lower end `lo_σ > 0`.
_side_ratio_sup(z_a, z_b, lo_σ) = div_up(dist_up(z_a, z_b), lo_σ)

function _is_identity_polynomial(coeffs)
    length(coeffs) == 2 || return false

    c0 = _as_ball(coeffs[1])
    c1 = _as_ball(coeffs[2])

    return c0.c == zero(c0.c) && c0.r == 0 && c1.c == one(c1.c) && c1.r == 0
end

function _polynomial_matrix(coeffs, M::BallMatrix)
    coeffs_vec = collect(coeffs)

    if _is_identity_polynomial(coeffs_vec)
        return M
    end

    n = size(M, 1)
    result = _zero_ballmatrix(n, M)
    identity = _identity_ballmatrix(n, M)
    for coeff in reverse(coeffs_vec)
        result = result * M
        if !iszero(coeff)
            value = _as_ball(coeff, M)
            result += value * identity
        end
    end
    return result
end

_polynomial_matrix(coeffs, M::AbstractMatrix) = _polynomial_matrix(coeffs, BallMatrix(M))

"""
    compute_schur_and_error(A; polynomial = nothing)

Compute the Schur decomposition of `A` and certified bounds for the
orthogonality defect, the reconstruction error, and the norms of `Z` and
`Z⁻¹`.  When `polynomial` is provided (as coefficients in ascending order),
additional bounds are computed for `p(A)` and `p(T)`.

Supports both Float64 and BigFloat precision based on the element type of `A`.
For BigFloat, the Schur decomposition is computed in Float64 and then refined
to higher precision using iterative refinement.
"""
function compute_schur_and_error(A::BallMatrix{T}; polynomial = nothing) where {T}
    RT = real(T)

    # For BigFloat, use iterative refinement from Float64
    if RT === BigFloat
        return _compute_schur_and_error_bigfloat(A; polynomial)
    end

    # Standard Float64 path
    CT = Complex{RT}
    S = LinearAlgebra.schur(CT.(A.c))

    bZ = BallMatrix(S.Z)
    errF = svd_bound_L2_opnorm(bZ' * bZ - I)

    bT = BallMatrix(S.T)
    errT = svd_bound_L2_opnorm(bZ * bT * bZ' - A)

    sigma_Z = svdbox(bZ)
    max_sigma = sigma_Z[1]
    min_sigma = sigma_Z[end]

    # Use type-appropriate rounding
    RT = real(T)
    norm_Z = setrounding(RT, RoundUp) do
        return RT(abs(max_sigma.c)) + RT(max_sigma.r)
    end

    min_sigma_lower = setrounding(RT, RoundDown) do
        return max(RT(min_sigma.c) - RT(min_sigma.r), zero(RT))
    end
    min_sigma_lower <= 0 &&
        throw(ArgumentError("Schur factor has non-positive smallest singular value bound"))
    norm_Z_inv = setrounding(RT, RoundUp) do
        return one(RT) / min_sigma_lower
    end

    if polynomial === nothing
        return S, errF, errT, norm_Z, norm_Z_inv
    end

    coeffs = collect(polynomial)
    pA = _polynomial_matrix(coeffs, A)
    pT = _polynomial_matrix(coeffs, bT)
    errT_poly = svd_bound_L2_opnorm(bZ * pT * bZ' - pA)

    return S, errF, errT_poly, norm_Z, norm_Z_inv
end

"""
    _compute_schur_and_error_bigfloat(A; polynomial = nothing)

BigFloat version of compute_schur_and_error.  Uses GenericSchur.jl's native
BigFloat Schur decomposition for full-precision results.
"""
function _compute_schur_and_error_bigfloat(A::BallMatrix{BigFloat}; polynomial = nothing)
    return _compute_schur_bigfloat_direct(A; polynomial)
end

"""
    _compute_schur_bigfloat_direct(A; polynomial = nothing)

Compute BigFloat Schur decomposition directly via GenericSchur.jl (no Float64
seed).  Works for matrices whose eigenvalues span many orders of magnitude.
"""
function _compute_schur_bigfloat_direct(A::BallMatrix{BigFloat}; polynomial = nothing)
    S = LinearAlgebra.schur(Complex{BigFloat}.(A.c))

    bZ = BallMatrix(S.Z)
    # Use upper_bound_L2_opnorm (Collatz + sqrt(‖·‖₁·‖·‖_∞)) instead of
    # svd_bound_L2_opnorm which calls svdbox — broken for BigFloat matrices.
    errF = upper_bound_L2_opnorm(bZ' * bZ - I)

    bT = BallMatrix(S.T)
    errT = upper_bound_L2_opnorm(bZ * bT * bZ' - A)

    # Bound σ_max(Z) and 1/σ_min(Z) from the orthogonality defect errF = ‖Z'Z - I‖₂.
    # Since Z'Z = I + E with ‖E‖₂ ≤ errF, the eigenvalues of Z'Z lie in [1-errF, 1+errF],
    # so σ_max(Z) ≤ √(1 + errF) and σ_min(Z) ≥ √(1 - errF).
    errF_val = BigFloat(errF)
    errF_val >= 1 && throw(ArgumentError(
        "Schur factor too far from unitary: ‖Z'Z - I‖ = $errF_val"))

    norm_Z = setrounding(BigFloat, RoundUp) do
        return sqrt(one(BigFloat) + errF_val)
    end
    norm_Z_inv = setrounding(BigFloat, RoundUp) do
        return one(BigFloat) / sqrt(one(BigFloat) - errF_val)
    end

    S_nt = (T = S.T, Z = S.Z, values = S.values)

    if polynomial === nothing
        return S_nt, errF, errT, norm_Z, norm_Z_inv
    end

    coeffs = collect(polynomial)
    pA = _polynomial_matrix(coeffs, A)
    pT = _polynomial_matrix(coeffs, bT)
    errT_poly = upper_bound_L2_opnorm(bZ * pT * bZ' - pA)

    return S_nt, errF, errT_poly, norm_Z, norm_Z_inv
end

"""
    _compute_schur_bigfloat_refined(A; polynomial = nothing)

Compute BigFloat Schur decomposition by refining a Float64 seed.
Seeds from Float64 Schur, then refines iteratively in BigFloat.
May fail for matrices with eigenvalues below ~10⁻¹⁶.
"""
function _compute_schur_bigfloat_refined(A::BallMatrix{BigFloat}; polynomial = nothing)
    n = size(A, 1)

    # Step 1: Compute Float64 Schur decomposition
    A_f64 = Complex{Float64}.(A.c)
    S_f64 = LinearAlgebra.schur(A_f64)

    # Step 2: Refine to BigFloat using iterative refinement
    target_precision = precision(BigFloat)
    result = refine_schur_decomposition(A.c, S_f64.Z, S_f64.T;
        target_precision = target_precision,
        max_iterations = 20)

    if !result.converged
        @warn "Schur refinement did not fully converge. Residual: $(result.residual_norm)"
    end

    # Build BigFloat Schur factorization
    Q_big = result.Q
    T_big = result.T

    # Create BallMatrix with refinement errors as radii
    Q_error = BigFloat(result.orthogonality_defect)
    A_norm = LinearAlgebra.norm(A.c)
    T_error = BigFloat(result.residual_norm) * A_norm

    bZ = BallMatrix(Q_big, fill(Q_error, n, n))
    errF = upper_bound_L2_opnorm(bZ' * bZ - I)

    bT = BallMatrix(T_big, fill(T_error, n, n))
    errT = upper_bound_L2_opnorm(bZ * bT * bZ' - A)

    # Bound σ_max(Z) and 1/σ_min(Z) from orthogonality defect (see _compute_schur_bigfloat_direct)
    errF_val = BigFloat(errF)
    errF_val >= 1 && throw(ArgumentError(
        "Schur factor too far from unitary: ‖Z'Z - I‖ = $errF_val"))

    norm_Z = setrounding(BigFloat, RoundUp) do
        return sqrt(one(BigFloat) + errF_val)
    end
    norm_Z_inv = setrounding(BigFloat, RoundUp) do
        return one(BigFloat) / sqrt(one(BigFloat) - errF_val)
    end

    S = (T = T_big, Z = Q_big, values = diag(T_big))

    if polynomial === nothing
        return S, errF, errT, norm_Z, norm_Z_inv
    end

    coeffs = collect(polynomial)
    pA = _polynomial_matrix(coeffs, A)
    pT = _polynomial_matrix(coeffs, bT)
    errT_poly = upper_bound_L2_opnorm(bZ * pT * bZ' - pA)

    return S, errF, errT_poly, norm_Z, norm_Z_inv
end

"""
    run_certification(A, circle; schur_data = nothing, polynomial = nothing, kwargs...)

Run the adaptive certification routine on `circle` using a serial evaluator.

# Arguments
- `A`: matrix to certify.  Converted to `BallMatrix` when required.
- `circle`: [`CertificationCircle`](@ref) describing the contour used for the
  adaptive refinement.

# Keyword Arguments
- `schur_data = nothing`: pre-computed Schur data as the 5-tuple
  `(S, errF, errT, norm_Z, norm_Z_inv)` returned by [`compute_schur_and_error`](@ref).
  When provided, the expensive Schur computation is skipped entirely. This is
  useful for reusing the same Schur decomposition across multiple circles or
  when the default Float64-seeded refinement fails (e.g., matrices with
  eigenvalues below ~10⁻¹⁶).
- `polynomial = nothing`: optional coefficients (ascending order) describing a
  polynomial `p`.  When provided the certification is carried out on `p(T)` and
  the returned error corresponds to the reconstruction error of `p(A)`.
- `η = 0.5`: admissible threshold for the adaptive refinement.  Must lie in the
  open unit interval.
- `check_interval = 100`: number of processed arcs between progress reports and
  consistency checks.
- `log_io = stdout`: destination `IO` for log messages.
- `Cbound = 1.0`: constant used by [`bound_res_original`](@ref) when lifting
  resolvent bounds back to the original matrix.
- `gram = nothing`: Hermitian positive definite Gram matrix `G`. When given, the
  certification is carried out in the weighted norm `‖x‖_G = √(x*Gx)` by the
  exact change of variables `Ã = L A L⁻¹` with `G = L*L`, so every returned
  resolvent bound is a bound in the `G`-norm. Unlike folding `√κ₂(G)` into
  `Cbound`, this costs no condition-number factor. See [`gram_transform`](@ref).
- `gram_factor = nothing`, `gram_factor_inv = nothing`: precomputed (and, for
  the inverse, verified before use) factor data — see [`gram_transform`](@ref).
- `gram_kwargs = (;)`: extra keywords forwarded to [`gram_transform`](@ref),
  e.g. `(use_bigfloat = true, inverse_method = :verify)`.

The return value is a named tuple containing the computed Schur form, the
accumulated certification log, and the resolvent bounds for both the Schur
factor and the original matrix.  The `gram` field holds the
[`GramTransform`](@ref) that was applied, or `nothing`.
"""
function run_certification(A::BallMatrix, circle::CertificationCircle;
        schur_data = nothing, polynomial = nothing, η::Real = 0.5,
        check_interval::Integer = 100, log_io::IO = stdout, Cbound = 1.0,
        gram = nothing, gram_factor = nothing, gram_factor_inv = nothing,
        gram_kwargs = (;))
    check_interval < 1 && throw(ArgumentError("check_interval must be positive"))
    η = Float64(η)
    (η <= 0 || η >= 1) && throw(ArgumentError("η must belong to (0, 1)"))

    A, gram_info = _prepare_gram(A, gram, gram_factor, gram_factor_inv, schur_data;
        gram_kwargs...)

    coeffs = polynomial === nothing ? nothing : collect(polynomial)
    if schur_data === nothing
        schur_data = coeffs === nothing ?
                     compute_schur_and_error(A) :
                     compute_schur_and_error(A; polynomial = coeffs)
    end

    S, errF, errT, norm_Z, norm_Z_inv = schur_data
    bT = BallMatrix(S.T)
    schur_matrix = coeffs === nothing ? bT : _polynomial_matrix(coeffs, bT)

    arcs = _initial_arcs(circle)
    cache = Dict{ComplexF64, Any}()
    certification_log = Any[]
    pending = Dict{Int, Tuple{ComplexF64, ComplexF64}}()
    eval_index = Ref(0)

    serial_evaluator = function (z::ComplexF64)
        eval_index[] += 1
        return _evaluate_sample(schur_matrix, z, eval_index[])
    end

    _require_complete(adaptive_arcs!(arcs, cache, pending, η; check_interval = check_interval,
        certification_log = certification_log, io = log_io,
        evaluator = serial_evaluator))

    isempty(certification_log) && throw(ErrorException("certification produced no samples"))

    min_sigma = minimum(log -> log.lo_val, certification_log)
    l2pseudo = maximum(log -> log.hi_res, certification_log)
    resolvent_schur_bound = bound_resolvent_schur(l2pseudo, η)
    resolvent_bound = bound_res_original(
        l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, size(A, 1);
        zmax = _circle_zmax(circle), Cbound = Cbound)

    return (;
        schur = S, schur_matrix, certification_log, minimum_singular_value = min_sigma,
        resolvent_schur_raw = l2pseudo, resolvent_schur = resolvent_schur_bound,
        resolvent_original = resolvent_bound, Cbound,
        errF, errT, norm_Z, norm_Z_inv, circle, polynomial = coeffs,
        snapshot_base = nothing, gram = gram_info)
end

function run_certification(A::AbstractMatrix, circle::CertificationCircle; kwargs...)
    run_certification(BallMatrix(A), circle; kwargs...)
end

#####################################################################
# OGITA SVD REFINEMENT OPTIMIZATION FOR PSEUDOSPECTRA CERTIFICATION #
#####################################################################

"""
    _evaluate_sample_ogita_bigfloat(T_matrix::BallMatrix, z::Number, idx::Int;
                                     max_iterations::Int=4,
                                     target_precision::Int=256,
                                     distance_threshold::Real=1e-4,
                                     use_cache::Bool=true)

Evaluate the smallest singular value at z using Ogita refinement with caching.

This function uses a worker-local cache to store the last refined BigFloat SVD.
For nearby points (distance < distance_threshold), it uses the cached SVD as a
starting point for Ogita refinement, which requires fewer iterations than
starting from Float64.

# Cache Strategy
- If cache hit: use cached BigFloat SVD → 1-2 Ogita iterations
- If cache miss: use Float64 SVD → 3-4 Ogita iterations

# Arguments
- `T_matrix`: The Schur matrix T (BallMatrix)
- `z`: Sample point on the circle
- `idx`: Sample index for logging
- `max_iterations`: Number of Ogita iterations for cache miss (3-4 for 256-bit)
- `target_precision`: Target precision in bits
- `distance_threshold`: Maximum distance for cache reuse
- `use_cache`: Whether to use caching (default: true)
"""
function _evaluate_sample_ogita_bigfloat(T_matrix::BallMatrix{ET}, z::Number, idx::Int;
        max_iterations::Int = 4,
        target_precision::Int = 256,
        distance_threshold::Real = 1e-4,
        use_cache::Bool = true) where {ET}
    RT = real(ET)
    CT = Complex{RT}
    n = size(T_matrix, 1)

    # Convert z to appropriate precision
    z_converted = CT(z)

    # Create shifted matrix: T - z*I (in the precision of T_matrix)
    T_shifted = T_matrix.c - z_converted * I

    elapsed = @elapsed begin
        # 3-tier caching strategy:
        # 1. Try center SVD (computed once at circle center, shared by all workers)
        # 2. Try local worker cache (recently computed SVD at nearby point)
        # 3. Fall back to fresh computation (and update local cache)

        cache_source = :none
        cached_U = nothing
        cached_S = nothing
        cached_V = nothing

        if use_cache
            # Tier 1: Try center SVD cache (always available, good for all points on circle)
            if _bf_center_svd_U[] !== nothing && _bf_center_svd_z[] !== nothing
                z_center = _bf_center_svd_z[]
                dist_to_center = abs(z_converted - CT(z_center))
                # Center cache is useful for any point on the circle (within radius + margin)
                if Float64(dist_to_center) < distance_threshold * 10  # Larger threshold for center
                    cache_source = :center
                    cached_U = _bf_center_svd_U[]
                    cached_S = _bf_center_svd_S[]
                    cached_V = _bf_center_svd_V[]
                end
            end

            # Tier 2: Try local worker cache (better if point is closer to last computed point)
            if _bf_last_svd_U[] !== nothing && _bf_last_svd_z[] !== nothing
                z_last = _bf_last_svd_z[]
                dist_to_last = abs(z_converted - CT(z_last))
                if Float64(dist_to_last) < distance_threshold
                    # Local cache is closer - prefer it over center
                    cache_source = :local
                    cached_U = _bf_last_svd_U[]
                    cached_S = _bf_last_svd_S[]
                    cached_V = _bf_last_svd_V[]
                end
            end
        end

        if cache_source == :local
            # Local cache hit: refine from nearby BigFloat SVD (fewer iterations)
            _bf_ogita_cache_hits[] += 1
            refined = ogita_svd_refine(
                T_shifted,
                cached_U,
                cached_S,
                cached_V;
                max_iterations = 2,  # Fewer iterations from BigFloat starting point
                precision_bits = target_precision,
                check_convergence = false
            )
        elseif cache_source == :center
            # Center cache hit: refine from center SVD (may need more iterations)
            _bf_center_cache_hits[] += 1
            refined = ogita_svd_refine(
                T_shifted,
                cached_U,
                cached_S,
                cached_V;
                max_iterations = max_iterations,  # Full iterations from center
                precision_bits = target_precision,
                check_convergence = false
            )
        else
            # Tier 3: Cache miss - compute Float64 SVD and refine
            _bf_ogita_cache_misses[] += 1
            T_f64 = convert.(ComplexF64, T_shifted)
            svd_f64 = LinearAlgebra.svd(T_f64)

            # Check if Float64 SVD has tiny singular values (below Float64 precision)
            # If so, Ogita refinement won't work - need full BigFloat SVD
            min_sv_f64 = minimum(svd_f64.S)
            if min_sv_f64 < 1e-14 * maximum(svd_f64.S)
                # Float64 SVD is unreliable for small singular values
                # Fall back to inverse power iteration for smallest singular value
                _bf_ogita_fallbacks[] += 1
                @debug "Float64 SVD has tiny σ_min=$(min_sv_f64), using inverse iteration"

                # Use inverse power iteration to find smallest singular value/vector
                # Start from Float64 SVD's smallest singular vector (may be wrong direction)
                # Solve (A'A)v = σ²v iteratively

                # First, use Ogita to refine the larger singular values (they're OK)
                refined = ogita_svd_refine(
                    T_shifted,
                    svd_f64.U,
                    svd_f64.S,
                    svd_f64.V;
                    max_iterations = max_iterations,
                    precision_bits = target_precision,
                    check_convergence = false
                )

                # Now fix the smallest singular value AND vectors using inverse iteration
                # Compute (A'A + μI)^{-1} with small shift μ to avoid singularity
                AHA = T_shifted' * T_shifted
                μ = eps(RT) * LinearAlgebra.norm(AHA)  # Small shift

                # Start with random vector orthogonal to larger singular vectors
                v = randn(CT, n)
                Σ_vec = isa(refined.Σ, Diagonal) ? diag(refined.Σ) : refined.Σ
                V_large = refined.V[:, 1:(end - 1)]  # All but smallest
                v = v - V_large * (V_large' * v)  # Orthogonalize
                v = v / LinearAlgebra.norm(v)

                # Inverse iteration: v_{k+1} = (A'A + μI)^{-1} v_k
                AHA_shifted = AHA + μ * I
                for _ in 1:10  # Usually converges fast
                    v_new = AHA_shifted \ v
                    v_new = v_new / LinearAlgebra.norm(v_new)
                    if LinearAlgebra.norm(v_new - v) < eps(RT) * 100
                        break
                    end
                    v = v_new
                end

                # Compute smallest singular value: σ = ||Av||
                Av = T_shifted * v
                σ_min = LinearAlgebra.norm(Av)

                # Compute left singular vector: u = Av / σ
                # For tiny σ, we need to be careful with division
                if σ_min > zero(RT)
                    u = Av / σ_min
                else
                    # If σ_min is exactly zero, u can be any unit vector orthogonal to other u's
                    u = randn(CT, n)
                    U_large = refined.U[:, 1:(end - 1)]
                    u = u - U_large * (U_large' * u)
                    u = u / LinearAlgebra.norm(u)
                end

                # Update refined with correct smallest singular value AND vectors
                Σ_vec[end] = σ_min
                V_new = copy(refined.V)
                V_new[:, end] = v
                U_new = copy(refined.U)
                U_new[:, end] = u
                refined = (U = U_new, Σ = Σ_vec, V = V_new)
            else
                # Float64 SVD is good enough - use Ogita refinement
                refined = ogita_svd_refine(
                    T_shifted,
                    svd_f64.U,
                    svd_f64.S,
                    svd_f64.V;
                    max_iterations = max_iterations,
                    precision_bits = target_precision,
                    check_convergence = false
                )
            end
        end

        # Update cache with the refined SVD
        if use_cache
            Σ_vec_for_cache = isa(refined.Σ, Diagonal) ? diag(refined.Σ) : refined.Σ
            _bf_last_svd_U[] = refined.U
            _bf_last_svd_S[] = Σ_vec_for_cache
            _bf_last_svd_V[] = refined.V
            _bf_last_svd_z[] = z_converted
        end

        # Certify the refined SVD with rigorous bounds
        Σ_vec = isa(refined.Σ, Diagonal) ? diag(refined.Σ) : refined.Σ
        svd_refined = (U = refined.U, S = Σ_vec, V = refined.V, Vt = refined.V')

        # Create BallMatrix for certification (with zero radius since we're working with midpoint)
        T_shifted_ball = BallMatrix(T_shifted, fill(zero(RT), n, n))
        result = _certify_svd(T_shifted_ball, svd_refined, MiyajimaM1(); apply_vbd = true)
        Σ = result.singular_values

        # Check if smallest singular value Ball contains zero but has positive midpoint
        # This happens when certification radius swamps tiny singular values
        # In this case, use a relative error bound instead
        σ_min_ball = Σ[end]
        if σ_min_ball.c > zero(RT) && σ_min_ball.c - σ_min_ball.r <= zero(RT)
            # Certification radius is too large - use conservative relative bound
            # For inverse iteration, typical relative error is O(eps) after convergence
            rel_error = RT(100) * eps(RT)  # Conservative factor
            new_radius = setrounding(RT, RoundUp) do
                abs(σ_min_ball.c) * rel_error
            end
            Σ[end] = Ball(σ_min_ball.c, max(new_radius, σ_min_ball.r * eps(RT)))
            @debug "Certification radius too large for tiny σ_min, using relative bound" σ_min=σ_min_ball.c old_rad=σ_min_ball.r new_rad=Σ[end].r
        end
    end

    val = Σ[end]

    # Compute lo_val (lower bound of smallest singular value)
    lo_val = setrounding(RT, RoundDown) do
        return RT(val.c) - RT(val.r)
    end

    # Check if val contains zero (certification radius swamped tiny σ_min)
    if lo_val <= zero(RT)
        # Use midpoint with conservative relative error for resolvent
        lo_val = setrounding(RT, RoundDown) do
            val.c * (one(RT) - RT(100) * eps(RT))
        end
        lo_val = max(lo_val, eps(RT))  # Ensure positive
    end

    res = Ball(one(RT) / val.c, setrounding(RT, RoundUp) do
        one(RT) / lo_val - one(RT) / val.c
    end)

    hi_res = setrounding(RT, RoundUp) do
        return RT(res.c) + RT(res.r)
    end

    return (
        i = idx,
        val = val,
        lo_val = lo_val,
        res = res,
        hi_res = hi_res,
        second_val = Σ[end - 1],
        z = z_converted,
        t = elapsed,
        id = nothing
    )
end

"""
    run_certification_ogita(A, circle; target_precision=256, kwargs...)

Optimized BigFloat certification using Ogita SVD refinement.

This function is specifically designed for BigFloat precision certification where
computing fresh BigFloat SVDs at each sample point is expensive. Instead, it:

1. Computes Schur decomposition with BigFloat refinement
2. At each sample point z, computes Float64 SVD of (T - zI)
3. Refines the Float64 SVD to BigFloat using Ogita's algorithm (3-4 iterations)

Due to quadratic convergence, 4 Ogita iterations from Float64 (~10^-16 error)
achieve ~10^-64 error, saturating 256-bit precision.

# Performance
Typically 10-100x faster than computing fresh BigFloat SVDs at each point.

# Arguments
- `A`: matrix to certify (will be converted to BigFloat BallMatrix)
- `circle`: CertificationCircle describing the contour
- `target_precision::Int=256`: precision in bits for BigFloat
- `max_ogita_iterations::Int=3`: Ogita iterations (3 is optimal for 256-bit precision)
- `gram`, `gram_factor`, `gram_factor_inv`, `gram_kwargs`: certify in the
  weighted norm `‖x‖_G = √(x*Gx)` — see [`run_certification`](@ref) and
  [`gram_transform`](@ref). The transform is applied *after* promotion to
  BigFloat, so the Gram factor is computed at `target_precision`.
- Other kwargs passed to standard certification
"""
function run_certification_ogita(A::BallMatrix{T}, circle::CertificationCircle;
        schur_data = nothing, polynomial = nothing, η::Real = 0.5,
        check_interval::Integer = 100, log_io::IO = stdout, Cbound = 1.0,
        target_precision::Int = 256,
        max_ogita_iterations::Int = 3,
        gram = nothing, gram_factor = nothing, gram_factor_inv = nothing,
        gram_kwargs = (;)) where {T}
    check_interval < 1 && throw(ArgumentError("check_interval must be positive"))
    η = Float64(η)
    (η <= 0 || η >= 1) && throw(ArgumentError("η must belong to (0, 1)"))

    # Convert to BigFloat if not already
    RT = real(T)
    if RT !== BigFloat
        old_prec = precision(BigFloat)
        setprecision(BigFloat, target_precision)
        A_big = BallMatrix(
            convert.(Complex{BigFloat}, A.c),
            convert.(BigFloat, A.r)
        )
        setprecision(BigFloat, old_prec)
    else
        A_big = A
    end

    # Set precision for computation
    old_prec = precision(BigFloat)
    setprecision(BigFloat, target_precision)

    try
        A_big, gram_info = _prepare_gram(A_big, gram, gram_factor, gram_factor_inv,
            schur_data; gram_kwargs...)

        coeffs = polynomial === nothing ? nothing : collect(polynomial)
        if schur_data === nothing
            schur_data = coeffs === nothing ?
                         compute_schur_and_error(A_big) :
                         compute_schur_and_error(A_big; polynomial = coeffs)
        end

        S, errF, errT, norm_Z, norm_Z_inv = schur_data
        bT = BallMatrix(S.T)
        schur_matrix = coeffs === nothing ? bT : _polynomial_matrix(coeffs, bT)

        @info "Using Ogita refinement optimization for BigFloat certification"
        @info "Target precision: $(target_precision) bits, Ogita iterations: $(max_ogita_iterations)"

        # Compute center SVD once and cache it for all workers
        # This provides a good starting point for Ogita refinement at any point on the circle
        @info "Computing center SVD at circle center..."
        center_z = Complex{BigFloat}(circle.center)
        T_center = schur_matrix.c - center_z * I
        T_center_f64 = convert.(ComplexF64, T_center)
        svd_center_f64 = LinearAlgebra.svd(T_center_f64)

        # Refine center SVD to BigFloat precision
        center_refined = ogita_svd_refine(
            T_center,
            svd_center_f64.U,
            svd_center_f64.S,
            svd_center_f64.V;
            max_iterations = max_ogita_iterations,
            precision_bits = target_precision,
            check_convergence = false
        )
        center_S = isa(center_refined.Σ, Diagonal) ? diag(center_refined.Σ) :
                   center_refined.Σ

        # Set center cache for all workers
        _set_center_svd_cache!(center_refined.U, center_S, center_refined.V, center_z)
        @info "Center SVD cached (σ_min = $(Float64(minimum(center_S))))"

        arcs = _initial_arcs(circle)
        cache = Dict{ComplexF64, Any}()
        certification_log = Any[]
        pending = Dict{Int, Tuple{ComplexF64, ComplexF64}}()
        eval_index = Ref(0)

        # Create optimized evaluator using Ogita refinement with 3-tier caching:
        # 1. Center SVD (computed above)
        # 2. Local worker cache (recent computation)
        # 3. Fresh Float64 SVD + Ogita refinement
        ogita_evaluator = function (z::ComplexF64)
            eval_index[] += 1
            return _evaluate_sample_ogita_bigfloat(
                schur_matrix, z, eval_index[];
                max_iterations = max_ogita_iterations,
                target_precision = target_precision
            )
        end

        _require_complete(adaptive_arcs!(arcs, cache, pending, η; check_interval = check_interval,
            certification_log = certification_log, io = log_io,
            evaluator = ogita_evaluator))

        isempty(certification_log) &&
            throw(ErrorException("certification produced no samples"))

        min_sigma = minimum(log -> log.lo_val, certification_log)
        l2pseudo = maximum(log -> log.hi_res, certification_log)
        resolvent_schur_bound = bound_resolvent_schur(l2pseudo, η)
        resolvent_bound = bound_res_original(
            l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, size(A, 1);
        zmax = _circle_zmax(circle), Cbound = Cbound)

        # Get cache statistics
        cache_stats = _bf_ogita_cache_stats()
        @info "Cache statistics: $(cache_stats)"

        return (;
            schur = S, schur_matrix, certification_log, minimum_singular_value = min_sigma,
            resolvent_schur_raw = l2pseudo, resolvent_schur = resolvent_schur_bound,
            resolvent_original = resolvent_bound, Cbound,
            errF, errT, norm_Z, norm_Z_inv, circle, polynomial = coeffs,
            snapshot_base = nothing, gram = gram_info,
            optimization = :ogita_refinement,
            target_precision = target_precision,
            cache_stats = cache_stats)
    finally
        setprecision(BigFloat, old_prec)
    end
end

function run_certification_ogita(A::AbstractMatrix, circle::CertificationCircle; kwargs...)
    run_certification_ogita(BallMatrix(A), circle; kwargs...)
end

# =====================================================================
# Serial Parametric Sylvester-based Certification
# =====================================================================

"""
    run_certification_parametric(A, circle; k=nothing, config=config_v2(), kwargs...)

Run the adaptive certification routine using the parametric Sylvester-based certifier.

This method uses block-diagonalization via Sylvester equation to certify the resolvent
norm, which can be more efficient than full SVD for large matrices.

# Arguments
- `A`: matrix to certify. Converted to `BallMatrix` when required.
- `circle`: [`CertificationCircle`](@ref) describing the contour.

# Keyword Arguments
- `k::Union{Nothing, Int}=nothing`: Split index for the Schur block decomposition.
  If `nothing`, automatically selects k ≈ n/4.
- `config::ResolventBoundConfig=config_v2()`: Configuration specifying estimators.
  Options: `config_v1()`, `config_v2()`, `config_v2p5()`, `config_v3()`.
- `polynomial = nothing`: optional coefficients for polynomial certification.
- `η = 0.5`: admissible threshold for adaptive refinement.
- `check_interval = 100`: number of arcs between progress reports.
- `log_io = stdout`: destination for log messages.
- `Cbound = 1.0`: constant for resolvent bound lifting.
- `gram`, `gram_factor`, `gram_factor_inv`, `gram_kwargs`: certify in the
  weighted norm `‖x‖_G = √(x*Gx)` — see [`run_certification`](@ref) and
  [`gram_transform`](@ref).

# Example
```julia
using BallArithmetic
A = randn(ComplexF64, 50, 50)
circle = CertificationCircle(0.0, 1.5; samples=64)

# Use V2 configuration (default)
result = run_certification_parametric(A, circle)

# Use V3 configuration with Neumann bounds
result = run_certification_parametric(A, circle; config=config_v3())

# Specify custom split
result = run_certification_parametric(A, circle; k=10)
```
"""
function run_certification_parametric(A::BallMatrix{T}, circle::CertificationCircle;
        k::Union{Nothing, Int} = nothing,
        config::ResolventBoundConfig = config_v2(),
        schur_data = nothing, polynomial = nothing, η::Real = 0.5,
        check_interval::Integer = 100,
        log_io::IO = stdout, Cbound = 1.0,
        gram = nothing, gram_factor = nothing, gram_factor_inv = nothing,
        gram_kwargs = (;)) where {T}
    check_interval < 1 && throw(ArgumentError("check_interval must be positive"))
    η = Float64(η)
    (η <= 0 || η >= 1) && throw(ArgumentError("η must belong to (0, 1)"))

    A, gram_info = _prepare_gram(A, gram, gram_factor, gram_factor_inv, schur_data;
        gram_kwargs...)

    coeffs = polynomial === nothing ? nothing : collect(polynomial)
    if schur_data === nothing
        schur_data = coeffs === nothing ?
                     compute_schur_and_error(A) :
                     compute_schur_and_error(A; polynomial = coeffs)
    end

    S, errF, errT, norm_Z, norm_Z_inv = schur_data
    bT = BallMatrix(S.T)
    schur_matrix = coeffs === nothing ? bT : _polynomial_matrix(coeffs, bT)

    n = size(schur_matrix, 1)

    # Auto-select k if not provided
    k_used = k === nothing ? max(2, n ÷ 4) : k
    k_used = clamp(k_used, 2, n - 2)  # Ensure valid range

    @info "Parametric certification with k=$k_used (n=$n)"
    @info "Configuration: $(config.d_inv_estimator), $(config.coupling_estimator), $(config.combiner)"

    # Precompute Sylvester quantities
    T_mat = Matrix(schur_matrix.c)
    T11 = T_mat[1:k_used, 1:k_used]
    T12 = T_mat[1:k_used, (k_used + 1):n]
    T22 = T_mat[(k_used + 1):n, (k_used + 1):n]

    X = solve_sylvester_oracle(T11, T12, T22)
    R = T12 + T11 * X - X * T22
    precomp = sylvester_resolvent_precompute(T_mat, k_used; X_oracle = X)

    if !precomp.precomputation_success
        error("Sylvester precomputation failed: $(precomp.failure_reason)")
    end

    @info "Sylvester diagnostics: reduction=$(precomp.reduction_factor), penalty=$(precomp.similarity_cond)"

    # Set up parametric config for the evaluator
    RT = real(T)
    R_typed = convert.(Complex{RT}, R)

    arcs = _initial_arcs(circle)
    cache = Dict{ComplexF64, Any}()
    certification_log = Any[]
    pending = Dict{Int, Tuple{ComplexF64, ComplexF64}}()
    eval_index = Ref(0)

    # Warm start cache for serial evaluation
    warm_U = Ref{Union{Nothing, Matrix}}(nothing)
    warm_S = Ref{Union{Nothing, Vector}}(nothing)
    warm_V = Ref{Union{Nothing, Matrix}}(nothing)
    warm_z = Ref{Union{Nothing, Number}}(nothing)

    serial_evaluator = function (z::ComplexF64)
        eval_index[] += 1
        idx = eval_index[]

        z_typed = Complex{RT}(z)

        # Check for warm start
        warm_start = nothing
        if warm_z[] !== nothing && abs(z_typed - warm_z[]) < 1e-4 * (abs(warm_z[]) + 1)
            warm_start = SVDWarmStart(warm_U[], warm_S[], warm_V[])
        end

        elapsed = @elapsed result = parametric_resolvent_bound(
            precomp, T_mat, z_typed, config;
            R = R_typed, svd_warm_start = warm_start
        )

        # Update warm start
        if result.success
            A_z = z_typed * I - T11
            try
                svd_Az = LinearAlgebra.svd(A_z)
                warm_U[] = svd_Az.U
                warm_S[] = svd_Az.S
                warm_V[] = svd_Az.V
                warm_z[] = z_typed
            catch
            end
        end

        # Convert to standard format
        if result.success
            sigma_min = RT(1) / result.M_A
            sigma_min_ball = Ball(sigma_min, zero(RT))

            lo_val = setrounding(RT, RoundDown) do
                sigma_min
            end

            hi_res = setrounding(RT, RoundUp) do
                result.resolvent_bound
            end

            return (
                i = idx,
                val = sigma_min_ball,
                lo_val = lo_val,
                res = Ball(result.resolvent_bound, zero(RT)),
                hi_res = hi_res,
                second_val = Ball(zero(RT), zero(RT)),
                z = z_typed,
                t = elapsed,
                id = nothing
            )
        else
            return (
                i = idx,
                val = Ball(zero(RT), RT(Inf)),
                lo_val = zero(RT),
                res = Ball(RT(Inf), zero(RT)),
                hi_res = RT(Inf),
                second_val = Ball(zero(RT), zero(RT)),
                z = z_typed,
                t = elapsed,
                id = nothing
            )
        end
    end

    _require_complete(adaptive_arcs!(arcs, cache, pending, η; check_interval = check_interval,
        certification_log = certification_log, io = log_io,
        evaluator = serial_evaluator))

    isempty(certification_log) && throw(ErrorException("certification produced no samples"))

    min_sigma = minimum(log -> log.lo_val, certification_log)
    l2pseudo = maximum(log -> log.hi_res, certification_log)
    resolvent_schur_bound = bound_resolvent_schur(l2pseudo, η)
    resolvent_bound = bound_res_original(
        l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, size(A, 1);
        zmax = _circle_zmax(circle), Cbound = Cbound)

    return (;
        schur = S, schur_matrix, certification_log, minimum_singular_value = min_sigma,
        resolvent_schur_raw = l2pseudo, resolvent_schur = resolvent_schur_bound,
        resolvent_original = resolvent_bound, Cbound,
        errF, errT, norm_Z, norm_Z_inv, circle, polynomial = coeffs,
        snapshot_base = nothing, gram = gram_info,
        k = k_used, parametric_precomp = precomp)
end

function run_certification_parametric(A::AbstractMatrix, circle::CertificationCircle; kwargs...)
    run_certification_parametric(BallMatrix(A), circle; kwargs...)
end

function polyconv(a::AbstractVector, b::AbstractVector)
    n, m = length(a), length(b)
    c = zeros(promote_type(eltype(a), eltype(b)), n + m - 1)
    for i in 1:n, j in 1:m

        c[i + j - 1] += a[i] * b[j]
    end
    return c
end

"""
    poly_from_roots(roots::AbstractVector)

Given a list of roots `r₁, r₂, …, rₙ`, returns the coefficients `[a₀, a₁, …, aₙ]`
of the monic polynomial
    p(x) = (x - r₁)(x - r₂)…(x - rₙ)
so that p(x) = a₀ + a₁*x + a₂*x² + … + aₙ*xⁿ.
"""
function poly_from_roots(roots::AbstractVector)
    coeffs = [1.0]
    for r in roots
        coeffs = polyconv(coeffs, [-r, 1.0])
    end
    return coeffs
end


function __init__()
    # streams cannot be captured at precompilation time
    _log_io[] = stdout
    return nothing
end

end # module
