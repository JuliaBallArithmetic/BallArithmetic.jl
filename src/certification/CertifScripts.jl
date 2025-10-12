module CertifScripts

using LinearAlgebra
using JLD2
using Base: dirname, mod1

using ..BallArithmetic: Ball, BallMatrix, svdbox, svd_bound_L2_opnorm

export dowork, adaptive_arcs!, bound_res_original, choose_snapshot_to_load,
       save_snapshot!, configure_certification!, set_schur_matrix!,
       compute_schur_and_error, CertificationCircle, points_on, run_certification, poly_from_roots

const _schur_matrix = Ref{Union{Nothing, BallMatrix}}(nothing)
const _job_channel = Ref{Any}(nothing)
const _result_channel = Ref{Any}(nothing)
const _certification_log = Ref{Any}(nothing)
const _snapshot_path = Ref{Union{Nothing, AbstractString}}(nothing)
const _log_io = Ref{IO}(stdout)

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

function _evaluate_sample(T::BallMatrix, z::ComplexF64, idx::Int)
    bz = Ball(z, 0.0)

    elapsed = @elapsed Σ = svdbox(T - bz * LinearAlgebra.I)

    val = Σ[end]
    res = 1 / val

    lo_val = setrounding(Float64, RoundDown) do
        return val.c - val.r
    end

    hi_res = setrounding(Float64, RoundUp) do
        return res.c + res.r
    end

    return (
        i = idx,
        val = val,
        lo_val = lo_val,
        res = res,
        hi_res = hi_res,
        second_val = Σ[end - 1],
        z = z,
        t = elapsed,
        id = nothing,
    )
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
        snapshot, io, check_interval::Integer, evaluator)

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

        ℓ = abs(z_b - z_a)
        ε = ℓ / σ_a

        sup_ε = setrounding(Float64, RoundUp) do
            return ε.c + ε.r
        end

        if sup_ε > η
            z_m = (z_a + z_b) / 2
            push!(arcs, (z_m, z_b))
            push!(arcs, (z_a, z_m))
            new += 1
        end

        processed += 1
        if processed % check_interval == 0
            @info "Processed $processed arcs (serial)" remaining=length(arcs) new_ratio=new / check_interval
            flush(io)
            new = 0
            if snapshot !== nothing
                cycle = save_snapshot!(arcs, cache, certification_log, pending, snapshot, cycle)
            end
        end
    end

    return nothing
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

            ℓ = abs(z_b - z_a)
            ε = ℓ / σ_a

            sup_ε = setrounding(Float64, RoundUp) do
                return ε.c + ε.r
            end

            if sup_ε > η
                z_m = (z_a + z_b) / 2
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
                    cycle = save_snapshot!(arcs, cache, certification_log, pending, snapshot, cycle)
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
    return nothing
end

"""
    adaptive_arcs!(arcs, cache, pending, η; kwargs...)

Drive the adaptive refinement routine.  When job channels are provided the
refinement uses asynchronous workers; otherwise the evaluation is carried out
serially using the supplied evaluator.
"""
function adaptive_arcs!(arcs::Vector{Tuple{ComplexF64, ComplexF64}},
        cache::Dict{ComplexF64, Any}, pending, η::Float64;
        check_interval::Integer = 1000, job_channel = nothing,
        result_channel = nothing, certification_log = nothing,
        snapshot = nothing, io = nothing, evaluator = nothing)

    job_channel = _resolve(job_channel, _job_channel)
    result_channel = _resolve(result_channel, _result_channel)
    certification_log = _ensure_certification_log(certification_log)
    snapshot = snapshot === nothing ? _snapshot_path[] : snapshot
    io = _ensure_io(io)

    if job_channel === nothing || result_channel === nothing
        evaluator === nothing &&
            throw(ArgumentError("serial refinement requires an evaluator"))
        return _adaptive_arcs_serial!(arcs, cache, pending, η, certification_log,
            snapshot, io, check_interval, evaluator)
    end

    return _adaptive_arcs_distributed!(arcs, cache, pending, η, job_channel,
        result_channel, certification_log, snapshot, io, check_interval)
end

"""
    bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, N)

Return an upper bound on the ℓ₁ resolvent norm of the original matrix
given the bounds obtained from the Schur form.
"""
function bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, N; Cbound = 1.0)
    l2pseudo_sup = _upper_bound(l2pseudo) / (1 - η)
    norm_Z_sup = max(_upper_bound_offset(norm_Z, 1), 0.0)
    norm_Z_inv_sup = max(_upper_bound_offset(norm_Z_inv, 1), 0.0)
    errF_sup = _upper_bound(errF)
    errT_sup = _upper_bound(errT)

    ϵ = max(max(errF_sup, errT_sup), max(norm_Z_sup, norm_Z_inv_sup))
    @info "The ϵ in the Schur theorems $ϵ"

    bound = setrounding(Float64, RoundUp) do
        numerator = 2 * (1 + ϵ^2) * l2pseudo_sup * Cbound
        denominator = 1 - 2 * ϵ * (1 + ϵ^2) * l2pseudo_sup
        return numerator / denominator
    end
    return bound
end

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
        @warn "Could not load snapshot file $(sorted[1]), possibly corrupted. Trying backup." exception=(e, catch_backtrace())
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

function _identity_ballmatrix(n::Integer)
    return BallMatrix(Matrix{ComplexF64}(I, n, n))
end

function _zero_ballmatrix(n::Integer)
    return BallMatrix(zeros(ComplexF64, n, n))
end

function _as_ball(value)
    value isa Ball && return value
    if value isa Complex
        return Ball(ComplexF64(value), 0.0)
    elseif value isa Real
        return Ball(Float64(value), 0.0)
    else
        throw(ArgumentError("unsupported value type $(typeof(value)) for ball conversion"))
    end
end

function _upper_bound(value)
    ball = _as_ball(value)
    return setrounding(Float64, RoundUp) do
        abs(ball.c) + ball.r
    end
end

function _upper_bound_offset(value, offset::Real)
    ball = _as_ball(value) - _as_ball(offset)
    return setrounding(Float64, RoundUp) do
        abs(ball.c) + ball.r
    end
end

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
    result = _zero_ballmatrix(n)
    identity = _identity_ballmatrix(n)
    for coeff in reverse(coeffs_vec)
        result = result * M
        if !iszero(coeff)
            value = _as_ball(coeff)
            result += value * identity
        end
    end
    return result
end

_polynomial_matrix(coeffs, M::AbstractMatrix) =
    _polynomial_matrix(coeffs, BallMatrix(M))

"""
    compute_schur_and_error(A; polynomial = nothing)

Compute the Schur decomposition of `A` and certified bounds for the
orthogonality defect, the reconstruction error, and the norms of `Z` and
`Z⁻¹`.  When `polynomial` is provided (as coefficients in ascending order),
additional bounds are computed for `p(A)` and `p(T)`.
"""
function compute_schur_and_error(A::BallMatrix; polynomial = nothing)
    S = LinearAlgebra.schur(Complex{Float64}.(A.c))

    bZ = BallMatrix(S.Z)
    errF = svd_bound_L2_opnorm(bZ' * bZ - I)

    bT = BallMatrix(S.T)
    errT = svd_bound_L2_opnorm(bZ * bT * bZ' - A)

    sigma_Z = svdbox(bZ)
    max_sigma = sigma_Z[1]
    min_sigma = sigma_Z[end]

    norm_Z = setrounding(Float64, RoundUp) do
        return abs(max_sigma.c) + max_sigma.r
    end

    min_sigma_lower = setrounding(Float64, RoundDown) do
        return max(min_sigma.c - min_sigma.r, 0.0)
    end
    min_sigma_lower <= 0 && throw(ArgumentError("Schur factor has non-positive smallest singular value bound"))
    norm_Z_inv = setrounding(Float64, RoundUp) do
        return 1 / min_sigma_lower
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
    run_certification(A, circle; polynomial = nothing, kwargs...)

Run the adaptive certification routine on `circle` using a serial evaluator.

# Arguments
- `A`: matrix to certify.  Converted to `BallMatrix` when required.
- `circle`: [`CertificationCircle`](@ref) describing the contour used for the
  adaptive refinement.

# Keyword Arguments
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

The return value is a named tuple containing the computed Schur form, the
accumulated certification log, and the resolvent bounds for both the Schur
factor and the original matrix.
"""
function run_certification(A::BallMatrix, circle::CertificationCircle;
        polynomial = nothing, η::Real = 0.5, check_interval::Integer = 100,
        log_io::IO = stdout, Cbound = 1.0)

    check_interval < 1 && throw(ArgumentError("check_interval must be positive"))
    η = Float64(η)
    (η <= 0 || η >= 1) && throw(ArgumentError("η must belong to (0, 1)"))

    coeffs = polynomial === nothing ? nothing : collect(polynomial)
    schur_data = coeffs === nothing ?
        compute_schur_and_error(A) :
        compute_schur_and_error(A; polynomial = coeffs)

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

    adaptive_arcs!(arcs, cache, pending, η; check_interval = check_interval,
        certification_log = certification_log, io = log_io,
        evaluator = serial_evaluator)

    isempty(certification_log) && throw(ErrorException("certification produced no samples"))

    min_sigma = minimum(log -> log.lo_val, certification_log)
    l2pseudo = maximum(log -> log.hi_res, certification_log)
    resolvent_bound = bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, size(A, 1); Cbound = Cbound)

    return (; schur = S, schur_matrix, certification_log, minimum_singular_value = min_sigma,
        resolvent_schur = l2pseudo, resolvent_original = resolvent_bound, Cbound,
        errF, errT, norm_Z, norm_Z_inv, circle, polynomial = coeffs,
        snapshot_base = nothing)
end

run_certification(A::AbstractMatrix, circle::CertificationCircle; kwargs...) =
    run_certification(BallMatrix(A), circle; kwargs...)

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

end # module
