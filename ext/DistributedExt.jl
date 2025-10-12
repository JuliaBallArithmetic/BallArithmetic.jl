module DistributedExt

using Distributed
using Distributed: RemoteException
using Base: dirname

using BallArithmetic
import BallArithmetic.CertifScripts

const _RemoteJob = Tuple{Int, ComplexF64}

function _load_certification_dependencies(pids)
    Distributed.@sync begin
        for pid in pids
            Distributed.@async Distributed.remotecall_eval(Main, pid, :(begin
                using LinearAlgebra
                using BallArithmetic
                using BallArithmetic.CertifScripts
            end))
        end
    end
end

function _set_schur_on_workers(pids, matrix)
    Distributed.@sync begin
        for pid in pids
            Distributed.@async Distributed.remotecall_wait(CertifScripts.set_schur_matrix!, pid, matrix)
        end
    end
end

function _cleanup_snapshots(basepath)
    for suffix in ("_A.jld2", "_B.jld2")
        filename = basepath * suffix
        isfile(filename) && rm(filename; force = true)
    end
end

function _run_certification_distributed(A::BallArithmetic.BallMatrix, circle::CertifScripts.CertificationCircle,
        worker_ids::Vector{Int}; polynomial = nothing, η::Real = 0.5, check_interval::Integer = 100,
        snapshot_path::Union{Nothing, AbstractString} = nothing, log_io::IO = stdout,
        channel_capacity::Integer = 1024, Cbound = 1.0, cleanup_workers::Bool)

    isempty(worker_ids) && throw(ArgumentError("no worker processes available for certification"))
    channel_capacity < 1 && throw(ArgumentError("channel_capacity must be positive"))
    check_interval < 1 && throw(ArgumentError("check_interval must be positive"))

    η = Float64(η)
    (η <= 0 || η >= 1) && throw(ArgumentError("η must belong to (0, 1)"))

    coeffs = polynomial === nothing ? nothing : collect(polynomial)
    schur_data = coeffs === nothing ?
        CertifScripts.compute_schur_and_error(A) :
        CertifScripts.compute_schur_and_error(A; polynomial = coeffs)

    S, errF, errT, norm_Z, norm_Z_inv = schur_data
    bT = BallArithmetic.BallMatrix(S.T)
    schur_matrix = coeffs === nothing ? bT : CertifScripts._polynomial_matrix(coeffs, bT)

    snapshot_base = snapshot_path === nothing ? tempname() : String(snapshot_path)
    mkpath(dirname(snapshot_base))
    cleanup_snapshot = snapshot_path === nothing

    certification_log = Any[]
    job_channel = nothing
    result_channel = nothing
    worker_tasks = Future[]
    cache = nothing
    pending = nothing

    try
        _load_certification_dependencies(worker_ids)
        _set_schur_on_workers(worker_ids, schur_matrix)

        job_channel = RemoteChannel(() -> Channel{_RemoteJob}(channel_capacity))
        result_channel = RemoteChannel(() -> Channel{NamedTuple}(channel_capacity))

        CertifScripts.configure_certification!(; job_channel = job_channel, result_channel = result_channel,
            certification_log = certification_log, snapshot = snapshot_base, io = log_io)

        worker_tasks = Future[]
        for pid in worker_ids
            push!(worker_tasks, Distributed.@spawnat pid CertifScripts.dowork(job_channel, result_channel))
        end

        arcs = CertifScripts._initial_arcs(circle)
        cache = Dict{ComplexF64, Any}()
        pending = Dict{Int, Tuple{ComplexF64, ComplexF64}}()

        CertifScripts.adaptive_arcs!(arcs, cache, pending, η; check_interval = check_interval,
            job_channel = job_channel, result_channel = result_channel,
            certification_log = certification_log, snapshot = snapshot_base, io = log_io)

        isempty(certification_log) && throw(ErrorException("certification produced no samples"))

        min_sigma = minimum(log -> log.lo_val, certification_log)
        l2pseudo = maximum(log -> log.hi_res, certification_log)
        resolvent_bound = CertifScripts.bound_res_original(l2pseudo, η, norm_Z, norm_Z_inv, errF, errT, size(A, 1); Cbound = Cbound)

        return (; schur = S, schur_matrix, certification_log, minimum_singular_value = min_sigma,
            resolvent_schur = l2pseudo, resolvent_original = resolvent_bound, Cbound,
            errF, errT, norm_Z, norm_Z_inv, circle, polynomial = coeffs,
            snapshot_base)
    finally
        if result_channel !== nothing && pending isa Dict
            wait_start = time()
            try
                while !isempty(pending)
                    if isready(result_channel)
                        result = try
                            take!(result_channel)
                        catch err
                            if err isa InvalidStateException
                                break
                            end
                            rethrow(err)
                        end
                        if cache isa Dict
                            CertifScripts._push_result!(cache, certification_log, result)
                        end
                        haskey(pending, result.i) && delete!(pending, result.i)
                        wait_start = time()
                    elseif !isempty(worker_tasks) && any(!isready, worker_tasks)
                        sleep(0.1)
                        if time() - wait_start > 30
                            @warn "timeout while draining pending certification results" pending=length(pending)
                            break
                        end
                    else
                        break
                    end
                end
            catch err
                if !(err isa InvalidStateException)
                    @warn "failed to drain pending certification results" exception=(err, catch_backtrace())
                end
            end
        end

        if job_channel !== nothing
            try
                close(job_channel)
            catch err
                if !(err isa InvalidStateException)
                    @warn "failed to close certification job channel" exception=(err, catch_backtrace())
                end
            end
        end

        for task in worker_tasks
            try
                fetch(task)
            catch err
                inner = err
                while true
                    if inner isa InvalidStateException
                        break
                    elseif inner isa RemoteException
                        inner = inner.captured.ex
                        continue
                    elseif inner isa TaskFailedException
                        inner = inner.task.exception
                        continue
                    elseif inner isa CompositeException && !isempty(inner.exceptions)
                        inner = first(inner.exceptions)
                        continue
                    end
                    inner = nothing
                    break
                end
                if inner isa InvalidStateException
                    continue
                end
                @warn "certification worker terminated with an error" exception=(err, catch_backtrace())
            end
        end

        if result_channel !== nothing
            try
                close(result_channel)
            catch err
                if !(err isa InvalidStateException)
                    @warn "failed to close certification result channel" exception=(err, catch_backtrace())
                end
            end
        end

        if cleanup_workers && !isempty(worker_ids)
            try
                rmprocs(worker_ids)
            catch err
                @warn "failed to remove certification workers" exception=(err, catch_backtrace())
            end
        end

        CertifScripts._job_channel[] = nothing
        CertifScripts._result_channel[] = nothing
        CertifScripts._certification_log[] = nothing
        CertifScripts._snapshot_path[] = nothing
        CertifScripts._log_io[] = stdout

        if cleanup_snapshot && (isfile(snapshot_base * "_A.jld2") || isfile(snapshot_base * "_B.jld2"))
            _cleanup_snapshots(snapshot_base)
        end
    end
end

"""
    CertifScripts.run_certification(A, circle, workers; kwargs...)

Distributed variant of [`CertifScripts.run_certification`](@ref) that evaluates
certification samples in parallel.

# Arguments
- `A`: matrix to certify.  Automatically converted to `BallMatrix` when
  necessary.
- `circle`: [`CertifScripts.CertificationCircle`](@ref) describing the contour
  used by the adaptive refinement.
- `workers`: one of
    * an integer specifying how many new worker processes should be created for
      the call,
    * a `Distributed.WorkerPool` whose members will be reused, or
    * an explicit collection of worker IDs.

# Keyword Arguments
- `polynomial = nothing`: coefficients (ascending order) defining an optional
  polynomial `p`.  When present the certification is performed on `p(T)` and the
  reconstruction error of `p(A)` is reported.
- `η = 0.5`: threshold steering the adaptive refinement.  Must belong to the
  open unit interval.
- `check_interval = 100`: number of processed arcs between progress reports and
  snapshot attempts.
- `snapshot_path = nothing`: base filename for alternating snapshot files.  A
  temporary path is used and cleaned up automatically when omitted.
- `log_io = stdout`: destination `IO` for log output.
- `channel_capacity = 1024`: capacity of the job and result channels shared
  with worker processes.
- `Cbound = 1.0`: constant used by [`CertifScripts.bound_res_original`](@ref)
  when translating Schur resolvent bounds to the original matrix.

The return value matches the serial flavour, exposing the Schur data,
certification log, and resolvent bounds in a named tuple.  When new workers are
spawned they are torn down automatically after the run.
"""
function CertifScripts.run_certification(A::BallArithmetic.BallMatrix, circle::CertifScripts.CertificationCircle,
        num_workers::Integer; kwargs...)
    num_workers < 1 && throw(ArgumentError("num_workers must be positive"))
    worker_ids = addprocs(num_workers)
    return _run_certification_distributed(A, circle, worker_ids; kwargs..., cleanup_workers = true)
end

function CertifScripts.run_certification(A::BallArithmetic.BallMatrix, circle::CertifScripts.CertificationCircle,
        pool::Distributed.WorkerPool; kwargs...)
    worker_ids = Distributed.workers(pool)
    return _run_certification_distributed(A, circle, collect(worker_ids); kwargs..., cleanup_workers = false)
end

function CertifScripts.run_certification(A::BallArithmetic.BallMatrix, circle::CertifScripts.CertificationCircle,
        worker_ids::AbstractVector{<:Integer}; kwargs...)
    return _run_certification_distributed(A, circle, collect(worker_ids); kwargs..., cleanup_workers = false)
end

function CertifScripts.run_certification(A::AbstractMatrix, circle::CertifScripts.CertificationCircle,
        num_workers::Integer; kwargs...)
    return CertifScripts.run_certification(BallArithmetic.BallMatrix(A), circle, num_workers; kwargs...)
end

function CertifScripts.run_certification(A::AbstractMatrix, circle::CertifScripts.CertificationCircle,
        pool::Distributed.WorkerPool; kwargs...)
    return CertifScripts.run_certification(BallArithmetic.BallMatrix(A), circle, pool; kwargs...)
end

function CertifScripts.run_certification(A::AbstractMatrix, circle::CertifScripts.CertificationCircle,
        worker_ids::AbstractVector{<:Integer}; kwargs...)
    return CertifScripts.run_certification(BallArithmetic.BallMatrix(A), circle, worker_ids; kwargs...)
end

end
