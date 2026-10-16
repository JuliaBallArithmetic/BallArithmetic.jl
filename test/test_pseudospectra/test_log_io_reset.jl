# `_log_io` must not hold a stream left over from precompilation.
#
# `const _log_io = Ref{IO}(stdout)` is evaluated when the module is precompiled, so the stream it
# captures belongs to the precompilation process and is dead in every session that loads the image.
# `_ensure_io(nothing)` returned it, and the first `flush(io)` after `check_interval` arcs then
# raised `ArgumentError: stream not initialized` -- killing any long refinement that had not passed
# `io` explicitly, on something unrelated to its mathematics.

using Test
using BallArithmetic
const CS = BallArithmetic.CertifScripts

eval_half(z) = (; z = ComplexF64(z), val = Ball(abs(ComplexF64(z)) - 0.5, 0.0),
    second_val = Ball(0.0, 0.0))

@testset "_log_io is this session's stdout" begin
    # __init__ calls _reset_log_io!, so after loading the module the stored stream is live
    @test CS._log_io[] === stdout
    @test CS._io_usable(CS._log_io[])
    @test CS._ensure_io(nothing) === stdout
    @test CS._ensure_io(devnull) === devnull        # an explicit io is always honoured
end

@testset "_io_usable does not throw on an uninitialised stream" begin
    # `isopen` RAISES rather than returning false for a stream in StatusUninit, which is the state
    # a precompiled `stdout` is in; the guard has to swallow that.
    dead = Base.PipeEndpoint()
    @test_throws ArgumentError isopen(dead)
    @test CS._io_usable(dead) == false
end

@testset "a stale _log_io does not kill a refinement" begin
    saved = CS._log_io[]
    try
        CS._log_io[] = Base.PipeEndpoint()          # exactly the precompilation leftover
        @test CS._ensure_io(nothing) === stdout     # falls back rather than handing back the corpse
        # a run long enough to cross check_interval, which is where the flush happens, and with no
        # explicit io so that the fallback is what is exercised
        circle = CS.CertificationCircle(0.0 + 0.0im, 1.0; samples = 8)
        arcs = collect(CS._initial_arcs(circle))
        cache = Dict{ComplexF64, Any}()
        res = CS.adaptive_arcs!(arcs, cache, Dict{Int, Any}(), 1.0; evaluator = eval_half,
            check_interval = 4)
        @test res.ok
        @test res.processed == 24
        @test res.reason === :complete
    finally
        CS._log_io[] = saved
    end
    @test CS._log_io[] === stdout
end

@testset "_reset_log_io! restores a usable stream" begin
    saved = CS._log_io[]
    try
        CS._log_io[] = Base.PipeEndpoint()
        @test !CS._io_usable(CS._log_io[])
        CS._reset_log_io!()
        @test CS._log_io[] === stdout
        @test CS._io_usable(CS._log_io[])
    finally
        CS._log_io[] = saved
    end
end
