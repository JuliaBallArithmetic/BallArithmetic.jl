# `_log_io` must not hold a stream left over from precompilation.
#
# `const _log_io = Ref{IO}(stdout)` is evaluated when the module is precompiled, so the stream it
# captured belonged to the precompilation process and was dead in every session that loaded the
# image. `_ensure_io(nothing)` returned it, and the first `flush(io)` after `check_interval` sides
# then raised `ArgumentError: stream not initialized`, killing any long refinement that had not
# passed `io` explicitly, on something unrelated to its mathematics. `main` holds `devnull` in the
# ref at precompilation and resolves it to this session's `stdout` in `__init__`.
#
# Ported from the `log-io-reset` branch, which fixed the same bug with an `_io_usable` guard and a
# `_reset_log_io!` resetter; neither exists in `main`, so the assertions naming them are gone and
# what remains is the behaviour both fixes were for.

using Test
using BallArithmetic
const CS = BallArithmetic.CertifScripts

eval_half(z) = (; z = ComplexF64(z), val = Ball(abs(ComplexF64(z)) - 0.5, 0.0),
    second_val = Ball(0.0, 0.0))

@testset "_log_io is this session's stdout" begin
    # __init__ resolves the ref, so after loading the module the stored stream is live
    @test CS._log_io[] === stdout
    flush(CS._log_io[])                             # raised on the precompiled handle
    @test CS._ensure_io(nothing) === stdout
    @test CS._ensure_io(devnull) === devnull         # an explicit io is always honoured
end

@testset "the default io does not kill a refinement" begin
    # a run long enough to cross check_interval, which is where the flush happens, and with no
    # explicit io so that the stored stream is what is exercised
    circle = CS.CertificationCircle(0.0 + 0.0im, 1.0; samples = 8)
    arcs = CS._initial_arcs(circle)
    cache = Dict{ComplexF64, Any}()
    res = CS.adaptive_arcs!(arcs, cache, Dict{Int, Any}(), 1.0; evaluator = eval_half,
        check_interval = 4)
    @test res.ok
    @test res.processed == 24
    @test res.reason === :complete
end

@testset "configure_certification! sets the stream the refinement flushes" begin
    saved = CS._log_io[]
    try
        CS.configure_certification!(io = devnull)
        @test CS._log_io[] === devnull
        @test CS._ensure_io(nothing) === devnull
    finally
        CS._log_io[] = saved
    end
    @test CS._log_io[] === stdout
end
