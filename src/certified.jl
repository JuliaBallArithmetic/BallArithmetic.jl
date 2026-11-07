# The flag that says that a result is proved, and the names it has carried.
#
# Every result type of this package whose computation can fail to prove what it claims carries
# one boolean field, `certified`, which is `true` only when the statement in that type's
# docstring was established with every rounding accounted for. `iscertified` reads it whatever
# the type, so that code built on the package does not depend on the field name.
#
# The name was not always the same. In version 0.3.0 the same flag was called `success` on some
# types, `verified` on others, `is_regular` on the regularity test and `precomputation_success`
# on the Sylvester precomputation; `certified` replaced all of them in 0.4. The old spellings
# still resolve, through `getproperty`, with a deprecation warning, and `_RENAMED_FLAGS` below is
# the table of which name belonged to which type.
#
# The boolean fields of this package that are deliberately absent from the table are other
# predicates: `converged` of `IterativeResult`, of `SchurRefinementResult` and of the other
# floating-point iterations, which says that an iteration reached a tolerance and proves nothing;
# `spectrum_covered` of `VerifyEigAllResult`, which asserts in addition to certification that the
# discs hold the whole spectrum; `empty` of `IterativeResult` and `ShavingResult`; `success` of
# `PreconditionerResult`, whose preconditioner is an approximate inverse and is never verified;
# `unsolvable` of `OverdeterminedResult`, which proves that the solution set is empty;
# `loop_closure` of `Enclosure`; `tolerance_achieved` of `AdaptiveSVDResult`; `is_right` of
# `VerifiedPolarResult`; `singular` of `GaussianEliminationResult`; `pencil` of
# `Miyajima2014aSchurNewtonResult`; and `used_fast_gamma` of `RumpOishi2024Result`.

"""
    iscertified(result) -> Bool

Whether `result` proves what it computes: `true` when the verification behind it succeeded, so
that the statement in the docstring of `typeof(result)` holds for every matrix and every vector
of the input balls, and `false` when some step of that verification did not apply, in which case
the enclosures carried by `result` prove nothing and usually have infinite radii.

This is the supported way to ask the question. The field it reads is called `certified` in version
0.4; in 0.3.0 it was called `success` on the verified decompositions, on the generalised
eigenvalue results and on most of the linear system solvers, `verified` on
[`KrawczykResult`](@ref), [`NKEigenpairResult`](@ref), [`Oishi2023Result`](@ref),
[`RumpOishi2024Result`](@ref) and [`VerifiedLinearSystemResult`](@ref), `is_regular` on
[`RegularityResult`](@ref) and `precomputation_success` on [`SylvesterResolventResult`](@ref).
Code that calls `iscertified` reads the flag in either version.

A type that carries no such flag raises an `ArgumentError` naming it, in place of returning
`false`: a `false` would read as "not proved", which is a claim about a computation that the type
never made. The flags that mean something else, and so are not read here, are `converged` of the
iterative methods and of the refinements, which is about an iteration reaching a tolerance;
`spectrum_covered` of [`VerifyEigAllResult`](@ref), which asserts in addition that the union of
the discs holds the whole spectrum; `empty` of [`IterativeResult`](@ref) and of
[`ShavingResult`](@ref); and `success` of [`PreconditionerResult`](@ref), whose preconditioner is
an approximate inverse.

# Example

```julia
B = randn(100, 100); A = B' * B + I
iscertified(verified_cholesky(Float64, A))     # true: A is positive definite
iscertified(verified_cholesky(Float64, [1.0 2.0; 2.0 1.0]))     # false: it is not
```
"""
function iscertified(result)
    throw(ArgumentError(string(typeof(result),
        " carries no flag meaning that the result is proved, so `iscertified` has no answer ",
        "for it; `propertynames(result)` lists what it does carry")))
end

# The types whose `certified` field means that the result is proved. One method each, reading the
# field directly, so that the deprecation branch of `getproperty` is not on the path.
const _CERTIFIED_RESULTS = (GEVResult, GaussianEliminationResult, HBRResult, KrawczykResult,
    MiyajimaGEVResult, NKEigenpairResult, Oishi2023Result, OverdeterminedResult,
    ParametricResolventResult, RegularityResult, RumpOishi2024Result, SylvesterResolventResult,
    VerifiedCholeskyResult, VerifiedLUResult, VerifiedLinearSystemResult, VerifiedPolarResult,
    VerifiedQRResult, VerifiedTakagiResult, VerifyLssResult)

for T in _CERTIFIED_RESULTS
    @eval iscertified(result::$T) = getfield(result, :certified)
end

# `VerifyEigAllResult` has a `certified` field with one entry per cluster, so the message above
# would be wrong for it.
function iscertified(result::VerifyEigAllResult)
    throw(ArgumentError(string(typeof(result), " certifies one cluster at a time: ",
        "`result.certified[i]` says whether cluster `i` satisfied the self-mapping test, and ",
        "`result.spectrum_covered` whether every cluster did, which is the stronger claim that ",
        "the discs hold the whole spectrum")))
end

@noinline function _renamed_flag(result, old::Symbol, new::Symbol)
    Base.depwarn(string("`", nameof(typeof(result)), ".", old,
            "` is deprecated: the field is `", new, "` since BallArithmetic 0.4, and ",
            "`iscertified(result)` reads it in either version."), :getproperty)
    return getfield(result, new)
end

# The name each flag had in version 0.3.0, by type. `n_verified` of `NKEigenpairsResult` is a
# count of certified eigenpairs rather than a flag, and is here because the same rename moved it.
#
# Two types that carry the flag have no entry. `VerifyLssResult` was `EpsilonInflationResult` in
# 0.3.0, with the field already called `certified`. `OverdeterminedResult` had `solvable` there,
# which said that the system was proved solvable; the current `certified` says that a finite
# enclosure was obtained, and `unsolvable` that the solution set is proved empty, so neither
# forwards from `solvable`.
const _RENAMED_FLAGS = ((GEVResult, (:success => :certified,)),
    (GaussianEliminationResult, (:success => :certified,)),
    (HBRResult, (:success => :certified,)),
    (KrawczykResult, (:verified => :certified,)),
    (MiyajimaGEVResult, (:success => :certified,)),
    (NKEigenpairResult, (:verified => :certified,)),
    (NKEigenpairsResult, (:n_verified => :n_certified,)),
    (Oishi2023Result, (:verified => :certified,)),
    (ParametricResolventResult, (:success => :certified,)),
    (RegularityResult, (:is_regular => :certified,)),
    (RumpOishi2024Result, (:verified => :certified,)),
    (SylvesterResolventResult, (:precomputation_success => :certified,)),
    (VerifiedCholeskyResult, (:success => :certified,)),
    (VerifiedLUResult, (:success => :certified,)),
    (VerifiedLinearSystemResult, (:verified => :certified,)),
    (VerifiedPolarResult, (:success => :certified,)),
    (VerifiedQRResult, (:success => :certified,)),
    (VerifiedTakagiResult, (:success => :certified,)))

for (T, renames) in _RENAMED_FLAGS
    old_names = Tuple(QuoteNode(first(p)) for p in renames)
    checks = [:(name === $(QuoteNode(first(p))) &&
                return _renamed_flag(result, $(QuoteNode(first(p))), $(QuoteNode(last(p)))))
              for p in renames]
    @eval begin
        # `name` is a literal at nearly every call site, so the comparisons fold away and
        # ordinary field access compiles to the `getfield` alone.
        @inline function Base.getproperty(result::$T, name::Symbol)
            $(checks...)
            return getfield(result, name)
        end
        # The deprecated spellings are properties that still resolve, so they belong in the
        # listing asked for the private ones, and nowhere else.
        Base.propertynames(result::$T, private::Bool) = private ?
                                                        (fieldnames($T)..., $(old_names...)) :
                                                        fieldnames($T)
    end
end
