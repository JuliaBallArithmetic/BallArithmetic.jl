# Tests of `iscertified`, the one accessor for the flag that says a result is proved, and of the
# spellings that flag had in version 0.3.0, which still resolve through `getproperty`.

using Test
using BallArithmetic
using LinearAlgebra
using Random

@testset "iscertified" begin
    rng = MersenneTwister(20261107)

    # one result of each type that carries the flag
    P = [4.0 1.0; 1.0 3.0]                      # symmetric positive definite
    A = [3.0 1.0; 1.0 4.0]
    bA = BallVector([1.0, 2.0])
    BA = BallMatrix(A)
    n = 6
    M = randn(rng, n, n) + n * I
    E = eigen(M)
    S = Symmetric(randn(rng, 4, 4))
    G = Symmetric(randn(rng, 4, 4))
    Gpd = Matrix(G * G + 4I)
    Gpd = (Gpd + Gpd') / 2
    Fs = eigen(Symmetric(Matrix(S)), Symmetric(Gpd))
    Gm = randn(rng, 8, 8) + 8I
    Tu = Matrix(UpperTriangular(0.3 * randn(rng, ComplexF64, 8, 8))) + Diagonal(ComplexF64.(1:8))
    Qc = Matrix(qr(randn(rng, ComplexF64, 3, 3)).Q)
    Asym = Qc * Diagonal([3.0, 2.0, 1.0]) * transpose(Qc)
    Ao = BallMatrix(randn(rng, 4, 2) + [1.0 0.0; 0.0 1.0; 0.5 0.5; 0.1 0.2])
    bo = BallVector(randn(rng, 4))
    syl = sylvester_resolvent_precompute(Tu, 3)

    results = (verified_cholesky(P), verified_lu(A), verified_qr(A), verified_polar(A),
        verified_takagi(Asym),
        krawczyk_linear_system(BA, bA), verifylss(BA, bA),
        interval_gaussian_elimination(BA, bA), hbr_method(BA, bA),
        verified_linear_solve_hmatrix(BA, bA), subsquares_method(Ao, bo),
        is_regular_sufficient_condition(BA),
        BallArithmetic.certify_eigenpair(M, E.values[1], E.vectors[:, 1]),
        verify_generalized_eigenpairs(Symmetric(Matrix(S)), Symmetric(Gpd), Fs.vectors,
            Fs.values),
        miyajima_gev_enclosure(BallMatrix(complex(Matrix(S))), BallMatrix(complex(Gpd)),
            Matrix(complex(Fs.vectors)), complex(collect(Fs.values))),
        oishi_2023_sigma_min_bound(BallMatrix(Gm), 3),
        rump_oishi_2024_sigma_min_bound(BallMatrix(Gm), 3),
        syl, parametric_resolvent_bound(syl, Tu, 0.5 + 0.0im))

    @testset "the accessor reads the flag: $(nameof(typeof(r)))" for r in results
        @test iscertified(r) isa Bool
        @test iscertified(r) === getfield(r, :certified)
        @test @inferred(iscertified(r)) === r.certified
    end

    @testset "every type of the table is exercised" begin
        @test :iscertified in names(BallArithmetic)
        # the instances above, one per type that carries the flag
        @test Set(nameof(typeof(r)) for r in results) ==
              Set(nameof(T) for T in BallArithmetic._CERTIFIED_RESULTS)
        for T in BallArithmetic._CERTIFIED_RESULTS
            @test :certified in fieldnames(T)
            @test hasmethod(iscertified, Tuple{T})
        end
        # the renamed spellings are gone from the types, and the new ones are there
        for (T, renames) in BallArithmetic._RENAMED_FLAGS, (old, new) in renames
            @test new in fieldnames(T)
            @test !(old in fieldnames(T))
        end
    end

    # the name of the flag in version 0.3.0, by type, and a result of that type
    by_type = Dict(nameof(typeof(r)) => r for r in results)
    by_type[:NKEigenpairsResult] = certify_eigenpairs(BallMatrix(Matrix(Symmetric(A))))
    renamed = [(by_type[nameof(T)], old, new)
               for (T, renames) in BallArithmetic._RENAMED_FLAGS for (old, new) in renames]

    @testset "the 0.3.0 spelling resolves: $(nameof(typeof(r))).$old" for (r, old, new) in renamed
        @test getproperty(r, old) === getfield(r, new)
        # `Base.depwarn` prints under `--depwarn=yes`, which `Pkg.test` sets
        if Base.JLOptions().depwarn == 1
            @test_deprecated getproperty(r, old)
        else
            @test getproperty(r, old) === getfield(r, new)
        end
        @test !(old in propertynames(r))
        @test old in propertynames(r, true)
        @test new in propertynames(r)
        @test propertynames(r) == fieldnames(typeof(r))
    end

    @testset "a type with no such flag is an error, not a false" begin
        for x in (1.0, BallMatrix(A), interval_gauss_seidel(BA, bA),
            compute_preconditioner(BA), certify_eigenpairs(BallMatrix(Matrix(Symmetric(A)))),
            verifyeigall(BallMatrix(M)))
            @test_throws ArgumentError iscertified(x)
            err = try
                iscertified(x)
            catch e
                e
            end
            @test occursin(string(typeof(x)), err.msg)
        end
        # the message for `verifyeigall` names the per-cluster flag and the covering claim
        err = try
            iscertified(verifyeigall(BallMatrix(M)))
        catch e
            e
        end
        @test occursin("certified[i]", err.msg) && occursin("spectrum_covered", err.msg)
    end

    @testset "ordinary field access is the getfield alone" begin
        r = verified_cholesky(P)
        f(x) = x.certified
        g(x) = x.residual_norm
        for h in (f, g)
            ir = string(code_typed(h, (typeof(r),))[1].first)
            @test occursin("getfield", ir)
            @test !occursin("_renamed_flag", ir)
        end
        f(r)        # compiled before the allocations are counted
        @test (@allocated f(r)) == 0
    end
end
