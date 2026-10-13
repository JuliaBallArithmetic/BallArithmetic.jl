"""
    BallArithmetic

Tools for rigorous linear algebra with floating-point balls. The
package builds on [`Ball`](@ref) numbers—midpoint/radius pairs that
track rounding error—to provide array, matrix, and vector types with
propagated uncertainty. High-level routines such as norm bounds,
eigenvalue certification, and singular value enclosures are exposed via
this module after it has been loaded.

Most functionality is available through exported symbols documented in
the respective source files. Loading the module also enables auxiliary
utilities such as rigorous rounding macros and promotion rules that let
`Ball` values interact seamlessly with base Julia numeric types.
"""
module BallArithmetic

include("numerical_test/multithread.jl")

using LinearAlgebra
using GenericLinearAlgebra
using GenericSchur

# By default OpenBLAS does not synchronise the floating-point control and status register across
# its threads, so a product under `setrounding` may be rounded to nearest in some of them. The
# INTLAB documentation, https://www.tuhh.de/ti3/rump/intlab/Octave/INTLAB_for_GNU_Octave.shtml,
# gives two remedies: OpenBLAS compiled with `CONSISTENT_FPCSR=1`, an option for x86 and x86_64
# only, or one computation thread. The first is used on x86_64 and the second elsewhere.
if Sys.ARCH == :x86_64
    using OpenBLASConsistentFPCSR_jll
else
    @warn "The behaviour of multithreaded OpenBlas on this architecture is unclear,
    we will fallback to single threaded OpenBLAS

    We refer to
    https://www.tuhh.de/ti3/rump/intlab/Octave/INTLAB_for_GNU_Octave.shtml
    "
end

function __init__()
    if Sys.ARCH == :x86_64
        @info "Switching to OpenBLAS with ConsistentFPCSR = 1 flag enabled, guarantees
        correct floating point rounding mode over all threads."
        BLAS.lbt_forward(OpenBLASConsistentFPCSR_jll.libopenblas_path; verbose = true)

        N = BLAS.get_num_threads()
        K = 1024
        if NumericalTest.rounding_test(N, K)
            @info "OpenBLAS is giving correct rounding on a ($K,$K) test matrix on $N threads"
        else
            @warn "OpenBLAS is not rounding correctly on the test matrix on $N threads"
            # `rounding_test(1, K)` leaves the BLAS on one thread
            if NumericalTest.rounding_test(1, K)
                @warn "The number of BLAS threads was set to 1 to ensure rounding mode is consistent"
            else
                BLAS.set_num_threads(N)
                _blas_rounding_failed()
            end
        end
    else
        N = BLAS.get_num_threads()
        if NumericalTest.rounding_test(1, 1024)
            @warn "The number of BLAS threads was set to 1 to ensure rounding mode is consistent"
        else
            BLAS.set_num_threads(N)
            _blas_rounding_failed()
        end
    end

    # Sums, differences and multiples of matrices under a rounding mode are Julia's own loops, not
    # OpenBLAS calls: the build above does not cover them. They are compared here, entry by entry,
    # with the directed operations.
    failed = NumericalTest.check_array_rounding()
    if isempty(failed)
        @info "Julia's matrix sums under a rounding mode agree with the directed operations on the test matrices"
    else
        @error "On this machine Julia's array operations under a rounding mode do NOT round as the mode says, for: " *
               join(failed, "; ") *
               ". The sums of ball matrices, and the sums inside the matrix products of ball " *
               "matrices (MMul3, MMul4, MMul5, oishi_MMul), fall back to entrywise directed " *
               "operations for Float32 and Float64."
    end
end

# The BLAS does not follow the rounding mode even on one thread: the products are bounded without
# it from here on (Isaia, 2026-10-13), see `NumericalTest.BLAS_ROUNDING_OK`.
function _blas_rounding_failed()
    NumericalTest.BLAS_ROUNDING_OK[] = false
    @error "The rounding test of the BLAS failed on 1 thread as well. Products of Float32 and " *
           "Float64 matrices are not evaluated under a rounding mode on this machine: they are " *
           "enclosed by the splitting scheme of Lange and Rump (prodK) and the Dot2 " *
           "accumulation of Ogita, Rump and Oishi, which is slower."
end

using RoundingEmulator, MacroTools, SetRounding

export ±, mid, rad, midtype, radtype

include("rounding/rounding.jl")
include("types/ball.jl")
export Ball, BallF64, BallComplexF64, inf, sup, ball_hull, intersect_ball, in0

include("types/array.jl")
export BallArray
export isvalid_enclosure, check_enclosure

include("types/matrix.jl")
export BallMatrix
include("types/vector.jl")
export BallVector
include("types/triangularize.jl")

export oishi_MMul

include("types/convertpromote.jl")
include("norm_bounds/rigorous_norm.jl")
export upper_bound_norm
include("norm_bounds/rigorous_opnorm_bounds.jl")
include("norm_bounds/rump_oishi_2024.jl")
include("norm_bounds/oishi_2023_schur.jl")
include("norm_bounds/triangular_inverse_bounds.jl")

export upper_bound_L1_opnorm, upper_bound_L2_opnorm, upper_bound_L_inf_opnorm
export rump_oishi_2024_triangular_bound
export Oishi2023Result, oishi_2023_sigma_min_bound, oishi_2023_optimal_block_size
export RumpOishi2024Result, rump_oishi_2024_sigma_min_bound,
       rump_oishi_2024_optimal_block_size
export psi_schur_factor, pi_norm
include("eigenvalues/gev.jl")
include("eigenvalues/upper_bound_spectral.jl")
include("eigenvalues/miyajima/gev_miyajima_procedures.jl")
include("eigenvalues/rump_lange_2023.jl")
export RigorousGeneralizedEigenvaluesResult, RigorousEigenvaluesResult,
       rigorous_generalized_eigenvalues, rigorous_eigenvalues, gevbox, evbox
include("decompositions/svd/rump_2011.jl")
include("decompositions/svd/miyajima_2014.jl")
include("error_free_transformations.jl")
include("eigenvalues/rump_verifyeigall.jl")
include("eigenvalues/miyajima_2014a.jl")
include("decompositions/svd/schur_gershgorin.jl")
include("decompositions/svd/miyajima_2014a_schurnewton.jl")
include("decompositions/svd/svd.jl")
include("decompositions/svd/adaptive_ogita_svd.jl")
export Miyajima2014aSchurNewtonResult, miyajima2014a_schurnewton, block_enclosure
export SchurGershgorinResult, RigorousSVDResult, schur_gershgorin_enclosure, rigorous_svd, svdbox, svd_bounds, svd_lower_bound_sigma_min, overlap_components,
       rigorous_svd_gpu, rigorous_svd_m4,
       OgitaSVDRefinementResult, AdaptiveSVDResult, ogita_svd_refine, adaptive_ogita_svd,
       ogita_iterations_for_precision, ogita_svd_refine_multifloat,
       SVDMethod, MiyajimaAuto, MiyajimaM1, MiyajimaM3, MiyajimaM4, certify_svd,
       clear_svd_cache!, svd_cache_stats, set_svd_cache!
export svd_bigfloat
export ogita_svd_refine_fast, ogita_svd_refine_hybrid
# `gev_invariant_subspaces` was removed on 2026-10-12: the eigenvalues and deflating subspaces of a
# pencil are enclosed by `verifyeigall(A, B)` (Algorithms 1 and 2 of Miyajima 2014).
include("eigenvalues/block_schur.jl")
export RigorousBlockSchurResult, rigorous_block_schur, extract_cluster_block,
       verify_block_schur_properties, estimate_block_separation
include("eigenvalues/verified_gev.jl")
export GEVResult, verify_generalized_eigenpairs, compute_beta_bound
include("eigenvalues/spectral_projection_schur.jl")
export SchurSpectralProjectorResult, compute_spectral_projector_schur,
       compute_spectral_projector_hermitian, project_vector_spectral,
       verify_spectral_projector_properties,
       SpectralCoefficientResult, compute_spectral_coefficient
include("eigenvalues/iterative_schur_refinement.jl")
export SchurRefinementResult, refine_schur_decomposition, rigorous_schur_bigfloat,
       newton_schulz_orthogonalize!
export SymmetricEigenRefinementResult, refine_symmetric_eigen
export refine_schur_double64, refine_schur_hybrid, refine_schur_multifloat
export refine_symmetric_eigen_double64, refine_symmetric_eigen_hybrid,
       refine_symmetric_eigen_multifloat
export ogita_svd_refine_fast, ogita_svd_refine_hybrid, ogita_svd_refine_multifloat
include("eigenvalues/ordschur_ball.jl")
include("eigenvalues/newton_kantorovich_eigenpair.jl")
export NKEigenpairResult, NKEigenpairsResult, certify_eigenpair, certify_eigenpairs

# Verified matrix decompositions (Rump & Ogita 2024)
# Rigorous residual computation helpers (must be before decomposition files)
include("decompositions/rump_ogita_2024.jl")
include("decompositions/verified_lu.jl")
include("decompositions/verified_cholesky.jl")
include("decompositions/verified_qr.jl")
include("decompositions/verified_polar.jl")
include("decompositions/verified_takagi.jl")
include("decompositions/polar_iterations.jl")

include("pseudospectra/rigorous_contour.jl")
include("pseudospectra/block_resolvent_floor.jl")
include("pseudospectra/svd_frame_floor.jl")
include("pseudospectra/pseudospectrum_cells.jl")
include("matrix_classifiers/is_M_matrix.jl")

include("matrix_properties/regularity.jl")
export RegularityResult
export is_regular_sufficient_condition, is_regular_spectral_radius, is_regular_gershgorin,
       is_regular_diagonal_dominance
export is_regular, is_singular_sufficient_condition

include("matrix_properties/determinant.jl")
export DeterminantResult
export det_hadamard, det_gershgorin, det_cramer, det_cofactor
export interval_det, contains_zero

include("linear_system/inflation.jl")
export VerifyLssResult, verifylss
include("linear_system/backward_substitution.jl")
include("linear_system/gaussian_elimination.jl")
export GaussianEliminationResult
export interval_gaussian_elimination, interval_gaussian_elimination_det
export is_regular_gaussian_elimination

include("linear_system/iterative_methods.jl")
export IterativeResult
export interval_initial_enclosure, interval_gauss_seidel, interval_jacobi

include("linear_system/hbr_method.jl")
export HBRResult
export hbr_method, hbr_method_simple

include("linear_system/shaving.jl")
export ShavingResult
export interval_shaving, sherman_morrison_inverse_update

include("linear_system/preconditioning.jl")
export PreconditionerType, MidpointInverse, LUFactorization, LDLTFactorization,
       IdentityPreconditioner
export PreconditionerResult
export compute_preconditioner, apply_preconditioner, is_well_preconditioned

# Krawczyk must come before overdetermined (overdetermined uses krawczyk_linear_system)
include("linear_system/krawczyk_complete.jl")
export KrawczykResult, krawczyk_linear_system

include("linear_system/overdetermined.jl")
export OverdeterminedResult
export select_subsquares, subsquares_method, multi_jacobi_method, interval_least_squares

include("linear_system/verified_linear_system_hmatrix.jl")
export VerifiedLinearSystemResult, verified_linear_solve_hmatrix

include("pseudospectra/sylvester_resolvent_bound.jl")
include("pseudospectra/CertifScripts.jl")
include("polynomials/poly_range.jl")
include("fft/ball_fft.jl")
include("linear_system/triangular_eigenvectors.jl")
include("linear_system/sylvester.jl")

export triangular_eigenvectors
export sylvester_miyajima_enclosure, triangular_sylvester_miyajima_enclosure
export verified_sylvester_enclosure, schur_sylvester_midpoint

# The resolvent of a block triangular matrix through the similarity S(X)
export SylvesterResolventResult, sylvester_resolvent_precompute, solve_sylvester_oracle
export triangular_inverse_inf_norm_bound, triangular_inverse_one_norm_bound,
       triangular_inverse_two_norm_bound
export psi_squared, similarity_condition_number
export NormEstimator, OneInfNorm, FrobeniusNorm
export DInverseEstimator, TriBacksub, NeumannOneInf, NeumannCollatz2
export CouplingEstimator, CouplingNone, CouplingARSolve, CouplingOffDirect
export ResolventBoundConfig, config_v1, config_v2, config_v3, config_v2p5
export ParametricResolventResult, parametric_resolvent_bound, SVDWarmStart
export estimate_2norm, find_optimal_split, compare_all_configs

# The routines below are stated for a matrix with exact entries: an approximate factor or
# iteration, or a verified decomposition of one matrix. Given a ball matrix they refuse, with the
# reason, in place of the `MethodError` or the "This conversion breaks rigour" that their
# arithmetic would meet on the way.
# run once when the package is precompiled, so that the call in `__init__` is compiled
NumericalTest.check_array_rounding()

for f in (:polar_newton, :polar_newton_schulz, :polar_qdwh, :cholesky_qr2,
    :triangular_eigenvectors,
    :verified_lu, :verified_qr, :verified_cholesky, :verified_polar, :verified_takagi,
    :verified_svd)
    @eval $f(A::BallMatrix, args...; kwargs...) = throw(ArgumentError(string($(string(f)),
        " is for a matrix with exact entries, not for a ball matrix; mid(A) is such a matrix, ",
        "and what is then computed is about mid(A) alone")))
end
for f in (:verified_lu, :verified_qr, :verified_cholesky, :verified_polar, :verified_takagi,
    :verified_svd)
    @eval $f(T::Type, A::BallMatrix, args...; kwargs...) = $f(A)
end

end
