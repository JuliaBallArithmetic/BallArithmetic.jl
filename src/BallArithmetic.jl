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
            @warn "OpenBLAS is not rounding correctly on the test matrix"
            @warn "The number of BLAS threads was set to 1 to ensure rounding mode is consistent"
            if !NumericalTest.rounding_test(1, K)
                @warn "The rounding test failed on 1 thread"
            end
        end
    else
        BLAS.set_num_threads(1)
        @warn "The number of BLAS threads was set to 1 to ensure rounding mode is consistent"
        if !NumericalTest.rounding_test(1, 1024)
            @warn "The rounding test failed on 1 thread"
        end
    end
end

using RoundingEmulator, MacroTools, SetRounding

export ±, mid, rad, midtype, radtype

include("rounding/rounding.jl")
include("types/ball.jl")
export Ball, BallF64, BallComplexF64, inf, sup, ball_hull, intersect_ball

include("types/array.jl")
export BallArray

include("types/matrix.jl")
export BallMatrix
include("types/vector.jl")
export BallVector
include("types/triangularize.jl")

include("types/convertpromote.jl")
include("norm_bounds/rigorous_norm.jl")
export upper_bound_norm
include("norm_bounds/rigorous_opnorm_bounds.jl")
include("norm_bounds/oishi.jl")
include("norm_bounds/oishi_triangular.jl")
include("norm_bounds/rump_oishi_2024.jl")

export upper_bound_L1_opnorm, upper_bound_L2_opnorm, upper_bound_L_inf_opnorm
export rump_oishi_2024_triangular_bound, backward_singular_value_bound
include("eigenvalues/gev.jl")
include("eigenvalues/upper_bound_spectral.jl")
include("eigenvalues/miyajima/proceduresMiyajima2010.jl")
include("eigenvalues/miyajima/gev_miyajima_procedures.jl")
include("eigenvalues/rump_2022a.jl")
include("eigenvalues/rump_lange_2023.jl")
export RigorousGeneralizedEigenvaluesResult, RigorousEigenvaluesResult,
    rigorous_generalized_eigenvalues, rigorous_eigenvalues, gevbox, evbox
export Rump2022aResult, rump_2022a_eigenvalue_bounds
export RumpLange2023Result, rump_lange_2023_cluster_bounds, refine_cluster_bounds
include("svd/singular_gerschgorin.jl")
include("svd/miyajima_vbd.jl")
include("svd/svd.jl")
include("svd/adaptive_ogita_svd.jl")
export MiyajimaVBDResult, RigorousSVDResult, miyajima_vbd, rigorous_svd, svdbox,
    rigorous_svd_m4, refine_svd_bounds_with_vbd,
    OgitaSVDRefinementResult, AdaptiveSVDResult, ogita_svd_refine, adaptive_ogita_svd,
    SVDMethod, MiyajimaM1, MiyajimaM4, RumpOriginal
include("eigenvalues/spectral_projectors.jl")
include("eigenvalues/block_schur.jl")
export RigorousSpectralProjectorsResult, miyajima_spectral_projectors,
    compute_invariant_subspace_basis, verify_projector_properties,
    projector_condition_number
export RigorousBlockSchurResult, rigorous_block_schur, extract_cluster_block,
    verify_block_schur_properties, estimate_block_separation,
    refine_off_diagonal_block, compute_block_sylvester_rhs
include("eigenvalues/verified_gev.jl")
export GEVResult, verify_generalized_eigenpairs, compute_beta_bound
include("eigenvalues/riesz_projections.jl")
export project_onto_eigenspace, project_onto_schur_subspace,
    verified_project_onto_eigenspace, compute_eigenspace_projector,
    compute_schur_projector
include("eigenvalues/spectral_projection_schur.jl")
export SchurSpectralProjectorResult, compute_spectral_projector_schur,
    compute_spectral_projector_hermitian, project_vector_spectral,
    verify_spectral_projector_properties
include("pseudospectra/rigorous_contour.jl")
include("matrix_classifiers/is_M_matrix.jl")

include("matrix_properties/regularity.jl")
export RegularityResult
export is_regular_sufficient_condition, is_regular_gershgorin, is_regular_diagonal_dominance
export is_regular, is_singular_sufficient_condition

include("matrix_properties/determinant.jl")
export DeterminantResult
export det_hadamard, det_gershgorin, det_cramer
export interval_det, contains_zero

include("linear_system/inflation.jl")
include("linear_system/backward_substitution.jl")
include("linear_system/gaussian_elimination.jl")
export GaussianEliminationResult
export interval_gaussian_elimination, interval_gaussian_elimination_det
export is_regular_gaussian_elimination

include("linear_system/iterative_methods.jl")
export IterativeResult
export interval_gauss_seidel, interval_jacobi

include("linear_system/hbr_method.jl")
export HBRResult
export hbr_method, hbr_method_simple

include("linear_system/shaving.jl")
export ShavingResult
export interval_shaving, sherman_morrison_inverse_update

include("linear_system/preconditioning.jl")
export PreconditionerType, MidpointInverse, LUFactorization, LDLTFactorization, IdentityPreconditioner
export PreconditionerResult
export compute_preconditioner, apply_preconditioner, is_well_preconditioned

include("linear_system/overdetermined.jl")
export OverdeterminedResult
export subsquares_method, multi_jacobi_method, interval_least_squares

include("linear_system/verified_linear_system_hmatrix.jl")
export VerifiedLinearSystemResult, verified_linear_solve_hmatrix

include("pseudospectra/CertifScripts.jl")
include("linear_system/sylvester.jl")

export sylvester_miyajima_enclosure, triangular_sylvester_miyajima_enclosure

end
