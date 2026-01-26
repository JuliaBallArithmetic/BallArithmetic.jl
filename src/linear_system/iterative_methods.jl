"""
    iterative_methods.jl

Iterative interval methods for solving linear systems Ax = b.
Implements Gauss-Seidel and Jacobi interval iteration methods.

# References
- Horáček, J. (2012), "Interval linear and nonlinear systems", PhD thesis
- Neumaier, A. (1990), "Interval Methods for Systems of Equations"
- Alefeld, G., Herzberger, J. (1983), "Introduction to Interval Computations"
"""

using LinearAlgebra

"""
    IterativeResult{T, VT}

Result from iterative interval linear system solver.

# Fields
- `solution::VT`: Enclosure of the solution set
- `converged::Bool`: Whether the iteration converged
- `iterations::Int`: Number of iterations performed
- `final_width::T`: Maximum width of final solution components
- `convergence_rate::T`: Observed convergence rate (ratio of successive widths)
"""
struct IterativeResult{T, VT}
    """Enclosure of the solution set."""
    solution::VT
    """Whether iteration converged."""
    converged::Bool
    """Number of iterations performed."""
    iterations::Int
    """Maximum width of final solution."""
    final_width::T
    """Convergence rate."""
    convergence_rate::T
end

"""
    interval_gauss_seidel(A::BallMatrix{T}, b::BallVector{T};
                          x0::Union{Nothing, BallVector{T}}=nothing,
                          max_iterations::Int=100,
                          tol::T=T(1e-10),
                          use_epsilon_inflation::Bool=true,
                          ϵ::T=T(1e-15),
                          r::T=T(0.001)) where {T}

Solve Ax = b using interval Gauss-Seidel iteration.

# Algorithm
The Gauss-Seidel method updates each component x_i using:

    x_i^(k+1) = (b_i - Σ_{j<i} a_{ij}x_j^(k+1) - Σ_{j>i} a_{ij}x_j^(k)) / a_{ii}

where updated components x_j^(k+1) for j < i are used immediately.

# Arguments
- `A`: Coefficient ball matrix (n×n)
- `b`: Right-hand side ball vector (n)
- `x0`: Initial enclosure (computed if not provided)
- `max_iterations`: Maximum number of iterations
- `tol`: Convergence tolerance (maximum component width)
- `use_epsilon_inflation`: Apply ε-inflation before intersection
- `ϵ`: Absolute inflation factor
- `r`: Relative inflation factor

# Returns
`IterativeResult` with solution enclosure and convergence information.

# Example
```julia
A = BallMatrix([3.0 1.0; 1.0 2.0], fill(1e-10, 2, 2))
b = BallVector([5.0, 4.0], fill(1e-10, 2))

result = interval_gauss_seidel(A, b)

if result.converged
    println("Solution: ", result.solution)
    println("Iterations: ", result.iterations)
end
```

# Notes
- Generally converges faster than Jacobi method
- Converges for strictly diagonally dominant matrices
- Uses most recent updates immediately (sequential updates)
- ε-inflation helps ensure non-empty intersection
"""
function interval_gauss_seidel(A::BallMatrix{T}, b::BallVector{T};
                                x0::Union{Nothing, BallVector{T}}=nothing,
                                max_iterations::Int=100,
                                tol::T=T(1e-10),
                                use_epsilon_inflation::Bool=true,
                                ϵ::T=T(1e-15),
                                r::T=T(0.001)) where {T}
    n = size(A, 1)

    # Check diagonal elements are non-zero
    for i in 1:n
        a_ii = A[i, i]
        if 0.0 ∈ a_ii
            error("Gauss-Seidel: Zero on diagonal at position $i")
        end
    end

    # Initialize with point solution if not provided
    if x0 === nothing
        # Use approximate inverse for initial guess
        A_mid = mid(A)
        b_mid = mid(b)
        x_approx = A_mid \ b_mid
        # Start with wide initial enclosure
        x0 = BallVector(x_approx, abs.(x_approx) .+ T(1.0))
    end

    x = copy(x0)
    prev_width = maximum(rad(x))
    convergence_rate = T(1.0)

    for iter in 1:max_iterations
        x_new = similar(x)

        # Gauss-Seidel update: use x_new[j] for j < i, x[j] for j ≥ i
        for i in 1:n
            # Compute b_i - Σ_{j<i} a_{ij}x_new[j] - Σ_{j>i} a_{ij}x[j]
            rhs = b[i]

            # Use updated values for j < i
            for j in 1:(i-1)
                rhs = rhs - A[i, j] * x_new[j]
            end

            # Use old values for j > i
            for j in (i+1):n
                rhs = rhs - A[i, j] * x[j]
            end

            # Divide by diagonal element
            x_new_i = rhs / A[i, i]

            # Apply ε-inflation if requested
            if use_epsilon_inflation
                x_new_i_inflated = Ball(mid(x_new_i), rad(x_new_i) * (1 + r) + ϵ)
            else
                x_new_i_inflated = x_new_i
            end

            # Intersect with previous value
            x_new[i] = intersect_ball(x_new_i_inflated, x[i])

            # Check if intersection is empty
            if rad(x_new[i]) < 0
                @warn "Gauss-Seidel: Empty intersection at component $i, iteration $iter"
                return IterativeResult(
                    x, false, iter, T(Inf), convergence_rate
                )
            end
        end

        # Check convergence
        current_width = maximum(rad(x_new))

        if current_width < tol
            return IterativeResult(
                x_new, true, iter, current_width, convergence_rate
            )
        end

        # Check if width is increasing (divergence)
        if iter > 1 && current_width > prev_width * 1.1
            @warn "Gauss-Seidel: Width increasing, may be diverging"
            return IterativeResult(
                x_new, false, iter, current_width, convergence_rate
            )
        end

        # Update convergence rate
        if prev_width > 0
            convergence_rate = current_width / prev_width
        end

        x = x_new
        prev_width = current_width
    end

    # Max iterations reached
    final_width = maximum(rad(x))
    return IterativeResult(
        x, false, max_iterations, final_width, convergence_rate
    )
end

"""
    interval_jacobi(A::BallMatrix{T}, b::BallVector{T};
                    x0::Union{Nothing, BallVector{T}}=nothing,
                    max_iterations::Int=100,
                    tol::T=T(1e-10),
                    use_epsilon_inflation::Bool=true,
                    ϵ::T=T(1e-15),
                    r::T=T(0.001)) where {T}

Solve Ax = b using interval Jacobi iteration.

# Algorithm
The Jacobi method updates all components simultaneously:

    x_i^(k+1) = (b_i - Σ_{j≠i} a_{ij}x_j^(k)) / a_{ii}

# Arguments
- `A`: Coefficient ball matrix (n×n)
- `b`: Right-hand side ball vector (n)
- `x0`: Initial enclosure (computed if not provided)
- `max_iterations`: Maximum number of iterations
- `tol`: Convergence tolerance (maximum component width)
- `use_epsilon_inflation`: Apply ε-inflation before intersection
- `ϵ`: Absolute inflation factor
- `r`: Relative inflation factor

# Returns
`IterativeResult` with solution enclosure and convergence information.

# Example
```julia
A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
b = BallVector([7.0, 6.0], fill(1e-10, 2))

result = interval_jacobi(A, b)

if result.converged
    println("Solution: ", result.solution)
    println("Convergence rate: ", result.convergence_rate)
end
```

# Notes
- Easily parallelizable (all updates independent)
- Converges for strictly diagonally dominant matrices
- Generally slower than Gauss-Seidel but more parallelizable
- All components updated simultaneously (parallel updates)
"""
function interval_jacobi(A::BallMatrix{T}, b::BallVector{T};
                         x0::Union{Nothing, BallVector{T}}=nothing,
                         max_iterations::Int=100,
                         tol::T=T(1e-10),
                         use_epsilon_inflation::Bool=true,
                         ϵ::T=T(1e-15),
                         r::T=T(0.001)) where {T}
    n = size(A, 1)

    # Check diagonal elements are non-zero
    for i in 1:n
        a_ii = A[i, i]
        if 0.0 ∈ a_ii
            error("Jacobi: Zero on diagonal at position $i")
        end
    end

    # Initialize with point solution if not provided
    if x0 === nothing
        # Use approximate inverse for initial guess
        A_mid = mid(A)
        b_mid = mid(b)
        x_approx = A_mid \ b_mid
        # Start with wide initial enclosure
        x0 = BallVector(x_approx, abs.(x_approx) .+ T(1.0))
    end

    x = copy(x0)
    prev_width = maximum(rad(x))
    convergence_rate = T(1.0)

    for iter in 1:max_iterations
        x_new = similar(x)

        # Jacobi update: all components use old values
        for i in 1:n
            # Compute b_i - Σ_{j≠i} a_{ij}x[j]
            rhs = b[i]

            for j in 1:n
                if j != i
                    rhs = rhs - A[i, j] * x[j]
                end
            end

            # Divide by diagonal element
            x_new_i = rhs / A[i, i]

            # Apply ε-inflation if requested
            if use_epsilon_inflation
                x_new_i_inflated = Ball(mid(x_new_i), rad(x_new_i) * (1 + r) + ϵ)
            else
                x_new_i_inflated = x_new_i
            end

            # Intersect with previous value
            x_new[i] = intersect_ball(x_new_i_inflated, x[i])

            # Check if intersection is empty
            if rad(x_new[i]) < 0
                @warn "Jacobi: Empty intersection at component $i, iteration $iter"
                return IterativeResult(
                    x, false, iter, T(Inf), convergence_rate
                )
            end
        end

        # Check convergence
        current_width = maximum(rad(x_new))

        if current_width < tol
            return IterativeResult(
                x_new, true, iter, current_width, convergence_rate
            )
        end

        # Check if width is increasing (divergence)
        if iter > 1 && current_width > prev_width * 1.1
            @warn "Jacobi: Width increasing, may be diverging"
            return IterativeResult(
                x_new, false, iter, current_width, convergence_rate
            )
        end

        # Update convergence rate
        if prev_width > 0
            convergence_rate = current_width / prev_width
        end

        x = x_new
        prev_width = current_width
    end

    # Max iterations reached
    final_width = maximum(rad(x))
    return IterativeResult(
        x, false, max_iterations, final_width, convergence_rate
    )
end

# Export functions
export IterativeResult
export interval_gauss_seidel, interval_jacobi
