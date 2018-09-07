using FastGaussQuadrature
import Base: size, show, minimum, maximum

struct Grid
    X::AbstractMatrix # Quadrature roots
    W::AbstractMatrix # Quadrature weights
    N::AbstractMatrix # Inverted weights for matrix elements
    nel::Integer # Number of finite elements
    n::Integer # Polynomial order of basis functions
    bl::Symbol # Left boundary condition
    br::Symbol # Right boundary condition
end

"""
    Grid(r, n[, bl[, br]])

Construct FEDVR grid, where the boundaries of the finite elements are
given by `r`. `n` is the order of the basis functions of the DVR. `bl`
and `br` decide the boundary conditions for the left and right edge of
the grid, respectively (:dirichlet0 by default)."""
function Grid(r::AbstractVector, n::Integer,
              bl::Symbol=:dirichlet0,
              br::Symbol=:dirichlet0)
    xₘ,wₘ = gausslobatto(n)
    xₘ = (xₘ .+ 1)/2
    lerp(a,b,t) = (1 .- t)*a + t*b
    nel = length(r)-1
    X = zeros(nel, n)
    W = zeros(nel, n)
    for i = 1:nel
        X[i,:] = lerp(r[i], r[i+1], xₘ)
        W[i,:] = 0.5*(r[i+1]-r[i])*wₘ
    end
    # Precalculate inverse weights for matrix elements
    N = zeros(nel, n)
    for i = 1:nel
        N[i,1] = 1/√(W[i,1] + (i == 1 ? 0 : W[i-1,end]))
        for m = 2:n-1
            N[i,m] = 1/√(W[i,m])
        end
        N[i,end] = 1/√(W[i,end] + (i == nel ? 0 : W[i+1,1]))
    end
    Grid(X,W,N,nel,n,bl,br)
end

"""
    elcount(grid)

Return number of finite elements in grid.
"""
elcount(grid::Grid) = grid.nel

"""
    elems(grid)

Return range of finite element indices in grid.
"""
elems(grid::Grid) = 1:elcount(grid)

"""
    order(grid)

Return polynomial order of basis functions for the DVR.
"""
order(grid::Grid) = grid.n

"""
    basecount(grid)

Return total number of basis functions in grid.
"""
basecount(grid::Grid) = elcount(grid)*order(grid) - (elcount(grid)-1) - (grid.bl == :dirichlet0 ? 1 : 0) - (grid.br == :dirichlet0 ? 1 : 0)

"""
    bases(grid, i)

Return range of basis function indices in element `i`. For an interior
element, this is always `1:n`, where `n` is the order of the `i`th
element. For the first element, in the case of a `:dirichlet0`
boundary condition, the result is `2:n`, otherwise
`1:n`. Equivalently, for the last element, in case of `:dirichlet0`
boundary conditions, the result is `1:n-1`."""
function bases(grid::Grid, i::Integer)
    n = order(grid)
    if i == 1 && grid.bl == :dirichlet0
        2:n
    elseif i == grid.nel && grid.br == :dirichlet0
        1:n-1
    else
        1:n
    end
end

minimum(grid::Grid) = min(grid.X[1],grid.X[end])
maximum(grid::Grid) = max(grid.X[1],grid.X[end])

function boundary_sel(grid::Grid, v::AbstractVector)
    if grid.bl == :dirichlet0 && grid.br == :dirichlet0
        v[2:end-1]
    elseif grid.bl == :dirichlet0
        v[2:end]
    elseif grid.br == :dirichlet0
        v[1:end-1]
    else
        v
    end
end

"""
    locs(grid)

Return locations of Gauss–Lobatto quadrature points.
"""
locs(grid::Grid) = boundary_sel(grid, [grid.X[:,1:end-1]'[:]..., grid.X[end]])

"""
    weights(grid)

Return weights of Gauss–Lobatto quadrature points.
"""
function weights(grid::Grid)
    bridges = grid.W[1:end-1,end] + grid.W[2:end,1]
    wl = grid.bl == :dirichlet0 ? 0 : grid.W[1]
    wr = grid.br == :dirichlet0 ? 0 : grid.W[end]
    boundary_sel(grid, [wl, [grid.W[:,2:end-1] [bridges;wr]]'[:]...])
end

"""
    find_interval(X, x, i, sel)

Find the interval of points in `x` covering the `i`:th row in
`X`. The search is started from the first element in `sel`."""
function find_interval(X, x, i, sel)
    lt(v) = x -> x < v
    a = something(findlast(lt(X[i,1]), x[sel[1]:end]), 0) + sel[1]
    b = max(a,sel[end])
    b = something(findlast(lt(X[i,end]), x[b:end]), 0) + b - 1
    if i == size(X,1) && b < length(x)
        # If last finite element and *not* :dirichlet0 boundary
        # conditions, include finite-element end-point in interval.
        b += 1
    end
    a:b
end

function show(io::IO, grid::Grid)
    write(io, "FEDVR Grid with $(elcount(grid)) elements of order $(order(grid)) [$(grid.bl),$(grid.br)]")
end

@recipe function plot(grid::Grid)
    markershape --> :circle
    label --> "FEDVR grid locations"
    xticks --> 1:order(grid)-1:basecount(grid)
    yticks --> [grid.X[:,1]..., grid.X[end,end]]
    xlabel --> "#"
    ylabel --> "x"
    locs(grid)
end

export elcount, elems, order, basecount, locs, weights
