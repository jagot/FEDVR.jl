"""
    project!(fun, grid, i, cⁱ)

Project the function `fun` onto the finite element of `grid` with index
`i`, writing the results in the vector of expansion coefficients `cⁱ`.
"""
function project!(fun, grid::Grid, i::Integer,
                  cⁱ::AbstractVector)
    n = order(grid)
    cⁱ[1] = grid.N[i,1]*grid.W[i,1]*fun(grid.X[i,1])
    for m = 2:n-1
        cⁱ[m] = √(grid.W[i,m])*fun(grid.X[i,m])
    end
    cⁱ[end] = grid.N[i,end]*grid.W[i,end]*fun(grid.X[i,end])
end

"""
    project(fun, basis)

Project the function `fun` onto the FEDVR `basis`.
"""
function project(fun, basis::Basis)
    grid = basis.grid
    C = zeros(grid.X)
    for i = elems(grid)
        project!(fun, grid, i, view(C, i, :))
    end
    # The bridge functions overlap the function both in element i and
    # element i+1; thus the contributions need to be added into one
    # expansion coefficient.
    C[2:elcount(grid),1] += C[1:elcount(grid)-1,end]
    
    [C[:,1:end-1]'[:]..., C[end]]
end

export project
