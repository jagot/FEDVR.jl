mutable struct Basis
    grid::Grid
    L′::AbstractArray
end

function Basis(r::AbstractVector, n::Integer, args...)
    grid = Grid(r, n, args...)
    L′ = eval_base_ders(grid)
    Basis(grid, L′)
end

"""
    find_interval(X, x, i, sel)

Find the interval of points in `x` covering the `i`:th row in
`X`. The search is started from the first element in `sel`."""
function find_interval(X, x, i, sel)
    le(v) = x -> x < v
    a = findlast(le(X[i,1]), x[sel[1]:end]) + sel[1]
    b = max(a,sel[end])
    b = findlast(le(X[i,end]), x[b:end]) + b - 1 + (i == size(X,1) ? 1 : 0)
    a:b
end

function eval_element!(xⁱ, Nⁱ, x, χ)
    n = length(xⁱ)
    for m in 1:n
        Lₘ = lagrange(xⁱ, m, x)
        χ[:,m] = Nⁱ[m]*Lₘ
        m > 1 && (χ[1,m] = 0)
    end
end

function evaluate!(basis::Basis, x::AbstractVector, χ::AbstractMatrix)
    g = basis.grid
    n = order(g)

    sel = 1:1
    for i in elems(g)
        sel = find_interval(g.X, x, i, sel)
        eval_element!(g.X[i,:], g.N[i,:], x[sel],
                      view(χ, sel, (1:n) + (i-1)*(n-1)))
    end
    if g.bl == :dirichlet0
        χ[:,1] = 0
    end
    if g.br == :dirichlet0
        χ[:,end] = 0
    end
    χ
end

function (basis::Basis)(x::AbstractVector)
    χ = spzeros(length(x),basecount(basis.grid))
    evaluate!(basis, x, χ)
end

function show(io::IO, basis::Basis)
    write(io, "FEDVR Basis($(basis.grid))")
end

@recipe function plot(basis::Basis, n=1001)
    x = linspace(basis.grid.X[1,1],basis.grid.X[end,end],n)
    xticks --> [basis.grid.X[:,1]..., basis.grid.X[end,end]]
    xlabel --> "x"
    legend --> false
    x,basis(x)
end
