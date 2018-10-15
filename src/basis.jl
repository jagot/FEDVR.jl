using LinearAlgebra

struct Basis
    grid::Grid
    L′::AbstractArray
end

function Basis(r::AbstractVector, n::Integer, args...)
    grid = Grid(r, n, args...)
    L′ = lagrangeder(grid)
    Basis(grid, L′)
end

function eval_element!(xⁱ, Nⁱ, x, χ, mrange)
    n = length(xⁱ)
    for (mi,m) in enumerate(mrange)
        Lₘ = lagrange(xⁱ, m, x)
        χ[:,mi] = Nⁱ[m]*Lₘ
    end
end

function evaluate!(basis::Basis, x::AbstractVector, χ::AbstractMatrix)
    g = basis.grid
    n = order(g)

    (last(x) < first(basis.grid.X) || first(x) > last(basis.grid.X)) && return χ

    sel = 1:1
    el1shift = n - length(bases(g, 1))
    for i in elems(g)
        sel = find_interval(g.X, x, i, sel)
        last(sel) < first(sel) && continue # This element is not covered by x
        b = bases(g, i)
        eval_element!(g.X[i,:], g.N[i,:], x[sel],
                      view(χ, sel,
                           eachindex(b) .+ (i-1)*(n-1) .- (i > 1 ? el1shift : 0)),
                      b)
    end
    χ
end

function (basis::Basis)(x::AbstractVector)
    χ = spzeros(length(x),basecount(basis.grid))
    evaluate!(basis, x, χ)
end

"""
    basis(f)

Generate the scalar operator corresponding to `f(x)` on the FEDVR
`basis`.
"""
(basis::Basis)(f::Function) = Diagonal(f.(locs(basis.grid)))

"""
    basis(I)

Return the identity matrix of `basis`.
"""
(basis::Basis)(::UniformScaling) = Diagonal(ones(basecount(basis.grid)))

function show(io::IO, basis::Basis)
    write(io, "FEDVR Basis($(basis.grid))")
end

@recipe function plot(basis::Basis, n=1001)
    x = range(basis.grid.X[1,1], stop=basis.grid.X[end,end], length=n)
    xticks --> [basis.grid.X[:,1]..., basis.grid.X[end,end]]
    xlabel --> "x"
    legend --> false
    x,evaluate(basis, x)
end

export evaluate
