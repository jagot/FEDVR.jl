type Basis
    grid::Grid
    f′::AbstractArray
end

function Basis(r::AbstractVector, n::Integer, args...)
    grid = Grid(r, n, args...)
    f′ = eval_base_ders(grid)
    Basis(grid, f′)
end

"""
    find_interval(X, x, i, sel)

Find the interval of points in `x` covering the `i`:th row in
`X`. The search is started from the first element in `sel`."""
function find_interval(X, x, i, sel)
    geq(v) = x -> x ≥ v
    a = findfirst(geq(X[i,1]), x[sel[1]:end]) + sel[1] - 1
    b = max(a,sel[end])
    b = findfirst(geq(X[i,end]), x[b:end]) + b - 1
    a:b
end

"""
    lagrange(xⁱ, m, x)

Calculate the Lagrange interpolating polynomial Lₘ(x), given the roots
`xⁱ`.

Lₘ(x) = ∏(j≠m) (x-xⁱⱼ)/(xⁱₘ-xⁱⱼ)
"""
function lagrange(xⁱ::AbstractVector, m::Integer, x)
    Lₘ = ones(x)
    for j in eachindex(xⁱ)
        j == m && continue
        Lₘ .*= (x-xⁱ[j])/(xⁱ[m]-xⁱ[j])
    end
    Lₘ
end

function eval_element!(xⁱ, wⁱ, wa, wb,
                       x, χ)
    n = length(xⁱ)
    for m in 1:n
        Lₘ = lagrange(xⁱ, m, x)
        χ[:,m] = Lₘ /
            if m in 2:n-1
                sqrt(wⁱ[m])
            elseif m == 1
                sqrt(wa+wⁱ[1])
            else
                sqrt(wⁱ[end]+wb)
            end
    end
end

function evaluate!(basis::Basis, x::AbstractVector, χ::AbstractMatrix)
    g = basis.grid
    N,n = size(g)

    sel = 1:1
    for i in 1:N-1
        sel = find_interval(g.X, x, i, sel)
        eval_element!(g.X[i,:], g.W[i,:],
                      i > 1 ? g.W[i-1,end] : 0,
                      i < N-1 ? g.W[i+1,1] : 0,
                      x[sel], view(χ, sel, (1:n) + (i-1)*(n-1)))
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
    N,n = size(basis.grid)
    χ = spzeros(length(x),(N-1)*(n-1)+1)
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
