using FastGaussQuadrature
import Base: size, show

type Grid
    X::AbstractMatrix
    W::AbstractMatrix
    N::Integer
    n::Integer
    bl::Symbol
    br::Symbol
end

function Grid(r::AbstractVector, n::Integer,
              bl::Symbol=:dirichlet0,
              br::Symbol=:dirichlet0)
    xₘ,wₘ = gausslobatto(n)
    N = length(r)
    X = zeros(N-1, n)
    W = zeros(N-1, n)
    for i = 1:N-1
        X[i,:] = 0.5*((r[i+1]-r[i])*xₘ + (r[i+1]+r[i]))
        W[i,:] = 0.5*(r[i+1]-r[i])*wₘ
    end
    Grid(X,W,N,n,bl,br)
end

elcount(grid::Grid) = grid.N - 1
order(grid::Grid) = grid.n
size(grid::Grid) = grid.N,grid.n
locs(grid::Grid) = [grid.X[:,1:end-1]'[:]..., grid.X[end]]
function weights(grid::Grid)
    bridges = grid.W[1:end-1,end] + grid.W[2:end,1]
    wl = grid.bl == :dirichlet0 ? 0 : grid.W[1]
    wr = grid.br == :dirichlet0 ? 0 : grid.W[end]
    [wl, [grid.W[:,2:end-1] [bridges;wr]]'[:]...]
end

function show(io::IO, grid::Grid)
    write(io, "FEDVR Grid with $(elcount(grid)) elements of order $(order(grid)) [$(grid.bl),$(grid.br)]")
end

@recipe function plot(grid::Grid)
    markershape --> :circle
    label --> "FEDVR grid locations"
    xticks --> 1:order(grid)-1:prod(size(grid))
    yticks --> [grid.X[:,1]..., grid.X[end,end]]
    xlabel --> "#"
    ylabel --> "x"
    locs(grid)
end
