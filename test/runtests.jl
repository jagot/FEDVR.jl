using FEDVR
using Base.Test

@testset "grid" begin
    N = 11
    n = 5
    breaks = linspace(0,1,N)
    grid = FEDVR.Grid(breaks, n)

    # We want the endpoints of the finite elements to match up exactly
    @test grid.X[1:end-1,end] == grid.X[2:end,1]

    @test grid.N[1] == 1/√(grid.W[1])
    @test grid.N[2,1] == 1/√(grid.W[1,end]+grid.W[2,1])

    @test elcount(grid) == N-1
    @test elems(grid) == 1:N-1
    @test order(grid) == n
    @test basecount(grid) == (N-1)*n - (N-2)
    Xp = locs(grid)
    @test length(Xp) == basecount(grid)
    @test Xp[1] == breaks[1]
    @test Xp[end] == breaks[end]

    println(grid)

    using RecipesBase
    RecipesBase.apply_recipe(Dict{Symbol,Any}(), grid)
end

@testset "lagrange" begin
    N = 11
    n = 5
    breaks = linspace(0,1,N)
    grid = FEDVR.Grid(breaks, n)
    e = m -> vec([zeros(m-1);1;zeros(n-m)])
    for i = elems(grid)
        for m = 1:n
            @test FEDVR.lagrange(grid.X[i,:], m, grid.X[i,m]) == 1
            @test FEDVR.lagrange(grid.X[i,:], m, grid.X[i,:]) == e(m)
        end
    end

    @test FEDVR.δ(-1, -1) == 1
    @test FEDVR.δ(-1, 1) == 0
    @test FEDVR.δ(0, 0) == 1
    @test FEDVR.δ(0, 1) == 0

    f = x -> x^4
    g = x -> 4x^3

    function vecdist(a::AbstractVector, b::AbstractVector,
                     ϵ = eps(eltype(a)))
        δ = √(sum(abs2, a-b))
        δ, δ/√(sum(abs2, a .+ ϵ))
    end

    for i = elems(grid)
        L′ = zeros(n,n)
        FEDVR.lagrangeder!(grid.X[i,:], grid.W[i,:], L′)
        c = f.(grid.X[i,:])
        c′ = L′'c
        c′a = g.(grid.X[i,:])
        @test vecdist(c′, c′a)[2] < 1e-13
    end
end
