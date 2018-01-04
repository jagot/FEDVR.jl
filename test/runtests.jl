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
    e = m -> [zeros(m-1);1;zeros(n-m)]
    for i = elems(grid)
        for m = 1:n
            @test FEDVR.lagrange(grid.X[i,:], m, grid.X[i,:]) == e(m)
        end
    end
end
