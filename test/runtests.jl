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
end
