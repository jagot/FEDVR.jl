using FEDVR
using Base.Test

function vecdist(a::AbstractVector, b::AbstractVector,
                 ϵ = eps(eltype(a)))
    δ = √(sum(abs2, a-b))
    δ, δ/√(sum(abs2, a .+ ϵ))
end

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

    gW = weights(grid)
    gN = [grid.N[:,1:end-1]'[:]..., grid.N[end]]
    @test vecdist(sqrt.(gW[2:end-1]),
                  1./gN[2:end-1])[1] < eps(Float64)

    @testset "intervals" begin
        for nn = 300:301
            x = linspace(breaks[1], breaks[end], nn)
            sel = 1:1
            sel = FEDVR.find_interval(grid.X, x, 1, sel)
            @test x[sel[1]] == breaks[1]
            for i = elems(grid)[1:end-1]
                sel = FEDVR.find_interval(grid.X, x, i, sel)
                @test x[sel[1]] >= breaks[i]
                @test x[sel[end]] < breaks[i+1]
            end
            i = elcount(grid)
            sel = FEDVR.find_interval(grid.X, x, i, sel)
            @test x[sel[1]] >= breaks[i]
            @test x[sel[end]] == breaks[end]
        end
    end

    @testset "misc" begin
        @test !isempty(string(grid))

        @test begin
            using RecipesBase
            RecipesBase.apply_recipe(Dict{Symbol,Any}(), grid)
            true
        end
    end
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

    L′ = FEDVR.lagrangeder(grid)

    for i = elems(grid)
        c = f.(grid.X[i,:])
        c′ = L′[i,:,:]'c
        c′a = g.(grid.X[i,:])
        @test vecdist(c′, c′a)[2] < 1e-13
    end
end

@testset "basis" begin
    N = 11
    n = 5
    breaks = linspace(0,1,N)
    basis = FEDVR.Basis(breaks, n)

    x = locs(basis.grid)
    χ = evaluate(basis, x)
    dχ = diag(χ)
    @test dχ[1] == dχ[end] == 0
    gN = [basis.grid.N[:,1:end-1]'[:]..., basis.grid.N[end]]
    @test dχ[2:end-1] == gN[2:end-1]

    @testset "misc" begin
        @test !isempty(string(basis))

        @test begin
            using RecipesBase
            RecipesBase.apply_recipe(Dict{Symbol,Any}(), basis)
            true
        end
    end
end

@testset "projections" begin
    breaks = linspace(0,1,11)
    n = 5
    basis = FEDVR.Basis(breaks, n, :dirichlet1, :dirichlet1)
    x = linspace(minimum(breaks),maximum(breaks),301)
    χ = evaluate(basis, x)

    f = x -> x^3 - 7x^2 + x^4 + 2
    ϕ = project(f, basis)

    @test vecdist(f.(x), χ*ϕ)[2] < 10eps(Float64)
end
