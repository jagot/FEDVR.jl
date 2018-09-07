using Pkg
Pkg.activate(".")

using FEDVR
using BlockMaps
using LinearAlgebra

N = 30
n = 10
L = 5.0
xx = range(0, stop=L, length=N+1)
basis = FEDVR.Basis(xx, n)
T = kinop(basis)
x = locs(basis.grid)
χ = evaluate(basis, x)

ψ = rand(Float64, size(x))
tt=similar(ψ)

mul!(tt, T, ψ)
