using LinearMaps

function potop(basis::Basis, V::Function)
    Vv = V.(locs(basis.grid))
    LinearMap(spdiagm(Vv), isposdef=all(Vv .> 0))
end

export potop
