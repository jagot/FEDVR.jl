using LinearMaps

function potop(basis::Basis, V::Function)
    Vv = V.(locs(basis.grid))
    LinearMap(spdiagm(0 => Vv), isposdef=all(Vv .> 0))
end

export potop
