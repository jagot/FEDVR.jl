using BlockMaps

function derop(basis::Basis, a, b)
    (a ∉ [0,1] || b ∉ [0,1]) &&
        error("Can only calculate derivative operators of orders 0–2!")

    g = basis.grid
    elrange = elems(g)
    n = order(g)

    f0 = zeros(elcount(g), n, n)
    for i = elrange
        f0[i,:,:] = speye(n)
    end
    fa = a == 1 ? basis.L′ : f0
    fb = b == 1 ? basis.L′ : f0

    d̃ = [zeros(n,n) for i = elrange]
    indices = Tuple{Integer,Integer}[]
    for i = elrange
        ii = (i-1)*(n-1)+1
        push!(indices, (ii,ii))
        sel = (1:n) + (i-1)*(n-1)
        fael = view(fa, i, :, :)
        fbel = view(fb, i, :, :)
        d̃el = d̃[i]
        for m = 1:n
            fm = fael[m,:]
            for m′ = 1:n
                fm′ = fbel[m′,:]
                d̃el[m,m′] = dot(fm,fm′.*basis.grid.W[i,:])
            end
        end
    end
    for i = 2:elcount(g)
        d̃⁻ = d̃[i-1][end,end]
        d̃⁺ = d̃[i][1,1]
        d̃[i-1][end,end] += d̃⁺
        d̃[i][1,1] += d̃⁻
    end
    indices,d̃
end

function derop(basis::Basis,o)
    a = o ÷ 2
    b = o - a
    derop(basis::Basis,a,b)
end

function kinop(basis::Basis)
    g = basis.grid
    n = order(g)
    elrange = elems(g)
    indices,t̃ = derop(basis,2)
    T = [zeros(n,n) for i = elrange]
    for i = elrange
        Tel = T[i]
        t̃el = t̃[i]
        for m = 1:n
            for m′ = 1:n
                Tel[m,m′] = 0.5g.N[i,m]*g.N[i,m′]*t̃el[m,m′]
            end
        end
    end

    if g.bl == :dirichlet0
        T[1][:,1] = 0
        T[1][1,:] = 0
    end
    if g.br == :dirichlet0
        T[end][:,end] = 0
        T[end][end,:] = 0
    end

    BlockMap(indices,T,
             clear_overlaps=true,
             overlap_tol=1e-8)
end

export kinop
