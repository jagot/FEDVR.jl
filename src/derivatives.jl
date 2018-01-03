using BlockMaps

function eval_base_ders(grid::Grid)
    n = order(grid)

    L′ = zeros(elcount(grid), n, n)
    for i = elems(grid)
        sel = (1:n) + (i-1)*(n-1)
        eval_base_ders_element!(grid.X[i,:], grid.W[i,:],
                                view(L′, i, :, :))
    end
    # if grid.bl == :dirichlet0
    #     L′[1,:,1] = 0
    # end
    # if grid.br == :dirichlet0
    #     L′[end,:,end] = 0
    # end

    L′
end

function derop(basis::Basis, a, b)
    (a ∉ [0,1] || b ∉ [0,1]) &&
        error("Can only calculate derivative operators of orders 0–2!")

    g = basis.grid
    # l = basecount(base.grid)
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
            for mp = 1:n
                fmp = fbel[mp,:]
                d̃el[m,mp] = dot(fm,fmp.*basis.grid.W[i,:])
            end
        end
    end
    indices,d̃
end

function derop(basis::Basis,o)
    a = o ÷ 2
    b = o - a
    derop(basis::Basis,a,b)
end

#=
\[\tilde{t}^i_{mm'} =
\sum_{m''} \d{f^i_m(m'')}{x} \d{f^i_{m'}(m'')}{x} w^i_{m''}\]
=#


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
            for mp = 1:n
                Tel[m,mp] = 0.5t̃el[m,mp]/sqrt(g.W[i,m]*g.W[i,mp])
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
