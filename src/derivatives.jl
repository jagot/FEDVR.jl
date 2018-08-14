using BlockMaps
using LinearAlgebra

function der_blocks(basis::Basis, a, b)
    (a ∉ [0,1] || b ∉ [0,1]) &&
        error("Can only calculate derivative operators of orders 0–2!")

    g = basis.grid
    elrange = elems(g)
    n = order(g)

    f0 = zeros(elcount(g), n, n)
    for i = elrange
        f0[i,:,:] = sparse(I, n, n)
    end
    fa = a == 1 ? basis.L′ : f0
    fb = b == 1 ? basis.L′ : f0

    d̃ = [zeros(n,n) for i = elrange]
    indices = Tuple{Integer,Integer}[]
    for i = elrange
        ii = (i-1)*(n-1) .+ 1
        push!(indices, (ii,ii))
        sel = (1:n) .+ (i-1)*(n-1)
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

function der_blocks(basis::Basis,o)
    a = o ÷ 2
    b = o - a
    der_blocks(basis::Basis,a,b)
end

function derop(basis::Basis,o)
    g = basis.grid
    n = order(g)
    elrange = elems(g)
    indices,d̃ = der_blocks(basis,o)
    D = [zeros(n,n) for i = elrange]
    for i = elrange
        Del = D[i]
        d̃el = d̃[i]
        for m = 1:n
            for m′ = 1:n
                Del[m,m′] = g.N[i,m]*g.N[i,m′]*d̃el[m,m′]
            end
        end
    end

    if g.bl == :dirichlet0
        D[1] = D[1][2:end,2:end]
        for i in 2:length(indices)
            indices[i] = (indices[i][1]-1,indices[i][2]-1)
        end
    end
    if g.br == :dirichlet0
        D[end] = D[end][1:end-1,1:end-1]
    end

    BlockMap(indices,D,
             overlaps=:split,
             overlap_tol=1e-8)
end

function kinop(basis::Basis)
    D = derop(basis, 2)
    blocks = map(D.blocks) do b
        BlockMaps.Block(0.5b.a, b.i, b.j)
    end
    BlockMap(size(D)..., blocks,
             issymmetric(D),
             ishermitian(D),
             isposdef(D),
             D.overlaps,
             D.overlap_tol)
end

export derop, kinop
