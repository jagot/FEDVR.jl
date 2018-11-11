using BlockBandedMatrices
using LinearAlgebra

function der_blocks(::Type{T}, basis::Basis, a, b) where T
    (a ∉ [0,1] || b ∉ [0,1]) &&
        error("Can only calculate derivative operators of orders 0–2!")

    g = basis.grid
    elrange = elems(g)
    n = order(g)

    f0 = zeros(elcount(g), n, n)
    for i = elrange
        f0[i,:,:] = sparse(I, n, n)
    end
    fa = a == 1 ? -basis.L′ : f0 # ∂ᴴ = -∂
    fb = b == 1 ? basis.L′ : f0

    d̃ = [zeros(T,n,n) for i = elrange]
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

der_blocks(::Type{T},basis::Basis,o) where T =
    der_blocks(T,basis::Basis,fld(o,2),cld(o,2))

function triderop(indices, D::AbstractVector{AbstractMatrix{T}}) where T
    tmp = Vector{T}(undef, maximum(maximum.(indices)) + size(last(D),1)-1)
    Dm = Tridiagonal(tmp[2:end], tmp, tmp[2:end])
    for (i,d) in zip(indices,D)
        s = i[1]:i[1]+size(d,1)-1
        @view(Dm[s,s]) .= d
    end
    Dm
end

function block_banded_derop(D::Vector{M}) where {T,M<:AbstractMatrix{T}}
    rows,lu = if length(D) > 1
        vcat([size(D[1],1)-1,1],
             vcat([[size(D[i],1)-2,1] for i=2:length(D)-1]...),
             size(D[end],1)-1),vcat(1,repeat([2,1],length(D)-1))
    else
        [size(D[1],1)],[0]
    end
    m = sum(rows)
    Dm = BlockSkylineMatrix(Zeros{T}(m,m), (rows,rows), (lu,lu))
    for i in 1:length(D)
        s = 1+(i>1)
        e = size(D[i],1)-(i<length(D))
        @view(Dm[Block(2i-1,2i-1)]) .= @view((D[i])[s:e,s:e])
        if i < length(D)
            @view(Dm[Block(2i,2i)]) .= @view((D[i])[end,end])
            @view(Dm[Block(2i-1,2i)]) .= @view((D[i])[s:e,end])
            @view(Dm[Block(2i,2i-1)]) .= reshape(@view((D[i])[end,s:e]), 1, length(s:e))
        end
        if i > 1
            @view(Dm[Block(2i-1,2i-2)]) .= @view((D[i])[s:e,1])
            @view(Dm[Block(2i-2,2i-1)]) .= reshape(@view((D[i])[1,s:e]), 1, length(s:e))
        end
        if i > 1 && i < length(D)
            @view(Dm[Block(2i-2,2i)]) .= @view((D[i])[1,end])
            @view(Dm[Block(2i,2i-2)]) .= @view((D[i])[end,1])
        end
    end
    Dm
end

function derop(::Type{T}, basis::Basis, o) where T
    g = basis.grid
    n = order(g)
    elrange = elems(g)
    indices,d̃ = der_blocks(T, basis, o)
    D = [zeros(T,n,n) for i = elrange]
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

    n == 2 && return triderop(indices, D)

    block_banded_derop(D)
end

derop(basis::Basis, o) =
    derop(eltype(basis), basis, o)

function kinop(::Type{T}, basis::Basis) where T
    D2 = derop(T, basis, 2)
    D2 ./= -2
    D2
end
kinop(basis::Basis) = kinop(eltype(basis), basis)

export derop, kinop
