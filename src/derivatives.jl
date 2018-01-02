"Kronecker δ function"
δ(a,b) = a == b ? 1 : 0

function eval_base_ders_element!(xⁱ::AbstractVector, wⁱ::AbstractVector,
                                 f′::AbstractMatrix)
    n = length(xⁱ)
    for m in 1:n
        f′[m,m] = (δ(m,n)-δ(m,1))/2wⁱ[m]
        for mp in 1:n
            mp == m && continue
            f = 1
            for k = 1:n
                k in [m,mp] && continue
                f *= (xⁱ[mp]-xⁱ[k])/(xⁱ[m]-xⁱ[k])
            end
            f′[m,mp] = f/(xⁱ[m]-xⁱ[mp])
        end
    end
end

function eval_base_ders(grid::Grid)
    N,n = size(grid)

    f′ = zeros(N-1, n, n)
    for i = 1:N-1
        sel = (1:n) + (i-1)*(n-1)
        eval_base_ders_element!(grid.X[i,:], grid.W[i,:],
                                view(f′, i, :, :))
    end
    # if grid.bl == :dirichlet0
    #     f′[1,:,1] = 0
    # end
    # if grid.br == :dirichlet0
    #     f′[end,:,end] = 0
    # end

    f′
end

function derop(basis::Basis, a, b)
    (a ∉ [0,1] || b ∉ [0,1]) &&
        error("Can only calculate derivative operators of orders 0–2!")
    
    N,n = size(basis.grid)
    l = (N-1)*n - (N-2)

    f0 = zeros(N-1, n, n)
    for i = 1:N-1
        f0[i,:,:] = speye(n)
    end
    fa = a == 1 ? basis.f′ : f0
    fb = b == 1 ? basis.f′ : f0
    
    d̃ = spzeros(l,l)
    for i = 1:N-1
        sel = (1:n) + (i-1)*(n-1)
        fael = view(fa, i, :, :)
        fbel = view(fb, i, :, :)
        d̃el = view(d̃, sel, sel)
        for m = 1:n
            fm = fael[m,:]
            for mp = 1:n
                fmp = fbel[mp,:]
                d̃el[m,mp] = dot(fm,fmp.*basis.grid.W[i,:])
            end
        end
    end
    d̃
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
    N,n = size(g)
    t̃ = derop(basis,2)
    T = spzeros(size(t̃)...)
    for i = 1:N-1
        sel = (1:n) + (i-1)*(n-1)
        Tel = view(T, sel, sel)
        t̃el = view(t̃, sel, sel)
        for m = 1:n
            for mp = 1:n
                Tel[m,mp] = 0.5t̃el[m,mp]/sqrt(g.W[i,m]*g.W[i,mp])
            end
        end
    end
    T
end
