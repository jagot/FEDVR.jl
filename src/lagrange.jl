"""
    lagrange(xⁱ, m, x)

Calculate the Lagrange interpolating polynomial Lⁱₘ(x), given the roots
`xⁱ`.

Lⁱₘ(x) = ∏(j≠m) (x-xⁱⱼ)/(xⁱₘ-xⁱⱼ)

Eq. (8) Rescigno2000
"""
function lagrange(xⁱ::AbstractVector, m::Integer, x::AbstractVector)
    Lₘ = fill(1.0, length(x))
    for j in eachindex(xⁱ)
        j == m && continue
        Lₘ .*= broadcast(-, x, xⁱ[j])/(xⁱ[m]-xⁱ[j])
    end
    Lₘ
end
lagrange(xⁱ::AbstractVector, m::Integer, x::Number) =
    lagrange(xⁱ, m, [x])[1]

"""
    δ(a,b)

Kronecker δ function"""
δ(a,b) = a == b ? 1 : 0

"""
    lagrangeder!(xⁱ, m, L′)

Calculate the derivative of the Lagrange interpolating polynomial
Lⁱₘ(x), given the roots `xⁱ`, *at* the roots, and storing the result
in `L′`.

Lⁱₘ′(x) = (xⁱₘ-xⁱₘ,)⁻¹ ∏(k≠m,m′) (xⁱₘ,-xⁱₖ)/(xⁱₘ-xⁱₖ), m≠m′,
          [δ(m,n) - δ(m,1)]/2wⁱₘ,                    m=m′

Eq. (20) Rescigno2000
"""
function lagrangeder!(xⁱ::AbstractVector, wⁱ::AbstractVector,
                      L′::AbstractMatrix)
    n = length(xⁱ)
    for m in 1:n
        L′[m,m] = (δ(m,n)-δ(m,1))/2wⁱ[m]
        for m′ in 1:n
            m′ == m && continue
            f = 1
            for k = 1:n
                k in [m,m′] && continue
                f *= (xⁱ[m′]-xⁱ[k])/(xⁱ[m]-xⁱ[k])
            end
            L′[m,m′] = f/(xⁱ[m]-xⁱ[m′])
        end
    end
end
