include("polarization.jl")
using IterTools, FLoops

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

function test_epsilon2_vs_rho(k, q, rho, beta2, beta3, epsilon2, epsilon3)
    m = length(rho)
    n = length(epsilon2)

    psi1 = zeros(m, n)
    nu1 = zeros(m, n)
    psi2 = zeros(m, n)
    nu2 = zeros(m, n)

    @floop for (i, j) in product(rho, epsilon2)
        psi1[i, :], nu1[i, :], psi2[i, :], nu2[i, :] = get_imbalanced_polarization(k, q, r, e2, epsilon3, beta2, beta3), epsilon2)
        println((i, j))
        flush(stdout)
    end
    
    return psi1, nu1, psi2, nu2
end


function imbalanced_polarization_e2_rho(k, q, rho, beta2, beta3, epsilon2, epsilon3)
    m = length(rho)
    n = length(epsilon2)

    psi1 = zeros(m, n)
    nu1 = zeros(m, n)
    psi2 = zeros(m, n)
    nu2 = zeros(m, n)

    for (i, j) in product(rho, epsilon2)
        psi1[i, :], nu1[i, :], psi2[i, :], nu2[i, :] = get_imbalanced_polarization(k, q, r, e2, epsilon3, beta2, beta3), epsilon2)
        println((i, j))
        flush(stdout)
    end
    
    return psi1, nu1, psi2, nu2
end