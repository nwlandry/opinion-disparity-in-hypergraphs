unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

function imbalanced_polarization_e2_rho_distributed()
    m = length(rho)
    n = length(epsilon2)

    psi1 = zeros(m, n)
    nu1 = zeros(m, n)
    psi2 = zeros(m, n)
    nu2 = zeros(m, n)
    
    for i in eachindex(rho)
        r = rho[i]
        results = pmap(e2 -> get_imbalanced_polarization(k, q, r, e2, epsilon3, beta2, beta3), epsilon2)
        psi1[i, :], nu1[i, :], psi2[i, :], nu2[i, :] = unzip(results)
        println(i)
        flush(stdout)
    end
    
    return psi1, nu1, psi2, nu2
    @floop for (i,j) in product(1:m,1:n)
end