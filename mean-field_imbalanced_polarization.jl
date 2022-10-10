# Gets the stability of the rescaled homogeneous equations
using Distributed, JSON

in_parallel = true

if in_parallel
    Distributed.addprocs(Sys.CPU_THREADS; exeflags="--project")
    @everywhere using Distributed, IntervalArithmetic, IntervalRootFinding, StaticArrays, LinearAlgebra
else
    using IntervalArithmetic, IntervalRootFinding, StaticArrays, LinearAlgebra
end


@everywhere begin
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

function get_polarization(k, q, rho, epsilon2, epsilon3, beta2, beta3)
    psi1_max = 0
    psi2_max = 0
    nu1_max = Inf
    nu2_max = Inf

    r2 = 1/(rho^2 + (1 - rho)^2) - 1
    r3 = 1/(rho^3 + (1 - rho)^3) - 1

    function f((x1, x2))
        dxdt1 = -x1 + beta2*k*(1 - x1)*(rho*(1 + r2*epsilon2)*x1 + (1 - rho)*(1 - epsilon2)*x2) +
            beta3*q*(1 - x1)*(rho^2*(1 + r3*epsilon3)*x1^2 + 2*rho*(1 - rho)*(1 - epsilon3)*x1*x2 +
            (1 - rho)^2*(1 - epsilon3)*x2^2)
        dxdt2 = -x2 + beta2*k*(1 - x2)*((1 - rho)*(1 + r2*epsilon2)*x2 + rho*(1 - epsilon2)*x1) +
            beta3*q*(1 - x2)*((1 - rho)^2*(1 + r3*epsilon3)*x2^2 + 2*(1 - rho)*rho*(1 - epsilon3)*x2*x1 +
            rho^2*(1 - epsilon3)*x1^2)
        return SVector(dxdt1, dxdt2)
    end

    function jacobian((x1, x2))
        J = zeros(2, 2)

        J[1, 1] = -1 + beta2*k*rho*(1 - x1)*(1 + r2*epsilon2) -
        beta2*k*((1 - rho)*(1 - epsilon2)*x2 + rho*(1 + r2*epsilon2)*x1) +
        2*beta3*q*(1 - x1)*(rho^2*(1 + r3*epsilon3)*x1 +
        (1 - rho)*rho*(1 - epsilon3)*x2) -
        beta3*q*(rho^2*(1 + r3*epsilon3)*x1^2 +
        2*(1 - rho)*rho*(1 - epsilon3)*x1*x2 + (1 - rho)^2*(1 - epsilon3)*x2^2)

        J[1, 2] = beta2*k*(1 - rho)*(1 - epsilon2)*(1 - x1) +
        2*beta3*q*(1 - rho)*rho*(1 - epsilon3)*(1 - x1)*x1 +
        2*beta3*q*(1 - rho)^2*(1 - epsilon3)*(1 - x1)*x2

        J[2, 1] = beta2*k*rho*(1 - epsilon2)*(1 - x2) +
        2*beta3*q*rho*(1 - rho)*(1 - epsilon3)*(1 - x2)*x2 +
        2*beta3*q*rho^2*(1 - epsilon3)*(1 - x2)*x1

        J[2, 2] = -1 + beta2*k*(1 - rho)*(1 - x2)*(1 + r2*epsilon2) -
        beta2*k*(rho*(1 - epsilon2)*x1 + (1 - rho)*(1 + r2*epsilon2)*x2) +
        2*beta3*q*(1 - x2)*((1 - rho)^2*(1 + r3*epsilon3)*x2 +
        rho*(1 - rho)*(1 - epsilon3)*x1) -
        beta3*q*((1 - rho)^2*(1 + r3*epsilon3)*x2^2 +
        2*rho*(1 - rho)*(1 - epsilon3)*x2*x1 + rho^2*(1 - epsilon3)*x1^2)
        
        return J
    end

    function spectral_abscissa(J)
        return maximum(real(eigvals(J)))
    end
    
    result = roots(f, IntervalBox(0..1, 0..1))
    for rt in result
        x = mid(rt.interval)
        nu = spectral_abscissa(jacobian(x))

        if (x[1] - x[2] > max(psi1_max, 0.01) && nu < 0)
            psi1_max = x[1] - x[2]
            nu1_max = nu
        elseif (x[2] - x[1] > max(psi2_max, 0.01) && nu < 0)
            psi2_max = x[2] - x[1]
            nu2_max = nu
        end
    end

    return psi1_max, nu1_max, psi2_max, nu2_max
end

end

function vary_rho_epsilon2(k, q, rho, beta2, beta3, epsilon2, epsilon3)
    m = length(rho)
    n = length(epsilon2)

    psi1 = zeros(m, n)
    nu1 = zeros(m, n)
    psi2 = zeros(m, n)
    nu2 = zeros(m, n)

    for i in eachindex(rho)
        r = rho[i]
        for j in eachindex(epsilon2)
            e2 = epsilon2[j]
            psi1[i, j], nu1[i, j], psi2[i, j], nu2[i, j] = get_polarization(k, q, r, e2, epsilon3, beta2, beta3)

            println((i, j, psi[i, j]))
            flush(stdout)
        end
    end
    return psi1, nu1, psi2, nu2
end

function vary_rho_epsilon2_distributed(k, q, rho, beta2, beta3, epsilon2, epsilon3)
    m = length(rho)
    n = length(epsilon2)

    psi1 = zeros(m, n)
    nu1 = zeros(m, n)
    psi2 = zeros(m, n)
    nu2 = zeros(m, n)
    
    for i in eachindex(rho)
        r = rho[i]
        results = pmap(e2 -> get_polarization(k, q, r, e2, epsilon3, beta2, beta3), epsilon2)
        psi1[i, :], nu1[i, :], psi2[i, :], nu2[i, :] = unzip(results)
        println(i)
        flush(stdout)
    end
    
    return psi1, nu1, psi2, nu2
end


m = 101
n = 101

k = 20
q = 20
rho = LinRange(0.48, 0.52, m)
epsilon2 = LinRange(0, 1.0, m)
epsilon3 = 0.95
beta2 = 0.2/k
beta3 = 4/q

if in_parallel
    psi1, nu1, psi2, nu2 = vary_rho_epsilon2_distributed(k, q, rho, beta2, beta3, epsilon2, epsilon3)
else
    psi1, nu1, psi2, nu2 = vary_rho_epsilon2(k, q, rho, beta2, beta3, epsilon2, epsilon3)
end

data = Dict("k"=>k, "q"=>q, "epsilon2"=>epsilon2, "epsilon3"=>epsilon3, "rho"=>rho, "beta2"=>beta2, "beta3"=>beta3, "psi1"=>psi1, "nu1"=>nu1, "psi2"=>psi2, "nu2"=>nu2)

open("Data/stability/mean-field_rho_epsilon2_polarization.json","w") do f
  JSON.print(f, data)
end