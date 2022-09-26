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

function get_polarization(epsilon2, epsilon3, beta2, beta3)
    psi_max = 0
    nu_max = Inf

    function f((x1, x2))
        dxdt1 = -x1 + 0.5*beta2*(1 - x1)*(x1 + x2 + epsilon2*(x1 - x2)) + 0.25*beta3*(1 - x1)*((x1 + x2)^2 + epsilon3*(3*x1^2 - 2*x1*x2 - x2^2))
        dxdt2 = -x2 + 0.5*beta2*(1 - x2)*(x2 + x1 + epsilon2*(x2 - x1)) + 0.25*beta3*(1 - x2)*((x2 + x1)^2 + epsilon3*(3*x2^2 - 2*x2*x1 - x1^2))
        return SVector(dxdt1, dxdt2)
    end

    function jacobian((x1, x2))
        J = zeros(2, 2)
        J[1, 1] = -1 + 0.5*beta2*(1 - x1)*(1 + epsilon2) - 0.5*beta2*(x1 + x2 + epsilon2*(x1 - x2)) +
                0.5*beta3*(1 - x1)*(x1 + x2 + epsilon3*(3*x1 - x2)) - 0.25*beta3*((x1 + x2)^2 + epsilon3*(3*x1^2 - 2*x1*x2 - x2^2))
        J[1, 2] = 0.5*beta2*(1 - x1)*(1 - epsilon2) + 0.5*beta3*(1 - x1)*(x1 + x2 - epsilon3*(x1 + x2))
        J[2, 1] = 0.5*beta2*(1 - x2)*(1 - epsilon2) + 0.5*beta3*(1 - x2)*(x2 + x1 - epsilon3*(x2 + x1))
        J[2, 2] = -1 + 0.5*beta2*(1 - x2)*(1 + epsilon2) - 0.5*beta2*(x2 + x1 + epsilon2*(x2 - x1)) +
                0.5*beta3*(1 - x2)*(x2 + x1 + epsilon3*(3*x2 - x1)) - 0.25*beta3*((x2 + x1)^2 + epsilon3*(3*x2^2 - 2*x2*x1 - x1^2))
        return J
    end

    function spectral_abscissa(J)
        return maximum(real(eigvals(J)))
    end
    
    result = roots(f, IntervalBox(0.0..0.5, 0.5..1.0))
    for rt in result
        x = mid(rt.interval)
        psi = abs(x[1] - x[2])


        if psi > 0.01 # filter out symmetric roots to save time
            nu = spectral_abscissa(jacobian(x))
            
            if (psi > psi_max && nu < 0)
                psi_max = psi
                nu_max = nu
            end
        end
    end

    return psi_max, nu_max
end

end

function vary_epsilon2_epsilon3(beta2, beta3, epsilon2, epsilon3)
    m = length(epsilon2)
    n = length(epsilon3)

    psi = zeros(m, n)
    nu = zeros(m, n)

    for i in eachindex(epsilon2)
        e2 = epsilon2[i]
        for j in eachindex(epsilon3)
            e3 = epsilon3[j]
            psi[i, j], nu[i, j] = get_polarization(epsilon2, epsilon3, beta2, beta3)

            println((i, j, psi[i, j]))
            flush(stdout)
        end
    end
    return psi, nu
end

function vary_beta2_beta3(beta2, beta3, epsilon2, epsilon3)
    m = length(beta2)
    n = length(beta3)

    psi = zeros(m, n)
    nu = zeros(m, n)

    for i in eachindex(beta2)
        b2 = beta2[i]
        for j in eachindex(beta3)
            b3 = beta3[j]
            psi[i, j], nu[i, j] = get_polarization(epsilon2, epsilon3, b2, b3)

            println((i, j, psi[i, j]))
            flush(stdout)
        end
    end
    return psi, nu
end

function vary_epsilon2_epsilon3_distributed(beta2, beta3, epsilon2, epsilon3)
    m = length(epsilon2)
    n = length(epsilon3)

    psi = zeros(m, n)
    nu = zeros(m, n)
    
    for i in eachindex(epsilon2)
        e2 = epsilon2[i]
        results = pmap(e3 -> get_polarization(e2, e3, beta2, beta3), epsilon3)
        psi[i, :], nu[i, :] = unzip(results)
        println(i)
        flush(stdout)
    end
    
    return psi, nu
end

function vary_beta2_beta3_distributed(beta2, beta3, epsilon2, epsilon3)
    m = length(beta2)
    n = length(beta3)

    psi = zeros(m, n)
    nu = zeros(m, n)
    
    for i in eachindex(beta2)
        b2 = beta2[i]
        results = pmap(b3 -> get_polarization(epsilon2, epsilon3, b2, b3), beta3)
        psi[i, :], nu[i, :] = unzip(results)
        println(i)
        flush(stdout)
    end
    
    return psi, nu
end


m = 100
n = 100

epsilon2 = LinRange(0, 1.0, m)
epsilon3 = LinRange(0.8, 1.0, n)

e2 = 0.5
e3 = 0.95

beta2 = LinRange(0, 1.0, m)
beta3 = LinRange(1.0, 7.0, n)

b2 = 0.2
b3 = 4

if in_parallel
    psi1, nu1 = vary_epsilon2_epsilon3_distributed(b2, b3, epsilon2, epsilon3)
    psi2, nu2 = vary_beta2_beta3_distributed(beta2, beta3, e2, e3)
else
    psi1, nu1 = vary_epsilon2_epsilon3(b2, b3, epsilon2, epsilon3)
    psi2, nu2 = vary_beta2_beta3(beta2, beta3, e2, e3)
end

data1 = Dict("beta2"=>b2, "beta3"=>b3, "epsilon2"=>epsilon2, "epsilon3"=>epsilon3, "psi"=>psi1, "nu"=>nu1)
data2 = Dict("beta2"=>beta2, "beta3"=>beta3, "epsilon2"=>e2, "epsilon3"=>e3, "psi"=>psi2, "nu"=>nu2)

open("Data/stability/theoretical_epsilon2_epsilon3_polarization.json","w") do f
  JSON.print(f, data1)
end

open("Data/stability/theoretical_beta_2_beta3_polarization.json","w") do f
    JSON.print(f, data2)
end