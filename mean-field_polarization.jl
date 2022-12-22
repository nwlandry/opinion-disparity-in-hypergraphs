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
include("src/polarization.jl")
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


m = 101
n = 101

epsilon2 = LinRange(0, 1.0, m)
epsilon3 = LinRange(0.8, 1.0, n)

e2 = 0.5
e3 = 0.95

beta2 = LinRange(0, 0.5, m)
beta3 = LinRange(3.0, 6.0, n)

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

open("Data/polarization/mean-field_epsilon2_epsilon3_polarization.json","w") do f
  JSON.print(f, data1)
end

open("Data/polarization/mean-field_beta2_beta3_polarization.json","w") do f
    JSON.print(f, data2)
end