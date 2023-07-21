# Gets the stability of the rescaled imbalanced homogeneous equations
using JSON

in_parallel = true

if in_parallel
    using Base.Threads, IntervalArithmetic, IntervalRootFinding, StaticArrays, LinearAlgebra
else
    using IntervalArithmetic, IntervalRootFinding, StaticArrays, LinearAlgebra
end

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
include("src/polarization.jl")


function vary_rho_epsilon2(k, q, rho, beta2, beta3, epsilon2, epsilon3, parallel)
    m = length(rho)
    n = length(epsilon2)

    psi1 = zeros(m, n)
    nu1 = zeros(m, n)
    psi2 = zeros(m, n)
    nu2 = zeros(m, n)

    if parallel
        println("Started running rho vs. epsilon2 in parallel:")
        Threads.@threads for (i, j) in collect(Iterators.product(1:m, 1:n))
            r = rho[i]
            e2 = epsilon2[j]
            psi1[i, j], nu1[i, j], psi2[i, j], nu2[i, j] = get_imbalanced_polarization(k, q, r, e2, epsilon3, beta2, beta3)
            print("$i, $j\n")
        end
    else
        println("Started running rho vs. epsilon2 sequentially:")
        for (i, j) in Iterators.product(1:m, 1:n)
            r = rho[i]
            e2 = epsilon2[j]
            psi1[i, j], nu1[i, j], psi2[i, j], nu2[i, j] = get_imbalanced_polarization(k, q, r, e2, epsilon3, beta2, beta3)
            print("$i, $j\n")
        end
    end
    return psi1, nu1, psi2, nu2
end

function vary_rho_epsilon3(k, q, rho, beta2, beta3, epsilon2, epsilon3, parallel)
    m = length(rho)
    n = length(epsilon3)

    psi1 = zeros(m, n)
    nu1 = zeros(m, n)
    psi2 = zeros(m, n)
    nu2 = zeros(m, n)

    if parallel
        println("Started running rho vs. epsilon3 in parallel:")
        Threads.@threads for (i, j) in collect(Iterators.product(1:m, 1:n))
            r = rho[i]
            e3 = epsilon3[j]
            psi1[i, j], nu1[i, j], psi2[i, j], nu2[i, j] = get_imbalanced_polarization(k, q, r, epsilon2, e3, beta2, beta3)
            print("$i, $j\n")
        end
    else
        println("Started running rho vs. epsilon3 sequentially:")
        for (i, j) in Iterators.product(1:m, 1:n)
            r = rho[i]
            e3 = epsilon3[j]
            psi1[i, j], nu1[i, j], psi2[i, j], nu2[i, j] = get_imbalanced_polarization(k, q, r, epsilon2, e3, beta2, beta3)
            print("$i, $j\n")
        end
    end
    
    return psi1, nu1, psi2, nu2
end


m = 101
n = 101

k = 20
q = 20
epsilon3 = 0.95
beta2 = 0.2/k
beta3 = 4/q

rho = LinRange(0.48, 0.52, m)
epsilon2 = LinRange(0, 1.0, m)
psi1, nu1, psi2, nu2 = vary_rho_epsilon2(k, q, rho, beta2, beta3, epsilon2, epsilon3, in_parallel)
data = Dict("k"=>k, "q"=>q, "epsilon2"=>epsilon2, "epsilon3"=>epsilon3, "rho"=>rho, "beta2"=>beta2, "beta3"=>beta3, "psi1"=>psi1, "nu1"=>nu1, "psi2"=>psi2, "nu2"=>nu2)

open("Data/polarization/mean-field_rho_epsilon2_polarization.json","w") do f
  JSON.print(f, data)
end

epsilon2 = 0.5
epsilon3 = LinRange(0.8, 1.0, m)
psi1, nu1, psi2, nu2 = vary_rho_epsilon3(k, q, rho, beta2, beta3, epsilon2, epsilon3, in_parallel)
data = Dict("k"=>k, "q"=>q, "epsilon2"=>epsilon2, "epsilon3"=>epsilon3, "rho"=>rho, "beta2"=>beta2, "beta3"=>beta3, "psi1"=>psi1, "nu1"=>nu1, "psi2"=>psi2, "nu2"=>nu2)

open("Data/polarization/mean-field_rho_epsilon3_polarization.json","w") do f
  JSON.print(f, data)
end