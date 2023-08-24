# Gets the stability of the rescaled homogeneous equations
using JSON

in_parallel = true

if in_parallel
    using Base.Threads, IntervalArithmetic, IntervalRootFinding, StaticArrays, LinearAlgebra
else
    using IntervalArithmetic, IntervalRootFinding, StaticArrays, LinearAlgebra
end


unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
include("src/opiniondisparity.jl")


function vary_epsilon2_epsilon3(beta2, beta3, epsilon2, epsilon3, parallel)
    m = length(epsilon2)
    n = length(epsilon3)

    psi = zeros(m, n)
    nu = zeros(m, n)

    if parallel
        println("Started running epsilon2 vs. epsilon3 in parallel:")
        Threads.@threads for (i, j) in collect(Iterators.product(1:m, 1:n))
            e2 = epsilon2[i]
            e3 = epsilon3[j]
            psi[i, j], nu[i, j] = get_opinion_disparity
            (e2, e3, beta2, beta3)
            print("$i, $j\n")
        end
    else
        println("Started running epsilon2 vs. epsilon3 sequentially:")
        for (i, j) in Iterators.product(1:m, 1:n)
            e2 = epsilon2[i]
            e3 = epsilon3[j]
            psi[i, j], nu[i, j] = get_opinion_disparity
            (e2, e3, beta2, beta3)
            print("$i, $j\n")
        end
    end

    return psi, nu
end

function vary_beta2_beta3(beta2, beta3, epsilon2, epsilon3)
    m = length(beta2)
    n = length(beta3)

    psi = zeros(m, n)
    nu = zeros(m, n)

    if parallel
        println("Started running beta2 vs. beta3 in parallel:")
        Threads.@threads for (i, j) in collect(Iterators.product(1:m, 1:n))
            b2 = beta2[i]
            b3 = beta3[j]
            psi[i, j], nu[i, j] = get_opinion_disparity
            (epsilon2, epsilon3, b2, b3)
            print("$i, $j\n")
        end
    else
        println("Started running beta2 vs. beta3 sequentially:")
        for (i, j) in Iterators.product(1:m, 1:n)
            b2 = beta2[i]
            b3 = beta3[j]
            psi[i, j], nu[i, j] = get_opinion_disparity
            (epsilon2, epsilon3, b2, b3)
            print("$i, $j\n")
        end
    end
    
    return psi, nu
end


m = 2
n = 2

epsilon2 = LinRange(0, 1.0, m)
epsilon3 = LinRange(0.8, 1.0, n)

e2 = 0.5
e3 = 0.95

beta2 = LinRange(0, 0.5, m)
beta3 = LinRange(3.0, 6.0, n)

b2 = 0.2
b3 = 4

psi1, nu1 = vary_epsilon2_epsilon3_distributed(b2, b3, epsilon2, epsilon3, in_parallel)
psi2, nu2 = vary_beta2_beta3_distributed(beta2, beta3, e2, e3, in_parallel)

data1 = Dict("beta2"=>b2, "beta3"=>b3, "epsilon2"=>epsilon2, "epsilon3"=>epsilon3, "psi"=>psi1, "nu"=>nu1)
data2 = Dict("beta2"=>beta2, "beta3"=>beta3, "epsilon2"=>e2, "epsilon3"=>e3, "psi"=>psi2, "nu"=>nu2)

open("Data/opiniondisparity/mean-field_epsilon2_epsilon3_opinion_disparity.json","w") do f
  JSON.print(f, data1)
end

open("Data/opiniondisparity/mean-field_beta2_beta3_opinion_disparity.json","w") do f
    JSON.print(f, data2)
end