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


k = 5
l = 5

epsilon2 = LinRange(0.0, 1.0, k)
epsilon3 = LinRange(0.95, 1.0, l)

m = 301
n = 301

beta2 = LinRange(0, 1.1, m)
beta3 = LinRange(0.5, 6.0, n)


data1 = Dict("beta2"=>beta2, "beta3"=>beta3, "epsilon2"=>epsilon2, "epsilon3"=>1.0)
data2 = Dict("beta2"=>beta2, "beta3"=>beta3, "epsilon2"=>1.0, "epsilon3"=>epsilon3)

if in_parallel
    for e2 in epsilon2
        psi, nu = vary_beta2_beta3_distributed(beta2, beta3, e2, 1.0)
        data1["psi-$e2"] = psi
        data1["nu-$e2"] = nu
    end

    for e3 in epsilon3
        psi, nu = vary_beta2_beta3_distributed(beta2, beta3, 1.0, e3)
        data2["psi-$e3"] = psi
        data2["nu-$e3"] = nu
    end

else
    for e2 in epsilon2
        psi, nu = vary_beta2_beta3(beta2, beta3, e2, 1.0)
        data1["psi-$e2"] = psi
        data1["nu-$e2"] = nu
    end

    for e3 in epsilon3
        psi, nu = vary_beta2_beta3(beta2, beta3, 1.0, e3)
        data2["psi-$e3"] = psi
        data2["nu-$e3"] = nu
    end
end

open("Data/stability/mean-field_polarization_boundaries_epsilon2.json","w") do f
  JSON.print(f, data1)
end

open("Data/stability/mean-field_polarization_boundaries_epsilon3.json","w") do f
    JSON.print(f, data2)
end