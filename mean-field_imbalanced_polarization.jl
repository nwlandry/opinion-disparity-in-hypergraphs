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



function vary_rho_epsilon2_distributed(k, q, rho, beta2, beta3, epsilon2, epsilon3)
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