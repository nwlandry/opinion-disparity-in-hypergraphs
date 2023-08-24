import json
import os
import sys

import numpy as np
import xgi

from src.GenerativeModels import uniform_planted_partition_hypergraph
from src.HypergraphContagion import get_x1_x2_in_parallel

rho = float(sys.argv[1])
epsilon2 = float(sys.argv[2])
epsilon3 = float(sys.argv[3])
num_realizations = 5

# parameters
n = 10000
k = 20
q = 20

fnamelist = []
for i in range(num_realizations):
    links = uniform_planted_partition_hypergraph(n, 2, k, epsilon2, rho)
    triangles = uniform_planted_partition_hypergraph(n, 3, q, epsilon3, rho)
    H = xgi.Hypergraph(links + triangles)

    fnamelist.append(f"Data/SBM/hypergraphs/{rho}-{epsilon2}-{epsilon3}-{i}.txt")
    xgi.write_edgelist(H, fnamelist[-1])
    print(f"Hypergraph {i} generated", flush=True)

# Epidemic parameters
gamma = 1
tmax = 100
fraction_to_average = 0.1
is_verbose = True
num_processes = len(os.sched_getaffinity(0))
num_sims = 1

numxpoints = 21
numypoints = 21

beta2tilde = 0.2
beta3tilde = 4

data = dict()

arglist = list()
for rho1 in np.linspace(0, 1, numxpoints):
    for rho2 in np.linspace(0, 1, numypoints):
        # different instances of hypergraphs
        for fname in fnamelist:
            community1 = {i for i in range(int(H.num_nodes * rho))}
            community2 = {i for i in range(int(H.num_nodes * rho), n)}

            beta2c = gamma / k
            beta3c = gamma / q

            beta2 = beta2tilde * beta2c
            beta3 = beta3tilde * beta3c

            beta = {2: beta2, 3: beta3}
            arglist.append(
                (
                    fname,
                    gamma,
                    beta,
                    community1,
                    community2,
                    rho1,
                    rho2,
                    tmax,
                    fraction_to_average,
                    num_sims,
                    is_verbose,
                )
            )

print("Started finding fixed points:", flush=True)
fixed_points = get_x1_x2_in_parallel(arglist, num_processes)

data["gamma"] = gamma
data["beta2"] = beta2tilde
data["beta3"] = beta3tilde
data["rho"] = rho
data["epsilon2"] = epsilon2
data["epsilon3"] = epsilon3
data["fixed-points"] = fixed_points

datastring = json.dumps(data)

with open(f"Data/opiniondisparity/{rho}-{epsilon2}-{epsilon3}.json", "w") as output_file:
    output_file.write(datastring)
