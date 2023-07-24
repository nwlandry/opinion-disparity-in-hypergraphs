import json
import os
import sys

import numpy as np
import xgi

from src import HypergraphContagion

rho = float(sys.argv[1])
epsilon2 = float(sys.argv[2])
epsilon3 = float(sys.argv[2])

# parameters
n = 10000
k = 20
q = 20

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
        # make the hypergraph
        H = xgi.uniform_HPPM(n, 2, rho, k, epsilon2) << xgi.uniform_HPPM(
            n, 3, rho, q, epsilon3
        )

        community1 = set(list(H.nodes)[: int(H.num_nodes * rho)])
        community2 = set(list(H.nodes)[int(H.num_nodes * (1 - rho)) :])
        mean_link_degree = H.nodes.degree(order=1).mean()
        mean_triangle_degree = H.nodes.degree(order=2).mean()

        beta2c = gamma / mean_link_degree
        beta3c = gamma / mean_triangle_degree

        beta2 = beta2tilde * beta2c
        beta3 = beta3tilde * beta3c

        beta = {2: beta2, 3: beta3}
        arglist.append(
            (
                H,
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

fixed_points = HypergraphContagion.get_x1_x2_in_parallel(arglist, num_processes)

data["gamma"] = gamma
data["beta2"] = beta2tilde
data["beta3"] = beta3tilde
data["epsilon2"] = epsilon2
data["epsilon3"] = epsilon3
data["fixed-points"] = fixed_points

datastring = json.dumps(data)

with open(f"Data/polarization/{rho}-{epsilon2}.json", "w") as output_file:
    output_file.write(datastring)
