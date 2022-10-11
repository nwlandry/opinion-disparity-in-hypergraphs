import json
import os
from fileinput import filename

import numpy as np
import xgi

import HypergraphContagion

# Epidemic parameters
gamma = 1
tmax = 100
fraction_to_average = 0.1
is_verbose = True
num_processes = len(os.sched_getaffinity(0))

num_sims = 1

output_filename = "polarization.json"

with open("Data/SBM/hypergraphs/epsilon_values.json") as file:
    data = json.loads(file.read())
epsilon2 = data["epsilon2"]
epsilon3 = data["epsilon3"]
n = len(epsilon2)
m = len(epsilon3)

data = dict()

for e2 in epsilon2:
    for e3 in epsilon3:
        fname = f"Data/SBM/hypergraphs/{e2}-{e3}.json"
        H = xgi.read_hypergraph_json(fname)
        community1 = set(list(H.nodes)[: int(H.num_nodes / 2)])
        community2 = set(list(H.nodes)[int(H.num_nodes / 2) :])
        mean_link_degree = H.nodes.degree(order=1).mean()
        mean_triangle_degree = H.nodes.degree(order=2).mean()

        beta2c = gamma / mean_link_degree
        beta3c = gamma / mean_triangle_degree

        beta2min = 0.0 * beta2c
        beta2max = 1 * beta2c

        beta3min = 2 * beta3c
        beta3max = 10 * beta3c

        beta2 = np.linspace(beta2min, beta2max, n)
        beta3 = np.linspace(beta3min, beta3max, m)

        arglist = list()
        for b2 in beta2:
            for b3 in beta3:
                beta = {2: b2, 3: b3}
                arglist.append(
                    (
                        fname,
                        gamma,
                        beta,
                        community1,
                        community2,
                        tmax,
                        fraction_to_average,
                        num_sims,
                        is_verbose,
                    )
                )
        print(f"epsilon2={e2}, epsilon3={e3} started", flush=True)
        psi = HypergraphContagion.get_polarization_in_parallel(
            arglist, num_processes
        )
        psi = np.reshape(psi, [n, m], order="C")
        print(f"epsilon2={e2}, epsilon3={e3} completed", flush=True)

        data[f"gamma-{e2}-{e3}"] = gamma
        data[f"beta2-{e2}-{e3}"] = beta2
        data[f"beta3-{e2}-{e3}"] = beta3
        data[f"psi-{e2}-{e3}"] = psi

datastring = json.dumps(data)

with open(f"Data/stability/{output_filename}", "w") as output_file:
    output_file.write(datastring)
