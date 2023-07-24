import json
import os

import numpy as np
import xgi

from src import HypergraphContagion

# Epidemic parameters
gamma = 1
tmax = 100
fraction_to_average = 0.1
is_verbose = True
num_processes = len(os.sched_getaffinity(0))
num_sims = 1

output_filename = "empirical_beta2_beta3_polarization.json"

with open("Data/SBM/hypergraphs/epsilon_values.json") as file:
    data = json.loads(file.read())

n = 41
m = 41
e2 = 0.5
e3 = 0.95

beta2tilde = np.round(np.linspace(0, 0.5, n), decimals=4)
beta3tilde = np.round(np.linspace(3, 6, m), decimals=4)

if e2 not in data["epsilon2"]:
    raise Exception(f"{e2} not in epsilon2.")
if e3 not in data["epsilon3"]:
    raise Exception(f"{e3} not in epsilon3.")

data = dict()

fname = f"Data/SBM/hypergraphs/{e2}-{e3}.json"
H = xgi.read_hypergraph_json(fname)
community1 = set(list(H.nodes)[: int(H.num_nodes / 2)])
community2 = set(list(H.nodes)[int(H.num_nodes / 2) :])
mean_link_degree = H.nodes.degree(order=1).mean()
mean_triangle_degree = H.nodes.degree(order=2).mean()

beta2c = gamma / mean_link_degree
beta3c = gamma / mean_triangle_degree

beta2 = beta2tilde * beta2c
beta3 = beta3tilde * beta3c

arglist = list()
for b2 in beta2:
    for b3 in beta3:
        beta = {2: b2, 3: b3}
        arglist.append(
            (
                H,
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

psi = HypergraphContagion.get_polarization_in_parallel(arglist, num_processes)
psi = np.reshape(psi, [n, m], order="C")

data["gamma"] = gamma
data["beta2"] = beta2tilde.tolist()
data["beta3"] = beta3tilde.tolist()
data["epsilon2"] = e2
data["epsilon3"] = e3
data["psi"] = psi.tolist()

datastring = json.dumps(data)

with open(f"Data/polarization/{output_filename}", "w") as output_file:
    output_file.write(datastring)
