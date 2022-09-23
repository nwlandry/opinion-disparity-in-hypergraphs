import json
import os
import numpy as np
import xgi

import HypergraphContagion

main_folder = os.getcwd()
data_folder = "Data"
dataset_folder = "SBM"

is_verbose = True
num_processes = len(os.sched_getaffinity(0))

# Epidemic parameters
gamma = 1
tmax = 100
fraction_to_average = 0.1

n = 100
m = 100
num_sims = 1

beta2tilde = 0.2
beta3tilde = 4

output_filename = "empirical_epsilon2_epsilon3_polarization.json"

with open(
    os.path.join(main_folder, data_folder, dataset_folder, "hypergraphs", "epsilon_values.json")
) as file:
    data = json.loads(file.read())
epsilon2 = data["epsilon2"]
epsilon3 = data["epsilon3"]

data = dict()
arglist = list()

for e2 in epsilon2:
    for e3 in epsilon3:
        fname = os.path.join(
            main_folder, data_folder, dataset_folder, "hypergraphs", f"{e2}-{e3}.json"
        )
        H = xgi.read_hypergraph_json(fname)
        community1 = set(list(H.nodes)[: int(H.num_nodes / 2)])
        community2 = set(list(H.nodes)[int(H.num_nodes / 2) :])
        mean_link_degree = H.nodes.degree(order=1).mean()
        mean_triangle_degree = H.nodes.degree(order=2).mean()

        beta2c = gamma / mean_link_degree
        beta3c = gamma / mean_triangle_degree

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
                tmax,
                fraction_to_average,
                num_sims,
                is_verbose,
            )
        )
polarization = HypergraphContagion.get_polarization_in_parallel(
    arglist, num_processes
)
polarization = np.reshape(polarization, [n, m], order="C")

data["gamma"] = gamma
data["beta2"] = beta2
data["beta3"] = beta3
data["epsilon2"] = epsilon2
data["epsilon3"] = epsilon3
data["polarization"] = polarization

datastring = json.dumps(data)

with open(
    os.path.join(main_folder, data_folder, dataset_folder, output_filename)
) as output_file:
    output_file.write(datastring)
