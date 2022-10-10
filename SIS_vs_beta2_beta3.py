import json
import os
from fileinput import filename

import numpy as np
import xgi

import HypergraphContagion

main_folder = os.getcwd()
data_folder = "Data"
dataset_folder = "SBM"

# Epidemic parameters
gamma = 1
tmax = 100
fraction_to_average = 0.1
is_verbose = True
num_processes = len(os.sched_getaffinity(0))
num_sims = 1

output_filename = "empirical_beta2_beta3_polarization.json"

with open(
    os.path.join(main_folder, data_folder, dataset_folder, "hypergraphs", "epsilon_values.json")
) as file:
    data = json.loads(file.read())

n = 41
m = 41
e2 = 0.5
e3 = 0.95

epsilon2 = data["epsilon2"]
epsilon3 = data["epsilon3"]

if e2 not in epsilon2:
    raise Exception(f"{e2} not in epsilon2.")
if e3 not in epsilon3:
    raise Exception(f"{e3} not in epsilon3.")

data = dict()

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

beta2min = 0.0 * beta2c
beta2max = 0.5 * beta2c

beta3min = 3 * beta3c
beta3max = 6 * beta3c

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
    os.path.join(main_folder, data_folder, "stability", output_filename)
) as output_file:
    output_file.write(datastring)
