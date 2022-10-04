import json
import os

import numpy as np
import xgi

from GenerativeModels import *

main_folder = os.getcwd()
data_folder = "Data"
dataset_folder = "SBM"

n = 10000
mean_link_degree = 20
mean_triangle_degree = 20
epsilon2 = np.round(np.linspace(0, 1, 41), decimals=3)
epsilon3 = np.round(np.linspace(0.8, 1, 41), decimals=3)

datastring = json.dumps({"epsilon2": epsilon2.tolist(), "epsilon3": epsilon3.tolist()})

with open(
    os.path.join(main_folder, data_folder, dataset_folder, "hypergraphs", "epsilon_values.json"), "w"
) as file:
    file.write(datastring)


for e2 in epsilon2:
    for e3 in epsilon3:
        edgelist = list()
        pairs = uniform_planted_partition_hypergraph(n, 2, mean_link_degree, e2)
        edgelist.extend(pairs)
        triangles = uniform_planted_partition_hypergraph(n, 3, mean_triangle_degree, e3)
        edgelist.extend(triangles)

        H = xgi.Hypergraph(edgelist)

        print(f"epsilon2={e2}, epsilon3={e3} completed", flush=True)

        xgi.write_hypergraph_json(
            H,
            os.path.join(
                main_folder,
                data_folder,
                dataset_folder,
                "hypergraphs",
                f"{e2}-{e3}.json",
            ),
        )
