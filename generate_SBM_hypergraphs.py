import json

import numpy as np
import xgi

from src.GenerativeModels import *

n = 10000
k = 20
q = 20
epsilon2 = np.round(np.linspace(0, 1, 41), decimals=4)
epsilon3 = np.round(np.linspace(0.5, 1, 41), decimals=4)

datastring = json.dumps({"epsilon2": epsilon2.tolist(), "epsilon3": epsilon3.tolist()})

with open("Data/SBM/hypergraphs/epsilon_values.json", "w") as file:
    file.write(datastring)


for e2 in epsilon2:
    for e3 in epsilon3:
        links = uniform_planted_partition_hypergraph(n, 2, k, e2)
        triangles = uniform_planted_partition_hypergraph(n, 3, q, e3)

        H = xgi.Hypergraph(links + triangles)

        print(f"epsilon2={e2}, epsilon3={e3} completed", flush=True)

        xgi.write_json(H, f"Data/SBM/hypergraphs/{e2}-{e3}.json")
