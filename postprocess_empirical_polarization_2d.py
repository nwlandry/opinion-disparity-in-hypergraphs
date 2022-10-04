import json
import shelve

import numpy as np

input_filename = "Data/stability/polarization.json"
output_filename = "Data/stability/empirical_polarization.json"

with open(input_filename) as file:
    data = json.load(file)
    epsilon2 = np.array(data["epsilon2"], dtype=float)
    epsilon3 = np.array(data["epsilon3"], dtype=float)

    psi = np.zeros((len(epsilon2), len(epsilon3)))
    for ii in range(len(epsilon2)):
        e2 = epsilon2[ii]
        for jj in range(len(epsilon3)):
            e3 = epsilon3[jj]
            polarization = np.array(data["polarization-" + str(e2) + "-" + str(e3)], dtype=float)
            psi[ii, jj] = np.max(polarization)

data = dict()
data["epsilon2"] = epsilon2.tolist()
data["epsilon3"] = epsilon3.tolist()
data["psi"] = psi.tolist()

datastring = json.dumps(data)

with open(output_filename, "w") as output_file:
    output_file.write(datastring)
