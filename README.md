# polarizability-of-hypergraphs
 
This repository accompanies the publication "The polarizability of hypergraphs with community structure" and provides all scripts necessary to reproduce all results and figures.

The contents of this repository are organized by directory.

## Top-level (no directory)

Plotting scripts:
* *plot_community_visualization.ipynb*: This notebook generates Fig. 1.
* *plot_empirical_polarization.ipynb*: This notebook generates the subplots in the left column of Fig. 3 from data in the "Data/polarization" folder.
* *plot_mean-field_polarization.ipynb*: This notebook generates the subplots in the right column of Fig. 3 from data in the "Data/polarization" folder as well as some of the subplots in Fig. 6.
* *plot_imbalanced_beta2c.ipynb*: This notebook generates Fig. 5.
* *plot_polarization_boundaries*: This notebook generates the subplots for Fig. 4.

Mean-field polarization:
* *mean-field_polarization.jl*: This script outputs JSON files of the polarization for (a) fixed $\widetilde{\beta}_2$ and $\widetilde{\beta}_3$ (varying $\epsilon_2$ and $\epsilon_3$) and (b) fixed $\epsilon_2$ and $\epsilon_3$ (varying $\widetilde{\beta}_2$ and $\widetilde{\beta}_3$).
* *mean-field_imbalanced_polarization.jl*: This script outputs a JSON file of $\psi_1$ and $\psi_2$ with respect to $\rho$ and $\epsilon_2$.
* *mean-field_polarization_boundaries.jl*: This script outputs JSON files of the polarization with respect to $\widetilde{\beta}_2$ and $\widetilde{\beta}_3$ for different values of (a) $\epsilon_2$ (holding $\epsilon_3$ fixed) and (b) $\epsilon_3$ (holding $\epsilon_2$ fixed).

Microscopic simulation:
* *SIS_vs_beta2_beta3.py*: This script runs the stochastic hypergraph SIS model on empirical datasets (generated with *generate_SBM_hypergraphs) for fixed $\epsilon_2$ and $\epsilon_3$ (varying $\widetilde{\beta}_2$ and $\widetilde{\beta}_3$) and outputs each to a JSON file.
* *SIS_vs_epsilon2_epsilon3.py*: This script runs the stochastic hypergraph SIS model on empirical datasets (created with *generate_SBM_hypergraphs.py*) for fixed $\widetilde{\beta}_2$ and $\widetilde{\beta}_3$ (varying $\epsilon_2$ and $\epsilon_3$) and outputs each to a JSON file.

Synthetic data:
* *generate_SBM_hypergraphs.py*: This script saves JSON files of synthetic m-HPPM hypergraphs with different $\epsilon_2$ and $\epsilon_3$ values.

## src
This directory contains all of the modules used in the scripts in the top-level directory.

* *GenerativeModels.py*: This module contains functions for generating m-HSBM hypergraphs, m-HPPM hypergraphs, and m-uniform Erdős–Rényi hypergraphs.
* *HypergraphContagion.py*: This module contains functions for simulating the SIS model on hypergraphs and outputting the polarization.
* *polarization.jl*: This module contains functions for computing the mean-field polarization for both the balanced and imbalanced cases.

## Mathematica
This directory contains all the notebooks to produce all phase plots and compute the epidemic thresholds.

* *beta2c_and_jacobian.nb*: This contains the calculations of the epidemic threshold for the balanced and imbalanced cases and the computation of the Jacobian matrices presented in Appendix C.
* *phase_plots.nb*: This generates phase diagrams for the balanced (Fig. 2) and imbalanced cases (Fig. 6).

## Data
This folder contains all the data used to generate the figures in the article.

### polarization

Most of these files contain the following fields: "gamma" (healing rate), "beta2" (pairwise infection rate), "beta3" (triangle infection rate), "epsilon2" (link community imbalance), "epsilon3" (triangle community imbalance), "psi" (polarization), and "nu" (the spectral abscissa, when able to be computed).

* *empirical_epsilon2_epsilon3_polarization.json*: This is the data corresponding to the top left panel in Fig. 3.
* *mean-field_epsilon2_epsilon3_polarization.json*: This is the data corresponding to the top right panel in Fig. 3.
* *empirical_beta2_beta3_polarization.json*: This is the data corresponding to the bottom left panel in Fig. 3.
* *mean-field_beta2_beta3_polarization.json*: This is the data corresponding to the bottom right panel in Fig. 3.
* *mean-field_polarization_boundaries_epsilon2.json*: This is the data corresponding to Fig. 4a.
* *mean-field_polarization_boundaries_epsilon2.json*: This is the data corresponding to Fig. 4b.
* *mean-field_rho_epsilon2_polarization.json*: This is the data corresponding to Fig. 6.

### vis
* *vis[1, 2, 3].json* and *pos[1, 2, 3].json* are the hypergraph and nodal positions corresponding to the [top, middle, bottom] panel of Fig. 1 respectively.

## Figures
This directory contains every figure in the manuscript as well as the individual images combined to make each figure. Each figure is stored as a PDF and either an SVG or PNG.

## Extra
This provides scripts that are not used in the manuscript but may prove helpful.

## requirements

Install the Python dependencies necessary to run the code by running
```
pip install -r requirements/requirements.txt
```
from the top-level directory.

Julia dependencies to install:
* Distributed
* JSON
* IntervalArithmetic
* IntervalRootFinding
* StaticArrays
* LinearAlgebra