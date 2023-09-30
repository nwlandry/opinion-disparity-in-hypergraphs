# Opinion disparity in hypergraphs with community structure
 
This repository accompanies the article ["Opinion disparity in hypergraphs with community structure"](https://doi.org/10.1103/PhysRevE.108.034311) by Nicholas Landry and Juan G. Restrepo and provides all scripts necessary to reproduce all results and figures.

The contents of this repository are organized by directory.

## Top-level (no directory)

Plotting scripts:
* *plot_illustrations.ipynb*: This notebook generates Figs. 1 and 2.
* *plot_balanced_opinion_disparity.ipynb*: This notebook generates Fig. 4 from data in the "Data/opiniondisparity" folder.
* *plot_opinion_disparity_boundaries*: This notebook generates the subplots for Fig. 5
* *plot_imbalanced_beta2c.ipynb*: This notebook generates Fig. 6.
* *plot_imbalanced_opinion_disparity.ipynb*: This notebook generates some of the subplots in Fig. 7 from data in the "Data/opiniondisparity" folder.

Mean-field opinion disparity:
* *mean-field_opinion_disparity.jl*: This script outputs JSON files of the opinion disparity for (a) fixed $\widetilde{\beta}_2$ and $\widetilde{\beta}_3$ (varying $\epsilon_2$ and $\epsilon_3$) and (b) fixed $\epsilon_2$ and $\epsilon_3$ (varying $\widetilde{\beta}_2$ and $\widetilde{\beta}_3$).
* *mean-field_imbalanced_opinion_disparity.jl*: This script outputs a JSON file of $\psi_1$ and $\psi_2$ with respect to $\rho$ and $\epsilon_2$.
* *mean-field_opinion_disparity_boundaries.jl*: This script outputs JSON files of the opinion disparity with respect to $\widetilde{\beta}_2$ and $\widetilde{\beta}_3$ for different values of (a) $\epsilon_2$ (holding $\epsilon_3$ fixed) and (b) $\epsilon_3$ (holding $\epsilon_2$ fixed).

Microscopic simulation:
* *SIS_vs_beta2_beta3.py*: This script runs the stochastic hypergraph SIS model on empirical datasets (generated with *generate_SBM_hypergraphs) for fixed $\epsilon_2$ and $\epsilon_3$ (varying $\widetilde{\beta}_2$ and $\widetilde{\beta}_3$) and outputs each to a JSON file.
* *SIS_vs_epsilon2_epsilon3.py*: This script runs the stochastic hypergraph SIS model on empirical datasets (created with *generate_SBM_hypergraphs.py*) for fixed $\widetilde{\beta}_2$ and $\widetilde{\beta}_3$ (varying $\epsilon_2$ and $\epsilon_3$) and outputs each to a JSON file.

Synthetic data:
* *generate_SBM_hypergraphs.py*: This script saves JSON files of synthetic m-HPPM hypergraphs with different $\epsilon_2$ and $\epsilon_3$ values.

## src
This directory contains all of the modules used in the scripts in the top-level directory.

* *GenerativeModels.py*: This module contains functions for generating m-HSBM hypergraphs, m-HPPM hypergraphs, and m-uniform Erdős–Rényi hypergraphs.
* *HypergraphContagion.py*: This module contains functions for simulating the SIS model on hypergraphs and outputting the opinion disparity.
* *opiniondisparity.jl*: This module contains functions for computing the mean-field opinion disparity for both the balanced and imbalanced cases.

## Mathematica
This directory contains all the notebooks to produce all phase plots and compute the epidemic thresholds.

* *beta2c_and_jacobian.nb*: This contains the calculations of the epidemic threshold for the balanced and imbalanced cases and the computation of the Jacobian matrices presented in Appendix C.
* *phase_plots.nb*: This generates phase diagrams for the balanced (Fig. 3) and imbalanced cases (Fig. 7).

## Data
This folder contains all the data used to generate the figures in the article.

### Opinion disparity

Most of these files contain the following fields: "gamma" (healing rate), "beta2" (pairwise infection rate), "beta3" (triangle infection rate), "epsilon2" (link community imbalance), "epsilon3" (triangle community imbalance), "psi" (opinion disparity), and "nu" (the spectral abscissa, when able to be computed).

* *empirical_epsilon2_epsilon3_opinion_disparity.json*: This is the data corresponding to the top left panel in Fig. 4.
* *mean-field_epsilon2_epsilon3_opinion_disparity.json*: This is the data corresponding to the top right panel in Fig. 4.
* *empirical_beta2_beta3_opinion_disparity.json*: This is the data corresponding to the bottom left panel in Fig. 4.
* *mean-field_beta2_beta3_opinion_disparity.json*: This is the data corresponding to the bottom right panel in Fig. 4.
* *mean-field_opinion_disparity_boundaries_epsilon2.json*: This is the data corresponding to Fig. 5a.
* *mean-field_opinion_disparity_boundaries_epsilon2.json*: This is the data corresponding to Fig. 5b.
* *mean-field_rho_epsilon2_opinion_disparity.json*: This is the data corresponding to Fig. 7.

### vis
* *vis[1, 2, 3].json* and *pos[1, 2, 3].json* are the hypergraph and nodal positions corresponding to the [top, middle, bottom] panel of Fig. 1 respectively.

## Figures
This directory contains every figure in the manuscript as well as the individual images combined to make each figure. Each figure is stored as a PDF and either an SVG or PNG.

## Extra
This provides scripts that are not used in the manuscript but may prove helpful. Not guaranteed to run.

## requirements

Install the Python dependencies necessary to run the code by running
```
pip install -r requirements.txt
```
from the top-level directory.

Julia dependencies to install:
* JSON
* IntervalArithmetic
* IntervalRootFinding
* StaticArrays
* LinearAlgebra
