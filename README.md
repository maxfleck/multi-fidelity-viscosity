# multi-fidelity-viscosity

Linear Multi-fidelity model based on Gaussian Process regression 
can be used to combine qualitative descriptioons known from simulations and quantitative accuracy from a few simulations to obtain something thats better than both on their own. 

Shear viscosities from a few molecular dynamics simulations as well as a few experimental shear viscosities enable a high-quality prediction of this transport property over a large range of thermodynamic state points.

The model relies on the univariate relationship between the reduced shear viscosity and the reduced residual entropy allowing the molecular simulation to extrapolate to regions not covered by the experimental training data.



builds on:
- [`FeOs`](https://github.com/vuejs/vue)
- [`GPy`](https://github.com/SheffieldML/GPy)
- [`emukit`](https://github.com/EmuKit/emukit)

## notebooks

### multi_fidelity_example.ipynb

Core of this repository. The entire approach can be reproduced using this notebook.

### poly_fit_reference.ipynb

Creates a reference based on all available data. Intended for comparison with the actual low-data approach.

## files

### entropy_scaling.py

Implementation of the core functions of our entropy scaling approach.

### multi_entropy_utils.py

Useful functions that build on entropy scaling and multi-fidelity approach.

## folders

### simulations

Contains all simulation results in a excel sheet

### butanol, propane, ...

Contains result plots and experimental data for the respective species
