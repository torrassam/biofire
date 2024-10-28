# IDH emergence from a vegetation-fires model

This repository provides the code to model plant-fire relationship in the Mediterranean and Boreal communities, as in xxx.

DOI:

Cite as: 

## Summary
This program models the dynamics of communities in fire prone environments. It includes 3 vegetation types that have different flammability (L), fire response (R), growth rate (c) and mortality rate (m). The model represents the time series of fractional vegetation cover (b) of the 3 plant types. The deterministic succession of plant (Tilman - Ecology, 1994) is perturbed by stochastic fires. Fire is represented as a non-stationary Poisson process, with average fire return time, 'tf'. The average fire return time depends on plant cover and flammability, leading to a fire-vegetation feedback.

## Content

The code is provided in Python and in Fortran language.

### Python Code

Python/ folder contains:

- ```biofire_main.py```: main file for generating the plant communities and simulating the time series of vegetation cover and fires.
- ```biofire_tools.py```: file containing the functions and subroutines used in the main.
- ```biofire_setting.json```: json file to set the simulation settings and the parameter values.

General command line:
```python generation.py -settings_file "biofire_setting.json"```

### Fortran Code



## References and Contacts
This code has been developed within the manuscript: XXX

Please refer to the paper for the underlying assumptions and equations.

Authors: Matilde Torrassa and Gabriele Vissio

For more information please contact Matilde Torrassa (matilde.torrassa@cimafoundation.org) concerning the Python code and Gabriele Vissio (gabriele.vissio@cnr.it) concerning the Fortran code.
