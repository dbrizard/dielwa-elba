# dielwa-elba: Dispersion of Elastic Waves in Elliptical Bars using Fraser's collocation method

A python module to compute the dispersion of elastic waves in elliptical bars
using the collocation method according to the following article of Fraser:

Fraser, W. B. (1969). Dispersion of elastic waves in elliptical bars. 
*Journal of Sound and Vibration*, 10(2), 247‑260. 
https://doi.org/10.1016/0022-460X(69)90199-0

Also contains a module to compute the dispersion of longitudinal elastic waves 
in round bars (Pochhammer-Chree equation).

XX Put example here.

## Dispersion curves visualization

Two methods are available to visualize the dispersion curves:
1. compute the values of the characteristic function on the whole (K,C) domain of interest. Dispersion curves are then visualized with a contour plot (with level 0 being the solutions) or a sign plot (most changes of sign being the solutions).
2. numerical solving of the characteristic equation with a prediction-correction algorithm. Prediction uses basic polynomial extrapolation, correction uses Regula Falsi method. 


### Warnings

The second option is only possible if the starting of the curve is known, ie. only for the first branch of the longitudinal mode. Higher branches are harder to catch and follow automatically.

The first method requires some car in the interpretation of the diagrams. 
Depending on the order of approximation (ie number of collocation points), 
either the real part or the imaginary part of the characteristic function is of interest.


## Contents

This repository contains the following Python and Fortran files:
* `ellipticReferenceSolutions.py` is the module used to plot the dispersion curves of Fraser's article;
* `round_bar_pochhammer_chree.py` contains the mother class to handle characteristic equations (it is applied here to the dispersion of longitudinal waves in round bars, with Pochhammer-Chree equation);
* `elliptical_bar_fraser.py` contains the class used to handle Fraser's approximate equations.
* `fraser_matrix.f90` computes Fraser's characteristic matrix. It is used to speed up calculations (10x approximate speed-up compared to pure Python version)
* `special_functions.f90` contains Bessel functions, used in the previous file.

## Installation

1. Extract all the files in the same folder;
2. Compile `special_functions.f90`:
  * `gfortran -c special_functions.f90 -fPIC`
3. Compile `fraser_matrix.f90` with `f2py` to make the `fraser_matrix` module available within Python:
  * `f2py -c -I. special_functions.o -m fraser_matrix fraser_matrix.f90`
4. The Python files should now be usable !


## Documentation and usage
Sorry, this is a small project, read the docstrings !


## Testing
Well, you can use `ellipticReferenceSolutions.py` to plot Fraser's curves and
compare them with the dispersio curves computed with `elliptical_bar_fraser.py`.
