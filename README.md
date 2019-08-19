# Welcome to pynumpm v0.2

Welcome to pynumpm, short for PyNumericalPhasematching! 

This package is meant to help you with the simulation of the phasematching spectrum of nonlinear processes, in particular of collinear three wave mixing processes.

**Note**: all physical quantities in this package are expressed in SI units (m, s, Hz).

## Getting started

### Prerequisites
pynumpm has been written for Python3, but this release should still work with Python2. Support for Python2 is not provided.

Pynumpm requires the following python packages:
    
* pip
* [numpy](http://www.numpy.org/)    
* [matplotlib](https://matplotlib.org)
* [scipy](https://www.scipy.org/)
* warnings
* logging
   
The setup will automatically take care of installing the missing packages, when possible.   

## Installation
To install the package, simply run 

`pip install pynumpm`

Alternatively, it is possible to install the package by downloading the package from _____ and, in the folder *pynumericalphasematching*, run

`python setup.py install`

## Examples 

The basic steps to run a simulation, are:

1. Create a `waveguide` object, using a suitable class from the `waveguide` module.
2. Create a `phasematching` object, using a suitable class from the `phasematching` module and loading the `waveguide` 
object into such object.
3. Run the `calculate_phasematching()` method of the phasematching object to calculate the phasematching spectrum.

### First steps
The next code creates an ideal waveguide with length L = 10mm and a poling period of 4.4$\mu$ um.
```python
import numpy as np
from pynumpm import waveguide

thiswaveguide = waveguide.Waveguide(length = 1e-2,
                                    poling_period = 4.4e-6)                                          
``` 
A `Waveguide` object does contains only information about the geometry of the waveguide under consideration, namely its 
length, its poling period (can be set to `np.infinity` if not used) and, for a `RealisticWaveguide` object, its profile 
(e.g., the width variation along the propagation axis). 
Therefore, a `Waveguide` object does not contain information to evaluate the nonlinear process of interest. 
These information must be provided using a `Phasematching` object.

The following lines load the `Waveguide` object created in the previous step into a `Phasematching` object and calculate
the phasematching spectrum as a function of \Delta\beta
```python
import numpy as np
from pynumpm import phasematching
from pynumpm import waveguide

z = np.array([0, 0.01])
thiswaveguide = waveguide.Waveguide(length = 1e-2,
                                    poling_period = 4.4e-6)   
deltabeta = np.linspace(-5000, 5000, 1000)
thisprocess = phasematching.PhasematchingDeltaBeta(waveguide=thiswaveguide)
thisprocess.deltabeta = deltabeta
thisprocess.calculate_phasematching()
thisprocess.plot()
```

For more example, consult the Tutorials.


## In development/ToDo list

* TODO: replace rectangular integration in Phasematching 1D and 2D with the sinc (the currect integration)
* Insert testing module
* Insert calculation of phasematched processes
* Insert bandwidth estimation (for 1D functions)
* TODO: Use FFT to calculate Simple 1D and 2D phasematching with user defined nonlinear profile
  (introduce in version 1.1).

## Author

* [Matteo Santandrea](mailto:mattsantand@gmail.com), University of Paderborn.

## License 

This project is licensed under the GNU GPLv3+ License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgment
* Benjamin Brecht, University of Paderborn (`Pump()` class)
* Marcello Massaro, University of Paderborn. Help with docstring, documentation, VCS, and cleaning up my messy work.