# Welcome to pynumpm v0.2

Welcome to pynumpm, short for PyNumericalPhasematching! 

This package is meant to help you with the simulation of the phasematching spectrum of nonlinear processes, in 
particular of collinear three wave mixing processes. It can numerically calculate the phasematching of ideal waveguides
and of waveguides with variable profile along their propagation axes. Moreover, it can calculate the phasematching
as a function of the wavevector mismatch $\Delta\beta$, of one scanning wavelength or of two scanning wavelength. 
This last functionality can be used for the calculation of *joint spectral amplitude* and *intensity* spectra and 
properties.

**Notes**
* All physical quantities in this package are expressed in SI units (m, s, Hz).
* In this package, the wavelengths are usually denoted as *red*, *green* and *blue* (or *r*, *g*, *b*). This implies 
that the blue field has shortest wavelength and the red has the longest wavelength. In case two fields have the same
wavelength, then the names are interchangeable.
* The functions describing the refractive indices, when used, must follow the convention of the standard Sellmeier's 
equation, i.e. they **must** accept the wavelength in $\mu$ m    

## Getting started

### Prerequisites
pynumpm has been written for Python3, but this release should still work with Python2. 
Support for Python2 is not provided.

pynumpm requires the following python packages:
    
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

Alternatively, it is possible to install the package by downloading the package from [GitHub](https://github.com/mattsantand/pynumericalphasematching)
and, in the folder *pynumericalphasematching*, run

`python setup.py install`

## Examples 

The basic steps to run a simulation, are:

1. Create a `waveguide` object, using a suitable class from the `waveguide` module.
2. Create a `phasematching` object, using a suitable class from the `phasematching` module and loading the `waveguide` 
object into such object.
3. Run the `calculate_phasematching()` method of the phasematching object to calculate the phasematching spectrum.

Here, the first steps for setting up a simple simulation are described. For more information and tutorial, check the 
[documentation](https://pynumericalphasematching.readthedocs.io/en/latest/). 

### First steps
The next code creates an ideal waveguide with length L = 10mm and a poling period of 4.4$\mu$ m.
```python
import numpy as np
from pynumpm import waveguide

thiswaveguide = waveguide.Waveguide(length = 1e-2,
                                    poling_period = 4.4e-6)                                          
``` 

The following lines load the `Waveguide` object created in the previous step into a `PhasematchingDeltaBeta` object and 
calculate the phasematching spectrum as a function of $\Delta\beta$
```python  

deltabeta = np.linspace(-5000, 5000, 1000)
thisprocess = phasematching.PhasematchingDeltaBeta(waveguide=thiswaveguide)
thisprocess.deltabeta = deltabeta
thisprocess.calculate_phasematching()
thisprocess.plot()
```

For more example, consult the Tutorials in the documentation.


## In development/ToDo list

* TODO: replace rectangular integration in Phasematching 1D and 2D with the sinc (the correct integration)
* Insert testing module
* Insert calculation of phasematched processes
* Insert bandwidth estimation (for 1D functions)
* TODO: Use FFT to calculate Simple 1D and 2D phasematching with user defined nonlinear profile
  (introduce in version 1.1).


## Documentation
The complete documentation of this package can be found [here](https://pynumericalphasematching.readthedocs.io/en/latest/).


## Author

* [Matteo Santandrea](mailto:mattsantand@gmail.com), University of Paderborn, [Integrated Quantum Optics](https://physik.uni-paderborn.de/silberhorn/) group

## License 

This project is licensed under the GNU GPLv3+ License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgment
* Benjamin Brecht, University of Paderborn (`Pump()` class)
* Marcello Massaro, University of Paderborn. Help with docstring, documentation, VCS, and cleaning up my messy work.