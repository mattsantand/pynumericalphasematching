# Welcome to pynumpm

Welcome to pynumpm, short for PyNumericalPhasematching! 

This package is meant to help you with the simulation of the phasematching spectrum of nonlinear processes.  

## Getting started

### Prerequisites
pynumpm has been written for Python3, but this release should still work with Python2.

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

## Scope of PyNumericalPhasematching
The package *pynumpm* is useful to calculate the phasematching spectrum of a variety of nonlinear processes.

In this version, it can calculate 

* the phasematching spectrum of a general nonlinear process, as a function of the wavevector mismatch
* the phasematching spectrum of a three wave mixing process as a function of one or two independent wavelength(s)

in the presence of a variation of the phase mismatch along the propagation axes of the guiding system.     


It consists of five modules

* waveguide, helpful to describe the properties of the guiding system  
* noise, used to describe the nose properties of the guiding system
* phasematching, used to calculate the phasematching spectrum
* pump, used to generate a 2D pump spectrum to compute joint spectral amplitudes (JSA)
* utilities, with some common functions

### Examples 

To run a simulation, you need to
1. Create a waveguide object
2. Load the waveguide object into a phasematching object
3. Run the *calculate_phasematching* method of the phasematching object

#### Waveguide definition

For example, the next snippet creates a waveguide object with a poling period of 16um, a nominal width of 7um and 1/f 
noise spectrum on its width, with a maximum deviation of 0.2um and plots its properties.
```
import numpy as np
from pynumpm import waveguide

z_mesh = np.linspace(0, 10, 1000)*1e-3
thiswaveguide = waveguide.Waveguide(z=z_mesh,
                                    poling = 16e-6,
                                    nominal_parameter_name = "Width [$\mu$m]",
                                    nominal_parameter = 7.0e-6)
thiswaveguide.create_noisy_waveguide(noise_profile="1/f",
                                     noise_amplitude=0.2)
thiswaveguide.plot_waveguide_properties(set_multiplier_x=1e3)                                                                                                         
```  

A second way to load the waveguide profile is by using the method `load_waveguide_profile`.
It is also possible to provide a custom poling configuration via the method `load_poling_structure`. 
Please, note that the integration of custom poling hasn't been fully tested so it might be buggy.

#### Phasematching simulation: 1D, phase-mismatch-dependent phasematching

The following snippet loads the previously created waveguide in a `PhasematchingDeltaBeta` object and calculates
the phasematching for a given `deltabeta` range, being `deltabeta` the wavevector mismatch of the interacting fields.
Given this definition, this object is suitable to simulate any general phasematched system. 
```
from pynumpm import phasematching

deltabeta = np.linspace(-5000, 5000, 1000)
thisprocess = phasematching.PhasematchingDeltaBeta(waveguide=thiswaveguide)
thisprocess.calculate_phasematching(deltabeta=deltabeta) 
thisprocess.plot()
```

#### Phasematching simulation: 1D, wavelength-dependent phasematching 

The following snippet loads the previous waveguide into a phasematching object and calculates the 1D phasematching 
spectrum for an SHG process pumped between 1540 and 1560nm and finally plots it.
```
from pynumpm import phasematching

thisprocess = phasematching.Phasematching1D(waveguide=thiswaveguide,
                                            n_red=n_effective,
                                            n_green=n_effective,
                                            n_blue=n_effective)
wl_red = np.linspace(1.540, 1.560, 1000) * 1e-6
thisprocess.red_wavelength = wl_red
thisprocess.set_nonlinearity_profile(profile_type="constant",
                                     first_order_coefficient=False)
thisprocess.calculate_phasematching()
thisprocess.plot() 
```

Here, `n_effective` is a function describing the refractive index of the light fields as a function of the wavelength 
and of the variable waveguide parameter - in this case, the waveguide width. 
In particular, it **needs** to be defined such that `n(parameter)(wavelength)` returns a float (or array, 
depending on `wavelength`).

To define the wavelength range, you can directly access the wavelengths using the attributes `red_wavelength`, 
`green_wavelength` and `blue_wavelength` of the classe `Phasematching1D`.
The class automatically detects which kind of process you are considering depending on the following criteria: 

* If only one wavelength is defined, then it is considered a *SHG process*, 
i.e. `red_wavelength` == `green_wavelength` == `blue_wavelength`/2.
The `red_wavelength` is also assigned to `input_wavelength`, while `blue_wavelength` is also assigned to `output_wavelength`. 
* If two wavelengths are defined (one array and one float), then it is considered a *SFG/DFG process*. 
The input vector is assigned to `input_wavelength` while the dependent output vector is assigned 
to `output_wavelength`.

The definition of `input_wavelength` and `output_wavelength` is important to define the plotting x-axis in the `plot`
routine.


## Author

* Matteo Santandrea, University of Paderborn

## License 
his project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


## Acknowledgment
* Benjamin Brecht, University of Paderborn (*pump.py* module)
