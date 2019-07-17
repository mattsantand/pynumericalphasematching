Tutorials
*********

If the dispersion functions (as a function of the wavelength) `n_red`, `n_green` and `n_blue` are available, it is
possible to calculate the phasematching as a function of one or two wavelength(s).

The next block of code calculates the poling period necessary for a sum frequency generation process with inputs at
1550nm and 890nm. Then it generates a 10mm long waveguide. Finally, it creates a Phasematching object that calculates the
phasematching of the process considering a fixed pump at 890nm and scanning the signal between 1530 and 1580 nm.
```python
import numpy as np
from pynumpm import waveguide, phasematching, utilities

poling = utilities.calculate_poling_period(1550e-9, 890e-9, 0,
                                           n_red,
                                           n_green,
                                           n_blue)[-1]
length = 10e-3
z = np.array([0, length])
thissimplewaveguide = waveguide.SimpleWaveguide(z=z,
                                                poling_period=poling)
thisprocess = phasematching.SimplePhasematching1D(waveguide=thissimplewaveguide,
                                                  n_red=n_red,
                                                  n_green=n_green,
                                                  n_blue=n_blue)
thisprocess.red_wavelength = np.linspace(1530, 1580, 1000) * 1e-9
thisprocess.green_wavelength = 890e-9
thisprocess.calculate_phasematching()
thisprocess.plot()
```
**Important note**

The function `n` describing the dispersion relation **must** follow these requirements:

1. When using a `SimplePhasematching1D` or `SimplePhasematching2D` object, `n` has the form `n=n(wl)`. It receives as inputs wavelengths in
\mu m (standard notation for Sellmeier equations) and returns the respective refractive index. E.g.
```python
def n(wl):
    """
    wl must be in \mu m
    """
    return A*wl + B*wl**2
```

2. When using a `Phasematching1D` or `Phasematching2D` object, `n` has the form `n=n(param)(wavelength)`, where `param`
is a generic parameter that influences the dispersion of the material (e.g. temperature or one fabrication parameter of
the nonlinear system). In particular, `n(param)` has to return a function that describes the refractive index as a
function of the wavelength, with the same convention as in point 1.

### More advanced capabilities
The next snippet creates a waveguide object with a poling period of 16um, a nominal width of 7um and 1/f
noise spectrum on its width, with a maximum deviation of 0.2um and plots its properties.
```python
import numpy as np
from pynumpm import waveguide

z_mesh = np.linspace(0, 10, 1000)*1e-3
thiswaveguide = waveguide.Waveguide(z=z_mesh,
                                    poling = 16e-6,
                                    nominal_parameter_name = "Width [$\mu$m]",
                                    nominal_parameter = 7.0e-6)
thiswaveguide.create_noisy_waveguide(noise_profile="1/f",
                                     noise_amplitude=0.2)
thiswaveguide.plot_waveguide_properties()
```

A second way to load the waveguide profile is by using the method `load_waveguide_profile`.
It is also possible to provide a custom poling configuration via the method `load_poling_structure`.
Please, note that the integration of custom poling hasn't been fully tested so it might be buggy.

#### Phasematching simulation: 1D, phase-mismatch-dependent phasematching

The following snippet loads the previously created waveguide in a `PhasematchingDeltaBeta` object and calculates
the phasematching for a given `deltabeta` range, being `deltabeta` the wavevector mismatch of the interacting fields.
Given this definition, this object is suitable to simulate any general phasematched system.
```python
from pynumpm import phasematching

deltabeta = np.linspace(-5000, 5000, 1000)
thisprocess = phasematching.PhasematchingDeltaBeta(waveguide=thiswaveguide)
thisprocess.calculate_phasematching(deltabeta=deltabeta)
thisprocess.plot()
```

#### Phasematching simulation: 1D, wavelength-dependent, three-wave mixing phasematching

The following snippet loads the previous waveguide into a phasematching object and calculates the 1D phasematching
spectrum for an SHG process pumped between 1540 and 1560nm and finally plots it.
```python
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

#### Phasematching simulation: 2D, wavelength-dependent, three-wave mixing phasematching

The following snippet loads the previous waveguide into a phasematching object and calculates the 1D phasematching
spectrum for a PDC process with signal and idler in the range (1540nm,1560nm) and (1200nm,1400nm) respectively.
```python
from pynumpm import phasematching

thisprocess = phasematching.Phasematching2D(waveguide=thiswaveguide,
                                            n_red=n_effective,
                                            n_green=n_effective,
                                            n_blue=n_effective)
wl_signal = np.linspace(1.540, 1.560, 1000) * 1e-6
wl_idler = np.linspace(1.2, 1.4, 1000)*1e-6
thisprocess.red_wavelength = wl_signal
thisprocess.green_wavelength = wl_idler
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
`green_wavelength` and `blue_wavelength` of the classe `Phasematching2D`.
It is necessary to define two wavelength ranges.
The `signal_wavelength` and the `idler_wavelength` are defined as the two input wavelength vectors, sorted in increasing
energy, i.e. `red_wavelength`<`green_wavelength`<`blue_wavelength`.

The definition of `signal_wavelength` and `idler_wavelength` is important to define the plotting x-axis in the `plot`
routine.
