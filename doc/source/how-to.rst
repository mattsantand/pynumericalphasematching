======
How-to
======

1. Defining a waveguide
====================

In :mod:`pynumpm.waveguide` two types of **waveguide** classes are provided:

* :class:`pynumpm.waveguide.Waveguide`, for the definition of an ideal waveguide;
* :class:`pynumpm.waveguide.RealisticWaveguide`, for the definition of a realistic waveguide.

Ideal waveguide
---------------
An ideal waveguide is defined using the class :class:`pynumpm.waveguide.Waveguide`.
To define an ideal waveguide, only its `length` and, if necessary, its poling period `poling_period` are required.
The next block of code creates a 10mm-long, ideal waveguide with poling period :math:`\Lambda = 16\mu\mathrm{m}`.

::

    from pynumpm.waveguide import Waveguide

    idealwaveguide = Waveguide(length = 10e-3,
                               poling_period = 16e-6)



Inhomogeneous waveguide
-----------------------
A real, inhomogeneous waveguide is defined using the class :class:`pynumpm.waveguide.RealisticWaveguide`.
The next block of code creates a 15mm-long waveguide, with a 10 :math:`\mu\mathrm{m}` poling, and a with a
parabolic width profile ranging between 7 and 7.5 :math:`\mu\mathrm{m}`.

::

    import numpy as np
    from pynumpm.waveguide import RealisticWaveguide

    length = 15e-3
    poling = 10e-6

    # Mesh definition. Discretize the propagation axis with 100um resolution
    z = np.arange(0, length, 100e-6)
    realwaveguide = RealisticWaveguide(z = z,
                                       poling_period = poling,
                                       nominal_parameter_name = "Waveguide width [$\mu$m]")

    # Define the waveguide profile as an array with the same shape as z
    profile = 0.5*(2*z/length-1)**2 + 7.
    # Load the profile in the waveguide
    realwaveguide.load_waveguide_profile(profile)
    # Plot the profile for confirmation
    realwaveguide.plot()

Adding noise
************

It is possible create waveguide structures with a profile having a specific noise spectrum. At the moment, Additive White Gaussian Noise
`AWGN <https://en.wikipedia.org/wiki/Additive_white_Gaussian_noise>`_, `1/f <https://en.wikipedia.org/wiki/Pink_noise>`_
 and `1/f2 <https://en.wikipedia.org/wiki/Brownian_noise>`_ are available.

The next snipped of code creates an inhomogeneous waveguide with a depth profile characterised by a mean depth of 8:math:`\mu\mathrm{m}`
and a 1/f noise of amplitude 0.2:math:`\mu\mathrm{m}`

::

    import numpy as np
    import matplotlib.pyplot as plt
    from pynumpm.waveguide import RealisticWaveguide

    length = 15e-3
    poling = 10e-6

    # Mesh definition. Discretize the propagation axis with 100um resolution
    z = np.arange(0, length, 100e-6)
    realwaveguide = RealisticWaveguide(z=z,
                                       poling_period=poling,
                                       nominal_parameter_name="Waveguide depth [$\mu$m]",
                                       nominal_parameter=8)
    # Create a noisy waveguide with a "1/f" noise spectrum and amplitude 0.2
    # This method accepts noise_profile equals to "awgn", "1/f" or "1/f2".
    realwaveguide.create_noisy_waveguide(noise_profile="1/f",
                                         noise_amplitude=0.2)
    # Plot the statistical properties of the waveguide
    realwaveguide.plot_waveguide_properties()
    plt.show()



Custom poling structure
***********************
It is possible to define a custom poling structure, with minimum feature size equal to the resolution of the mesh
employed to discretize the waveguide.

.. warning:: This feature is still under development.

::

    import numpy as np
    from pynumpm.waveguide import RealisticWaveguide

    length = 10e-3

    # Mesh definition. Discretize the propagation axis with 100um resolution
    z = np.arange(0, length, 100e-6)
    realwaveguide = RealisticWaveguide(z=z,
                                       nominal_parameter_name="Waveguide width [$\mu$m]",
                                       nominal_parameter=7)

    # Define the poling structure based on the z-mesh by providing a vector with the same shape of the z-mesh and
    # containing only +1 and -1, indicating the orientation of the poling domains.
    # For simplicity, we build here a periodic poling with period equal to 0.2um. However, any sequence is allowed.
    poling_structure = np.ones(shape=z.shape)
    poling_structure[::2] = -1
    realwaveguide.load_poling_structure(poling_structure)

2. Spectrum of an ideal waveguide
==============================
Once a *Waveguide* object is defined, it is possible to calculate its phasematching spectrum using one of the classes
provided in the module :mod:`pynumpm.phasematching`.
To calculate the spectrum of an ideal waveguide, use the classes defined as *Simple___* in conjunction with
:class:`pynumpm.waveguide.Waveguide` objects.
Three types of functions are available to calculate the phasematching spectra:

* *PhasematchingDeltaBeta*, to calculate the spectrum as a function of the phase mismatch :math:`\Delta\beta`;
* *Phasematching1D*, to calculate the spectrum of a **three-wave mixing process** scanning one input wavelength and
keeping other fixed;
* *Phasematchinbg2D*, to calculate the spectrum of a **three-wave mixing process** scanning two input wavelengths.

When calculating the spectra as a function of the wavelength, it is necessary to provide the dispersion relations of the
system. If the calculation is performed on a :class:`pynumpm.waveguide.RealisticWaveguide`, the dispersion relations must
depend also on the parameter describing the waveguide profile.

:math:`\Delta\beta` dependent
-----------------------------

The next block of code creates a 2cm-long ideal waveguide and calculate its spectrum as a function of :math:`\Delta\beta`,
for :math:`\Delta\beta\in [-1000, 1000] \mathrm{m}^{-1}`.

::

    from pynumpm.waveguide import Waveguide
    from pynumpm.phasematching import SimplePhasematchingDeltaBeta
    import matplotlib.pyplot as plt

    # Define the ideal waveguide
    length = 20e-3
    idealwaveguide = Waveguide(length=length)

    # Define the phasematching calculation, based on the waveguide object provided.
    idealphasematching = SimplePhasematchingDeltaBeta(waveguide=idealwaveguide)
    idealphasematching.deltabeta = np.arange(-1000, 1000, 1)

    # Perform the calculation.
    # normalized is set to True to have the phasematching bounded between [0,1]. If false, the spectrum will scale with
    # the waveguide length.
    phi = idealphasematching.calculate_phasematching(normalized=True)

    idealphasematching.plot()
    plt.show()



Wavelength dependent: 1D
------------------------
The next block of code creates a 2cm-long, ideal waveguide and calculates its phasematching spectrum for the sum-frequency
generation process 1550nm(TE) + 890nm(TM) -> 565.4nm(TE), with polarisation defined in parentheses. The spectrum is
calculated with the field at 890 fixed and the one at 1550nm scanned within 10nm.
The function :func:`pynumpm.utils.calculate_poling_period` is used to compute the correct poling period for the central
wavelengths of the process.

::

    from pynumpm import waveguide, phasematching, utils
    import matplotlib.pyplot as plt

    length = 20e-3
    red_wl0 = 1550e-9
    red_span = 10e-9
    green_wl0 = 890e-9
    # Use the utilities module to calculate the poling period of the process
    poling_period = utils.calculate_poling_period(red_wl0, green_wl0, 0, nTE, nTM, nTE)
    print("The correct poling period is {0}".format(poling_period))

    # Define the waveguide
    thiswaveguide = waveguide.Waveguide(length=length,
                                        poling_period=poling_period)

    # Define the phasematching process
    thisphasematching = phasematching.SimplePhasematching1D(waveguide=thiswaveguide,
                                                            n_red=ny,
                                                            n_green=nz,
                                                            n_blue=ny,
                                                            order=1)
    # Define the range for the scanning wavelength
    thisphasematching.red_wavelength = np.linspace(red_wl0-red_span/2, red_wl0+red_span/2, 1000)
    thisphasematching.green_wavelength = green_wl0
    # Calculate the phasematching spectrum
    thisphasematching.calculate_phasematching()
    # Plot
    thisphasematching.plot()
    plt.show()

Wavelength dependent: 2D
------------------------
The next block of code creates a 4cm-long, ideal waveguide and calculates its phasematching spectrum for the parametric
 down conversion (PDC) process 775nm (TE) -> 1550nm(TE) + 1550nm(TM), with polarisation defined in parentheses.
 The spectrum is calculated scannning the signal and idler fields at 1550nm within 10nm.
The function :func:`pynumpm.utils.calculate_poling_period` is used to compute the correct poling period for the central
wavelengths of the process.

3. Spectrum of an inhomogeneous waveguide
======================================

:math:`\Delta\beta` dependent
-----------------------------

Wavelength dependent: 1D
------------------------

Wavelength dependent: 2D
------------------------


4. Definition of a pump spectrum
=============================

5. JSA calculations
================


6. Utilities
=========

Poling period calculation
-------------------------

Phasematching point calculation
-------------------------------

Bandwidth calculation
---------------------
