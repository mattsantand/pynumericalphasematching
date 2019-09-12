======
How-to
======

1. Defining a waveguide
=======================

In :mod:`pynumpm.waveguide` two types of **waveguide** classes are provided:

* :class:`pynumpm.waveguide.Waveguide`, for the definition of an ideal waveguide;
* :class:`pynumpm.waveguide.RealisticWaveguide`, for the definition of a realistic waveguide.

Ideal waveguide
---------------
An ideal waveguide is defined using the class :class:`pynumpm.waveguide.Waveguide`.
To define an ideal waveguide, only its `length` and, if necessary, its `poling_period` are required.
The following block of code creates a 10mm-long, ideal waveguide with poling period :math:`\Lambda = 16\mu\mathrm{m}`.

::

    from pynumpm.waveguide import Waveguide

    idealwaveguide = Waveguide(length = 10e-3,
                               poling_period = 16e-6)



Inhomogeneous waveguide
-----------------------
A real, inhomogeneous waveguide is defined using the class :class:`pynumpm.waveguide.RealisticWaveguide`.
The following block of code creates a 15mm-long waveguide, with a 10 :math:`\mu\mathrm{m}` poling, and a with a
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

It is possible create waveguide structures with a profile having a specific noise spectrum. At the moment, `AWGN <https://en.wikipedia.org/wiki/Additive_white_Gaussian_noise>`_, `1/f <https://en.wikipedia.org/wiki/Pink_noise>`_
and `1/f2 <https://en.wikipedia.org/wiki/Brownian_noise>`_ noise spectra are available.

The following snipped of code creates an inhomogeneous waveguide with a depth profile characterised by a mean depth of
:math:`8\mu\mathrm{m}` and a 1/f noise of amplitude :math:`0.2\mu\mathrm{m}`

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

    # Define the poling structure based on the z-mesh by providing a vector with the same shape
    # of the z-mesh and containing only +1 and -1, indicating the orientation of the poling domains.
    # For simplicity, we build here a periodic poling with period equal to 0.2um. However, any
    # sequence is allowed.
    poling_structure = np.ones(shape=z.shape)
    poling_structure[::2] = -1
    realwaveguide.load_poling_structure(poling_structure)

2. Spectrum of an ideal waveguide
=================================
Once a *Waveguide* object is defined, it is possible to calculate its phasematching spectrum using one of the classes
provided in the module :mod:`pynumpm.phasematching`.
To calculate the spectrum of an ideal waveguide, use the classes defined as *Simple___* in conjunction with :class:`pynumpm.waveguide.Waveguide` objects.
Three types of functions are available to calculate the phasematching spectra:

* *PhasematchingDeltaBeta*, to calculate the spectrum as a function of the phase mismatch :math:`\Delta\beta`;
* *Phasematching1D*, to calculate the spectrum of a **three-wave mixing process** scanning one input wavelength and keeping other fixed;
* *Phasematchinbg2D*, to calculate the spectrum of a **three-wave mixing process** scanning two input wavelengths.

When calculating the spectra as a function of the wavelength, it is necessary to provide the dispersion relations of the
system. If the calculation is performed on a :class:`pynumpm.waveguide.RealisticWaveguide`, the dispersion relations must
depend also on the parameter describing the waveguide profile.

.. warning::
    When calculating the spectrum as a function of the wavelength, the dispersion functions :math:`n = n(\lambda)` must be provided.
    They must follow the conventions of Sellmeier equations, i.e. must accept the wavelength in :math:`\mu\mathrm{m}` (the API will convert automatically the units).

:math:`\Delta\beta` dependent
-----------------------------

The following block of code creates a 2cm-long ideal waveguide and calculate its spectrum as a function of :math:`\Delta\beta`,
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
    # normalized is set to True to have the phasematching bounded between [0,1]. If false,
    # the spectrum will scale with the waveguide length.
    phi = idealphasematching.calculate_phasematching(normalized=True)

    idealphasematching.plot()
    plt.show()


Wavelength dependent: 1D
------------------------
The following block of code creates a 2cm-long, ideal waveguide and calculates its phasematching spectrum for the sum-frequency
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
    thisprocess = phasematching.SimplePhasematching1D(waveguide=thiswaveguide,
                                                            n_red=ny,
                                                            n_green=nz,
                                                            n_blue=ny,
                                                            order=1)
    # Define the range for the scanning wavelength
    thisprocess.red_wavelength = np.linspace(red_wl0-red_span/2, red_wl0+red_span/2, 1000)
    thisprocess.green_wavelength = green_wl0
    # Calculate the phasematching spectrum
    thisprocess.calculate_phasematching()
    # Plot
    thisprocess.plot()
    plt.show()

Wavelength dependent: 2D
------------------------

The following block of code creates a 4cm-long, ideal waveguide and calculates its phasematching spectrum for the parametric
down conversion (PDC) process 775nm (TE) -> 1550nm(TE) + 1550nm(TM), with polarisation defined in parentheses.
The spectrum is calculated scannning the signal and idler fields at 1550nm within 10nm.
The function :func:`pynumpm.utils.calculate_poling_period` is used to compute the correct poling period for the central
wavelengths of the process.

::

    from pynumpm import waveguide, phasematching, utils
    import matplotlib.pyplot as plt

    length = 20e-3
    red_wl0 = 1550e-9
    red_span = 10e-9
    green_wl0 = 1550e-9
    green_span = 10e-9
    # Use the utilities module to calculate the poling period of the process
    poling_period = utils.calculate_poling_period(red_wl0, green_wl0, 0, nTE, nTM, nTE)
    print("The correct poling period is {0}".format(poling_period))

    # Define the waveguide
    thiswaveguide = waveguide.Waveguide(length=length,
                                        poling_period=poling_period)

    # Define the phasematching process
    thisprocess = phasematching.SimplePhasematching2D(waveguide=thiswaveguide,
                                                            n_red=nTE,
                                                            n_green=nTM,
                                                            n_blue=nTE,
                                                            order=1)
    # Define the range for the scanning wavelength
    thisprocess.red_wavelength = np.linspace(red_wl0 - red_span / 2,
                                             red_wl0 + red_span / 2,
                                             1000)
    thisprocess.green_wavelength = np.linspace(green_wl0 - green_span / 2,
                                               green_wl0 + green_span / 2,
                                               1000)
    # Calculate the phasematching spectrum
    thisprocess.calculate_phasematching()
    # Plot
    thisprocess.plot()
    plt.show()

3. Spectrum of an inhomogeneous waveguide
=========================================
Passing a :class:`pynumpm.waveguide.RealisticWaveguide` object to a *Phasematching___* object, one can easily calculate
the phasematching spectrum of a custom-defined waveguide.

.. warning::
    The calculation of a wavelength-dependent spectrum requires the correct definition of the dispersion relation passed
    to the Phasematching object. The dispersion relations must be encoded as a function dependent on the variable describing
    the waveguide profile, returning the dispersion relation as a function of the wavelength, i.e.
    :math:`n = n(parameter)(\lambda)`.

.. warning::

    The dispersion as a function of :math:`\lambda` must follow the conventions of Sellmeier equations, i.e. must accept
    the wavelength in :math:`\mu\mathrm{m}` (the API will convert automatically the units).

:math:`\Delta\beta` dependent
-----------------------------
The following block of code creates a 2cm-long waveguide with a 1/f2 noise on the :math:`\Delta\beta` having a maximum amplitude
of :math:`\delta\beta_{max} = 100\mathrm{m}^{-1}` and calculates its spectrum in the range :math:`\Delta\beta\in[-5000, 5000] \mathrm{m}^{-1}`.

.. note::

    The calculation is performed assuming calculating the phasematching spectrum over a range :math:`\Delta\beta_0`,
    while the phasemismatch changes along the waveguide by a factor :math:`\delta\beta(z)`, i.e. :math:`\Delta\beta(z) = \Delta\beta_0 + \delta\beta(z)`.

.. note::

    Setting the `nominal_parameter=0` for the :class:`pynumpm.waveguide.RealisticWaveguide` ensures it to be phasematched.

::

    from pynumpm.waveguide import RealisticWaveguide
    from pynumpm.phasematching import PhasematchingDeltaBeta
    import matplotlib.pyplot as plt

    # Waveguide definition
    length = 20e-3
    z = np.linspace(0, length, 1000)
    thiswaveguide = RealisticWaveguide(z=z,
                                       nominal_parameter=0,
                                       nominal_parameter_name=r"$\Delta\beta$")
    thiswaveguide.create_noisy_waveguide(noise_profile="1/f2",
                                         noise_amplitude=100.0)
    thiswaveguide.plot()

    # Phasematching calculation

    thisprocess = PhasematchingDeltaBeta(waveguide=thiswaveguide)
    deltabeta = np.linspace(-5000, 5000, 1000)
    thisprocess.deltabeta = deltabeta
    thisprocess.calculate_phasematching(normalized=True)
    thisprocess.plot(verbose=True)
    plt.show()

Wavelength dependent: 1D
------------------------
The following block of code creates a 3cm-long waveguide and simulates the effects of a temperature inhomogeneity during the operation
of the system. The waveguide has an average temperature of :math:`40^\circ\mathrm{C}` and a 1/f noise with maximum amplitude
of :math:`3^\circ\mathrm{C}`.

The process is analogous to the one seen in section 2.

::

    from pynumpm import waveguide, phasematching, utils
    import matplotlib.pyplot as plt

    length = 30e-3  # length in m
    dz = 1e-6  # discretization in m
    z = np.arange(0, length + dz, dz)

    # Define the dispersion relations
    # n = n(parameter)(wavelength)
    nte, ntm = custom_sellmeier()

    # Define the process wavelengths
    red_wl0 = 1550e-9
    red_span = 20e-9
    green_wl0 = 890e-9

    # Calculate the poling period
    poling_period = utils.calculate_poling_period(red_wl0, green_wl0, 0, nte(40), ntm(40), nte(40), 1)
    print("The poling period is poling period: ", poling_period)

    # Define the waveguide
    thiswaveguide = waveguide.RealisticWaveguide(z=z,
                                                 poling_period=poling_period,
                                                 nominal_parameter=40,
                                                 nominal_parameter_name=r"Waveguide temperature [$^\circ$ C]")
    thiswaveguide.create_noisy_waveguide(noise_profile="1/f",
                                         noise_amplitude=3)
    thiswaveguide.plot_waveguide_properties()

    # Calculate the phasematching
    thisprocess = phasematching.Phasematching1D(waveguide=thiswaveguide,
                                                n_red=nte,
                                                n_green=ntm,
                                                n_blue=nte)
    thisprocess.red_wavelength = np.linspace(red_wl0-red_span/2,
                                             red_wl0+red_span/2,
                                             1000)
    thisprocess.green_wavelength = green_wl0
    phi = thisprocess.calculate_phasematching()
    thisprocess.plot()
    plt.show()


Wavelength dependent: 2D
------------------------

::

    from pynumpm import waveguide, utils, phasematching
    import matplotlib.pyplot as plt

    length = 25e-3  # length in m
    dz = 100e-6  # discretization in m

    nte, ntm = custom_sellmeier()
    T0 = 25

    poling_period = utils.calculate_poling_period(1.55e-6, 0, 0.55e-6, nte(T0), ntm(T0),
                                                  nte(T0), 1)
    print("Poling period: ", poling_period)
    z = np.arange(0, length + dz, dz)
    thiswaveguide = waveguide.RealisticWaveguide(z=z,
                                                 poling_period=poling_period,
                                                 nominal_parameter=T0,
                                                 nominal_parameter_name=r"WG temperature[$^\circ$C]")
    thiswaveguide.create_noisy_waveguide(noise_profile="1/f",
                                         noise_amplitude=1.0)
    thisprocess = phasematching.Phasematching2D(waveguide=thiswaveguide,
                                                n_red=nte,
                                                n_green=ntm,
                                                n_blue=nte)

    thisprocess.red_wavelength = np.linspace(1.50e-6, 1.6e-6, 100)
    thisprocess.blue_wavelength = np.linspace(0.549e-6, 0.551e-6, 1000)
    thisprocess.calculate_phasematching()
    thisprocess.plot()
    plt.show()


4. Definition of a pump spectrum
================================

5. JSA calculations
===================


6. Utilities
============

Phasematching point calculation
-------------------------------

Bandwidth calculation
---------------------
