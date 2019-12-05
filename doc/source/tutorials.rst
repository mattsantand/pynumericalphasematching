=========
Tutorials
=========

Here you can find some tutorials to understand the capabilities of this API.

.. warning:: This section is under rework

Effect of a parabolic gradient of the phase mismatch
****************************************************

Let's study the effect of a parabolic gradient of the phase mismatch for waveguides with different lengths.
Let's approximate :math:`\Delta\beta(z) = \Delta\beta_0 + \delta\beta(z)`, where :math:`\delta\beta(z)` is a parabolic
function.

::

    import numpy as np
    from pynumpm import waveguide, phasematching

    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)

    lengths = [1, 5, 10, 50]
    for length in lengths:
        z = np.linspace(0, length * 1e-3, 1000)
        thiswaveguide = waveguide.RealisticWaveguide(z=z,
                                                     nominal_parameter_name=r"$\Delta\beta$ [1/m]",
                                                     nominal_parameter=0)
        profile = 200 * (2 * z / z.max() - 1) ** 2
        thiswaveguide.load_waveguide_profile(profile)
        thiswaveguide.plot(ax=ax1)
        thisprocess = phasematching.PhasematchingDeltaBeta(waveguide=thiswaveguide)
        thisprocess.deltabeta = np.linspace(-2000, 2000, 1000)
        thisprocess.calculate_phasematching(normalized=True)
        thisprocess.plot(ax=ax2)

    plt.figure(fig1.number)
    plt.legend(lengths)
    plt.figure(fig2.number)
    plt.legend(lengths)
    plt.show()

In the previous code, we created a number of waveguides with different lengths. For each waveguide, we loaded its
profile :math:`\delta\beta(z)` along the propagation axis using the method :func:`pynumpm.waveguide.RealisticWaveguide.load_waveguide_profile`.
Next, we loaded in `thisprocess.deltabeta` the range of :math:`\Delta\beta` that we want to analyse.
Finally, with the method :func:`pynumpm.phasematching.SimplePhasematchingDeltaBeta.calculate_phasematching` we calculated
the phasematching for the different waveguides.

Effect of a random variation of the waveguide width
***************************************************

In the next code, we study the impact of random variations of the waveguide widths on the resulting phasematching,
as a function of one scanning wavelength, for an SFG process.
Here, we assume as waveguide width an additive white gaussian process with amplitude 0.2.

.. attention:: The simulation of processes with variable dispersion relations requires very specific inputs. In
               particular, the equations `n` specifying the variation of the dispersion as a function of the waveguide
               parameter must be in the form `n=n(param)(wl)`, i.e. `n(param)` must return a dispersion function
               `n(wl)`. Moreover, the function `n(wl)` must follow the convention of Sellmeier's equation, i.e. the
               input waveguide must be in :math:`\mu\mathrm{m}`.

.. note:: We assume that the correct Sellmeier have been loaded and stored in the variables `nx`, `ny`, `nz`.

::

    import numpy as np
    from pynumpm import waveguide, phasematching, utils

    length = 20 # 2cm
    nominal_width = 7 # waveguide width = 7um
    wl_red0 = 1550e-9
    wl_green0 = 800e-9
    span_red = 10e-9


    z = np.linspace(0, length,10000)*1e-3
    poling_period = utils.calculate_poling_period(wl_red0, wl_green0, 0,
                                                  ny(nominal_width), nz(nominal_width), ny(nominal_width))
    thiswaveguide = waveguide.RealisticWaveguide(z = z
                                                 poling_period = poling_period,
                                                 nominal_parameter = nominal_width,
                                                 nominal_parameter_name = "Width [$\mu$m]")
    thiswaveguide.create_noisy_waveguide(noise_profile="awgn",
                                         noise_amplitude=0.2)
    thiswaveguide.plot_waveguide_properties()

    thisprocess = phasematching.Phasematching1D(waveguide=waveguide
                                                n_red=ny,
                                                n_green=nz,
                                                n_blue=ny)
    # let's scan 10nm around 1550nm
    thisprocess.red_wavelength = np.linspace(wl_red0-span_red/2, wl_red0+span_red/2, 1000)
    thisprocess.green_wavelength = wl_green0
    thisprocess.calculate_phasematching()
    thisprocess.plot()
    plt.show()


Calculation of JSI properties
*****************************

In the following code, we calculate the phasematching spectrum of a parametric down conversion (PDC) process
and of its pulsed pump. Next, we calculate the joint spectral intensity of the process and evaluate its Schmidt mode
distribution, the Schmidt number and its purity.

For this simulation, we assume that the waveguide width is a 1/f random process with amplitude 0.1.

::

    import numpy as np
    from pynumpm import waveguide, phasematching, jsa, utils

    length = 20 # 2cm
    nominal_width = 7 # waveguide width = 7um
    wl_red0 = 1550e-9
    wl_green0 = 1550e-9
    span_red = 10e-9
    span_green = 10e-9


    z = np.linspace(0, length,10000)*1e-3
    poling_period = utils.calculate_poling_period(wl_red0, wl_green0, 0,
                                                  ny(nominal_width), nz(nominal_width), ny(nominal_width))
    thiswaveguide = waveguide.RealisticWaveguide(z = z
                                                 poling_period = poling_period,
                                                 nominal_parameter = nominal_width,
                                                 nominal_parameter_name = "Width [$\mu$m]")
    thiswaveguide.create_noisy_waveguide(noise_profile="1/f",
                                         noise_amplitude=0.1)
    thiswaveguide.plot_waveguide_properties()

    thisprocess = phasematching.Phasematching2D(waveguide=waveguide
                                                n_red=ny,
                                                n_green=nz,
                                                n_blue=ny)
    # let's scan 10nm around 1550nm
    thisprocess.red_wavelength = np.linspace(wl_red0-span_red/2, wl_red0+span_red/2, 1000)
    thisprocess.green_wavelength = np.linspace(wl_green0-span_green/2, wl_green0+span_green/2, 1000)
    thisprocess.calculate_phasematching()
    thisprocess.plot()

    thispump = jsa.Pump(process=jsa.Process.PDC)
    thispump.wavelength1 = thisprocess.wavelength1
    thispump.wavelength2 = thisprocess.wavelength2
    # set the bandwidth to 1nm
    thispump.pump_width = 1e-9
    thispump.plot()

    # load the pump and the phasematching to calculate the JSA
    thisjsa = jsa.JSA(phasematching=thisprocess,
                      pump=thispump)
    thisjsa.calculate_JSA()
    K, _, _ = thisjsa.calculate_schmidt_decomposition()
    print("This process has a Schmidt number K = {0}, corresponding to a purity of {1}".format(K, 1/K))
    thisjsa.plot_schmidt_coefficients(ncoeff=20)
    thisjsa.plot(plot_pump=True)
    plt.show()


Important notes
***************

The function `n` describing the dispersion relation **must** follow these requirements:

1.  When using a :class:`pynumpm.SimplePhasematching1D` or :class:`pynumpm.SimplePhasematching2D` object, `n` has the
    form `n=n(wl)`. It receives as inputs wavelengths in :math:`\mu\mathrm{m}`  (standard notation for Sellmeier equations)
    and returns the respective refractive index. E.g.

::

    def n(wl):
        """
        wl must be in microns
        """
        return A*wl + B*wl**2


2.  When using a `Phasematching1D` or `Phasematching2D` object, `n` has the form `n=n(param)(wavelength)`, where `param`
    is a generic parameter that influences the dispersion of the material (e.g. temperature or one fabrication parameter
    of the nonlinear system). In particular, `n(param)` has to return a function that describes the refractive index as
    a function of the wavelength, with the same convention as in point 1.
