"""

.. module: examples.py
.. moduleauthor: Matteo Santandrea <matteo.santandrea@upb.de>

Examples to use the PyNumericalPhasematching package.
"""

import numpy as np
import matplotlib.pyplot as plt


def custom_sellmeier():
    """
    Temperature dependend sellmeier for 5% MgO-doped congruent lithium niobate (from wikipedia)
    :return:
    """

    def ne(temperature):
        f = (temperature - 24.5) * (temperature + 570.82)
        a1 = 5.756
        a2 = 0.0983
        a3 = 0.2020
        a4 = 189.32
        a5 = 12.52
        a6 = 1.32e-2
        b1 = 2.860e-6
        b2 = 4.7e-8
        b3 = 6.113e-8
        b4 = 1.516e-4
        return lambda wl: np.sqrt(a1 + b1 * f + (a2 + b2 * f) / (wl ** 2 - (a3 + b3 * f) ** 2) + (a4 + b4 * f) / (
                wl ** 2 - a5 ** 2) - a6 * wl ** 2)

    def no(temperature):
        f = (temperature - 24.5) * (temperature + 570.82)
        a1 = 5.5653
        a2 = 0.1185
        a3 = 0.2091
        a4 = 89.61
        a5 = 10.85
        a6 = 1.97e-2
        b1 = 7.941e-7
        b2 = 3.134e-8
        b3 = -4.641e-9
        b4 = 2.188e-6
        return lambda wl: np.sqrt(a1 + b1 * f + (a2 + b2 * f) / (wl ** 2 - (a3 + b3 * f) ** 2) + (a4 + b4 * f) / (
                wl ** 2 - a5 ** 2) - a6 * wl ** 2)

    return no, ne


def example_simple_waveguide():
    import pynumpm.waveguide as WG
    length = 1e-2
    wg = WG.Waveguide(length=length,
                      poling_period=10e-6)
    print(wg)


def tutorial_example_realistic_waveguide():
    from pynumpm.waveguide import RealisticWaveguide

    length = 15e-3
    poling = 10e-6

    # Mesh definition. Discretize the propagation axis with 100um resolution
    z = np.arange(0, length, 100e-6)
    realwaveguide = RealisticWaveguide(z=z,
                                       poling_period=poling,
                                       nominal_parameter_name="Waveguide width [$\mu$m]")

    # Define the waveguide profile as an array with the same shape as z
    profile = 0.5 * (2 * z / length - 1) ** 2 + 7.
    # Load the profile in the waveguide
    realwaveguide.load_waveguide_profile(profile)
    # Plot the profile for confirmation
    realwaveguide.plot()
    plt.show()


def tutorial_example_noisy_waveguide():
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


def tutorial_custom_poling():
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


def example_waveguide():
    """
    This function illustrates the use of the RealisticWaveguide module and the RealisticWaveguide class.

    """
    import pynumpm.waveguide as WG
    z = np.linspace(0, 0.020, 1000)
    uniform_waveguide = WG.RealisticWaveguide(z=z, poling_period=np.infty,
                                              nominal_parameter=7., nominal_parameter_name="Width")
    uniform_waveguide.plot()

    noisy_waveguide = WG.RealisticWaveguide(z=z, nominal_parameter=7., nominal_parameter_name="Width",
                                            poling_period=np.infty)
    noisy_waveguide.create_noisy_waveguide(noise_profile="awgn",
                                           noise_amplitude=0.1)
    noisy_waveguide.plot()
    noisy_waveguide = WG.RealisticWaveguide(z=z, nominal_parameter=7., nominal_parameter_name=r"Width [$\mu$m]",
                                            poling_period=np.infty)
    noisy_waveguide.create_noisy_waveguide(noise_profile="1/f",
                                           noise_amplitude=0.1)
    noisy_waveguide.plot()
    noisy_waveguide.plot_waveguide_properties()
    plt.show()


def example_noise():
    """
    Examples to create noise profiles

    """
    from pynumpm.noise import NoiseFromSpectrum, CorrelatedNoise, NoiseProfile
    z = np.linspace(0, 20, 10000)
    thisnoise = NoiseFromSpectrum(z=z, noise_amplitude=0.2, offset=0.4, profile_spectrum="awgn")
    thisnoise.plot_noise_properties()
    othernoise = CorrelatedNoise(z=z, noise_amplitude=0.2, correlation_length=0.2)
    othernoise.plot_noise_properties()
    concatenate_noise = NoiseProfile.concatenate(thisnoise, othernoise)
    concatenate_noise.plot_noise_properties()
    plt.show()


def example_simple_phasematching_deltabeta():
    from pynumpm import waveguide, phasematching

    thiswaveguide = waveguide.Waveguide(length=25e-3)
    deltabeta = np.linspace(-5000, 5000, 1000)
    thisprocess = phasematching.SimplePhasematchingDeltaBeta(waveguide=thiswaveguide)
    thisprocess.deltabeta = deltabeta
    thisprocess.calculate_phasematching(normalized=False)
    print(thisprocess.calculate_integral())
    thisprocess.plot()
    plt.show()


def tutorial_example_ideal_deltabeta():
    from pynumpm.waveguide import Waveguide
    from pynumpm.phasematching import SimplePhasematchingDeltaBeta

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


def tutorial_example_simple1dphasematching():
    NY, NZ = custom_sellmeier()
    nTE = lambda wl: NY(30)(wl)
    nTM = lambda wl: NZ(30)(wl)

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
                                                      n_red=nTE,
                                                      n_green=nTM,
                                                      n_blue=nTE,
                                                      order=1)
    # Define the range for the scanning wavelength
    thisprocess.red_wavelength = np.linspace(red_wl0 - red_span / 2, red_wl0 + red_span / 2, 1000)
    thisprocess.green_wavelength = green_wl0
    # Calculate the phasematching spectrum
    thisprocess.calculate_phasematching()
    # Plot
    thisprocess.plot()
    plt.show()


def tutorial_example_simple2dphasematching():
    NY, NZ = custom_sellmeier()
    nTE = lambda wl: NY(30)(wl)
    nTM = lambda wl: NZ(30)(wl)

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
    thisprocess.red_wavelength = np.linspace(red_wl0 - red_span / 2, red_wl0 + red_span / 2, 1000)
    thisprocess.green_wavelength = np.linspace(green_wl0 - green_span / 2, green_wl0 + green_span / 2, 1000)
    # Calculate the phasematching spectrum
    thisprocess.calculate_phasematching()
    # Plot
    thisprocess.plot()
    plt.show()


def tutorial_example_phasematching_deltabeta_realistic_wg():
    from pynumpm.waveguide import RealisticWaveguide
    from pynumpm.phasematching import PhasematchingDeltaBeta

    z = np.linspace(0, 0.02, 1000)
    thiswaveguide = RealisticWaveguide(z=z, nominal_parameter=0, nominal_parameter_name=r"$\Delta\beta$")
    thiswaveguide.create_noisy_waveguide(noise_profile="1/f2",
                                         noise_amplitude=500.0)
    thiswaveguide.plot()

    deltabeta = np.linspace(-5000, 5000, 1000)
    thisprocess = PhasematchingDeltaBeta(waveguide=thiswaveguide)
    thisprocess.deltabeta = deltabeta
    thisprocess.calculate_phasematching(normalized=True)
    thisprocess.plot(verbose=True)
    plt.show()


def tutorial_example_1Dphasematching():
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
    thisprocess.red_wavelength = np.linspace(red_wl0 - red_span / 2, red_wl0 + red_span / 2, 1000)
    thisprocess.green_wavelength = green_wl0
    phi = thisprocess.calculate_phasematching()
    thisprocess.plot()
    plt.show()


def example_1D_SFG():
    from pynumpm import waveguide, phasematching, utils

    length = 30e-3  # length in m
    dz = 50e-6  # discretization in m
    z = np.arange(0, length + dz, dz)

    nte, ntm = custom_sellmeier()

    poling_period = utils.calculate_poling_period(1.55e-6, 0, 0.55e-6, nte(20), ntm(20), nte(20), 1)
    print("Poling period: ", poling_period)

    thiswaveguide = waveguide.RealisticWaveguide(z=z,
                                                 poling_period=poling_period,
                                                 nominal_parameter=20,
                                                 nominal_parameter_name=r"Temperature [deg]")
    thiswaveguide.create_noisy_waveguide(noise_profile="1/f",
                                         noise_amplitude=0.1)
    thiswaveguide.plot_waveguide_properties()
    thisprocess = phasematching.Phasematching1D(waveguide=thiswaveguide,
                                                n_red=nte,
                                                n_green=ntm,
                                                n_blue=nte)
    wl_red = np.linspace(1.50, 1.6, 1000) * 1e-6
    thisprocess.red_wavelength = wl_red
    thisprocess.blue_wavelength = 0.55e-6
    thisprocess.set_nonlinearity_profile(profile_type="constant",
                                         first_order_coefficient=False)
    phi = thisprocess.calculate_phasematching()
    thisprocess.plot()
    plt.show()


def tutorial_example_2Dphasematching():
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


def example_jsa1():
    from pynumpm import waveguide, utils, phasematching, jsa

    length = 5e-3  # length in m

    nte, ntm = custom_sellmeier()
    T0 = 25

    poling_period = utils.calculate_poling_period(1.55e-6, 0, 0.55e-6, nte(T0), ntm(T0),
                                                  nte(T0), 1)
    print("Poling period: ", poling_period)
    z = np.array([0, length])
    thiswaveguide = waveguide.Waveguide(length=length,
                                        poling_period=poling_period)
    thisprocess = phasematching.SimplePhasematching2D(waveguide=thiswaveguide,
                                                      n_red=nte(T0),
                                                      n_green=ntm(T0),
                                                      n_blue=nte(T0))

    thisprocess.red_wavelength = np.linspace(1.50e-6, 1.6e-6, 500)
    thisprocess.blue_wavelength = np.linspace(0.549e-6, 0.551e-6, 1000)
    thisprocess.calculate_phasematching()
    thisprocess.plot()

    # the process is an SFG process
    thispump = jsa.Pump(process=jsa.Process.SFG)
    thispump.wavelength1 = thisprocess.wavelength1
    thispump.wavelength2 = thisprocess.wavelength2
    # set the bandwidth to 1nm
    thispump.pump_width = 1e-9
    thispump.plot()

    # load the pump and the phasematching to calculate the JSA
    thisjsa = jsa.JSA(phasematching=thisprocess,
                      pump=thispump)
    thisjsa.calculate_JSA()
    thisjsa.calculate_schmidt_decomposition()
    thisjsa.plot_schmidt_coefficients(ncoeff=20)
    thisjsa.plot(plot_pump=True)
    plt.show()


def tutorial_calculate_wavelengths():
    from pynumpm import utils

    nte, ntm = custom_sellmeier()
    success, res = utils.calculate_phasematching_point([549.5e-9, "b"], 4.55e-6, nte(20), ntm(20), nte(20), hint=[900e-9, 1550e-9])


if __name__ == '__main__':
    import logging

    FORMAT = "%(asctime)s.%(msecs)03d -- %(filename)s:%(lineno)s - %(funcName)20s() :>> %(message)s"
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    # example_simple_waveguide()
    # example_waveguide()
    # example_noise()
    # example_simple_phasematching_deltabeta()
    # tutorial_example_realistic_waveguide()
    # tutorial_example_noisy_waveguide()
    # tutorial_example_ideal_deltabeta()
    # tutorial_example_simple1dphasematching()
    # tutorial_example_simple2dphasematching()
    # tutorial_example_phasematching_deltabeta_realistic_wg()
    # tutorial_example_1Dphasematching()
    # tutorial_example_2Dphasematching()
    # example_jsa1()
    tutorial_calculate_wavelengths()
