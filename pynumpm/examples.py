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


def example_phasematching_deltabeta():
    from pynumpm import waveguide, phasematching

    z = np.linspace(0, 0.02, 1000)
    thiswaveguide = waveguide.RealisticWaveguide(z=z, nominal_parameter=0, nominal_parameter_name=r"$\Delta\beta$")
    thiswaveguide.create_noisy_waveguide(noise_profile="1/f",
                                         noise_amplitude=500.0)
    thiswaveguide.plot()

    deltabeta = np.linspace(-5000, 5000, 1000)
    thisprocess = phasematching.PhasematchingDeltaBeta(waveguide=thiswaveguide)
    thisprocess.deltabeta = deltabeta
    thisprocess.calculate_phasematching(normalized=True)
    thisprocess.plot(verbose=True)
    plt.show()


def example_simple1D_phasematching():
    from pynumpm import waveguide, phasematching, utils

    nte, ntm = custom_sellmeier()

    length = 10e-3
    poling = utils.calculate_poling_period(1550e-9, 890e-9, 0,
                                           ntm(20),
                                           ntm(20),
                                           ntm(20))
    print(poling)
    thissimplewaveguide = waveguide.Waveguide(length=length,
                                              poling_period=poling)
    thisprocess = phasematching.SimplePhasematching1D(waveguide=thissimplewaveguide,
                                                      n_red=ntm(20),
                                                      n_green=ntm(20),
                                                      n_blue=ntm(20))
    thisprocess.red_wavelength = np.linspace(1530, 1580, 1000) * 1e-9
    thisprocess.green_wavelength = 890e-9
    thisprocess.calculate_phasematching()
    thisprocess.plot()
    plt.show()


def example_1D_phasematching():
    from pynumpm import waveguide, phasematching, utils

    length = 30e-3  # length in m
    dz = 1e-6  # discretization in m
    z = np.arange(0, length + dz, dz)

    nte, ntm = custom_sellmeier()

    poling_period = utils.calculate_poling_period(1.55e-6, 890e-9, 0, nte(40), ntm(40), nte(40), 1)
    print("Poling period: ", poling_period)

    thiswaveguide = waveguide.RealisticWaveguide(z=z,
                                                 poling_period=poling_period,
                                                 nominal_parameter=40,
                                                 nominal_parameter_name=r"Wg width [$\mu$m]")
    thiswaveguide.create_noisy_waveguide(noise_profile="1/f",
                                         noise_amplitude=3)
    thiswaveguide.plot_waveguide_properties()
    thisprocess = phasematching.Phasematching1D(waveguide=thiswaveguide,
                                                n_red=nte,
                                                n_green=ntm,
                                                n_blue=nte)
    wl_red = np.linspace(1.540, 1.560, 1000) * 1e-6
    thisprocess.red_wavelength = wl_red
    thisprocess.green_wavelength = 890e-9
    thisprocess.set_nonlinearity_profile(profile_type="constant",
                                         first_order_coefficient=False)
    # thisprocess.set_wavelengths()
    phi = thisprocess.calculate_phasematching()
    thisprocess.plot()
    plt.show()


def example_test_load_wg():
    # TODO: put this function in a test module
    from pynumpm import waveguide, phasematching, utils

    simplewg = waveguide.Waveguide(length=1, poling_period=1)
    realwg = waveguide.RealisticWaveguide(z=np.array([0, 1]))
    simplepm = phasematching.SimplePhasematchingDeltaBeta(waveguide=simplewg)
    simplepm2 = phasematching.SimplePhasematchingDeltaBeta(waveguide=realwg)
    # realpm = phasematching.PhasematchingDeltaBeta(waveguide=simplewg)
    realpm2 = phasematching.PhasematchingDeltaBeta(waveguide=realwg)


def example_test_1DPM():
    # TODO: put this function in a test module
    from pynumpm import waveguide, phasematching, utils

    nte, ntm = custom_sellmeier()

    length = 20e-3
    poling = utils.calculate_poling_period(1550e-9, 890e-9, 0,
                                           nte(20),
                                           ntm(20),
                                           nte(20))
    thissimplewaveguide = waveguide.Waveguide(length=length,
                                              poling_period=poling)
    ideal_process = phasematching.SimplePhasematching1D(waveguide=thissimplewaveguide,
                                                        n_red=nte(20),
                                                        n_green=ntm(20),
                                                        n_blue=nte(20))
    ideal_process.red_wavelength = np.linspace(1530, 1580, 1000) * 1e-9
    ideal_process.green_wavelength = 890e-9
    phi1 = ideal_process.calculate_phasematching()

    dz = 50e-6  # discretization in m
    z = np.arange(0, length + dz, dz)

    thiswaveguide = waveguide.RealisticWaveguide(z=z,
                                                 poling_period=poling,
                                                 nominal_parameter=20,
                                                 nominal_parameter_name=r"Temperature")
    thisprocess = phasematching.Phasematching1D(waveguide=thiswaveguide,
                                                n_red=nte,
                                                n_green=ntm,
                                                n_blue=nte)

    thisprocess.red_wavelength = ideal_process.red_wavelength
    thisprocess.green_wavelength = ideal_process.green_wavelength
    phi2 = thisprocess.calculate_phasematching()
    plt.plot(ideal_process.red_wavelength * 1e9, abs(phi1) ** 2 - abs(phi2) ** 2, ":")
    plt.figure()
    plt.plot(ideal_process.red_wavelength * 1e9, np.imag(phi1), ":", color="tab:blue")
    plt.plot(ideal_process.red_wavelength * 1e9, np.imag(phi2), ":", color="tab:orange")
    plt.plot(ideal_process.red_wavelength * 1e9, np.real(phi1), "-", color="tab:blue")
    plt.plot(ideal_process.red_wavelength * 1e9, np.real(phi2), "-", color="tab:orange")
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


def example_simple2D_phasematching():
    from pynumpm import waveguide, utils, phasematching

    length = 25e-3  # length in m

    nte, ntm = custom_sellmeier()
    T0 = 25

    poling_period = utils.calculate_poling_period(1.55e-6, 0, 0.55e-6, nte(T0), ntm(T0), nte(T0), 1)
    print("Poling period: ", poling_period)
    thiswaveguide = waveguide.Waveguide(length=length,
                                        poling_period=poling_period)
    thisprocess = phasematching.SimplePhasematching2D(waveguide=thiswaveguide,
                                                      n_red=nte(T0),
                                                      n_green=ntm(T0),
                                                      n_blue=nte(T0))

    thisprocess.red_wavelength = np.linspace(1.50e-6, 1.6e-6, 100)
    thisprocess.blue_wavelength = np.linspace(0.549e-6, 0.551e-6, 1000)
    thisprocess.calculate_phasematching()
    thisprocess.plot()
    plt.show()


def example_2D_phasematching():
    from pynumpm import waveguide, utils, phasematching

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
    thispump.signal_wavelength = thisprocess.wavelength1
    thispump.idler_wavelength = thisprocess.wavelength2
    # set the bandwidth to 1nm
    thispump.pump_width = 1e-9
    thispump.plot()

    # load the pump and the phasematching to calculate the JSA
    thisjsa = jsa.JSA(phasematching=thisprocess,
                      pump=thispump)
    thisjsa.calculate_JSA()
    thisjsa.calculate_schmidt_decomposition()
    thisjsa.plot_schmidt_coefficients()
    thisjsa.plot(plot_pump=True)
    plt.show()


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
    # example_phasematching_deltabeta()
    example_simple1D_phasematching()
    example_1D_phasematching()
    example_test_1DPM()
    example_test_load_wg()
    example_1D_SFG()
    example_simple2D_phasematching()
    example_2D_phasematching()
    example_jsa1()
