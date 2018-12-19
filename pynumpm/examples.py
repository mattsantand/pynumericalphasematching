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


def example_waveguide():
    """
    This function illustrates the use of the Waveguide module and the Waveguide class.

    """
    import pynumpm.waveguide as WG
    z = np.linspace(0, 0.020, 1000)
    uniform_waveguide = WG.Waveguide(z=z, nominal_parameter=7., nominal_parameter_name="Width")
    uniform_waveguide.plot()

    noisy_waveguide = WG.Waveguide(z=z, nominal_parameter=7., nominal_parameter_name="Width")
    noisy_waveguide.create_noisy_waveguide(noise_profile="awgn",
                                           noise_amplitude=0.1)
    noisy_waveguide.plot()

    noisy_waveguide = WG.Waveguide(z=z, nominal_parameter=7., nominal_parameter_name="Width")
    noisy_waveguide.create_noisy_waveguide(noise_profile="1/f",
                                           noise_amplitude=0.1,
                                           nominal_parameter=3.0)
    noisy_waveguide.plot()
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


def example_phasematching_deltabeta():
    from pynumpm import waveguide, phasematching

    z = np.linspace(0, 0.02, 1000)
    thiswaveguide = waveguide.Waveguide(z=z, nominal_parameter=0, nominal_parameter_name=r"\Delta\beta")
    thiswaveguide.create_noisy_waveguide(noise_profile="1/f",
                                         noise_amplitude=500.0)
    thiswaveguide.plot()

    deltabeta = np.linspace(-5000, 5000, 1000)
    thisprocess = phasematching.PhasematchingDeltaBeta(waveguide=thiswaveguide)
    thisprocess.calculate_phasematching(deltabeta=deltabeta,
                                        normalized=True)
    thisprocess.plot(verbose=True)


def example_1D_phasematching():
    from pynumpm import waveguide, phasematching, utilities

    length = 30e-3  # length in m
    dz = 50e-6  # discretization in m
    z = np.arange(0, length + dz, dz)

    nte, ntm = custom_sellmeier()

    poling_period = utilities.calculate_poling_period(1.55e-6, 1.55e-6, 0, ntm(40), ntm(40), ntm(40), 1)[-1]
    print("Poling period: ", poling_period)

    thiswaveguide = waveguide.Waveguide(z=z,
                                        poling_period=poling_period,
                                        nominal_parameter=40,
                                        nominal_parameter_name=r"Wg width [$\mu$m]")
    thiswaveguide.create_noisy_waveguide(noise_profile="1/f",
                                         noise_amplitude=3)
    thisprocess = phasematching.Phasematching1D(waveguide=thiswaveguide,
                                                n_red=ntm,
                                                n_green=ntm,
                                                n_blue=ntm)
    wl_red = np.linspace(1.540, 1.560, 1000) * 1e-6
    thisprocess.red_wavelength = wl_red
    # thisprocess.set_wavelengths(wl_red, wl_red, 0, constlam="shg")
    thisprocess.set_nonlinearity_profile(profile_type="constant",
                                         first_order_coefficient=False)
    phi = thisprocess.calculate_phasematching()
    thisprocess.plot()


def example_1D_SFG():
    from pynumpm import waveguide, phasematching, utilities

    length = 30e-3  # length in m
    dz = 50e-6  # discretization in m
    z = np.arange(0, length + dz, dz)

    nte, ntm = custom_sellmeier()

    poling_period = utilities.calculate_poling_period(1.55e-6, 0, 0.55e-6, nte(40), ntm(40), nte(40), 1)[-1]
    print("Poling period: ", poling_period)

    thiswaveguide = waveguide.Waveguide(z=z,
                                        poling_period=poling_period,
                                        nominal_parameter=40,
                                        nominal_parameter_name=r"Wg width [$\mu$m]")
    thiswaveguide.create_noisy_waveguide(noise_profile="1/f",
                                         noise_amplitude=0)
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


def example_2D_phasematching():
    from pynumpm import waveguide, utilities, phasematching, pump

    length = 20e-3  # length in m
    dz = 100e-6  # discretization in m

    nte, ntm = custom_sellmeier()
    T0 = 25

    lamr, lamg, lamb, poling_period = utilities.calculate_poling_period(1.55e-6, 0, 0.55e-6, nte(T0), ntm(T0),
                                                                        nte(T0), 1)
    print("Poling period: ", poling_period)
    z = np.arange(0, length + dz, dz)
    thiswaveguide = waveguide.Waveguide(z=z,
                                        poling_period=poling_period,
                                        nominal_parameter=T0,
                                        nominal_parameter_name=r"WG temperature[$^\circ$C]")
    # thiswaveguide.create_noisy_waveguide(noise_profile="1/f",
    #                                      noise_amplitude=3)
    thisprocess = phasematching.Phasematching2D(waveguide=thiswaveguide,
                                                n_red=nte,
                                                n_green=ntm,
                                                n_blue=nte,
                                                process="sfg")
    thisprocess.set_red(central_wavelength=1.55e-6,
                        delta_lambda=20e-9,
                        n_points=100)
    thisprocess.set_blue(central_wavelength=0.55e-6,
                         delta_lambda=1e-9,
                         n_points=100)
    thisprocess.calculate_phasematching()
    thisprocess.plot_phasematching()

    pump = pump.Pump(process="SFG")
    signal_wl = thisprocess.red_wavelength
    idler_wl = thisprocess.blue_wavelength
    SIG, ID = np.meshgrid(signal_wl, idler_wl)
    pump_center = lamg
    pump_width = 2E-9
    pump.signal_wavelength = SIG
    pump.idler_wavelength = ID
    pump.pump_center = pump_center
    pump.pump_width = pump_width
    p = pump.pump()
    res = pump.pump()
    res /= (abs(res) ** 2).max()
    plt.figure()
    plt.imshow(abs(res) ** 2)
    plt.colorbar()
    thisprocess.calculate_JSA(pump=pump)
    thisprocess.plot_JSI()
    print("K: ", thisprocess.calculate_schmidt_number(verbose=True))


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    # example_waveguide()
    # example_noise()
    # example_phasematching_deltabeta()
    # example_1D_phasematching()
    example_1D_SFG()
    # example_2D_phasematching()
    plt.show()
