"""

.. module: examples.py
.. moduleauthor: Matteo Santandrea <matteo.santandrea@upb.de>

Examples to use the PyNumericalPhasematching package.
"""

import numpy as np
import matplotlib.pyplot as plt


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
    thisprocess = phasematching.PhasematchingDeltaBeta(waveguide = thiswaveguide)
    thisprocess.calculate_phasematching(deltabeta=deltabeta,
                                        normalized=True)
    thisprocess.plot()


if __name__ == '__main__':
    # example_waveguide()
    # example_noise()
    example_phasematching_deltabeta()
    plt.show()