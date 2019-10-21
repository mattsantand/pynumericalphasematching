import numpy as np
import matplotlib.pyplot as plt
from pynumpm import waveguide, noise, phasematching, jsa, utils


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


def test_waveguide():
    ideal_waveguide = waveguide.Waveguide(length=0.01,
                                          poling_period=16e-6)
    print(ideal_waveguide)

    z = np.linspace(0, 10, 1000) * 1e-3
    poling = -4e-6
    real_waveguide = waveguide.RealisticWaveguide(z=z,
                                                  poling_period=poling,
                                                  nominal_parameter=7.,
                                                  nominal_parameter_name="Width")
    print(real_waveguide)
    real_waveguide.plot_waveguide_properties()
    plt.show()


def test_noise():
    # From spectral distribution
    z = np.linspace(0, 10, 1000) * 1e-3
    noise1 = noise.NoiseFromSpectrum(z=z,
                                     offset=-2,
                                     noise_amplitude=-0.1,
                                     profile_spectrum="1/f")
    noise2 = noise.CorrelatedNoise(z=z,
                                   offset=-2,
                                   noise_amplitude=0.1,
                                   correlation_length=4e-3)
    print(noise1)
    print(noise2)
    fig, _ = noise1.plot_noise_properties(ls="--", lw=1, color="tab:blue")
    noise2.plot_noise_properties(fig=fig, ls="-.", lw=1, color="tab:orange")

    concatenated_noise = noise1.concatenate(noise2)
    concatenated_noise2 = noise2.concatenate(noise1)
    sum_noise = noise1 + noise2
    fig2, ax = concatenated_noise.plot_noise_properties()
    concatenated_noise2.plot_noise_properties(fig=fig2)
    sum_noise.plot_noise_properties()

    plt.show()


def test_simple_phasematching():
    ny, nz = custom_sellmeier()
    thiswaveguide = waveguide.Waveguide(length=25e-3)
    deltabeta = np.linspace(-5000, 5000, 1000)
    thisprocess = phasematching.SimplePhasematchingDeltaBeta(waveguide=thiswaveguide)
    thisprocess.deltabeta = deltabeta
    thisprocess.calculate_phasematching(normalized=False)
    print(thisprocess.calculate_integral())
    thisprocess.plot()
    print(utils.bandwidth(thisprocess.deltabeta, abs(thisprocess.phi) ** 2))

    thisprocess2 = phasematching.SimplePhasematchingDeltaBeta(waveguide=thiswaveguide)
    thisprocess2.deltabeta = deltabeta
    thisprocess2.calculate_phasematching()
    print(thisprocess2.calculate_integral())
    thisprocess2.plot()

    poling_period = utils.calculate_poling_period(1550e-9, 0, 532e-9, ny(150), nz(150), ny(150))
    thiswaveguide2 = waveguide.Waveguide(length=25e-3, poling_period=poling_period)
    thisprocess3 = phasematching.SimplePhasematching1D(waveguide=thiswaveguide2,
                                                       n_red=ny(150),
                                                       n_green=nz(150),
                                                       n_blue=ny(150))
    thisprocess3.red_wavelength = np.linspace(1530, 1570, 1000) * 1e-9
    thisprocess3.blue_wavelength = 532e-9
    thisprocess3.calculate_phasematching()
    thisprocess3.plot()
    print(utils.bandwidth(thisprocess3.red_wavelength, abs(thisprocess3.phi) ** 2))

    thisprocess4 = phasematching.SimplePhasematching2D(waveguide=thiswaveguide2,
                                                       n_red=ny(150),
                                                       n_green=nz(150),
                                                       n_blue=ny(150))
    thisprocess4.red_wavelength = np.linspace(1530, 1570, 1000) * 1e-9
    thisprocess4.blue_wavelength = np.linspace(530, 534, 200) * 1e-9
    thisprocess4.calculate_phasematching()
    thisprocess4.plot()
    thisprocess4.plot_deltabeta_contour()

    plt.show()


def test_phasematching():
    ny, nz = custom_sellmeier()
    z = np.linspace(0, 25, 1000) * 1e-3

    thiswaveguide = waveguide.RealisticWaveguide(z=z, nominal_parameter_name=r"$\Delta\beta$", nominal_parameter=0)
    thiswaveguide.create_noisy_waveguide(noise_profile="1/f", noise_amplitude=1000)
    thiswaveguide.plot_waveguide_properties()

    thisprocess = phasematching.PhasematchingDeltaBeta(waveguide=thiswaveguide)
    thisprocess.deltabeta = np.linspace(-5000, 5000, 1000)
    thisprocess.calculate_phasematching(normalized=True)
    print(thisprocess.calculate_integral())
    thisprocess.plot()

    poling_period = utils.calculate_poling_period(1550e-9, 0, 532e-9, ny(150), nz(150), ny(150))
    thiswaveguide2 = waveguide.RealisticWaveguide(z=z,
                                                  poling_period=poling_period,
                                                  nominal_parameter_name=r"$T$",
                                                  nominal_parameter=150)
    thiswaveguide2.create_noisy_waveguide(noise_profile="1/f", noise_amplitude=5)

    thisprocess2 = phasematching.Phasematching1D(waveguide=thiswaveguide2,
                                                 n_red=ny,
                                                 n_green=nz,
                                                 n_blue=ny)
    thisprocess2.red_wavelength = np.linspace(1530, 1570, 1000) * 1e-9
    thisprocess2.blue_wavelength = 532e-9
    thisprocess2.calculate_phasematching()
    thisprocess2.plot()

    thisprocess3 = phasematching.Phasematching2D(waveguide=thiswaveguide2,
                                                 n_red=ny,
                                                 n_green=nz,
                                                 n_blue=ny)
    thisprocess3.red_wavelength = np.linspace(1530, 1570, 200) * 1e-9
    thisprocess3.blue_wavelength = np.linspace(530, 534, 200) * 1e-9
    thisprocess3.calculate_phasematching()
    thisprocess3.plot()

    plt.show()


if __name__ == '__main__':
    # test_waveguide()
    # test_noise()
    # test_simple_phasematching()
    test_phasematching()
