import numpy as np
import matplotlib.pyplot as plt
from pynumpm import waveguide, noise, phasematching, jsa


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
    pass


if __name__ == '__main__':
    test_waveguide()
