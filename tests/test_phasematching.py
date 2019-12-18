import numpy as np
import matplotlib.pyplot as plt
from pynumpm import waveguide, phasematching, utils
from examples import custom_sellmeier
import pytest

ny, nz = custom_sellmeier()


def test_simplePhasematchingDeltabeta():
    length = 24e-3

    thiswaveguide = waveguide.Waveguide(length=length,
                                        poling_period=+np.infty)

    # SimplePhasematchingDeltaBeta
    thisprocess = phasematching.SimplePhasematchingDeltaBeta(waveguide=thiswaveguide)
    db = np.linspace(-1000, 1000, 2000)
    thisprocess.deltabeta = db
    phi = thisprocess.calculate_phasematching()

    correct_phi = np.sinc(db * length / 2 / np.pi) * np.exp(1j * db * length / 2)
    assert np.all(np.isclose(correct_phi, phi))


def test_poling_period():
    nred = nz(20)
    ngreen = ny(20)
    nblue = ny(20)

    wlred = 1550e-9
    wlgreen = 870e-9
    wlblue = (wlred ** -1 + wlgreen ** -1) ** -1

    correct_poling = 1 / (nblue(wlblue * 1e6) / wlblue - ngreen(wlgreen * 1e6) / wlgreen - nred(wlred * 1e6) / wlred)
    poling_period = utils.calculate_poling_period(wlred, wlgreen, 0, nred, ngreen, nblue)
    assert poling_period == correct_poling

    wlred = 1550e-9
    wlblue = 550e-9
    wlgreen = (wlblue ** -1 - wlred ** -1) ** -1

    correct_poling = 1 / (nblue(wlblue * 1e6) / wlblue - ngreen(wlgreen * 1e6) / wlgreen - nred(wlred * 1e6) / wlred)
    poling_period = utils.calculate_poling_period(wlred, 0, wlblue, nred, ngreen, nblue)
    assert poling_period == correct_poling

    wlgreen = 1550e-9
    wlblue = 550e-9
    wlred = (wlblue ** -1 - wlgreen ** -1) ** -1

    correct_poling = 1 / (nblue(wlblue * 1e6) / wlblue - ngreen(wlgreen * 1e6) / wlgreen - nred(wlred * 1e6) / wlred)
    poling_period = utils.calculate_poling_period(0, wlgreen, wlblue, nred, ngreen, nblue)
    assert poling_period == correct_poling


def test_SimplePhasematching1D():
    length = 15e-3

    nred = ny(20)
    ngreen = nz(20)
    nblue = ny(20)

    red_wl0 = 1550e-9
    red_span = 10e-9
    green_wl0 = 890e-9

    wlred = red_wl0
    wlgreen = green_wl0
    wlblue = (wlred ** -1 + wlgreen ** -1) ** -1

    poling_period = utils.calculate_poling_period(wlred, wlgreen, 0, nred, ngreen, nblue)
    assert poling_period == 1 / (
            nblue(wlblue * 1e6) / wlblue - ngreen(wlgreen * 1e6) / wlgreen - nred(wlred * 1e6) / wlred)

    # Define the waveguide
    thiswaveguide = waveguide.Waveguide(length=length,
                                        poling_period=poling_period)

    # Define the phasematching process
    thisprocess = phasematching.SimplePhasematching1D(waveguide=thiswaveguide,
                                                      n_red=nred,
                                                      n_green=ngreen,
                                                      n_blue=nblue,
                                                      order=1)
    # Define the range for the scanning wavelength
    red = np.linspace(red_wl0 - red_span / 2, red_wl0 + red_span / 2, 1000)
    green = green_wl0
    blue = (red ** -1 + green ** -1) ** -1
    thisprocess.red_wavelength = red
    thisprocess.green_wavelength = green
    # Calculate the phasematching spectrum
    phi = thisprocess.calculate_phasematching()

    db = 2 * np.pi * (nblue(blue * 1e6) / blue -
                      ngreen(green * 1e6) / green -
                      nred(red * 1e6) / red -
                      1 / poling_period)
    correct_phi = np.sinc(db * length / 2 / np.pi) * np.exp(-1j * db * length / 2)
    assert np.all(np.isclose(correct_phi, phi))

    # Test 2
    # Define the range for the scanning wavelength
    red = np.linspace(red_wl0 - red_span / 2, red_wl0 + red_span / 2, 1000)
    blue = wlblue
    green = (blue ** -1 - red ** -1) ** -1
    print(red, green, blue)

    thisprocess = phasematching.SimplePhasematching1D(waveguide=thiswaveguide,
                                                      n_red=nred,
                                                      n_green=ngreen,
                                                      n_blue=nblue,
                                                      order=1)
    thisprocess.red_wavelength = red
    thisprocess.blue_wavelength = blue
    # Calculate the phasematching spectrum
    phi = thisprocess.calculate_phasematching()

    db = 2 * np.pi * (nblue(blue * 1e6) / blue -
                      ngreen(green * 1e6) / green -
                      nred(red * 1e6) / red -
                      1 / poling_period)
    correct_phi = np.sinc(db * length / 2 / np.pi) * np.exp(-1j * db * length / 2)
    assert np.all(np.isclose(correct_phi, phi))


def test_SimplePhasematching2D():
    length = 10e-3

    nred = ny(20)
    ngreen = nz(20)
    nblue = ny(20)

    red_wl0 = 1550e-9
    red_span = 10e-9
    green_wl0 = 890e-9
    blue_span = 2e-9

    wlred = red_wl0
    wlgreen = green_wl0
    wlblue = (wlred ** -1 + wlgreen ** -1) ** -1

    poling_period = utils.calculate_poling_period(wlred, wlgreen, 0, nred, ngreen, nblue)
    assert poling_period == 1 / (
            nblue(wlblue * 1e6) / wlblue - ngreen(wlgreen * 1e6) / wlgreen - nred(wlred * 1e6) / wlred)

    # Define the waveguide
    thiswaveguide = waveguide.Waveguide(length=length,
                                        poling_period=poling_period)

    # Define the phasematching process
    thisprocess = phasematching.SimplePhasematching2D(waveguide=thiswaveguide,
                                                      n_red=nred,
                                                      n_green=ngreen,
                                                      n_blue=nblue,
                                                      order=1)
    # Define the range for the scanning wavelength
    red = np.linspace(red_wl0 - red_span / 2, red_wl0 + red_span / 2, 1000)
    blue = np.linspace(wlblue - blue_span / 2, wlblue + blue_span / 2, 1000)
    RED, BLUE = np.meshgrid(red, blue)
    GREEN = (BLUE ** -1 - RED ** -1) ** -1
    thisprocess.red_wavelength = red
    thisprocess.blue_wavelength = blue
    # Calculate the phasematching spectrum
    phi = thisprocess.calculate_phasematching()

    db = 2 * np.pi * (nblue(BLUE * 1e6) / BLUE -
                      ngreen(GREEN * 1e6) / GREEN -
                      nred(RED * 1e6) / RED -
                      1 / poling_period)
    correct_phi = np.sinc(db * length / 2 / np.pi) * np.exp(-1j * db * length / 2)
    assert np.all(np.isclose(correct_phi, phi))

    # Test 2
    # Define the phasematching process
    thisprocess = phasematching.SimplePhasematching2D(waveguide=thiswaveguide,
                                                      n_red=nred,
                                                      n_green=ngreen,
                                                      n_blue=nblue,
                                                      order=1)
    # Define the range for the scanning wavelength
    red = np.linspace(red_wl0 - red_span / 2, red_wl0 + red_span / 2, 1000)
    green = np.linspace(wlgreen - 10e-9 / 2, wlgreen + 10e-9 / 2, 1000)
    RED, GREEN = np.meshgrid(red, green)
    BLUE = (GREEN ** -1 + RED ** -1) ** -1
    thisprocess.red_wavelength = red
    thisprocess.green_wavelength = green
    # Calculate the phasematching spectrum
    phi = thisprocess.calculate_phasematching()

    db = 2 * np.pi * (nblue(BLUE * 1e6) / BLUE -
                      ngreen(GREEN * 1e6) / GREEN -
                      nred(RED * 1e6) / RED -
                      1 / poling_period)
    correct_phi = np.sinc(db * length / 2 / np.pi) * np.exp(-1j * db * length / 2)
    assert np.all(np.isclose(correct_phi, phi))


def test_PhasematchingDeltaBeta():
    length = 15e-3
    z = np.linspace(0, length, 2000)
    thiswaveguide = waveguide.RealisticWaveguide(z=z,
                                                 poling_period=+np.infty,
                                                 nominal_parameter=0,
                                                 nominal_parameter_name="db")

    # SimplePhasematchingDeltaBeta
    thisprocess = phasematching.PhasematchingDeltaBeta(waveguide=thiswaveguide)
    db = np.linspace(-1000, 1000, 2000)
    thisprocess.deltabeta = db
    phi = thisprocess.calculate_phasematching()
    correct_phi = np.sinc(db * length / 2 / np.pi) * np.exp(1j * db * length / 2)
    assert np.all(abs(correct_phi - phi) < 1e-3)
    profile = np.load("pink_noise_wg_profile.npy") * 2000

    length = 24.5e-3
    z = np.linspace(0, length, len(profile))
    thiswaveguide = waveguide.RealisticWaveguide(z=z,
                                                 poling_period=+np.infty,
                                                 nominal_parameter=0,
                                                 nominal_parameter_name="db")
    thiswaveguide.load_waveguide_profile(profile)
    thisprocess = phasematching.PhasematchingDeltaBeta(waveguide=thiswaveguide)
    db = np.linspace(-1000, 1000, 2000)
    thisprocess.deltabeta = db
    phi = thisprocess.calculate_phasematching()
    correct_phi = np.load("spectrum_pink_waveguide_deltabeta.npy")
    assert np.all(abs(correct_phi - phi) < 1e-4)


def test_Phasematching1D():
    nominal_parameter = 20
    wlred0 = 1550e-9
    wlgreen0 = 775e-9
    wlblue0 = (wlred0 ** -1 + wlgreen0 ** -1) ** -1

    nred = nz
    ngreen = nz
    nblue = nz

    poling_period = utils.calculate_poling_period(wlred0, wlgreen0, 0,
                                                  nred(nominal_parameter), ngreen(nominal_parameter),
                                                  nblue(nominal_parameter))

    length = 25e-3
    z = np.linspace(0, length, 5000)
    thiswaveguide = waveguide.RealisticWaveguide(z=z,
                                                 poling_period=poling_period,
                                                 nominal_parameter=nominal_parameter,
                                                 nominal_parameter_name="Temperature")
    thisprocess = phasematching.Phasematching1D(waveguide=thiswaveguide,
                                                n_red=nred,
                                                n_green=ngreen,
                                                n_blue=nblue)
    redwl = np.linspace(wlred0 - 2e-9, wlred0 + 2e-9, 1000)
    thisprocess.red_wavelength = redwl
    thisprocess.green_wavelength = wlgreen0
    bluewl = (redwl ** -1 + wlgreen0 ** -1) ** -1
    phi = thisprocess.calculate_phasematching()
    db = 2 * np.pi * (nblue(nominal_parameter)(bluewl * 1e6) / bluewl -
                      ngreen(nominal_parameter)(wlgreen0 * 1e6) / wlgreen0 -
                      nred(nominal_parameter)(redwl * 1e6) / redwl -
                      1 / poling_period)
    correct_phi = np.sinc(db * length / 2 / np.pi) * np.exp(-1j * db * length / 2)
    assert np.all(np.abs(correct_phi - phi) < 1e-3)


    # profile = np.load("pink_noise_wg_profile.npy") * 10 + nominal_parameter


if __name__ == '__main__':
    # pytest.main([__file__])
    test_Phasematching1D()
