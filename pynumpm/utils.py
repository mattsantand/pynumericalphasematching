import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import scipy.optimize as opt
import logging
import enum
from typing import Callable, List


class Propagation(enum.Enum):
    COPROPAGATION = -1
    """Sign of the signal in copropagatio"""
    COUNTEPROPAGATION = +1
    """Sign of signal in counterpropagation"""


def calculate_poling_period(lamr: float = 0, lamg: float = 0, lamb: float = 0,
                            nr: Callable[[float], float] = None, ng: Callable[[float], float] = None,
                            nb: Callable[[float], float] = None, order=1, propagation_type=Propagation.COPROPAGATION):
    """
    Function to calculate the poling period of a specific process. To ensure energy conservation, specify only two
    wavelengths (in meter) and leave the free one to 0

    :param lamr: Wavelength of the red field [m].
    :type lamr: float
    :param lamg: Wavelength of the green field [m].
    :type lamg: float
    :param lamb: Wavelength of the blue field [m].
    :type lamb: float
    :param nr: Function returning the refractive index for the red field.
    :type nr: Function
    :param ng: Function returning the refractive index for the green field.
    :type ng: Function
    :param nb: Function returning the refractive index for the blue field.
    :type nb: Function
    :param order: Order of the process. Default: 1
    :type order: int
    :return: List of red wavelength, green wavelength, blue wavelength and poling period.
    """

    if not isinstance(propagation_type, Propagation):
        raise TypeError("propagation_type must be of the type pynumpm.utils.Propagation")
    if lamb == 0:
        lamb = 1. / (1. / abs(lamg) + 1. / abs(lamr))
    if lamg == 0:
        lamg = 1. / (1. / abs(lamb) - 1. / abs(lamr))
    if lamr == 0:
        lamr = 1. / (1. / abs(lamb) - 1. / abs(lamg))
    Lambda = order / (nb(abs(lamb) * 1e6) / lamb -
                      ng(abs(lamg) * 1e6) / lamg +
                      propagation_type.value * nr(abs(lamr) * 1e6) / lamr)
    return lamr, lamg, lamb, Lambda


def calculate_phasematching_point(fixed_wl: List[float, str], poling_period: float, nr: Callable[[float], float],
                                  ng: Callable[[float], float], nb: Callable[[float], float], hint: List[float, float],
                                  order: int = 1, verbose: bool = False):
    """
    Function to calculate the phasematching point, given a wavelength and the poling period of the structure.

    :param fixed_wl: Wavelength to be kept constant during the calculation.
    :type fixed_wl: typing.List
    :param poling_period: Poling period of the structure
    :type poling_period: float
    :param nr: Function returning the refractive index for the red field.
    :type nr: Function
    :param ng: Function returning the refractive index for the green field.
    :type ng: Function
    :param nb: Function returning the refractive index for the blue field.
    :type nb: Function
    :param hint: List of the two wavelengths to be used as first hints to find the solution.
    :type hint: typing.List
    :param order: Order of the process. Default: 1
    :type order: int
    :param verbose: Set the calculation to be verbose or not. Default: False
    :type verbose: bool
    :return: List of red wavelength, green wavelength, blue wavelength and poling period.

    This function tries to minimise the energy and momentum conservation equations to find the possible phasematched
    processes, given a fixed wavelength and the poling period, using the `fsolve`routine from `scipy`.

    The `fixed_wl` variable is a list (or tuple) containing the value and the field name ("r", "g" or "b" for red,
    green and blue). For example, if one wants to define the green field as fixed at 800nm, the `fixed_wl` must be in
    the form [800e-9, "g"].
    """
    lam, constlam = fixed_wl
    lam *= 1e6
    poling_period_um = poling_period * 1e6
    hint = np.array([i * 1e6 for i in hint])

    def zb(w):
        wb = lam
        wg, wr = w

        return np.array([nb(abs(wb)) / wb - ng(abs(wg)) / wg - nr(abs(wr)) / wr - 1.0 * order / poling_period_um,
                         1.0 / (1.0 / abs(wb) - 1.0 / abs(wg)) - abs(wr)])

    def zg(w):
        wg = lam
        wb, wr = w
        return np.array([nb(abs(wb)) / wb - ng(abs(wg)) / wg - nr(abs(wr)) / wr - 1.0 * order / poling_period_um,
                         1.0 / (1.0 / abs(wb) - 1.0 / abs(wg)) - abs(wr)])

    def zr(w):
        wr = lam
        wb, wg = w
        return np.array([nb(abs(wb)) / wb - ng(abs(wg)) / wg - nr(abs(wr)) / wr - 1.0 * order / poling_period_um,
                         1.0 / (1.0 / abs(wb) - 1.0 / abs(wg)) - abs(wr)])

    def zshg(w):
        wshg, wfund = w
        return np.array([nb(abs(wshg)) / wshg - ng(abs(wfund)) / wfund - nr(
            abs(wfund)) / wfund - order / poling_period_um, abs(wfund) - 2. * abs(wshg)])

    if constlam == 'shg':
        out = opt.fsolve(zshg, hint, full_output=True)
        if (out[2] == 1):
            # arr = np.sort(np.array([out[0][0], out[0][1], out[0][1]]))
            arr = np.array([out[0][0], out[0][1], out[0][1]])  # edited by Matteo on 24.10.2017
            return True, np.array([arr[0], arr[1], arr[2], poling_period_um]) * 1e-6
        else:
            if verbose:
                print("Error?:\n", out)
            return False, np.array([np.nan, np.nan, np.nan, np.nan])

    else:
        out = [False, False, False]
        if constlam == 'b':
            out = opt.fsolve(zb, hint, full_output=True)
        if constlam == 'g':
            out = opt.fsolve(zg, hint, full_output=True)
        if constlam == 'r':
            out = opt.fsolve(zr, hint, full_output=True)

        if out[2] == 1:
            # arr = np.sort(np.array([lam, out[0][0], out[0][1]]))
            arr = np.array([lam, out[0][0], out[0][1]])
            return True, np.array([arr[0], arr[1], arr[2], poling_period_um]) * 1e-6
        else:
            if verbose:
                print("Error?:\n", out)
            return False, np.array([np.nan, np.nan, np.nan, np.nan])


def bandwidth(wl, phi, **kwargs):
    """
    Calculates the bandwidth of a given phasematching spectrum fitting it with savgol_filter and then approximating it
    with a UnivariateSpline.

    :param wl: Wavelengths
    :type wl: Array
    :param phi: Phasematching intensity
    :type phi: Array
    :return: FWHM bandwidth of the phasematching intensity

    Additional parameters

    :param window_size: Savgol_filter  window_size parameter
    :type window_size: int
    :param polynomial_order: Savgol_filter polynomial_order parameter
    :type polynomial_order: int
    """

    window_size = kwargs.get("window_size", 71)
    polynomial_order = kwargs.get("polynomial_order", 9)

    smoothed = savgol_filter(phi, window_size, polynomial_order)

    spline = UnivariateSpline(wl * 1e9, np.abs(smoothed) ** 2 - np.max(np.abs(smoothed) ** 2) / 2, s=0)
    r1, r2 = spline.roots()
    bw = r2 - r1

    return bw


def calculate_profile_properties(z=None, profile=None):
    """
    Function to calculate the noise properties (autocorrelation and power density spectrum) of the noise on the
    waveguide profile
    :param z: z mesh of the system
    :type z: `numpy:numpy.ndarray`
    :param profile: Profile of the varying variable of the waveguide.
    :type profile: `numpy:numpy.ndarray`

    :return z_autocorr, autocorrelation, f, power_spectrum: Returns the autocorrelation profile (z axis included)
    and the power spectrum (frequency and power)
    """
    logger = logging.getLogger(__name__)
    logger.info("Calculating profile properties")
    if z is None:
        raise IOError("The z mesh is missing. Please, can you be so kind to provide me the discretization of the axis?")
    if profile is None:
        raise IOError("Oh dear! It looks like you have an empty profile! What do you want me to calculate about THAT? "
                      "Please provide a non-empty profile...")

    f = np.fft.fftshift(np.fft.fftfreq(len(z), np.diff(z)[0]))
    noise_spectrum = np.fft.fft(profile)
    power_spectrum = noise_spectrum * np.conj(noise_spectrum)
    autocorrelation = np.fft.ifftshift(np.fft.ifft(power_spectrum))
    power_spectrum = np.fft.fftshift(power_spectrum)
    z_autocorr = np.fft.fftshift(np.fft.fftfreq(len(f), np.diff(f)[0]))
    return z_autocorr, autocorrelation, f, power_spectrum
