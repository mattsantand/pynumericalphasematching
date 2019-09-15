import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import scipy.optimize as opt
import logging
import enum
from typing import Callable, List


class Propagation(enum.Enum):
    """
    Enum class containing the two possible propagation configurations, *copropagation* and *counterpropagation*. It is
    used to change the sign of the red_wavelength in the calculations of the deltabeta in this module.

    """
    COPROPAGATION = -1
    """Sign of the signal in copropagation"""
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
    :param propagation_type: Type of the propagation (co- or counter-propagation). Default is coprop.
    :type propagation_type: :class:`pynumpm.utils.Propagation`
    :return: The poling period.
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
    return Lambda


def calculate_phasematching_point(fixed_wl, poling_period: float, nr: Callable[[float], float],
                                  ng: Callable[[float], float], nb: Callable[[float], float],
                                  hint, order: int = 1, verbose: bool = False):
    """
    Function to calculate the phasematching point, given a wavelength and the poling period of the structure.

    This function tries to minimise the energy and momentum conservation equations to find the possible phasematched
    processes, given a fixed wavelength and the poling period, using the `fsolve`routine from `scipy`.

    The `fixed_wl` variable is a list (or tuple) containing the value and the field name ("r", "g" or "b" for red,
    green and blue). For example, if one wants to define the green field as fixed at 800nm, the `fixed_wl` must be in
    the form [800e-9, "g"].

    In case of an SHG calculation, `fixed_wl` can receive as second parameter the string "shg".

    .. note:: In case of an SHG calculation, the value of the constant wavelength is not used.



    :param fixed_wl: Wavelength to be kept constant during the calculation.
    :type fixed_wl: list
    :param poling_period: Poling period of the structure
    :type poling_period: float
    :param nr: Function returning the refractive index for the red field.
    :type nr: Function
    :param ng: Function returning the refractive index for the green field.
    :type ng: Function
    :param nb: Function returning the refractive index for the blue field.
    :type nb: Function
    :param hint: List of the two wavelengths to be used as first hints to find the solution.
    :type hint: list
    :param order: Order of the process. Default: 1
    :type order: int
    :param verbose: Set the calculation to be verbose or not. Default: False
    :type verbose: bool
    :return: List of red wavelength, green wavelength, blue wavelength and poling period.

    """
    lam, constlam = fixed_wl
    # convert all the units in um
    lam *= 1e6
    poling_period_um = poling_period * 1e6
    hint = np.array([i * 1e6 for i in hint])

    # List of functions describing energy and momentum conservation, to be minimised.
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
    Function to calculate the bandwidth of a given phasematching spectrum fitting it with savgol_filter and then approximating it
    with a UnivariateSpline.

    .. warning:: This function has not been tested thoroughly.

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



