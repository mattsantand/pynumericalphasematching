import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import scipy.optimize as opt
import logging
import enum


class Propagation(enum.Enum):
    COPROPAGATION = -1
    """Sign of the signal in copropagatio"""
    COUNTEPROPAGATION = +1
    """Sign of signal in counterpropagation"""

def calculate_poling_period(lamr=0, lamg=0, lamb=0, nr=None, ng=None, nb=None, order=1, **kwargs):
    """
    Function to calculate the poling period of a specific process. To ensure energy conservation, specify only 2
    wavelengths (in meter) and leave the free one to 0

    :param lamr:
    :param lamg:
    :param lamb:
    :param nr:
    :param ng:
    :param nb:
    :param order:
    :param kwargs:
    :return:
    """
    propagation_type = kwargs.get("propagation_type", "copropagation")
    if (lamb == 0):
        lamb = 1. / (1. / abs(lamg) + 1. / abs(lamr))
    if (lamg == 0):
        lamg = 1. / (1. / abs(lamb) - 1. / abs(lamr))
    if (lamr == 0):
        lamr = 1. / (1. / abs(lamb) - 1. / abs(lamg))
    if propagation_type.lower() == "copropagation":
        Lambda = order / (nb(abs(lamb) * 1e6) / lamb - ng(abs(lamg) * 1e6) / lamg - nr(abs(lamr) * 1e6) / lamr)
    elif propagation_type.lower() == "counterpropagation":
        Lambda = order / (nb(abs(lamb) * 1e6) / lamb - ng(abs(lamg) * 1e6) / lamg + nr(abs(lamr) * 1e6) / lamr)
    else:
        raise ValueError("Don't know " + propagation_type)
    return lamr, lamg, lamb, Lambda


def deltabeta(lamr=None, lamg=None, lamb=None, nr=None, ng=None, nb=None, poling=np.infty, order=1,
              propagation=Propagation.COPROPAGATION):
    """
    Function to calculate the delta
    :param lamr:
    :param lamg:
    :param lamb:
    :param nr:
    :param ng:
    :param nb:
    :param poling:
    :param order:
    :return:
    """
    importError = True
    if lamr is None and lamg is not None and lamb is not None:
        importError = False
        lamr = (lamb ** -1 - lamg ** -1) ** -1
    if lamg is None and lamr is not None and lamb is not None:
        importError = False
        lamg = (lamb ** -1 - lamr ** -1) ** -1
    if lamb is None and lamr is not None and lamg is not None:
        importError = False
        lamb = (lamr ** -1 + lamg ** -1) ** -1

    if importError:
        raise ValueError("Only one of the input wavelengths can be 'None'")

    return 2 * np.pi * (nb(lamb * 1e6) / lamb - ng(lamg * 1e6) / lamg +
                        propagation.value * nr(lamr * 1e6) / lamr - order / poling)


def calculate_phasematching_point(fixed_wl, poling_period, nb, ng, nr, hint, order=1, verbose=False):
    lam, constlam = fixed_wl
    lam *= 1e6
    poling_period_um = poling_period * 1e6
    hint = [i * 1e6 for i in hint]

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
        if constlam == 'b':
            out = opt.fsolve(zb, hint, full_output=True)
        if constlam == 'g':
            out = opt.fsolve(zg, hint, full_output=True)
        if constlam == 'r':
            out = opt.fsolve(zr, hint, full_output=True)

        if (out[2] == 1):
            # arr = np.sort(np.array([lam, out[0][0], out[0][1]]))
            arr = np.array([lam, out[0][0], out[0][1]])
            return True, np.array([arr[0], arr[1], arr[2], poling_period_um]) * 1e-6
        else:
            if verbose:
                print("Error?:\n", out)
            return False, np.array([np.nan, np.nan, np.nan, np.nan])


def bandwidth(wl, phi, **kwargs):
    """
    Calculates the bandwidth of a given phasematching phi on axis wl from a fitting with savgol_filter

    :param wl: Wavelengths
    :type wl: Array
    :param phi: Phasematching intensity
    :type phi: Array
    :return: FWHM bandwidth of the phasematching intensity

    Additional parameters

    :param window_size: Savgol_filter parameter
    :type window_size: int
    :param polynomial_order: Savgol_filter parameter
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
