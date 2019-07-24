import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import logging


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


# def deltabeta(lam1, lam2, free_param, nr, ng, nb, poling=np.infty):
#     """
#     Function to calculate the delta
#     :param lam1:
#     :param lam2:
#     :param free_param:
#     :param nr:
#     :param ng:
#     :param nb:
#     :param poling:
#     :return:
#     """
#     if free_param == "b":
#         wlr, wlg = lam1, lam2
#         wlb = (wlr ** -1 + wlg ** -1) ** -1
#     elif free_param == "g":
#         wlr, wlb = lam1, lam2
#         wlg = (wlb ** -1 - wlr ** -1) ** -1
#     elif free_param == "r":
#         wlg, wlb = lam1, lam2
#         wlr = (wlb ** -1 - wlg ** -1) ** -1
#     else:
#         raise ValueError("Wrong label for free parameter")
#     return wlr, wlg, wlb, 2 * np.pi * (nb(wlb) / wlb - ng(wlg) / wlg - nr(wlr) / wlr - 1 / poling)


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

