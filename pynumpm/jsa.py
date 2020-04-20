# coding=utf-8
"""
Module to simulate the 2D spectrum of a pump field for the simulation of joint spectral amplitudes of nonlinear
processes.

"""
import logging
from scipy.special import hermite, factorial
from scipy.constants import c as _sol
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import enum
import warnings
from typing import Union
from pynumpm import phasematching


class Process(enum.Enum):
    """
    Enum class used to describe the nonlinear processes. It is used to change the way the pump spectrum is calculated.

    """

    SFG = "SFG. Input is the wavelength1, Output is the wavelength2."
    DFG = "DFG. Input is the wavelength1, Output is the wavelength2."
    PDC = "PDC."
    BWPDC = "BWPDC. Wavelength1 is the forward propagating field, wavelength2 is the backward one."  # TODO: Check this


class Pump(object):
    """
    Pump class. It is used to describe 2D pump functions for the calculation of JSA.
    The pump is modelled as

    .. math::
        \\alpha(\omega_1, \omega_2) = \mathrm{HG}_n(\omega_1, \omega_2)\cdot
        \mathrm{exp}\\left\\lbrace -\\frac{(\omega_p - \omega_{p,0})^2}{2\\sigma^2}\\right\\rbrace
        \cdot \mathrm{Chirp}
        \cdot \mathrm{filter}

    Initialize the pump object calling the class and passing a suitable `Process` element.

    The parameters of the pump must be assigned by the user. The following attributes can be modified:

    :var pump_spectrum:
    :var wavelength1:
    :var pump_centre:
    :var pump_width:
    :var pump_temporal_mode:
    :var pump_chirp:
    :var pump_delay:
    :var filter_width:

    """

    def __init__(self, process: Process):
        """

        :param process: Process under investigation. An element of the class `pynumpm.jsa.Process`.
        :type process: Process
        """

        self.__process = None
        self.process = process

        self.__pump_centre = None
        self.__pump_wavelength2D = None
        self.__pump_width = None
        self.__filter_pump = False
        self.__pump_delay = 0
        self.__pump_chirp = 0
        self.__pump_temporal_mode = 0
        self.__pump_filter_width = None
        self.__wavelength1 = None
        self.__wavelength2 = None
        self.__wavelength1_2D = None
        self.__wavelength2_2D = None
        self.__pump_spectrum = None
        self.__correct_pump_width = None

    @property
    def pump_spectrum(self):
        return self.__pump_spectrum

    @pump_spectrum.setter
    def pump_spectrum(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("The 'pump_spectrum' must be a numpy.ndarray object.")
        if value.shape[0] != len(self.wavelength1):
            raise ValueError("The input pump spectrum is not compatible with the signal wavelength set. "
                             "Make sure pump_spectrum.shape[0] == len(wavelength1)")
        if value.shape[1] != len(self.wavelength2):
            raise ValueError("The input pump spectrum is not compatible with the idler wavelength set. "
                             "Make sure pump_spectrum.shape[1] == len(wavelength2)")
        self.__pump_spectrum = value

    @property
    def wavelength1(self):
        return self.__wavelength1

    @wavelength1.setter
    def wavelength1(self, value):
        self.__wavelength1 = value

    @property
    def wavelength2(self):
        return self.__wavelength2

    @wavelength2.setter
    def wavelength2(self, value):
        self.__wavelength2 = value

    @property
    def process(self):
        return self.__process

    @process.setter
    def process(self, value):
        if not isinstance(value, Process):
            raise TypeError("The type of 'process' must be pynumpm.jsa.Process")
        self.__process = value

    @property
    def pump_centre(self):
        return self.__pump_centre

    @pump_centre.setter
    def pump_centre(self, value):
        self.__pump_centre = value

    @property
    def pump_width(self):
        return self.__pump_width

    @pump_width.setter
    def pump_width(self, value):
        self.__pump_width = value

    @property
    def filter_width(self):
        return self.__pump_filter_width

    @filter_width.setter
    def filter_width(self, value):
        self.__filter_pump = True
        self.__pump_filter_width = value

    @property
    def pump_delay(self):
        return self.__pump_delay

    @pump_delay.setter
    def pump_delay(self, value):
        self.__pump_delay = value

    @property
    def pump_chirp(self):
        return self.__pump_chirp

    @pump_chirp.setter
    def pump_chirp(self, value):
        self.__pump_chirp = value

    @property
    def pump_temporal_mode(self):
        return self.__pump_temporal_mode

    @pump_temporal_mode.setter
    def pump_temporal_mode(self, value: int):
        errormsg = "The pump temporal mode has to be non-negative integer"
        if not isinstance(value, int):
            raise TypeError(errormsg)
        if value < 0:
            raise ValueError(errormsg)
        self.__pump_temporal_mode = value

    @property
    def wavelength1_2D(self):
        return self.__wavelength1_2D

    @property
    def wavelength2_2D(self):
        return self.__wavelength2_2D

    def _hermite_mode(self, x: float):
        """ A normalised Hermite-Gaussian function """
        # On 22.11.2017, Matteo changed all the self.pump_width to self.__correct_pump_width
        # _result = hermite(self.pump_temporal_mode)((self.pump_center - x) /
        #                                            self.pump_width) *    \
        #     exp(-(self.pump_center - x)**2 / (2 * self.pump_width**2)) /\
        #     sqrt(factorial(self.pump_temporal_mode) * sqrt(pi) *
        #          2**self.pump_temporal_mode * self.pump_width)
        # TODO: Check the correctness of the __correct_pump_width parameter
        _result = hermite(self.pump_temporal_mode)((self.pump_centre - x) /
                                                   self.__correct_pump_width) * \
                  np.exp(-(self.pump_centre - x) ** 2 / (2 * self.__correct_pump_width ** 2)) / \
                  np.sqrt(factorial(self.pump_temporal_mode) * np.sqrt(np.pi) *
                          2 ** self.pump_temporal_mode * self.__correct_pump_width)
        return _result

    def __set_wavelengths(self):
        text = ""
        error = False
        if self.wavelength1 is None:
            error = True
            text += "You need to set wavelength1. "
        if self.wavelength2 is None:
            error = True
            text += "You need to set wavelength2."
        if error:
            raise ValueError(text.rstrip())

        self.__wavelength1_2D, self.__wavelength2_2D = np.meshgrid(self.wavelength1,
                                                                   self.wavelength2)
        message = "The pump central wavelength hasn't been set. Inferring its value from the wavelength1/wavelength2 arrays"
        if self.process == Process.PDC or self.process == Process.BWPDC:
            self.pump_wavelength = 1.0 / (1.0 / self.__wavelength1_2D +
                                          1.0 / self.__wavelength2_2D)
            if self.pump_centre is None:
                warnings.warn(message, UserWarning)
                self.pump_centre = (self.wavelength1.mean() ** -1 + self.wavelength2.mean() ** -1) ** -1

        elif self.process == Process.SFG:
            self.pump_wavelength = 1.0 / (1.0 / self.__wavelength2_2D -
                                          1.0 / self.__wavelength1_2D)
            if self.pump_centre is None:
                warnings.warn(message, UserWarning)
                self.pump_centre = (self.wavelength2.mean() ** -1 - self.wavelength1.mean() ** -1) ** -1

        elif self.process == Process.DFG:
            self.pump_wavelength = 1.0 / (1.0 / self.__wavelength1_2D -
                                          1.0 / self.__wavelength2_2D)
            if self.pump_centre is None:
                warnings.warn(message, UserWarning)
                self.pump_centre = (self.wavelength1.mean() ** -1 - self.wavelength2.mean() ** -1) ** -1
        else:
            raise NotImplementedError("The process {0} has not been implemented yet.".format(self.process))

    def calculate_pump_spectrum(self):
        """
        This function calculates the pump function.

        :return: matrix containing the pump function in wavelength1 and wavelength2 frequency plane
        """
        logger = logging.getLogger(__name__)
        self.__set_wavelengths()

        # self.pump_width /= 2 * sqrt(log(2))
        # self.pump_width = self.pump_width /( 2 * sqrt(log(2)))
        self.__correct_pump_width = self.pump_width / (2 * np.sqrt(np.log(2)))
        if self.__filter_pump:
            self.filter_width *= np.sqrt(2)
            _filter = np.zeros(np.shape(self.pump_wavelength), float)
            logger.debug("Pump wavelength: %f", np.shape(self.pump_wavelength))
            for i in range(len(self.wavelength1)):
                logger.debug("Loop index: %d", i)
                for j in range(len(self.wavelength2)):
                    if self.pump_wavelength[j, i] < self.pump_centre - \
                            0.5 * self.filter_width:
                        pass
                    elif self.pump_wavelength[j, i] <= self.pump_centre + \
                            0.5 * self.filter_width:
                        _filter[j, i] = 1
                    else:
                        pass
            _pump_function = self._hermite_mode(self.pump_wavelength) * \
                             np.exp(1j * 2 * np.pi * _sol / self.pump_wavelength *
                                    self.pump_delay) * \
                             np.exp(1j * (2 * np.pi * _sol / self.pump_wavelength) ** 2 *
                                    self.pump_chirp) * _filter
        else:
            _pump_function = self._hermite_mode(self.pump_wavelength) * \
                             np.exp(1j * 2 * np.pi * _sol / self.pump_wavelength *
                                    self.pump_delay) * \
                             np.exp(1j * (2 * np.pi * _sol / self.pump_wavelength) ** 2 *
                                    self.pump_chirp)
        self.__pump_spectrum = _pump_function
        return _pump_function

    def plot(self, ax=None, light_plot=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        if self.pump_spectrum is None:
            self.calculate_pump_spectrum()

        if light_plot:
            x, y = self.wavelength1 * 1e9, self.wavelength2 * 1e9,
            ax.imshow(abs(self.pump_spectrum) ** 2, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
                      aspect="auto")
            warnings.warn("The light_plot mode is compatible only with linear meshes of the signal/idler wavelengths.")
        else:
            ax.pcolormesh(self.wavelength1 * 1e9, self.wavelength2 * 1e9, abs(self.pump_spectrum) ** 2)

        ax.set_title("Pump intensity")
        if self.process == Process.SFG or self.process == Process.DFG:
            ax.set_xlabel(r"$\lambda_{input}$ [nm]")
            ax.set_ylabel(r"$\lambda_{output}$ [nm]")
        else:
            ax.set_xlabel(r"$\lambda_{signal}$ [nm]")
            ax.set_ylabel(r"$\lambda_{idler}$ [nm]")
        plt.tight_layout()


class JSA(object):
    def __init__(self, phasematching: Union[phasematching.SimplePhasematching2D, phasematching.Phasematching2D],
                 pump: Pump):
        self.__phasematching = phasematching
        self.__pump = None
        self.pump = pump
        self.__JSA = None
        self.__JSI = None
        self.__K = None
        self.__marginal1 = None
        self.__marginal2 = None
        self.__singular_values = None

    @property
    def phasematching(self):
        return self.__phasematching

    @phasematching.setter
    def phasematching(self, value):
        self.__phasematching = value

    @property
    def pump(self):
        return self.__pump

    @pump.setter
    def pump(self, value: Pump):
        if not isinstance(value, Pump):
            raise TypeError("The pump must be an object of the class pynumpm.jsa.Pump")
        if value.pump_spectrum is None:
            value.calculate_pump_spectrum()
        self.__pump = value

    @property
    def jsa(self):
        return self.__JSA

    @property
    def jsi(self):
        return self.__JSI

    @property
    def schmidt_number(self):
        return self.__K

    @property
    def marginal1(self):
        return self.__marginal1

    @property
    def marginal2(self):
        return self.__marginal2

    @property
    def singular_values(self):
        return self.__singular_values

    def calculate_JSA(self):
        """
        Function to calculate the JSA.

        :param pump_width: Pump object. Signal and idler wavelengths of the pump are overwritten to match the one of the
        phasematching process
        :type pump: :class:`~pynumpm.jsa.Pump`
        :return:
        """
        logger = logging.getLogger(__name__)
        logger.info("Calculating JSA")
        signal_wl = self.phasematching.wavelength1
        idler_wl = self.phasematching.wavelength2

        # d_wl_signal = np.diff(signal_wl)[0]
        # d_wl_idler = np.diff(self.phasematching.wavelength2)[0]

        WL_SIGNAL, WL_IDLER = np.meshgrid(signal_wl, idler_wl)
        self.pump.wavelength1 = WL_SIGNAL
        self.pump.wavelength2 = WL_IDLER

        JSA = self.pump.pump_spectrum * self.phasematching.phi

        d_wl_signal = np.gradient(signal_wl)
        d_wl_idler = np.gradient(idler_wl)
        DWSIG, DWIDL = np.meshgrid(d_wl_signal, d_wl_idler)
        JSA /= np.sqrt((abs(JSA * DWSIG * DWIDL) ** 2).sum())
        JSI = abs(JSA) ** 2
        self.__marginal1 = (JSI * abs(DWIDL)).sum(axis=0)
        self.__marginal2 = (JSI * abs(DWSIG)).sum(axis=1)
        self.__JSA = JSA
        self.__JSI = JSI
        return self.__JSA, self.__JSI

    def calculate_schmidt_decomposition(self, verbose=False):
        """
        Function to calculate the Schmidt decomposition.

        :param bool verbose: Print to screen the Schmidt number and the purity of the state.

        :return: the Schmidt number.
        """
        logger = logging.getLogger(__name__)
        if self.jsa is None:
            self.calculate_JSA()
        U, self.__singular_values, V = np.linalg.svd(self.__JSA)
        self.__singular_values /= np.sqrt((self.__singular_values ** 2).sum())
        self.__K = 1 / (self.__singular_values ** 4).sum()
        text = "Schmidt number K: {K}\nPurity: {P}".format(K=self.schmidt_number, P=1 / self.schmidt_number)
        if verbose:
            print(text)
        logger.info(text)
        logger.debug("Check normalization: sum of s^2 = " + str((abs(self.__singular_values) ** 2).sum()))
        return self.__K, self.__singular_values, U, V

    def plot_schmidt_coefficients(self, ncoeff: int = 20, ax: plt.Axes = None):
        """
        Function to plot the first n distribution of the Schmidt coefficients.

        :param ncoeff: Number of coefficients to plot. Default: 20
        :type ncoeff: int
        :param ax: Handles to the axis object where to plot.
        :param ax: `matplotlib.axes.Axes`
        :return:
        """
        if self.__singular_values is None:
            self.calculate_schmidt_decomposition()
        if ax is None:
            plt.figure()
            ax = plt.gca()
        plt.sca(ax)
        plt.bar(range(ncoeff), self.__singular_values[:ncoeff], 0.8)
        plt.xlabel("Mode number $i$")
        plt.ylabel(r"$\sqrt{\lambda_i}$")
        return ax

    def plot(self, ax=None, light_plot=False, normalized=True, title="JSI", plot_pump=False):
        """
        Function to plot JSI. Pass ax handle through "ax" to plot in a specified axis environment.

        :param ax: Axes handles
        :type ax: matplotlib.pyplot.axes
        :param light_plot:  Flag to allow plotting in the *light* mode. The light_plot mode is compatible only with
                            linear meshes of the signal/idler wavelengths. Default is False.
        :type light_plot: bool
        :param normalized: Flag to plot the JSI normalized or unnormalized. Default is True.
        :type normalized: bool
        :param plot_pump: Flag to plot the pump spectrum overlayed as contour plot. Default is False.
        :type plot_pump: bool.
        :param kwargs:
        :return: the axes handle for the plot
        """
        logger = logging.getLogger(__name__)
        if self.jsa is None:
            logger.info("The JSA was not calculated. I'll try to calculate it right away.")
            self.calculate_JSA()
            # raise ValueError("You need to calculate the JSA first, use the command calculate_jsa()")
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        title = title
        x = self.phasematching.wavelength1 * 1e9
        y = self.phasematching.wavelength2 * 1e9

        jsi_to_plot = self.jsi
        if normalized:
            logger.debug("The user wants the normalised JSI.")
            jsi_to_plot /= jsi_to_plot.max()

        if light_plot:
            im = ax.imshow(jsi_to_plot, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
                           aspect="auto")
            warnings.warn("The light_plot mode is compatible only with linear meshes of the signal/idler wavelengths.")
        else:
            im = ax.pcolormesh(x, y, jsi_to_plot)

        if plot_pump:
            logger.debug("The user wants the pump contours on the JSI plot.")
            X, Y = np.meshgrid(x, y)
            Z = abs(self.pump.pump_spectrum) ** 2
            sigmas = np.arange(0, 4, 1)[::-1]
            levels = np.exp(-sigmas)
            warnings.warn("The sigmas indicated in the JSI plot refer to the intensity of the pump. "
                          "If you want the field information, you need to convert them accordingly.")
            labels_dict = {level: str(len(levels) - i - 1) + r"$\sigma$" for i, level in enumerate(levels)}
            CS = ax.contour(X, Y, Z / Z.max(), levels, colors="w", linestyles="-",
                            linewidths=levels * 6)
            ax.clabel(CS, fmt=labels_dict, fontsize=9, inline=1)

        plt.gcf().colorbar(im)
        ax.set_xlabel(r"$\lambda_{signal}$ [nm]")
        ax.set_ylabel(r"$\lambda_{idler}$ [nm]")
        ax.set_title(title)
        plt.tight_layout()
        return ax

    def plot_marginals(self, ax=None, **kwargs):
        """
        Function to plot the marginals of the JSI.

        :param ax:  Axes handles for the *two* axes where to draw the marginals. The input can be None or a list of
                    the two axes handles. If None, a new plot is generated with two subplots, one for each marginal.
                    Default: *None*
        :type ax:  matplotlib.pyplot.axes
        :return: the Axes handles
        """
        if ax is None:
            fig, ax = plt.subplots(1, 2)
        else:
            if type(ax) != list or (type(ax) == list and len(ax) != 2):
                raise ValueError(
                    "I need two different axes to plot the marginals. ax should be a list of axes handles [ax0, ax1]")

        suptitle = kwargs.get("suptitle", "Marginals")
        print(self.phasematching.wavelength1 * 1e9)
        print(self.__marginal1)
        ax[0].plot(self.phasematching.wavelength1 * 1e9, self.marginal1)
        ax[1].plot(self.phasematching.wavelength2 * 1e9, self.marginal2)
        plt.suptitle(suptitle)
        return ax
