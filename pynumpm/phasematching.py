# coding=utf-8
"""
.. module:: phasematching.py
.. moduleauthor:: Matteo Santandrea <matteo.santandrea@upb.de>
   :synopsis: Module to calculate the phasematching of a given waveguide, as specified by the classes provided in the
   pynumpm.waveguide module.

This module is used to calculate different types of phasematching:
    * :class:`~pynumpm.phasematching.PhasematchingDeltaBeta`: 1D phasematching spectrum, given the wavevector mismatch range to be analyzed.
    * :class:`~pynumpm.phasematching.Phasematching1D`: 1D phasematching spectrum, given the wavelength range to be analyzed and the Sellmeier equations.
    * :class:`~pynumpm.phasematching.Phasematching2D`: 2D phasematching spectrum, given the wavelength range to be analyzed and the Sellmeier equations.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
import scipy.interpolate as interp
from scipy.integrate import simps
from pynumpm import waveguide as Waveguide
from typing import Union, Callable
from tqdm import tqdm
import warnings

_REF_INDEX_TYPE0 = Callable[[float], float]
_REF_INDEX_TYPE1 = Callable[[float], Callable[[float], float]]


# TODO: replace rectangular integration in Phasematching 1D and 2D with the sinc (the currect integration)
# TODO: Use FFT to calculate Simple 1D and 2D phasematching with user
#  defined nonlinear profile
#  (introduce in version 1.1).

class SimplePhasematchingDeltaBeta(object):
    """
    Base class used for the calculation of phasematching as a function of :math:`\Delta\\beta` for uniform waveguides.

    The accessible attributes in this class are:

    :param waveguide: Waveguide object
    :type waveguide: :class:`~pynumpm.waveguide.Waveguide`
    :param deltabeta: vector containing the values of :math:`\Delta\\beta` needed to calculate the phasematching
    :type deltabeta: :class:`~numpy:numpy.ndarray`
    :param phi: Vector containing the phasematching spectrum
    :type phi: :class:`~numpy:numpy.ndarray`
    """

    def __init__(self, waveguide: Union[Waveguide.Waveguide, Waveguide.RealisticWaveguide]):
        """

        :param waveguide: Waveguide object describing the properties of a uniform waveguide.
        :type waveguide: pynumpm.waveguide.Waveguide

        """

        self._waveguide = None
        self.waveguide = waveguide
        self._deltabeta = None
        self._phi = None

    def __repr__(self):
        text = f"{self.__class__.__name__} object.\n\tWaveguide: {self.waveguide}."
        return text

    @property
    def waveguide(self):
        """
        :return: The waveguide object used in the calculation.

        """
        return self._waveguide

    @waveguide.setter
    def waveguide(self, waveguide: Waveguide.Waveguide):
        if isinstance(waveguide, (Waveguide.Waveguide, Waveguide.RealisticWaveguide)):
            self._waveguide = waveguide
        else:
            raise TypeError("You need to pass an object from the pynumpm.waveguide.Waveguide or "
                            "pynumpm.waveguide.RealisticWaveguide class.")

    @property
    def deltabeta(self) -> np.ndarray:
        """
        :return: The :math:`\Delta\\beta` vector used for the phasematching calculation.

        """
        return self._deltabeta

    @deltabeta.setter
    def deltabeta(self, value: np.ndarray):
        """
        Define the :math:`\Delta\\beta` vector used for the phasematching calculation.

        :param value: Array of :math:`\Delta\\beta` [*1/m*]
        :type value: numpy.ndarray
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("The deltabeta has to be a numpy.ndarray")
        self._deltabeta = value

    @property
    def phi(self):
        """
        :return: The phasematching complex amplitude

        """
        return self._phi

    def calculate_phasematching(self, normalized=True):
        """
        Function that calculates the phasematching.
        Prior to the evaluation of the phasematching, it is necessary to set the :math:`\Delta\\beta` vector by
        assigning it to the variable `deltabeta`.

        :param normalized: Sets the normalization of the phasematching. If True, the
                           phasematching will be normalized to the unit length (i.e., the maximum will be in [0,1]).
                           Default: *False*.
        :type normalized: bool
        :return: the function returns the complex-valued phasematching spectrum.

        """
        logger = logging.getLogger(__name__)
        logger.info("SimplePhasematchingDeltaBeta: Calculating the phasematching.")
        if self.deltabeta is None:
            raise ValueError("You need to define a delta beta range.")

        db = self.deltabeta - 2 * np.pi / self.waveguide.poling_period
        length = self.waveguide.length
        self._phi = np.sinc(db * length / 2 / np.pi) * np.exp(1j * db * length / 2)
        if not normalized:
            self._phi *= length
        return self.phi

    def plot(self, ax=None, normalized=False, **kwargs):
        """
        Function to plot the phasematching intensity.

        :param ax: Optional argument. Handle of the axis of the plot. Default: None
        :param normalized: Optional argument. If True, normalizes the plotted phasematching to have the maximum to 1.
                           Default: False
        :type normalized: bool
        :return: the axis handle of the plot
        """
        if ax is None:
            fig = plt.figure()
            plt.subplot(111)
            ax = plt.gca()
        if self.phi is None:
            raise IOError(
                "I'd really like to plot something nice, but you have not calculated the phasematching yet, "
                "so this would only be a white canvas.")
        y = abs(self.phi) ** 2
        if normalized:
            y = abs(self.phi) ** 2 / y.max()
        ax.plot(self.deltabeta, y, ls=kwargs.get("ls", "-"),
                lw=kwargs.get("lw", 3),
                color=kwargs.get("color"),
                label=kwargs.get("label"))
        plt.title("Phasematching")
        plt.xlabel(r"$\Delta\beta$ [m$^{-1}$]")
        plt.ylabel("Intensity [a.u.]")
        plt.tight_layout()
        return ax

    def calculate_integral(self):
        """
        Function to calculate the integral of the phasematching curve. It uses the function `simps` from the scipy module.

        :return: Intensity integral
        """
        return simps(abs(self.phi) ** 2, self.deltabeta)


class PhasematchingDeltaBeta(SimplePhasematchingDeltaBeta):
    """
    This class is used to simulate phasematching of systems considering only the wavevector mismatch
    (:math:`\Delta\\beta`).

    """

    def __init__(self, waveguide: Waveguide.RealisticWaveguide):
        """
        Initialization of the class requires the following parameter:

        :param waveguide: Waveguide object, as provided by the class waveguide.
        :type waveguide: :class:`~pynumpm.waveguide.RealisticWaveguide`

        """
        super(PhasematchingDeltaBeta, self).__init__(waveguide)
        self._cumulative_delta_beta = None
        self._cumulative_exp = None
        self._cumulative_sinc = None
        self._noise_length_product = None

    @property
    def waveguide(self):
        return self._waveguide

    @waveguide.setter
    def waveguide(self, waveguide: Waveguide.RealisticWaveguide):
        if isinstance(waveguide, Waveguide.RealisticWaveguide):
            self._waveguide = waveguide
        else:
            raise TypeError("You need to pass an object from the pynumpm.waveguide.RealisticWaveguide class.")

    @property
    def noise_length_product(self):
        return self._noise_length_product

    def calculate_phasematching(self, normalized=True):
        """
        Function that calculates the phasematching in case of inhomogeneous waveguide.
        Prior to the evaluation of the phasematching, it is necessary to set the :math:`\Delta\\beta` vector by
        assigning it to the variable `deltabeta`.

        :param normalized: Sets the normalization of the phasematching. If the normalization is on (True), the phasematching will be normalized to the unit length (i.e., the maximum will be in [0,1]). Default: False.
        :type normalized: bool
        :return: the function returns the complex-valued phasematching spectrum.

        """
        logger = logging.getLogger(__name__)
        logger.info("PhasematchingDeltaBeta: Calculating the phasematching.")
        if self.deltabeta is None:
            raise ValueError("You need to define a delta beta range.")
        self._cumulative_delta_beta = np.zeros(shape=len(self.deltabeta), dtype=complex)
        self._cumulative_exp = np.ones(shape=len(self.deltabeta), dtype=complex)
        self._cumulative_sinc = np.zeros(shape=len(self.deltabeta), dtype=complex)
        for i in tqdm(range(len(self.waveguide.z) - 1)):
            dz = self.waveguide.z[i + 1] - self.waveguide.z[i]
            this_deltabeta = self.deltabeta + self.waveguide.profile[i] - 2 * np.pi / self.waveguide.poling_period
            x = this_deltabeta * dz / 2
            self._cumulative_sinc += dz * np.sinc(x / np.pi) * np.exp(1j * x) * np.exp(
                1j * self._cumulative_delta_beta)
            self._cumulative_delta_beta += this_deltabeta * dz

        self._phi = self._cumulative_sinc
        if normalized:
            self._phi /= self.waveguide.length
        self._noise_length_product = abs(self.waveguide.profile).max() * self.waveguide.length
        return self.phi

    def plot(self, ax=None, normalized=False, verbose=False):
        """
        Function to plot the phasematching intensity.

        :param ax: Optional argument. Handle of the axis of the plot. Default: None
        :param normalized: Optional argument. If True, normalizes the plotted phasematching to have the maximum to 1. Default: False
        :type normalized: bool
        :param verbose: Optional. If True, writes the main information in the plot. Default: False
        :type verbose: bool
        :return: the axis handle of the plot
        """
        if ax is None:
            fig = plt.figure()
            plt.subplot(111)
            ax = plt.gca()
        if self.phi is None:
            raise IOError(
                "I'd really like to plot something nice, but you have not calculated the phasematching yet, "
                "so this would only be a white canvas.")
        y = abs(self.phi) ** 2
        if normalized:
            y = abs(self.phi) ** 2 / y.max()
        ax.plot(self.deltabeta, y)
        plt.title("Phasematching")
        plt.xlabel(r"$\Delta\beta$ [m$^{-1}$]")
        plt.ylabel("Intensity [a.u.]")
        if verbose:
            integral = self.calculate_integral()
            noise_length_product = self.noise_length_product
            text = "Integral: {integ:.3}\n".format(integ=integral) + \
                   r"$\sigma L$ = {sigmaL:.3}".format(sigmaL=noise_length_product)
            x0, x1 = plt.xlim()
            y0, y1 = plt.ylim()
            x = .7 * (x1 - x0) + x0
            y = .7 * y1
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(x, y, text, bbox=props)
        plt.tight_layout()
        return ax


class SimplePhasematching1D(object):
    """Class to calculate the 1D-phasematching, i.e. having one fixed wavelength and scanning another one (the third is
        fixed due to energy conservation.
        The convention for labelling wavelength is

        .. math::

            |\\lambda_{red}| \\geq |\\lambda_{green}| \\geq |\\lambda_{blue}|

        i.e. according to their "energy".

        """

    def __init__(self, waveguide: Union[Waveguide.Waveguide, Waveguide.RealisticWaveguide],
                 n_red: Union[_REF_INDEX_TYPE0, _REF_INDEX_TYPE1],
                 n_green: Union[_REF_INDEX_TYPE0, _REF_INDEX_TYPE1],
                 n_blue: Union[_REF_INDEX_TYPE0, _REF_INDEX_TYPE1],
                 order: int = 1, backpropagation: bool = False):
        """
        Initialization of the class requires the following parameters:

        :param waveguide: Waveguide object. Use the Waveguide class in the Waveguide module to define this object.
        :type waveguide: :class:`~pynumpm.waveguide.RealisticWaveguide`
        :param n_red: refractive index for the "red" wavelength. It has to be a lambda function of a lambda function,
                      i.e. n(variable_parameter)(wavelength in um)
        :param n_green: refractive index for the "green" wavelength. It has to be a lambda function of a lambda
                        function, i.e. n(variable_parameter)(wavelength in um)
        :param n_blue: refractive index for the "blue" wavelength. It has to be a lambda function of a lambda function,
                       i.e. n(variable_parameter)(wavelength in um)
        :param order: order of phasematching. Default: 1
        :param bool backpropagation: Set to True if it is necessary to calculate a backpropagation configuration.

        """
        self._waveguide = None
        self.waveguide = waveguide
        self._phi = None
        self._wavelengths_set = False
        self._n_red = n_red
        self._n_green = n_green
        self._n_blue = n_blue
        # ====================================================
        self.order = order
        # TODO: check if and how the poling order interferes when the poling structure is set
        self.process = None
        self._red_wavelength = None
        self._green_wavelength = None
        self._blue_wavelength = None
        self._input_wavelength = None
        self._output_wavelength = None
        if backpropagation:
            self.propagation_type = "backpropagation"
        else:
            self.propagation_type = "copropagation"
        # self._nonlinear_profile = None
        self.scanning_wavelength = None
        self._cumulative_delta_beta = None
        self._cumulative_exponential = None
        self._lamr0 = None
        self._lamg0 = None
        self._lamb0 = None

    @property
    def waveguide(self):
        return self._waveguide

    @waveguide.setter
    def waveguide(self, waveguide: Union[Waveguide.Waveguide, Waveguide.RealisticWaveguide]):
        """
        Method to load a waveguide object.

        :param waveguide:
        :return:
        """
        if isinstance(waveguide, (Waveguide.Waveguide, Waveguide.RealisticWaveguide)):
            self._waveguide = waveguide
        else:
            raise TypeError("You need to pass an object from the pynumpm.waveguide.Waveguide or "
                            "pynumpm.waveguide.RealisticWaveguide class.")

    @property
    def phi(self):
        """
        One dimensional phasematching spectrum (complex valued function) numpy.ndarray

        """
        return self._phi

    @property
    def n_red(self):
        return self._n_red

    @property
    def n_green(self):
        return self._n_green

    @property
    def n_blue(self):
        return self._n_red

    @property
    def red_wavelength(self):
        return self._red_wavelength

    @red_wavelength.setter
    def red_wavelength(self, value: Union[None, float, np.ndarray]):
        """
        Red wavelength of the process.

        :param value: None, Single float or vector of float, containing the "red" wavelengths, in meters.
        :type value: float, numpy.ndarray
        :return:
        """
        self._red_wavelength = value

    @property
    def green_wavelength(self):
        return self._green_wavelength

    @green_wavelength.setter
    def green_wavelength(self, value: Union[None, float, np.ndarray]):
        """
        Green wavelength of the process.

        :param value: None, Single float or vector of float, containing the "green" wavelengths, in meters.
        :type value: float, numpy.ndarray
        :return:
        """
        self._green_wavelength = value

    @property
    def blue_wavelength(self):
        return self._blue_wavelength

    @blue_wavelength.setter
    def blue_wavelength(self, value: Union[None, float, np.ndarray]):
        """
        Blue wavelength of the process.

        :param value: None, Single float or vector of float, containing the "blue" wavelengths, in meters.
        :type value: float, numpy.ndarray
        :return:
        """
        self._blue_wavelength = value

    @property
    def input_wavelength(self):
        """
        Input (scanning) wavelength of the process. It cannot be set, it is automatically detected.

        """
        return self._input_wavelength

    @property
    def output_wavelength(self):
        """
        Output (scanning) wavelength of the process. It cannot be set, it is automatically detected.

        """
        return self._output_wavelength

    @property
    def wavelengths_set(self):
        return self._wavelengths_set

    @property
    def lamr0(self):
        return self._lamr0

    @property
    def lamg0(self):
        return self._lamg0

    @property
    def lamb0(self):
        return self._lamb0

    def set_wavelengths(self):
        logger = logging.getLogger(__name__)
        num_of_none = (self.red_wavelength is None) + \
                      (self.green_wavelength is None) + \
                      (self.blue_wavelength is None)
        logger.info("Number of wavelengths set to 'None': " + str(num_of_none))
        if num_of_none > 2:
            raise ValueError("It would be cool to know in which wavelength range I should calculate the phasematching!")
        elif num_of_none == 2:
            # calculate SHG
            self.process = "shg"
            logger.info("Calculating wavelengths for SHG")
            if self.red_wavelength is not None:
                self.green_wavelength = self.red_wavelength
                self.blue_wavelength = self.red_wavelength / 2.
            elif self.green_wavelength is not None:
                self.red_wavelength = self.green_wavelength
                self.blue_wavelength = self.green_wavelength / 2.
            elif self.blue_wavelength is not None:
                self.red_wavelength = self.blue_wavelength / 2.
                self.green_wavelength = self.blue_wavelength / 2.
            else:
                logger.error("An error occurred in __set_wavelengths. When setting the SHG wavelengths, "
                             "all the wavelengths are set but the number of none is 2."
                             "Red wavelength: {r}\nGreen wavelength: {g}\nBlue wavelength: {b}".format(
                    r=self.red_wavelength,
                    g=self.green_wavelength,
                    b=self.blue_wavelength))
                raise ValueError("Something unexpected happened in set_wavelength. "
                                 "Check the log please and chat with the developer.")
            self._input_wavelength = self.red_wavelength
            self._output_wavelength = self.blue_wavelength
        elif num_of_none == 1:
            logger.info("Calculating wavelengths for sfg/dfg")
            self.process = "sfg/dfg"
            if self.red_wavelength is None:
                if type(self.green_wavelength) == np.ndarray:
                    self._input_wavelength = self.green_wavelength
                    logger.info("The input wavelength is the green")
                else:
                    self._input_wavelength = self.blue_wavelength
                    logger.info("The input wavelength is the blue")
                self.red_wavelength = (self.blue_wavelength ** -1 - self.green_wavelength ** -1) ** -1
                self._output_wavelength = self.red_wavelength
            elif self.green_wavelength is None:
                if type(self.red_wavelength) == np.ndarray:
                    self._input_wavelength = self.red_wavelength
                    logger.info("The input wavelength is the red")
                else:
                    self._input_wavelength = self.blue_wavelength
                    logger.info("The input wavelength is the blue")
                self.green_wavelength = (self.blue_wavelength ** -1 - self.red_wavelength ** -1) ** -1
                self._output_wavelength = self.green_wavelength
            elif self.blue_wavelength is None:
                if type(self.red_wavelength) == np.ndarray:
                    self._input_wavelength = self.red_wavelength
                    logger.info("The input wavelength is the red")
                else:
                    self._input_wavelength = self.green_wavelength
                    logger.info("The input wavelength is the green")
                self.blue_wavelength = (self.red_wavelength ** -1 + self.green_wavelength ** -1) ** -1
                self._output_wavelength = self.blue_wavelength
            else:
                logger.error("An error occurred in __set_wavelengths. When setting the SFG/DFG wavelengths, "
                             "all the wavelengths are set but the number of none is 1."
                             "Red wavelength: {r}\nGreen wavelength: {g}\nBlue wavelength: {b}".format(
                    r=self.red_wavelength,
                    g=self.green_wavelength,
                    b=self.blue_wavelength))
                raise ValueError("Something unexpected happened in set_wavelength. "
                                 "Check the log please and chat with the developer.")
        self._wavelengths_set = True
        self._lamr0 = self.red_wavelength.mean() if type(self.red_wavelength) == np.ndarray else self.red_wavelength
        self._lamg0 = self.green_wavelength.mean() if type(
            self.green_wavelength) == np.ndarray else self.green_wavelength
        self._lamb0 = self.blue_wavelength.mean() if type(self.blue_wavelength) == np.ndarray else self.blue_wavelength
        return self.red_wavelength, self.green_wavelength, self.blue_wavelength

    def calculate_delta_k(self):
        logger = logging.getLogger(__name__)
        if self.propagation_type == "copropagation":
            dd = 2 * pi * (self.n_blue(abs(self.blue_wavelength) * 1e6) / self.blue_wavelength -
                           self.n_green(abs(self.green_wavelength) * 1e6) / self.green_wavelength -
                           self.n_red(abs(self.red_wavelength) * 1e6) / self.red_wavelength -
                           float(self.order) / self.waveguide.poling_period)
            logger.debug("DK shape in __calculate_delta_k: " + str(dd.shape))
            return dd
        elif self.propagation_type == "backpropagation":
            dd = 2 * pi * (self.n_blue(abs(self.blue_wavelength) * 1e6) / self.blue_wavelength -
                           self.n_green(abs(self.green_wavelength) * 1e6) / self.green_wavelength +
                           self.n_red(abs(self.red_wavelength) * 1e6) / self.red_wavelength -
                           float(self.order) / self.waveguide.poling_period)
            logger.debug("DK shape in __calculate_delta_k: " + str(dd.shape))
            return dd
        else:
            raise NotImplementedError("I don't know what you asked!\n" + self.propagation_type)

    def calculate_phasematching(self, normalized=True):
        """
        This function is the core of the class. It calculates the phasematching of the process, considering one
        wavelength fixed and scanning the other two.

        :param normalized: If True, the phasematching is limited in [0,1]. Otherwise, the maximum depends on the
                                waveguide length, Default: True
        :type normalized: bool
        :return: the complex-valued phasematching spectrum
        """
        if not self.wavelengths_set:
            self.set_wavelengths()

        logger = logging.getLogger(__name__)
        logger.info("Calculating phasematching")
        db = self.calculate_delta_k()
        self._phi = np.sinc(db * self.waveguide.length / 2 / np.pi) * np.exp(-1j * db * self.waveguide.length / 2)
        if not normalized:
            self._phi /= self.waveguide.length
        return self.phi

    def calculate_integral(self):
        """
        Calculate the phasematching intensity integral

        :return: the phasematching intensity integral
        """
        if self.phi is not None:
            return simps(abs(self.phi) ** 2, self.scanning_wavelength)
        else:
            raise RuntimeError("You need to evaluate the phasematching spectrum, before I can calculate its integral.")

    def plot(self, ax=None, plot_intensity=True, plot_input=True, **kwargs):
        """
        Plot the phasematching intensity/amplitude.

        :param ax: Axis handle for the plot. If None, plots in a new figure. Default is None.
        :param plot_intensity: Set to True to plot the intensity profile, False to plot the amplitude and phase.
                               Default to True.
        :type plot_intensity: bool
        :param plot_input: Select the x axis for the plot. If True, use the `input_wavelength` as input, otherwise use `output_wavelength`.
        :type plot_intensity: bool
        :param kwargs: :func:`matplotlib.pyplot.plot` **kwargs arguments
        :return: figure and axis handle
        """
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        else:
            fig = plt.gcf()

        if plot_input:
            wl = self.input_wavelength
        else:
            wl = self.output_wavelength

        if plot_intensity:
            plt.plot(wl * 1e9, abs(self.phi) ** 2, ls=kwargs.get("ls", "-"),
                     lw=kwargs.get("lw", 3),
                     color=kwargs.get("color"),
                     label=kwargs.get("label"))
        else:
            plt.plot(wl * 1e9, np.abs(self.phi), label="Amplitude", **kwargs)
            plt.gca().set_ylabel("Amplitude")
            plt.gca().twinx().plot(wl * 1e9, np.unwrap(np.angle(self.phi)), ls=":", color="k", label="Phase", **kwargs)
            plt.gca().set_ylabel("Phase [rad]")

        xlabel = kwargs.get("xlabel", "Wavelength [nm]")
        ylabel = kwargs.get("ylabel", "Intensity [a.u.]")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Phasematching")
        fig = plt.gcf()
        plt.tight_layout()
        return fig, ax


class Phasematching1D(SimplePhasematching1D):
    """
    Class to calculate the 1D-phasematching, i.e. having one fixed wavelength and scanning another one (the third is
    fixed due to energy conservation.
    The convention for labelling wavelength is

    .. math::

        |\\lambda_{red}| \\geq |\\lambda_{green}| \\geq |\\lambda_{blue}|

    i.e. according to their "energy".

    """

    def __init__(self, waveguide: Waveguide.RealisticWaveguide, n_red: _REF_INDEX_TYPE1, n_green: _REF_INDEX_TYPE1,
                 n_blue: _REF_INDEX_TYPE1, order: int = 1, backpropagation: bool = False):
        """
        Initialization of the class requires the following parameters:

        :param waveguide: Waveguide object. Use the Waveguide class in the Waveguide module to define this object.
        :type waveguide: :class:`~pynumpm.waveguide.RealisticWaveguide`
        :param n_red: refractive index for `red` wavelength. It must be a function of a function, i.e.
                      n(parameter)(wavelength). The wavelength **must** be in micron (usual convention for
                      Sellmeier's equations)
        :type n_red: function of function
        :param n_green: refractive index for `green` wavelength. It must be a function of a function, i.e.
                        n(parameter)(wavelength). The wavelength **must** be in micron (usual convention for
                        Sellmeier's equations)
        :type n_green: function of function
        :param n_blue: refractive index for "blue" wavelength. It must be a function of a function, i.e.
                       n(parameter)(wavelength). The wavelength **must** be in micron (usual convention for
                       Sellmeier's equations)
        :type n_blue: function of function
        :param order: order of phasematching. Default: 1
        :type order: int
        :param backpropagation: Set to True if it is necessary to calculate a backpropagation configuration.
        :type backpropagation: bool

        """
        super().__init__(waveguide=waveguide,
                         n_red=n_red,
                         n_green=n_green,
                         n_blue=n_blue,
                         order=order,
                         backpropagation=backpropagation)
        self._noise_length_product = None
        self._delta_beta0_profile = None
        self._lamr0 = None
        self._lamg0 = None
        self._lamb0 = None
        self._nonlinear_profile = None
        self._nonlinear_profile_set = False

    @property
    def waveguide(self):
        return self._waveguide

    @waveguide.setter
    def waveguide(self, waveguide: Waveguide.RealisticWaveguide):
        """
        Method to load a waveguide object.

        :param waveguide:
        :return:
        """
        if isinstance(waveguide, Waveguide.RealisticWaveguide):
            self._waveguide = waveguide
        else:
            raise TypeError("You need to pass an object from the pynumpm.waveguide.Waveguide or "
                            "pynumpm.waveguide.RealisticWaveguide class.")

    @property
    def deltabeta_profile(self):
        """
        Profile of the :math:`\Delta\\beta` errpr
        
        """
        return self._delta_beta0_profile

    @property
    def nonlinear_profile(self):
        return self._nonlinear_profile

    @property
    def noise_length_product(self):
        """
        Product between the sample length and the maximum :math:`\Delta\\beta` variation for the process. If this value
        is above 10, the phasematching is likely to be noisy.

        """
        return self._noise_length_product

    def set_nonlinearity_profile(self, profile_type="constant", first_order_coeff=False, **kwargs):
        """
        Method to set the nonlinearity profile g(z), with either a constant profile or a variety of different windowing functions.

        :param str profile_type: Type of nonlinearity profile to consider. Possible options are
                                 [constant/gaussian/hamming/bartlett/hanning/blackman/kaiser].
        :param bool first_order_coeff: Select whether to simulate the reduction of efficiency due to quasi-phase
                                       matching or not.
        :param kwargs: Additional parameters to specify different variables of the `profile_type` used. Only effective
                       if `profile_type` is *"gaussian"* or *"kaiser"*.
        :return: The function returns the nonlinearity profile of the system.

        The different types of profile available are:

        * constant: Uniform nonlinear profile.
        * gaussian: :math:`g(z) = \\mathrm{e}^{-\\frac{(z-L/2)^2}{2\\sigma^2}}`. Set the :math:`\sigma` of the gaussian
                    profile with the `kwargs` argument `sigma_g_norm`, defining the standard deviation of the gaussian
                    profile in units of the length (defauls to 0.5, i.e. L/2).
        * hamming: :func:`numpy.hamming`
        * bartlett :func:`numpy.bartlett`
        * hanning: :func:`numpy.hanning`
        * blackman: :func:`numpy.blackman`
        * kaiser: :func:`numpy.kaiser`. Set the :math:`\\beta` parameter of the *Kaiser* profile with the `kwargs`
                  argument `beta`,
        """
        logger = logging.getLogger(__name__)
        logger.info("Setting the nonlinear profile.")
        logger.info("Profile type: {pt}".format(pt=profile_type))
        if profile_type == "constant":
            logger.debug("Value of first_order_coeff: {foc}".format(foc=first_order_coeff))
            if first_order_coeff:
                g = lambda z: 2 / pi
            else:
                g = lambda z: 1.
        elif profile_type == "gaussian":
            if first_order_coeff:
                coeff = 2 / np.pi
            else:
                coeff = 1
            sigma_norm = kwargs.get("sigma_g_norm", 0.5)
            # I drop the phase in the exponential term of the delta beta in the main loop
            g = lambda z: coeff * np.exp(
                -(z - self.waveguide.length / 2.) ** 2 / (2 * (sigma_norm * self.waveguide.length) ** 2))
        elif profile_type == "hamming":
            g = np.hamming(len(self.waveguide.z))
            g = interp.interp1d(self.waveguide.z, g, kind="cubic")
        elif profile_type == "bartlett":
            g = np.bartlett(len(self.waveguide.z))
            g = interp.interp1d(self.waveguide.z, g, kind="cubic")
        elif profile_type == "hanning":
            g = np.hanning(len(self.waveguide.z))
            g = interp.interp1d(self.waveguide.z, g, kind="cubic")
        elif profile_type == "blackman":
            g = np.blackman(len(self.waveguide.z))
            g = interp.interp1d(self.waveguide.z, g, kind="cubic")
        elif profile_type == "kaiser":
            g = np.kaiser(len(self.waveguide.z), kwargs.get("beta", 1.))
            g = interp.interp1d(self.waveguide.z, g, kind="cubic")
        else:
            raise ValueError("The nonlinear profile {0} has not been implemented yet.".format(profile_type))
        self._nonlinear_profile_set = True
        self._nonlinear_profile = g
        return self.nonlinear_profile

    def plot_nonlinearity_profile(self):
        """
        Function to plot the nonlinearity profile

        """
        x = self.waveguide.z
        y = self.nonlinear_profile(self.waveguide.z)
        plt.plot(x, y)

    def _calculate_local_neff(self, posidx):
        local_parameter = self.waveguide.profile[posidx]
        try:
            n_red = self.n_red(local_parameter)
        except:
            raise RuntimeError("Something happened here! 'local parameter' was {0}".format(local_parameter))
        try:
            n_green = self.n_green(local_parameter)
        except:
            raise RuntimeError("Something happened here! 'local parameter' was {0}".format(local_parameter))
        try:
            n_blue = self.n_blue(local_parameter)
        except:
            raise RuntimeError("Something happened here! 'local parameter' was {0}".format(local_parameter))
        return n_red, n_green, n_blue

    def calculate_delta_k(self, wl_red=None, wl_green=None, wl_blue=None, n_red=None, n_green=None, n_blue=None):
        """
        Overload of the method to calculate delta_k. This is necessary since this class is used to calculate the
        phasematching for a sample with variable dispersions.

        :param wl_red: *Red* wavelength of the process.
        :type wl_red: float or numpy.ndarray
        :param wl_green: *Green* wavelength of the process.
        :type wl_green: float or numpy.ndarray
        :param wl_blue: *Blue* wavelengh of the process.
        :type wl_blue: float or numpy.ndarray
        :param n_red: Function returning the refractive index for the *red* field as a function of the wavelength.
        :type n_red: function
        :param n_green: Function returning the refractive index for the *green* field as a function of the wavelength.
        :type n_green: function
        :param n_blue: Function returning the refractive index for the *blue* field as a function of the wavelength.
        :type n_blue: function
        :return:
        """
        logger = logging.getLogger(__name__)
        if self.propagation_type == "copropagation":
            dd = 2 * pi * (n_blue(abs(wl_blue) * 1e6) / wl_blue -
                           n_green(abs(wl_green) * 1e6) / wl_green -
                           n_red(abs(wl_red) * 1e6) / wl_red -
                           float(self.order) / self.waveguide.poling_period)
            logger.debug("DK shape in __calculate_delta_k: " + str(dd.shape))
            return dd
        elif self.propagation_type == "backpropagation":

            return 2 * pi * (n_blue(wl_blue * 1e6) / wl_blue -
                             n_green(wl_green * 1e6) / wl_green +
                             n_red(wl_red * 1e6) / wl_red -
                             float(self.order) / self.waveguide.poling_period)
        else:
            raise NotImplementedError("I don't know what you asked!\n" + self.propagation_type)

    def calculate_phasematching(self, normalized=True):
        """
        This function calculates the phasematching of the process. Use Phasematching1D.red_wavelength/green_wavelength/
        blue_wavelength to set the wavelengths of the process:

        * For SHG calculations, set **either** the red_wavelength or green_wavelength as a numpy.ndarray of wavelengths
          (in meters). The class will detect automatically that it has received a single input and will automatically
          calculate the SHG.
        * For other processes, set one wavelength as a numpy.ndarray and the other, fixed one as a float. The third
          will be calculated automatically.

        This function does not support PDC calculations. Use :class:`pynumpm.phasematching.SimplePhasematching2D` or
        :class:`pynumpm.phasematching.Phasematching2D` instead.

        :param bool normalized: If True, the phasematching is limited in [0,1]. Otherwise, the maximum depends on the
                                waveguide length. Default: True

        :return: the complex-valued phasematching spectrum
        """
        if not self.wavelengths_set:
            self.set_wavelengths()

        logger = logging.getLogger(__name__)
        logger.info("Calculating phasematching")

        if not self._nonlinear_profile_set:
            self.set_nonlinearity_profile(profile_type="constant", first_order_coefficient=False)
        if self.waveguide.poling_structure_set:
            logger.info("Poling period is not set. Calculating from structure.")
        else:
            logger.info("Poling period is set. Calculating with constant poling structure.")

        tmp_dk = self.calculate_delta_k(self.red_wavelength, self.green_wavelength, self.blue_wavelength,
                                        *self._calculate_local_neff(0))
        self._cumulative_delta_beta = np.zeros(shape=tmp_dk.shape)
        self._cumulative_exponential = np.zeros(shape=self._cumulative_delta_beta.shape, dtype=complex)
        logger.debug("Shape cumulative deltabeta:" + str(self._cumulative_delta_beta.shape))
        logger.debug("Shape cum_exp:" + str(self._cumulative_exponential.shape))
        self._delta_beta0_profile = np.nan * np.ones(shape=self.waveguide.z.shape)

        dz = np.gradient(self.waveguide.z)

        # for idx, z in enumerate(self.waveguide.z[:-1]):
        for idx in tqdm(range(1, len(self.waveguide.z)), ncols=100):
            z = self.waveguide.z[idx]
            # 1) retrieve the current parameter (width, thickness, ...)
            n_red, n_green, n_blue = self._calculate_local_neff(idx)
            # 2) evaluate the current phasemismatch
            DK = self.calculate_delta_k(self.red_wavelength, self.green_wavelength, self.blue_wavelength,
                                        n_red, n_green, n_blue)
            self._delta_beta0_profile[idx] = self.calculate_delta_k(self.lamr0, self.lamg0, self.lamb0, n_red,
                                                                    n_green,
                                                                    n_blue)
            # 4) add the phasemismatch to the past phasemismatches (first summation, over the delta k)
            self._cumulative_delta_beta += DK
            # 5) evaluate the (cumulative) exponential (second summation, over the exponentials)
            if self.waveguide.poling_structure_set:
                self._cumulative_exponential += self.nonlinear_profile(z) * dz[idx] * self.waveguide.poling_structure[
                    idx] * \
                                                (np.exp(-1j * dz[idx] * self._cumulative_delta_beta) -
                                                 np.exp(-1j * dz[idx] * (self._cumulative_delta_beta - DK)))
            else:
                self._cumulative_exponential += self.nonlinear_profile(z) * dz[idx] * np.exp(
                    -1j * dz[idx] * self._cumulative_delta_beta)

        logger.info("Calculation terminated")
        self._phi = self._cumulative_exponential  # * self.waveguide.dz
        if normalized:
            self._phi /= self.waveguide.length
        self._noise_length_product = abs(self._delta_beta0_profile).max() * self.waveguide.z[-1]
        return self.phi

    def plot_deltabeta_error(self, ax=None):
        # self.calculate_phasematching_error()
        if ax is None:
            plt.figure()
            ax = plt.gca()
        else:
            ax = ax
        ax.plot(self.waveguide.z, self._delta_beta0_profile)
        return ax


class SimplePhasematching2D(object):
    """
    Class for simple phasematching 2D
    """

    def __init__(self, waveguide: Union[Waveguide.Waveguide, Waveguide.RealisticWaveguide] = None,
                 n_red: Union[_REF_INDEX_TYPE0, _REF_INDEX_TYPE1] = None,
                 n_green: Union[_REF_INDEX_TYPE0, _REF_INDEX_TYPE1] = None,
                 n_blue: Union[_REF_INDEX_TYPE0, _REF_INDEX_TYPE1] = None,
                 order: int = 1, backpropagation: bool = False):
        self._waveguide = None
        self.waveguide = waveguide
        self._n_red = n_red
        self._n_green = n_green
        self._n_blue = n_blue
        self._order = order
        self._red_wavelength = None
        self._green_wavelength = None
        self._blue_wavelength = None
        self._pump_centre = None
        self._wavelength1 = None
        self._wavelength2 = None
        self._backpropagation = backpropagation
        if self._backpropagation:
            self.propagation_type = "backpropagation"
        else:
            self.propagation_type = "copropagation"
        self._WL_RED = None
        self._WL_GREEN = None
        self._WL_BLUE = None
        self._phi = None

    @property
    def waveguide(self):
        return self._waveguide

    @waveguide.setter
    def waveguide(self, waveguide: Union[Waveguide.Waveguide, Waveguide.RealisticWaveguide, None]):
        if isinstance(waveguide, (Waveguide.Waveguide, Waveguide.RealisticWaveguide)):
            self._waveguide = waveguide
        elif waveguide is None:
            warnings.warn("You have not provided any waveguide object to calculate the phasematching. If you don't "
                          "load  an external phasematching, nothing will work.")
        else:
            raise TypeError("You need to pass an object from the pynumpm.waveguide.Waveguide or "
                            "pynumpm.waveguide.RealisticWaveguide class.")

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value

    @property
    def phi(self):
        return self._phi

    @property
    def red_wavelength(self):
        return self._red_wavelength

    @red_wavelength.setter
    def red_wavelength(self, value):
        self._red_wavelength = value

    @property
    def green_wavelength(self):
        return self._green_wavelength

    @green_wavelength.setter
    def green_wavelength(self, value):
        self._green_wavelength = value

    @property
    def blue_wavelength(self):
        return self._blue_wavelength

    @blue_wavelength.setter
    def blue_wavelength(self, value):
        self._blue_wavelength = value

    @property
    def pump_centre(self):
        return self._pump_centre

    @property
    def wavelength1(self):
        return self._wavelength1

    @property
    def wavelength2(self):
        return self._wavelength2

    def load_phasematching(self, wavelength1, wavelength2, phasematching):
        """
        Function used to load externally the phasematching spectrum

        :param wavelength1:
        :param wavelength2:
        :param phasematching:
        :return:
        """
        logger = logging.getLogger(__name__)
        logger.debug("Loading externally the phasematching spectrum")
        self._wavelength1 = wavelength1
        self._wavelength2 = wavelength2
        self._phi = phasematching
        logger.info("User-provided phasematching has been loaded the phasematching spectrum")

    def set_wavelengths(self):
        logger = logging.getLogger(__name__)
        num_of_none = (self.red_wavelength is None) + \
                      (self.green_wavelength is None) + \
                      (self.blue_wavelength is None)
        logger.info("Number of wavelengths set to 'None': " + str(num_of_none))
        if num_of_none != 1:
            logger.info(
                "num_of_none != 1, the user has left more than 1 wavelength ranges unknown. An error will be raised.")
            logger.debug("Wavelengths set:", self.red_wavelength, self.green_wavelength, self.blue_wavelength,
                         num_of_none)
            raise ValueError(
                "Here you must be more precise. I need exactly 2 wavelength ranges, so only one wavelength can be none")
        else:
            for i in [self.red_wavelength, self.green_wavelength, self.blue_wavelength]:
                if i is not None:
                    if type(i) != np.ndarray:
                        raise ValueError("The wavelengths have to be either None or an array.")
            if self.red_wavelength is None:
                self._wavelength1 = self.green_wavelength
                self._wavelength2 = self.blue_wavelength
                self._pump_centre = (self.blue_wavelength.mean() ** -1 - self.green_wavelength.mean() ** -1) ** -1
                self._WL_GREEN, self._WL_BLUE = np.meshgrid(self.green_wavelength, self.blue_wavelength)
                self._WL_RED = (self._WL_BLUE ** -1 - self._WL_GREEN ** -1) ** -1
            elif self.green_wavelength is None:
                self._pump_centre = (self.blue_wavelength.mean() ** -1 - self.red_wavelength.mean() ** -1) ** -1
                self._wavelength1 = self.red_wavelength
                self._wavelength2 = self.blue_wavelength
                self._WL_RED, self._WL_BLUE = np.meshgrid(self.red_wavelength, self.blue_wavelength)
                self._WL_GREEN = (self._WL_BLUE ** -1 - self._WL_RED ** -1) ** -1
            elif self.blue_wavelength is None:
                self._wavelength1 = self.red_wavelength
                self._wavelength2 = self.green_wavelength
                self._pump_centre = (self.green_wavelength.mean() ** -1 + self.red_wavelength.mean() ** -1) ** -1
                self._WL_RED, self._WL_GREEN = np.meshgrid(self.red_wavelength, self.green_wavelength)
                self._WL_BLUE = (self._WL_RED ** -1 + self._WL_GREEN ** -1) ** -1
            else:
                logging.info("An error occurred while setting the wavelengths.")

            logging.debug("Wavelength matrices sizes: {0},{1},{2}".format(self._WL_RED.shape, self._WL_GREEN.shape,
                                                                          self._WL_BLUE.shape))

    def calculate_phasematching(self, normalized=True):
        """
        Function to calculate the phasematching spectrum.

        :param normalized:
        :return:
        """
        length = self.waveguide.length
        poling_period = self.waveguide.poling_period
        self.set_wavelengths()
        if self._backpropagation:
            deltabeta = 2 * np.pi * (self._n_blue(self._WL_BLUE * 1e6) / self._WL_BLUE -
                                     self._n_green(self._WL_GREEN * 1e6) / self._WL_GREEN +
                                     self._n_red(self._WL_RED * 1e6) / self._WL_RED -
                                     1 / poling_period)
        else:
            deltabeta = 2 * np.pi * (self._n_blue(self._WL_BLUE * 1e6) / self._WL_BLUE -
                                     self._n_green(self._WL_GREEN * 1e6) / self._WL_GREEN -
                                     self._n_red(self._WL_RED * 1e6) / self._WL_RED -
                                     1 / poling_period)

        self._phi = np.sinc(deltabeta * length / 2 / np.pi) * np.exp(-1j * deltabeta * length / 2)
        if not normalized:
            self._phi /= length
        return self.phi

    def plot(self, **kwargs):
        """
        Function to plot phasematching. Pass ax handle through "ax" to plot in a specified axis environment.

        :param kwargs:
        :return:
        """

        plot_intensity = kwargs.get("plot_intensity", True)
        ax = kwargs.get("ax", None)
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        phi = abs(self.phi)
        if plot_intensity:
            phi = phi ** 2

        cmap = kwargs.get("cmap", "viridis")
        vmin = kwargs.get("vmin", phi.min())
        vmax = kwargs.get("vmax", phi.max())

        im = ax.pcolormesh(self.wavelength1 * 1e9, self.wavelength2 * 1e9, phi, cmap=cmap, vmin=vmin,
                           vmax=vmax)
        if kwargs.get("cbar", True):
            cbar = plt.colorbar(im)
        else:
            cbar = None
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Wavelength [nm]")
        ax.set_title("Phasematching")
        fig = plt.gcf()
        d = {"fig": fig,
             "ax": ax,
             "im": im,
             "vmin": vmin,
             "vmax": vmax,
             "cbar": cbar}
        plt.tight_layout()
        return d


class Phasematching2D(SimplePhasematching2D):
    """
        Class to calculate the 2D-phasematching, i.e. having one fixed wavelength and scanning another one (the third is
        fixed due to energy conservation.
        The convention for labelling wavelength is

        .. math::

            |\\lambda_{red}| \\geq |\\lambda_{green}| \\geq |\\lambda_{blue}|

        i.e. according to their "energy".

        Accessible attributes of the class are

        :param waveguide: waveguide object
        :type waveguide: :class:`pynumpm.waveguide.RealisticWaveguide`
        :param phi: One dimensional phasematching spectrum (complex valued function)
        :type phi: :class:`~numpy:numpy.ndarray`
        :param n_red: refractive index for `red` wavelength. It must be a function of a function, i.e.
                      n(parameter)(wavelength). The wavelength **must** be in micron (usual convention for
                      Sellmeier's equations)
        :type n_red: function of function
        :param n_green: refractive index for `green` wavelength. It must be a function of a function, i.e.
                        n(parameter)(wavelength). The wavelength **must** be in micron (usual convention for
                        Sellmeier's equations)
        :type n_green: function of function
        :param n_blue: refractive index for "blue" wavelength. It must be a function of a function, i.e.
                       n(parameter)(wavelength). The wavelength **must** be in micron (usual convention for Sellmeier's
                       equations)
        :type n_blue: function of function
        :param int order: order of phasematching
        :param red_wavelength: None, Single float or vector of float, containing the "red" wavelengths, in meters.
        :type red_wavelength: :class:`numpy:numpy.ndarray`
        :param green_wavelength: None, Single float or vector of float, containing the "green" wavelengths, in meters.
        :type green_wavelength: :class:`numpy:numpy.ndarray`
        :param blue_wavelength: None, Single float or vector of float, containing the "blue" wavelengths, in meters.
        :type blue_wavelength: :class:`numpy:numpy.ndarray`
        :param wavelength1: Signal (scanning) wavelength of the process. It cannot be set, it is automatically set
                                  to be the lowest energetic wavelength of the user-defined wavelength ranges.
        :type wavelength1: :class:`numpy:numpy.ndarray`
        :param wavelength2: Output (scanning) wavelength of the process. It cannot be set, it is automatically set
                            to be the highest energetic wavelength of the user-defined wavelength ranges.
        :type wavelength2: :class:`numpy:numpy.ndarray`

        """

    def __init__(self, waveguide: Waveguide.RealisticWaveguide, n_red: _REF_INDEX_TYPE1, n_green: _REF_INDEX_TYPE1,
                 n_blue: _REF_INDEX_TYPE1, order: int = 1, backpropagation: bool = False):
        """
        Initialization of the class requires the following parameters:

        :param waveguide: Waveguide object. Use the Waveguide class in the Waveguide module to define this object.
        :type waveguide: :class:`~pynumpm.waveguide.RealisticWaveguide`
        :param n_red: refractive index for the "red" wavelength. It has to be a lambda function of a lambda function, i.e. n(variable_parameter)(wavelength in um)
        :param n_green: refractive index for the "green" wavelength. It has to be a lambda function of a lambda function, i.e. n(variable_parameter)(wavelength in um)
        :param n_blue: refractive index for the "blue" wavelength. It has to be a lambda function of a lambda function, i.e. n(variable_parameter)(wavelength in um)
        :param order: order of phasematching. Default: 1
        :param bool backpropagation: Set to True if it is necessary to calculate a backpropagation configuration.

        """
        super().__init__(waveguide=waveguide, n_red=n_red, n_green=n_green, n_blue=n_blue,
                         order=order, backpropagation=backpropagation)
        self.__nonlinear_profile_set = False
        self.__nonlinear_profile = None
        self.__cumulative_deltabeta = None
        self.__cumulative_exponential = None

    @property
    def waveguide(self):
        return self._waveguide

    @waveguide.setter
    def waveguide(self, waveguide: Waveguide.RealisticWaveguide):
        if isinstance(waveguide, Waveguide.RealisticWaveguide):
            self._waveguide = waveguide
        else:
            raise TypeError("You need to pass an object from the pynumpm.waveguide.Waveguide or "
                            "pynumpm.waveguide.RealisticWaveguide class.")

    @property
    def nonlinear_profile(self):
        return self.__nonlinear_profile

    def set_nonlinearity_profile(self, profile_type="constant", first_order_coeff=False, **kwargs):
        """
        Method to set the nonlinearity profile g(z), with either a constant profile or a variety of different windowing functions.

        :param str profile_type: Type of nonlinearity profile to consider. Possible options are [constant/gaussian/hamming/bartlett/hanning/blackman/kaiser].
        :param bool first_order_coeff: Select whether to simulate the reduction of efficiency due to quasi-phase matching or not.
        :param kwargs: Additional parameters to specify different variables of the `profile_type` used. Only effective if `profile_type` is *"gaussian"* or *"kaiser*.
        :return: The function returns the nonlinearity profile of the system.

        The different types of profile available are:

        * constant: Uniform nonlinear profile.
        * gaussian: :math:`g(z) = \\mathrm{e}^{-\\frac{(z-L/2)^2}{2\\sigma^2}}`.
        * hamming: :func:`numpy.hamming`
        * bartlett :func:`numpy.bartlett`
        * hanning: :func:`numpy.hanning`
        * blackman: :func:`numpy.blackman`
        * kaiser: :func:`numpy.kaiser`

        If *profile_type* is set to *"gaussian"*, then `**kwargs` accets the keyword `sigma_g_norm`, defining the standard
        deviation of the gaussian profile in units of the length (defauls to 0.5, i.e. L/2).
        If *profile_type* is set to *"kaiser"*, then `**kwargs` accepts the keyword `beta`, describing the
        :math:`\\beta` parameter of the Kaiser distribution.
        """
        logger = logging.getLogger(__name__)
        logger.info("Setting the nonlinear profile.")
        logger.info("Profile type: {pt}".format(pt=profile_type))
        if profile_type == "constant":
            logger.debug("Value of first_order_coeff: {foc}".format(foc=first_order_coeff))
            if first_order_coeff:
                g = lambda z: 2 / pi
            else:
                g = lambda z: 1.
        elif profile_type == "gaussian":
            if first_order_coeff:
                coeff = 2 / np.pi
            else:
                coeff = 1
            sigma_norm = kwargs.get("sigma_g_norm", 0.5)
            # I drop the phase in the exponential term of the delta beta in the main loop
            g = lambda z: coeff * np.exp(
                -(z - self.waveguide.length / 2.) ** 2 / (2 * (sigma_norm * self.waveguide.length) ** 2))
        elif profile_type == "hamming":
            g = np.hamming(len(self.waveguide.z))
            g = interp.interp1d(self.waveguide.z, g, kind="cubic")
        elif profile_type == "bartlett":
            g = np.bartlett(len(self.waveguide.z))
            g = interp.interp1d(self.waveguide.z, g, kind="cubic")
        elif profile_type == "hanning":
            g = np.hanning(len(self.waveguide.z))
            g = interp.interp1d(self.waveguide.z, g, kind="cubic")
        elif profile_type == "blackman":
            g = np.blackman(len(self.waveguide.z))
            g = interp.interp1d(self.waveguide.z, g, kind="cubic")
        elif profile_type == "kaiser":
            g = np.kaiser(len(self.waveguide.z), kwargs.get("beta", 1.))
            g = interp.interp1d(self.waveguide.z, g, kind="cubic")
        else:
            raise ValueError("The nonlinear profile {0} has not been implemented yet.".format(profile_type))
        self.__nonlinear_profile_set = True
        self.__nonlinear_profile = g
        return self.nonlinear_profile

    def __calculate_local_neff(self, posidx):
        local_parameter = self.waveguide.profile[posidx]
        try:
            n_red = self._n_red(local_parameter)
        except:
            raise RuntimeError("Something happened here! 'local parameter' was {0}".format(local_parameter))
        try:
            n_green = self._n_green(local_parameter)
        except:
            raise RuntimeError("Something happened here! 'local parameter' was {0}".format(local_parameter))
        try:
            n_blue = self._n_blue(local_parameter)
        except:
            raise RuntimeError("Something happened here! 'local parameter' was {0}".format(local_parameter))
        return n_red, n_green, n_blue

    def __calculate_delta_k(self, wl_red, wl_green, wl_blue, n_red, n_green, n_blue):
        if self.propagation_type == "copropagation":
            dd = 2 * pi * (n_blue(abs(wl_blue) * 1e6) / wl_blue -
                           n_green(abs(wl_green) * 1e6) / wl_green -
                           n_red(abs(wl_red) * 1e6) / wl_red -
                           float(self.order) / self.waveguide.poling_period)
            return dd
        elif self.propagation_type == "backpropagation":

            return 2 * pi * (n_blue(wl_blue * 1e6) / wl_blue -
                             n_green(wl_green * 1e6) / wl_green +
                             n_red(wl_red * 1e6) / wl_red -
                             float(self.order) / self.waveguide.poling_period)
        else:
            raise NotImplementedError("I don't know what you asked!\n" + self.propagation_type)

    def calculate_phasematching(self, normalized=True):
        """
        This function is the core of the class. It calculates the 2D phasematching of the process, scanning the two
        user-defined wavelength ranges.

        :param bool normalized: If True, the phasematching is limited in [0,1]. Otherwise, the maximum depends on the
                                waveguide length, Default: True

        :return: the complex-valued phasematching spectrum
        """
        logger = logging.getLogger(__name__)
        logger.info("Calculating phasematching")
        if self.waveguide.poling_structure_set:
            logger.info("Poling period is not set. Calculating from structure.")
        else:
            logger.info("Poling period is set. Calculating with constant poling structure.")

        if not self.__nonlinear_profile_set:
            self.set_nonlinearity_profile(profile_type="constant", first_order_coefficient=False)

        self.set_wavelengths()
        self.__cumulative_deltabeta = np.zeros(shape=(len(self.wavelength2), len(self.wavelength1)),
                                               dtype=complex)
        self.__cumulative_exponential = np.zeros(shape=self.__cumulative_deltabeta.shape, dtype=complex)
        dz = np.gradient(self.waveguide.z)
        for idx in tqdm(range(1, len(self.waveguide.z) - 1), ncols=100):
            z = self.waveguide.z[idx]
            # 1) retrieve the current parameter (width, thickness, ...)
            n_red, n_green, n_blue = self.__calculate_local_neff(idx)
            # 2) evaluate the current phasemismatch
            DK = self.__calculate_delta_k(self._WL_RED, self._WL_GREEN, self._WL_BLUE, n_red, n_green, n_blue)
            # 4) add the phasemismatch to the past phasemismatches (first summation, over the delta k)
            self.__cumulative_deltabeta += DK
            # 5) evaluate the (cumulative) exponential (second summation, over the exponentials)
            if self.waveguide.poling_structure_set:
                # TODO: rewrite this as a sum over the sinc, instead with rectangular approximation
                self.__cumulative_exponential += self.nonlinear_profile(z) * self.waveguide.poling_structure[idx] * dz[
                    idx] * \
                                                 (np.exp(-1j * dz[idx] * self.__cumulative_deltabeta) -
                                                  np.exp(-1j * dz[idx] * (self.__cumulative_deltabeta - DK)))
            else:
                self.__cumulative_exponential += self.nonlinear_profile(z) * dz[idx] * np.exp(
                    -1j * dz[idx] * self.__cumulative_deltabeta)
        if normalized:
            self._phi = 1 / self.waveguide.length * self.__cumulative_exponential
        logger.info("Calculation terminated")
        return self.phi

    def slice_phasematching(self, const_wl):
        """
        Slice the phasematching. The function interpolates the phasematching in the direction of the wavelength
        to be kept fixed (**fix_wl**) and evaluating the interpolation at the value provided (**value**). In this way,
        it is possible to cut along wavelength not present in the original grid.

        :param float const_wl: Constant wavelength where to cut the phasematching. The program detects whether it is
                               within the signal or idler range.

        :return wl, phi: scanning wavelength and interpolated complex-valued phasematching spectrum.
        """
        # TODO: What happens in case of wavelength degeneracy?
        logger = logging.getLogger(__name__)
        f_real = interp.interp2d(self.wavelength1, self.wavelength2, np.real(self.phi), kind='linear')
        f_imag = interp.interp2d(self.wavelength1, self.wavelength2, np.imag(self.phi), kind='linear')
        logger.debug("Constant wl: " + str(const_wl))
        if self.wavelength1.min() <= const_wl <= self.wavelength1.max():
            wl = self.wavelength2
            phi = f_real(const_wl, self.wavelength2) + 1j * f_imag(const_wl, self.wavelength2)
        elif self.wavelength2.min() <= const_wl <= self.wavelength2.max():
            wl = self.wavelength1
            phi = f_real(self.wavelength1, const_wl) + 1j * f_imag(self.wavelength1, const_wl)
        else:
            raise NotImplementedError(
                "My dumb programmer hasn't implemented the slice along a line not parallel to the axes...")
        return wl, phi

    def extract_max_phasematching_curve(self, ax=None, **kwargs):
        """
        Extract the curve of max phasematching. Useful to estimate GVM.

        :param ax: Axis handle.
        :return:
        """
        # TODO: this function has been reimplemented. There is an offset sometimes in between the reconstructed peak
        #  and the real curve

        signal = self.wavelength1
        idler = []

        for idx, wl in enumerate(signal):
            idl, pm_cut = self.slice_phasematching(const_wl=wl)
            idler.append(idl[pm_cut.argmax()])
        IDLER = np.array(idler)
        p_idler = np.polyfit(signal, IDLER, deg=2)
        idler = np.polyval(p_idler, signal)
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        self.plot(ax=ax)
        ax.plot(signal * 1e9, idler * 1e9, "k", lw=3)
        plt.tight_layout()
        return signal, idler
