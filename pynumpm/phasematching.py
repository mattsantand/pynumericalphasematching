# coding=utf-8
"""
.. module:: phasematching.py
.. moduleauthor:: Matteo Santandrea <matteo.santandrea@upb.de>
   :platform: Windows (tested), Unix
   :synopsis: Module to calculate the phasematching of a given waveguide, as specified by the Waveguide class.

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
from pynumpm import jsa


# TODO: replace rectangular integration in Phasematching 1D and 2D with the sinc (the currect integration)

class PhasematchingDeltaBeta(object):
    """
    This class is used to simulate phasematching of systems considering only the wavevector mismatch (:math:`\Delta\\beta`).
    The accessible attributes in this class are:

    :param waveguide: Waveguide object
    :type waveguide: :class:`~pynumpm.waveguide.Waveguide`
    :param deltabeta: vector containing the values of :math:`\Delta\\beta` needed to calculate the phasematching
    :type deltabeta: :class:`~numpy:numpy.ndarray`
    :param phi: Vector containing the phasematching spectrum
    :type phi: :class:`~numpy:numpy.ndarray`
    :param noise_length_product: Value of the noise length product :math:`\sigma L`
    :type noise_length_product: float

    """

    def __init__(self, waveguide):
        """
        Initialization of the class requires the following parameter:

        :param waveguide: Waveguide object, as provided by the class waveguide.
        :type waveguide: :class:`~pynumpm.waveguide.Waveguide`

        """
        self.__waveguide = waveguide
        self.__deltabeta = None
        self.__phi = 0
        self.__cumulative_delta_beta = None
        self.__cumulative_exp = None
        self.__cumulative_sinc = None
        self.__noise_length_product = None

    @property
    def waveguide(self):
        return self.__waveguide

    @property
    def deltabeta(self):
        return self.__deltabeta

    @deltabeta.setter
    def deltabeta(self, value):
        self.__deltabeta = value

    @property
    def phi(self):
        return self.__phi

    @phi.setter
    def phi(self, value):
        self.__phi = value

    @property
    def noise_length_product(self):
        return self.__noise_length_product

    def load_waveguide(self, waveguide):
        self.__waveguide = waveguide

    def calculate_phasematching(self, normalized=False):
        """
        Function that calculates the phasematching.
        Prior to the evaluation of the phasematching, it is necessary to set the :math:`\Delta\\beta` vector by
        assigning it to the variable `deltabeta`.

        :param deltabeta: Vector describing the deltabeta space to be scanned.
        :type deltabeta: numpy.ndarray
        :param normalized: Sets the normalization of the phasematching. If the normalization is on (True), the phasematching will be normalized to the unit length (i.e., the maximum will be in [0,1]). Default: False.
        :type normalized: bool
        :return: the function returns the complex-valued phasematching spectrum.

        """
        logger = logging.getLogger(__name__)
        if self.deltabeta is None:
            raise ValueError("You need to define a delta beta range.")
        logger.info("Calculating the phasematching.")
        self.__cumulative_delta_beta = np.zeros(shape=len(self.deltabeta), dtype=complex)
        self.__cumulative_exp = np.ones(shape=len(self.deltabeta), dtype=complex)
        self.__cumulative_sinc = np.zeros(shape=len(self.deltabeta), dtype=complex)

        for i in range(len(self.waveguide.z) - 1):
            dz = self.waveguide.z[i + 1] - self.waveguide.z[i]
            this_deltabeta = self.deltabeta + self.waveguide.profile[i] - 2 * np.pi / self.waveguide.poling_period
            x = this_deltabeta * dz / 2
            self.__cumulative_sinc += dz * np.sinc(x / np.pi) * np.exp(1j * x) * np.exp(
                1j * self.__cumulative_delta_beta)
            self.__cumulative_delta_beta += this_deltabeta * dz
        self.__phi = self.__cumulative_sinc
        if normalized:
            self.__phi /= self.waveguide.z[-1]
        self.__noise_length_product = abs(self.waveguide.profile).max() * self.waveguide.length
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
                "I'd really like to plot something nice, but you have not calculated the phasematching yet, to this would only be a white canvas.")
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
            print(x, y, text)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(x, y, text, bbox=props)
        return ax

    def calculate_integral(self):
        """
        Function to calculate the integral of the phasematching curve. It uses the function `simps` from the scipy module.

        :return: Intensity integral
        """
        return simps(abs(self.phi) ** 2, self.deltabeta)


class Phasematching1D(object):
    """
    Class to calculate the 1D-phasematching, i.e. having one fixed wavelength and scanning another one (the third is
    fixed due to energy conservation.
    The convention for labelling wavelength is

    .. math::

        |\\lambda_{red}| \\geq |\\lambda_{green}| \\geq |\\lambda_{blue}|

    i.e. according to their "energy".

    Accessible attributes of the class are

    :param waveguide: waveguide object
    :type waveguide: :class:`pynumpm.waveguide.Waveguide`
    :param phi: One dimensional phasematching spectrum (complex valued function)
    :type phi: :class:`~numpy:numpy.ndarray`
    :param n_red: refractive index for `red` wavelength. It must be a function of a function, i.e.
        n(parameter)(wavelength). The wavelength **must** be in micron (usual convention for Sellmeier's equations)
    :type n_red: function of function
    :param n_green: refractive index for `green` wavelength. It must be a function of a function, i.e.
        n(parameter)(wavelength). The wavelength **must** be in micron (usual convention for Sellmeier's equations)
    :type n_green: function of function
    :param n_blue: refractive index for "blue" wavelength. It must be a function of a function, i.e.
        n(parameter)(wavelength). The wavelength **must** be in micron (usual convention for Sellmeier's equations)
    :type n_blue: function of function
    :param int order: order of phasematching
    :param str propagation_type:  copropagating or counterpropagating
    :param red_wavelength: None, Single float or vector of float, containing the "red" wavelengths, in meters.
    :type red_wavelength: :class:`numpy:numpy.ndarray`
    :param green_wavelength: None, Single float or vector of float, containing the "green" wavelengths, in meters.
    :type green_wavelength: :class:`numpy:numpy.ndarray`
    :param blue_wavelength: None, Single float or vector of float, containing the "blue" wavelengths, in meters.
    :type blue_wavelength: :class:`numpy:numpy.ndarray`
    :param input_wavelength: Input (scanning) wavelength of the process. It cannot be set, it is automatically detected.
    :type input_wavelength: :class:`numpy:numpy.ndarray`
    :param output_wavelength: Output (scanning) wavelength of the process. It cannot be set, it is automatically detected.
    :type output_wavelength: :class:`numpy:numpy.ndarray`

    """

    def __init__(self, waveguide, n_red, n_green, n_blue, order=1, backpropagation=False):
        """
        Initialization of the class requires the following parameters:

        :param waveguide: Waveguide object. Use the Waveguide class in the Waveguide module to define this object.
        :type waveguide: :class:`~pynumpm.waveguide.Waveguide`
        :param n_red: refractive index for the "red" wavelength. It has to be a lambda function of a lambda function, i.e. n(variable_parameter)(wavelength in um)
        :param n_green: refractive index for the "green" wavelength. It has to be a lambda function of a lambda function, i.e. n(variable_parameter)(wavelength in um)
        :param n_blue: refractive index for the "blue" wavelength. It has to be a lambda function of a lambda function, i.e. n(variable_parameter)(wavelength in um)
        :param order: order of phasematching. Default: 1
        :param bool backpropagation: Set to True if it is necessary to calculate a backpropagation configuration.

        """
        self.__waveguide = waveguide
        self.__phi = None
        self.__wavelength_set = False
        self.__n_red = n_red
        self.__n_green = n_green
        self.__n_blue = n_blue
        # ====================================================
        self.order = order
        # TODO: check if and how the poling order interferes when the poling structure is set
        self.process = None
        self.__red_wavelength = None
        self.__green_wavelength = None
        self.__blue_wavelength = None
        self.__input_wavelength = None
        self.__output_wavelength = None
        if backpropagation:
            self.propagation_type = "backpropagation"
        else:
            self.propagation_type = "copropagation"
        self.__nonlinear_profile_set = False
        self.__nonlinear_profile = None
        self.__noise_length_product = None
        self.scanning_wavelength = None
        self.__cumulative_delta_beta = None
        self.__cumulative_exponential = None
        self.__delta_beta0_profile = None
        self.__lamr0 = None
        self.__lamg0 = None
        self.__lamb0 = None

    @property
    def nonlinear_profile(self):
        return self.__nonlinear_profile

    @property
    def waveguide(self):
        return self.__waveguide

    @property
    def phi(self):
        return self.__phi

    @property
    def n_red(self):
        return self.__n_red

    @property
    def n_green(self):
        return self.__n_green

    @property
    def n_blue(self):
        return self.__n_red

    @property
    def red_wavelength(self):
        return self.__red_wavelength

    @red_wavelength.setter
    def red_wavelength(self, value):
        self.__red_wavelength = value

    @property
    def green_wavelength(self):
        return self.__green_wavelength

    @green_wavelength.setter
    def green_wavelength(self, value):
        self.__green_wavelength = value

    @property
    def blue_wavelength(self):
        return self.__blue_wavelength

    @blue_wavelength.setter
    def blue_wavelength(self, value):
        self.__blue_wavelength = value

    @property
    def input_wavelength(self):
        return self.__input_wavelength

    @property
    def output_wavelength(self):
        return self.__output_wavelength

    @property
    def noise_length_product(self):
        return self.__noise_length_product

    def load_waveguide(self, waveguide):
        self.__waveguide = waveguide

    def __set_wavelengths(self):
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
            self.__input_wavelength = self.red_wavelength
            self.__output_wavelength = self.blue_wavelength
        elif num_of_none == 1:
            logger.info("Calculating wavelengths for sfg/dfg")
            self.process = "sfg/dfg"
            if self.red_wavelength is None:
                if type(self.green_wavelength) == np.ndarray:
                    self.__input_wavelength = self.green_wavelength
                    logger.info("The input wavelength is the green")
                else:
                    self.__input_wavelength = self.blue_wavelength
                    logger.info("The input wavelength is the blue")
                self.red_wavelength = (self.blue_wavelength ** -1 - self.green_wavelength ** -1) ** -1
                self.__output_wavelength = self.red_wavelength
            elif self.green_wavelength is None:
                if type(self.red_wavelength) == np.ndarray:
                    self.__input_wavelength = self.red_wavelength
                    logger.info("The input wavelength is the red")
                else:
                    self.__input_wavelength = self.blue_wavelength
                    logger.info("The input wavelength is the blue")
                self.green_wavelength = (self.blue_wavelength ** -1 - self.red_wavelength ** -1) ** -1
                self.__output_wavelength = self.green_wavelength
            elif self.blue_wavelength is None:
                if type(self.red_wavelength) == np.ndarray:
                    self.__input_wavelength = self.red_wavelength
                    logger.info("The input wavelength is the red")
                else:
                    self.__input_wavelength = self.green_wavelength
                    logger.info("The input wavelength is the green")
                self.blue_wavelength = (self.red_wavelength ** -1 + self.green_wavelength ** -1) ** -1
                self.__output_wavelength = self.blue_wavelength
            else:
                logger.error("An error occurred in __set_wavelengths. When setting the SFG/DFG wavelengths, "
                             "all the wavelengths are set but the number of none is 1."
                             "Red wavelength: {r}\nGreen wavelength: {g}\nBlue wavelength: {b}".format(
                    r=self.red_wavelength,
                    g=self.green_wavelength,
                    b=self.blue_wavelength))
                raise ValueError("Something unexpected happened in set_wavelength. "
                                 "Check the log please and chat with the developer.")
        self.__wavelength_set = True
        self.__lamr0 = self.red_wavelength.mean() if type(self.red_wavelength) == np.ndarray else self.red_wavelength
        self.__lamg0 = self.green_wavelength.mean() if type(
            self.green_wavelength) == np.ndarray else self.green_wavelength
        self.__lamb0 = self.blue_wavelength.mean() if type(self.blue_wavelength) == np.ndarray else self.blue_wavelength
        return self.red_wavelength, self.green_wavelength, self.blue_wavelength

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

    def plot_nonlinearity_profile(self):
        """
        Function to plot the nonlinearity profile

        """
        x = self.waveguide.z
        y = self.nonlinear_profile(self.waveguide.z)
        plt.plot(x, y)

    def __calculate_local_neff(self, posidx):
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

    def __calculate_delta_k(self, wl_red, wl_green, wl_blue, n_red, n_green, n_blue):
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
        This function is the core of the class. It calculates the phasematching of the process, considering one
        wavelength fixed and scanning the other two.

        :param bool normalized: If True, the phasematching is limited in [0,1]. Otherwise, the maximum depends on the
        waveguide length, Default: True

        :return: the complex-valued phasematching spectrum
        """
        if not self.__wavelength_set:
            self.__set_wavelengths()

        logger = logging.getLogger(__name__)
        logger.info("Calculating phasematching")

        if not self.__nonlinear_profile_set:
            self.set_nonlinearity_profile(profile_type="constant", first_order_coefficient=False)
        if self.waveguide.poling_structure_set:
            logger.info("Poling period is not set. Calculating from structure.")
        else:
            logger.info("Poling period is set. Calculating with constant poling structure.")

        tmp_dk = self.__calculate_delta_k(self.red_wavelength, self.green_wavelength, self.blue_wavelength,
                                          *self.__calculate_local_neff(0))
        self.__cumulative_delta_beta = np.zeros(shape=tmp_dk.shape)
        self.__cumulative_exponential = np.zeros(shape=self.__cumulative_delta_beta.shape, dtype=complex)
        logger.debug("Shape cumulative deltabeta:" + str(self.__cumulative_delta_beta.shape))
        logger.debug("Shape cum_exp:" + str(self.__cumulative_exponential.shape))
        self.__delta_beta0_profile = np.nan * np.ones(shape=self.waveguide.z.shape)
        dz = np.diff(self.waveguide.z)
        for idx, z in enumerate(self.waveguide.z[:-1]):
            # 1) retrieve the current parameter (width, thickness, ...)
            n_red, n_green, n_blue = self.__calculate_local_neff(idx)
            # 2) evaluate the current phasemismatch
            DK = self.__calculate_delta_k(self.red_wavelength, self.green_wavelength, self.blue_wavelength,
                                          n_red, n_green, n_blue)
            self.__delta_beta0_profile[idx] = self.__calculate_delta_k(self.__lamr0, self.__lamg0, self.__lamb0, n_red,
                                                                       n_green,
                                                                       n_blue)
            # 4) add the phasemismatch to the past phasemismatches (first summation, over the delta k)
            self.__cumulative_delta_beta += DK
            # 5) evaluate the (cumulative) exponential (second summation, over the exponentials)
            if self.waveguide.poling_structure_set:
                self.__cumulative_exponential += self.nonlinear_profile(z) * dz[idx] * self.waveguide.poling_structure[
                    idx] * \
                                                 (np.exp(-1j * dz[idx] * self.__cumulative_delta_beta) -
                                                  np.exp(-1j * dz[idx] * (self.__cumulative_delta_beta - DK)))
            else:
                self.__cumulative_exponential += self.nonlinear_profile(z) * dz[idx] * np.exp(
                    -1j * dz[idx] * self.__cumulative_delta_beta)

        logger.info("Calculation terminated")
        self.__phi = self.__cumulative_exponential  # * self.waveguide.dz
        if normalized:
            self.__phi /= self.waveguide.length
        self.__noise_length_product = abs(self.__delta_beta0_profile).max() * self.waveguide.z[-1]
        return self.phi

    def calculate_integral(self):
        """
        Calculate the phasematching intensity integral

        :return: the phasematching intensity integral
        """
        return simps(abs(self.phi) ** 2, self.scanning_wavelength)

    def plot_deltabeta_error(self, ax=None):
        # self.calculate_phasematching_error()
        if ax is None:
            plt.figure()
            ax = plt.gca()
        else:
            ax = ax
        ax.plot(self.waveguide.z, self.__delta_beta0_profile)
        return ax

    def plot(self, ax=None, plot_intensity=True, plot_input=True, **kwargs):
        """
        Plot the phasematching intensity/amplitude.

        :param ax: Axis handle for the plot. If None, plots in a new figure. Default is None.
        :param bool plot_intensity: Set to True to plot the intensity profile, False to plot the amplitude and phase.
        Default to True.
        :param bool plot_input: Select the x axis for the plot. If True, use the `input_wavelength` as input, otherwise use `output_wavelength`.
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
        return fig, ax


class Simple2DPhasematching(object):
    def __init__(self, waveguide, n_red, n_green, n_blue, order=1, backpropagation=False):
        self.__waveguide = waveguide
        self.__n_red = n_red
        self.__n_green = n_green
        self.__n_blue = n_blue
        # ====================================================
        self.order = order
        # self.process = kwargs.get("process", "PDC").lower()
        self.__red_wavelength = None
        self.__green_wavelength = None
        self.__blue_wavelength = None
        self.__signal_wavelength = None
        self.__idler_wavelength = None
        self.__backpropagation = backpropagation
        if self.__backpropagation:
            self.propagation_type = "backpropagation"
        else:
            self.propagation_type = "copropagation"

        self.__phi = None

    @property
    def phi(self):
        return self.__phi

    @property
    def red_wavelength(self):
        return self.__red_wavelength

    @red_wavelength.setter
    def red_wavelength(self, value):
        self.__red_wavelength = value

    @property
    def green_wavelength(self):
        return self.__green_wavelength

    @green_wavelength.setter
    def green_wavelength(self, value):
        self.__green_wavelength = value

    @property
    def blue_wavelength(self):
        return self.__blue_wavelength

    @blue_wavelength.setter
    def blue_wavelength(self, value):
        self.__blue_wavelength = value

    def __set_wavelengths(self):
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
                self.signal_wavelength = self.green_wavelength
                self.idler_wavelength = self.blue_wavelength
                self.pump_centre = (self.blue_wavelength.mean() ** -1 - self.green_wavelength.mean() ** -1) ** -1
                self.__WL_GREEN, self.__WL_BLUE = np.meshgrid(self.green_wavelength, self.blue_wavelength)
                self.__WL_RED = (self.__WL_BLUE ** -1 - self.__WL_GREEN ** -1) ** -1
            elif self.green_wavelength is None:
                self.pump_centre = (self.blue_wavelength.mean() ** -1 - self.red_wavelength.mean() ** -1) ** -1
                self.signal_wavelength = self.red_wavelength
                self.idler_wavelength = self.blue_wavelength
                self.__WL_RED, self.__WL_BLUE = np.meshgrid(self.red_wavelength, self.blue_wavelength)
                self.__WL_GREEN = (self.__WL_BLUE ** -1 - self.__WL_RED ** -1) ** -1
            elif self.blue_wavelength is None:
                self.signal_wavelength = self.red_wavelength
                self.idler_wavelength = self.green_wavelength
                self.pump_centre = (self.green_wavelength.mean() ** -1 + self.red_wavelength.mean() ** -1) ** -1
                self.__WL_RED, self.__WL_GREEN = np.meshgrid(self.red_wavelength, self.green_wavelength)
                self.__WL_BLUE = (self.__WL_RED ** -1 + self.__WL_GREEN ** -1) ** -1
            else:
                logging.info("An error occurred while setting the wavelengths.")

            logging.debug("Wavelength matrices sizes: {0},{1},{2}".format(self.__WL_RED.shape, self.__WL_GREEN.shape,
                                                                          self.__WL_BLUE.shape))

    def calculate_phasematching(self, normalized=True):
        length = self.__waveguide.z.max() - self.__waveguide.z.min()
        poling_period = self.__waveguide.poling_period
        self.__set_wavelengths()
        if self.__backpropagation:
            deltabeta = 2 * np.pi * (self.__n_blue(self.__WL_BLUE * 1e6) / self.__WL_BLUE -
                                     self.__n_green(self.__WL_GREEN * 1e6) / self.__WL_GREEN +
                                     self.__n_red(self.__WL_RED * 1e6) / self.__WL_RED -
                                     1 / poling_period)
        else:
            deltabeta = 2 * np.pi * (self.__n_blue(self.__WL_BLUE * 1e6) / self.__WL_BLUE -
                                     self.__n_green(self.__WL_GREEN * 1e6) / self.__WL_GREEN -
                                     self.__n_red(self.__WL_RED * 1e6) / self.__WL_RED -
                                     1 / poling_period)

        self.__phi = np.sinc(deltabeta * length / 2 / np.pi) * np.exp(-1j * deltabeta * length / 2)
        if not normalized:
            self.__phi /= length
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

        im = ax.pcolormesh(self.signal_wavelength * 1e9, self.idler_wavelength * 1e9, phi, cmap=cmap, vmin=vmin,
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
        return d


class Phasematching2D(object):
    """
        Class to calculate the 2D-phasematching, i.e. having one fixed wavelength and scanning another one (the third is
        fixed due to energy conservation.
        The convention for labelling wavelength is

        .. math::

            |\\lambda_{red}| \\geq |\\lambda_{green}| \\geq |\\lambda_{blue}|

        i.e. according to their "energy".

        Accessible attributes of the class are

        :param waveguide: waveguide object
        :type waveguide: :class:`pynumpm.waveguide.Waveguide`
        :param phi: One dimensional phasematching spectrum (complex valued function)
        :type phi: :class:`~numpy:numpy.ndarray`
        :param n_red: refractive index for `red` wavelength. It must be a function of a function, i.e.
            n(parameter)(wavelength). The wavelength **must** be in micron (usual convention for Sellmeier's equations)
        :type n_red: function of function
        :param n_green: refractive index for `green` wavelength. It must be a function of a function, i.e.
            n(parameter)(wavelength). The wavelength **must** be in micron (usual convention for Sellmeier's equations)
        :type n_green: function of function
        :param n_blue: refractive index for "blue" wavelength. It must be a function of a function, i.e.
            n(parameter)(wavelength). The wavelength **must** be in micron (usual convention for Sellmeier's equations)
        :type n_blue: function of function
        :param int order: order of phasematching
        :param str propagation_type:  copropagating or counterpropagating
        :param red_wavelength: None, Single float or vector of float, containing the "red" wavelengths, in meters.
        :type red_wavelength: :class:`numpy:numpy.ndarray`
        :param green_wavelength: None, Single float or vector of float, containing the "green" wavelengths, in meters.
        :type green_wavelength: :class:`numpy:numpy.ndarray`
        :param blue_wavelength: None, Single float or vector of float, containing the "blue" wavelengths, in meters.
        :type blue_wavelength: :class:`numpy:numpy.ndarray`
        :param signal_wavelength: Signal (scanning) wavelength of the process. It cannot be set, it is automatically set
        to be the lowest energetic wavelength of the user-defined wavelength ranges.
        :type signal_wavelength: :class:`numpy:numpy.ndarray`
        :param idler_wavelength: Output (scanning) wavelength of the process. It cannot be set, it is automatically set
        to be the highest energetic wavelength of the user-defined wavelength ranges.
        :type idler_wavelength: :class:`numpy:numpy.ndarray`

        """

    def __init__(self, waveguide, n_red, n_green, n_blue, order=1, backpropagation=False):
        """
        Initialization of the class requires the following parameters:

        :param waveguide: Waveguide object. Use the Waveguide class in the Waveguide module to define this object.
        :type waveguide: :class:`~pynumpm.waveguide.Waveguide`
        :param n_red: refractive index for the "red" wavelength. It has to be a lambda function of a lambda function, i.e. n(variable_parameter)(wavelength in um)
        :param n_green: refractive index for the "green" wavelength. It has to be a lambda function of a lambda function, i.e. n(variable_parameter)(wavelength in um)
        :param n_blue: refractive index for the "blue" wavelength. It has to be a lambda function of a lambda function, i.e. n(variable_parameter)(wavelength in um)
        :param order: order of phasematching. Default: 1
        :param bool backpropagation: Set to True if it is necessary to calculate a backpropagation configuration.

        """
        self.waveguide = waveguide
        # these n_xxx functions are function that accept one parameter and provide a lambda function (n = n(wavelength))
        self.__n_red = n_red
        self.__n_green = n_green
        self.__n_blue = n_blue
        # ====================================================
        self.order = order
        # self.process = kwargs.get("process", "PDC").lower()
        self.__red_wavelength = None
        self.__green_wavelength = None
        self.__blue_wavelength = None
        self.__signal_wavelength = None
        self.__idler_wavelength = None
        if backpropagation:
            self.propagation_type = "backpropagation"
        else:
            self.propagation_type = "copropagation"
        self.__nonlinear_profile_set = False
        self.__nonlinear_profile = None

    @property
    def signal_wavelength(self):
        return self.__signal_wavelength

    @signal_wavelength.setter
    def signal_wavelength(self, value):
        self.__signal_wavelength = value

    @property
    def nonlinear_profile(self):
        return self.__nonlinear_profile

    @property
    def idler_wavelength(self):
        return self.__idler_wavelength

    @idler_wavelength.setter
    def idler_wavelength(self, value):
        self.__idler_wavelength = value

    @property
    def red_wavelength(self):
        return self.__red_wavelength

    @red_wavelength.setter
    def red_wavelength(self, value):
        self.__red_wavelength = value

    @property
    def green_wavelength(self):
        return self.__green_wavelength

    @green_wavelength.setter
    def green_wavelength(self, value):
        self.__green_wavelength = value

    @property
    def blue_wavelength(self):
        return self.__blue_wavelength

    @blue_wavelength.setter
    def blue_wavelength(self, value):
        self.__blue_wavelength = value

    @property
    def n_red(self):
        return self.__n_red

    @property
    def n_green(self):
        return self.__n_green

    @property
    def n_blue(self):
        return self.__n_blue

    def load_waveguide(self, waveguide):
        self.__waveguide = waveguide

    def __set_wavelengths(self):
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
                self.signal_wavelength = self.green_wavelength
                self.idler_wavelength = self.blue_wavelength
                self.pump_centre = (self.blue_wavelength.mean() ** -1 - self.green_wavelength.mean() ** -1) ** -1
                self.__WL_GREEN, self.__WL_BLUE = np.meshgrid(self.green_wavelength, self.blue_wavelength)
                self.__WL_RED = (self.__WL_BLUE ** -1 - self.__WL_GREEN ** -1) ** -1
            elif self.green_wavelength is None:
                self.pump_centre = (self.blue_wavelength.mean() ** -1 - self.red_wavelength.mean() ** -1) ** -1
                self.signal_wavelength = self.red_wavelength
                self.idler_wavelength = self.blue_wavelength
                self.__WL_RED, self.__WL_BLUE = np.meshgrid(self.red_wavelength, self.blue_wavelength)
                self.__WL_GREEN = (self.__WL_BLUE ** -1 - self.__WL_RED ** -1) ** -1
            elif self.blue_wavelength is None:
                self.signal_wavelength = self.red_wavelength
                self.idler_wavelength = self.green_wavelength
                self.pump_centre = (self.green_wavelength.mean() ** -1 + self.red_wavelength.mean() ** -1) ** -1
                self.__WL_RED, self.__WL_GREEN = np.meshgrid(self.red_wavelength, self.green_wavelength)
                self.__WL_BLUE = (self.__WL_RED ** -1 + self.__WL_GREEN ** -1) ** -1
            else:
                logging.info("An error occurred while setting the wavelengths.")

            logging.debug("Wavelength matrices sizes: {0},{1},{2}".format(self.__WL_RED.shape, self.__WL_GREEN.shape,
                                                                          self.__WL_BLUE.shape))

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

        self.__set_wavelengths()
        self.__cumulative_deltabeta = np.zeros(shape=(len(self.idler_wavelength), len(self.signal_wavelength)),
                                               dtype=complex)
        self.__cumulative_exponential = np.zeros(shape=self.__cumulative_deltabeta.shape, dtype=complex)
        dz = np.diff(self.waveguide.z)
        for idx, z in enumerate(self.waveguide.z[:-1]):
            # 1) retrieve the current parameter (width, thickness, ...)
            n_red, n_green, n_blue = self.__calculate_local_neff(idx)
            # 2) evaluate the current phasemismatch
            DK = self.__calculate_delta_k(self.__WL_RED, self.__WL_GREEN, self.__WL_BLUE, n_red, n_green, n_blue)
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
            self.phi = 1 / self.waveguide.length * self.__cumulative_exponential
        logger.info("Calculation terminated")
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

        im = ax.pcolormesh(self.signal_wavelength * 1e9, self.idler_wavelength * 1e9, phi, cmap=cmap, vmin=vmin,
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
        return d

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
        f_real = interp.interp2d(self.signal_wavelength, self.idler_wavelength, np.real(self.phi), kind='linear')
        f_imag = interp.interp2d(self.signal_wavelength, self.idler_wavelength, np.imag(self.phi), kind='linear')
        logger.debug("Constant wl: " + str(const_wl))
        if self.signal_wavelength.min() <= const_wl <= self.signal_wavelength.max():
            wl = self.idler_wavelength
            phi = f_real(const_wl, self.idler_wavelength) + 1j * f_imag(const_wl, self.idler_wavelength)
        elif self.idler_wavelength.min() <= const_wl <= self.idler_wavelength.max():
            wl = self.signal_wavelength
            phi = f_real(self.signal_wavelength, const_wl) + 1j * f_imag(self.signal_wavelength, const_wl)
        else:
            raise NotImplementedError(
                "MY dumb programmer hasn't implemented the slice along a line not parallel to the axes...")
        return wl, phi

    def extract_max_phasematching_curve(self, ax=None, **kwargs):
        """
        Extract the curve of max phasematching. Useful to estimate GVM.

        :param ax: Axis handle.
        :return:
        """
        # TODO: this function has been reimplemented. There is an offset sometimes in between the reconstructed peak
        #  and the real curve

        signal = self.signal_wavelength
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
        return signal, idler
