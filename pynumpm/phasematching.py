# coding=utf-8
"""
Module to calculate the phasematching of a given waveguide, as specified by the Waveguide class.

It can calculate different types of phasematching:
    * :class:`PhasematchingDeltaBeta`: 1D phasematching spectrum, given the wavevector mismatch range to be analyzed.
    * :class:`Phasematching1D`: 1D phasematching spectrum, given the wavelength range to be analyzed and the Sellmeier equations.
    * :class:`Phasematching2D`: 2D phasematching spectrum, given the wavelength range to be analyzed and the Sellmeier equations.

"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
import scipy.interpolate as interp
import copy
import pprint
from scipy.integrate import simps


# TODO: replace rectangular integration with the sinc (the currect integration)

class PhasematchingDeltaBeta(object):
    """
    This class is used to simulate phasematching of systems considering only the wavevector mismatch (:math:`\Delta\\beta).
    """

    def __init__(self, waveguide):
        """

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
        """
        Waveguide object, as defined by the user during initialization.
        """
        return self.__waveguide

    @property
    def deltabeta(self):
        """
        Array of the :math:`\Delta\\beta` values, as defined by the user.
        """
        return self.__deltabeta

    @deltabeta.setter
    def deltabeta(self, value):
        self.__deltabeta = value
        logger = logging.getLogger(__name__)
        logger.debug("Delta beta vector set")

    @property
    def phi(self):
        """
        Phasematching spectrum (complex valued numpy.ndarray)

        """
        return self.__phi

    @phi.setter
    def phi(self, value):
        self.__phi = value

    @property
    def noise_length_product(self):
        return self.__noise_length_product

    def calculate_phasematching(self, deltabeta, normalized=False):
        """
        Function that calculates the phasematching, given the vector of deltabeta (wavevector mismatch)

        :param deltabeta: Vector describing the deltabeta space to be scanned.
        :type deltabeta: numpy.ndarray
        :param normalized: Sets the normalization of the phasematching. If the normalization is on (True), the phasematching will be normalized to the unit length (i.e., the maximum will be in [0,1]). Default: False.
        :type normalized: bool
        :return: the function returns the complex-valued phasematching spectrum.
        """
        logger = logging.getLogger(__name__)
        self.deltabeta = deltabeta
        logger.info("Calculating the phasematching.")
        self.__cumulative_delta_beta = np.zeros(shape=len(self.deltabeta), dtype=complex)
        self.__cumulative_exp = np.ones(shape=len(self.deltabeta), dtype=complex)
        self.__cumulative_sinc = np.zeros(shape=len(self.deltabeta), dtype=complex)

        for i in range(len(self.waveguide.z) - 1):
            dz = self.waveguide.z[i + 1] - self.waveguide.z[i]
            this_deltabeta = self.deltabeta + self.waveguide.profile[i]
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
        :param normalized: Optional argument. If True, normalizes the plotted phasematchig to have the maximum to 1. Default: False
        :type normalized: bool
        :param verbose: Optional. If True, writes the main information in the plot. Default: False
        :type verbose: bool
        :return:
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
            plt.text(x, y, text)

    def calculate_integral(self):
        """
        Function to calculate the integral of the phasematching curve. It uses the function `simps` from the scipy module.

        :return: Phasematching integral
        """
        return simps(abs(self.phi) ** 2, self.deltabeta)


class Phasematching1D(object):
    """
    Class to calculate the 1D-phasematching, i.e. having one fixed wavelength and scanning another one (the third is
    fixed due to energy conservation.
    The convention for labelling wavelength is abs(lambda_red) >= abs(lambda_green) >= abs(lambda_blue), i.e. according
    to their "energy".

    :param waveguide: waveguide object
    :type waveguide: :class:`pynumpm.waveguide.waveguide`
    :param phi: phasematching spectrum (complex valued function)
    :param n_red: refractive index for "red" wavelength
    :param n_green: refractive index for "green" wavelength
    :param n_blue: refractive index for "blue" wavelength
    :param order: order of phasematching
    :param constlam: wavelength to be kept fixed (can be "r","g","b" or "shg")
    :param process: type of process: "SFG", "SHG", "bwSFG", "bwSHG".
    :param propagation_type:  copropagating or counterpropagating
    :param red_wavelength: None, Single float or vector of float, containing the "red" wavelengths, in meters.
    :param green_wavelength: None, Single float or vector of float, containing the "green" wavelengths, in meters.
    :param blue_wavelength: None, Single float or vector of float, containing the "blue" wavelengths, in meters.

    """

    def __init__(self, waveguide, n_red, n_green, n_blue, order=1, backpropagation=False):
        """
        Variables:

        :param waveguide: Waveguide object. Use the Waveguide class in the Waveguide module to define this object.
        :param n_red: refractive index for the "red" wavelength. It has to be a lambda function of a lambda function, i.e. n(variable_parameter)(wavelength in um)
        :param n_green: refractive index for the "green" wavelength. It has to be a lambda function of a lambda function, i.e. n(variable_parameter)(wavelength in um)
        :param n_blue: refractive index for the "blue" wavelength. It has to be a lambda function of a lambda function, i.e. n(variable_parameter)(wavelength in um)
        :param process: type of process: "SFG", "SHG", "bwSFG", "bwSHG".
        :param order: order of phasematching. Default: 1

        """
        self.__waveguide = waveguide
        self.__phi = None
        self.__wavelength_set = False
        self.__n_red = n_red
        self.__n_green = n_green
        self.__n_blue = n_blue
        self.__constlam = None
        # ====================================================
        self.order = order
        # TODO: check if and how the poling order interferes when the poling structure is set
        self.process = None
        self.__red_wavelength = None
        self.__green_wavelength = None
        self.__blue_wavelength = None
        if backpropagation:
            self.propagation_type = "backpropagation"
        else:
            self.propagation_type = "copropagation"
        self._nonlinear_profile_set = False
        self.__noise_length_product = None
        self.scanning_wavelength = None
        self.__cumulative_delta_beta = None
        self.__cumulative_exponential = None
        self.__delta_beta0_profile = None
        self.__lamr0 = None
        self.__lamg0 = None
        self.__lamb0 = None

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
    def constlam(self):
        return self.__constlam

    @property
    def noise_length_product(self):
        return self.__noise_length_product

    def set_wavelengths(self):
        logger = logging.getLogger(__name__)
        num_of_none = (self.red_wavelength is None) + \
                      (self.green_wavelength is None) + \
                      (self.blue_wavelength is None)
        if num_of_none > 2:
            logger.info("Test", self.red_wavelength, self.green_wavelength, self.blue_wavelength, num_of_none)
            raise ValueError("It would be cool to know in which wavelength range I should calculate the phasematching!")
        elif num_of_none == 2:
            # calculate SHG
            self.process = "shg"
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
                logger.info("An error occurred in set_wavelengths. When setting the SHG wavelengths, "
                            "all the wavelengths are set but the number of none is 2."
                            "Red wavelength: {r}\nGreen wavelength: {g}\nBlue wavelength: {b}".format(
                    r=self.red_wavelength,
                    g=self.green_wavelength,
                    b=self.blue_wavelength))
                raise ValueError("Something unexpected happened in set_wavelength. "
                                 "Check the log please and chat with the developer.")
        elif num_of_none == 1:
            self.process = "sfg/dfg"
            if self.red_wavelength is None:
                self.red_wavelength = (self.blue_wavelength ** -1 - self.green_wavelength ** -1) ** -1
            elif self.green_wavelength is None:
                self.green_wavelength = (self.blue_wavelength ** -1 - self.red_wavelength ** -1) ** -1
            elif self.blue_wavelength is None:
                self.blue_wavelength = (self.red_wavelength ** -1 + self.green_wavelength ** -1) ** -1
            else:
                logger.info("An error occurred in set_wavelengths. When setting the SFG/DFG wavelengths, "
                            "all the wavelengths are set but the number of none is 1."
                            "Red wavelength: {r}\nGreen wavelength: {g}\nBlue wavelength: {b}".format(
                    r=self.red_wavelength,
                    g=self.green_wavelength,
                    b=self.blue_wavelength))
                raise ValueError("Something unexpected happened in set_wavelength. "
                                 "Check the log please and chat with the developer.")
        self.__wavelength_set = True
        self.__lamr0 = self.red_wavelength.mean() if type(self.red_wavelength) == np.ndarray else self.red_wavelength
        self.__lamg0 = self.green_wavelength.mean() if type(self.green_wavelength) == np.ndarray else self.green_wavelength
        self.__lamb0 = self.blue_wavelength.mean() if type(self.blue_wavelength) == np.ndarray else self.blue_wavelength
        return self.red_wavelength, self.green_wavelength, self.blue_wavelength

    def set_nonlinearity_profile(self, profile_type="constant", first_order_coeff=False, **kwargs):
        """
        Method to set the nonlinear profile. As a default it is set to "constant", i.e. birefringent or QPM.
        If **profile_type** is set to "constant", the you can select whether to use the first order Fourier coefficient (2/pi) using
        the keyword *first_order_coefficient* =**True**/False.

        If **profile_type** is set to *"gaussian"*, then you can select the width of the gaussian g(z) using the keyword
        *sigma_g_norm*, that reads the sigma in units of the length. Defauls to 0.5 (i.e. L/2).

        :param profile_type:
        :param kwargs:
        :return:
        """
        logger = logging.getLogger(__name__)
        logger.info("Setting the nonlinear profile.")
        logger.debug("Profile type: {pt}".format(pt=profile_type))
        if profile_type == "constant":
            logger.debug("First order coefficient: {foc}".format(foc=first_order_coeff))
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
        self.nonlinear_profile = g
        return self.nonlinear_profile

    def plot_nonlinearity_profile(self):
        x = self.waveguide.z
        y = self.nonlinear_profile(self.waveguide.z)
        plt.plot(x, y)

    def __calculate_local_neff(self, posidx):
        """
        Function to calculate the local effective refractive index
        :param posidx:
        :return:
        """
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
        """
        Computes the delta k.
        The wavelengths are provided in meter.
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
        This function is the core. Calculates the phasematching of the process, considering one wavelength fixed and
        scanning the other two.

        :param normalized: If True, the phasematching is limited in [0,1]. Otherwise, the maximum depends on the waveguide length, Default: True
        :type normalized: bool
        :return:
        """
        if not self.__wavelength_set:
            self.set_wavelengths()

        logger = logging.getLogger(__name__)
        logger.info("Calculating phasematching")

        if not self._nonlinear_profile_set:
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
        self.__delta_beta0_profile = np.zeros(shape=self.waveguide.z.shape)
        dz = self.waveguide.z[1] - self.waveguide.z[0]
        for idx, z in enumerate(self.waveguide.z):
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
                self.__cumulative_exponential += self.nonlinear_profile(z) * self.waveguide.poling_structure[idx] * \
                                                 (np.exp(-1j * dz * self.__cumulative_delta_beta) -
                                                  np.exp(-1j * dz * (self.__cumulative_delta_beta - DK)))
            else:
                self.__cumulative_exponential += self.nonlinear_profile(z) * np.exp(
                    -1j * dz * self.__cumulative_delta_beta)

        logger.info("Calculation terminated")
        self.__phi = self.__cumulative_exponential * self.waveguide.dz
        if normalized:
            self.__phi /= self.waveguide.length
        self.__noise_length_product = abs(self.__delta_beta0_profile).max() * self.waveguide.z[-1]
        return self.phi

    def calculate_integral(self):
        return simps(abs(self.phi) ** 2, self.scanning_wavelength)

    def plot_phasematching_error(self, ax=None):
        # self.calculate_phasematching_error()
        if ax is None:
            plt.figure()
            ax = plt.gca()
        else:
            ax = ax
        ax.plot(self.waveguide.z, self.dk_profile)
        return ax

    def plot(self, plot_intensity=True, xaxis="r", **kwargs):
        fig = kwargs.get("fig", None)
        ax = kwargs.get("ax", None)

        if ax is None:
            if fig is None:
                plt.figure()
            else:
                plt.figure(fig.number)
            ax = plt.gca()
        else:
            ax = plt.gca()
        if xaxis == self.constlam:
            raise ValueError("axis_wl has to be different than " + str(self.constlam))
        else:
            if xaxis == "r":
                wl = self.red_wavelength
            elif xaxis == "g":
                wl = self.green_wavelength
            elif xaxis == "b":
                wl = self.blue_wavelength
            else:
                raise ValueError("I don't know what {0} is.".format(xaxis))

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
        d = {"fig": fig,
             "ax": ax}
        return d


class Phasematching2D(object):
    """
    Phasematching2d class

    """

    def __init__(self, waveguide, n_red, n_green, n_blue, **kwargs):
        """
        Initialize the process to be studied.

        :param waveguide: An Object of the class Waveguide, properly initialized.
        :param n_red: function to compute the Sellmeier depending on the parameter that varies in the waveguide
            (i.e., n(w) if the width changes along the waveguide)
        :param n_green: function to compute the Sellmeier depending on the parameter that varies in the waveguide
            (i.e., n(w) if the width changes along the waveguide)
        :param n_blue: function to compute the Sellmeier depending on the parameter that varies in the waveguide
            (i.e., n(w) if the width changes along the waveguide)

        Additional parameters are:

        * :param process: [**PDC**/SFG]: process to be studied. If process is PDC, then you must specify the red
        and green wavelengths; if process is SFG, you must specify red and blue wavelengths.
        * :param order: [**1**]: order of the phasematching process
          :type order: int

    Other parameters related to this objects are:

        * :param __red_is_set: Parameter that is set to True if the red wl has been initialized
          :type __red_is_set: bool
        * :param __green_is_set: Parameter that is set to True if the green wl has been initialized
          :type __green_is_set: bool
        * :param __blue_is_set: Parameter that is set to True if the blue wl has been initialized
          :type __blue_is_set: bool

        """
        self.waveguide = waveguide
        self.__red_is_set = False
        self.__green_is_set = False
        self.__blue_is_set = False
        # these n_xxx functions are function that accept one parameter and provide a lambda function (n = n(wavelength))
        self.n_red = n_red
        self.n_green = n_green
        self.n_blue = n_blue
        # ====================================================
        self.order = kwargs.get("order", 1.)
        self.process = kwargs.get("process", "PDC").lower()
        self.red_wavelength = None
        self.green_wavelength = None
        self.blue_wavelength = None
        self.propagation_type = kwargs.get("propagation_type", "copropagation")
        if self.process[:2] == "bw":
            self.propagation_type = "backpropagation"
            self.process = self.process[2:]

    # TODO: redo the set_wavelength functions
    def set_red(self, **kwargs):
        """
        Function to set the red wavelength.

        :param central_wavelength: Central wavelength, in meters
        :param delta_lambda: Range to be scanned, in meters
        :param n_points: Number of points to sample
        :param kwargs:
        :return:
        """
        logger = logging.getLogger(__name__)
        self.n_points_red = kwargs.get("n_points", 100)
        if "start" in kwargs.keys() and "end" in kwargs.keys():
            initial_wl = kwargs.get("start")
            final_wl = kwargs.get("end")
            self.red_wavelength = np.linspace(initial_wl, final_wl, self.n_points_red)
            self.red_cwl = (initial_wl + final_wl) / 2.
            self.red_delta_wl = (final_wl - initial_wl) / 2.
        else:
            self.red_cwl = kwargs.get("central_wavelength")
            self.red_delta_wl = kwargs.get("delta_lambda")
            initial_wl = self.red_cwl - self.red_delta_wl / 2.
            final_wl = self.red_cwl + self.red_delta_wl / 2.
            self.red_wavelength = np.linspace(initial_wl, final_wl, self.n_points_red)

        self.__red_is_set = True
        if self.__red_is_set:
            logger.info("Red wavelength has been set: %f:%f:%f",
                        self.red_wavelength[0] * 1e9, self.red_wavelength[1] * 1e9 - self.red_wavelength[0] * 1e9,
                        self.red_wavelength[-1] * 1e9)

    def set_green(self, **kwargs):
        """
        Function to set the green wavelength.

        :param central_wavelength: Central wavelength, in meters
        :param delta_lambda: Range to be scanned, in meters
        :param n_points: Number of points to sample
        :param kwargs:
        :return:
        """
        logger = logging.getLogger(__name__)
        self.n_points_green = kwargs.get("n_points", 100)

        if "start" in kwargs.keys() and "end" in kwargs.keys():
            initial_wl = kwargs.get("start")
            final_wl = kwargs.get("end")
            self.green_wavelength = np.linspace(initial_wl, final_wl, self.n_points_green)
            self.green_cwl = (initial_wl + final_wl) / 2.
            self.green_delta_wl = (final_wl - initial_wl) / 2.
        else:
            self.green_cwl = kwargs.get("central_wavelength")
            self.green_delta_wl = kwargs.get("delta_lambda")
            initial_wl = self.green_cwl - self.green_delta_wl / 2.
            final_wl = self.green_cwl + self.green_delta_wl / 2.
            self.green_wavelength = np.linspace(initial_wl, final_wl, self.n_points_green)
        self.__green_is_set = True
        if self.__green_is_set:
            logger.info("Green wavelength has been set: %f:%f%:%f",
                        self.green_wavelength[0] * 1e9, self.green_wavelength[1] * 1e9 - self.green_wavelength[0] * 1e9,
                        self.green_wavelength[-1] * 1e9)

    def set_blue(self, **kwargs):
        """
        Function to set the blue wavelength.

        :param central_wavelength: Central wavelength, in meters
        :param delta_lambda: Range to be scanned, in meters
        :param n_points: Number of points to sample
        :param kwargs:
        :return:
        """
        logger = logging.getLogger(__name__)
        self.n_points_blue = kwargs.get("n_points", 100)

        if "start" in kwargs.keys() and "end" in kwargs.keys():
            initial_wl = kwargs.get("start")
            final_wl = kwargs.get("end")
            self.blue_wavelength = np.linspace(initial_wl, final_wl, self.n_points_blue)
            self.blue_cwl = (initial_wl + final_wl) / 2.
            self.blue_delta_wl = (final_wl - initial_wl) / 2.
        else:
            self.blue_cwl = kwargs.get("central_wavelength")
            self.blue_delta_wl = kwargs.get("delta_lambda")
            initial_wl = self.blue_cwl - self.blue_delta_wl / 2.
            final_wl = self.blue_cwl + self.blue_delta_wl / 2.
            self.blue_wavelength = np.linspace(initial_wl, final_wl, self.n_points_blue)

        self.__blue_is_set = True
        if self.__blue_is_set:
            logger.info("Blue wavelength has been set: %f:%f:%f",
                        self.blue_wavelength[0] * 1e9, self.blue_wavelength[1] * 1e9 - self.blue_wavelength[0] * 1e9,
                        self.blue_wavelength[-1] * 1e9)

    def calculate_local_neff(self, posidx):
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
        """
        Computes the delta k.
        The wavelengths are provided in meter.
        """
        # print(self.waveguide.poling_period)
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

    def calculate_phasematching(self, verbose=False):
        """
        This function is the core. Calculates the phasematching of the process, considering one wavelength fixed and scanning the other two.

        :return:
        """
        logger = logging.getLogger(__name__)
        logger.info("Calculating phasematching")
        if self.waveguide.poling_structure_set:
            logger.info("Poling period is not set. Calculating from structure.")
        else:
            logger.info("Poling period is set. Calculating with constant poling structure.")

        # edited at 28/09/2017 because the previous for loop was wrong!
        if self.process == "pdc":
            logger.info("Calculating for PDC")
            # calculate pdc phasematching. Assumes given the lambda_signal and lambda_idler
            # (i.e., lambda_red and lambda_green)
            if not (self.__red_is_set and self.__green_is_set):
                raise Exception(
                    "You must set the 'red' and 'green' (i.e., signal and idler, with wl_red <= wl_green) wavelengths before performing this calculation!")

            self._WL_RED, self._WL_GREEN = np.meshgrid(self.red_wavelength, self.green_wavelength)
            self._WL_BLUE = 1. / (1. / abs(self._WL_RED) + 1. / abs(self._WL_GREEN))
            # TODO: maybe format the following message?
            logger.info(self._WL_RED.mean(), self._WL_GREEN.mean(), self._WL_BLUE.mean())
            self.__cumulative_deltabeta = np.zeros(shape=(len(self.green_wavelength), len(self.red_wavelength)),
                                                   dtype=complex)

        elif self.process == "sfg":
            # calculate sfg phasematching
            if not (self.__red_is_set and self.__blue_is_set):
                raise Exception(
                    "You must set the 'red' and 'blue' (i.e., input and output, with wl_input <= wl_output) wavelengths before performing this calculation!")

            self._WL_RED, self._WL_BLUE = np.meshgrid(self.red_wavelength, self.blue_wavelength)
            self._WL_GREEN = 1. / (1. / abs(self._WL_BLUE) - 1. / abs(self._WL_RED))
            self.__cumulative_deltabeta = np.zeros(shape=(len(self.blue_wavelength), len(self.red_wavelength)),
                                                   dtype=complex)

        self.__cumulative_exponential = np.zeros(shape=self.__cumulative_deltabeta.shape, dtype=complex)
        dz = self.waveguide.z[1] - self.waveguide.z[0]
        dz = np.diff(self.waveguide.z).mean()  # TODO: this is RISKYYYYYYYY
        for idx, z in enumerate(self.waveguide.z):
            if verbose:
                if idx % 20 == 0:
                    logger.info("z = ", z * 1e3, " mm")
            # 1) retrieve the current parameter (width, thickness, ...)
            n_red, n_green, n_blue = self.calculate_local_neff(idx)
            # 2) evaluate the current phasemismatch
            DK = self.__calculate_delta_k(self._WL_RED, self._WL_GREEN, self._WL_BLUE, n_red, n_green, n_blue)
            # 4) add the phasemismatch to the past phasemismatches (first summation, over the delta k)
            self.__cumulative_deltabeta += DK
            # 5) evaluate the (cumulative) exponential (second summation, over the exponentials)
            if self.waveguide.poling_structure_set:
                self.__cumulative_exponential += self.waveguide.poling_structure[idx] * \
                                                 (np.exp(-1j * dz * self.__cumulative_deltabeta) -
                                                  np.exp(-1j * dz * (self.__cumulative_deltabeta - DK)))
            else:
                self.__cumulative_exponential += np.exp(-1j * dz * self.__cumulative_deltabeta)

        logger.info("Calculation terminated")
        self.phi = 1 / self.waveguide.length * self.__cumulative_exponential * dz
        return self.phi

    def extract_shg(self, **kwargs):
        """
        Function to extract the SHG plot from the phasematching.
        At the moment it works only if you are calculating the phasematching through "PDC"

        :param kwargs:
        :return:
        """
        if self.process == "pdc":
            if np.all(self.red_wavelength == self.green_wavelength):
                phi = abs(self.phi) ** 2
                self.shg = phi.diagonal()
                return self.red_wavelength, self.shg
            else:
                raise ValueError("signal and idler don't match")
        else:
            raise NotImplementedError("Wrong process, I can compute only from ")

    def slice_phasematching(self, **kwargs):
        """
        Slice the phasematching. The function interpolates the phasematching in the direction of the wavelength
        to be kept fixed (**fix_wl**) and evaluating the interpolation at the value provided (**value**). In this way, it is possible to cut
        along wavelength not present in the original grid.

        * :param fix_wl: [**red*/green/blue] Wavelength to be kept fixed while slicing
          :type fix_wl: str
        * :param value: [**red_cwl**] Value of the wavelength to be kept fixed
          :type value: float
        * :param show: If True, plots the slice
          :type show: bool

        """
        fix_wl = kwargs.get("fix_wl", "red").lower()
        scan_wl = kwargs.get("scan_wl", "green").lower()
        value_wl = kwargs.get("value", self.red_cwl)

        if self.process == "pdc":
            if fix_wl == "red":
                phasematching = interp.interp1d(self.red_wavelength, self.phi, axis=1)
                wl = copy.deepcopy(self.green_wavelength)
            elif fix_wl == "green":
                phasematching = interp.interp1d(self.green_wavelength, self.phi, axis=0)
                wl = copy.deepcopy(self.red_wavelength)
            else:
                # here, you need to provide the (CW) pump wl of the process (pdc)...
                blue_wl = value_wl
                # ... and also the wavelength you are scanning (default the lowest)
                scan_wl = kwargs.get("scan_wl", "red").lower()
                # 2D interpolation of the phasematching
                f_real = interp.interp2d(self.red_wavelength, self.green_wavelength, np.real(self.phi), kind='cubic')
                f_imag = interp.interp2d(self.red_wavelength, self.green_wavelength, np.imag(self.phi), kind='cubic')

                if scan_wl == "red":
                    # I am scanning with the lowest wavelength
                    wl = copy.deepcopy(self.red_wavelength)
                    phasematching = np.zeros(shape=wl.shape, dtype=complex)
                    for idx, curr_wl in enumerate(wl):
                        other_wl = (blue_wl ** -1 - curr_wl ** -1) ** -1
                        phasematching[idx] = f_real(curr_wl, other_wl)[0] + 1j * f_imag(curr_wl, other_wl)[0]
                elif scan_wl == "green":
                    wl = copy.deepcopy(self.green_wavelength)
                    phasematching = np.zeros(shape=wl.shape, dtype=complex)
                    for idx, curr_wl in enumerate(wl):
                        other_wl = (blue_wl ** -1 - curr_wl ** -1) ** -1
                        phasematching[idx] = f_real(other_wl, curr_wl)[0] + 1j * f_imag(other_wl, curr_wl)[0]
                else:
                    raise ValueError("Provide the scanning wavelength (red/green)")
                self.phasematching_cut = np.array(phasematching)
                return wl, self.phasematching_cut

        elif self.process == "sfg":
            # print("process = sfg")
            if fix_wl == "red":
                # The user said that he has fixed the red wavelength. Therefore, calculate the interpolation of
                # the phasematching as a function of the red wavelength. Assume as independent variable the blue.

                # phasematching is the interpolation of the matrix in order to get the phi = f(blue/green) parametrically
                # wrt red, i.e. new_phi = phasematching(my_red_wavelength) --> new_phi is an array whose "axis" is the
                # blue (or the green) wavelengths
                phasematching = interp.interp1d(self.red_wavelength, self.phi, axis=1)
                if scan_wl == "blue":
                    wl = copy.deepcopy(self.blue_wavelength)
                elif scan_wl == "green":
                    # if the wavelength to be scanned is the pump, then its value depends on the red_wl (which is decided by the user) and by the blue
                    wl = (self.blue_wavelength ** -1 - value_wl ** -1) ** -1
            elif fix_wl == "blue":
                # phasematching is the interpolation of the matrix in order to get the phi = f(red/green) parametrically
                # wrt blue, i.e. new_phi = phasematching(my_blue_wavelength) --> new_phi is an array whose "axis" is the
                # red (or the green) wavelengths
                phasematching = interp.interp1d(self.blue_wavelength, self.phi, axis=0)
                if scan_wl == "red":
                    wl = copy.deepcopy(self.red_wavelength)
                elif scan_wl == "green":
                    wl = (self.blue_wl ** -1 - self.red_wavelength - 1) ** -1
            elif fix_wl == "green":
                # I want to keep fixed the pump!
                print("You asked me to do a difficult thingy!!")
                # here, you need to provide the (CW) pump wl of the process (sfg)...
                green_wl = value_wl
                # ... and also the wavelength you are scanning (default the lowest)
                scan_wl = kwargs.get("scan_wl", "red").lower()

                # 2D interpolation of the phasematching. I need to separate the real and imaginary part
                # for the interpolation
                f_real = interp.interp2d(self.red_wavelength, self.blue_wavelength, np.real(self.phi), kind='cubic')
                f_imag = interp.interp2d(self.red_wavelength, self.blue_wavelength, np.imag(self.phi), kind='cubic')
                if scan_wl == "red":
                    # I am scanning with the lowest wavelength
                    wl = copy.deepcopy(self.red_wavelength)
                    phasematching = np.zeros(shape=self.red_wavelength.shape, dtype=complex)
                    for idx, _red_wl in enumerate(self.red_wavelength):
                        _blue_wl = (_red_wl ** -1 + green_wl ** -1) ** -1
                        phasematching[idx] = f_real(_red_wl, _blue_wl)[0] + 1j * f_imag(_red_wl, _blue_wl)[0]
                elif scan_wl == "blue":
                    wl = copy.deepcopy(self.blue_wavelength)
                    phasematching = np.zeros(shape=self.blue_wavelength.shape, dtype=complex)
                    for idx, _blue_wl in enumerate(self.blue_wavelength):
                        _red_wl = (_blue_wl ** -1 - green_wl ** -1) ** -1
                        phasematching[idx] = f_real(_red_wl, _blue_wl)[0] + 1j * f_imag(_red_wl, _blue_wl)[0]
                else:
                    raise ValueError("Provide the scanning wavelength (red/blue)")
                # self.phasematching_cut = interp.interp1d(wl, np.array(phasematching), kind='cubic')
                # phasematching = interp.interp1d(wl, np.array(phasematching), kind='cubic')
                # print type(self.phasematching_cut(wl))

                # here, phasematching is an ARRAY of the phasematching amplitudes as a function of the red/blue wavelength.
                self.phasematching_cut = np.array(phasematching)
                return wl, self.phasematching_cut
            else:
                raise ValueError("There is no {0} wavelength.".format(fix_wl))
        else:
            raise ValueError("The process is wrong, I cannot calculate the slice!")
        self.phasematching_cut = phasematching(value_wl)

        if kwargs.get("plot", False):
            plt.figure()
            plt.plot(wl * 1e9, abs(self.phasematching_cut) ** 2)
            plt.show()

        return wl, self.phasematching_cut

    def plot_phasematching(self, **kwargs):
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
        active_axes = self.__check_active_axes()
        phi = abs(self.phi)
        if plot_intensity:
            phi = phi ** 2

        cmap = kwargs.get("cmap", "Greens")
        vmin = kwargs.get("vmin", phi.min())
        vmax = kwargs.get("vmax", phi.max())
        # try:
        # TODO: check whether to flipud(phi)
        im = ax.pcolormesh(active_axes[0] * 1e9, active_axes[1] * 1e9, phi, cmap=cmap, vmin=vmin, vmax=vmax)
        # except:
        #     print "Cannot set the colormap"
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

    def __check_active_axes(self):
        """
        Function to check which are the useful axes for the plot (depending on which wavelengths have been initialized)
        :return:
        """
        list_wl = [[self.red_wavelength, self.__red_is_set],
                   [self.green_wavelength, self.__green_is_set],
                   [self.blue_wavelength, self.__blue_is_set]]

        active_axes = [i[0] for i in list_wl if i[1]]
        return active_axes

    def calculate_JSA(self, thispump):
        """
        Function to calculate the JSA.
        Requires as an input a pump Object (from the class Pump written by Benni)

        :param pump:
        :return:
        """
        logger = logging.getLogger(__name__)
        logger.info("Calculating JSA")
        if self.process == "pdc":
            d_wl1 = abs(self.red_wavelength[1] - self.red_wavelength[0])
            d_wl2 = abs(self.green_wavelength[1] - self.green_wavelength[0])
            wl1 = self.red_wavelength
            wl2 = self.green_wavelength
        elif self.process == "sfg":
            d_wl1 = abs(self.red_wavelength[1] - self.red_wavelength[0])
            d_wl2 = abs(self.blue_wavelength[1] - self.blue_wavelength[0])
            wl1 = self.red_wavelength
            wl2 = self.blue_wavelength

        WL1, WL2 = np.meshgrid(wl1, wl2)
        if thispump.signal_wavelength is None:
            thispump.signal_wavelength = WL1
        if thispump.idler_wavelength is None:
            thispump.idler_wavelength = WL2
        self.pump = thispump.pump()
        self.JSA = self.pump * self.phi
        self.JSA /= np.sqrt((abs(self.JSA) ** 2).sum() * d_wl1 * d_wl2)
        self.JSI = abs(self.JSA) ** 2
        return self.JSA, self.JSI

    def calculate_marginals(self):
        if self.process == "pdc":
            self.marginal_red = self.JSI.sum(axis=0) * abs(self.green_wavelength[1] - self.green_wavelength[0])
            self.marginal_green = self.JSI.sum(axis=1) * abs(self.red_wavelength[1] - self.red_wavelength[0])
            return (self.red_wavelength, self.marginal_red), (self.green_wavelength, self.marginal_green)
        elif self.process == "sfg":
            self.marginal_red = self.JSI.sum(axis=0) * abs(self.blue_wavelength[1] - self.blue_wavelength[0])
            self.marginal_blue = self.JSI.sum(axis=1) * abs(self.red_wavelength[1] - self.red_wavelength[0])
            return (self.red_wavelength, self.marginal_red), (self.blue_wavelength, self.blue_green)

    def plot_JSI(self, **kwargs):
        """
        Function to plot JSI. Pass ax handle through "ax" to plot in a specified axis environment.

        :param kwargs:
        :return:
        """
        ax = kwargs.get("ax", None)
        title = kwargs.get("title", "JSI")

        if ax is None:
            plt.figure()
            ax = plt.gca()
        if self.process == "pdc":
            x = self.red_wavelength * 1e9
            y = self.green_wavelength * 1e9
        elif self.process == "sfg":
            x = self.red_wavelength * 1e9
            y = self.blue_wavelength * 1e9

        im = ax.pcolormesh(x, y, self.JSI)

        if kwargs.get("plot_pump", False):
            print("Plot Pump")
            X, Y = np.meshgrid(x, y)
            Z = abs(self.pump) ** 2
            CS = ax.contour(X, Y, Z / Z.max(), 4, colors="w", ls=":", lw=0.5)
            ax.clabel(CS, fontsize=9, inline=1)

        plt.gcf().colorbar(im)
        ax.set_xlabel("Signal [nm]")
        ax.set_ylabel("Idler [nm]")
        ax.set_title(title)

    def calculate_schmidt_number(self, verbose=False):
        """
        Function to calculate the Schidt decomposition.

        :return:
        """
        logger = logging.getLogger(__name__)
        U, self.singular_values, V = np.linalg.svd(self.JSA)
        self.singular_values /= np.sqrt((self.singular_values ** 2).sum())
        self.K = 1 / (self.singular_values ** 4).sum()

        logger.debug("Check normalization: sum of s^2 = ", (abs(self.singular_values) ** 2).sum())
        logger.info("K = ", self.K)
        return self.K

    def extract_max_phasematching_curve(self, **kwargs):
        """
        Extract the curve of max phasematching. Useful to estimate GVM.

        :return:
        """
        signal = self.red_wavelength
        idler = []

        for idx, wl in enumerate(signal):
            idl, pm_cut = self.slice_phasematching(fix_wl="red", value=wl)
            idler.append(idl[pm_cut.argmax()])
        IDLER = np.array(idler)
        p_idler = np.polyfit(signal, IDLER, deg=2)
        idler = np.polyval(p_idler, signal)
        if kwargs.get("plot", False):
            self.plot_phasematching()
            plt.plot(signal * 1e9, idler * 1e9, "k", lw=3)
            plt.show()

        return signal, idler

    def print_properties(self):
        pprint.pprint(vars(self))

    def find_gvm(self):
        signal, idler = self.extract_max_phasematching_curve(plot=False)
        grad = np.diff(idler * 1e9) / np.diff(signal * 1e9)
        IDL = idler[abs(grad).argmin()]
        SIG = signal[abs(grad).argmin()]
        if self.process == "pdc":
            pump = 1. / (1. / IDL + 1. / SIG)
            return SIG, IDL, pump
        elif self.process == "sfg":
            pump = 1. / (1. / IDL - 1. / SIG)
            return SIG, pump, IDL
