# coding=utf-8
"""
Module to simulate the 2D spectrum of a pump field for the simulation of joint spectral amplitudes of nonlinear processes.

"""
import logging
from scipy.special import hermite, factorial
from scipy.constants import c as _sol
import matplotlib.pyplot as plt
import numpy as np
import enum
import warnings


class Process(enum.Enum):
    SFG = "SFG. Signal is the input, Idler is the output."
    DFG = "DFG. Signal is the input, Idler is the output."
    PDC = "PDC."
    BWPDC = "BWPDC. Signal is the for the backward propagating field."  # TODO: Check this


class Pump(object):
    """ Provides the pump function of a nonlinear process.

    === Public methods ===
    pump -- generates the pump function from the provided
            parameters

    === Private methods ===
    _hermite_mode -- normalised Hermite-Gaussian function

    === Public Variables ===
    pump_center -- pump central wavelength [m] (default: None)
    pump_wavelength -- matrix containing the pump wavelengths
                       in the signal and idler frequency plane [m]
                       (default: None)
    pump_width -- the intensity FWHM of the pump pulses [m] (default: None)
    signal_wavelength -- matrix containing the signal wavelengths [m]
                         (default: None)
    idler_wavelength -- matrix containing the idler wavelengths [m]
                        (default: None)
    type -- keyword defining the pump type; must be in ['normal',
            'filtered', custom] (default: 'normal')
    process -- nonlinear process (default: 'PDC')
    pump_delay -- temporal pump delay with respect to a reference [s]
                  (default: 0)
    pump_chirp -- quadratic chirp parameter of the pump pulse [s**2]
                  (default: 0)
    pump_temporal_mode -- temporal mode order of the pump pulse
                          (default: 0)
    pump_filter_width -- intensity FWHM of a spectral filter [m]
                         (default: 100)
    sol -- speed of light [m] (default: 299792458)

    === Private Variables ===
    _result -- about every calculation result
    _pump_function -- matrix containing the pump function
    _filter -- matrix containing the filter function

    """

    def __init__(self, process: Process):
        """ Initialise a pump with default parameters. """

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
        self.__signal_wavelength = None
        self.__idler_wavelength = None
        self.__signal_wavelength2D = None
        self.__idler_wavelength2D = None
        self.__pump_spectrum = None
        self.__correct_pump_width = None

    @property
    def pump_spectrum(self):
        return self.__pump_spectrum

    @property
    def signal_wavelength(self):
        return self.__signal_wavelength

    @signal_wavelength.setter
    def signal_wavelength(self, value):
        self.__signal_wavelength = value

    @property
    def idler_wavelength(self):
        return self.__idler_wavelength

    @idler_wavelength.setter
    def idler_wavelength(self, value):
        self.__idler_wavelength = value

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
    def signal_wavelength2D(self):
        return self.__signal_wavelength2D

    @property
    def idler_wavelength2D(self):
        return self.__idler_wavelength2D

    def _hermite_mode(self, x):
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
        if self.signal_wavelength is None:
            error = True
            text += "You need to set the signal wavelength. "
        if self.idler_wavelength is None:
            error = True
            text += "You need to set the idler wavelength."
        if error:
            raise ValueError(text.rstrip())

        self.__signal_wavelength2D, self.__idler_wavelength2D = np.meshgrid(self.signal_wavelength,
                                                                            self.idler_wavelength)
        message = "The pump central wavelength hasn't been set. Inferring its value from the signal/idler arrays"
        if self.process == Process.PDC or self.process == Process.BWPDC:
            self.pump_wavelength = 1.0 / (1.0 / self.__signal_wavelength2D +
                                          1.0 / self.__idler_wavelength2D)
            if self.pump_centre is None:
                warnings.warn(message, UserWarning)
                self.pump_centre = (self.signal_wavelength.mean() ** -1 + self.idler_wavelength.mean() ** -1) ** -1

        elif self.process == Process.SFG:
            self.pump_wavelength = 1.0 / (1.0 / self.__idler_wavelength2D -
                                          1.0 / self.__signal_wavelength2D)
            if self.pump_centre is None:
                warnings.warn(message, UserWarning)
                self.pump_centre = (self.idler_wavelength.mean() ** -1 - self.signal_wavelength.mean() ** -1) ** -1

        elif self.process == Process.DFG:
            self.pump_wavelength = 1.0 / (1.0 / self.__signal_wavelength2D -
                                          1.0 / self.__idler_wavelength2D)
            if self.pump_centre is None:
                warnings.warn(message, UserWarning)
                self.pump_centre = (self.signal_wavelength.mean() ** -1 - self.idler_wavelength.mean() ** -1) ** -1
        else:
            raise NotImplementedError("The process {0} has not been implemented yet.".format(self.process))

    def calculate_pump_spectrum(self):
        """ Calculates the pump function

        === Returns ===
        _pump_function -- matrix containing the pump function in
                          signal and idler frequecy plane
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
            for i in range(len(self.signal_wavelength)):
                logger.debug("Loop index: %d", i)
                for j in range(len(self.idler_wavelength)):
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
            x, y = self.signal_wavelength * 1e9, self.idler_wavelength * 1e9,
            ax.imshow(abs(self.pump_spectrum) ** 2, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
                      aspect="auto")
            warnings.warn("The light_plot mode is compatible only with linear meshes of the signal/idler wavelengths.")
        else:
            ax.pcolormesh(self.signal_wavelength * 1e9, self.idler_wavelength * 1e9, abs(self.pump_spectrum) ** 2)

        ax.set_title("Pump intensity")
        if self.process == Process.SFG or self.process == Process.DFG:
            ax.set_xlabel(r"$\lambda_{input}$ [nm]")
            ax.set_ylabel(r"$\lambda_{output}$ [nm]")
        else:
            ax.set_xlabel(r"$\lambda_{signal}$ [nm]")
            ax.set_ylabel(r"$\lambda_{idler}$ [nm]")
        plt.tight_layout()


class JSA(object):
    def __init__(self, phasematching, pump: Pump):
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
    def JSA(self):
        return self.__JSA

    @property
    def JSI(self):
        return self.__JSI

    @property
    def K(self):
        return self.__K

    @property
    def marginal1(self):
        return self.__marginal1

    @property
    def marginal2(self):
        return self.__marginal2

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
        signal_wl = self.phasematching.signal_wavelength
        idler_wl = self.phasematching.idler_wavelength

        # d_wl_signal = np.diff(signal_wl)[0]
        # d_wl_idler = np.diff(self.phasematching.idler_wavelength)[0]

        WL_SIGNAL, WL_IDLER = np.meshgrid(signal_wl, idler_wl)
        self.pump.signal_wavelength = WL_SIGNAL
        self.pump.idler_wavelength = WL_IDLER

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

    def calculate_schmidt_number(self, verbose=False):
        """
        Function to calculate the Schidt decomposition.

        :param bool verbose: Print to screen the Schmidt number and the purity of the state.

        :return: the Schmidt number.
        """
        logger = logging.getLogger(__name__)
        if self.JSA is None:
            self.calculate_JSA()
        U, self.__singular_values, V = np.linalg.svd(self.__JSA)
        self.__singular_values /= np.sqrt((self.__singular_values ** 2).sum())
        self.__K = 1 / (self.__singular_values ** 4).sum()
        text = "Schmidt number K: {K}\nPurity: {P}".format(K=self.K, P=1 / self.K)
        if verbose:
            print(text)
        logger.info(text)
        logger.debug("Check normalization: sum of s^2 = " + str((abs(self.__singular_values) ** 2).sum()))
        return self.__K

    def plot(self, ax=None, light_plot=False, **kwargs):
        """
        Function to plot JSI. Pass ax handle through "ax" to plot in a specified axis environment.

        :param kwargs:
        :return:
        """
        if self.JSA is None:
            raise ValueError("You need to calculate the JSA first, use the command calculate_jsa()")
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        title = kwargs.get("title", "JSI")
        x = self.phasematching.signal_wavelength * 1e9
        y = self.phasematching.idler_wavelength * 1e9

        if light_plot:
            im = ax.imshow(self.JSI, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
                           aspect="auto")
            warnings.warn("The light_plot mode is compatible only with linear meshes of the signal/idler wavelengths.")
        else:
            im = ax.pcolormesh(x, y, self.JSI)

        if kwargs.get("plot_pump", False):
            print("Plot Pump")
            X, Y = np.meshgrid(x, y)
            Z = abs(self.pump.pump_spectrum) ** 2
            CS = ax.contour(X, Y, Z / Z.max(), 4, colors="w", ls=":", lw=0.5)
            ax.clabel(CS, fontsize=9, inline=1)

        plt.gcf().colorbar(im)
        ax.set_xlabel(r"$\lambda_{signal}$ [nm]")
        ax.set_ylabel(r"$\lambda_{idler}$ [nm]")
        ax.set_title(title)
        plt.tight_layout()

    def plot_marginals(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 2)
        else:
            if type(ax) != list or (type(ax) == list and len(ax) != 2):
                raise ValueError(
                    "I need two different axes to plot the marginals. ax should be a list of axes handles [ax0, ax1]")

        suptitle = kwargs.get("suptitle", "Marginals")
        print(self.phasematching.signal_wavelength * 1e9)
        print(self.__marginal1)
        ax[0].plot(self.phasematching.signal_wavelength * 1e9, self.marginal1)
        ax[1].plot(self.phasematching.idler_wavelength * 1e9, self.marginal2)
        plt.suptitle(suptitle)


def main():
    import numpy
    import matplotlib.pyplot as plt
    pump = Pump(process=Process.PDC)
    signal_wl = numpy.linspace(790E-9, 810E-9, 150)
    idler_wl = numpy.linspace(790E-9, 810E-9, 250)
    pump_centre = 400E-9
    pump_width = 1E-9
    pump.signal_wavelength = signal_wl
    pump.idler_wavelength = idler_wl
    pump.pump_center = pump_centre
    pump.pump_width = pump_width
    pump.pump_filter_width = 1.0E-9
    pump.__filter_pump = 'filtered'
    result = pump.pump_spectrum
    result /= abs(result).max()
    plt.figure(figsize=(4, 4))
    plt.contourf(pump.signal_wavelength2D, pump.idler_wavelength2D, abs(result), 10)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
