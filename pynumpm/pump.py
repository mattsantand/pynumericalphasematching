# coding=utf-8
"""
Module to simulate the 2D spectrum of a pump field for the simulation of joint spectral amplitudes of nonlinear processes.

"""
import logging
from numpy import exp, pi, sqrt, shape, zeros, log
from scipy.special import hermite, factorial


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

    def __init__(self, pump_center=None, pump_wavelength=None,
                 pump_width=None, signal_wavelength=None,
                 idler_wavelength=None, pump_type="normal",
                 process="PDC", pump_delay=0, pump_chirp=0,
                 pump_temporal_mode=0, pump_filter_width=100):
        """ Initialise a pump with default parameters. """
        self.pump_center = pump_center
        self.pump_wavelength = pump_wavelength
        self.pump_width = pump_width
        self.signal_wavelength = signal_wavelength
        self.idler_wavelength = idler_wavelength
        self.type = pump_type
        self.process = process
        self.pump_delay = pump_delay
        self.pump_chirp = pump_chirp
        self.pump_temporal_mode = pump_temporal_mode
        self.pump_filter_width = pump_filter_width
        self.sol = 299792458.0

    def _hermite_mode(self, x):
        """ A normalised Hermite-Gaussian function """
        # On 22.11.2017, Matteo changed all the self.pump_width to self.correct_pump_width
        # _result = hermite(self.pump_temporal_mode)((self.pump_center - x) /
        #                                            self.pump_width) *    \
        #     exp(-(self.pump_center - x)**2 / (2 * self.pump_width**2)) /\
        #     sqrt(factorial(self.pump_temporal_mode) * sqrt(pi) *
        #          2**self.pump_temporal_mode * self.pump_width)
        _result = hermite(self.pump_temporal_mode)((self.pump_center - x) /
                                                   self.correct_pump_width) * \
                  exp(-(self.pump_center - x) ** 2 / (2 * self.correct_pump_width ** 2)) / \
                  sqrt(factorial(self.pump_temporal_mode) * sqrt(pi) *
                       2 ** self.pump_temporal_mode * self.correct_pump_width)
        return _result

    def pump(self):
        """ Calculates the pump function

        === Returns ===
        _pump_function -- matrix containing the pump function in
                          signal and idler frequecy plane
        """
        logger = logging.getLogger(__name__)
        # self.pump_width /= 2 * sqrt(log(2))
        # self.pump_width = self.pump_width /( 2 * sqrt(log(2)))
        self.correct_pump_width = self.pump_width / (2 * sqrt(log(2)))
        if self.process.upper() in ['PDC', 'BWPDC']:
            self.pump_wavelength = 1.0 / (1.0 / self.signal_wavelength +
                                          1.0 / self.idler_wavelength)
        elif self.process.upper() == 'SFG':
            self.pump_wavelength = 1.0 / (1.0 / self.idler_wavelength -
                                          1.0 / self.signal_wavelength)
        elif self.process.upper() == 'DFG':
            self.pump_wavelength = 1.0 / (1.0 / self.signal_wavelength -
                                          1.0 / self.idler_wavelength)
        if self.type.upper() == 'NORMAL':
            _pump_function = self._hermite_mode(self.pump_wavelength) * \
                             exp(1j * 2 * pi * self.sol / self.pump_wavelength *
                                 self.pump_delay) * \
                             exp(1j * (2 * pi * self.sol / self.pump_wavelength) ** 2 *
                                 self.pump_chirp)
        elif self.type.upper() == 'FILTERED':
            self.pump_filter_width *= sqrt(2)
            _filter = zeros(shape(self.pump_wavelength), float)
            logger.debug("Pump wavelength: %f", shape(self.pump_wavelength))
            for i in range(len(self.signal_wavelength)):
                logger.debug("Loop index: %d", i)
                for j in range(len(self.idler_wavelength)):
                    if self.pump_wavelength[j, i] < self.pump_center - \
                            0.5 * self.pump_filter_width:
                        pass
                    elif self.pump_wavelength[j, i] <= self.pump_center + \
                            0.5 * self.pump_filter_width:
                        _filter[j, i] = 1
                    else:
                        pass
            _pump_function = self._hermite_mode(self.pump_wavelength) * \
                             exp(1j * 2 * pi * self.sol / self.pump_wavelength *
                                 self.pump_delay) * \
                             exp(1j * (2 * pi * self.sol / self.pump_wavelength) ** 2 *
                                 self.pump_chirp) * _filter
        elif self.type.upper() == 'CUSTOM':
            _pump_function = None
            # from custom_pump import custom_pump
            # _pump_function = custom_pump(self.pump_wavelength,
            #                              self.pump_center,
            #                              self.pump_width)
        else:
            _pump_function = None
        return _pump_function


def main():
    import numpy
    import matplotlib.pyplot as plt
    pump = Pump()
    signal_wl = numpy.linspace(790E-9, 810E-9, 150)
    idler_wl = numpy.linspace(790E-9, 810E-9, 250)
    SIG, ID = numpy.meshgrid(signal_wl, idler_wl)
    pump_center = 400E-9
    pump_width = 1E-9
    pump.signal_wavelength = SIG
    pump.idler_wavelength = ID
    pump.pump_center = pump_center
    pump.pump_width = pump_width
    pump.pump_filter_width = 1.0E-9
    pump.type = 'filtered'
    result = pump.pump()
    result /= abs(result).max()
    plt.figure(figsize=(4, 4))
    plt.contourf(SIG, ID, abs(result), 10)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
