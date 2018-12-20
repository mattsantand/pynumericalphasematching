# -*- coding: utf-8 -*-
"""
Created on 26.09.2017 08:30

.. module: 
.. moduleauthor: Matteo Santandrea <matteo.santandrea@upb.de>
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pynumpm import utilities


class NoiseProfile(object):
    def __init__(self, z=None, noise_amplitude=0., offset=0.):
        if z is None:
            raise ValueError("z cannot be None")
        if noise_amplitude < 0:
            warnings.warn("noise_amplitude is negative. Taking the absolute value", UserWarning)
        self.__noise_amplitude = abs(noise_amplitude)

        self.__z = z
        self.__offset = offset
        self.__dz = np.diff(self.z)[0]
        self.__profile = self.__noise_amplitude * np.ones(shape=self.z.shape) + self.__offset
        self.__autocorrelation_is_calculated = False
        self.f = None

    @property
    def z(self):
        return self.__z

    @property
    def dz(self):
        return self.__dz

    @property
    def profile(self):
        return self.__profile

    @property
    def noise_amplitude(self):
        return self.__noise_amplitude

    @property
    def offset(self):
        return self.__offset

    def concatenate(self, other):
        if self.dz != other.dz:
            raise ValueError("The resolution of the two noise spectra are different. Set them to equal.")
        new_z = np.append(self.z, other.z + self.z[-1])
        newprofile = np.append(self.profile, other.profile)
        newnoise = NoiseProfile(z=new_z)
        newnoise.__profile = newprofile
        return newnoise

    def plot_noise_properties(self, fig=None, **kwargs):
        z_autocorr, autocorr, f, power_spectrum = utilities.calculate_profile_properties(self.z, self.profile)
        if fig is None:
            fig = plt.figure()
        else:
            fig = plt.figure(fig.number)

        plt.subplot(211)
        l1, = plt.plot(self.z, self.profile, **kwargs)
        plt.title("Noise")
        plt.xlabel("z")
        plt.ylabel("Noise")

        plt.subplot(234)
        l2, = plt.semilogy(z_autocorr, abs(autocorr) ** 2, label="Calculated autocorrelation",
                           **kwargs)
        plt.title("|R(z)|^2")
        plt.xlabel("z")

        plt.subplot(235)
        l3, = plt.loglog(f, abs(power_spectrum) ** 2, **kwargs)
        plt.title("|S(f)|^2")
        plt.xlabel("f")
        plt.subplot(236)
        plt.hist(self.profile, bins=int(np.sqrt(len(self.profile))))
        plt.title("Histogram of the noise.")
        plt.tight_layout()
        return fig, [l1, l2, l3]


class NoiseFromSpectrum(NoiseProfile):
    def __init__(self, z=None, offset=0, noise_amplitude=0., profile_spectrum=None):
        NoiseProfile.__init__(self, z, offset=offset, noise_amplitude=noise_amplitude)
        if profile_spectrum is None:
            raise ValueError("profile_spectrum must be set")
        if profile_spectrum.lower() in ["awgn", "1/f", "1/f2", "pink"]:
            if profile_spectrum.lower() == "pink":
                profile_spectrum = "1/f"
            self.__profile_spectrum = profile_spectrum.lower()
        else:
            raise ValueError("profile_spectrum has to be 'awgn', '1/f' or '1/f2'")

        # print "I am going to calculate the profile"
        self.__profile = self.generate_noise()
        # print "Profile calculated"

    @property
    def profile_spectrum(self):
        return self.__profile_spectrum

    @property
    def profile(self):
        return self.__profile

    def generate_noise(self):
        logger = logging.getLogger(__name__)
        logger.info("Generating {s} spectrum.".format(s=self.profile_spectrum))
        length = self.z[-1] - self.z[0]
        npoints = len(self.z)
        if npoints % 2 == 0:
            npoints += 1
            extended = True
        else:
            extended = False
        df = 1. / length
        frequency_coefficients = np.zeros(shape=npoints, dtype=complex)
        half_size = int((npoints - 1) / 2)

        if self.profile_spectrum == "awgn":
            exponent = 0
        elif self.profile_spectrum == "1/f":
            exponent = 1
        elif self.profile_spectrum == "1/f2":
            exponent = 2
        else:
            raise ValueError("Unknown self.profile_spectrum value. Cannot set the exponent.")

        for idx in range(1, half_size + 1):
            fk = idx * df / (2 * np.pi)
            ck = 1. / fk ** exponent
            phase = np.random.uniform(0, 2 * np.pi)
            frequency_coefficients[half_size + idx] = ck * np.exp(1j * phase)
            frequency_coefficients[half_size - idx] = ck * np.exp(-1j * phase)
        y = np.fft.ifft(np.fft.ifftshift(frequency_coefficients))
        y *= self.noise_amplitude / abs(y).max()
        if extended:
            y = y[:-1]
        return np.real(y) + self.offset


class CorrelatedNoise(NoiseProfile):
    def __init__(self, z=None, offset=0, noise_amplitude=0., correlation_length=0.):
        NoiseProfile.__init__(self, z=z, offset=offset, noise_amplitude=noise_amplitude)
        if correlation_length < 0:
            warnings.warn("correlation_length is negative. I will get the absolute value", UserWarning)
        self.__correlation_length = abs(correlation_length)
        NoiseProfile.__profile = self.generate_noise()

    @property
    def correlation_length(self):
        return self.__correlation_length

    @property
    def profile(self):
        return self.__profile

    def generate_noise(self):
        logger = logging.getLogger(__name__)
        sigma = self.noise_amplitude
        if self.correlation_length == 0:
            r = 0
        else:
            r = np.exp(-self.dz / self.correlation_length)
        logger.info("Correlation factor: %f", r)
        zz = self.z - self.z[-1] / 2.
        self.ideal_correlation_function = sigma ** 2 * np.exp(- abs(zz) / self.correlation_length)
        y = np.zeros(self.z.shape)
        # generate the first point drawing it from a gaussian distribution centered on the mean and with sigma std.
        # y[0] = sigma * np.random.randn()
        y[0] = 0
        for i in range(1, len(self.z)):
            y[i] = r * y[i - 1] + sigma * np.sqrt(1 - r ** 2) * np.random.randn()
        # y += self.offset
        return y + self.offset
