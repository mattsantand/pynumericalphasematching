# -*- coding: utf-8 -*-
"""
Created on 26.09.2017 08:30

.. module: 
.. moduleauthor: Matteo Santandrea <matteo.santandrea@upb.de>
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erfc
import matplotlib.pyplot as plt
import RSoftFEM

_LONGRANGENOISE = ["quadratic", "quartic", "gaussian", "awgn", "1/f", "1/f2"]


class Noise(object):
    def __init__(self, **kwargs):
        self.z = kwargs.get("z", np.linspace(0, 20, 1000))
        self.dz = self.z[1] - self.z[0]
        self.offset = kwargs.get("offset", 0.)
        self.long_range_noise = kwargs.get("long_range_noise", "gaussian")
        if not (self.long_range_noise.lower() in _LONGRANGENOISE):
            raise ValueError("long_range_noise variable has to be in {0}".format(_LONGRANGENOISE))
        self.long_range_parameter = kwargs.get("long_range_parameter")
        if self.long_range_noise.lower() in _LONGRANGENOISE[:3]:
            self.short_range_noise = kwargs.get("short_range_noise", "gaussian")
            if not self.short_range_noise.lower() == "gaussian":
                raise NotImplementedError("This short noise has not yet been implemented")
            self.short_range_parameter = kwargs.get("short_range_parameter")
            self.n_points_long_range = kwargs.get("number_points_long_range", 10)
            self.n_points_short_range = kwargs.get("number_points_short_range", 300)
            self.noise = self.generate_long_range_noise() + self.generate_short_range_noise() + self.offset
        else:
            self.noise = self.generate_noise() + self.offset
        self.calculate_autocorrelation()

    def __add__(self, other):
        if np.any((self.z - other.z) != 0):
            raise AttributeError("The z vectors are different!")
        noise = self.noise + other.noise
        newObject = Noise(z=self.z, offset=0, long_range_noise="AWGN", long_range_parameter=0)
        newObject.offset = noise.mean()
        newObject.noise = noise
        newObject.long_range_noise = self.long_range_noise + "+" + other.long_range_noise
        newObject.calculate_autocorrelation()
        return newObject

    def max_deviation(self):
        return abs(self.noise - self.noise.mean()).max()

    def generate_noise(self):
        length = self.z[-1] - self.z[0]
        npoints = len(self.z)
        if npoints % 2 == 0:
            npoints += 1
            extended = True
        else:
            extended = False
        df = 1. / length
        C = np.zeros(shape=npoints, dtype=complex)
        half_size = (npoints - 1) / 2

        if self.long_range_noise == "AWGN":
            exponent = 0
        elif self.long_range_noise == "1/f":
            exponent = 1
        elif self.long_range_noise == "1/f2":
            exponent = 2
        else:
            raise ValueError("Unknown self.long_range_noise value. Cannot set the exponent.")

        for idx in range(1, half_size + 1):
            fk = idx * df / (2 * np.pi)
            ck = 1. / fk ** exponent
            phase = np.random.uniform(0, 2 * np.pi)
            C[half_size + idx] = ck * np.exp(1j * phase)
            C[half_size - idx] = ck * np.exp(-1j * phase)
        y = np.fft.ifft(np.fft.ifftshift(C))
        y *= self.long_range_parameter / abs(y).max()
        if extended:
            y = y[:-1]
        return np.real(y)

    def generate_short_range_noise(self):
        if self.short_range_noise.lower() == "gaussian":
            self.short_range = np.random.randn(len(self.z)) * self.short_range_parameter
        else:
            raise NotImplementedError("This short range has not yet been implemented")
        return self.short_range

    def generate_long_range_noise(self):
        xx = np.linspace(0, self.z.max(), self.n_points_long_range)
        if self.long_range_noise.lower() == "gaussian":
            long_range = np.random.randn(self.n_points_long_range) * self.long_range_parameter
        elif self.long_range_noise.lower() == "quadratic":
            points = np.linspace(-1, 1, self.n_points_long_range)
            long_range = points ** 2 * self.long_range_parameter
        else:
            raise NotImplementedError("This long range has not yet been implemented")
        # interpolate for the short mesh
        y = interp1d(xx, long_range, kind="cubic", fill_value="extend")
        return y(self.z)

    def calculate_noise(self, **kwargs):
        noise = self.generate_long_range_noise() + self.generate_short_range_noise() + self.offset
        if kwargs.get("plot", False):
            plt.figure()
            plt.plot(self.z, noise)
        return noise

    def calculate_autocorrelation(self):
        self.f = np.linspace(-1. / (2 * (self.z[1] - self.z[0])), 1. / (2 * (self.z[1] - self.z[0])), len(self.z))
        self.noise_spectrum = np.fft.fft(self.noise)
        self.power_spectrum = self.noise_spectrum * np.conj(self.noise_spectrum)
        self.autocorrelation = np.fft.ifftshift(np.fft.ifft(self.power_spectrum))
        self.power_spectrum = np.fft.fftshift(self.power_spectrum)
        return self.autocorrelation, self.power_spectrum

    def calculate_noise(self, noise_spectrum):
        return np.fft.ifftshift(np.fft.ifft(noise_spectrum))

    def plot_noise_properties(self, fig=None, ax=None, **kwargs):
        if self.autocorrelation is None:
            self.calculate_autocorrelation()
        if fig is None:
            plt.figure()
        else:
            plt.figure(fig.number)

        plt.subplot(211)
        l1, = plt.plot(self.z, self.noise, **kwargs)
        plt.title("Noise")
        plt.xlabel("z")
        plt.ylabel("Noise")
        plt.subplot(223)
        l2, = plt.semilogy(np.linspace(-self.z.max() / 2., self.z.max() / 2., len(self.z)),
                           abs(self.autocorrelation) ** 2,
                           **kwargs)
        plt.title("|R(z)|^2")
        plt.xlabel("z")
        plt.subplot(224)
        l3, = plt.loglog(self.f, abs(self.power_spectrum) ** 2, **kwargs)
        plt.title("|S(f)|^2")
        plt.xlabel("f")
        plt.tight_layout()
        return l1, l2, l3


class StandardNoise(object):
    def __init__(self, z=np.linspace(0, 20, 1000), offset=0):
        self.z = z
        self.dz = self.z[1] - self.z[0]
        self.offset = offset
        self.__autocorrelation_is_calculated = False
        self.__profile = None
        self.ideal_correlation_function = None

    def concatenate(self, other):
        z = np.append(self.z, other.z[1:] + self.z[-1])
        print z[-1]
        noise = np.append(self.noise, other.noise[1:])
        profile = np.append(self.profile, other.profile[1:])
        new = StandardNoise(z=z)
        new.noise = noise
        new.profile = profile
        new.offset = profile.mean() - noise.mean()
        return new

    def generate_noise_from_pds(self, noise_type):
        length = self.z[-1] - self.z[0]
        npoints = len(self.z)
        if npoints % 2 == 0:
            npoints += 1
            extended = True
        else:
            extended = False
        df = 1. / length
        C = np.zeros(shape=npoints, dtype=complex)
        half_size = (npoints - 1) / 2

        if noise_type.lower() == "awgn":
            exponent = 0
        elif noise_type == "1/f":
            exponent = 1
        elif noise_type == "1/f2":
            exponent = 2
        else:
            raise ValueError("Unknown self.long_range_noise value. Cannot set the exponent.")

        for idx in range(1, half_size + 1):
            fk = idx * df / (2 * np.pi)
            ck = 1. / fk ** exponent
            phase = np.random.uniform(0, 2 * np.pi)
            C[half_size + idx] = ck * np.exp(1j * phase)
            C[half_size - idx] = ck * np.exp(-1j * phase)
        y = np.fft.ifft(np.fft.ifftshift(C))
        y *= self.sigma / abs(y).max()
        # y += self.offset
        if extended:
            y = y[:-1]
        return np.real(y)

    def generate_correlated_noise(self, correlation_length):
        sigma = self.sigma
        self.correlation_length = correlation_length
        if self.correlation_length == 0:
            r = 0
        else:
            r = np.exp(-self.dz / correlation_length)
        print "Correlation factor: ", r
        zz = self.z - self.z[-1] / 2.
        self.ideal_correlation_function = sigma ** 2 * np.exp(- abs(zz) / correlation_length)
        y = np.zeros(self.z.shape)
        # generate the first point drawing it from a gaussian distribution centered on the mean and with sigma std.
        # y[0] = sigma * np.random.randn()
        y[0] = 0
        for i in range(1, len(self.z)):
            y[i] = r * y[i - 1] + sigma * np.sqrt(1 - r ** 2) * np.random.randn()
        # y += self.offset
        return y

    def generate_regular_noise(self, noise_type, **kwargs):
        if noise_type == "linear":
            y = np.linspace(0, 1, len(self.z)) * self.sigma
        elif noise_type == "quadratic":
            y = np.linspace(0, 1, len(self.z)) ** 2 * self.sigma
        elif noise_type == "logistic":
            y = 1. / (1 + np.exp(- kwargs.get("k", 1) * (self.z - self.z.mean()))) - 0.5
            y *= self.sigma / abs(y).max()
        else:
            raise ValueError("Unknown noise type")
        return y

    def generate_noise(self, noise_type, sigma, **kwargs):
        self.noise_type = noise_type
        self.sigma = sigma
        if self.noise_type == "correlated":
            if kwargs.has_key("correlation_length"):
                self.correlation_length = kwargs["correlation_length"]
                self.noise = self.generate_correlated_noise(self.correlation_length)
            else:
                raise ValueError("Dictionary is missing the 'correlation_length'.")
        elif self.noise_type.lower() in ["awgn", "1/f", "1/f2"]:
            self.noise = self.generate_noise_from_pds(self.noise_type)
        elif self.noise_type.lower() in ["linear", "quadratic", "logistic"]:
            self.noise = self.generate_regular_noise(self.noise_type, **kwargs)
        else:
            raise ValueError("Unknown noise type.")
        self.profile = self.noise + self.offset
        return self.noise

    def max_deviation(self):
        return abs(self.noise - self.noise.mean()).max()

    def calculate_noise_properties(self):
        # self.f = np.linspace(-1. / (2 * (self.z[1] - self.z[0])), 1. / (2 * (self.z[1] - self.z[0])), len(self.z))

        # print("Length-amplitude product: ", self.z[-1] * abs(self.noise).max())
        print("Calculating autocorrelation")
        self.f = np.fft.fftshift(np.fft.fftfreq(len(self.z), np.diff(self.z)[0]))
        self.noise_spectrum = np.fft.fft(self.noise)
        self.power_spectrum = self.noise_spectrum * np.conj(self.noise_spectrum)
        self.autocorrelation = np.fft.ifftshift(np.fft.ifft(self.power_spectrum))
        self.power_spectrum = np.fft.fftshift(self.power_spectrum)
        self.z_autocorr = np.fft.fftshift(np.fft.fftfreq(len(self.f), np.diff(self.f)[0]))
        self.__autocorrelation_is_calculated = True
        return self.z_autocorr, self.autocorrelation, self.f, self.power_spectrum

    @property
    def profile(self):
        return self.__profile

    @profile.setter
    def profile(self, value):
        self.__profile = value

    def plot_noise_properties(self, fig=None, ax=None, **kwargs):
        if not self.__autocorrelation_is_calculated:
            self.calculate_noise_properties()
        if fig is None:
            fig = plt.figure()
        else:
            fig = plt.figure(fig.number)

        plt.subplot(211)
        l1, = plt.plot(self.z, self.noise, **kwargs)
        plt.title("Noise")
        plt.xlabel("z")
        plt.ylabel("Noise")

        plt.subplot(234)
        l2, = plt.semilogy(self.z_autocorr, abs(self.autocorrelation) ** 2, label="Calculated autocorrelation",
                           **kwargs)
        # if self.ideal_correlation_function is not None:
        #     plt.gca().twinx().semilogy(self.z_autocorr,
        #                  self.ideal_correlation_function, ":",
        #                  label="Ideal autocorrelation")
        plt.title("|R(z)|^2")
        plt.xlabel("z")

        plt.subplot(235)
        l3, = plt.loglog(self.f, abs(self.power_spectrum) ** 2, **kwargs)
        plt.title("|S(f)|^2")
        plt.xlabel("f")
        plt.subplot(236)
        plt.hist(self.noise, bins=int(np.sqrt(len(self.noise))))
        plt.tight_layout()
        return fig, ax, [l1, l2, l3]


def main_Standard_Noise():
    z = np.linspace(0, 1, 1000)
    thisnoise = StandardNoise(z=z, offset=0)
    # thisnoise.generate_noise(noise_type="correlated", sigma=800, correlation_length=1)
    thisnoise.generate_noise(noise_type="logistic", sigma=1, k=10)
    thisnoise2 = StandardNoise(z=z, offset=thisnoise.profile[-1] * 2)
    # thisnoise.generate_noise(noise_type="correlated", sigma=800, correlation_length=1)
    thisnoise2.generate_noise(noise_type="logistic", sigma=1, k=10)

    noise = np.append(thisnoise.noise, thisnoise2.noise + 2 * thisnoise.noise[-1])
    z = np.linspace(0, 1, len(noise))
    plt.plot(z, noise)
    obj = thisnoise.concatenate(thisnoise2)
    plt.figure()
    plt.plot(thisnoise.z, thisnoise.noise, ":")
    plt.plot(thisnoise2.z, thisnoise2.noise, ".")
    plt.plot(obj.z, obj.profile, "-.")


if __name__ == "__main__":
    # thisnoise = Noise(short_range_parameter=0.01, long_range_parameter=0.1)
    # thisnoise.generate_short_range_noise()
    # thisnoise.generate_long_range_noise()
    # thisnoise.calculate_noise()
    # thisnoise.calculate_autocorrelation()
    # thisnoise.plot_noise_properties()

    # thisnoise = Noise(long_range_noise="AWGN", long_range_parameter=0., offset=7.0)
    # thisnoise.plot_noise_properties()
    # othernoise = Noise(long_range_noise="1/f", long_range_parameter=0.2, offset=7.0)
    # othernoise.plot_noise_properties()
    # sumnoise = thisnoise + othernoise
    # sumnoise.plot_noise_properties()

    main_Standard_Noise()
    plt.show()
