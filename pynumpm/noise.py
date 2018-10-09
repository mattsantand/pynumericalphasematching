# -*- coding: utf-8 -*-
"""
Created on 26.09.2017 08:30

.. module: 
.. moduleauthor: Matteo Santandrea <matteo.santandrea@upb.de>
"""

import numpy as np
import matplotlib.pyplot as plt


class NoiseProfile(object):
    def __init__(self, z=None, noise_amplitude=0., offset=0.):
        if z is None:
            raise ValueError("z cannot be None")
        if noise_amplitude < 0:
            raise Warning("noise_amplitude is negative. Taking the absolute value")
        self.__noise_amplitude = abs(noise_amplitude)

        self.__z = z
        self.__offset = offset
        self.__dz = np.diff(self.z)[0]
        self.__profile = np.zeros(shape=self.z.shape)
        self.__autocorrelation_is_calculated = False
        self.f = None
        self.noise_spectrum = None
        self.power_spectrum = None
        self.autocorrelation = None
        self.power_spectrum = None
        self.z_autocorr = None

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

    def calculate_noise_properties(self):
        print("Calculating noise properties")
        self.f = np.fft.fftshift(np.fft.fftfreq(len(self.z), np.diff(self.z)[0]))
        self.noise_spectrum = np.fft.fft(self.profile)
        self.power_spectrum = self.noise_spectrum * np.conj(self.noise_spectrum)
        self.autocorrelation = np.fft.ifftshift(np.fft.ifft(self.power_spectrum))
        self.power_spectrum = np.fft.fftshift(self.power_spectrum)
        self.z_autocorr = np.fft.fftshift(np.fft.fftfreq(len(self.f), np.diff(self.f)[0]))
        self.__autocorrelation_is_calculated = True
        return self.z_autocorr, self.autocorrelation, self.f, self.power_spectrum

    def plot_noise_properties(self, fig=None, ax=None, **kwargs):
        if not self.__autocorrelation_is_calculated:
            self.calculate_noise_properties()
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
        l2, = plt.semilogy(self.z_autocorr, abs(self.autocorrelation) ** 2, label="Calculated autocorrelation",
                           **kwargs)
        plt.title("|R(z)|^2")
        plt.xlabel("z")

        plt.subplot(235)
        l3, = plt.loglog(self.f, abs(self.power_spectrum) ** 2, **kwargs)
        plt.title("|S(f)|^2")
        plt.xlabel("f")
        plt.subplot(236)
        plt.hist(self.profile, bins=int(np.sqrt(len(self.profile))))
        plt.tight_layout()
        return fig, ax, [l1, l2, l3]


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
        # print "Generating {s} spectrum.".format(s=self.profile_spectrum)
        length = self.z[-1] - self.z[0]
        npoints = len(self.z)
        if npoints % 2 == 0:
            npoints += 1
            extended = True
        else:
            extended = False
        df = 1. / length
        frequency_coefficients = np.zeros(shape=npoints, dtype=complex)
        half_size = (npoints - 1) / 2

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
            raise Warning("correlation_length is negative. I will get the absolute value")
        self.__correlation_length = abs(correlation_length)
        NoiseProfile.__profile = self.generate_noise()

    @property
    def correlation_length(self):
        return self.__correlation_length

    @property
    def profile(self):
        return self.__profile

    def generate_noise(self):
        sigma = self.noise_amplitude
        if self.correlation_length == 0:
            r = 0
        else:
            r = np.exp(-self.dz / self.correlation_length)
        # print "Correlation factor: ", r
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


# class Noise(object):
#     def __init__(self, **kwargs):
#         self.z = kwargs.get("z", np.linspace(0, 20, 1000))
#         self.dz = self.z[1] - self.z[0]
#         self.offset = kwargs.get("offset", 0.)
#         self.long_range_noise = kwargs.get("long_range_noise", "gaussian")
#         if not (self.long_range_noise.lower() in _LONGRANGENOISE):
#             raise ValueError("long_range_noise variable has to be in {0}".format(_LONGRANGENOISE))
#         self.long_range_parameter = kwargs.get("long_range_parameter")
#         if self.long_range_noise.lower() in _LONGRANGENOISE[:3]:
#             self.short_range_noise = kwargs.get("short_range_noise", "gaussian")
#             if not self.short_range_noise.lower() == "gaussian":
#                 raise NotImplementedError("This short noise has not yet been implemented")
#             self.short_range_parameter = kwargs.get("short_range_parameter")
#             self.n_points_long_range = kwargs.get("number_points_long_range", 10)
#             self.n_points_short_range = kwargs.get("number_points_short_range", 300)
#             self.noise = self.generate_long_range_noise() + self.generate_short_range_noise() + self.offset
#         else:
#             self.noise = self.generate_noise() + self.offset
#         self.calculate_autocorrelation()
#
#     def __add__(self, other):
#         if np.any((self.z - other.z) != 0):
#             raise AttributeError("The z vectors are different!")
#         noise = self.noise + other.noise
#         newObject = Noise(z=self.z, offset=0, long_range_noise="AWGN", long_range_parameter=0)
#         newObject.offset = noise.mean()
#         newObject.noise = noise
#         newObject.long_range_noise = self.long_range_noise + "+" + other.long_range_noise
#         newObject.calculate_autocorrelation()
#         return newObject
#
#     def max_deviation(self):
#         return abs(self.noise - self.noise.mean()).max()
#
#     def generate_noise(self):
#         length = self.z[-1] - self.z[0]
#         npoints = len(self.z)
#         if npoints % 2 == 0:
#             npoints += 1
#             extended = True
#         else:
#             extended = False
#         df = 1. / length
#         C = np.zeros(shape=npoints, dtype=complex)
#         half_size = (npoints - 1) / 2
#
#         if self.long_range_noise == "AWGN":
#             exponent = 0
#         elif self.long_range_noise == "1/f":
#             exponent = 1
#         elif self.long_range_noise == "1/f2":
#             exponent = 2
#         else:
#             raise ValueError("Unknown self.long_range_noise value. Cannot set the exponent.")
#
#         for idx in range(1, half_size + 1):
#             fk = idx * df / (2 * np.pi)
#             ck = 1. / fk ** exponent
#             phase = np.random.uniform(0, 2 * np.pi)
#             C[half_size + idx] = ck * np.exp(1j * phase)
#             C[half_size - idx] = ck * np.exp(-1j * phase)
#         y = np.fft.ifft(np.fft.ifftshift(C))
#         y *= self.long_range_parameter / abs(y).max()
#         if extended:
#             y = y[:-1]
#         return np.real(y)
#
#     def generate_short_range_noise(self):
#         if self.short_range_noise.lower() == "gaussian":
#             self.short_range = np.random.randn(len(self.z)) * self.short_range_parameter
#         else:
#             raise NotImplementedError("This short range has not yet been implemented")
#         return self.short_range
#
#     def generate_long_range_noise(self):
#         xx = np.linspace(0, self.z.max(), self.n_points_long_range)
#         if self.long_range_noise.lower() == "gaussian":
#             long_range = np.random.randn(self.n_points_long_range) * self.long_range_parameter
#         elif self.long_range_noise.lower() == "quadratic":
#             points = np.linspace(-1, 1, self.n_points_long_range)
#             long_range = points ** 2 * self.long_range_parameter
#         else:
#             raise NotImplementedError("This long range has not yet been implemented")
#         # interpolate for the short mesh
#         y = interp1d(xx, long_range, kind="cubic", fill_value="extend")
#         return y(self.z)
#
#     def calculate_noise(self, **kwargs):
#         noise = self.generate_long_range_noise() + self.generate_short_range_noise() + self.offset
#         if kwargs.get("plot", False):
#             plt.figure()
#             plt.plot(self.z, noise)
#         return noise
#
#     def calculate_autocorrelation(self):
#         self.f = np.linspace(-1. / (2 * (self.z[1] - self.z[0])), 1. / (2 * (self.z[1] - self.z[0])), len(self.z))
#         self.noise_spectrum = np.fft.fft(self.noise)
#         self.power_spectrum = self.noise_spectrum * np.conj(self.noise_spectrum)
#         self.autocorrelation = np.fft.ifftshift(np.fft.ifft(self.power_spectrum))
#         self.power_spectrum = np.fft.fftshift(self.power_spectrum)
#         return self.autocorrelation, self.power_spectrum
#
#     def calculate_noise(self, noise_spectrum):
#         return np.fft.ifftshift(np.fft.ifft(noise_spectrum))
#
#     def plot_noise_properties(self, fig=None, ax=None, **kwargs):
#         if self.autocorrelation is None:
#             self.calculate_autocorrelation()
#         if fig is None:
#             plt.figure()
#         else:
#             plt.figure(fig.number)
#
#         plt.subplot(211)
#         l1, = plt.plot(self.z, self.noise, **kwargs)
#         plt.title("Noise")
#         plt.xlabel("z")
#         plt.ylabel("Noise")
#         plt.subplot(223)
#         l2, = plt.semilogy(np.linspace(-self.z.max() / 2., self.z.max() / 2., len(self.z)),
#                            abs(self.autocorrelation) ** 2,
#                            **kwargs)
#         plt.title("|R(z)|^2")
#         plt.xlabel("z")
#         plt.subplot(224)
#         l3, = plt.loglog(self.f, abs(self.power_spectrum) ** 2, **kwargs)
#         plt.title("|S(f)|^2")
#         plt.xlabel("f")
#         plt.tight_layout()
#         return l1, l2, l3


def example_Noise():
    z = np.linspace(0, 20, 10000)
    thisnoise = NoiseFromSpectrum(z=z, noise_amplitude=0.2, offset=0.4, profile_spectrum="awgn")
    thisnoise.plot_noise_properties()
    othernoise = CorrelatedNoise(z=z, noise_amplitude=0.2, correlation_length=0.2)
    othernoise.plot_noise_properties()
    concatenate_noise = NoiseProfile.concatenate(thisnoise, othernoise)
    concatenate_noise.plot_noise_properties()
    plt.show()


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

    # main_Standard_Noise()
    example_Noise()
