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


def calculate_profile_properties(z: np.ndarray = None, profile: np.ndarray = None):
    """
    Function to calculate the noise properties (autocorrelation and power density spectrum) of the noise on the
    waveguide profile

    :param z: z mesh of the system
    :type z: numpy.ndarray
    :param profile: Profile of the varying variable of the waveguide.
    :type profile: numpy.ndarray

    :return: z_autocorr, autocorrelation, f, power_spectrum: Returns the autocorrelation profile (z axis included)
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


class NoiseProfile(object):
    """
    Base class to define a generic noise profile.

    Initialize the noise object passing a numpy array containing the mesh along z, the noise amplitude and an offset.

    :param z: linearly spaced space mesh [*meter*].
    :type z: numpy.ndarray
    :param noise_amplitude: Amplitude of the noise profile.
    :type noise_amplitude: float
    :param offset: Offset of the noise profile
    :type offset: float

    The following block of code initialises and plots the profile of

    """

    def __init__(self, z: np.ndarray = None, noise_amplitude: float = 0., offset: float = 0.):
        logger = logging.getLogger(__name__)
        logger.debug("Creating NoiseProfile object. "
                     "z.shape={0}; noise_amplitude={1}; offset={2}.".format(z.shape, noise_amplitude, offset))
        if z is None:
            raise ValueError("z cannot be None")
        if noise_amplitude < 0:
            warnings.warn("noise_amplitude is negative. Taking the absolute value", UserWarning)
        self.__noise_amplitude = abs(noise_amplitude)

        self.__z = z
        self.__offset = offset
        self.__length = self.z.max() - self.z.min()
        self.__dz = np.diff(self.z)[0]
        self.__profile = self.__noise_amplitude * np.ones(shape=self.z.shape) + self.__offset
        self.__autocorrelation_is_calculated = False
        logger.debug("NoiseProfile object creates successfully.")

    def __repr__(self):
        text = f"{self.__class__.__name__} object at {hex(id(self))}.\n\tLength: {self.length} m\n\t" \
               f"Noise amplitude:{self.noise_amplitude}\n\tNoise offset:{self.offset}"
        return text

    @property
    def length(self):
        """
        Length of the structure

        """
        return self.__length

    @property
    def z(self):
        """
        Z-mesh used to discretise the profile

        """
        return self.__z

    @property
    def dz(self):
        """
        Discretisation unit cell.

        """
        return self.__dz

    @property
    def profile(self):
        """
        Profile of the structure.

        """
        return self.__profile

    @property
    def noise_amplitude(self):
        """
        Noise amplitude of the noise profile.

        """
        return self.__noise_amplitude

    @property
    def offset(self):
        """
        Offset of the noise profile.

        """
        return self.__offset

    def concatenate(self, other):
        """
        Method to concatenate two noise tracks.

        .. warning:: This method has not been tested completely.

        :param other: Another instance of a NoiseProfile
        :type other: :class:`pynumpm.NoiseProfile`

        :return:
        """
        logger = logging.getLogger(__name__)
        logger.debug("Concatenating two objects"
                     "other={0}".format(other))
        if self.dz != other.dz:
            raise ValueError("The resolution of the two noise spectra are different. Set them to equal.")
        new_z = np.append(self.z, other.z + self.z[-1])
        newprofile = np.append(self.profile, other.profile)
        newnoise = NoiseProfile(z=new_z)
        newnoise.__profile = newprofile
        logger.debug("Concatenation successful.")
        return newnoise

    def load_noise_profile(self, noise_profile: np.ndarray):
        """
        Method used to load a user-generated noise profile.

        :param noise_profile: Array containing the noise profile
        :type noise_profile: numpy.ndarray

        """
        if not isinstance(noise_profile, np.ndarray):
            raise TypeError("'noise_profile' must be a numpy.ndarray object.")

        if noise_profile.shape != self.z.shape:
            raise ValueError("The shape of 'noise_profile' is {0}, while the shape of the discretization mesh is"
                             "{1}. They must be consistent.".format(noise_profile.shape, self.z.shape))
        self.__profile = noise_profile
        self.__offset = self.profile.mean()
        self.__noise_amplitude = self.profile - self.profile.mean()

    def plot_noise_properties(self, fig=None, **plotkwargs):
        """
        Function to plot the noise properties.

        :param fig: Figure handle, if the plot needs to be in a specific figure
        :param plotkwargs: Dictionary of properties to be passed to the plotting functions. This can be used e.g. to
                           define the colours and the size of the lines
        :return: fig, [ax1, ax2, ax3]. The handles to the figure object and the three axes of the figure.
        """
        logger = logging.getLogger(__name__)
        logger.debug("Plotting noise properties.")
        z_autocorr, autocorr, f, power_spectrum = calculate_profile_properties(self.z, self.profile)
        if fig is None:
            fig = plt.figure()
        else:
            fig = plt.figure(fig.number)

        plt.subplot(211)
        ax1 = plt.gca()
        l1, = plt.plot(self.z, self.profile, **plotkwargs)
        plt.title("Noise")
        plt.xlabel("z")
        plt.ylabel("Noise")

        plt.subplot(234)
        ax2 = plt.gca()
        l2, = plt.semilogy(z_autocorr, abs(autocorr) ** 2, label="Calculated autocorrelation",
                           **plotkwargs)
        plt.title("|R(z)|^2")
        plt.xlabel("z")

        plt.subplot(235)
        ax3 = plt.gca()
        l3, = plt.loglog(f, abs(power_spectrum) ** 2, **plotkwargs)
        plt.title("|S(f)|^2")
        plt.xlabel("f")
        plt.subplot(236)
        plt.hist(self.profile, bins=int(np.sqrt(len(self.profile))))
        plt.title("Histogram of the noise.")
        plt.tight_layout()
        return fig, [ax1, ax2, ax3]


class NoiseFromSpectrum(NoiseProfile):
    """
    Class to create a noise profile given a specific noise power spectrum. It inherits from NoiseProfile.
    It can create `Additive White Gaussian Noise (awgn) <https://en.wikipedia.org/wiki/Additive_white_Gaussian_noise>`_,
    `1/f noise <https://en.wikipedia.org/wiki/Pink_noise>`_ and
    `1/f2 noise <https://en.wikipedia.org/wiki/Brownian_noise>`_.

    Initialize the noise object passing a numpy array containing the mesh along z, the noise amplitude, an offset
    and a string describing the power spectrum of the noise.

    :param z: linearly spaced space mesh [*meter*].
    :type z: numpy.ndarray
    :param noise_amplitude: Amplitude of the noise profile.
    :type noise_amplitude: float
    :param offset: Offset of the noise profile
    :type offset: float
    :param profile_spectrum: Noise profile of the simulated structure. Can be one of "awgn", "1/f", "1/f2".
    :type profile_spectrum: str

    """

    def __init__(self, z: np.ndarray = None, offset: float = 0, noise_amplitude: float = 0.,
                 profile_spectrum: str = None):

        logger = logging.getLogger(__name__)
        logger.debug("Creating a NoiseFromSpectrum object. "
                     "z.shape={0}; offset={1}; noise_amplitude={2}; profile_spectrum={3}.".format(z.shape,
                                                                                                  offset,
                                                                                                  noise_amplitude,
                                                                                                  profile_spectrum))
        NoiseProfile.__init__(self, z, offset=offset, noise_amplitude=noise_amplitude)
        if profile_spectrum is None:
            raise ValueError("'profile_spectrum' must be set")
        if profile_spectrum.lower() in ["awgn", "1/f", "1/f2"]:
            self.__profile_spectrum = profile_spectrum.lower()
        else:
            raise ValueError("'profile_spectrum' has to be 'awgn', '1/f' or '1/f2'")
        self.__profile = self._generate_noise()
        logger.debug("NoiseFromSpectrum object created.")

    def __repr__(self):
        text = f"{self.__class__.__name__} object at {hex(id(self))}.\n\tLength: {self.length} m\n\t" \
               f"Noise amplitude:{self.noise_amplitude}\n\tNoise offset:{self.offset};" \
               f"\n\tProfile spectrum: {self.profile_spectrum}"
        return text

    @property
    def profile_spectrum(self):
        """
        Type of noise spectrum of the structure

        """
        return self.__profile_spectrum

    @property
    def profile(self):
        """
        Profile of the structure

        """
        return self.__profile

    def _generate_noise(self):
        """
        Function that generates the noise profile.

        """
        # This function generates the noise.

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
        logger.debug("Noise profile generated.")
        return np.real(y) + self.offset


class CorrelatedNoise(NoiseProfile):
    """
    Class to describe correlated noise.

    ..warning:: This class hasn't been tested completely. It might be buggy.
    """

    def __init__(self, z=None, offset=0, noise_amplitude=0., correlation_length=0.):
        """

        :param z:
        :param offset:
        :param noise_amplitude:
        :param correlation_length:
        """
        logger = logging.getLogger(__name__)
        logger.debug("Creating CorrelatedNoise object. z.shape={0}; offset={1}; noise_amplitude={2}; "
                     "correlation_length={3}".format(z.shape, offset, noise_amplitude, correlation_length))
        NoiseProfile.__init__(self, z=z, offset=offset, noise_amplitude=noise_amplitude)
        if correlation_length < 0:
            warnings.warn("correlation_length is negative. I will get the absolute value", UserWarning)
        self.__correlation_length = abs(correlation_length)
        NoiseProfile.__profile = self.generate_noise()
        logger.debug("CorrelatedNoise object created.")

    def __repr__(self):
        text = f"{self.__class__.__name__} object at {hex(id(self))}.\n\tLength: {self.length} m\n\t" \
               f"Noise amplitude:{self.noise_amplitude}\n\tNoise offset:{self.offset};" \
               f"\n\tCorrelation length: {self.correlation_length}."
        return text

    @property
    def correlation_length(self):
        return self.__correlation_length

    @property
    def profile(self):
        return self.__profile

    def generate_noise(self):
        logger = logging.getLogger(__name__)
        logger.debug("Generating noise profile.")
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
        logger.debug("Noise profile generated correctly.")
        return y + self.offset
