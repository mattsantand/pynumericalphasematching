# -*- coding: utf-8 -*-
"""
Waveguide module.

This module is used to describe waveguide objects, that can be used to calculate the phasematching.
The _Waveguide_ class can describe a waveguide with a uniform profile, with an externally provided profile or a noisy
profile.

Content:
    * class _Waveguide_: waveguide class to describe the waveguide.
    * function _example_: function to explain how to use the waveguide module.
"""

import numpy as np
import matplotlib.pyplot as plt
import Noise


class Waveguide(object):
    """
    Waveguide class.

    It is used to describe waveguides.

    Attributes:
        * z: numpy array of the space discretization [in meter]. Linear spacing is required.
        * dz: spatial discretization [in meter].
        * length: waveguide length [in meter].
        * profile: numpy array containing the waveguide profile. Has the same dimension of _z_.
        * poling_period: poling period of the structure [in meter].
        * poling_period_um: poling period of the structure [in micron].
        * poling_structure: vector containing the orientation of the chi(2), if the poling orientation is specified.
        * poling_structure_set: boolean that tracks if the poling structure has been set.
        * nominal_parameter: the nominal fabrication parameter of the waveguide.
        * nominal_parameter_name: the name of the fabrication parameter under investigation.

    Methods:
        * create_uniform_waveguide: function to create a uniform waveguide profile
        * load_waveguide_profile: function to load a user-defined waveguide profile.
        * create_noisy_waveguide: function to create a noisy waveguide.
        * load_poling_structure: function to load a poling structure.
        * plot: function to plot the waveguide profile.

    """

    def __init__(self, z=None, poling_period=None, nominal_parameter=1., nominal_parameter_name=""):
        """
        Constructor of the _Waveguide_ class.

        Input parameters:

        :param z: linearly spaced space mesh.
        :type z: numpy.ndarray
        :param poling_period: poling period of the structure [m].
        :type poling_period: float
        :param nominal_parameter: nominal parameter of the structure [variable units, depend on the Sellmeier used].
        :type nominal_parameter: float
        :param nominal_parameter_name: name of the nominal parameter (used for the axes). Allows LaTeX commands.
        :type nominal_parameter_name: string
        """
        self.__z = z
        self.__dz = np.diff(self.z)[0]
        self.__length = self.z[-1] - self.z[0]
        self.__poling_period = poling_period
        self.__nominal_parameter = nominal_parameter
        self.__waveguide_profile = self.nominal_parameter * np.ones(shape=self.z.shape)
        self.__nominal_parameter_name = nominal_parameter_name
        self.__poling_structure = None

    @property
    def z(self):
        """Space mesh [in meter]"""
        return self.__z

    @property
    def dz(self):
        """Space discretization [in meter]"""
        return self.__dz

    @property
    def length(self):
        """Crystal length [in meter]"""
        return self.__length

    @property
    def profile(self):
        """Waveguide profile (array)"""
        return self.__waveguide_profile

    @property
    def poling_period(self):
        """Poling period [in meter]"""
        return self.__poling_period

    @property
    def poling_period_um(self):
        """Poling period [in micron]"""
        return self.__poling_period * 1e6

    @property
    def nominal_parameter_name(self):
        """Name of the nominal fabrication parameter."""
        return self.__nominal_parameter_name

    @property
    def poling_structure_set(self):
        """Boolean to describe if the poling structure is set."""
        return self.poling_structure is not None

    @property
    def poling_structure(self):
        """Array containing the poling structure."""
        return self.__poling_structure

    @property
    def nominal_parameter(self):
        """Nominal fabrication parameter of the waveguide"""
        return self.__nominal_parameter

    @nominal_parameter.setter
    def nominal_parameter(self, value):
        """Nominal fabrication parameter of the waveguide"""
        self.__nominal_parameter = value

    def create_uniform_waveguide(self, nominal_parameter):
        """
        Function to create a uniform waveguide profile.

        :param nominal_parameter:
        :return:
        """
        y = nominal_parameter * np.ones(shape=self.z.shape)
        self.nominal_parameter = nominal_parameter
        return y

    def load_waveguide_profile(self, waveguide_profile):
        """
        Function to load the waveguide profile.

        :param waveguide_profile:
        :return:
        """
        if waveguide_profile.shape != self.z.shape:
            raise ValueError("The shape of the waveguide_profile {s1} is different from the z mesh {s2}".format(
                s1=waveguide_profile.shape,
                s2=self.z.shape))
        else:
            self.__waveguide_profile = waveguide_profile
            self.nominal_parameter = self.profile.mean()

    def create_noisy_waveguide(self, noise_profile="1/f", noise_amplitude=0.2, nominal_parameter=None):
        """
        Function to create a noisy waveguide.

        :param noise_profile:
        :param noise_amplitude:
        :param nominal_parameter:
        :return:
        """
        if nominal_parameter is None:
            nominal_parameter = self.nominal_parameter
        thisnoise = Noise.NoiseFromSpectrum(z=self.z,
                                            offset=nominal_parameter,
                                            profile_spectrum=noise_profile,
                                            noise_amplitude=noise_amplitude)
        self.load_waveguide_profile(thisnoise.profile)

    def load_poling_structure(self, poling_structure):
        if poling_structure.shape != self.profile.shape:
            raise ValueError("The poling_structure must have the same shape as the waveguide profile!")
        self.__poling_structure = poling_structure

    def plot(self, ax=None):
        """
        Function to plot the waveguide profile.

        :param ax: handle to axis, if you want to plot in specific axes.
        :return: fig, ax: handle to figure and axis objects
        """
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        else:
            plt.sca(ax)
            fig = plt.gcf()

        ax.plot(self.z * 1e3, self.profile)
        ax.set_xlabel("Length [mm]")
        ax.set_ylabel(self.nominal_parameter_name)
        ax.set_title("Waveguide profile")
        return fig, ax


def example():
    z = np.linspace(0, 0.020, 10000)
    thiswaveguide = Waveguide(z=z, nominal_parameter=7., nominal_parameter_name="Width")
    thiswaveguide.create_noisy_waveguide()
    thiswaveguide.plot()
    plt.show()


if __name__ == '__main__':
    example()