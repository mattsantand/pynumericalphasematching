

import numpy as np
import matplotlib.pyplot as plt
from pynumpm import noise


class Waveguide(object):
    """
    Waveguide class.

    It is used to describe waveguide profiles. It can generate noisy profiles (via the functions in the :mod:`noise`
    module), it can load user-defined profiles (they must be consistent with the user specified mesh).
    Moreover, the user can specify a poling structure (functionality unused in the CalculatePhasematching at the
    moment).

    """

    # Attributes:
    #     * z: numpy array of the space discretization [in meter]. Linear spacing is required.
    #     * dz: spatial discretization [in meter].
    #     * length: waveguide length [in meter].
    #     * profile: numpy array containing the waveguide profile. Has the same dimension of _z_.
    #     * poling_period: poling period of the structure [in meter].
    #     * poling_period_um: poling period of the structure [in micron].
    #     * poling_structure: vector containing the orientation of the chi(2), if the poling orientation is specified.
    #     * poling_structure_set: boolean that tracks if the poling structure has been set.
    #     * nominal_parameter: the nominal fabrication parameter of the waveguide.
    #     * nominal_parameter_name: the name of the fabrication parameter under investigation.
    #
    # Methods:
    #     * create_uniform_waveguide: function to create a uniform waveguide profile
    #     * load_waveguide_profile: function to load a user-defined waveguide profile.
    #     * create_noisy_waveguide: function to create a noisy waveguide.
    #     * load_poling_structure: function to load a poling structure.
    #     * plot: function to plot the waveguide profile.

    def __init__(self, z=None, poling_period=None, nominal_parameter=1., nominal_parameter_name=""):
        """
        Initialize the waveguide by providing a z-mesh and the nominal parameter of the profile. This will automatically
        generate a uniform profile with the specified nominal parameter.

        :param z: linearly spaced space mesh [in **meter**].
        :type z: numpy.ndarray
        :param poling_period: poling period of the structure [in **meter**].
        :type poling_period: float
        :param nominal_parameter: nominal parameter of the structure [variable units, depend on the Sellmeier used].
        :type nominal_parameter: float
        :param nominal_parameter_name: name of the nominal parameter (used for the axes). LaTeX syntax is allowed.
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
        """*Uniformly* spaced mesh [in meter]"""
        return self.__z

    @property
    def dz(self):
        """Space discretization [in meter]. It is automatically calculated from the input z."""
        return self.__dz

    @property
    def length(self):
        """Crystal length [in meter]. It is automatically calculated from the input z."""
        return self.__length

    @property
    def profile(self):
        """Waveguide profile (array). Array containing the values of the waveguide profile to be simulated (e.g.,
        the waveguide width or temperature profile).
        """
        return self.__waveguide_profile

    @property
    def poling_period(self):
        """Poling period [in meter]."""
        return self.__poling_period

    @property
    def poling_period_um(self):
        """Poling period [in micron]."""
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
        """
        Nominal fabrication parameter of the waveguide
        """
        return self.__nominal_parameter

    @nominal_parameter.setter
    def nominal_parameter(self, value):
        """Nominal fabrication parameter of the waveguide"""
        self.__nominal_parameter = value

    def create_uniform_waveguide(self, nominal_parameter):
        """
        Function to create a uniform waveguide profile.

        :param nominal_parameter: Nominal parameter of the waveguide.
        """
        y = nominal_parameter * np.ones(shape=self.z.shape)
        self.nominal_parameter = nominal_parameter
        #return y

    def load_waveguide_profile(self, waveguide_profile):
        """
        Function to load a user-defined waveguide profile.

        :param waveguide_profile: array with the waveguide profile. It *must* have the same shape as the z-mesh
        :type waveguide_profile: numpy.ndarray
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
        Function to create a noisy waveguide with a 1/f-like spectrum.

        :param noise_profile: String identifying the noise profile. Default is *1/f*. Options are AWGN | 1/f | 1/f2
        :type noise_profile: str
        :param nominal_parameter: Nominal parameter of the waveguide. Default is *None*. If unspecified, it uses the one initialized when defining the object.
        :type nominal_parameter: float
        :param noise_amplitude: Range of the noise (in the same units as :py:meth:`~Waveguide.Waveguide.create_noisy_waveguide.nominal_parameter`)
        :type noise_amplitude: float
        """
        if nominal_parameter is None:
            nominal_parameter = self.nominal_parameter
        thisnoise = noise.NoiseFromSpectrum(z=self.z,
                                            offset=nominal_parameter,
                                            profile_spectrum=noise_profile,
                                            noise_amplitude=noise_amplitude)
        self.load_waveguide_profile(thisnoise.profile)

    def load_poling_structure(self, poling_structure):
        """
        Function to load the poling structure.

        .. warning:: This function is untested.

        :param poling_structure: Array containing the orientation of the poling.
        :type poling_structure: numpy.ndarray
        """
        if poling_structure.shape != self.profile.shape:
            raise ValueError("The poling_structure must have the same shape as the waveguide profile!")
        self.__poling_structure = poling_structure
        self.__poling_period = +np.infty

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