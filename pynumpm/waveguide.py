import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from pynumpm import noise, utils
import logging


class Waveguide(object):
    def __init__(self, length: float, poling_period: float = +np.infty):
        """
        Base class for the description of a waveguide object.

        :param length: Length of the waveguide [**meter**].
        :type length: float
        :param poling_period: Poling period of the structure, [**meter**]. Default: +np.infty
        :type poling_period: float
        """
        if np.isinf(poling_period):
            warnings.warn("The user has not provided a poling period. The default value of +np.infty will be used.",
                          UserWarning)
        self.__poling_period = poling_period
        self.__poling_structure_set = False
        self.__length = length

    @property
    def poling_period(self):
        """Poling period [**meter**]."""
        return self.__poling_period

    @property
    def poling_period_um(self):
        """Poling period [**micrometer**]."""
        return self.__poling_period * 1e6

    @property
    def length(self):
        """Crystal length [**meter**]."""
        return self.__length


class RealisticWaveguide(Waveguide):
    """
    RealisticWaveguide class.

    It is used to describe waveguide profiles. It can generate noisy profiles (via the functions in the :mod:`noise`
    module), it can load user-defined profiles (they must be consistent with the user specified mesh).
    Moreover, the user can specify a poling structure (functionality unused in the CalculatePhasematching at the
    moment).

    """

    def __init__(self, z: np.ndarray, poling_period: float = np.infty, nominal_parameter: float = 1.,
                 nominal_parameter_name: str = ""):
        """
        Initialize the waveguide by providing a z-mesh and the nominal parameter of the profile. This will automatically
        generate a uniform profile with the specified nominal parameter.

        :param z: linearly spaced space mesh [**meter**].
        :type z: numpy.ndarray
        :param poling_period: poling period of the structure [**meter**].
        :type poling_period: float
        :param nominal_parameter: nominal parameter of the structure [variable units, depend on the Sellmeier used].
        Default: None.
        :type nominal_parameter: float
        :param nominal_parameter_name: name of the nominal parameter (used for the axes). LaTeX syntax is allowed.
        Default: empty string.
        :type nominal_parameter_name: string
        """

        self.__z = z
        self.__dz = np.diff(self.z)[0]
        self.__nominal_parameter = nominal_parameter
        # when an object of this class is initialized, this call creates a uniform waveguide
        self.__waveguide_profile = self.load_waveguide_profile()
        self.__nominal_parameter_name = nominal_parameter_name
        length = self.z[-1] - self.z[0]
        Waveguide.__init__(self, length, poling_period)
        if self.__nominal_parameter_name == "":
            warnings.warn(
                "The name of the variable parameter was left empty. "
                "That's really not a great name..",
                UserWarning)
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
    def profile(self):
        """Waveguide profile (array).
        Array containing the values of the waveguide profile to be simulated (e.g., the waveguide
        width or temperature profile).
        """
        return self.__waveguide_profile

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

    def load_waveguide_profile(self, waveguide_profile: np.ndarray = None):
        """
        Function to load a user-defined waveguide profile.

        :param waveguide_profile: Array with the waveguide profile, with respect to the parameter under investigation,
        or *None*. If *None*, it will create a uniform waveguide with the a constant `waveguide.nominal_parameter`.
        If an array, it *must* have the same shape as the z-mesh. Default: None
        :type waveguide_profile: numpy.ndarray
        :return: The numpy.ndarray containing the waveguide profile.
        """
        # If the waveguide profile is None, then create a uniform waveguide
        if waveguide_profile is None:
            self.__waveguide_profile = self.nominal_parameter * np.ones(shape=self.z.shape)
        else:
            if waveguide_profile.shape != self.z.shape:
                raise ValueError("The shape of the waveguide_profile {s1} is different from the z mesh {s2}".format(
                    s1=waveguide_profile.shape,
                    s2=self.z.shape))
            else:
                self.__waveguide_profile = waveguide_profile
                self.nominal_parameter = self.profile.mean()

        return self.__waveguide_profile

    def create_noisy_waveguide(self, noise_profile: str = "1/f", noise_amplitude: float = 0.2):
        """
        Function to create a noisy waveguide with a predefined noise spectrum.

        :param noise_profile: String identifying the noise profile. Default is *1/f*. Options are AWGN | 1/f | 1/f2
        :type noise_profile: str
        :param noise_amplitude: Range of the noise (in the same units as :py:meth:`~RealisticWaveguide.RealisticWaveguide.create_noisy_waveguide.nominal_parameter`)
        :type noise_amplitude: float
        :return: A numpy.ndarray containing the generated profile of the waveguide.
        """
        # Generate a noise profile using the NoiseFromSpectrum function.
        thisnoise = noise.NoiseFromSpectrum(z=self.z,
                                            offset=self.nominal_parameter,
                                            profile_spectrum=noise_profile,
                                            noise_amplitude=noise_amplitude)
        profile = self.load_waveguide_profile(thisnoise.profile)
        return profile

    def load_poling_structure(self, poling_structure):
        """
        Function to load the poling structure of the waveguide. This function can be used to create custom poling
        structures, such as apodized poling, e.g. `[1] <https://arxiv.org/abs/1410.7714>`_ and
        `[2] <https://arxiv.org/abs/1704.03683>`_
        If the poling structure is set via this function, the poling period of the waveguide is set to +numpy.inftz
        .. warning:: The effectiveness of this function in the calculation of the phasematching spectra is untested.

        :param poling_structure: Array containing the orientation of the poling.
        :type poling_structure: numpy.ndarray
        """
        if poling_structure.shape != self.profile.shape:
            raise ValueError("The poling_structure must have the same shape as the waveguide profile!")
        self.__poling_structure = poling_structure
        self.__poling_period = +np.infty
        return self.__poling_period

    def plot(self, ax: matplotlib.axes.Axes = None):
        """
        Function to plot the waveguide profile.

        :param ax: handle to axis, if you want to plot in specific axes.
        :type ax: matplotlib.axes.Axes
        :return: fig, ax: handle to figure and axis objects
        """
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        else:
            plt.sca(ax)
            fig = plt.gcf()

        ax.plot(self.z, self.profile)
        ax.set_xlabel("z [m]")
        ax.set_ylabel(self.nominal_parameter_name)
        ax.set_title("Waveguide profile")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        return fig, ax

    def plot_waveguide_properties(self, fig: matplotlib.figure.Figure = None):
        """
        Function to plot the waveguide properties in a figure. This function plots the waveguide profile,
        the power spectrum, autocorrelation and histogram distribution of the noise.

        :param fig: Handle of the figure where the plots should be displayed. If *None*, then opens a new figure. Default
        is None.
        :type fig: matplotlib.figure.Figure
        :return:
        """
        if fig is None:
            fig = plt.figure()

        z_autocorr, autocorr, f, power_spectrum = utils.calculate_profile_properties(self.z, self.profile)
        if fig is None:
            fig = plt.figure()
        else:
            fig = plt.figure(fig.number)

        plt.subplot(211)
        ax0 = plt.gca()
        l1, = plt.plot(self.z, self.profile)
        plt.title("Waveguide profile")
        plt.xlabel("z [m]")
        ax0.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.ylabel(self.nominal_parameter_name)

        plt.subplot(234)
        ax1 = plt.gca()
        l2, = plt.semilogy(z_autocorr, abs(autocorr) ** 2, label="Calculated autocorrelation")
        plt.title("|R(z)|^2")
        plt.xlabel("z [m]")
        ax0.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.grid(axis="both")

        plt.subplot(235)
        ax2 = plt.gca()
        l3, = plt.loglog(f, abs(power_spectrum) ** 2)
        plt.grid(axis="both")
        plt.title("|S(f)|^2")
        plt.xlabel("f")

        plt.subplot(236)
        ax3 = plt.gca()
        plt.hist(self.profile, bins=int(np.sqrt(len(self.profile))), density=True)
        plt.title("Distribution of " + self.nominal_parameter_name.lower().split("[")[0])
        plt.xlabel(self.nominal_parameter_name)
        plt.ylabel("Frequencies")
        plt.tight_layout()
        return fig, [ax0, ax1, ax2, ax3], [l1, l2, l3]
