import numpy as np
import matplotlib.pyplot as plt
import warnings
from pynumpm import noise, utils
import logging

LOOKUPTABLE = {0: " [m]",
               2: " [cm]",
               3: " [mm]",
               6: r" [$\mu$m]",
               9: " [nm]"}


class SimpleWaveguide(object):
    def __init__(self, z=None, poling_period=None):
        if z is None:
            raise ValueError("Please, provide the waveguide mesh 'z'.")
        if poling_period is None:
            raise ValueError(
                "Please, provide the poling period of the structure ('poling_period'). If no poling is required, set poling_period = +np.infty")
        self.__z = z
        self.__poling_period = poling_period
        self.__poling_structure_set = False
        self.__length = self.z[-1] - self.z[0]

    @property
    def z(self):
        return self.__z

    @property
    def poling_period(self):
        return self.__poling_period

    @property
    def poling_period_um(self):
        return self.__poling_period * 1e6

    @property
    def poling_structure_set(self):
        """Boolean to describe if the poling structure is set."""
        return self.__poling_structure_set

    @property
    def length(self):
        return self.__length


class Waveguide(object):
    """
    Waveguide class.

    It is used to describe waveguide profiles. It can generate noisy profiles (via the functions in the :mod:`noise`
    module), it can load user-defined profiles (they must be consistent with the user specified mesh).
    Moreover, the user can specify a poling structure (functionality unused in the CalculatePhasematching at the
    moment).

    """

    def __init__(self, z: np.ndarray, poling_period: float = None, nominal_parameter: float = 1.,
                 nominal_parameter_name: str = "parameter"):
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
        if poling_period is None:
            raise ValueError(
                "Please, provide the poling period of the structure ('poling_period'). If no poling is required, set poling_period = +np.infty")
        self.__z = z
        self.__dz = np.diff(self.z)[0]
        self.__length = self.z[-1] - self.z[0]
        self.__poling_period = poling_period
        self.__nominal_parameter = nominal_parameter
        self.__waveguide_profile = self.load_waveguide_profile()
        self.__nominal_parameter_name = nominal_parameter_name
        if self.__nominal_parameter_name == "parameter":
            warnings.warn(
                "The name of the variable parameter was left empty. It will be called 'parameter', but that's really not a great name",
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
        # return y

    def load_waveguide_profile(self, waveguide_profile=None):
        """
        Function to load a user-defined waveguide profile.

        :param waveguide_profile: array with the waveguide profile. It *must* have the same shape as the z-mesh
        :type waveguide_profile: numpy.ndarray
        """
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
        else:
            if self.nominal_parameter != nominal_parameter:
                warnings.warn(
                    "Attention. The nominal parameter set to create the noisy profile is different from the one set for the ideal structure. I will overwrite the ideal one.")
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

    def plot(self, ax=None, set_multiplier_x=1):
        """
        Function to plot the waveguide profile.

        :param ax: handle to axis, if you want to plot in specific axes.
        :param set_multiplier_x: set a scaling factor for the x axis. Use a power of 10. Default to 1.
        :return: fig, ax: handle to figure and axis objects
        """
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        else:
            plt.sca(ax)
            fig = plt.gcf()

        exponent = int(np.log10(set_multiplier_x))
        set_multiplier_x = 10 ** exponent  # round the set_multiplier to be a power of 10.

        ax.plot(self.z * set_multiplier_x, self.profile)
        ax.set_xlabel("z" + LOOKUPTABLE[exponent])
        ax.set_ylabel(self.nominal_parameter_name)
        ax.set_title("Waveguide profile")
        return fig, ax

    def plot_waveguide_properties(self, fig=None, set_multiplier_x=1):
        """
        Function to plot the waveguide properties in a figure. This function plots the waveguide profile,
        it's spectrum and .....

        :param fig: Handle of the figure where the plots should be displayed. If None, then opens a new figure. Default
        is None.
        :param set_multiplier_x: set a scaling factor for the x axis. Use a power of 10. Default to 1.
        :return:
        """
        if fig is None:
            fig = plt.figure()

        exponent = int(np.log10(set_multiplier_x))
        set_multiplier_x = 10 ** exponent  # round the set_multiplier to be a power of 10.
        z_autocorr, autocorr, f, power_spectrum = utils.calculate_profile_properties(self.z, self.profile)
        if fig is None:
            fig = plt.figure()
        else:
            fig = plt.figure(fig.number)

        plt.subplot(211)
        ax0 = plt.gca()
        l1, = plt.plot(self.z * set_multiplier_x, self.profile)
        plt.title("Waveguide profile")
        plt.xlabel("z" + LOOKUPTABLE[exponent])
        plt.ylabel(self.nominal_parameter_name)

        plt.subplot(234)
        ax1 = plt.gca()
        l2, = plt.semilogy(z_autocorr, abs(autocorr) ** 2, label="Calculated autocorrelation")
        plt.title("|R(z)|^2")
        plt.xlabel("z")

        plt.subplot(235)
        ax2 = plt.gca()
        l3, = plt.loglog(f, abs(power_spectrum) ** 2)
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
