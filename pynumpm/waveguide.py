import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from pynumpm import noise
import logging


class Waveguide(object):
    """
    Base class for the description of a waveguide object. Initialize the object providing its length (in meter) and,
    if necessary, its poling period.

    :param length: Length of the waveguide [*meter*].
    :type length: float
    :param poling_period: Poling period of the structure, [*meter*]. Default: +numpy.infty
    :type poling_period: float

    The following block of code initialises a 10mm-long waveguide with a poling period of 16 :math:`\mu\mathrm{m}`::

        thiswaveguide = waveguide.Waveguide(length = 10e-3,
                                            poling_period = 16e-6)

    """

    def __init__(self, length: float, poling_period: float = +np.infty):
        if not isinstance(length, float):
            raise ValueError("'length' must be a float.")

        if not isinstance(poling_period, float):
            raise ValueError("'length' must be a float.")

        if np.isinf(poling_period):
            warnings.warn("The user has not provided a poling period. The default value of +numpy.infty will be used.",
                          UserWarning)
        self._poling_period = poling_period
        self._length = length
        self._waveguide_profile = None

    def __repr__(self):
        text = f"{self.__class__.__name__} object at {hex(id(self))}.\n\tLength: {self.length} m\n\tpoling period:{self.poling_period_um}um"
        return text

    @property
    def poling_period(self):
        """
        The poling period of the structure, in m.

        """
        return self._poling_period

    @property
    def poling_period_um(self):
        """
        The poling period of the structure, in :math:`\mu\mathrm{m}`.

        """
        return self._poling_period * 1e6

    @property
    def length(self):
        """
        The length of the structure, in m.

        """
        return self._length


class RealisticWaveguide(Waveguide):
    """
    Class for the description of waveguides with variable profiles. It can generate noisy profiles (via the functions in
    the :mod:`pynumpm.noise` module) and it can load user-defined profiles and/or poling structures (they must be
    consistent with the user specified mesh).

    Initialize the waveguide by providing a z-mesh,a poling and the nominal parameter of the profile. This will
    automatically generate a uniform profile with the specified nominal parameter.

    :param z: linearly spaced space mesh [*meter*].
    :type z: numpy.ndarray
    :param poling_period: poling period of the structure [*meter*].
    :type poling_period: float
    :param nominal_parameter: nominal parameter of the structure [variable units depending on the Sellmeier used].
                              Default: None.
    :type nominal_parameter: float
    :param nominal_parameter_name: name of the nominal parameter (used for the axes). LaTeX syntax is allowed.
                                   Default: empty string.
    :type nominal_parameter_name: string

    The following block of code initialises and plots the profile of
    a 15mm-long, 7 :math:`\mu\mathrm{m}`-wide waveguide with a poling period of
    9 :math:`\mu\mathrm{m}`. The waveguide is discretized over a z-mesh with steps of 10 :math:`\mu\mathrm{m}`.
    The waveguide width is characeterised by a noise with a 1/f spectrum with amplitude 0.2 :math:`\mu\mathrm{m}`.
    ::

        z = np.arange(0, 15e-3, 10e-6)
        thiswaveguide = waveguide.RealisticWaveguide(z = z,
                                                     poling_period = 9e-6,
                                                     nominal_parameter = 7e-6,
                                                     nominal_parameter_name = "Width")
        thiswaveguide.create_noisy_waveguide(noise_profile = "1/f",
                                             noise_amplitude = 0.2)
        thiswaveguide.plot_waveguide_properties()

    """

    def __init__(self, z: np.ndarray, poling_period: float = np.infty, nominal_parameter: float = 1.,
                 nominal_parameter_name: str = ""):

        if not isinstance(z, np.ndarray):
            raise TypeError("'z' must be a numpy.ndarray.")
        if len(z.shape) > 2 or (len(z.shape) == 2 and z.shape[1] != 1):
            raise ValueError("z is a {0} array. It needs to be a 1D array instead.".format(z.shape))
        if len(z.shape) == 2:
            # in this case, z.shape = (xxx, 1). We need to reduce it to (xxx,)
            warnings.warn("z has the shape {0}. Reshaping it to ({1},)".format(z.shape, z.shape[0]))
            z = z.reshape(-1, )

        if isinstance(nominal_parameter, int):
            nominal_parameter = float(nominal_parameter)
        if not isinstance(nominal_parameter, float):
            raise TypeError("'nominal_parameter' must be a float")
        if not isinstance(nominal_parameter_name, str):
            raise TypeError("'nominal_parameter_name' must be a string")

        self._z = z
        length = self.z[-1] - self.z[0]

        Waveguide.__init__(self, length, poling_period)
        self._nominal_parameter = nominal_parameter
        # when an object of this class is initialized, the following call creates a uniform waveguide
        self._waveguide_profile = self.load_waveguide_profile()
        self._nominal_parameter_name = nominal_parameter_name
        if self._nominal_parameter_name == "":
            warnings.warn(
                "The name of the variable parameter was left empty. "
                "That's really not a great name...",
                UserWarning)
        self._poling_structure = None

    def __repr__(self):
        text = f"{self.__class__.__name__} object at {hex(id(self))}.\n\tLength: {self.length}m" \
               f"\n\tPoling: {self.poling_period_um} um." \
               f"\n\t{self.nominal_parameter_name}: {self.nominal_parameter}" \
               f"\n\tDiscretization: {self.dz}" \
               f"\n\tPoling structure set: {self.poling_structure_set}"
        return text

    @property
    def z(self):
        """
        Array containing the z-mesh of the structure.

        :return:
        """
        return self._z

    @property
    def dz(self):
        """
        Resolution of the z-mesh.

        :return:
        """
        dz = np.diff(self.z)[0]
        return dz

    @property
    def profile(self):
        """
        Waveguide profile (array).
        Array containing the values of the waveguide profile to be simulated (e.g., the waveguide
        width or temperature profile).

        """
        return self._waveguide_profile

    @property
    def nominal_parameter_name(self):
        """
        Name of the nominal fabrication parameter.

        """
        return self._nominal_parameter_name

    @property
    def poling_structure_set(self):
        """
        Boolean to describe if the poling structure is set.

        """
        return self.poling_structure is not None

    @property
    def poling_structure(self):
        """
        Array containing the poling structure.

        """
        return self._poling_structure

    @property
    def nominal_parameter(self):
        """
        Nominal fabrication parameter of the waveguide.

        """
        return self._nominal_parameter

    @nominal_parameter.setter
    def nominal_parameter(self, value):
        self._nominal_parameter = value

    def load_waveguide_profile(self, waveguide_profile: np.ndarray = None):
        """
        Function to load a user-defined waveguide profile.

        :param waveguide_profile: Array with the waveguide profile, with respect to the parameter under investigation,
                                  or *None*. If *None*, it will create a uniform waveguide with the a constant
                                  `waveguide.nominal_parameter`. If an array, it *must* have the same shape as the
                                  z-mesh. Default: None
        :type waveguide_profile: numpy.ndarray
        :return: The numpy.ndarray containing the waveguide profile.

        """
        # If the waveguide profile is None, then create a uniform waveguide
        if waveguide_profile is None:
            self._waveguide_profile = self.nominal_parameter * np.ones(shape=self.z.shape)
        else:
            if waveguide_profile.shape != self.z.shape:
                raise ValueError("The shape of the waveguide_profile {s1} is different from the z mesh {s2}".format(
                    s1=waveguide_profile.shape,
                    s2=self.z.shape))
            else:
                self._waveguide_profile = waveguide_profile
                self.nominal_parameter = self.profile.mean()

        return self._waveguide_profile

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
        `[2] <https://arxiv.org/abs/1704.03683>`_.
        If the poling structure is set via this function, the poling period of the waveguide is *automatically* set to
        +`numpy.infty`.

        .. warning:: The correct behaviour of this function is untested.

        :param poling_structure: Array containing the orientation of the poling.
        :type poling_structure: numpy.ndarray
        :return: The poling structure

        """
        if poling_structure.shape != self.profile.shape:
            raise ValueError("The poling_structure must have the same shape as the waveguide profile!")
        self._poling_structure = poling_structure
        self._poling_period = +np.infty
        return self._poling_structure

    def plot(self, ax: matplotlib.pyplot.Axes = None):
        """
        Function to plot the waveguide profile. If an axis handle is passed, it will plot in those axes.

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
        plt.tight_layout()
        return fig, ax

    def plot_waveguide_properties(self, fig=None, **plotkwargs):
        """
        Function to plot the waveguide properties in a figure. This function plots the waveguide profile,
        the power spectrum, autocorrelation and histogram distribution of the noise.
        If a Figure handle is passed, it plots in said Figure.

        :param fig: Handle of the figure where the plots should be displayed. If *None*, then opens a new figure.
                    Default is None.
        :type fig: matplotlib.figure.Figure

        """
        logger = logging.getLogger(__name__)
        logger.debug("Plotting noise properties.")
        z_autocorr, autocorr, f, power_spectrum = noise.calculate_profile_properties(self.z, self.profile)
        if fig is None:
            fig = plt.figure()
        else:
            fig = plt.figure(fig.number)
        list_of_axes = fig.get_axes()
        if list_of_axes == []:
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(234)
            ax3 = plt.subplot(235)
            ax4 = plt.subplot(236)
        else:
            if len(list_of_axes) == 4:
                ax1, ax2, ax3, ax4 = list_of_axes
            else:
                raise ConnectionError("The figure does not have the correct number of axes (4).")

        plt.sca(ax1)
        l1, = plt.plot(self.z, self.profile, **plotkwargs)
        plt.title("Noise")
        plt.xlabel("z")
        plt.ylabel("Noise")

        plt.sca(ax2)
        l2, = plt.semilogy(z_autocorr, abs(autocorr) ** 2, label="Calculated autocorrelation",
                           **plotkwargs)
        plt.title("|R(z)|^2")
        plt.xlabel("z")

        plt.sca(ax3)
        l3, = plt.loglog(f, abs(power_spectrum) ** 2, **plotkwargs)
        plt.title("|S(f)|^2")
        plt.xlabel("f")

        plt.sca(ax4)
        plt.hist(self.profile, bins=int(np.sqrt(len(self.profile))), **plotkwargs)
        plt.title("Histogram of the noise.")
        plt.tight_layout()
        return fig, [ax1, ax2, ax3]
