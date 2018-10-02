# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from collections import OrderedDict
import Noise


class Waveguide(object):
    def __init__(self, **kwargs):
        """
        Initialization of the waveguide. Inputs:

        * "length": waveguide length, in meter [default: 16mm]
        * "poling": poling period, in micron [default: infinite (for bulk)]. Read below for more info.
        * "dz": mesh discretization discretizing the propagation axis, in meters [default: poling period]

        In this function are calculated:

        * "z": 1D array of the points in which the waveguide is discretized. If *poling=np.inf*, then the number of
points is the value stored in *z_discretization*. If *poling* is a non infinite number, then *z_discretization* is
redefined to take into account the poling unit.
        * "dz": distance between two points of the mesh of the waveguide.

        :param kwargs:
        """
        if kwargs.has_key("z"):
            self.z = kwargs.get("z")
        else:
            self.length = kwargs.get("length", 16e-3)
            self.dz = abs(kwargs.get("dz", 5e-5))
            self.z = np.arange(0, self.length + self.dz, self.dz)
        self.poling = kwargs.get("poling", np.inf) * 1e-6
        self.length = self.z[-1]
        self.waveguide_model = OrderedDict()
        self.waveguide_model["length"] = round(self.length * 1e3, 3)
        self.poling_structure_set = False
        # dictionary used for plotting
        self.dictu = {"width": "um",
                      "temperature": "*C"}

    @property
    def profile(self):
        return self.f_profile(self.z)

    def evaluate_profile(self, **kwargs):
        """
        This function returns the parameter - dependent on the position y along the waveguide -
        that influences the phasematching properties.
        Suppose that n = n(eps). This function returns eps = f(z).

        Inputs:

        :param parameter: [**width**/temperature]: Describes the variable that influences the variation in n_effective.
        :type parameter: str
        :param profile: [**uniform** / even power / gaussian noise / custom]: Type of profile
        :type profile: str
        :param nominal_parameter: Nominal parameter value for the parameter. (3.0 if width, 1080 if temperature)
        :type nominal_parameter: float

        Additional parameters:

            * If "profile" == "even power"

                    :param n_points: Number of points were to evaluate the even polynomial [default: 10]
                    :param d_parameter:  overall variation of the nominal parameter along the length of the sample [default: nominal_parameter * 0.01]
                    :param order: order of the even polynomial [default: 2]


            * If "profile" == "gaussian noise"
                    :param n_points: Number of points were to evaluate the even polynomial [default: 10]
                    :param offset: Offset of the nominal parameter [default: 0].
                    :param relative_noise:  Relative noise [default: 0.005] . Used to compute the sigma of the gaussian noise (sigma = nominal_parameter*relative_noise)


            * If "profile" == "custom"
                    :param data: Dataset to be interpolated
                    :param method: [fit/interpolation/none]
                    :param kind: parameter if method is "interpolation" [**cubic** / linear]

        :return:
        """

        parameter = kwargs.get("parameter", "width")  # describes the variable that can be addressed. At the moment,
        # is width and temperature

        self.profile_type = kwargs.get("profile", "uniform")  # Type of model

        # Depending on the parameter that varies, initialize differently the nominal parameter
        if parameter.lower() == "width":
            default_nominal_parameter = 7.0
        elif parameter.lower() == "temperature":
            default_nominal_parameter = 1080.
        else:
            default_nominal_parameter = None

        self.nominal_parameter = kwargs.get("nominal_parameter", default_nominal_parameter)

        self.waveguide_model["type of profile"] = self.profile_type
        self.waveguide_model["variable parameter"] = parameter
        method = kwargs.get("method", "fit")
        n_points = kwargs.get("n_points", 10)
        fitting_degree = kwargs.get("fitting_degree", int(n_points / 2))

        # Depending on the type of profile specified, interpolate the relative f_profile function to calculate
        # param = f_profile(z)
        if self.profile_type.lower() == "uniform":
            print "Generating uniform waveguide"
            self.sample_z = [0, self.length]
            self.sample_parameter = [self.nominal_parameter, self.nominal_parameter]
            self.f_profile = interp.interp1d(self.sample_z, self.sample_parameter, kind="linear")

        elif self.profile_type.lower() == "even power":
            print "Generating 'even power' waveguide"
            delta_param = kwargs.get("d_parameter", self.nominal_parameter * 0.01)
            order = kwargs.get("order", 2)
            self.waveguide_model["order"] = order
            self.waveguide_model["variation"] = delta_param
            self.sample_z = np.linspace(0, self.length, n_points)
            self.sample_parameter = self.nominal_parameter + delta_param * np.linspace(-1, 1, n_points) ** order
            self.f_profile = interp.interp1d(self.sample_z, self.sample_parameter, kind="cubic")

        elif self.profile_type.lower() == "gaussian noise":
            print "Generating waveguide with gaussian noise"
            rel_noise = kwargs.get("relative_noise", 0.005)  # relative amount of white noise on top of the wg width
            offset = kwargs.get("offset", 0.)
            self.waveguide_model["relative noise"] = rel_noise
            self.sample_z = np.linspace(0, self.length, n_points)
            self.sample_parameter = np.random.normal(self.nominal_parameter + offset,
                                                     self.nominal_parameter * rel_noise,
                                                     n_points)
            if method == "fit":
                print "Fitting (degree:" + str(fitting_degree) + ")"
                self.f_profile = lambda z: np.polyval(
                    np.polyfit(self.sample_z, self.sample_parameter, deg=fitting_degree), z)
            else:
                print "Interpolating"
                interpkind = kwargs.get("kind", "linear")
                print "Kind ", interpkind
                self.f_profile = interp.interp1d(self.sample_z, self.sample_parameter, kind=interpkind)


        elif self.profile_type.lower() == "custom":
            # Custom model: you must provide data for it
            print "Generating waveguide from custom data"
            self.sample_parameter = kwargs.get("data")
            self.sample_z = kwargs.get("z", np.linspace(0, self.length, len(self.sample_parameter)))
            # check that self.sample_parameter is a valid array (or list of number)
            if type(self.sample_parameter) is not np.ndarray:
                if type(self.sample_parameter) is list and type(self.sample_parameter[0]) is (float or int):
                    self.sample_parameter = np.array(self.sample_parameter)
                else:
                    print "I have no idea what you passed me"
                    return None

            # if the nominal parameter is not defined, then assign it to the average value of the data
            if self.nominal_parameter is None:
                self.nominal_parameter = self.sample_parameter.mean()

            self.waveguide_model["data"] = self.sample_parameter

            if method == "fit":
                print "Fitting (degree:", fitting_degree, ")"
                self.f_profile = lambda z: np.polyval(
                    np.polyfit(self.sample_z, self.sample_parameter, deg=fitting_degree), z)
            elif method == "interpolating":
                print "Interpolating"
                interpkind = kwargs.get("kind", "linear")
                self.f_profile = interp.interp1d(self.sample_z, self.sample_parameter, kind=interpkind)
            elif method == "none":
                self.f_profile = lambda z: self.sample_parameter[np.where(self.z == z)]
            else:
                raise NotImplementedError("Method {0} currently not implemented".format(method))
        else:
            print "Unknown model"

        return self.sample_z, self.sample_parameter, self.f_profile

    def add_standard_noise(self, variable_parameter="width", nominal_parameter=7.0, noise_type="1/f",
                           noise_parameter=0):
        thisnoise = Noise.NoiseFromSpectrum(z=self.z,
                                            offset=nominal_parameter,
                                            profile_spectrum=noise_type,
                                            noise_amplitude=noise_parameter)
        self.evaluate_profile(parameter=variable_parameter,
                              profile="custom",
                              data=thisnoise.profile,
                              method="none")

    def plot_profile(self, ax=None):
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

        ax.plot(self.z * 1e3, self.f_profile(self.z))
        ax.set_xlabel("Length [mm]")
        ylabel = self.waveguide_model["variable parameter"] + " waveguide profile [" + \
                 self.dictu[self.waveguide_model["variable parameter"]] + "]"
        ax.set_ylabel(ylabel)
        ax.set_title("Waveguide profile")
        return fig, ax

    def load_poling_profile(self, poling_structure):
        """
        Function to load the poling profile, if you need to calculate directly the effect of the poling structure.

        :param poling_structure: array of +1 and -1 describing the poling structure. It MUST have the same length (-1) of self.z
        :return:
        """
        if self.poling < 1e3:
            raise ImportWarning(
                "You are setting a poling structure even if you have specified the poling period. Are you sure of what you are doing?")
        if len(poling_structure) != len(self.z):
            raise ImportError(
                "I cannot assign the poling structure to the selected mesh. Mesh length: {ml}, poling structure length: {pl}".format(
                    ml=(len(self.z)), pl=len(poling_structure)))
        elif np.any(abs(poling_structure) != 1):
            raise ValueError("The poling structure has to have only +1 and -1")
        else:
            self.poling_structure = poling_structure
        self.poling_structure_set = True
        return self.poling_structure_set

    def plot_poling(self, ax=None):
        """
        Function to plot the poling profile.

        :param ax: handle to axis, if you want to plot in specific axes.
        :return: fig, ax: handle to figure and axis objects
        """
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        else:
            plt.sca(ax)
            fig = plt.gcf()

        ax.plot(self.z * 1e3, self.poling_structure)
        ax.set_xlabel("Position [mm]")
        ax.set_ylabel("Poling orientation")
        ax.set_title("Poling profile")
        return fig, ax


def example():
    z = np.linspace(0, 0.020, 10000)
    thiswaveguide = Waveguide(z=z, nominal_parameter=7., variable_parameter="Width")
    thiswaveguide.createNoisyWaveguide()
    thiswaveguide.plot_profile()
    plt.show()


if __name__ == '__main__':
    example()
