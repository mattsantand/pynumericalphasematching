import numpy as np
import matplotlib.pyplot as plt
from pynumpm import waveguide, phasematching, jsa, utils
from examples import custom_sellmeier


def test_pump():
    length = 5e-3  # length in m

    nte, ntm = custom_sellmeier()
    T0 = 25

    poling_period = utils.calculate_poling_period(1.55e-6, 0, 0.55e-6, nte(T0), ntm(T0),
                                                  nte(T0), 1)
    thiswaveguide = waveguide.Waveguide(length=length,
                                        poling_period=poling_period)
    thisprocess = phasematching.SimplePhasematching2D(waveguide=thiswaveguide,
                                                      n_red=nte(T0),
                                                      n_green=ntm(T0),
                                                      n_blue=nte(T0))

    thisprocess.red_wavelength = np.linspace(1.50e-6, 1.6e-6, 500)
    thisprocess.blue_wavelength = np.linspace(0.549e-6, 0.551e-6, 1000)
    thisprocess.calculate_phasematching()

    # the process is an SFG process
    thispump = jsa.Pump(process=jsa.Process.SFG)
    thispump.wavelength1 = thisprocess.wavelength1
    thispump.wavelength2 = thisprocess.wavelength2
    # set the bandwidth to 1nm
    thispump.pump_width = 1e-9
    thispump.plot()

    # load the pump and the phasematching to calculate the JSA
    thisjsa = jsa.JSA(phasematching=thisprocess,
                      pump=thispump)
    thisjsa.calculate_JSA()
    thisjsa.calculate_schmidt_decomposition()
    thisjsa.plot_schmidt_coefficients(ncoeff=20)
    thisjsa.plot(plot_pump=True)
    plt.show()
