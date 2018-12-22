Phasematching
*************

With this module you can easily calculate the phasematching of a non-ideal waveguide.

Classes
=======

.. automodule:: pynumpm.phasematching
   :members:

Example of usage
================

>>> thisprocess = phasematching.Phasematching2D(waveguide=thiswaveguide,
                                            n_red=n_effective,
                                            n_green=n_effective,
                                            n_blue=n_effective)
>>>wl_signal = np.linspace(1.540, 1.560, 1000) * 1e-6
>>>wl_idler = np.linspace(1.2, 1.4, 1000)*1e-6
>>>thisprocess.red_wavelength = wl_signal
>>>thisprocess.green_wavelength = wl_idler
>>>thisprocess.set_nonlinearity_profile(profile_type="constant",
                                     first_order_coefficient=False)
>>>thisprocess.calculate_phasematching()
>>>thisprocess.plot()