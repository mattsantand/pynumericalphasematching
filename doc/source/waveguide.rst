Waveguide
*********

With this module you can easily describe waveguide profiles to be fed in the :mod:`pynumpm.phasematching` module.

Classes
=======

.. automodule:: pynumpm.waveguide
   :members:

Example of usage
================

Define the z-mesh (in meters). The waveguide will have the same length as the mesh.

>>> z = np.linspace(0, 0.020, 10000)

Create a waveguide on the mesh. In this case, the code will generate a waveguide with nominal parameter *7*. The name of
the parameter is set to "Width :math:`[\\mum]`" to enable LaTeX formatting of the y-axis label.

>>> thiswaveguide = Waveguide(z=z, nominal_parameter=7., nominal_parameter_name=r"Width [$\mu$m]")

Create a 1/f noise on top of the generated waveguide

>>> thiswaveguide.create_noisy_waveguide()

Plot the waveguide profile

>>> thiswaveguide.plot()

