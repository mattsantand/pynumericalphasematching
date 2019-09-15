`Waveguide` module
******************

With this module you can easily describe waveguide profiles to be fed in the :mod:`pynumpm.phasematching` module.

Two classes are available:

* :class:`pynumpm.waveguide.Waveguide`, for the simulation of simple, ideal system with translational invariance;
* :class:`pynumpm.waveguide.RealisticWaveguide`, for the simulation of systems with a parameter that varies along their
  propagation axis.

The objects of this class are printable (the have a __repr__ method).

Waveguide
---------

.. autoclass:: pynumpm.waveguide.Waveguide
   :members:

RealisticWaveguide
------------------
.. autoclass:: pynumpm.waveguide.RealisticWaveguide
   :members:

..
   .. automodule:: pynumpm.waveguide
      :members:
