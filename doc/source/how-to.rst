======
How-to
======

1. Defining a waveguide
====================

In :mod:`pynumpm.waveguide` two types of **waveguide** classes are provided:

* :class:`pynumpm.waveguide.Waveguide`, for the definition of an ideal waveguide;
* :class:`pynumpm.waveguide.RealisticWaveguide`, for the definition of a realistic waveguide.

Ideal waveguide
---------------
To define an ideal waveguide, only its `length` and, if necessary, its poling period `poling_period` are required.
The next block of code creates a 10mm-long, ideal waveguide with poling period :math:`\Lambda = 16\mu\mathrm{m}`.
::

    from pynumpm.waveguide import Waveguide

    idealwaveguide = Waveguide(length = 10e-3,
                               poling_period = 16e-6)



Inhomogeneous waveguide
-----------------------

Adding noise
************

Custom poling structure
***********************



2. Spectrum of an ideal waveguide
==============================

:math:`\Delta\beta` dependent
-----------------------------

Wavelength dependent: 1D
------------------------

Wavelength dependent: 2D
------------------------


3. Spectrum of an inhomogeneous waveguide
======================================

:math:`\Delta\beta` dependent
-----------------------------

Wavelength dependent: 1D
------------------------

Wavelength dependent: 2D
------------------------


4. Definition of a pump spectrum
=============================

5. JSA calculations
================


6. Utilities
=========

Poling period calculation
-------------------------

Phasematching point calculation
-------------------------------

Bandwidth calculation
---------------------
