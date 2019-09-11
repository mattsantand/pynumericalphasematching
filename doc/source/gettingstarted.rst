===============
Getting started
===============

Background
==========

This API has been developed to solve the task of calculating the (one- or two-dimensional) phasematching spectrum of
a wide variety of `nonlinear optical processes <https://en.wikipedia.org/wiki/Nonlinear_optics#Frequency-mixing_processes>`_ processes.
The package is best suited for the analysis of three-wave mixing phenomena; however, some of its tools can be used to
study also higher-order optical nonlinear processes (four-wave mixing, etc.).

This API consists of *four* conceptually distinct modules:

* the module :mod:`pynumpm.waveguide` is used to define the guiding structure under investigation. In here, one can define two
types of guiding structures: *ideal* or *real*. An ideal structure is modelled as homogeneous along its **z** optical axis,
whereas a *real* structure allows the definition of a profile describing the variation of its properties along the
optical **z** axis;
* the module :mod:`pynumpm.noise` is mostly used by the :mod:`pynumpm.waveguide` to model noisy structures. However, it
can be accessed by the user, that can use its function to re-use the predefined noise profiles in case the basic functionalities
provided in the :mod:`pynumpm.waveguide` are not enough;
* the module :mod:`pynumpm.phasematching` is the core module of the API. It provides different types of functions for
the calculation of phasematching spectra;
* the module :mod:`pynumpm.jsa` provides classes that can be used to simulate the behaviour of the simulated phasematching when
pumped with a pulsed light source. No quantum effects are taken into consideration and no time evolution has been taken into
account. Therefore, this module works under the assumption of low nonlinear efficiency.


Installation
============

This package has been developed for Python3, but should work for Python2 as well. Support for Python2 is not provided.

It can be downloaded or cloned from its `GitHub page <https://github.com/mattsantand/pynumericalphasematching>`_.
To install the package, simply run

::

    python setup.py install

in the folder *pynumericalphasematching*.

As soon as possible, the package will be uploaded on PyPI.