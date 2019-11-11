==========
References
==========

Theoretical background
----------------------
The phasematching spectrum of a nonlinear optical process in an *ideal* guiding structure (waveguide, fibre, etc...) of
length :math:`L` can be calculated using the equation

.. math::

    \phi &= \frac{1}{L}\int_0^L \exp\left\lbrace \mathrm{i}\Delta\beta z\right\rbrace dz\\
    &= sinc\left(\frac{\Delta\beta L}{2}\right)\exp\left\lbrace \mathrm{i}\frac{\Delta\beta L}{2}\right\rbrace,


where :math:`\Delta\beta` is the phase mismatch of the process, :math:`z` is the optical axis of the system and
:math:`sinc(x) = sin(x)/x`. For example, in a three-wave mixing process, the :math:`\Delta\beta` is usually given by

.. math::

    \Delta\beta & = \beta_3 - \beta_2 - \beta_1\\
                & = 2\pi \left(\frac{n_3}{\lambda_3} - \frac{n_2}{\lambda_2} - \frac{n_1}{\lambda_1}\right),

being :math:`\beta_i` the wavevector of the :math:`i^{th}` field.

If, however, the structure is not ideal and its properties change along the propagation axis *z*, the phase mismatch
:math:`\Delta\beta` may vary along the guiding structure, i.e. :math:`\Delta\beta = \Delta\beta(z)`.
In this case, the phasematching spectrum is given by

.. math::

    \phi = \frac{1}{L}\int_0^L \exp\left\lbrace \mathrm{i} \int_0^z\Delta\beta(\xi)d\xi \right\rbrace dz.

The integral :math:`\int_0^z\Delta\beta(\xi)d\xi` is necessary to correctly keep track of the phase acquired by the
fields as they travel along an inhomogeneous medium.

While this API can be used to calculate the spectrum of an ideal waveguide, this task is quite trivial, since an analytic
expression exist (see first integral).
This API was developed to calculate the phasematching of an inhomogeneous structure for a given profile :math:`\Delta\beta(z)`
or, equivalently, if the variation of the refractive index :math:`n(z)` along the guiding structure is known.

Useful literature
-----------------

* `Helmfrid et al., Influence of randomly varying domain lengths and nonuniform effective index on second-harmonic generation in quasi-phase-matching waveguides, JOSAB, 1991 <https://www.osapublishing.org/josab/abstract.cfm?uri=josab-8-4-797>`_
* `Francis-Jones et al., Characterisation of longitudinal variation in photonic crystal fibre, Opt. Expr., 2016 <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-24-22-24836>`_

Publications
------------
The code developed in this package has been used to write the following scientific papers:

* `M. Santandrea et al., Fabrication limits of waveguides in nonlinear crystals and their impact on quantum optics applications, 2019 New J. Phys. 21 033038 <https://iopscience.iop.org/article/10.1088/1367-2630/aaff13>`_
* `M. Santandrea et al., General framework for the analysis of imperfections in nonlinear systems, arXiv:1906.09857 <https://arxiv.org/abs/1906.09857>`_
* `M. Santandrea et al., Characterisation of Inhomogeneities in Ti:LiNbO3 waveguides, arXiv:1906.10018  <https://arxiv.org/abs/1906.10018>`_

