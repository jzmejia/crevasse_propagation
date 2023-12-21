.. _overview:

==============
Model Overview
==============

Parameters
**********




Notation Comparison
*******************


+---------------------------------+-------------------+--------------------+--------------------+
| parameter                       | Weertman  [#f1]_  |van der Veen [#f2]_ | Poinar [#f3]_      |
+=================================+===================+====================+====================+
| ice thickness                   |                   | H                  | H                  |
+---------------------------------+-------------------+--------------------+--------------------+
| depth                           | y                 | b                  | z                  |
+---------------------------------+-------------------+--------------------+--------------------+
| crevasse depth                  | L                 | d                  | d                  |
+---------------------------------+-------------------+--------------------+--------------------+
| depth to water surface          | :math:`-y_0`      | a                  | w                  |
+---------------------------------+-------------------+--------------------+--------------------+
| average tensile stress          | :math:`\sigma_A`  | :math:`R_{xx}`     | :math:`\sigma`     |
+---------------------------------+-------------------+--------------------+--------------------+
| tensile stress                  | :math:`\sigma_T`  | :math:`\sigma_{xx}`| :math:`\sigma_y`'  |
+---------------------------------+-------------------+--------------------+--------------------+
| compressive hydrostatic stress  | :math:`\sigma_C`  | L                  |                    |
+---------------------------------+-------------------+--------------------+--------------------+
| shear modulus                   | G                 |                    | :math:`\mu`        |
+---------------------------------+-------------------+--------------------+--------------------+
| crevasse opening displacement   | D(y)              |                    | e(d,z)             |
+---------------------------------+-------------------+--------------------+--------------------+






.. [#f1] Weertman (1996). Dislocation Based Fracture Mechanics
.. [#f2] van der Veen (1998). Fracture mechanics approach to penetration of surface creasses on glaciers
.. [#f3] Poinar et al., (2017). Drainage of southeast Greenland firn aquifer water through crevasses to the bed.   

.. ### Comparison with notation used in the literature

.. | parameter                       |    1964    |       1983        |    1996     | van der veen  | matlab |   poinar    |
.. | ------------------------------- | :--------: | :---------------: | :---------: | :-----------: | ------ | :---------: |
.. | ice thickness                   |            |                   |             |       H       |        |      H      |
.. | variable depth                  |            |                   |      y      |       b       | Z      |      z      |
.. | crevasse depth                  |            |         L         |      L      |       d       |        |      d      |
.. | depth to water surface          |            |                   |   -$y_o$    |       a       | dw     |      w      |
.. | height of water column          |            |                   |             |      d-a      | b      |             |
.. | average tensile stress          |            |         T         | $\sigma _A$ |   $R_{xx}$    |        |  $\sigma$   |
.. | tensile/compressive stress $^1$ | $\tau (x)$ | $\sigma _{xx}(y)$ | $\sigma _T$ | $\sigma_{xx}$ |        | $\sigma'_y$ |
.. | compressive hydrostatic stress  |            |                   | $\sigma_C$  |      $L$      |        |
.. | shear modulus                   |            |       $\mu$       |      G      |               |        |    $\mu$    |
.. | constant                        |  $\alpha$  |                   | $\alpha _i$ |               |        |
.. | stress intensity factor         |            |                   |     $K$     |               |        |
.. | K at crack tip                  |            |                   |    $K_I$    |               |        |
.. | critical K for ice              |            |                   |  $K_{gc}$   |               | KIC    |     KIC     |
.. | net Burgers vector              |            |                   |    $b_T$    |               |        |
.. | crevasse opening displacement   |            |                   |   $D(y)$    |               |        |   e(d,z)    |
.. | crevasse width                  |            |                   |             |               |        |    W(z)     |


.. $^1$ deviatoric stress in the crevasse opening direction (first principal stress)
