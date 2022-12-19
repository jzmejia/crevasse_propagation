# CrevProp: A crevasse propagation model
<!-- docs: passing, read the docs -->
[![Documentation Status](https://readthedocs.org/projects/crevasse-propagation/badge/?version=latest)](https://crevasse-propagation.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/jzmejia/crevasse_propagation)
![GitHub top language](https://img.shields.io/github/languages/top/jzmejia/crevasse_propagation)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/jzmejia/crevasse_propagation)

![GitHub last commit](https://img.shields.io/github/last-commit/jzmejia/crevasse_propagation)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/jzmejia/crevasse_propagation)

[crevprop documentation](https://crevasse-propagation.readthedocs.io/en/latest/)

## Description

A thermo-visco-elastic modoel for hydraulically driven crevasse propagation through cold ice.  This
model was originally created by Kristin Poinar (Poinar et al., 2017), written in MatLab. The Python implementation was initially created by jzmejia Aug 2021. This repo contains the python implementaion of crevprop which includes added functionality, project restructuring, and additional/expanded equations.

## Usage

```python
from crevprop.iceblock import IceBlock 

# initialize model geometry for a domain with an ice thickness of 500 m and vertical resolution of 1 m.
ib = IceBlock(500, 1, dt=0.5, thermal_freq=2)

```


## Important
This model is currently under development and is not ready for public use.  Module is currently unstable and we can not guarantee the validity of results or functionality until version 1 release scheduled for early 2023. 

## Model Overview






### Model Parameters
Note: This content will be moved to the documentation and removed from the README. 

Model parameters to navigate the equations used in `crevasse_propagation` within relevant literature (e.g., Poinar et al., 2017; Van der Veen 2007; Weertman 1964, 1983, 1996).

| parameter                 |               |     | value/units    |
| ------------------------- | ------------- | --- | -------------- |
| Ice thickness             | H             |     | m              |
| Ice surface elevation     | s             |     | m.a.s.l.       |
| Crevase spacing           | R             |     | 50 m           |
| Surface runoff rate       | $\dot b$      |     | 0.5 m a$^{-1}$ |
| Water depth in crevasse   | w             |     | m              |
| Longitudinal stress       | $\sigma _{y}$ |     |                |
| Shear modulus             | $\mu$         | G   | 0.07-3.9 GPa   |
| Water flux                | Q             |     |                |
| Fracture toughness of ice | $K_{IC}$      |     |                |




### Comparison with notation used in the literature

| parameter                       |    1964    |       1983        |    1996     | van der veen  | matlab |   poinar    |
| ------------------------------- | :--------: | :---------------: | :---------: | :-----------: | ------ | :---------: |
| ice thickness                   |            |                   |             |       H       |        |      H      |
| variable depth                  |            |                   |      y      |       b       | Z      |      z      |
| crevasse depth                  |            |         L         |      L      |       d       |        |      d      |
| depth to water surface          |            |                   |   -$y_o$    |       a       | dw     |      w      |
| height of water column          |            |                   |             |      d-a      | b      |             |
| average tensile stress          |            |         T         | $\sigma _A$ |   $R_{xx}$    |        |  $\sigma$   |
| tensile/compressive stress $^1$ | $\tau (x)$ | $\sigma _{xx}(y)$ | $\sigma _T$ | $\sigma_{xx}$ |        | $\sigma'_y$ |
| compressive hydrostatic stress  |            |                   | $\sigma_C$  |      $L$      |        |
| shear modulus                   |            |       $\mu$       |      G      |               |        |    $\mu$    |
| constant                        |  $\alpha$  |                   | $\alpha _i$ |               |        |
| stress intensity factor         |            |                   |     $K$     |               |        |
| K at crack tip                  |            |                   |    $K_I$    |               |        |
| critical K for ice              |            |                   |  $K_{gc}$   |               | KIC    |     KIC     |
| net Burgers vector              |            |                   |    $b_T$    |               |        |
| crevasse opening displacement   |            |                   |   $D(y)$    |               |        |   e(d,z)    |
| crevasse width                  |            |                   |             |               |        |    W(z)     |


$^1$ deviatoric stress in the crevasse opening direction (first principal stress)
