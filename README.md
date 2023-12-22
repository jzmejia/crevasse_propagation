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

A thermo-visco-elastic modoel for hydraulically driven crevasse
propagation through cold ice.  This model was originally created by
Kristin Poinar (Poinar et al., 2017), written in MatLab. The Python
implementation was initially created by jzmejia Aug 2021. This repo
contains the python implementaion of crevprop which includes added 
functionality, project restructuring, and additional/expanded equations.

## Usage

```python
import pandas as pd
from crevprop.iceblock import IceBlock 

# initialize model geometry for a domain with an ice thickness
# of 1000 m and vertical resolution (dz spacing) of 1 m.
creep_df = pd.read_csv('creep_deformation_file_name.csv', 
                        names=['t','z'])  

ib=IceBlock(1000,
            1, 
            dt=1, 
            years_to_run=2,
            thermal_freq=10,
            crev_spacing=30,
            u_surf=200,
            T_profile=df,
            sigmaT0=120e3,
            creep_table=creep_df,
            include_creep=True,
            Qin_annual=5000,
            shear_modulus=0.1e9
           )
```

## Important

This model is currently under development and is not ready for public use.
Module is currently unstable and we can not guarantee the validity of results
or functionality until version 1 release.

## Model Overview

### Model Parameters

Note: This content will be moved to the documentation and removed from 
the `README.md`.

Model parameters to navigate the equations used in `crevasse_propagation` 
within relevant literature (e.g., Poinar et al., 2017; Van der Veen 2007;
Weertman 1964, 1983, 1996).

| parameter                 |               | module         | units |
| ------------------------- | ------------- | -------------- | ----- |
| Ice thickness             | H             | `ice_thickness`| m |
| Crevase spacing           | R             | `crev_spacing` | m  |
| Water depth in crevasse   | w             | `water_depth`  | m  |
| Longitudinal stress       | $\sigma _{y}$ | `sigma_crev`   | kPa  | |
| Shear modulus             | $\mu$   | `shear_modulus`,`mu`| 0.07-3.9 GPa |
| Water flux                | Q             | `Qin`          | m $^{2}$ per timestep |
| Water flux initialization |               | `Qin_annual`   | m $^{2}$ a $^{-1}$ |
| Fracture toughness of ice | $K_{IC}$      | `fracture_toughness` | Pa m $^{0.5}$  |

### Comparison with notation used in the literature

Comparison between Weertman (1964, 1983, 1996), van der veen 2007, and Poinar 2017.

| parameter                      |   1964   |  1983  |   1996    |van der veen| poinar |
| ------------------------------ | :------: | :----: | :-------: | :---------:| :----: |
| ice thickness                  |          |        |           |  H     | H |
| variable depth                 |          |        |     y     |  b     | z |
| crevasse depth                 |          |   L    |     L     |  d     | d |
| depth to water surface         |          |        | - $y_o$   |  a     | w |
| height of water column         |          |        |           |  d-a   |   |
| average tensile stress         |          |   T    |$\sigma _A$|$R_{xx}$|$\sigma$|
| tensile stress $^1$  |$\tau (x)$|$\sigma _{xx}(y)$|$\sigma _T$|$\sigma_{xx}$|$\sigma'_y$|
| compressive hydrostatic stress |          |        |$\sigma_C$ | $L$    | |
| shear modulus                  |          | $\mu$  |     G     |        |$\mu$ |
| constant                       | $\alpha$ |        |$\alpha _i$|        | |
| stress intensity factor        |          |        | $K$       |        | |
| K at crack tip                 |          |        | $K_I$     |        | |
| critical K for ice             |          |        | $K_{gc}$  |        | KIC|
| net Burgers vector             |          |        | $b_T$     |        | |
| crevasse opening displacement  |          |        | $D(y)$    |        |e(d,z)|
| crevasse width                 |          |        |           |        |W(z)|

$^1$ deviatoric stress in the crevasse opening direction (first principal stress)
