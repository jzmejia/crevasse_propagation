# Crevasse Propagation Model
[![Documentation Status](https://readthedocs.org/projects/crevasse-propagation/badge/?version=latest)](https://crevasse-propagation.readthedocs.io/en/latest/?badge=latest)



## Description
thermo-visco-elastic modoel for hydraulically driven crevasse propagation through cold ice.  
model created by K. Poinar (Poinar et al., 2017)  
python implementation by jzmejia Aug 2021 (start)


## Important
This model is currently under construction and is not ready for public use.  We can not guarantee the validity of any model outputs at this time.





## Model Overview


### Model Parameters  

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




### Notation used in the literature

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
