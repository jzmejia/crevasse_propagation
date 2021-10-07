# crevasse_propagation


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

| parameter                     |    1964    |       1983        |    1996     | van der veen  | matlab |
| ----------------------------- | :--------: | :---------------: | :---------: | :-----------: | ------ |
| ice thickness                 |            |                   |             |       H       |        |
| variable depth                |            |                   |      y      |       b       | Z      |
| crevasse depth                |            |         L         |      L      |       d       |        |
| depth to water surface        |            |                   |   -$y_o$    |       a       | dw     |
| height of water column        |            |                   |             |      d-a      | b      |
| **stress**                    |            |                   |             |               |        |
| average tensile stress        |            |         T         | $\sigma _A$ |   $R_{xx}$    |        |
| tensile or compressive stress | $\tau (x)$ | $\sigma _{xx}(y)$ | $\sigma _T$ | $\sigma_{xx}$ |        |
| compressive hydrostatic       |            |                   | $\sigma_C$  |      $L$      |        |
| shear modulus                 |            |       $\mu$       |      G      |               |        |
| constant                      |  $\alpha$  |                   | $\alpha _i$ |               |        |
| stress intensity factor       |            |                   |     $K$     |               |        |
| K at crack tip                |            |                   |    $K_I$    |               |        |
| critical K for ice            |            |                   |  $K_{gc}$   |               | KIC    |
| net Burgers vector            |            |                   |    $b_T$    |               |        |
| crevasse opening displacement |            |                   |   $D(y)$    |               |        |
