"""
The main container for the crevasse propagation model, holding and 
initializing model geometry
"""
import numpy as np
from numpy.lib.function_base import diff
from numpy import sqrt, abs
from numpy.polynomial import Polynomial as P
from scipy.constants import g, pi

import math as math
# import pandas as pd
from typing import (
    Dict,
    List,
    Any
)

from .temperature_field import ThermalModel
from . import physical_constants as pc

from .crevasse import Crevasse


class Ice(object):
    """Material properties of ice

    Parameters
    ----------


    Attributes
    ----------
    density:
    T: float, int
        ice temperature in degrees Celcius 
    """

    def __init__(self,
                 density=917.,
                 temperature=0,
                 fracture_toughness=10e3):
        """_summary_

        Parameters
        ----------
        density : int, optional
            ice denisty in kg/m^3, by default 917 kg/m^3
        temperature : int, optional
            ice temperature in degrees Celsius, by default 0C
        fracture_toughness : float, optional
            fracture toughness of ice, by default 10e3
        """

        self.density = density
        self.T = temperature
        self.T_kelvin = self.C_to_K(temperature)

        self.specific_heat_capacity = self.calc_specific_heat_capacity()
        self.heat_capacity_intercept = 2115.3
        self.heat_capacity_slope = 7.79293

        # note: thermal conductivity at 0 deg C unit W/mK
        self.thermal_conductivity = self.ki = 2.1

        self.latient_heat_of_freezing = self.Lf = 3.35e5
        self.kappa = self.thermal_diffusivity()
        self.units = self._set_unit()

        self.fracture_toughness = fracture_toughness

    def C_to_K(self, C):
        """Convert temperature in degrees Celsius to Kelvin"""
        return C + 273.15

    def calc_specific_heat_capacity(self, T):
        """specific heat capacity for pure ice (J/kg/K)

        Specific heat capacity, c, per unit mass of ice in SI units. 
        Note: c of dry snow and ice does not vary with density 
        because the heat needed to warm the air and vapor between 
        grains is neglibible. (see Cuffey, ch 9, pp 400)

        c = 152.5 + 7.122(T)

        Parameters
        ----------
        T: float
            ice temperature in degrees Celcius

        Returns
        -------
        c: float 
            specific heat capacity of ice in Jkg^-1K^-1

        """
        return 152.5 + 7.122 * self.C_to_K(T)

    def thermal_conductivity_pure_ice(self, T=0):
        return 9.828 * np.exp(-5.7e-3 * self.C_to_K(T))
    
    def van_dusen(density):
        """Depth dependant thermal conductivity for dry snow, firn, and ice
        Van Dusen 1929

        This equation typically gives a lower limit in most cases

        Parameters
        ----------
        density : (float)
            density of dry snow, firn, or glacier ice in kg/m^3

        """
        return 2.1e-2 + 4.2e-4 * density + 2.2e-9 * density**3
    
    def schwerdtfeger(self, density):
        # density must be less than the density of pure ice, find threshold to use here
        pure_ice = self.thermal_conductivity_pure_ice
        pass
    
    def thermal_conductivitiy_firn(self, x, relationship="density"):
        """calculate thermal conductivity of firn
        
        This function implements the depth or density dependant
        empirical relationships described by Oster and Albert (2022).
        
        Use the density relation for depths from 0-48 m. When the
        the density of pure ice is entered into this equation a thermal
        conductivity `k_{firn}(p_ice)=2.4` W/mK which is the known 
        thermal conductivity of pure ice at -25 deg C.
        
        The depth dependant relationship predicts the thermal 
        conductivity of pure ice for depths around 100-110 m. This is
        consistant with the field-measured depth of the firn-ice
        transitions. 

        Parameters
        ----------
        x : float
            density or depth used in calculation. Must correspond to 
            choice of relationship. 
        relationship : str, optional
            must be "density" or "depth", by default "density"
        """
        # function of density vs function of depth
        if relationship == "density":
            k_firn = 0.144 * np.exp(0.00308 * x)
        elif relationship == "depth":
            k_firn = 0.536 * np.exp(0.0144 * x)
        
    
    def calc_thermal_conductivity(self, T=0, density=917, method="empirical"):
        """calc thermal conductivy using specified method"""
        if method == "van_dusen":
            kt = self.van_dusen(density)
        elif method == "schwerdtfeger":
            kt = 1
        elif method == "empirical":
            kt = self.thermal_conductivity_pure_ice(T)

        return kt
    
    

    def thermal_diffusivity(self):
        return self.thermal_conductivity / self.density / self.heat_capacity

    def _set_unit(self):
        units = {
            'density': 'kg/m^3',
            'thermal conductivity': 'J/m/K/s',
            'thermal diffusivity': 'm^2/s',
            'latient heat of freezing': 'J/kg',
            'heat capacity': 'J/kg/K',
            'melting point at 1 atm': 'K',
            'fracture toughness': 'MPa m^-1/2',
            'driving stress': 'kPa'
        }
        return units


class IceBlock(Ice):
    """
    Two-dimensional ice block geometry defining model domain.

    The model's geometry

    Attributes
    ----------
    ice_thickness : float, int
        thickness of ice block in meters
    dx : float, int
        horizontal (x-coordinate) sampling within ice block (m)
        determined by diffusion lengthscale 
    dz : float, int
        vertical sampling resolution within ice block (m)
    x : np.array
        x-coordinates of model domain ranging from 0 at downstream 
        boundaryto -length at upstream boundary, with spacing of dx. 
        Unit of meters. This array defines the x coordinates of the 
        iceblock overwhich to run the model. 
    z : np.array
        z-coordinates (verical) of model domain ranging from 0 at the 
        ice surface to the ice thickness at the base of the ice sheet, 
        spacing of dz. Unit of meters. This array defines the z 
        coordinates of the iceblock overwhich to run the crevasse 
        propagation model. Note that the thermal model will run with a 
        minimum vertical resolution of 5m to reduce computational costs 
        (dz>=5m for thermal model). 
    length : int, float
        length of horizontal component of model domain in meters. 
    dt : float, int
        Timestep in seconds to run crevasse model (seconds)
    dt_T : float, int
        timestep in seconds to run thermal model (seconds)
    crev_spacing : float, int
        Spacing between crevasses in crevasse field (m)
    crev_count : int
        Number of crevasses within model domain
    crev_locs : List[Tuple]
        positional information corresponding to crevasses within 
        crevasse field. Tuple entries contain the (x-coordinate, depth) 
        of a crevasse, with the number of list entries equal to the 
        current crevasse count. Values are used by ``ThermalModel`` when
        solving for ice temperature and refreezing. 
    temperature : ThermalModel
        an instance of ThermalModel populated with model geometry and 
        initial conditions specified when calling ``__init__``
    u_surf : float, optional
        Ice surface velocity within domain (meters per year).
        Defaults to 100 (m/year).
    fracture_toughness : float
        value to use for fracture toughness of ice (kPa), defaults 
        to value defined in physical_constants.py (0.1 kPa)
    ice_density : float, int
        ice density to use throughout ice block in units of kg/m^3.
        Defaults to value set in physical_constants.py (917 kg/m^3)


    Notes
    -----
    future versions aim to allow a density profile.

    """

    def __init__(
        self,
        ice_thickness,
        dz,
        dt=1,
        thermal_freq=5,
        crev_spacing=30,
        crev_count=None,
        T_profile=None,
        T_surface=None,
        T_bed=None,
        u_surf=100.,
        fracture_toughness=10e3,
        ice_density=917
    ):
        """
        Parameters
        ----------
        ice_thickness : float, int
            thickness of ice block in meters
        dz : float, int
            vertical sampling resolution within ice block (m)
        dt : float, int
            Timestep in days to run crevasse model (days)
            defaults to 0.5 days. 
        crev_spacing : float, int
            Spacing between crevasses in crevasse field (m)
        thermal_freq : float, int
            Multiple of timestep to run thermal model. 1 would run the 
            thermal model at every timestep whereas a value of 5 would 
            run the model after every 5 timesteps. Defaults to 5.
        T_profile : np.array, pd.Series, pd.DataFrame, optional
            Temperature profile for upstream boundary condition. The 
            profile will be interpolated to match the thermal model's 
            vertical resolution. A value for ``T_profile`` is required 
            to run ``ThermalModel``.
        T_surface : float, optional
            Ice surface temperature, degrees C. Defaults to 0.
        T_bed : float, int, optional
            Temperature boundary condition at Defaults to None.
        u_surf : float, optional
            Ice surface velocity within domain (meters per year).
            Defaults to 100 (m/year).
        fracture_toughness : float
            value to use for fracture toughness of ice (kPa), defaults 
            to value defined in ``physical_constants.py`` (0.1 kPa)
        ice_density : float, int
            ice density to use throughout ice block in units of kg/m^3.
            Defaults to value set in physical_constants.py (917 kg/m^3)
            future versions aim to allow a density profile.

        """

        super().__init__(ice_density, fracture_toughness)

        # material properties
        # self.density = ice_density
        # self.fracture_toughness = fracture_toughness

        # time domain
        self.dt = dt * pc.SECONDS_IN_DAY
        self.dt_T = self._thermal_timestep(dt, thermal_freq)

        self.crev_spacing = crev_spacing
        self.crev_count = crev_count if crev_count else 1

        # ice block geometry
        self.length = self.calc_length(u_surf)
        self.dx = (0.5 * self.length)/round(0.5 * self.length
                                            / self.diffusion_length())
        self.x = np.arange(-self.dx-self.length, self.dx, self.dx)

        self.ice_thickness = ice_thickness
        self.dz = dz
        self.z = np.arange(-self.ice_thickness, self.dz, self.dz)

        # crevasse field
        self.crev_locs = [(-self.length, -3)]

        # temperature field

        self.temperature = self._init_temperatures(T_profile, T_surface, T_bed)

        # ice velocity
        self.u_surf = u_surf / pc.SECONDS_IN_YEAR

        # stress field

        # <NOTE: should round this value>
        self.x_advect = round(abs(self.u_surf) * self.dt, 4)

    # def get_length(self):
    #     pass

    # def set_length(self):
    #     # if hasattr(self,"length"):

    #     pass

    def diffusion_length(self):
        """calculate the horizontal diffusion of heat through ice, m"""
        return np.sqrt(pc.THERMAL_DIFFUSIVITY * self.dt_T)

    def _init_geometry(self):
        """initialize ice block geometry

        Return
        ------
        x : np.array
            1-D array defining the horizontal (x-direction) 
            [-dx - length : dx : 0]
        z : np.array
            1-D array defining the vertical (z-direction) 
            [0 : dz : ice_thickness]

        """
        x = np.arange(-self.dx-self.length, self.dx, self.dx)
        z = np.arange(-self.ice_thickness, self.dz, self.dz)
        return x, z

    def _thermal_timestep(self, timestep, thermal_freq):
        if round(365 % (timestep*thermal_freq)) != 0:
            raise ValueError(
                "thermal_freq must divide 365 evenly")
        return round(self.dt * thermal_freq)

    def _init_temperatures(self, T_profile, T_surface, T_bed):
        return None if isinstance(T_profile, type(None)) else ThermalModel(
            self.ice_thickness, self.length, self.dt_T, self.dz, self.dx,
            self.crev_locs, T_profile, T_surface, T_bed)

    def advect_domain(self):
        """increase domain length to allow crevasses to move downstream

        For the model's timestep `dt` and user-defined ice velocity 
        `u_surf` allow the 2-D model domain to expand in length by 
        adding ice at the up-stream boundary. 

        `IceBlock` attributes are modified by this function
        `.length` increases by the distance advected in each timestep
        `.x` will reflect new domain length [-length,dx,0]
        """
        pass

    def stress_field(self):
        pass

    def calc_length(self, usurf):
        """Calculate initial length of ice block using class init args.

        ice block created to initially have 1 year of ice at the 
        downstream end of the ice block ahead of the first crevasse's 
        location. This is specified to keep the crevasse cold. Otherwise
        the downstream boundary condition becomes diffusively influenced
        by the downstream-most crevasse.

        condition: if model is run from t=0 and no crevasses exist the
        ice block length will just be the 1 year of ice ahead of the 
        first crev. If class initialized for a pre-existing crevasse 
        field the length depends on the number of crevasses and spacing 
        of the crevasse field in addition to the one year of ice at the 
        downstream end.

        Parameters
        ----------
        usurf 
        crev_count
        crev_spacing

        """
        # if crev_count and crev_count >= 1:
        #     length = crev_count*crev_spacing + usurf
        # else:
        #     length = usurf
        return self.crev_count * self.crev_spacing + usurf


# class CrevasseField:
#     def __init__(self,
#                  crev_spacing,
#                  crev_count=None
#                  ):
#         self.crev_spacing = crev_spacing
#         self.crev_count = crev_count if crev_count else 1


# class Crevasse:
#     def __init__(self, Qin):
#         # material properties
#         self.ice_thickness = None
#         self.Rxx = None
#         self.depth = None

#         # forcing
#         self.Qin = Qin

#         # creasse geometry
#         self.D = None

#         # water
#         self.water_depth = self.depth - self.calc_water_height()

#     def F(self, use_approximation=False):
#         """Finite ice thickness correction for stress intensity factor

#         from van der Veen (1998) equation 6
#         F(lambda) where lambda = crevasse depth / ice thickness

#         correction accounts for the ~12% increase in stress intensity
#         factor that accounts for material properties (e.g., the crack
#         tip plastic zone-area where plastic deformation occurs ahead
#         of the crack's tip.).

#         Note
#         ----
#         correction is only used for the tensile stress component of
#           K_I (mode 1)
#
#         Parameters
#         ----------
#         crevasse_depth: float, int
#             depth below surface in meters
#         ice_thickness: float, int
#             ice thickness in meters
#         use_approx: bool
#             whether to use shallow crevasse approximation.
#             if True return 1.12 instead of the polynomial expansion.
#             Defaults to False

#         Returns
#         -------
#         F(lambda): float
#             stress intensity correction factor
#         """
#         p = P([1.12, -0.23, 10.55, -21.72, 30.39])
#         return 1.12 if use_approx else p(self.depth/self.ice_thickness)

#     def tensile_stress(self):
#         """calculate tensile stress

#         Note
#         ----
#         an approximatioin of the polynomial coefficient can be used
#         if the ratio between crevasse_depth and ice thickness < 0.2
#         Future work could add an if statement, but should test if the
#         full computation with numpy.polynomial.Polynomial is faster
#         than using the conditional.

#         Equation from van der Veen 1998 where the stress intensity
#           factor (K_I)::

#             K_I(1) = F(lambda)*Rxx*sqrt(pi*crevasse_depth)
#             where lambda = crevasse_depth / ice_thickness and

#             F(lambda) = 1.12 - 0.23*lambda + 10.55*lambda**2
#                   - 12.72*lambda**3
#                   + 30.39*lambda**4

#         For shallow crevasses::

#             F(lambda->0) = 1.12 * Rxx * sqrt(pi*crevasse_depth)


#         Parameters
#         ----------
#         Rxx :
#             far-field stress or tensile resistive stress
#         crevasse_depth : float
#             crevasse depth below ice surface in m
#         ice_thickness : float
#             ice thickness in meters

#         Returns
#         -------
#             stress intensity factor's tensile component
#         """
#         return self.F(self.depth, self.ice_thickness
#                       ) * self.Rxx * sqrt(pi * self.depth)

#     def calc_water_height(self):
#         """calc water high in crevasse using van der Veen 2007

#         van der Veen 1998/2007 equation to estimate the net stress
#         intensity factor (KI) for mode I crack opening where::

#             KI = tensile stress - lithostatic stress + water pressure(1)
#             KI = 1.12 * Rxx * sqrt(pi * ice_thickness)               (2)
#             - 0.683 * ice_density * g * ice_thickness**1.5
#             + 0.683 * water_density * g * water_height**1.5


#         because KI = KIC (the `fracture_toughness` of ice) when a crack
#         opens, we set KI=KIC in equation 2 and solve for water_height::

#             water_height = (( KIC                                   (3)
#             - 1.12 * Rxx * sqrt(pi * ice_thickness)
#             + 0.683 * ice_density * g * ice_thickness**1.5)
#             / 0.683 * water_density * g )**2/3

#         **Assumptions**: Rxx is constant with depth, doesn't
#                           account for firn

#         Note
#         ----
#         using the function `tensile_stress()` will calculate the full
#         tensile stress term instead of using the approximation of 1.12
#         shown in equations 2 and 3. An if statement can be added,
#         however, numpy's polynomial function is quite fast.

#         Parameters
#         ----------
#         Rxx:
#             far field tensile stress
#         crevasse_depth: float
#             crevasse depth below ice surface (m)
#         fracture_toughness: float
#             fracture toughness of ice units of MPa*m^1/2
#         ice_thickness: float): ice thickness in meters
#         ice_density: float, optional
#             Density of ice/firn.Defaults to DENSITY_ICE = 917 kg/m^3.

#         Returns
#         -------
#         water_height: float
#             water height above crevasse bottom (m)
#             values (0, crevasse_depth) -> boundaries rep a water-free
#             crevasse (=0) or a copletely full crevasse (=crevase_depth).
#         """
#         return (
#             (
#                 self.fracture_toughness
#                 - self.tensile_stress(self.Rxx, self.depth,self.ice_thickness)
#                 + 0.683 * self.ice_density * g * sqrt(pi) * self.depth ** 1.5
#             )
#             / (0.683 * DENSITY_WATER * g * sqrt(pi))
#         ) ** (2 / 3)

#     def elastic_displacement(self, z, sigma_T, mu,
#                              alpha=(1-POISSONS_RATIO),
#                              has_water=True
#                              ):
#         # define D and alpha for a water-free crevasse
#         sigma_A = self.applied_stress(sigma_T, self.depth,
#                                       self.water_depth, has_water=has_water)
#         # define constant to advoid repeated terms in D equation
#         c1 = (2*alpha)/(mu*pi)
#         D = (c1*pi*sigma_A * diff_squares(self.depth, z)
#              + c1*DENSITY_ICE*g*self.depth*diff_squares(self.depth, z)
#              - c1*DENSITY_ICE * g * z ** 2 * 0.5 *
#              np.log(sum_over_diff(self.depth, diff_squares(self.depth, z)))
#              )
#         if has_water:
#             c1 = c1 * DENSITY_WATER * g
#             D = (D
#                  - c1 * diff_squares(self.depth, self.water_depth) *
#                  diff_squares(self.depth, z)
#                  + (c1 * (z**2-self.water_depth**2)*0.5*np.log(abs(
    #                 sum_over_diff(
#                     diff_squares(self.depth, self.water_depth),
#                     diff_squares(self.depth, z)))))
#                  - c1*self.water_depth*z*np.log(abs(
    #                   sum_over_diff(self.water_depth
#                       *diff_squares(self.depth, z),
#                       z*diff_squares(self.depth, self.water_depth))))
#                  + c1*self.water_depth**2 *
#                  np.log(abs(sum_over_diff(diff_squares(self.depth, z),
#                         diff_squares(self.depth, self.water_depth))))
#                  )
#         return abs(D)

#     def sigma(self):
#         return (self.sigma_T - (2 * DENSITY_ICE * g * self.depth) / pi
#                 - DENSITY_WATER * g * self.water_depth
#                 + (2/pi)*DENSITY_WATER * g * self.water_depth * math.asin(
#                 self.water_depth/self.depth)
#                 + ((2*DENSITY_WATER*g) / pi) * math.sqrt(
#                 self.depth ** 2 - self.water_depth ** 2)
#                 )

#     def applied_stress(self, has_water=False):
#         sigma_A = self.traction_stress - (2*DENSITY_ICE*g*self.depth)/pi
#         if has_water:
#             sigma_A = (sigma_A
#                        - DENSITY_WATER*g*self.water_depth
#                        + (2/pi) * DENSITY_WATER * g * self.water_depth *
#                        np.arcsin(self.water_depth/self.depth)
#                        + (2 * DENSITY_WATER * g * (
#                            self.depth**2 - self.water_depth**2)**(.5))/pi
#                        )
#         return sigma_A


# def alpha(dislocation_type="edge", crack_opening_mode=None):
#     if dislocation_type == "screw" or crack_opening_mode in [3, 'iii', 'III']:
#         alpha = 1
#     elif dislocation_type == "edge" or crack_opening_mode in [1, 2, 'I', 'II']:
#         alpha = 1 - POISSONS_RATIO
#     else:
#         print(f'incorrect function inputs,assuming edge dislocation alpha=1-v')
#         alpha = 1 - POISSONS_RATIO
#     return alpha


# # math helper functions to simplify the
# def sum_over_diff(x, y):
#     return (x+y) / (x-y)


# def diff_squares(x, y):
#     return np.sqrt(x**2 - y**2)


# def density_profile(depth, C=0.02, ice_density=917., snow_density=350.):
#     """empirical density-depth relationship from Paterson 1994

#     Parameters
#     ----------
#     depth : float/array
#         depth
#     C : float, optional
#         constant variable with site. 0.0165 m^-1 < C < 0.0314 m^-1.
#           Defaults to 0.02 m^-1
#     ice_density : (float, optional)
#         Defaults to 917.
#     snow_density : (float, optional):
#         Defaults to 350.

#     Returns
#     -------
#         snow density at depth
#     """
#     return ice_density - (ice_density - snow_density) * np.exp(-C*depth)


# Model Geometry
#   + → x
#   ↓
#   z

# ‾‾‾‾⎡‾‾‾‾\                /‾‾‾‾‾‾‾‾‾         ⎤
#     ⎜     \              /                   ⎟
#     ⎜      \<-- D(z) -->/                    ⎟
#     ⎜       \          /                     ⎟
#     d        \--------/  <--- water surface  ⎦
#     ⎜         \wwwwww/
#     ⎜          \wwww/
#     ⎜           \ww/
#     ⎣  crevasse  \/
#         depth

# STRESS INTENSITY FACTOR
# For a fracture to propagate
#        KI >= KIC
# stress @ crev tip must >= fracture toughness of ice
# where KI is the stress intensity factor
# which describes the stresses at the fracture's tip
# the material's fracture toughness (KIC)
