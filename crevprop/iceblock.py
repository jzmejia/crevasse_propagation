"""
The main container for the crevasse propagation model, holding and 
initializing model geometry.
"""
import numpy as np
from numpy.lib.function_base import diff
from numpy import sqrt, abs
from numpy.polynomial import Polynomial as P
from scipy.constants import g, pi

import math as math
import pandas as pd
from typing import (
    Dict,
    List,
    Tuple,
    Any,
    Optional,
    Union,
)

from .temperature_field import ThermalModel
from .physical_constants import DENSITY_ICE, DENSITY_WATER, POISSONS_RATIO
from . import physical_constants as pc


class IceBlock():
    """
    Two-dimensional ice block geometry defining model domain.
    
    Attributes
    ----------
    ice_thickness : float, int
        thickness of ice block in meters
    dx : float, int
        horizontal (x-coordinate) sampling within ice block (m)
    dz : float, int
        vertical sampling resolution within ice block (m)
    x : np.array
        x-coordinates of model domain ranging from 0 at downstream boundary
        to -length at upstream boundary, with spacing of dx. Unit of meters.
        This array defines the x coordinates of the iceblock overwhich
        to run the model. 
    z : np.array
        z-coordinates (verical) of model domain ranging from 0 at the ice
        surface to the ice thickness at the base of the ice sheet, spacing 
        of dz. Unit of meters. This array defines the z coordinates of the
        iceblock overwhich to run the crevasse propagation model. Note that
        the thermal model will run with a minimum vertical resolution of 5m
        to reduce computational costs (dz>=5m for thermal model). 
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
        positional information corresponding to crevasses within crevasse field. 
        Tuple entries contain the (x-coordinate, depth) of a crevasse, with the
        number of list entries equal to the current crevasse count. These values 
        are used by the ``ThermalModel`` when solving for ice temperature and 
        refreezing. 
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
    
    
    
    Note
    ----
    future versions aim to allow a density profile.
    """
    def __init__(
        self,
        ice_thickness,
        dx,
        dz,
        dt,
        crev_spacing,
        crev_count=None,
        thermal_freq=10,
        T_profile=None,
        T_surface=None,
        T_bed=None,
        u_surf=100.,
        fracture_toughness=100e3,
        ice_density=DENSITY_ICE
    ):
        """
        Parameters
        ----------
        ice_thickness : float, int
            thickness of ice block in meters
        dx : float, int
            horizontal (x-coordinate) sampling within ice block (m)
        dz : float, int
            vertical sampling resolution within ice block (m)
        dt : float, int
            Timestep in days to run crevasse model (days)
        crev_spacing : float, int
            Spacing between crevasses in crevasse field (m)
        thermal_freq : float, int
            Multiple of timestep to run thermal model. 1 would run the 
            thermal model at every timestep whereas a value of 10 would 
            run the model after every 10 timesteps. Defaults to 10.
        T_profile : np.array, pd.Series, pd.DataFrame, optional
            Temperature profile for upstream boundary condition. The 
            profile will be interpolated to match the thermal model's 
            vertical resolution. A value for ``T_profile`` is required to 
            run ``ThermalModel``.
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

        # material properties
        self.density = ice_density
        self.fracture_toughness = fracture_toughness

        # time domain
        self.dt = dt * pc.SECONDS_IN_DAY
        # self.t = 0

        # ice block geometry
        self.ice_thickness = ice_thickness
        self.dx = dx
        self.dz = dz
        self.u_surf = u_surf

        # crevasse field
        self.crev_spacing = crev_spacing
        self.crev_count = crev_count if crev_count else 1
        self.crev_locs = [(-self.length, -3)]

        self.x, self.z, self.length = self._init_geometry()

        # temperature field
        self.dt_T = self._thermal_timestep(dt, thermal_freq)
        self.temperature = self._init_temperatures(T_profile, T_surface, T_bed)

    # def get_length(self):
    #     pass

    # def set_length(self):
    #     # if hasattr(self,"length"):

    #     pass

    def _toarray(self, start, stop, step):
        return np.linspace(start, stop, abs(round((start-stop)/step))+1)

    def _init_geometry(self):
        """initialize ice block geometry
        """
        x = self._toarray(-self.dx-self.length, 0, self.dx)
        z = self._toarray(-self.ice_thickness, 0, self.dz)
        length = self.crev_count * self.crev_spacing + self.u_surf
        return x, z, length

    def advect_domain(self):
        """advect domain downglacier in accordance with ice velocity
        """
        pass

    def _thermal_timestep(self, timestep, thermal_freq):
        if round(365 % (timestep*thermal_freq)) != 0:
            raise ValueError(
                "thermal_freq must divide 365 evenly")
        return round(self.dt * thermal_freq)

    def _init_temperatures(self, T_profile, T_surface, T_bed):
        return ThermalModel(self.ice_thickness, self.length, self.dt_T,
                            self.dz, self.crev_locs, T_profile, T_surface, T_bed) if T_profile else None


# class CrevasseField:
#     def __init__(self,
#                  crev_spacing,
#                  crev_count=None
#                  ):
#         self.crev_spacing = crev_spacing
#         self.crev_count = crev_count if crev_count else 1


class Crevasse:
    def __init__(self, Qin):
        # material properties
        self.ice_thickness = None
        self.Rxx = None
        self.depth = None

        # forcing
        self.Qin = Qin

        # creasse geometry
        self.D = None

        # water
        self.water_depth = self.depth - self.calc_water_height()

    def F(self, use_approximation=False):
        """Finite ice thickness correction for the stress intensity factor

        from van der Veen (1998) equation 6
        F(lambda) where lambda = crevasse depth / ice thickness

        correction accounts for the ~12% increase in the stress intensity
        factor that accounts for material properties such aas the crack
        tip plastic zone (i.e., area where plastic deformation occurs ahead
        of the crack's tip.).

        Note
        ----
        correction is only used for the tensile stress component of K_I (mode 1)

        Parameters
        ----------
        crevasse_depth: float, int
            depth below surface in meters
        ice_thickness: float, int
            ice thickness in meters
        use_approximation: bool
            whether to use shallow crevasse approximation.
            if True return 1.12 instead of the polynomial expansion.
            Defaults to False

        Returns
        -------
        F(lambda): float
            stress intensity correction factor
        """
        p = P([1.12, -0.23, 10.55, -21.72, 30.39])
        return 1.12 if use_approximation else p(self.depth / self.ice_thickness)

    def tensile_stress(self):
        """calculate tensile stress

        Note
        ----
        an approximatioin of the polynomial coefficient can be used
        if the ratio between crevasse_depth and ice thickness is less than
        0.2. Future work could add an if statement, but should test if the
        full computation with numpy.polynomial.Polynomial is faster than the
        conditional.

        Equation from van der Veen 1998 where the stress intensity factor (K_I)::
            
            K_I(1) = F(lambda)*Rxx*sqrt(pi*crevasse_depth)
            where lambda = crevasse_depth / ice_thickness and
            
            F(lambda) = 1.12 - 0.23*lambda + 10.55*lambda**2 - 12.72*lambda**3 + 30.39*lambda**4

        For shallow crevasses::
        
            F(lambda->0) = 1.12 * Rxx * sqrt(pi*crevasse_depth)

        
        Parameters
        ----------
        Rxx : 
            far-field stress or tensile resistive stress
        crevasse_depth : float
            crevasse depth below ice surface in m
        ice_thickness : float
            ice thickness in meters

        Returns
        -------
            stress intensity factor's tensile component
        """
        return self.F(self.depth, self.ice_thickness
                      ) * self.Rxx * sqrt(pi * self.depth)

    def calc_water_height(self):
        """calc water high in crevasse using van der Veen 2007

        van der Veen 1998/2007 equation to estimate the net stress intensity
        factor (KI) for mode I crack opening where::

            KI = tensile stress - lithostatic stress + water pressure        (1)
            KI = 1.12 * Rxx * sqrt(pi * ice_thickness)                       (2)
            - 0.683 * ice_density * g * ice_thickness**1.5
            + 0.683 * water_density * g * water_height**1.5


        because KI = KIC (the `fracture_toughness` of ice) when a crack
        opens, we set KI=KIC in equation 2 and solve for water_height::

            water_height = (( KIC                                            (3)
            - 1.12 * Rxx * sqrt(pi * ice_thickness)
            + 0.683 * ice_density * g * ice_thickness**1.5)
            / 0.683 * water_density * g )**2/3

        **Assumptions**: Rxx is constant with depth - not accounting for firn
        
        Note
        ----
        using the function `tensile_stress()` will calculate the full
        tensile stress term instead of using the approximation of 1.12
        shown in equations 2 and 3. An if statement can be added,
        however, numpy's polynomial function is quite fast.

        Parameters
        ----------
        Rxx:
            far field tensile stress
        crevasse_depth: float
            crevasse depth below ice surface (m)
        fracture_toughness: float
            fracture toughness of ice units of MPa*m^1/2
        ice_thickness: float): ice thickness in meters
        ice_density: float, optional
            Density of ice/firn.Defaults to DENSITY_ICE = 917 kg/m^3.

        Returns
        -------
        water_height: float
            water height above crevasse bottom (m)
            values (0, crevasse_depth) -> boundaries rep a water-free
            crevasse (=0) or a copletely full crevasse (=crevase_depth).
        """
        return (
            (
                self.fracture_toughness
                - self.tensile_stress(self.Rxx, self.depth, self.ice_thickness)
                + 0.683 * self.ice_density * g * sqrt(pi) * self.depth ** 1.5
            )
            / (0.683 * DENSITY_WATER * g * sqrt(pi))
        ) ** (2 / 3)

    def elastic_displacement(self, z, sigma_T, mu,
                             alpha=(1-POISSONS_RATIO),
                             has_water=True
                             ):
        # define D and alpha for a water-free crevasse
        sigma_A = self.applied_stress(sigma_T, self.depth,
                                      self.water_depth, has_water=has_water)
        # define constant to advoid repeated terms in D equation
        c1 = (2*alpha)/(mu*pi)
        D = (c1*pi*sigma_A * diff_squares(self.depth, z)
             + c1*DENSITY_ICE*g*self.depth*diff_squares(self.depth, z)
             - c1*DENSITY_ICE * g * z ** 2 * 0.5 *
             np.log(sum_over_diff(self.depth, diff_squares(self.depth, z)))
             )
        if has_water:
            c1 = c1 * DENSITY_WATER * g
            D = (D
                 - c1 * diff_squares(self.depth, self.water_depth) *
                 diff_squares(self.depth, z)
                 + (c1 * (z**2-self.water_depth**2)*0.5*np.log(abs(sum_over_diff(
                    diff_squares(self.depth, self.water_depth), diff_squares(self.depth, z)))))
                 - c1*self.water_depth*z*np.log(abs(sum_over_diff(self.water_depth*diff_squares(
                     self.depth, z), z*diff_squares(self.depth, self.water_depth))))
                 + c1*self.water_depth**2 *
                 np.log(abs(sum_over_diff(diff_squares(self.depth, z),
                                          diff_squares(self.depth, self.water_depth))))
                 )
        return abs(D)

    def sigma(self):
        return (self.sigma_T - (2 * DENSITY_ICE * g * self.depth) / pi
                - DENSITY_WATER * g * self.water_depth
                + (2/pi)*DENSITY_WATER * g * self.water_depth * math.asin(
                self.water_depth/self.depth)
                + ((2*DENSITY_WATER*g) / pi) * math.sqrt(
                self.depth ** 2 - self.water_depth ** 2)
                )

    def applied_stress(self, has_water=False):
        sigma_A = self.traction_stress - (2*DENSITY_ICE*g*self.depth)/pi
        if has_water:
            sigma_A = (sigma_A
                       - DENSITY_WATER*g*self.water_depth
                       + (2/pi) * DENSITY_WATER * g * self.water_depth *
                       np.arcsin(self.water_depth/self.depth)
                       + (2 * DENSITY_WATER * g * (
                           self.depth**2 - self.water_depth**2)**(.5)) / pi
                       )
        return sigma_A


def alpha(dislocation_type="edge", crack_opening_mode=None):
    if dislocation_type == "screw" or crack_opening_mode in [3, 'iii', 'III']:
        alpha = 1
    elif dislocation_type == "edge" or crack_opening_mode in [1, 2, 'I', 'II']:
        alpha = 1 - POISSONS_RATIO
    else:
        print(f'incorrect inputs to function, assuming an edge dislocation alpha=1-v')
        alpha = 1 - POISSONS_RATIO
    return alpha


# math helper functions to simplify the
def sum_over_diff(x, y):
    return (x+y) / (x-y)


def diff_squares(x, y):
    return np.sqrt(x**2 - y**2)


def density_profile(depth, C=0.02, ice_density=917., snow_density=350.):
    """empirical density-depth relationship from Paterson 1994

    Parameters
    ----------
    depth : float/array
        depth
    C : float, optional
        constant variable with site. 0.0165 m^-1 < C < 0.0314 m^-1. Defaults to 0.02 m^-1
    ice_density : (float, optional)
        Defaults to 917.
    snow_density : (float, optional):
        Defaults to 350.

    Returns
    -------
        snow density at depth
    """
    return ice_density - (ice_density - snow_density) * np.exp(-C*depth)





# Model Geometry
#   + ??? x
#   ??? 
#   z                
                 
# ???????????????????????????\                /???????????????????????????         ???
#     ???     \              /                   ???
#     ???      \<-- D(z) -->/                    ???
#     ???       \          /                     ???
#     d        \--------/  <--- water surface  ???
#     ???         \wwwwww/
#     ???          \wwww/
#     ???           \ww/
#     ???  crevasse  \/
#         depth

# STRESS INTENSITY FACTOR
# For a fracture to propagate
#        KI >= KIC
# stress @ crev tip must >= fracture toughness of ice
# where KI is the stress intensity factor
# which describes the stresses at the fracture's tip
# the material's fracture toughness (KIC)
