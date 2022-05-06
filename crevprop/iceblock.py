""" 
IceBlock
 --  --  --  -- 
The main container for the crevasse propagation model, holding and 
initializing model geometry.




"""
import numpy as np
from numpy.lib.function_base import diff
from numpy import sqrt, abs
from numpy.polynomial import Polynomial as P
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

from .physical_constants import DENSITY_ICE, DENSITY_WATER, FRACTURE_TOUGHNESS, POISSONS_RATIO, g, pi
from . import physical_constants as pc


class IceBlock(object):
    def __init__(
        self,
        ice_thickness,
        dx,
        dz,
        dt,
        crev_spacing,
        thermal_freq=10,
        T_profile=None,
        T_surface=None,
        T_bed=None,
        t=None,
        u_surf=100.,
        fracture_toughness=FRACTURE_TOUGHNESS,
        ice_density=DENSITY_ICE
    ):
        """A 2-D container containing model domain elements.
        
        
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
            vertical resolution. A value for T_profile is required to 
            run ThermalModel.
        T_surface : float, optional
            Ice surface temperature, degrees C. Defaults to 0.
        T_bed : float, int, optional
            Temperature boundary condition at Defaults to None.
        u_surf : float, optional
            Ice surface velocity within domain (meters per year).
            Defaults to 100.
        fracture_toughness : float
            value to use for fracture toughness of ice (kPa), defaults 
            to value defined in physical_constants.py (0.1 kPa)
        ice_density : float, int
            ice density to use throughout ice block in units of kg/m^3.
            Defaults to value set in physical_constants.py (917 kg/m^3)
            NOTE: future versions aim to allow a density profile.
        """

        # material properties
        self.density = ice_density
        self.fracture_toughness = fracture_toughness
        
        
        # time domain

        self.dt = dt * pc.SECONDS_IN_DAY
        
        
        # ice block geometry

        self.ice_thickness = ice_thickness
        self.dx = dx
        self.dz = dz
        self.u_surf = u_surf
        self.length = self.crev_count * self.crev_spacing + self.u_surf
        self.crev_spacing = crev_spacing
        self.crev_count = 1
        
        # temperature field
        self.dt_T = self._thermal_timestep(dt, thermal_freq)
        self.temperature = self._init_temperatures(T_profile, T_surface, T_bed)
        

    def get_length(self):
        pass
    
    def set_length(self):
        # if hasattr(self,"length"):
            
        pass
    
    
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
                         self.dz, T_profile, T_surface, T_bed) if T_profile else None


class ThermalModel(object):
    def __init__(
        self,
        ice_thickness: Union[int, float],
        length: Union[int, float],
        dt_T: Union[int, float],
        dz: Union[int, float],
        T_profile,
        T_surface=None,
        T_bed=None
    ):
        """Apply temperature advection and diffusion through ice block.

        Parameters
        ----------
        ice_thickness : int, float
            ice thickness in meters.
        length : int, float
            Length of model domain in the x-direction (m).
        dt_T : int, float
            thermal model timestep in seconds.
        dz : int, float:
            vertical resolution for domain (m). Value is set to 5 m if 
            input value is below 5 to reduce computational load. 
        T_profile : pd.Series, pd.DataFrame, np.Array
            Temperatures within ice column to set upstream boundary 
            condition within ice block. Temperatures in deg C with 
            corresponding depth from ice surface (m). 
        T_surface : float, int
            Air temperature in deg C. Defaults to 0.
        T_bed : float, int
            Temperature at ice-bed interface in deg C. Defaults to None.
        """
        
        # geometry
        self.length = length
        self.ice_thickness = ice_thickness
        self.dt = dt_T
        self.dx = (0.5*self.length) / round(0.5*self.length /
                                            self._diffusion_lengthscale())
        self.dz = dz if self._ge(dz, 5) else 5
        self.z = np.arange(-self.ice_thickness, self.dz, self.dz) if isinstance(
            self.dz, int) else self._toarray(-self.ice_thickness, self.dz, self.dz)
        self.x = self._toarray(-self.dx-self.length, 0, self.dx)


        # Boundary Conditions
        self.T_surface = T_surface if T_surface else 0
        self.T_bed = T_bed if T_bed else 0
        self.T_upglacier = self._set_upstream_bc(T_profile)
        # left = upglacier end, right = downglacier
        self.T = np.outer(self.T_upglacier, np.linspace(1, 0.99, self.x.size))
        self.Tdf = pd.DataFrame(
            data=self.T, index=self.z, columns=np.round(self.x))
        self.T_crev = 0
        # self.crev_locs = 0

    def _diffusion_lengthscale(self):
        return np.sqrt(1.090952729018252e-6 * self.dt)

    def _ge(self, n, thresh):
        return True if n >= thresh else False

    def _toarray(self, start, stop, step):
        return np.linspace(start, stop, abs(round((start-stop)/step))+1)

    def _set_upstream_bc(self, Tprofile):
        """interpolate temperature profile data points to match z res.

        Args:
            Tprofile [Tuple(array, array), pd.DataFrame]: Temperature 
                profile data points. temperature, elevation = Tuple
                your array must be structured such that 

        Returns:
            [np.array]: Ice temperatures for entire ice block in deg C.
        """
        # interpolate temperature profile to match z (vertical resolution.)
        if isinstance(Tprofile, pd.DataFrame):
            t, z = (1, 0) if Tprofile[Tprofile.columns[0]
                                      ].is_monotonic else (0, 1)
            t = Tprofile[Tprofile.columns[t]].values
            z = Tprofile[Tprofile.columns[z]].values
        elif isinstance(Tprofile, Tuple):
            t, z = Tprofile

        T = np.interp(self.z, z, t)

        # smooth temperature profile with a 25 m window
        win = self.dz*25+2
        T = pd.Series(T[:win]).append(pd.Series(T).rolling(
            win, min_periods=1, center=True).mean()[win:])

        # apply basal temperature condition
        idx = 0 if z[0] < -1 else -1
        T[idx] = self.T_bed

        return T.values

    def t_resample(self, dz):
        if dz % self.dz == 0:
            T = self.Tdf[self.Tdf.index.isin(self.z[::dz].tolist())].values
        # ToDo add else statement/another if statement
        return T

    def A_matrix(self):
        """create the A matrix to solve for future temperatures

        [y] = [A]^-1[b] where A is the A matrix and b is current temperatures
                        throughout ice block. Y are the temperatures at the
                        same points at the next timestep

        NOTE: As the iceblock advects downglacier and the domain's length
              increases until reaching the specified maximum A will need
              to be recalculated. 

        WARNING: Function does not currently add boundary conditions for
        crevasse locations. 

        Returns:
            A (np.ndarray): square matrix with coefficients to calculate
                            temperatures within the ice block 
        """
        nx = self.x.size

        sx = pc.THERMAL_DIFFUSIVITY * self.dt / self.dx ** 2
        sz = pc.THERMAL_DIFFUSIVITY * self.dt / self.dz ** 2

        # create inversion matrix A
        A = np.eye(self.T.size)
        for i in range(nx, self.T.size - nx):
            if i % nx != 0 and i % nx != nx-1:
                A[i, i] = 1 + 2*sx + 2*sz
                A[i, i-nx] = A[i, i+nx] = -sz
                A[i, i+1] = A[i, i-1] = -sx

        # Apply crevasse location boundary conditions to A
        return A

    def _execute(self):
        """Solve for future temp w/ implicit finite difference scheme
        """
        pass

    def thermal_conductivity(density):
        """
        depth dependant thermal conductivity formula, Van Dusen 1929

        note, this gives a lower limit in most cases

        Args:
            density (float): ice/snow density in kg/m^3
        """
        return 2.1e-2 + 4.2e-4 * density + 2.2e-9 * density**3

    def specific_heat_capacity(temperature):
        """specific heat capacity for pure ice of temperature in Kelvin

        Specific heat capacity, c, per unit mass of ice in SI units. 
        Note: c of dry snow and ice does not vary with density because the 
        heat needed to warm the air and vapor between grains is neglibible.
        (see Cuffey, ch 9, pp 400)

        c = 152.5 + 7.122(T)

        Args:
            temperature (float): ice temperature in degrees Kelvin

        Returns:
            float : specific heat capacity (c) in Jkg^-1K^-1
        """

        return 152.5 + 7.122 * temperature

    def thermal_diffusivity(thermal_conductivity, density, specific_heat_capacity):
        """

        Args:
            thermal_conductivity (float): W/mK
            density (float): kg/m^3
            specific_heat_capacity (float): J/kgK

        Returns:
            thermal_diffusivity (float): units of m^2/s
        """
        return thermal_conductivity/(density * specific_heat_capacity)


class CrevasseField():
    pass




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

        NOTE: correction is only used for the tensile stress component of
        K_I (mode 1)

        Args:
            crevasse_depth : depth below surface in meters
            ice_thickness : in meters
            use_approximation (bool): defaults to False
                if True return 1.12 instead of the polynomial expansion

        Returns:
            F(lambda) float : stress intensity correction factor


            """
        p = P([1.12, -0.23, 10.55, -21.72, 30.39])
        return 1.12 if use_approximation else p(self.depth / self.ice_thickness)


    def tensile_stress(self):
        """

        <Note: an approximatioin of the polynomial coefficient can be used
        if the ratio between crevasse_depth and ice thickness is less than
        0.2. Future work could add an if statement, but should test if the
        full computation with numpy.polynomial.Polynomial is faster than the
        conditional.>

        Equation from van der Veen 1998
        stress intensity factor K_I(1) = F(lambda)*Rxx*sqrt(pi*crevasse_depth)
        where lambda = crevasse_depth / ice_thickness and
        F(lambda) =    1.12 - 0.23*lambda + 10.55*lambda**2
                    - 12.72*lambda**3 + 30.39*lambda**4

        For shallow crevasses
            F(lambda->0) = 1.12 * Rxx * sqrt(pi*crevasse_depth)

        Args:
            Rxx (): far-field stress or tensile resistive stress
            crevasse_depth (float): crevasse depth below ice surface in m
            ice_thickness (float): ice thickness in meters

        Returns:
            stress intensity factor's tensile component
        """
        return self.F(self.depth, self.ice_thickness
                      ) * self.Rxx * sqrt(pi * self.depth)


    def calc_water_height(self):
        """calc water high in crevasse using van der Veen 2007

        van der Veen 1998/2007 equation to estimate the net stress intensity
        factor (KI) for mode I crack opening where

        KI = tensile stress - lithostatic stress + water pressure        (1)
        KI = 1.12 * Rxx * sqrt(pi * ice_thickness)                       (2)
            - 0.683 * ice_density * g * ice_thickness**1.5
            + 0.683 * water_density * g * water_height**1.5


        because KI = KIC (the `fracture_toughness` of ice) when a crack
        opens, we set KI=KIC in equation 2 and solve for water_height:

        water_height = (( KIC                                            (3)
                        - 1.12 * Rxx * sqrt(pi * ice_thickness)
                        + 0.683 * ice_density * g * ice_thickness**1.5)
                        / 0.683 * water_density * g )**2/3

        Assumptions:
            - Rxx is constant with depth - not accounting for firn
        Note:
            using the function `tensile_stress()` will calculate the full
            tensile stress term instead of using the approximation of 1.12
            shown in equations 2 and 3. An if statement can be added,
            however, numpy's polynomial function is quite fast.

        Args:
            Rxx ([type]): far field tensile stress
            crevasse_depth (float): crevasse depth below ice surface (m)
            fracture_toughness (float): fracture toughness of ice
                units of MPa*m^1/2
            ice_thickness (float): ice thickness in meters
            ice_density (float, optional): [description].
                        Defaults to DENSITY_ICE = 917 kg/m^3.

        Returns:
            water_height (float): water height above crevasse bottom (m)
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
                 - c1 * diff_squares(self.depth, self.water_depth) * diff_squares(self.depth, z)
                + (c1 * (z**2-self.water_depth**2)*0.5*np.log(abs(sum_over_diff(
                    diff_squares(self.depth, self.water_depth), diff_squares(self.depth, z)))))
                - c1*self.water_depth*z*np.log(abs(sum_over_diff(self.water_depth*diff_squares(
                    self.depth, z), z*diff_squares(self.depth, self.water_depth))))
                + c1*self.water_depth**2 *
                np.log(abs(sum_over_diff(diff_squares(self.depth,z),
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

    Args:
        depth (float/array): depth
        C (float, optional): constant variable with site
            0.0165 m^-1 < C < 0.0314 m^-1
            defaults to 0.02 m^-1
        ice_density (float, optional): [description]. Defaults to 917.
        snow_density (float, optional): [description]. Defaults to 350.

    Returns:
        snow density at depth
    """
    return ice_density - (ice_density - snow_density) * np.exp(-C*depth)

        
        
        
"""
fracture.py




Model Geometry

  + → x
  ↓ 
  z                
                 
‾‾‾‾⎡‾‾‾‾\                /‾‾‾‾‾‾‾‾‾         ⎤
    ⎜     \              /                   ⎟
    ⎜      \<-- D(z) -->/                    ⎟
    ⎜       \          /                     ⎟
    d        \--------/  <--- water surface  ⎦
    ⎜         \wwwwww/
    ⎜          \wwww/
    ⎜           \ww/
    ⎣  crevasse  \/
        depth


"""





# STRESS INTENSITY FACTOR
# For a fracture to propagate 
#        KI >= KIC
# stress @ crev tip must >= fracture toughness of ice
# where KI is the stress intensity factor
# which describes the stresses at the fracture's tip
# the material's fracture toughness (KIC)


