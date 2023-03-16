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
from .crevasse import CrevasseField, Crevasse


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
                 ice_density=917.,
                 ice_temperature=0,
                 fracture_toughness=100e3):
        """_summary_

        Parameters
        ----------
        density : int, optional
            ice denisty in kg/m^3, by default 917 kg/m^3
        temperature : int, optional
            ice temperature in degrees Celsius, by default 0C
        fracture_toughness : float, optional
            fracture toughness of ice in Pa, by default 0.1 MPa
        """

        self.ice_density = ice_density
        self.ice_temperature = ice_temperature

        self.specific_heat_capacity = 2097
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

    def van_dusen(self, density):
        """Depth dependant thermal conductivity for dry snow, firn, ice
        Van Dusen (1929)

        This equation typically gives a lower limit in most cases

        Parameters
        ----------
        density : (float)
            density of dry snow, firn, or glacier ice in kg/m^3

        """
        return 2.1e-2 + 4.2e-4 * density + 2.2e-9 * density**3

    def schwerdtfeger(self, density):
        # density must be less than the density of pure ice,
        # find threshold to use here
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
        return k_firn

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
        """calculate thermal diffusivity

        thermal_conductivity / density * specific_heat_capacity


        Returns
        -------
        thermal diffusivity with units of m^2/s
        """
        return self.thermal_conductivity / (
            self.ice_density * self.specific_heat_capacity)

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
        dt=0.5,
        thermal_freq=2,
        crev_spacing=30,
        crev_count=None,
        T_profile=None,
        T_surface=None,
        T_bed=None,
        u_surf=100.,
        fracture_toughness=None,
        ice_density=None
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
            value to use for fracture toughness of ice (Pa), defaults
            to 0.1 MPa. 
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

        # ice block geometry
        self.length = self.calc_length(u_surf, crev_count, crev_spacing)
        self.dx = self.calc_dx()
        self.x = np.arange(-self.dx-self.length, self.dx, self.dx)

        self.ice_thickness = ice_thickness
        self.dz = dz
        self.z = np.arange(-self.ice_thickness, self.dz, self.dz)


        # ice velocity
        self.u_surf = u_surf / pc.SECONDS_IN_YEAR
        
        
        # crevasse field
        
        self.crevasses = self._init_crevfield(crev_spacing, crev_count)
        self.crev_locs = [(-self.length, -3)]
        self.bluelayer = self.crevasses.bluelayer
        

        # temperature field
        self.temperature = self._init_temperatures(T_profile, T_surface, T_bed)

        

        # stress field

        #
        self.x_advect = round(abs(self.u_surf) * self.dt, 4)

    # def get_length(self):
    #     pass

    # def set_length(self):
    #     # if hasattr(self,"length"):

    #     pass

    def _max_crevasses(self):
        
        return max_num_crev
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

    def _init_temperatures(self, T_profile, T_surface, T_bed):
        return None if isinstance(T_profile, type(None)) else ThermalModel(
            self.ice_thickness, self.length, self.dt_T, self.dz, self.dx,
            self.crev_locs, T_profile, T_surface, T_bed,
            thermal_conductivity=self.ki,
            ice_density=self.ice_density,
            latient_heat_of_freezing_ice=self.Lf,
            thermal_diffusivity=self.kappa
        )

    def _init_crevfield(self, crev_spacing, crev_count):
        """Initialize CrevasseField for given model geometry"""
        crevasse_field = CrevasseField(self.x, self.z,
                                       self.dx, self.dz, self.dt,
                                       self.ice_thickness,
                                       self.length,
                                       self.fracture_toughness,
                                       crev_spacing,
                                       crev_count=crev_count
                                       )
        return crevasse_field

    def _thermal_timestep(self, timestep, thermal_freq):
        if round(365 % (timestep*thermal_freq)) != 0:
            raise ValueError(
                "thermal_freq must divide 365 evenly")
        return round(self.dt * thermal_freq)

    def diffusion_length(self):
        """diffusion lengthscale for thermal model timestep dt_T"""
        return np.sqrt(self.kappa * self.dt_T)

    def calc_dx(self):
        """calculate model dx from diffusion lengthscale"""
        return (0.5 * self.length)/round(0.5 * self.length
                                         / self.diffusion_length())

    def advect_domain(self):
        """increase domain length to allow crevasses to move downstream

        For the model's timestep `dt` and user-defined ice velocity
        `u_surf` allow the 2-D model domain to expand in length by
        adding ice at the up-stream boundary.

        `IceBlock` attributes are modified by this function
        `.length` increases by the distance advected in each timestep
        `.x` will reflect new domain length [-length,dx,0]
        """

        # TODO! because xadvect can be smaller than dx, calculate di
        # from cumulative timestep counter n, and subtract added, the
        # tracker of how much the domain has grown over model run
        # added will need to be initalized before this
        # n will need to be initialized outside or given to function

        # di = round(n * self.x_advect / self.dx) - added

        # add conditional for if di >= dx, if not, don't add anything
        # recalculate length
        # recalculate x

        # reassign ice surface temperature if variable
        # recalculate T/update thermal model
        # track crevasse location (downglacier-most)

        # detach domain if necessary
        # update thermal model upglacier boundary condition req

        # update u and v velocity profiles if applicable

        pass

    def calc_length(self, usurf, crev_count, crev_spacing):
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
        crev_count = crev_count if crev_count else 1
        return crev_count * crev_spacing + usurf
