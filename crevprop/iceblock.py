"""
Copyright (c) 2021-2023 by Jessica Mejia <jzmejia@buffalo.edu>



The main container for the crevasse propagation model, holding and
initializing model geometry
"""
import numpy as np
# from numpy.lib.function_base import diff
from numpy import abs
# from numpy.polynomial import Polynomial as P
# from scipy.constants import g, pi

from dataclasses import dataclass


import math as math
from dataclasses import dataclass
# import pandas as pd
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Union
)

from .ice import Ice
from .temperature_field import ThermalModel
from . import physical_constants as pc
from .crevasse_field import CrevasseField


@dataclass
class geometry():
    """Class for defining 2D model geometry"""
    ice_thickness: float
    dz: float
    dt: float
    u_surf: float
    crev_spacing: int
    xmove: float = None
    length: float = None
    dx: float = None

    def __post_init__(self):
        """add and update calculated attributes 
        """
        self.length = self.crev_spacing + self.u_surf
        self.dt = self.dt * pc.SECONDS_IN_DAY
        self.u_surf = self.u_surf / pc.SECONDS_IN_YEAR
        self.xmove = np.abs(self.u_surf) * self.dt

    @property
    def z(self) -> np.array:
        """1D array defining the vertical (z-direction) axis (m)"""
        return np.arange(-self.ice_thickness, self.dz, self.dz)

    @property
    def x(self) -> np.array:
        """1D array defining the horizontal (x-direction) axis (m)"""
        return np.arange(-self.length, self.dx, self.dx)


@dataclass
class ModelOptions:
    """Class for keeping track of options for crev prop calc


    Parameters
    ----------
        blunt : bool, by default False
            whether to blunt the stresses on individual crevasses, used
            with creep calculation and sets sigmaTcrev=0 for 
            interior crevasses within crevasse field, by default False
        include_creep : bool, bydefault False
            consider creep closure in model, by default False
            NOTE: creep not yet supported by model due to data input 
            requirements
        never_closed : bool, by default True
            always allow melt into the crevasse regardless of 
            near-surface pintching. This only affects the Qin calc.
            by default True.
        water_compressive : bool, by default False
            whether to allow water into crevasse when the longitudinal 
            stress on the crevasse is negative (compressive stress 
            regeime). If false, don't allow water into crevasse if in a
            compressive regeime. This affects the Qin calculation. 
            by default False
    """
    blunt: bool = False
    include_creep: bool = False
    never_closed: bool = True
    water_compressive: bool = False


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
        years_to_run=2,
        thermal_freq=2,
        crev_spacing=30,
        T_profile=None,
        T_surface=None,
        T_bed=None,
        u_surf=200.,
        fracture_toughness=100e3,
        ice_density=917,
        blunt=False,
        include_creep=False,
        never_closed=True,
        compressive=False
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
        self.ibg = geometry(ice_thickness, dz, dt, u_surf, crev_spacing)

        # time domain
        # initialize model to time=0
        self.t = 0
        self.time = 0
        # self.doy = 1
        self.num_years = years_to_run

        self.thermal_freq = thermal_freq
        self.dt_T = self._thermal_timestep(dt, thermal_freq)

        # ice block geometry
        self.dx = self.calc_dx(self.ibg.length)
        setattr(self.ibg, 'dx', self.dx)

        # crevasse field
        # can multiply by int (num years to form new crevs/track for)
        self.max_crevs = round(
            self.ibg.u_surf/self.ibg.crev_spacing) * years_to_run

        # temporary way to store crevasse info
        self.crev_locs = [(-self.ibg.length, -3)]
        # temporary storage for refreezing to bass back and forth

        # temperature field
        self.temperature = ThermalModel(self.ibg, self.dt_T, self.crev_locs,
                                        T_profile, T_surface, T_bed,
                                        thermal_conductivity=self.ki,
                                        ice_density=self.ice_density,
                                        latient_heat_of_freezing_ice=self.Lf,
                                        thermal_diffusivity=self.kappa
                                        ) if T_profile is not None else None

        self.virtualblue = self._get_virtualblue()

        self.crev_field = CrevasseField(self.ibg,
                                        self.fracture_toughness,
                                        self.max_crevs,
                                        self.virtualblue,
                                        ModelOptions(blunt, include_creep,
                                                     never_closed, compressive)
                                        )
        self.xmove = round(abs(self.ibg.u_surf) * self.ibg.dt, 4)

        self.detached = False

    def advect_domain(self, t):
        """increase domain length to allow crevasses to move downstream

        For the model's timestep `dt` and user-defined ice velocity
        `u_surf` allow the 2-D model domain to expand in length by
        adding ice at the up-stream boundary.

        `IceBlock` attributes are modified by this function
        `.length` increases by the distance advected in each timestep
        `.x` will reflect new domain length [-length,dx,0]
        """

        # 1. if domain is detached from upglacier boundary
        # set downstream BC based on the month/season using history
        # from the previous year's runs

        # TODO! because xmove can be smaller than dx, calculate di
        # from cumulative timestep counter n, and subtract added, the
        # tracker of how much the domain has grown over model run
        # added will need to be initalized before this
        # n will need to be initialized outside or given to function

        # di = round(n * self.x_advect / self.dx) - added

        # add conditional for if di >= dx, if not, don't add anything
        # recalculate length
        # recalculate x

        # setattr(ibg, 'x', value)

        # reassign ice surface temperature if variable
        # recalculate T/update thermal model
        # track crevasse location (downglacier-most)

        # detach domain if necessary
        # update thermal model upglacier boundary condition req

        # update u and v velocity profiles if applicable

        pass

    def increment_time(self):
        """run model for one timestep



        1. Update IceBlock geometry
        2. Find new 


        """

        # 1. advect domain
        # update IceBlock geometry

        # update crevasse field geometry
        # self.crevasse_field.run_through_time(input what is needed)
        # find Qin to use in this timestep
        # execute fracture mechanics scheme
        #

        pass

    def update_temperatures(self):
        # for timesteps at the thermal model frequency calculate
        # temperatures for iceblock

        # 1. update thermal model geometry to match iceblock
        # 2. reassign/update crevasse locations in thermal model
        # 3. execute model `ThermalModel.calc_temperature()`
        # 4. recalculate refreezing
        #       this updates ThermalModel instance with new bluelayer
        #       values that will be used in the next temperature calc
        #       and returns the new virtualblue that needs to go to
        #       crevasses for their calculation in next timesteps

        # REFREEZING NEEDS TO BE PASSED TO CREVASSE FIELD

        pass

    def propagate_fractures(self):
        pass

    def _get_virtualblue(self):
        """gets and adjust refreezing values from ThermalModel

        executes `ThermalModel.refreezing()` on current instance defined
        within `self.temperature`, then updates outputs for `IceBlock` 
        `dz` and `dt`. Two list objects are returned, each containing
        np.array objects with values, array objects are in the same
        order as `ThermalModel.crevasses` and 

        Returns
        -------
        virtualblue_left, virtualblue_right : List
            Lists of np.array objects containing potential refreezing
            rates corresponding with crevasses in crevasse field. 
        """
        virtualblue_l, virtualblue_r = self.temperature.refreezing()

        virtualblue_left = []
        virtualblue_right = []

        for num, crev in enumerate(virtualblue_l):
            lhs = crev/self.thermal_freq
            rhs = virtualblue_r[num]/self.thermal_freq

            if self.dx != self.temperature.dx:
                lhs = np.interp(self.ibg.z, self.temperature.z, lhs)
                rhs = np.interp(self.ibg.z, self.temperature.z, rhs)

            virtualblue_left.append(lhs)
            virtualblue_right.append(rhs)

        return virtualblue_left, virtualblue_right

    def _thermal_timestep(self, timestep, thermal_freq):
        if round(365 % (timestep*thermal_freq)) != 0:
            raise ValueError(
                "thermal_freq must divide 365 evenly")
        return round(self.ibg.dt * thermal_freq)

    def diffusion_length(self):
        """diffusion lengthscale for thermal model timestep dt_T"""
        return np.sqrt(self.kappa * self.dt_T)

    def calc_dx(self, length):
        """calculate model dx from diffusion lengthscale"""
        return (0.5 * length)/round(0.5 * length
                                    / self.diffusion_length())

    # def calc_length(self, usurf, crev_count, crev_spacing):
    #     """Calculate initial length of ice block using class init args.

    #     ice block created to initially have 1 year of ice at the
    #     downstream end of the ice block ahead of the first crevasse's
    #     location. This is specified to keep the crevasse cold. Otherwise
    #     the downstream boundary condition becomes diffusively influenced
    #     by the downstream-most crevasse.

    #     condition: if model is run from t=0 and no crevasses exist the
    #     ice block length will just be the 1 year of ice ahead of the
    #     first crev. If class initialized for a pre-existing crevasse
    #     field the length depends on the number of crevasses and spacing
    #     of the crevasse field in addition to the one year of ice at the
    #     downstream end.

    #     Parameters
    #     ----------
    #     usurf : float, int
    #     crev_count : int
    #     crev_spacing  : int

    #     """
    #     crev_count = crev_count if crev_count else 1
    #     return crev_count * crev_spacing + usurf
