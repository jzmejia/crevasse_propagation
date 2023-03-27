import math as math

from typing import Union
import numpy as np

from .crevasse import Crevasse


class CrevasseField():
    """Container for all crevasses within model domain

    Primary roles:
    1. Gets crevasse field and model domain geometry from `IceBlock` 
    2. Initializes, drives, and tracks all crevasses within domain as 
        model runs through time and crevasses are advected downglacier


    Attributes
    ----------
    fracture_toughness: int
        fracture toughness of ice in Pa m^0.5
    crev_spacing: int
        spacing of crevasses within crevasse field in meters. Value
        defined in `IceBlock` and remains unchanged.
    sigmaT0: float, int, optional
        maximum far field tensile stress, amplitude of the stress
        field that the ice mass gets advected through,
        by default 120e3 Pa m ^ 0.5
    PFA_depth: float, int, optional
        meters penetration required to access the perrenial firn
        aquifer, by default None
    crev_count: int
        number of crevasses in crevasse field
    crev_locs: list of tuples
        x-coordinates of each crevasse in field with current depths
    """

    def __init__(self,
                 z: np.array,
                 dx: float,
                 dz,
                 dt: float,
                 ice_thickness,

                 x: np.array,
                 length: float,

                 fracture_toughness,
                 crev_spacing: Union[int, float],
                 max_crevs,
                 sigmaT0=120e3,

                 PFA_depth=None,
                 blunt=False,
                 include_creep=False,
                 never_closed=True,
                 water_compressive=False  # , *args, **kwargs
                 ):
        """

        Parameters
        ----------
        x : np.array
            x-coordinates of model geometry/domain. 
        z : np.array
            depth-vector corresponding to ice block thickness
        dx : float
            horizontal sampling resolution of ice block (m)
        dz : int, float
            vertical sampling resolution of ice block (m)
        dt : float
            timestep of model runs
        ice_thickness : float, int
            thickness of ice block in meters
        length : float, int
            length of ice block in meters
        fracture_toughness : int
            fracture toughness of ice in Pa m^0.5
        crev_spacing : int
            spacing of crevasses within crevasse field in meters.
        sigmaT0 : float, int, optional
            maximum far field tensile stress, amplitude of the stress
            field that the ice mass gets advected through, 
            by default 120e3 Pa m^0.5
        PFA_depth : float, int, optional
            meters penetration required to access the perrenial firn 
            aquifer, by default None
        include_creep : bool, optional
            consider creep closure in model, by default False
            NOTE: creep not yet supported by model due to data input 
            requirements
        blunt : bool, optional
            whether to blunt the stresses on individual crevasses, used
            with creep calculation and sets sigmaTcrev=0 for 
            interior crevasses within crevasse field, by default False
        never_closed : bool, optional
            always allow melt into the crevasse regardless of 
            near-surface pintching. This only affects the Qin calc.
            by default True.
        water_compressive : bool, optional
            whether to allow water into crevasse when the longitudinal 
            stress on the crevasse is negative (compressive stress 
            regeime). If false, don't allow water into crevasse if in a
            compressive regeime. This affects the Qin calculation. 
            by default False

        """
        # model geometry and domian management
        self.ice_thickness = ice_thickness
        self.x = x  # will change over time
        self.z = z
        self.dx = dx
        self.dz = dz
        self.dt = dt
        self.length = length  # will change over time

        self.crev_spacing = crev_spacing
        self.max_crevs = max_crevs

        # ice properties
        self.fracture_toughness = fracture_toughness
        self.ice_softness = 1e8  # how soft is the ice

        #  run create_crevasse at this point then
        # identify crevasses in instances object by calling
        # Crevasse.instances e.g., self.crevasses=Crevasse.instances
        # at bottom for now make a method so that i don't need all this
        self.crevasses = self.create_crevasse()
        self.crev_locs = [(-self.length, -0.1)]
        self.crev_count = len(self.crev_locs)  # self.crevasse_list()

        # temporary things needed for stress field
        self.sigmaT0 = sigmaT0
        self.sigmaCrev = 0  # changes sigma on each crev
        self.wps = 3e3  # positive strain area width (m) half-wavelength

        self.PFA_depth = PFA_depth

        # model options
        self.blunt = blunt
        self.include_creep = include_creep
        self.never_closed = never_closed
        self.water_compressive = water_compressive

        self.crev_instances = Crevasse.instances

    def expand_domain(self):
        pass

    def propagate_fractures(self):
        pass

    def crevasse_list(self):
        # example
        crev_count = self.crev_instances.length()
        print(f"number of crevasses in crevasse field: {crev_count}")
        return crev_count

    def stress_field(self):
        """define stress field based on model geometry

        stress field = sin wave with an amplitude matching
        Rxx and a wavelength matching crevasse field length

        a separate function will determine position within stress
        field as model progresses (i.e., you will need to know the 
        stresses at each crevasse location based on where they fall
        within the stress field defined here.) Potentially move to 
        crevasse field (or calculated here and then input to crevasse 
        field class)

        sigma_T = sigma_T0 * sin(-pi/ (wps * x))
        where 
        sigma_T = defines the stress sigma_T for all x in model domain
        sigma_T0 = maximum tensile stress within stress field (Rxx)
        wps = width of positive strain area (half wave length)
        x = x-vector of model/iceblock 

        """
        pass

    def create_crevasse(self):
        """create and initialize a new crevasse"""
        # new crevasses will be put in their designated location on the
        # upstream edge of the model domain boundary of iceblock

        # print statement

        # initialize via running Crevasse class at init

        # add storage terms to class and update attrs/data
        # newCrev = Crevasse(self.z, self.dz, self.ice_thickness,)
        Qin = 1e-4  # zero value
        sigmaCrev = 10e4
        Crevasse(self.z,
                 self.dz,
                 self.ice_thickness,
                 self.x[0],
                 Qin,
                 self.ice_softness,
                 sigmaCrev,
                 fracture_toughness=self.fracture_toughness)
