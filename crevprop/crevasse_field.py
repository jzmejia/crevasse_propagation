import math as math
import numpy as np

from dataclasses import dataclass
# from typing import Any, Tuple


from .crevasse import Crevasse


@dataclass
class StressField:
    """
    sigmaT0 : int
        maximum tensile stress (xx) used to define the
        amplitude of the stress field that IceBlock gets
        advected down. Units in Pa
    wpa : int
        width of positve strain area
        half-wavelength of stress field
        defines the length of the stress field
        that experiences positive straining (m)
    blunt : bool
        Whether to blunt stresses within crevasse field.
        Defaults to False.
    """
    sigmaT0: int
    wpa: int
    blunt: bool = False
    # sigma_crev: Any = 0

    def sigmaT(self, x):
        """calculate tensile stress for location x from upstream boundary."""
        return self.sigmaT0 * np.sin(-np.pi/self.wpa*x)


class CrevasseField():
    """Container for all crevasses within model domain

    Primary roles:
    1. Gets crevasse field and model domain geometry from `IceBlock`
    2. Initializes, drives, and tracks all crevasses within domain as
        model runs through time and crevasses are advected downglacier


    Parameters
    ----------
    geometry : obj
        geometry class object for model domain
    fracture_toughness : int
        fracture toughness of ice in Pa m^0.5
    crev_spacing : int
        spacing of crevasses within crevasse field in meters.
    sigmaT0 : float, int, optional
        maximum far field tensile stress, amplitude of the stress field
        iceblock is advected through, by default 120e3 Pa m^0.5
    wpa : float, int, optional
        width of positive strain area in meters that ice advects
        through. Defaults to 1, 500 m. Positive strain area begins
        at up-glacier edge of ice block at - length/x[0], increasing
        towards downglacier end of IceBlock.
    PFA_depth : float, int, optional
        meters penetration required to access the perrenial firn
        aquifer, by default None
    include_creep : bool, optional
        consider creep closure in model, by default False
        NOTE: creep not yet supported by model due to data input
        requirements
    blunt : bool, optional
        whether to blunt the stresses on individual crevasses, used
        with creep calculation and sets sigmaTcrev = 0 for
        interior crevasses within crevasse field, by default False
    never_closed : bool, optional
        always allow melt into the crevasse regardless of
        near-surface pintching. This only affects the Qin calc.
        by default True.
    water_compressive : bool, optional
        whether to allow water into crevasse when the longitudinal
        stress on the crevasse is negative(compressive stress regeime).
        If false, don't allow water into crevasse if in a
        compressive regeime. This affects the Qin calculation.
        by default False


    Attributes
    ----------
    crev_count : int
        number of crevasses in crevasse field
    xcoords : list of tuples
        x-coordinates of each crevasse in field with current depths
    advected_distance: list of floats
        crevasse locations in meters from upglacier boundary of iceblock
    ice_softness :

    """

    def __init__(self,
                 geometry,
                 fracture_toughness,
                 virtualblue,
                 comp_options,
                 sigmaT0=120e3,
                 wpa=1500,
                 PFA_depth=None,
                 ):

        # model geometry and domian management
        self.geometry = geometry
        self.comp_options = comp_options
        self.t = 0

        # define stress field
        self.stress_field = StressField(sigmaT0, wpa, self.comp_options.blunt)

        # potential refreezing rate to use for new crevasses
        self.virtualblue0 = self.deconvolve_refreezing(virtualblue)

        # ice properties
        self.fracture_toughness = fracture_toughness
        self.ice_softness = 1e8  # how soft is the ice

        # crevasses
        self.xcoords = []

        self.crevasses = self.create_crevasse()

        # self.crev_locs = [(-self.geometry.length, -0.1)]
        # self.crev_count = len(self.crev_locs)  # self.crevasse_list()

        self.Qin = 0
        self.PFA_depth = PFA_depth

        self.crev_instances = Crevasse.instances

    @property
    def advected_distance(self) -> list[float]:
        """Crevasse distance from upglacier IceBlock edge in meters"""
        return [-(self.geometry.length-abs(x)) for x in self.xcoords]

    @property
    def crev_info(self):

    def deconvolve_refreezing(self, vb_tuple: tuple[list, list]):
        left, right = vb_tuple
        return left[0], right[0]

    def update_virtualblue(self, vb_tuple: tuple[list, list]):
        """update virtualblue with recalculated values"""
        lhs, rhs = vb_tuple
        for idx, crev in enumerate(self.crev_instances):
            crev.virtualblue_left = lhs[idx]
            crev.virtualblue_right = rhs[idx]

    def evolve_crevasses(self, t, updated_geometry):
        # update model geometry and time
        self.geometry = updated_geometry
        self.t = t

        for idx, crevasse in enumerate(self.crev_instances):
            Qin = self.Qin  # fix this
            sigmaCrev = self.stress_field.sigmaT(self.advected_distance[idx])

            crevasse.evolve(Qin, sigmaCrev)

        #     crevasse.propagate_fracture(Qin,virtual_blue)
        # update crevasse field with propagated crevase info
        # return anything that is needed

    def crevasse_list(self):
        # example
        crev_count = self.crev_instances.length()
        print(f"number of crevasses in crevasse field: {crev_count}")
        return crev_count

    def create_crevasse(self):
        """create and initialize a new crevasse"""
        # new crevasses will be put in their designated location on the
        # upstream edge of the model domain boundary of iceblock

        # print statement

        # initialize via running Crevasse class at init

        Qin = 0  # zero value
        crev = Crevasse(self.geometry.z,
                        self.geometry.dz,
                        self.geometry.ice_thickness,
                        self.geometry.length,
                        Qin,
                        self.ice_softness,
                        round(self.stress_field.sigmaT(0)),
                        self.virtualblue0,
                        self.t,
                        fracture_toughness=self.fracture_toughness)

        self.xcoords.append(-self.geometry.length)
        return crev
