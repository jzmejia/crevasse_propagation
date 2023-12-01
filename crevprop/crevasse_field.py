import math as math
import numpy as np

from dataclasses import dataclass
# from typing import Any, Tuple


from .crevasse import Crevasse
from .physical_constants import SECONDS_IN_YEAR, SECONDS_IN_DAY


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
        """calculate tensile stress x meters from upstream boundary."""
        return self.sigmaT0 * np.cos(-np.pi/self.wpa*x)


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
    mu :

    """

    def __init__(self,
                 geometry,
                 fracture_toughness,
                 virtualblue,
                 comp_options,
                 ice_density,
                 sigmaT0=120e3,
                 wpa=1500,
                 PFA_depth=None,
                 creep_table=None
                 ):

        # model geometry and domian management
        self.geometry = geometry
        self.comp_options = comp_options
        self.t = 0
        self.ice_density = ice_density

        # define stress field
        self.stress_field = StressField(sigmaT0, wpa, self.comp_options.blunt)

        # potential refreezing rate to use for new crevasses
        self.virtualblue0 = self.deconvolve_refreezing(virtualblue)

        # ice properties
        self.fracture_toughness = fracture_toughness
        self.mu = 1e8  # 

        # crevasses
        self.xcoords = []
        
        # initialize creep 
        self.creep = self.creep_init(creep_table)

        self.crevasses = self.create_crevasse()

        # self.crev_locs = [(-self.geometry.length, -0.1)]
        # self.crev_count = len(self.crev_locs)  # self.crevasse_list()

        # Qin for timestep using annual value of 10000m^2/year 
        # for 0.5 day timestep we are inputting 13.7m^2 at each timestep
        # or 27.4m^2/day
        self.Qin = round(10000/SECONDS_IN_YEAR*self.geometry.dt,1)
        self.PFA_depth = PFA_depth

        self.crev_instances = Crevasse.instances
        
        
        
        # initialize creep
        # 1. accept in data
        # 2. reduce data set with model params 
        # (sigma range) and length of time to run model
        
    @property
    def advected_distance(self):
        """Crevasse distance from upglacier IceBlock edge in meters"""
        return [-(self.geometry.length-abs(x)) for x in self.xcoords]

    @property
    def crev_info(self):
        """return a list of properties for all crevasses in field
        list[tuple[float, float, float]]
        Returns
        -------
        crev_info: list[tuple[float,float,float]]
            tuple of crevasse properties
            xcoord : xcoordinate of crevasse (location) (m)
            depth : crevasse depth in meters 
            NOTE: check val is negative
            water_depth : water depth below ice surface in meters
        """
        crev_info = []
        for crev in self.crev_instances:
            crev_info.append((crev.xcoord, crev.depth, crev.water_depth))
        print("xcoord, depth, water_depth")
        return crev_info

    @property
    def crev_num(self) -> int:
        "number of crevasses in crevasse field"
        return len(self.crev_instances)

    def deconvolve_refreezing(self, vb_tuple):
        left, right = vb_tuple
        return left[0], right[0]

    def update_virtualblue(self, vb_tuple):
        """update virtualblue with recalculated values
        
        : tuple[list, list]
        """
        lhs, rhs = vb_tuple
        for idx, crev in enumerate(self.crev_instances):
            crev.virtualblue_left = lhs[idx]
            crev.virtualblue_right = rhs[idx]

    def evolve_crevasses(self, t, updated_geometry):
        # update model geometry and time
        self.geometry = updated_geometry
        self.t = t

        # calculate stress and Qin for each crevasse and evolve
        for idx, crevasse in enumerate(self.crev_instances):
            Qin = self.Qin  # fix this
            sigmaCrev = self.stress_field.sigmaT(self.advected_distance[idx])

            crevasse.evolve(Qin, sigmaCrev, t)

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
                        -self.geometry.length,
                        self.geometry.dt,
                        Qin,
                        self.mu,
                        round(self.stress_field.sigmaT(0)),
                        self.virtualblue0,
                        self.t,
                        ice_density=self.ice_density,
                        fracture_toughness=self.fracture_toughness,
                        creep_table=self.creep
                        )

        self.xcoords.append(-self.geometry.length)
        return crev
    
    def creep_init(self, data):
        df1 = data
        min_sigma=self.stress_field.sigmaT(
            self.geometry.u_surf*SECONDS_IN_YEAR
            *self.geometry.num_years)/1e3
        max_sigma=self.stress_field.sigmaT0/1e3
        sigma = inclusive_slice(df1.columns.get_level_values(
            "sigma").drop_duplicates().to_numpy(),min_sigma,max_sigma)
        yrs = inclusive_slice(df1.columns.get_level_values(
            "years").drop_duplicates().to_numpy(),0,self.geometry.num_years)
        df = df1.loc(axis=1)[yrs,sigma,:]
        return df
        

def inclusive_slice(a, a_min, a_max, pad=None):
    """Array subset with values on or outside of given range
    
    Given an interval, the array is clipped to the closest values
    corresponding to the interval edges such that the resulting
    array has the shortest length while encompassing the entire
    value range given by a_min and a_max.
    
    No check is performed to ensure ``a_min < a_max``
    
    Parameters
    ----------
    a : array_like
        Array containing elements to clip.
    a_min, a_max : array_like or None
        Minimum and maximum value. If ``None``, clipping is not 
        performed on the corresponding edge. 
        Only one of `a_min` and `a_max` may be ``None``. 
        Both are broadcast against `a`.
    pad : int or None, optional
        Include additional a values on either side of range.
        Defaults to None.
    
    Returns
    -------
    clipped_array : ndarray
        An array with elements of `a` corresponding to the 
        inclusive range of `a_min` to `a_max`
    
    Examples
    --------
    >>> a = np.array([0,30,60,90,120,150,180])
    >>> inclusive_slice(a,100,120)
    array([90,120])
    >>> inclusive_slice(a,40,61.7)
    array([30,60,90])
    
    """
    # account for optional padding of window
    i_min = -1 if pad is None else -1-pad
    i_max = 0 if pad is None else pad
    return a[np.argwhere(a<=a_min).item(i_min):
        np.argwhere(a>=a_max).item(i_max)+1]