"""

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

STRESS INTENSITY FACTOR
For a fracture to propagate
       KI >= KIC
stress @ crev tip must >= fracture toughness of ice
where KI is the stress intensity factor
which describes the stresses at the fracture's tip
the material's fracture toughness (KIC)
STRESS INTENSITY FACTOR
For a fracture to propagate
       KI >= KIC
stress @ crev tip must >= fracture toughness of ice
where KI is the stress intensity factor
which describes the stresses at the fracture's tip
the material's fracture toughness (KIC)
Syntax from van der Veen
dw = depth to water ( or water depth) = distance from ice surface to
   the top of the water column within crevasee
d = crevasse depth = crevasse depth below ice surface
b = water height above crevasse tip
dw = d - b

1. Find crack geometry and shape given water input(R, b, Nyrs) and
  background stress(sigmaT: + compression, - tensile) and physical
   constants(poissons ratio, shear modulus)

Takes into account
1. Elastic opening(based on Krawczynski 2009)
2. Viscous closure(based on Lilien Elmer results)
3. Refreezing rate(diffusion and temperature gradient at sidewalls)

"""
import math as math
import numpy as np
from numpy import sqrt, abs
from numpy.polynomial import Polynomial as P
from scipy.constants import g, pi
from typing import Union, Tuple, List

from .physical_constants import DENSITY_ICE, DENSITY_WATER, POISSONS_RATIO, SECONDS_IN_DAY

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
        constant variable with site. 0.0165 m^-1 < C < 0.0314 m^-1. 
        Defaults to 0.02 m^-1
    ice_density : (float, optional)
        Defaults to 917.
    snow_density : (float, optional):
        Defaults to 350.

    Returns
    -------
        snow density at depth
    """
    return ice_density - (ice_density - snow_density) * np.exp(-C*depth)


class Crevasse():
    """Crevasse formed considering elastic, creep, and refreezing

    K_I - Stress intensity factor (mode I crack opening)
    K_IC - Fracture toughness of Ice
    NOTE: A crevasse propagates when K_I = K_IC

    W(z) = E(z) + C(z) - F(z)
    Crevasse shape = Elastic deformation
                    + creep (viscous) deformation
                    - refreezing

    Model Components
    1. Elastic deformation - linear elastic fracture mechanics
    2. Viscous Creep deformation
    3. Refreezing

    Parameters
    ----------
    z: np.array
        depth-vector corresponding to ice block thickness (m)
    dz: int
        vertical sampling resolution of ice block (m)
    ice_thickness: int
        thickness of ice block in meters
    x: np.array
        x-coordinates of model geometry/domain. 
    Qin: Union[int, float]
        meltwater input to crevasse
    ice_softness: int
        ice softness (mu)
    sigmaCrev: float
        applied stress on crevasse
    virblue: Tuple
        potential refreezing along crevasse walls (left,right)
    t0: int
        current model time in seconds when class initialized.
    ice_density: int, optional
        _description_, by default 917
    fracture_toughness: int, optional
        fracture toughness of ice in Pa m^0.5, by default 100e3
    include_creep: bool, optional
        consider creep closure in model, by default False
        NOTE: creep not yet supported by model due to data input 
        requirements, by default False
    never_closed: bool, optional
        always allow melt into the crevasse regardless of near-surface 
        pintching. This only affects the Qin calc., by default False

    Attributes
    ----------
    mu : int
        ice softness
    depth : float
        Crevasse depth in meters below ice surface (positive)
    volume : float
        Crevasse volume (m^2)
    Vwater : float
        Volume of water within crevasse (m^2)
    water_depth : float
        Depth of water within crevasse in meters from ice surface
    Vfrz : float
        Volume refrozen within crevasse
    virtualblue_left : np.array
        Potential refreezing over timestep dt along the left side of 
        crevasse wall for entire ice thickness. 
    virtualblue_right : np.array
        Potential refreezing over timestep dt along the right side of 
        crevasse wall for entire ice thickness. 
    closed : bool, default False
        Has the crevasse closed due to creep?
    FTHF : bool
        Has the crevasse fractured the entire ice thickness?
    alpha : float
        alpha computed for dislocation type, defaults to 
    """
    instances = []

    def __init__(self,
                 z: np.array,
                 dz: int,
                 ice_thickness: int,
                 x: float,
                 Qin: Union[int, float],
                 ice_softness: int,
                 sigmaCrev: float,
                 virblue: Tuple,
                 t0: int,
                 ice_density=917,
                 fracture_toughness=100e3,
                 include_creep=False,
                 never_closed=False
                 ):
        self.z = z
        self.dz = dz
        self.ice_thickness = ice_thickness
        self.xcoord = x
        self.t0 = t0
        self.fracture_toughness = fracture_toughness
        self.mu = ice_softness
        self.ice_density = ice_density

        # self.flotation_depth = (1-self.ice_density/1000) * self.ice_thickness
        self.sigmaCrev = sigmaCrev

        # dynamic, set initial crev conditions
        self.depth = 0.1
        self.volume = 1e-4

        # water-filled crevasse
        self.Qin = Qin
        # self.Vmelt = 0
        # self.Vpfa = 0

        self.Vwater = 1e-4
        # depth = distance from ice surface to water surface in crevasse
        self.water_depth = 0
        # height = height of water above crevasse tip
        self.water_height = 0
        # TODO: make depth or height a computed property

        # self.water_depth = self.depth - self.calc_water_height()

        # volume of water refrozen in crevasse, curent timestep
        self.Vfrz = 0
        self.virtualblue_left, self.virtualblue_right = virblue

        # volume of water refrozen in crevasse, prev timestep
        self.Vfrz_prev = 0

        # crevasse wall displacement D(z)
        self.walls = np.zeros(len(self.z))
        self.left_wall = -self.walls
        self.right_wall = self.walls

        # optional
        self.include_creep = include_creep  # note: reqs data
        self.never_closed = never_closed
        self.closed = False
        self.approximate_F = False

        self.FTHF = False  # full thickness hydrofracture achieved?
        self.alpha = self.calc_alpha(mode=1)

        # tolerance for viscoelastic crevasse extension
        self.ztol = 1e-3  # m depth tolerance for
        # volume error fractional ratio (Vwater-Vcrev)/Vwater
        self.voltol = 1e-3

        Crevasse.instances.append(self)

    # def __iter__(self):
    #     return iter(self.instances)

    def age(self, t):
        """How many days since the crevasse formed?"""
        return (t-self.t0)/SECONDS_IN_DAY

    # def set_virtualblue(self, ib_virblue):

    def evolve(self, Qin, sigmaCrev):
        """evolve crevasse for new timestep and inputs

        this function allows the crevasse's shape to evolve in response
        to water inputs and changing applied stress on the crevasse.


        Parameters
        ----------
        Qin : float
            volume of water input to crevasse during a timestep of model
            units of m^2
        sigmaCrev : float
            applied stress on crevasse in Pa
        """
        setattr(self, 'sigmaCrev', sigmaCrev)
        # Vwater = Qin

        if Qin > 1e-4 & self.depth < (self.ice_thickness - 10):
            self.crevmorph(Qin)

        elif self.depth >= (self.ice_thickness - 10):
            # do nothing fracture mechanics related but set water level
            # to floation to best approximate a moulin
            dw = self.flotation_depth(self.depth)
            setattr(self, 'FTHF', True)

            # calculate Vfrz for this timestep
        else:
            # there is no water to fill the crevasse
            dw = 0

            # if old crevasse calculate elastic closure of surface ditch
            # e.g., englacial void near ice surface
            # artificially assign a "depth" of crevasse to highest place
            # where wall profile > 0

        # TODO: after all these calculations you'll need to reset class
        # attrs to updated values

        pass

    def flotation_depth(self, crevasse_depth):
        """water depth required for flotation"""
        return (1-DENSITY_ICE/DENSITY_WATER)*crevasse_depth

    def _growth(self, z, Dleft, Dright, Vwater) -> bool:
        """will the crevasse grow given current water depth and Qin?"""
        return True if self.crevasse_volume(z, self.water_depth, Dleft, Dright
                                            ) > Vwater else False

    def crevmorph(self, Qin):
        """
        find crevasse shape givin water input and background stress


        this function adds elastic opening, creep closure, and 
        refreezing to make a crevasse. For a given 

        Parameters
        ----------
        Qin : float
            Liquid water input to crevasse for timestep dt. 
        """

        Z_elastic = max(self.depth, 0.1)
        dz = 1
        dy = 0.01  # z spacing resolution to use if crevasse is shallow
        # finer z-resolution for crevasse calculations
        y = np.arange(-Z_elastic, dy, dy)

        Vwater = Qin
        Vcrev = 1e-15  # init crev volume to something very small

        # current crevasse wall locations, redefining here to new y
        Dleft0 = np.interp(y, self.z, self.left_wall)
        Dright0 = np.interp(y, self.z, self.right_wall)

        growth = self._growth(y, Dleft0, Dright0, Vwater)

        while abs(Vwater-Vcrev)/Vwater > self.voltol & dz > self.ztol:

            # 1. calc water depth for KI(cracktip) = KIC
            water_depth = self.calc_water_depth(Z_elastic)

            # elastic crack geometry
            # right now filling with nans outside of crev size, do i
            # want to do this if i have to subtract or set to zero then
            # nan later for plotting?
            E0 = self.elastic_displacement(y, self.water_depth, self.depth)
            E = self.elastic_displacement(y, water_depth, Z_elastic)

            # Elastic differential opening
            EDiff = E - E0

            # Apply elastic opening to crevasse walls
            Dleft = np.minimum(Dleft0 - EDiff, np.zeros_like(Dleft0))
            Dright = np.maximum(Dright0 + EDiff, np.zeros_like(Dright0))

            # REFREEZING CONTRIBUTION TO CREVASSE WIDTH
            # 1. convert virtual blue (freezing IF crevasses goes that
            # deep) to actual blue (freezing that actually occurs)

            crev_idx = np.where(np.logical_and(self.z > -Z_elastic,
                                               self.z < -water_depth))

            blueband_left = self.virtualblue_left[crev_idx]
            blueband_right = self.virtualblue_right[crev_idx]

            # Creep Closure - calculated from data for area
            # if self.creep_closing:
            #     CDiff = self.creep_closing()
            # else:
            #     CDiff = np.zeros_like(FDiff_left)

            #

        pass

    def creep_closing(self):
        pass

    def refreezing(self, crevasse_depth, water_depth):
        """Refreezing contribution to crevasse width

        Parameters
        ----------
        water_depth : float
        crevasse_depth : float
        """
        pass

    # everything below are class methods added from fracture.py
    # NOTE: all describe linear elastic fracture mechanics

    def elastic_displacement(self,
                             Z,
                             water_depth,
                             crevasse_depth,
                             has_water=True
                             ):
        """elastic crevasse wall displacement from applied stress sigmaT



        Weertmen's elastic equiation giving shape of crevasse based on 
        material prperties of ice, crevasse depth, water depth etc. 


        Parameters
        ----------
        Z : np.array
            depth array to find crevasse wall locations along
        water_depth : float
            depth from ice surface to water in crevasse in meters
        crevasse_depth : float
            depth of crevasse in m (positive value)
        has_water : bool, optional
            does the crevasse have any water in it? by default True

        Returns
        -------
        D : np.ndarray
            crevasse wall displacement for each depth given in Z array
            units m, values are all positive. 
        """

        # define D and alpha for a water-free crevasse
        sigma_A = self.applied_stress(self.sigmaCrev, crevasse_depth,
                                      water_depth, has_water=has_water)

        # define constant to avoid repeated terms in D equation
        c1 = (2*self.alpha)/(self.mu*pi)

        # take supset of depth array to avoide dividing by zero at
        # crevasse tip
        z = Z[Z > -crevasse_depth]

        # Wall displacement D(z) for a water-free crevasse
        D = (c1 * pi * sigma_A * diff_squares(crevasse_depth, z)
             + c1 * self.ice_density * g * crevasse_depth * diff_squares(
                 crevasse_depth, z)
             - c1 * self.ice_density * g * z ** 2 * 0.5 * np.log(
                 sum_over_diff(crevasse_depth,
                               diff_squares(crevasse_depth, z))))

        # Add 4 extra terms to D(z) for water added to crevasse
        if has_water:
            c1 = c1 * DENSITY_WATER * g
            D = (D
                 - c1 * diff_squares(crevasse_depth, water_depth) *
                 diff_squares(crevasse_depth, z)
                 + (c1 * (z**2 - water_depth**2) * 0.5 * np.log(abs(
                     sum_over_diff(diff_squares(crevasse_depth, water_depth),
                                   diff_squares(crevasse_depth, z)))))
                 - c1 * water_depth * z * np.log(abs(sum_over_diff(
                     water_depth * diff_squares(crevasse_depth, z),
                     z * diff_squares(crevasse_depth, water_depth))))
                 + c1 * water_depth**2 * np.log(abs(sum_over_diff(
                     diff_squares(crevasse_depth, z),
                     diff_squares(crevasse_depth, water_depth))))
                 )

        Dz = np.copy(Z)
        Dz[Dz > -crevasse_depth] = abs(D)
        # Dz[-(z.size+1)] = 0
        Dz[Dz <= -crevasse_depth] = 0  # was np.nan and < only

        return Dz

    def applied_stress(self,
                       sigma_T,
                       crevasse_depth,
                       water_depth,
                       has_water=True
                       ):
        """calculate applied stress Rxx on crevasse walls

        Parameters
        ----------
        sigma_T : float, int
            far field stress
        crevasse_depth : float
            crevasse depth below ice surface in meters
        water_depth : float, int
            water depth below ice surface in meters
        has_water : bool, optional
            is there any water within crevasse? by default True

        Returns
        -------
        applied_stress: float
            stress applied to crevasse walls (Rxx)
        """
        sigma_A = sigma_T - (2 * self.ice_density * g * crevasse_depth)/pi
        if has_water or water_depth:
            sigma_A = (sigma_A
                       - DENSITY_WATER*g*water_depth
                       + (2/pi)*DENSITY_WATER*g*water_depth *
                       np.arcsin(water_depth/crevasse_depth)
                       + (2*DENSITY_WATER*g *
                          (crevasse_depth**2-water_depth**2)**(.5))/pi
                       )
        return sigma_A

    def calc_water_depth(self, crevasse_depth):
        """calculate water depth in crevasse and correct for small crevs

        Apply the Hook formulation described in `.calc_water_height()` 
        and transfrom from height of water column to depth of water
        below ice surface. Also, apply a slight correction for small
        crevasses such that the water height does not overflow the
        crevasse. This correction is applied to crevasses shallower than
        30 m. 



        Parameters
        ----------
        crevasse_depth : float
            crevasse depth in meters. 

        Returns
        -------
        water depth : float
            water depth in meters from the ice surface
        """

        if crevasse_depth >= 30:
            water_height = self.calc_water_height(crevasse_depth)
        else:
            d1 = 30
            d2 = 40
            b1 = self.calc_water_height(d1)
            b2 = self.calc_water_height(d2)
            water_height = b1 + (b2-b1)/(d2-d1)*(crevasse_depth-d1)

        return max(0, crevasse_depth - water_height)

    def calc_water_height(self, crevasse_depth):
        """calc water high in crevasse using Hooke text book formulation

        Linear Elastic Fracture Mechanics

        van der Veen 1998/2007 equation to estimate the net stress
        intensity factor (KI) for mode I crack opening where::

            KI = tensile stress - lithostatic stress + water pressure(1)
            KI = 1.12 * Rxx * sqrt(pi * ice_thickness)               (2)
            - 0.683 * ice_density * g * ice_thickness**1.5
            + 0.683 * water_density * g * water_height**1.5


        because KI = KIC (the `fracture_toughness` of ice) when a crack
        opens, we set KI=KIC in equation 2 and solve for water_height::

            water_height = (( KIC                                    (3)
            - 1.12 * Rxx * sqrt(pi * ice_thickness)
            + 0.683 * ice_density * g * ice_thickness**1.5)
            / 0.683 * water_density * g )**2/3

        **Assumptions**: Rxx constant w/ depth, doesn't account for firn

        Note
        ----
        using the function `tensile_stress()` will calculate the full
        tensile stress term instead of using the approximation of 1.12
        shown in equations 2 and 3. An `if statement` can be added,
        however, `numpy`'s `polynomial` function is quite fast.

        calculation uses class attrs `Crevasse.sigmaCrev` the normal 
        stress responsible for crevasse opening (i.e., the resistive 
        stress R$_{xx}$)

        Parameters
        ----------
        crevasse_depth: float
            crevasse depth below ice surface (m)

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
                - self.tensile_stress(crevasse_depth)
                + 0.683*self.ice_density*g*sqrt(pi) * (crevasse_depth**1.5)
            )
            / (0.683*DENSITY_WATER*g*sqrt(pi))
        ) ** (2/3)

    def tensile_stress(self, crevasse_depth):
        """calculate tensile stress

        LEFM

        Note
        ----
        an approximatioin of the polynomial coefficient can be used
        if the ratio between crevasse_depth and ice thickness < 0.2
        Future work could add an if statement, but should test if the
        full computation with numpy.polynomial.Polynomial is faster than
        the conditional.

        Equation from van der Veen 1998 where the stress intensity 
        factor (K_I)::

            K_I(1) = F(lambda)*Rxx*sqrt(pi*crevasse_depth)
            where lambda = crevasse_depth / ice_thickness and

            F(lambda) = 1.12 - 0.23*lambda + 10.55*lambda**2 
                        - 12.72*lambda**3 + 30.39*lambda**4

        For shallow crevasses::

            F(lambda->0) = 1.12 * Rxx * sqrt(pi*crevasse_depth)


        Parameters
        ----------
        crevasse_depth : float
            crevasse depth below ice surface in m

        Returns
        -------
            stress intensity factor's tensile component
        """
        return self.F(crevasse_depth)*self.sigmaCrev*sqrt(pi*crevasse_depth)

    def F(self, crevasse_depth):
        """Finite ice thickness correction for stress intensity factor

        LEFM

        from van der Veen (1998) equation 6
        F(lambda) where lambda = crevasse depth / ice thickness

        correction accounts for the ~12% increase in the stress 
        intensity factor that accounts for material properties such as 
        the crack tip plastic zone (i.e., area where plastic deformation
        occurs ahead of the crack's tip.).

        Note
        ----
        correction is only used for the tensile stress component of K_I 
        (mode I)

        Parameters
        ----------
        crevasse_depth: float, int
            depth below surface in meters
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
        return 1.12 if self.approximate_F else p(crevasse_depth/self.ice_thickness)

    def crevasse_volume(self, z, water_depth, Dleft, Dright):
        """calculate volume of water filled crevasse

        Parameters
        ----------
            z : np.array
                vertical spacing/coordinates corresponding to crevasse 
                wall profiles Dleft and Dright
            water_depth : float, int
                water depth in crevasse
            Dleft : np.array
                profile of left crevasse wall
            Dright : np.array
                profile of crevasse walls

        Returns
        -------
            volume : float
                two-dimensional crevasse volume in square meters


        NOTE: Adapted from Poinar Matlab Script using matlab's `trapz`
        function. An important difference is that in matlab `trapz(X,Y)`
        for integration using two variables but for the `numpy.trapz()`
        function used here, this is inverted such that `np.trapz(Y,X)`.

        """
        # find index of water depth and only consider that in calc.
        idx = np.abs(z+water_depth).argmin() + 1

        volume = np.abs(np.trapz(Dleft[:idx], z[:idx])) \
            + np.abs(np.trapz(Dright[:idx], z[:idx]))
        return volume

    def calc_alpha(edge_dislocation=True, mode=1):
        """define alpha from dislocation type and mode of crack opening

        alpha   = 1 - v,    for edge dislocations  
                = 1,        for screw discolations 

        where v is Poisson's Ratio which is prescribed in 
        `physical_constants` as 0.3 following Simmons and Wang 1971

        edge discolations imply the crack is either a plane strain shear
        crack (mode II) or a plane strain tensile crack (mode I)
        screw dislocations imply the crack is in an antiplane strain 
        shear crack (mode III crack) 

        Parameters
        ----------
        edge_dislocation : bool, optional
            consider edge dislocaitons only if true, if false consider 
            screw dislocations, by default True

        crack_opening_mode : int, str, optional
            mode of crack opening to use, by default 1

        Returns
        -------
        alpha : float
        """
        if edge_dislocation or mode in [1, 2, 'I', 'II', 'i', 'ii']:
            alpha = 1 - POISSONS_RATIO
        elif not edge_dislocation or mode in [3, 'iii', 'III']:
            alpha = 1
        else:
            print(f'incorrect function inputs, assuming edge '
                  'dislocation alpha=1-v')
            alpha = 1 - POISSONS_RATIO
        return alpha

    # Igore everything below - -- temporary notes from matlab script

    # D10 - crevasse position on left
    # D20 - crevasse position on right
    # y0 - z(vertical)
    # dw0 - water depth
    # Z0 - crevasse depth that the elastic module has suggested
    #        integrate on this value to refine it

    # KIC - stress intensity factor
    # kk - crevasse counter = - for plotting
    # PFAaccessed - for firn aquifer(Qin) fractured to depth
    #        have reached the aquifer depth? bool - prints info is the
    #       only use here
    # n - time step for printing info
    # dtScale - how much bigger is the time steps for the thermal model

    # Returns
    # -------
    #    Z - interval used in interval splitting to test different
    #        crevasse depths.
    #        comes out of elastic equation if elastic is the only force
    #        acting
    #         during fracture.
    #            Z will be larger than Ztrue or correct - rename to
    #            Z_elastic for similar
    #             since it is only taking into consideration elastic
    #             fracture mechanism
    #     Ztrue - final depth - rename
    #     dw - water depth
    #     D - mean of d1 and d2
    #     D1 - left crevasse profile against zgrid
    #     D2 - right crevasse profile against zgrid
    #     Fdiff1 - freezing component of change in volume/profile
    # def sigma(self):
    #     return (self.sigma_T - (2 * DENSITY_ICE * g * self.depth) / pi
    #             - DENSITY_WATER * g * self.water_depth
    #             + (2/pi)*DENSITY_WATER * g * self.water_depth * math.asin(
    #             self.water_depth/self.depth)
    #             + ((2*DENSITY_WATER*g) / pi) * math.sqrt(
    #             self.depth ** 2 - self.water_depth ** 2)
    #             )
