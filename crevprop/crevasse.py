"""Crevasse """
import math as math

import numpy as np
from numpy.lib.function_base import diff
from numpy import sqrt, abs
from numpy.polynomial import Polynomial as P
from scipy.constants import g, pi

from .physical_constants import DENSITY_ICE, DENSITY_WATER, POISSONS_RATIO

# STRESS INTENSITY FACTOR
# For a fracture to propagate
#        KI >= KIC
# stress @ crev tip must >= fracture toughness of ice
# where KI is the stress intensity factor
# which describes the stresses at the fracture's tip
# the material's fracture toughness (KIC)
# Syntax from van der Veen
# dw = depth to water ( or water depth) = distance from ice surface to
#    the top of the water column within crevasee
# d = crevasse depth = crevasse depth below ice surface
# b = water height above crevasse tip
# dw = d - b

# 1. Find crack geometry and shape given water input(R, b, Nyrs) and
#   background stress(sigmaT: + compression, - tensile) and physical
#    constants(poissons ratio, shear modulus)

# Takes into account
# 1. Elastic opening(based on Krawczynski 2009)
# 2. Viscous closure(based on Lilien Elmer results)
# 3. Refreezing rate(diffusion and temperature gradient at sidewalls)


class CrevasseField:
    """Container for all crevasses within model domain

    Primary roles:
    1. Gets crevasse field and model domain geometry from `IceBlock` 
    2. Initializes, drives, and tracks all crevasses within domain as 
        model runs through time and crevasses are advected downglacier


    """

    def __init__(self,
                 x,
                 z,
                 dx,
                 dz,
                 dt,
                 ice_thickness,
                 length,
                 fracture_toughness,
                 crev_spacing,
                 sigmaT=120e3,
                 crev_count=None,
                 crev_locs=None,
                 ):

        self.ice_thickness = ice_thickness
        self.x = x
        self.z = z
        self.dx = dx
        self.dz = dz
        self.dt = dt
        self.length = length

        self.crev_locs = crev_locs if crev_locs else [(-self.length, -0.1)]
        self.crev_spacing = crev_spacing
        self.crev_count = crev_count if crev_count else 1

        # Pa m^0.5 fracture toughness of ice
        self.fracture_toughness = fracture_toughness
        self.ice_softness = 1e8  # how soft is the ice

        # maximum tensile stress (xx) const, amp of stress field that ice mass gets advected through
        self.sigmaT0 = sigmaT  # far-field tensile stress
        self.sigmaCrev = 0  # changes sigma on each crev
        self.wps = 3e3  # width of positive strain area half-wavelength (m)

        # weather to blunt the stresses (sigmaTcrev=0 for interior crevasses) used with creep calc
        self.blunt = False
        self.include_creep = True  # use creep model or not
        # always allow melt in, regardless of near-surface pinching only affects Vin calc
        self.never_closed = True

        # allow water when longitundinal stress is negative (compressive stress regeime)
        # don't allow water in to crev if in compressive regeime -- affects Qin calc
        self.water_compressive = False
        self.PFA_depth = 30  # meters penetration required to access perrenial firn aquifer


class Crevasse:
    """Crevasse formed considering elastic, creep, and refreezing



    K_I - Stress intensity factor (mode I crack opening)
    K_IC - Fracture toughness of Ice

    NOTE A crevasse propagates when K_I = K_IC




    W(z) = E(z) + C(z) - F(z)
    Crevasse shape = Elastic deformation + creep (viscous) deformation
                        - refreezing

    Model Components
    1. Elastic deformation - linear elastic fracture mechanics 
    2. Viscous Creep deformation
    3. Refreezing


    accessible/returnable valuables for each timestep

    crevasse depth
    Vcrev
    Vwater

    Vfrz
    refreezing component of change in crevasse profile (fDiff)

    wall displacement profiles
    water height (dw)



    """

    def __init__(self,
                 z,
                 dz,
                 ice_thickness,
                 Qin,
                 ice_softness,
                 sigmaCrev,
                 ice_density=917,
                 fracture_toughness=100e3,
                 include_creep=True,
                 never_closed=False
                 ):
        """

        Parameters
        ----------
        Qin : _type_
            _description_
        ice_softness : float
            Shear modulus of ice in units of GPa
            accepted values range from 0.07-3.9 GPa


        """
        self.z = z
        self.dz = z
        self.ice_thickness = ice_thickness

        self.fracture_toughness = fracture_toughness
        self.mu = ice_softness
        # sigmaCrev = stress applied to/felt by crevasse
        self.sigmaCrev = sigmaCrev
        self.ice_density = ice_density

        self.depth = 0.1
        self.volume = 1e-4

        # water-filled crevasse
        self.Qin = Qin
        self.Vmelt = 0
        self.Vpfa = 0
        self.Vwater = 1e-4
        # depth = distance from ice surface to water surface in crevasse
        self.water_depth = 0
        # height = height of water above crevasse tip
        self.water_height = 0
        self.flotation_depth = (1-self.ice_density/1000) * self.ice_thickness

        # self.water_depth = self.depth - self.calc_water_height()

        # volume of water refrozen in crevasse, curent timestep
        self.Vfrz = 0

        # volume of water refrozen in crevasse, prev timestep
        self.Vfrz_prev = 0

        # crevasse wall displacement D(z)
        self.walls = np.zeros(len(self.z))

        # optional
        self.include_creep = include_creep
        self.never_closed = never_closed
        self.closed = False

        self.FTHF = False  # full thickness hydrofracture achieved?

        self.alpha = self.calc_alpha(mode=1)

        # tolerance for viscoelastic crevasse extension
        self.ztol = 1e-3  # meters depth tolerance for viscoelastic crevasse extension
        # volume error fractional ratio (Vwater-Vcrev)/Vwater
        self.voltol = 1e-3

    # three crevasse processes considered here

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


        Returns
        -------

        """
        setattr(self, 'sigmaCrev', sigmaCrev)
        Vwater = Qin

        if Vwater > 1e-4 & self.depth < (self.ice_thickness - 10):
            self.crevmorph(Qin)

        elif self.depth >= (self.ice_thickness - 10):
            # do nothing fracture mechanics related but set water level
            # to floation to best approximate a moulin
            dw = (1-DENSITY_ICE/DENSITY_WATER)*self.depth
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

    def crevmorph(self, Qin):
        """
        find crevasse shape givin water input and background stress


        this function adds elastic opening, creep closure, and 
        refreezing to make a crevasse. For a given 

        """
        # required initializations/adjustments

        # interval used in interval splitting to test different crevasse
        # depths. This comes out of the elastic equation as if they are
        # the only forces acting on crevasse during fracture.
        # Z_elastic > true crev depth
        Z_elastic = max(self.depth, 0.1)
        dz = 1
        dy = 0.01  # z spacing resolution to use if crevasse is shallow

        Vwater = Qin
        Vcrev = 1e-15  # init crev volume to something very small

        # 1. determine if the crevasse will grow in this timestep
        #    crevasse will grow if Qin (water input/volume) > crevasse volume

        # conditionals: did the crevasse creep or freeze closed?

        # Begin loop on matching Vwater = Vcrevase (solving for the crevasse
        # volume required to support the input water)
        # change crevasse volume to something very small

        # loop depends on tolerance values
        while abs(Vwater-Vcrev)/Vwater > self.voltol & dz > self.ztol:

            y = np.arange(-Z_elastic, dy, dy)

            # 1. assign water depth so that KI=KIC
            water_depth = self.calc_water_depth(Z_elastic)

        pass

    def elastic_opening(self):
        """Linear elastic fracture mechanics for crevasse opening




        """

        #
        dy = 0.01  # vertical spacing to use for shallow crevasses

        pass

    def creep_closing(self):
        pass

    def refreezing(self):
        pass

    # everything below are class methods added from fracture.py
    # NOTE: all describe linear elastic fracture mechanics

    def elastic_displacement(self,
                             Z,
                             water_depth,
                             crevasse_depth,
                             has_water=True
                             ):
        """elastic crevasse wall displacement from applied stress sigmaT.



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
        sigma_A = self.applied_stress(self.sigma_T, crevasse_depth,
                                      water_depth, has_water=has_water)

        # define constant to advoid repeated terms in D equation
        c1 = (2*self.alpha)/(self.mu*pi)

        # take supset of depth array to avoide dividing by zero at crevasse tip
        z = Z[Z > -crevasse_depth]

        # Wall displacement D(z) for a water-free crevasse
        D = (c1 * pi * sigma_A * diff_squares(crevasse_depth, z)
             + c1 * self.ice_density * g * crevasse_depth * diff_squares(
                 crevasse_depth, z)
             - c1 * self.ice_density * g * z ** 2 * 0.5 * np.log(
                 sum_over_diff(crevasse_depth, diff_squares(crevasse_depth, z))))

        # Add 4 extra terms to D(z) for the addition of water to crevasse
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
        Dz[-(z.size+1)] = 0
        Dz[Dz < -crevasse_depth] = np.nan

        return Dz

    def applied_stress(self, sigma_T, crevasse_depth, water_depth, has_water=True):
        sigma_A = sigma_T - (2 * self.ice_density * g * crevasse_depth)/pi
        if has_water or water_depth:
            sigma_A = (sigma_A
                       - DENSITY_WATER * g * water_depth
                       + (2/pi) * DENSITY_WATER * g * water_depth *
                       np.arcsin(water_depth/crevasse_depth)
                       + (2 * DENSITY_WATER * g * (
                           crevasse_depth**2 - water_depth**2)**(.5)) / pi
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
        """

        if crevasse_depth >= 30:
            water_height = self.calc_water_height(crevasse_depth)
        else:
            d1 = 30
            d2 = 40
            b1 = self.calc_water_height(d1)
            b2 = self.calc_water_height(d2)
            water_height = b1 + (b2-b1)/(d2-d1)*(crevasse_depth - d1)

        return max(0, crevasse_depth - water_height)

    def calc_water_height(self, crevasse_depth):
        """calc water high in crevasse using Hooke text book formulation

        LEFM

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
                + 0.683 * self.ice_density * g *
                sqrt(pi) * (crevasse_depth ** 1.5)
            )
            / (0.683 * DENSITY_WATER * g * sqrt(pi))
        ) ** (2 / 3)

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
        Rxx : 
            tensile resistive stress in Pa
        crevasse_depth : float
            crevasse depth below ice surface in m
        ice_thickness : float
            ice thickness in meters

        Returns
        -------
            stress intensity factor's tensile component
        """
        return self.F(crevasse_depth, self.ice_thickness
                      ) * self.sigmaCrev * sqrt(pi * crevasse_depth)

    def F(self, crevasse_depth, use_approx=False):
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
        return 1.12 if use_approx else p(crevasse_depth / self.ice_thickness)

    # def sigma(self):
    #     return (self.sigma_T - (2 * DENSITY_ICE * g * self.depth) / pi
    #             - DENSITY_WATER * g * self.water_depth
    #             + (2/pi)*DENSITY_WATER * g * self.water_depth * math.asin(
    #             self.water_depth/self.depth)
    #             + ((2*DENSITY_WATER*g) / pi) * math.sqrt(
    #             self.depth ** 2 - self.water_depth ** 2)
    #             )

    # def crev_morph(self):
    #     """geometry solver
    #     """
    #     pass

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
        idx = np.abs(z+water_depth).argmin()+1

        volume = np.abs(np.trapz(Dleft[:idx], z[:idx])) \
            + np.abs(np.trapz(Dright[:idx], z[:idx]))
        return volume

    def calc_alpha(edge_dislocation=True, mode=1):
        """define alpha from dislocation type and mode of crack opening

        alpha

        Parameters
        ----------
        edge_dislocation : bool, optional
            consider edge dislocaitons only if true, if false consider screw 
            dislocations, by default True

        crack_opening_mode : int, str, optional
            mode of crack opening to use, by default 1

        Returns
        -------
        _type_
            _description_
        """
        if edge_dislocation or mode in [1, 2, 'I', 'II', 'i', 'ii']:
            alpha = 1 - POISSONS_RATIO
        elif not edge_dislocation or mode in [3, 'iii', 'III']:
            alpha = 1 - POISSONS_RATIO
        else:
            print(f'incorrect function inputs, assuming edge dislocation alpha=1-v')
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

    # Igore everything below - -- temporary notes from matlab script

    # D10 - crevasse position on left
    # D20 - crevasse position on right
    # y0 - z(vertical)
    # dw0 - water depth
    # Z0 - crevasse depth that the elastic module has suggested - we will
    #        integrate on this value to refine it

    # KIC - stress intensity factor
    # kk - crevasse counter = - for plotting
    # PFAaccessed - for firn aquifer(Qin) fractured to depth
    #        have reached the aquifer depth? bool - prints info is the only use here
    # n - time step for printing info
    # dtScale - how much bigger is the time steps for the thermal model

    # Returns
    # -------
    #    Z - interval used in interval splitting to test different crevasse depths.
    #        comes out of elastic equation if elastic is the only force acting
    #         during fracture.
    #            Z will be larger than Ztrue or correct - rename to Z_elastic for similar
    #             since it is only taking into consideration elastic fracture mechanism
    #     Ztrue - final depth - rename
    #     dw - water depth
    #     D - mean of d1 and d2
    #     D1 - left crevasse profile against zgrid
    #     D2 - right crevasse profile against zgrid
    #     Fdiff1 - freezing component of change in volume/profile
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
