"""Crevasse """

from numpy.lib.function_base import diff
from .physical_constants import DENSITY_ICE, DENSITY_WATER, POISSONS_RATIO
import numpy as np
from numpy import sqrt, abs
from numpy.polynomial import Polynomial as P
import math as math
from scipy.constants import g, pi


class Crevasse:
    """Class describing a single crevasse
    """

    def __init__(self, Qin):
        """

        Parameters
        ----------
        Qin : _type_
            _description_
        """
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
        """Finite ice thickness correction for stress intensity factor

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

        **Assumptions**: Rxx constant with depth, 
        sdoesn't account for firn

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
                 + (c1 * (z**2 - self.water_depth**2) * 0.5 * np.log(abs(
                     sum_over_diff(diff_squares(self.depth, self.water_depth),
                                   diff_squares(self.depth, z)))))
                 - c1 * self.water_depth * z * np.log(abs(
                     sum_over_diff(self.water_depth * diff_squares(self.depth, z),
                                   z * diff_squares(self.depth, self.water_depth))))
                 + c1 * self.water_depth**2 * np.log(abs(
                     sum_over_diff(diff_squares(self.depth, z),
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
