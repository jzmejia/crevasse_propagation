from .physical_constants import DENSITY_ICE, DENSITY_WATER, FRACTURE_TOUGHNESS, POISSONS_RATIO, g, pi
import numpy as np
from numpy import sqrt
from numpy.polynomial import Polynomial as P
import math as math


# STRESS INTENSITY FACTOR
# For a fracture to propagate KI (the stress intensity factor)
# which describes the stresses at the fracture's tip must reach
# the material's fracture toughness (KIC)


def F(crevasse_depth, ice_thickness):
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

    Returns:
        F(lambda) float : stress intensity correction factor


    """
    p = P([1.12, -0.23, 10.55, -21.72, 30.39])
    return p(crevasse_depth / ice_thickness)


def tensile_stress(Rxx, crevasse_depth, ice_thickness):
    """[summary]

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
    return F(crevasse_depth, ice_thickness) * Rxx * sqrt(pi * crevasse_depth)


def water_height(
    Rxx,
    fracture_toughness: float,
    crevasse_depth: float,
    ice_thickness: float,
    ice_density=DENSITY_ICE,
):
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
            fracture_toughness
            - tensile_stress(Rxx, crevasse_depth, ice_thickness)
            + 0.683 * ice_density * g * sqrt(pi) * crevasse_depth ** 1.5
        )
        / (0.683 * DENSITY_WATER * g * sqrt(pi))
    ) ** (2 / 3)


def water_depth(Rxx,
                crevasse_depth,
                ice_thickness,
                fracture_toughness=FRACTURE_TOUGHNESS,
                ice_density=DENSITY_ICE
                ):
    return crevasse_depth - water_height(Rxx, fracture_toughness,
                                         crevasse_depth, ice_thickness, ice_density)


def sigma_A(has_water=False):
    pass


def D(y,
      ice_density,
      sigma_T,
      mu,
      crevasse_depth,
      water_surface,
      alpha,
      has_water=True
      ):
    # define D and alpha for a water-free crevasse
    sigma_A = sigma_T - (2 * ice_density*g*crevasse_depth)/pi
    D = ((2*alpha*sigma_A)/mu * sqrt(crevasse_depth**2-y**2)
         + ((2*alpha*ice_density*g)/(pi*mu) *
         crevasse_depth*sqrt(crevasse_depth**2-y**2))
         - ((2*alpha*ice_density*g*y**2)/(2*pi*G)*np.log(
             (crevasse_depth+sqrt(crevasse_depth**2-y**2))
             / (crevasse_depth+sqrt(crevasse_depth**2-y**2))))
         )
    if has_water:
        sigma_A = (sigma_A - DENSITY_WATER*g*water_surface
                   + (2/pi)*DENSITY_WATER*g*water_surface *
                   np.arcsin(water_surface/crevasse_depth)
                   + (2*DENSITY_WATER*g*(crevasse_depth**2-water_surface**2)**0.5)/pi
                   )

    return


def integrate_b11(x, x_, c):
    """solve integral of the form b11 from Weertman 1996

    I1  = integral of x'dx'/(x^2-x'^2)sqrt(c^2-x'^2)
        = 1 / 2*sqrt(c^2-x^2) ln|a+b/b-a| for x^2<c^2 
            where a = sqrt(c^2-x^2) and b is sqrt(c^2-x'2)

    Args:
        x ([type]): [description]
        x2 ([type]): [description]
        c ([type]): [description]
    """


def sigma(
    sigma_T, crevasse_depth, water_depth,
):
    return (
        sigma_T
        - (2 * DENSITY_ICE * g * crevasse_depth) / pi
        - DENSITY_WATER * g * water_depth
        + (2/pi)*DENSITY_WATER * g * water_depth * math.asin(
            water_depth/crevasse_depth)
        + ((2*DENSITY_WATER*g) / pi) * math.sqrt(
            crevasse_depth ** 2 - water_depth ** 2)
    )
