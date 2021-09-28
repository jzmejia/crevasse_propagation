from .physical_constants import DENSITY_ICE, DENSITY_WATER, POISSONS_RATIO, g, pi
import numpy as np
from numpy.polynomial import Polynomial as P
import math as math


# STRESS INTENSITY FACTOR
# For a fracture to propagate KI (the stress intensity factor)
# which describes the stresses at the fracture's tip must reach
# the material's fracture toughness (KIC)


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
    p = P([1.12, -0.23, 10.55, -21.72, 30.39])
    return p(crevasse_depth/ice_thickness)*Rxx*np.sqrt(pi*crevasse_depth)


def water_hight(Rxx,
                fracture_toughness: float,
                crevasse_depth: float,
                ice_thickness: float,
                ice_density=DENSITY_ICE
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
    return ((fracture_toughness
            - tensile_stress(Rxx, crevasse_depth, ice_thickness)
            + 0.683 * ice_density * g * crevasse_depth**1.5)
            / (0.683 * DENSITY_WATER * g))**(2/3)


def water_height_to_depth(water_height, crevasse_depth):
    """convert water level above crack tip to depth below ice surface

    Args:
        water_height ([float]): water height above crack tip (m)
        crevasse_depth ([float]): crack tip depth below surface (m)

    Returns:
        water depth: water level depth below ice surface (m)
    """
    return crevasse_depth - water_height


def crevasse_volume():
    pass


def B():
    pass


def wall_displacement(y,
                      longitudinal_stress,
                      crevasse_depth,
                      water_depth,
                      shear_modulus,
                      ):

    # c = 2*alpha/mu = 2*(1-poisson's ratio)/shear_modulus = 1.4/shear_mod
    c = (2 * (1 - POISSONS_RATIO)) / (shear_modulus*pi)
    # for conciseness
    z = crevasse_depth
    d = water_depth
    term1 = sigma(longitudinal_stress, z, d) * rds(z, y) * pi
    D = c * (term1
             + DENSITY_ICE*g*z * rds(z, y)
             - DENSITY_WATER*g * rds(z, d) * rds(z, y)
             - 0.5*DENSITY_ICE*g*y**2 * np.log((z+rds(z, y))/(z-rds(z, y)))
             + 0.5*DENSITY_WATER*g*(y**2-d**2)*np.log(math.abs((rds(z, d)+rds(z, y))
                                                               / (rds(z, d)-rds(z, y))))
             - DENSITY_WATER*g*d*y * np.log(abs((d*rds(z, y)+y*rds(z, d))
                                                / (d*rds(z, y)-y*rds(z, d))))
             + DENSITY_WATER*g*d**2 * np.log(abs((rds(z, y)+rds(z, d))
                                                 / (rds(z, y)-rds(z, d))))
             )
    return D


def sigma(longitudinal_stress,
          crevasse_depth,
          water_depth,
          ):
    return (longitudinal_stress
            - (2*DENSITY_ICE*g*crevasse_depth) / pi
            - DENSITY_WATER*g*water_depth
            + (2/pi)*DENSITY_WATER*g*water_depth *
            math.asin(water_depth/crevasse_depth)
            + ((2*DENSITY_WATER*g)/pi) *
            math.sqrt(crevasse_depth**2-water_depth**2)
            )


def rds(x, y):
    """return the square root of the difference of squares

    sqrt(x**2 - y**2)

    Args:
        x ([type]): [description]
        y ([type]): [description]

    Returns:
        [type]: [description]
    """
    return math.sqrt(x**2-y**2)
