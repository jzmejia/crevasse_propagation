"""
fracture mechanics used in crevasse propagation model

Linear elastic fracture mechanics scheme for crevasse propagation


LEFM based on Weertman's dislocation theory and application to crevasses
with adaptations from van der Veen (1998)


Theory
------
STRESS INTENSITY FACTOR
For a fracture to propagate
       KI >= KIC
stress @ crev tip must >= fracture toughness of ice
where KI is the stress intensity factor
which describes the stresses at the fracture's tip
the material's fracture toughness (KIC)

"""
from numpy.lib.function_base import diff
from .physical_constants import DENSITY_ICE, DENSITY_WATER, FRACTURE_TOUGHNESS, POISSONS_RATIO
import numpy as np
from numpy import sqrt, abs
from numpy.polynomial import Polynomial as P
import math as math
from scipy.constants import g, pi


def F(crevasse_depth, ice_thickness, use_approximation=False):
    """Finite ice thickness correction for the stress intensity factor

    from van der Veen (1998) equation 6

     F(:math:`\lambda`) where :math:`\lambda`=crevasse_depth/ice_thickness

    correction accounts for the ~12% increase in the stress intensity
    factor that accounts for material properties such as the crack
    tip plastic zone (i.e., area where plastic deformation occurs ahead
    of the crack's tip.).

    Notes
    -----
    correction only used for tensile stress component of K_I (mode 1)


    Parameters
    ----------
    crevasse_depth : float, int
        depth below surface in meters
    ice_thickness : float, int
        ice thickness in meters
    use_approximation: bool
        whether to use the shallow crevasse approximation. 
        If ``True`` use 1.12 instead of the polynomial expansion, 
        if ``False`` use full polynomial expansion in calculation. 
        Defaults to ``False``.


    Returns
    -------
    F(lambda) float : stress intensity correction factor

    """
    p = P([1.12, -0.23, 10.55, -21.72, 30.39])
    return 1.12 if use_approximation else p(crevasse_depth / ice_thickness)


def tensile_stress(Rxx, crevasse_depth, ice_thickness):
    """Calculate the stress intensity factor's tensile compoent.

    An approximatioin of the polynomial coefficient can be 
    used if the ratio between ``crevasse_depth`` and ``ice_thickness``
    is less than 0.2. Future work could add an if-statement, 
    but should test if the full computation with 
    ``numpy.polynomial.Polynomial`` is faster than the conditional.

    van der Veen (1998)'s formulation of the stress intensity factor for 
    mode I crack opening (:math:`K_I`):

    .. math:: K_{I}(1) = F(\lambda) R_{xx}  \sqrt{\pi d}

    where :math:`\lambda = d/H` where *d* is crevasse depth, *H* is 
    ice thickness, and 

    .. math:: F(\lambda) = 1.12 - 0.23\lambda + 10.55\lambda^2 
    - 12.72\lambda^3 + 30.39\lambda^4

    For shallow crevasses, 
    :math:`F(\lambda \Rightarrow 0) = 1.12  R_{xx}  \sqrt{\pi d}`

    Parameters
    ----------
    Rxx: 
        far-field stress or tensile resistive stress
    crevasse_depth: float
        crevasse depth below ice surface in meters
    ice_thickness: float
        ice thickness in meters

    Returns
    -------
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
    factor (KI) for mode I crack opening where::

        KI = tensile stress - lithostatic stress + water pressure    (1)
        KI = 1.12 * Rxx * sqrt(pi * ice_thickness)                   (2)
            - 0.683 * ice_density * g * ice_thickness**1.5
            + 0.683 * water_density * g * water_height**1.5

    because KI = KIC (the ``fracture_toughness`` of ice) when a crack
    opens, we set KI=KIC in equation 2 and solve for water_height::

        water_height = (( KIC                                        (3)
                      - 1.12 * Rxx * sqrt(pi * ice_thickness)
                      + 0.683 * ice_density * g * ice_thickness**1.5)
                      / 0.683 * water_density * g )**2/3


    **Assumptions**: Rxx constant with depth, doesn't account for firn


    Notes
    -----
    using the function ``tensile_stress()`` will calculate the full
    tensile stress term instead of using the approximation of 1.12
    shown in equations 2 and 3. An if statement can be added,
    however, numpy's polynomial function is quite fast.


    Parameters
    ----------
    Rxx : float
        Far field tensile stress
    crevasse_depth : float
        crevasse depth below ice surface (m)
    fracture_toughness : float
        fracture toughness of ice, units of MPa*m^1/2
    ice_thickness : float
        ice thickness in meters
    ice_density : float, optional
        Defaults to DENSITY_ICE = 917 kg/m^3.


    Returns
    -------
    water_height : float 
        water height above crevasse bottom (m)
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
    """Convert water height in crevasse to depth below surface in meters


    Parameters
    ----------
    Rxx
    crevasse_depth : float, int
        crevasse depth in meters below surface (positive)
    ice_thickness: float, int
        local ice thickness in metesr
    fracture_toughness: float, int
        fracture toughness of ice
    ice_density : float
        ice density in kg/m^3, defaults to 917 kg/m^3


    Returns
    -------
    : float

    """
    return crevasse_depth - water_height(Rxx, fracture_toughness,
                                         crevasse_depth, ice_thickness,
                                         ice_density)


def sigma(sigma_T, crevasse_depth, water_depth):
    """Calculate sigma

    Parameters
    ----------
    sigma_T : float, int
    crevasse_depth : float, int
        crevasse depth from ice surface (m), positive.
    water_depth : float, int
        depth from ice surface to water surface within crevasse (m).

    """
    return (
        sigma_T
        - (2 * DENSITY_ICE * g * crevasse_depth) / pi
        - DENSITY_WATER * g * water_depth
        + (2/pi)*DENSITY_WATER * g * water_depth * math.asin(
            water_depth/crevasse_depth)
        + ((2*DENSITY_WATER*g) / pi) * math.sqrt(
            crevasse_depth ** 2 - water_depth ** 2)
    )


def applied_stress(traction_stress,
                   crevasse_depth,
                   water_depth,
                   has_water=False):
    """calculated applied stress (sigma_A)

    Parameters
    ----------
    traction_stress
    crevasse_depth : 
        crevasse depth in meters from ice surface
    water_depth : float, int
        distance from ice surface to water column within crevasse in m.
    has_water : bool, optional
        Is there water within the crevasse? Defaults to False.

    Returns
    -------
        sigma_A

    """
    sigma_A = traction_stress - (2*DENSITY_ICE*g*crevasse_depth)/pi
    if has_water:
        sigma_A = (sigma_A
                   - DENSITY_WATER*g*water_depth
                   + (2/pi) * DENSITY_WATER * g * water_depth *
                   np.arcsin(water_depth/crevasse_depth)
                   + (2 * DENSITY_WATER * g * (
                       crevasse_depth**2 - water_depth**2)**(.5)) / pi
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


def elastic_displacement(z,
                         sigma_T,
                         mu,
                         crevasse_depth,
                         water_depth,
                         alpha=(1-POISSONS_RATIO),
                         has_water=True
                         ):
    """calculate elastic crevasse wall displacement from applied stress sigma_T.


    Parameters
    ----------
    z 
    sigma_T : 
    mu : float, int
        ice softness
    crevasse_depth : float, int
        distance between ice surface and crevasse tip in m (positive).
    water_depth : 
        distance between ice and water surface within crevasse (m).
    alpha : tuple, optional
        Defaults to (1-POISSONS_RATIO).
    has_water : bool, optional
        Is there water within the crevasse? Defaults to True.

    """
    # define D and alpha for a water-free crevasse
    sigma_A = applied_stress(sigma_T, crevasse_depth,
                             water_depth, has_water=has_water)
    # define constant to advoid repeated terms in D equation
    c1 = (2*alpha)/(mu*pi)
    D = (c1*pi*sigma_A * diff_squares(crevasse_depth, z)
         + c1*DENSITY_ICE*g*crevasse_depth*diff_squares(crevasse_depth, z)
         - c1*DENSITY_ICE*g*z**2*0.5 *
         np.log(sum_over_diff(crevasse_depth, diff_squares(crevasse_depth, z)))
         )
    if has_water:
        c1 = c1*DENSITY_WATER*g
        D = (D - c1*diff_squares(crevasse_depth,
             water_depth)*diff_squares(crevasse_depth, z)
             + (c1*(z**2-water_depth**2)*0.5*np.log(abs(sum_over_diff(
                 diff_squares(crevasse_depth, water_depth),
                 diff_squares(crevasse_depth, z)))))
             - c1*water_depth*z*np.log(abs(sum_over_diff(
                 water_depth*diff_squares(crevasse_depth, z),
                 z*diff_squares(crevasse_depth, water_depth))))
             + c1*water_depth**2 *
             np.log(abs(sum_over_diff(diff_squares(crevasse_depth, z),
                    diff_squares(crevasse_depth, water_depth))))
             )
    return abs(D)


# math helper functions to simplify the
def sum_over_diff(x, y):
    """calcualte x+y / x-y

    Parameters
    ----------
    x : float, int
    y : float, int

    Returns
    -------
    : float

    """
    return (x+y) / (x-y)


def diff_squares(x, y):
    """calculate the squareroot of the difference of squares
    Parameters
    ----------
    x : float, int
    y : float, int

    Returns
    -------
    : float

    """
    return np.sqrt(x**2 - y**2)


def density_profile(depth, C=0.02, ice_density=917., snow_density=350.):
    """empirical density-depth relationship from Paterson 1994

    Parameters
    ----------
    depth : float, array
        depth below ice surface in m
    C : float, optional
        constant variable with site. Use 0.0165 m^-1 < C < 0.0314 m^-1
        Defaults to 0.02 m^-1
    ice_density : float, optional
        value to use for ice density in kg/m^3. Defaults to 917.
    snow_density : float, optional
        value to use for snow density (fresh). Defaults to 350.

    Returns
    -------
    : float, array 
        snow density at depth

    """
    return ice_density - (ice_density - snow_density) * np.exp(-C*depth)
