"""
(c) 2024 Jessica Mejia

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

from scipy.integrate import trapz
from numpy.lib.function_base import diff
from .physical_constants import (
    DENSITY_ICE,
    DENSITY_WATER,
    FRACTURE_TOUGHNESS,
    POISSONS_RATIO,
)
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
    """Calculate the stress intensity factor's tensile component.

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
            + 0.683 * ice_density * g * sqrt(pi) * crevasse_depth**1.5
        )
        / (0.683 * DENSITY_WATER * g * sqrt(pi))
    ) ** (2 / 3)


def water_depth(
    Rxx,
    crevasse_depth,
    ice_thickness,
    fracture_toughness=FRACTURE_TOUGHNESS,
    ice_density=DENSITY_ICE,
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
    return crevasse_depth - water_height(
        Rxx, fracture_toughness, crevasse_depth, ice_thickness, ice_density
    )


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
        + (2 / pi)
        * DENSITY_WATER
        * g
        * water_depth
        * math.asin(water_depth / crevasse_depth)
        + ((2 * DENSITY_WATER * g) / pi) * math.sqrt(crevasse_depth**2 - water_depth**2)
    )


def applied_stress(
    traction_stress, crevasse_depth, water_depth, has_water=False, density=DENSITY_ICE
):
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
    sigma_A = traction_stress - (2 * density * g * crevasse_depth) / pi
    if has_water:
        sigma_A = (
            sigma_A
            - DENSITY_WATER * g * water_depth
            + (2 / pi)
            * DENSITY_WATER
            * g
            * water_depth
            * np.arcsin(water_depth / crevasse_depth)
            + (2 * DENSITY_WATER * g * (crevasse_depth**2 - water_depth**2) ** (0.5))
            / pi
        )
    return sigma_A


def alpha(dislocation_type="edge", crack_opening_mode=None):
    if dislocation_type == "screw" or crack_opening_mode in [3, "iii", "III"]:
        alpha = 1
    elif dislocation_type == "edge" or crack_opening_mode in [1, 2, "I", "II"]:
        alpha = 1 - POISSONS_RATIO
    else:
        print(f"incorrect function inputs, assuming edge dislocation alpha=1-v")
        alpha = 1 - POISSONS_RATIO
    return alpha


def elastic_displacement(
    z,
    sigma_T,
    mu,
    crevasse_depth,
    water_depth,
    alpha=(1 - POISSONS_RATIO),
    has_water=True,
    density=DENSITY_ICE,
):
    """calculate elastic crevasse wall displacement from applied stress sigma_T.


    Calculates Weertman's D-equation describing crevasse wall
    displacement with depth

    D(z) - Horizontal crevasse wall displacement with depth z


    Parameters
    ----------
    z : np.array
        negative veertical distance from ice surface
    sigma_T :
    mu : float, int
        ice softness/shear modulus
    crevasse_depth : float, int
        distance between ice surface and crevasse tip in m (positive).
    water_depth :
        depth of water surface within crevasse (negative)
    alpha : tuple, optional
        Defaults to (1-POISSONS_RATIO).
    has_water : bool, optional
        Is there water within the crevasse? Defaults to True.

    """
    # define D and alpha for a water-free crevasse
    sigma_A = applied_stress(
        sigma_T, crevasse_depth, water_depth, has_water=has_water, density=density
    )
    # define constant to advoid repeated terms in D equation
    c1 = (2 * alpha) / (mu * pi)
    D = (
        c1 * pi * sigma_A * diff_squares(crevasse_depth, z)
        + c1 * density * g * crevasse_depth * diff_squares(crevasse_depth, z)
        - (
            c1
            * density
            * g
            * z**2
            * 0.5
            * np.log(sum_over_diff(crevasse_depth, diff_squares(crevasse_depth, z)))
        )
    )
    if has_water:
        c1 = c1 * DENSITY_WATER * g
        D = (
            D
            - c1
            * diff_squares(crevasse_depth, water_depth)
            * diff_squares(crevasse_depth, z)
            + (
                c1
                * (z**2 - water_depth**2)
                * 0.5
                * np.log(
                    abs(
                        sum_over_diff(
                            diff_squares(crevasse_depth, water_depth),
                            diff_squares(crevasse_depth, z),
                        )
                    )
                )
            )
            - c1
            * water_depth
            * z
            * np.log(
                abs(
                    sum_over_diff(
                        water_depth * diff_squares(crevasse_depth, z),
                        z * diff_squares(crevasse_depth, water_depth),
                    )
                )
            )
            + c1
            * water_depth**2
            * np.log(
                abs(
                    sum_over_diff(
                        diff_squares(crevasse_depth, z),
                        diff_squares(crevasse_depth, water_depth),
                    )
                )
            )
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
    return (x + y) / (x - y)


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


def density_profile(depth, C=0.02, ice_density=917.0, snow_density=350.0):
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
    return ice_density - (ice_density - snow_density) * np.exp(-C * depth)


def paterson_empirical_relation(b, rho_i=917, rho_s=350, C=0.02):
    """depth dependant near-surface density

    Curve representing the transition from surface density to ice at
    depth.

    rho(b) = density_ice - (density_ice - density_surface)*e^-Cb


    Parameters
    ----------
    b : np.array
        array of depth below ice/snow surface in meters
    rho_i : int, optional
        ice density in kg/m^3, by default 917
    rho_s : int, optional
        surface densit in kg/m^3, by default 350 for seasonal snow
    C : float, optional
        empirical constant in units of m^-1 and can range from 0.0165
        to 0.0314 m^-1, by default 0.02 m^-1. Controls the rate at which
        firn density approaches ice.

    Returns
    -------
    rho(b) : np.array
        density as a function of depth in kg/m^3 corresponding to the
        b array input to function.
    """
    return rho_i - (rho_i - rho_s) * np.exp(-C * b)


def G(b, d, H):
    """Functional expression G in KI2 evaluation.

    G(gamma,lambda) where gamma=b/d, lambda=d/H
    from Tada et al., 1973 p 2.27, based on polynomial curve fitting to
    numerically derived solutions for the second component of the stress
    intensity factor KI2 for mode I crack opening. Implementiation also
    in van der veen, 1998 equ 12 and use in the intergral described in
    equ 14. p37.

    Parameters
    ----------
    b : float, int
        depth below surface
    d : float, int
        crevasse depth in meters.
    H : float, int
        ice thickness in meters

    Returns
    -------
    G : float
    """
    gamma = b / d
    lam = d / H
    G = (
        3.52 * (1 - gamma) / (1 - lam) ** (3 / 2)
        - (4.35 - 5.28 * gamma) / (1 - lam) ** 0.5
        + ((1.3 - 0.3 * gamma ** (3.2)) / (1 - gamma**2) ** 0.5 + 0.83 - 1.76 * gamma)
        * (1 - (1 - gamma) * lam)
    )
    return G


def overburden_stress(b):
    """ice overburden stress as a function of depth acconting for firn

    Parameters
    ----------
    b : int, float
        depth b in meters

    Returns
    -------
    float, int
        overburden stress in kPa
    """
    rho_i = 917
    rho_s = 350
    g = 9.81
    C = 0.02
    return -rho_i * g * b + ((rho_i - rho_s) / C) * g * (1 - np.exp(-C * b))


def f2(b, d, H, rhoi=917, rhos=350, C=0.02):
    """integral within the equation for KI(2) van der veen 1998"""
    return (-b + (rhoi - rhos) / (rhoi * C) * (1 - np.exp(-C * b))) * G(b, d, H)


def f3(a, b, d, H):
    return (b - a) * G(b, d, H)


def evaluate_KI1(d, H, Rxx, simplify=True, crev_spacing=None):
    """Stress intensity factor for tensile normal stress

    Parameters
    ----------
    d : int, float
        crevasse depth in meters
    H : int, float
        ice thickness in meters
    Rxx : int, float
        resistive stress in kPa, normal stress responsible for crevasse opening
        defined as the full stress sigma_xx minus the weight-induced
        lithostatic stress, L.
    simplify : bool, optional
        use simplification for shallow crevasses instead of polynomial
        curve fit to numerically computered stress intensity factors
        accounting for crevasse depth in proportion to ice thickness,
        by default True
    crev_spacing : int, float, optional
        spacing between crevasses in meters. If a value is given
        calculations will include the effect of shielding within the
        crevasse field (resulting in shallower initial crevasse depths).
        By default None, and the effect of shielding is ignored.

    Returns
    -------
    KI1: float, int
        KI1 in units of MPa
    """
    if crev_spacing:
        W = crev_spacing / 2
        S = W / (W + d)
        n = 1 / np.sqrt(np.pi)
        p = P(
            [
                n,
                n * 0.5,
                n * (3 / 8),
                n * (5 / 16),
                n * (35 / 128),
                n * (63 / 256),
                n * (231 / 1024),
                22.501,
                -63.502,
                58.045,
                -17.577,
            ]
        )
        KI1 = p(S) * Rxx * np.sqrt(np.pi * d * S)
    else:
        p = P([1.12, -0.23, 10.55, -21.72, 30.39])
        F = 1.12 if simplify else p(d / H)
        KI1 = F * Rxx * np.sqrt(np.pi * d)
    return KI1 / 1e6


def KI1_shielding(d, Rxx, crev_spacing):
    """KI1 accounting for opening and consider effect of shielding

    Parameters
    ----------
    d : _type_
        _description_
    Rxx : _type_
        _description_
    crev_spacing : float, int, optional
        spacing between crevasses in meters. If a value is given
        calculations will include the effect of shielding within the
        crevasse field (resulting in shallower initial crevasse depths).
        By default None, and the effect of shielding is ignored.

    Returns
    -------
    float, int
        KI1 in kPa
    """
    W = crev_spacing / 2
    S = W / (W + d)
    n = 1 / np.sqrt(np.pi)
    p = P(
        [
            n,
            n * 0.5,
            n * (3 / 8),
            n * (5 / 16),
            n * (35 / 128),
            n * (63 / 256),
            n * (231 / 1024),
            22.501,
            -63.502,
            58.045,
            -17.577,
        ]
    )
    return (p(S) * Rxx * np.sqrt(np.pi * d * S)) / 1e6


def evaluate_KI2(d, H, rhoi=917, rhos=350, C=0.02):
    """KI(2) in units of MPa"""
    x = np.arange(0, d, 0.01)
    f = f2(x, d, H, rhoi=rhoi, rhos=rhos, C=C)
    i_trapz = trapz(f, x)
    return (((2 * rhoi * 9.81) / np.sqrt(np.pi * d)) * i_trapz) / 1e6


def evaluate_KI3(a, d, H):
    """stress intensity factor component for a water-filled crevasse

    Parameters
    ----------
    a : float, int
        water depth below surface in meters
    d : float, int
        crevasse depth in meters
    H : float, int
        ice thickness in meters

    Returns
    -------
    KI3: float
        The third component of the stress intensity factor for mode I
        opening in units of kPa.
    """
    x = np.arange(a, d, 0.01)
    f = f3(a, x, d, H)
    return ((2 * 1000 * 9.81) / np.sqrt(np.pi * d) * trapz(f, x)) / 1e6 if d > a else 0


def evaluate_KInet(d, H, Rxx, a, rhoi=917, rhos=350, C=0.02):
    return (
        evaluate_KI1(d, H, Rxx)
        + evaluate_KI2(d, H, rhoi=rhoi, rhos=rhos, C=C)
        + evaluate_KI3(a, d, H)
    )


def penetration_depth_equ(d, H, Rxx, KIC, rhos=350, crev_spacing=None):
    """calculate crevasse penetration depth for a single crevasse

    Parameters
    ----------
    d : np.array, list, iterable
        depth vector to use for calculations
    H : int, float
        ice thickness in meters.
    Rxx : float, int
        Tensile stress in kPa
    KIC : float, int
        fracture toughness in MPa
    rhos : int, optional
        surface density to use in kg/m^3, by default 350 kg/m^3 for firn
    crev_spacing : float, int, optional
        spacing between crevasses in meters. If a value is given
        calculations will include the effect of shielding within the
        crevasse field (resulting in shallower initial crevasse depths).
        By default None, and the effect of shielding is ignored.

    Returns
    -------
    crevasse depth: float, int
        initial crevasse depth in meters.
    """
    K1 = (
        KI1_shielding(d, Rxx, crev_spacing) if crev_spacing else evaluate_KI1(d, H, Rxx)
    )
    return K1 + evaluate_KI2(d, H, rhos=rhos) - KIC


def find_crev_depth(H, Rxx, KIC, rhos=350, crev_spacing=None, d=None):
    """find initial crevasse depth for a water-free crevasse

    Parameters
    ----------
    H : float, int
        ice thickness in meters
    Rxx : float, int
        surface stress responsible for crevasse opening in kPa
    KIC : float, int
       fracture toughness of ice in MPa
    rhos : int, optional
        surface density, defautls to 350 kg/m^3, by default 350
    crev_spacing : float, int, optional
        spacing between crevasses in meters. If a value is given
        calculations will include the effect of shielding within the
        crevasse field (resulting in shallower initial crevasse depths).
        By default None, and the effect of shielding is ignored.
    d : np.array
        depth array to use to solve for initial crevasse. Units in
        meters. NOTE: the resolution of this array corresponds to the
        precision of the calculated result. Defaults to np.arange(50)
        producing crevasse depth rounded with no decimal places.

    Returns
    -------
    crev_depth: float, int
        inital crevasse depth upon creation in meters.
    """
    d = np.arange(50)
    if crev_spacing:
        xx = np.array(
            [
                KI1_shielding(x, Rxx, crev_spacing) + evaluate_KI2(x, H, rhos=rhos)
                for x in d
            ]
        )
    else:
        xx = np.array(
            [evaluate_KI1(x, H, Rxx) + evaluate_KI2(x, H, rhos=rhos) for x in d]
        )
    d_final = d[np.where(xx >= KIC)[0][-1]] if len(d[np.where(xx >= KIC)[0]]) > 0 else 0
    return round(d_final, 1)
