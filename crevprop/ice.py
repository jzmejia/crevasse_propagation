"""
Copyright (c) 2021-2024 by Jessica Mejia <jzmejia@buffalo.edu>



The main container for the crevasse propagation model, holding and
initializing model geometry
"""

import numpy as np


class Ice(object):
    """Material properties of ice

    Parameters
    ----------


    Attributes
    ----------
    density:
    T: float, int
        ice temperature in degrees Celcius
    """

    def __init__(self, ice_density=917.0, ice_temperature=0, fracture_toughness=100e3):
        """_summary_

        Parameters
        ----------
        density : int, optional
            ice denisty in kg/m^3, by default 917 kg/m^3
        temperature : int, optional
            ice temperature in degrees Celsius, by default 0C
        fracture_toughness : float, optional
            fracture toughness of ice in Pa, by default 0.1 MPa
        """

        self.ice_density = ice_density
        self.ice_temperature = ice_temperature

        self.specific_heat_capacity = 2097
        self.heat_capacity_intercept = 2115.3
        self.heat_capacity_slope = 7.79293

        # note: thermal conductivity at 0 deg C unit W/mK
        self.thermal_conductivity = self.ki = 2.1

        self.latient_heat_of_freezing = self.Lf = 3.35e5
        self.kappa = self.thermal_diffusivity()
        self.units = self._set_unit()

        self.fracture_toughness = fracture_toughness

    def C_to_K(self, C):
        """Convert temperature in degrees Celsius to Kelvin"""
        return C + 273.15

    def calc_specific_heat_capacity(self, T):
        """specific heat capacity for pure ice (J/kg/K)

        Specific heat capacity, c, per unit mass of ice in SI units.
        Note: c of dry snow and ice does not vary with density
        because the heat needed to warm the air and vapor between
        grains is neglibible. (see Cuffey, ch 9, pp 400)

        c = 152.5 + 7.122(T)

        Parameters
        ----------
        T: float
            ice temperature in degrees Celcius

        Returns
        -------
        c: float
            specific heat capacity of ice in Jkg^-1K^-1

        """
        return 152.5 + 7.122 * self.C_to_K(T)

    def thermal_conductivity_pure_ice(self, T=0):
        return 9.828 * np.exp(-5.7e-3 * self.C_to_K(T))

    def van_dusen(self, density):
        """Depth dependant thermal conductivity for dry snow, firn, ice
        Van Dusen (1929)

        This equation typically gives a lower limit in most cases

        Parameters
        ----------
        density : (float)
            density of dry snow, firn, or glacier ice in kg/m^3

        """
        return 2.1e-2 + 4.2e-4 * density + 2.2e-9 * density**3

    def schwerdtfeger(self, density):
        # density must be less than the density of pure ice,
        # find threshold to use here
        pure_ice = self.thermal_conductivity_pure_ice
        pass

    def thermal_conductivitiy_firn(self, x, relationship="density"):
        """calculate thermal conductivity of firn

        This function implements the depth or density dependant
        empirical relationships described by Oster and Albert (2022).

        Use the density relation for depths from 0-48 m. When the
        the density of pure ice is entered into this equation a thermal
        conductivity `k_{firn}(p_ice)=2.4` W/mK which is the known
        thermal conductivity of pure ice at -25 deg C.

        The depth dependant relationship predicts the thermal
        conductivity of pure ice for depths around 100-110 m. This is
        consistant with the field-measured depth of the firn-ice
        transitions.

        Parameters
        ----------
        x : float
            density or depth used in calculation. Must correspond to
            choice of relationship.
        relationship : str, optional
            must be "density" or "depth", by default "density"
        """
        # function of density vs function of depth
        if relationship == "density":
            k_firn = 0.144 * np.exp(0.00308 * x)
        elif relationship == "depth":
            k_firn = 0.536 * np.exp(0.0144 * x)
        return k_firn

    def calc_thermal_conductivity(self, T=0, density=917, method="empirical"):
        """calc thermal conductivy using specified method"""
        if method == "van_dusen":
            kt = self.van_dusen(density)
        elif method == "schwerdtfeger":
            kt = 1
        elif method == "empirical":
            kt = self.thermal_conductivity_pure_ice(T)

        return kt

    def thermal_diffusivity(self):
        """calculate thermal diffusivity

        thermal_conductivity / density * specific_heat_capacity


        Returns
        -------
        thermal diffusivity with units of m^2/s
        """
        return self.thermal_conductivity / (
            self.ice_density * self.specific_heat_capacity
        )

    def _set_unit(self):
        units = {
            "density": "kg/m^3",
            "thermal conductivity": "J/m/K/s",
            "thermal diffusivity": "m^2/s",
            "latient heat of freezing": "J/kg",
            "heat capacity": "J/kg/K",
            "melting point at 1 atm": "K",
            "fracture toughness": "MPa m^-1/2",
            "driving stress": "kPa",
        }
        return units
