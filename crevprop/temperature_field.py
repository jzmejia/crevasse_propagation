"""
temperature_field.py



Components
    1. Temperature of ice block
        - determined from observations & boundary conditions
    2. Refreezing


2-D Thermal Model
    - 0ºC for ice in contact with each crevasse

    2-D semi-implicit finite-difference scheme
    staggered leapfrog method for vertical advection
        upward motion scaled linearly from -b @surface to 0 @bed
        the entire domain is advected horizontally at 200 m/a 
        ^ Plug flow with Lagrangian reference frame

    Domain:
        250 m long
        200 m buffer region at the downstream end

        Plug flow simulation
        model domain expands in each timestep at the upstream end 
        according to the ice velocity u, theregby pushing the domain
        downstream

        @500 m length, the upper 200 m is disgarded annually
        We retain for use as an upstream boundary condition over the 
        subsequent year (Poinar 2015)

        This model configuration allows the model's domain to track the
        crevasse field as it advects downglacier and evolvs thermo-mechanically

    Thermal model components
        - horizontal diffusion
        - vertical diffusion
        - latent heat transfer from refreezing
        


"""
import numpy as np


class PureIce:
    def __init__(self, Temperature):
        self.T = Temperature
        self.Lf = 333.5
        self.density = 9.17
        self.specific_heat_capacity = self.specific_heat_capacity()
        self.thermal_conductivity = self.thermal_conductivity()
        self.thermal_diffusivity = self.thermal_conductivity / (
            self.density * self.specific_heat_capacity)
        
        
        def specific_heat_capacity(self):
            """specific heat capacity for pure ice (J/kg/K)

            Specific heat capacity, c, per unit mass of ice in SI units. 
            Note: c of dry snow and ice does not vary with density because the 
            heat needed to warm the air and vapor between grains is neglibible.
            (see Cuffey, ch 9, pp 400)

            c = 152.5 + 7.122(T)

            Args:
                temperature (float): ice temperature in degrees Kelvin

            Returns:
                float : specific heat capacity (c) in Jkg^-1K^-1
            """

            return 152.5 + 7.122 * self.T
        
        def thermal_conductivity(self):
            """calc thermal conductivy for pure ice (W/m/K)"""
            return 9.828 * np.exp(-5.7e-3 * self.T)



# for not pure ice

def thermal_conductivity(density):
    """
    depth dependant thermal conductivity for dry snow, firn, and glacier ice
    Van Dusen 1929

    note, this gives a lower limit in most cases

    Args:
        density (float): ice/snow density in kg/m^3
    """
    return 2.1e-2 + 4.2e-4 * density + 2.2e-9 * density**3



def thermal_diffusivity(thermal_conductivity, density, specific_heat_capacity):
    """

    Args:
        thermal_conductivity (float): W/mK
        density (float): kg/m^3
        specific_heat_capacity (float): J/kgK

    Returns:
        thermal_diffusivity (float): units of m^2/s
    """
    return thermal_conductivity/(density * specific_heat_capacity)
