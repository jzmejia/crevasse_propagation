"""
A two-dimensional thermal model used to solve for temperature
within the ice block defining model geometry. 

Solvers used include a semi-implicit finite-difference scheme
staggered leapfrog method for vertical advection
upward motion scaled linearly from -b at the ice surface to 0 at the bed
the entire domain is advected horizontally at 200 m/a 
Plug flow with Lagrangian reference frame

For the plug flow simulation the model's domain expands for each 
timestep at the upstream end of the horizontal model domain. 
The amount of ice added at the upstream end is determined by
the annual ice velocity `u`, which when applied, pushes the model
domain downstream. Once the model's horzontal domain exceeds a 
length of 500 m, we remove the uppermost 200 m of domain. This model configuration allows the model's domain to track the
crevasse field as it advects downglacier and evolvs thermo-mechanically

Thermal model components include
- horizontal diffusion
- vertical diffusion
- latent heat transfer from refreezing
"""
import numpy as np


class PureIce:
    """Thermal equations applicable to pure ice.
    
    Parameters
    ----------
    Temperature: float, int
        Ice temperature in degrees Kelvin (K).
    
    Attributes
    ----------
    T: float, int
        ice temperature in degrees Kelvin (K)
    Lf: float
        Latient heat of fusion for ice
    density: float
        density of pure ice
    """
    def __init__(self, Temperature):
        """
        Parameters
        ----------
        Temperature : float
            ice temperature in degree K.
        """
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

            Parameters
            ----------
            temperature: float
                ice temperature in degrees Kelvin

            Returns
            -------
            c: float 
                specific heat capacity of ice in Jkg^-1K^-1
            """

            return 152.5 + 7.122 * self.T
        
        def thermal_conductivity(self):
            """calc thermal conductivy for pure ice (W/m/K)"""
            return 9.828 * np.exp(-5.7e-3 * self.T)



# for not pure ice
def thermal_conductivity(density):
    """Depth dependant thermal conductivity for dry snow, firn, and glacier ice
    Van Dusen 1929

    This equation typically gives a lower limit in most cases

    Parameters
    ----------
    density : (float)
        density of dry snow, firn, or glacier ice in kg/m^3
    """
    return 2.1e-2 + 4.2e-4 * density + 2.2e-9 * density**3



def thermal_diffusivity(thermal_conductivity, density, specific_heat_capacity):
    """ Thermal diffusivity calculation
    
    Parameters
    ----------
    thermal_conductivity : (float)
        in W/mK
    density : (float) 
        kg/m^3
    specific_heat_capacity : (float)
        J/kgK

    Returns
    -------
    thermal_diffusivity : (float)
        units of m^2/s
    """
    return thermal_conductivity/(density * specific_heat_capacity)
