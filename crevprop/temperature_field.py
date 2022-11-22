"""
A two-dimensional thermal model used to solve for temperature
within the ice block defining model geometry. 

Solvers used include a semi-implicit finite-difference scheme
staggered leapfrog method for vertical advection
upward motion scaled linearly from -b at the ice surface to 0 at 
the bed. the entire domain is advected horizontally at 200 m/a 
Plug flow with Lagrangian reference frame

For the plug flow simulation the model's domain expands for each 
timestep at the upstream end of the horizontal model domain. 
The amount of ice added at the upstream end is determined by
the annual ice velocity `u`, which when applied, pushes the model
domain downstream. Once the model's horzontal domain exceeds a 
length of 500 m, we remove the uppermost 200 m of domain. This 
model configuration allows the model's domain to track the crevasse 
field as it advects downglacier and evolvs thermo-mechanically

Thermal model components include:
- horizontal diffusion
- vertical diffusion
- latent heat transfer from refreezing

"""
import pandas as pd
import numpy as np
from typing import Union, Tuple

from .physical_constants import DENSITY_ICE
from . import physical_constants as pc


class ThermalModel():
    """Thermal model used to create IceBlock's temperature field.

    Notes
    -----
    on ``ThermalModel`` geometry and relationship to ``IceBlock``
    This class set's up the geometry of the thermal model which differs
    ``IceBlock``. ThermalModel's horizontal ``dx`` and vertical ``dz``
    resolution differs from IceBlock, whereby ``dx`` is equal to the 
    diffusive lengthscale of heat through the ice (e.g., becomes a 
    function of ice properties and model timestep). We set the horzontal
    grid resolution of the ThermalModel equal to this value to enable 
    thermal calculations. 
    ``dz`` (vertical or depth) spacing can differ from ``IceBlock``, as 
    the thermal model will run with a corser vertical resolution to save 
    on computational expense. 


    Attributes
    ----------
    dt : int, float
        thermal model timestep in seconds
    diffusive_lengthscale: float
        distance of thermal diffusion within the ice during the 
        thermal model's timestep, unit in meters. 
    length : float
    ice_thickness
    dx : float, int
        x-coordinate spacing in meters
    dz : float, int
        z-coordinate (depth) spacing in meters
    z : np.array
        z-coordinate (depth) array of model coordinates
    x : np.array
        x-coordinates of model domain, from 0 at down-stream end 
        of ice block, to -L at up-stream end. 
    crevasses 
        crevasse coorindates within model domain
    T_surface : int, float
        Temperature at ice surface in deg C.
    T_bed : int, float
        Basal boundary condition used for thermal model, deg C.
        Defaults to 0 deg C
    T_upglacier : np.array
        Temperature profile to use for upstream boundary condition 
        of thermal model. requries temperatures in deg C with depth 
        from ice surface
    T : np.ndarray
        Temperature at dx, dz thorughout iceblock, deg C
    Tdf : pd.DataFrame
        Dataframe representation of Temperatures thoughout iceblock
        cols = x,coords, index = z,depth coords, values correspond to
        ice temperature in deg C.
    T_crev : float, int
        Temperature at crevasse walls in deg C.
    solver : str
        type of solver to use, defaults to ``explicit``
    """

    def __init__(
        self,
        ice_thickness: Union[int, float],
        length: Union[int, float],
        dt_T: Union[int, float],
        dz: Union[int, float],
        dx: float,
        crevasses,
        T_profile,
        T_surface=None,
        T_bed=None,
        solver=None,
        udef=0
    ):
        """Apply temperature advection and diffusion through ice block.

        Parameters
        ----------
        ice_thickness : int, float
            ice thickness in meters.
        length : int, float
            Length of model domain in the x-direction (m).
        dt_T : int, float
            thermal model timestep in seconds.
        dz : int, float:
            vertical resolution for domain (m). Value is set to 5 m if 
            input value is below 5 to reduce computational load. 
        T_profile : pd.Series, pd.DataFrame, np.Array
            Temperatures within ice column to set upstream boundary 
            condition within ice block. Temperatures in deg C with 
            corresponding depth from ice surface (m). 
        T_surface : float, int
            Air temperature in deg C. Defaults to 0.
        T_bed : float, int
            Temperature at ice-bed interface in deg C. Defaults to None.
        udef : deformation velocity, defaults to 0 m/a
            Will be able to assign an array for variable deformation
            velocity with depth but this is not yet supported.
        """
        # geometry
        self.length = length
        self.ice_thickness = ice_thickness
        self.dt = dt_T
        # self.diffusive_lengthscale = self._diffusion_lengthscale()
        # self.dx = (0.5*self.length) / round(0.5*self.length /
        # self.diffusive_lengthscale)
        self.dz = dz if self._ge(dz, 5) else 5
        self.dx = dx
        # NOTE: end of range = dx or dz to make end of array = 0
        self.z = np.arange(-self.ice_thickness, self.dz, self.dz) if isinstance(
            self.dz, int) else np.arange(-self.ice_thickness, self.dz, self.dz)
        self.x = np.arange(-self.dx-self.length, 0, self.dx)

        self.crevasses = crevasses

        self.udef = udef  # defaults to 0, can be int/float/depth vector

        # Boundary Conditions
        self.T_surface = T_surface if T_surface else 0
        self.T_bed = T_bed if T_bed else 0
        self.T_upglacier = self._set_upstream_bc(T_profile)
        # left = upglacier end, right = downglacier
        self.T = np.outer(self.T_upglacier, np.linspace(1, 0.99, self.x.size))

        # initialize temperatures used for leap-frog advection
        self.T0 = None
        self.Tnm1 = None
        self.Tdf = pd.DataFrame(
            data=self.T, index=self.z, columns=np.round(self.x))
        self.T_crev = 0

        self.solver = solver if solver else "explicit"

        # For solver to consider horizontal ice velocity a udef term
        # needs to be added at some point
        # For sovler to consider vertical ice velocity need ablation

        # self.crev_locs = 0

    def _diffusion_lengthscale(self):
        """calculate the horizontal diffusion of heat through ice, m"""
        return np.sqrt(pc.THERMAL_DIFFUSIVITY * self.dt_T)

    def _ge(self, n, thresh):
        """greater than"""
        return True if n >= thresh else False

    def _set_upstream_bc(self, Tprofile):
        """interpolate Tprofile to thermal model vertical resolution.

        NOTE: elevation should be negative and ordered as 
        `[-ice_thickness:0]`

        Parameters
        ----------
        Tprofile : [Tuple(array, array), pd.DataFrame]
            Temperature profile data points. (temperature, elevation)
            your array must be structured such that 

        Returns
        -------
        np.array
            Ice temperatures for entire ice block in deg C.
        """
        # interpolate temperature profile to match z
        if isinstance(Tprofile, pd.DataFrame):
            t, z = (1, 0) if Tprofile[Tprofile.columns[0]
                                      ].is_monotonic else (0, 1)
            t = Tprofile[Tprofile.columns[t]].values
            z = Tprofile[Tprofile.columns[z]].values
        elif isinstance(Tprofile, Tuple):
            t, z = Tprofile

        T = np.interp(self.z, z, t)

        # smooth temperature profile with a 25 m window
        win = self.dz*25+2
        T = pd.Series(T[:win]).append(pd.Series(T).rolling(
            win, min_periods=1, center=True).mean()[win:])

        # apply basal temperature condition
        idx = 0 if z[0] < -1 else -1
        T[idx] = self.T_bed

        return T.values

    def t_resample(self, dz):
        """resample temperatures to match z resolution"""
        if dz % self.dz == 0:
            T = self.Tdf[self.Tdf.index.isin(self.z[::dz].tolist())].values
        # ToDo add else statement/another if statement
        return T

    # def _calc_thermal_diffusivity(self):

    #     return

    def A_matrix(self):
        """create the A matrix to solve for future temperatures

        **A** solves for future temperatures within iceblock following::

            [y] = [A]^-1[b] 

            where A is the A matrix and b is current temperatures
            throughout ice block. Y are the temperatures at the
            same points at the next timestep



        Notes
        -----
        As the iceblock advects downglacier and the domain's length
        increases until reaching the specified maximum A will need
        to be recalculated. 

        internal deformation is not currently considered

        Returns
        -------
        A : np.ndarray
            square matrix with coefficients to calculate temperatures 
            within the ice block 
        """
        nx = self.x.size
        nz = self.z.size

        sx = pc.THERMAL_DIFFUSIVITY * self.dt / self.dx ** 2
        sz = pc.THERMAL_DIFFUSIVITY * self.dt / self.dz ** 2

        # Apply crevasse location boundary conditions to A
        # by creating a list of matrix indicies that shouldn't be assigned
        crev_idx = []
        for crev in self.crevasses:
            # surface coordinate already covered by surface boundary condition
            # use to find depth values
            crev_x = (nx * nz - 1) - abs(round(crev[0]/self.dx))
            crev_depth = crev[1]
            if crev_depth >= 2*self.dz and crev_depth < self.ice_thickness:
                crev_idx.extend(
                    np.arange(crev_x-(np.floor(crev_depth/self.dz)-1)*nx, crev_x, nx))

        # create inversion matrix A
        A = np.eye(self.T.size)

        # internal deformation (horizontal )
        for i in range(nx, self.T.size - nx):
            if i % nx != 0 and i % nx != nx-1 and i not in crev_idx:
                A[i, i] = 1 + 2*sx + 2*sz - self.dt/self.dx * self.udef
                A[i, i-nx] = A[i, i+nx] = -sz
                A[i, i-1] = -sx
                A[i, i+1] = -sx + self.dt / self.dx * self.udef

        return A

    def _calc_temperature(self):
        """Solve for future temp w/ implicit finite difference scheme

        Solve for future temperatures while storing temperature fields
        for the the previous two timesteps 

        """
        A = self.A_matrix()

        pass

    def refreezing(self):
        bluelayer = self.dt * pc.THERMAL_CONDUCTIVITY_ICE / \
            (pc.LATIENT_HEAT_OF_FUSION * DENSITY_ICE) * ()
        pass


# General thermal equations for pure ice and glacier ice

class PureIce(object):
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
            Note: c of dry snow and ice does not vary with density 
            because the heat needed to warm the air and vapor between 
            grains is neglibible. (see Cuffey, ch 9, pp 400)

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
    """Depth dependant thermal conductivity for dry snow, firn, and ice
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
