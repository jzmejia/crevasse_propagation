""" 
IceBlock
 --  --  --  -- 
The main container for the crevasse propagation model, holding and 
initializing model geometry.




"""


from typing import (
    Dict,
    List,
    Tuple,
    Any,
    Optional,
    Union,
)
import numpy as np
import pandas as pd
import crevprop.physical_constants as pc


class IceBlock(object):
    def __init__(
        self,
        ice_thickness,
        dz,
        dx,
        crev_spacing,
        timestep,
        thermal_freq,
        T_profile,
        T_surface=None,
        T_bed=None,
        u_surf=100.
    ):
        """A 2-dimensional container containing model domain elements.
        
        
        Parameters
        ----------
        ice_thickness : float, int
            thickness of ice block in meters
        dz : float, int
            vertical sampling resolution within ice block (m)
        dx : float, int
        crev_spacing : float, int
            Crevasse spacing (m)
        timestep : float
            Timestep in days to run crevasse model
        thermal_freq : float, int
            Multiple of timestep to run thermal model. 1 would run the 
            thermal model at every timestep whereas a value of 10 would 
            run the model after every 10 timesteps.
        T_profile : np.array, pd.Series, pd.DataFrame, optional
            Temperature profile for upstream boundary condition. The 
            profile will be interpolated to match the thermal model's 
            vertical resolution.
        T_surface : float, optional
            Ice surface temperature, degrees C. Defaults to 0.
        T_bed (_type_, optional): 
            Temperature boundary condition at Defaults to None.
        u_surf (float, optional): 
            Ice surface velocity within domain (meters per year).
            Defaults to 100.
        """

        self.ice_thickness = ice_thickness
        self.dx = dx
        self.dz = dz
        self.crev_spacing = crev_spacing
        self.crev_count = 1
        # self.crev_field =
        self.u_surf = u_surf
        self.length = self.crev_count * self.crev_spacing + self.u_surf
        self.dt = timestep * pc.SECONDS_IN_DAY
        self.dt_T = self._thermal_timestep(timestep, thermal_freq)
        self.temperature = self._init_temperatures(T_profile, T_surface, T_bed)

    def _init_geometry(self):
        """initialize ice block geometry

        """
        x = self._toarray(-self.dx-self.length, 0, self.dx)
        z = self._toarray(-self.ice_thickness, 0, self.dz)
        pass

    def advect_domain(self):
        """advect domain downglacier in accordance with ice velocity
        """
        pass

    def _thermal_timestep(self, timestep, thermal_freq):
        if round(365 % (timestep*thermal_freq)) != 0:
            raise ValueError(
                "thermal_freq must divide 365 evenly")
        return round(self.dt * thermal_freq)

    def _init_temperatures(self, T_profile, T_surface, T_bed):
        T = ThermalModel(self.ice_thickness, self.length, self.dt_T,
                         self.dz, T_profile, T_surface, T_bed)
        return T


class ThermalModel(object):
    def __init__(
        self,
        ice_thickness: Union[int, float],
        length: Union[int, float],
        dt_T: Union[int, float],
        dz: Union[int, float],
        T_profile,
        T_surface=None,
        T_bed=None
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
        """
        
        
        self.length = length
        self.ice_thickness = ice_thickness
        self.dt = dt_T
        self.dx = (0.5*self.length) / round(0.5*self.length /
                                            self._diffusion_lengthscale())
        self.dz = dz if self._ge(dz, 5) else 5
        self.z = np.arange(-self.ice_thickness, self.dz, self.dz) if isinstance(
            self.dz, int) else self._toarray(-self.ice_thickness, self.dz, self.dz)
        self.x = self._toarray(-self.dx-self.length, 0, self.dx)

        # Boundary Conditions
        self.T_surface = T_surface if T_surface else 0
        self.T_bed = T_bed if T_bed else 0
        self.T_upglacier = self._set_upstream_bc(T_profile)
        # left = upglacier end, right = downglacier
        self.T = np.outer(self.T_upglacier, np.linspace(1, 0.99, self.x.size))
        self.Tdf = pd.DataFrame(
            data=self.T, index=self.z, columns=np.round(self.x))
        self.T_crev = 0
        # self.crev_locs = 0

    def _diffusion_lengthscale(self):
        return np.sqrt(1.090952729018252e-6 * self.dt)

    def _ge(self, n, thresh):
        return True if n >= thresh else False

    def _toarray(self, start, stop, step):
        return np.linspace(start, stop, abs(round((start-stop)/step))+1)

    def _set_upstream_bc(self, Tprofile):
        """interpolate temperature profile data points to match z res.

        Args:
            Tprofile [Tuple(array, array), pd.DataFrame]: Temperature 
                profile data points. temperature, elevation = Tuple
                your array must be structured such that 

        Returns:
            [np.array]: Ice temperatures for entire ice block in deg C.
        """
        # interpolate temperature profile to match z (vertical resolution.)
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
        if dz % self.dz == 0:
            T = self.Tdf[self.Tdf.index.isin(self.z[::dz].tolist())].values
        # ToDo add else statement/another if statement
        return T

    def A_matrix(self):
        """create the A matrix to solve for future temperatures

        [y] = [A]^-1[b] where A is the A matrix and b is current temperatures
                        throughout ice block. Y are the temperatures at the
                        same points at the next timestep

        NOTE: As the iceblock advects downglacier and the domain's length
              increases until reaching the specified maximum A will need
              to be recalculated. 

        WARNING: Function does not currently add boundary conditions for
        crevasse locations. 

        Returns:
            A (np.ndarray): square matrix with coefficients to calculate
                            temperatures within the ice block 
        """
        nx = self.x.size

        sx = pc.THERMAL_DIFFUSIVITY * self.dt / self.dx ** 2
        sz = pc.THERMAL_DIFFUSIVITY * self.dt / self.dz ** 2

        # create inversion matrix A
        A = np.eye(self.T.size)
        for i in range(nx, self.T.size - nx):
            if i % nx != 0 and i % nx != nx-1:
                A[i, i] = 1 + 2*sx + 2*sz
                A[i, i-nx] = A[i, i+nx] = -sz
                A[i, i+1] = A[i, i-1] = -sx

        # Apply crevasse location boundary conditions to A
        return A

    def _execute(self):
        """Solve for future temp w/ implicit finite difference scheme
        """
        pass

    def thermal_conductivity(density):
        """
        depth dependant thermal conductivity formula, Van Dusen 1929

        note, this gives a lower limit in most cases

        Args:
            density (float): ice/snow density in kg/m^3
        """
        return 2.1e-2 + 4.2e-4 * density + 2.2e-9 * density**3

    def specific_heat_capacity(temperature):
        """specific heat capacity for pure ice of temperature in Kelvin

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

        return 152.5 + 7.122 * temperature

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


def CrevField():
    pass
