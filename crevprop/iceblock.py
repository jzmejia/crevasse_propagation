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
    TypeVar
)
import numpy as np
import pandas as pd
import crevprop.physical_constants as pc


class IceBlock:
    def __init__(
        self,
        ice_thickness,
        dz,
        dx,
        crev_spacing,
        timestep,
        thermal_freq,
        u_surf=100.
    ):
        self.ice_thickness = ice_thickness
        self.dx = dx
        self.dz = dz
        self.crev_spacing = crev_spacing
        self.crev_count = 1
        self.u_surf = u_surf
        self.length = self.crev_count * self.crev_spacing + self.u_surf
        self.dt = timestep * pc.SECONDS_IN_DAY
        self.dt_T = self._thermal_timestep(timestep, thermal_freq)

    def _thermal_timestep(self, timestep, thermal_freq):
        if round(365 % (timestep*thermal_freq)) != 0:
            raise ValueError(
                "thermal_freq must divide 365 evenly to align the thermal model")
        return round(self.dt * thermal_freq)


class ThermalModel:
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
        sx = pc.THERMAL_DIFFUSIVITY * self.dt / self.dx ** 2
        sz = pc.THERMAL_DIFFUSIVITY * self.dt / self.dz ** 2

        A = np.eye()

        return A
