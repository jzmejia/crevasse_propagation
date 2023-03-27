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
import matplotlib.pyplot as plt


def flatten(nested):
    """return flatened list"""
    return [item for elem in nested for item in elem]


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
    Lf : float
        Latient heat of freezing for ice in kJ/kg
    ki : float
        Thermal conductivity of ice in W/m/K
    ice_density : float, int
        Ice density in kg/m^3
        NOTE: future versions will allow for depth dependant densities
        for ice and snow/firn. Upon implementation accepted dtypes will
        include np.array objects or pd.DataFrame/pd.Series objects.
    """

    def __init__(
        self,
        ice_thickness: Union[int, float],
        length: Union[int, float],
        dt_T: Union[int, float],
        dz: Union[int, float],
        dx: float,
        x: np.array,
        crevasses,
        T_profile,
        T_surface=None,
        T_bed=None,
        # solver=None,
        udef=0,
        thermal_conductivity=2.1,
        ice_density=917,
        latient_heat_of_freezing_ice=3.35e5,
        thermal_diffusivity=1.09e-6
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
        thermal_freq : int
            frequency thermal model is run in comparison to crevasse
            runs. dt * thermal_freq = dt_T
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
        ice_density : optional, float
            ice density in kg/m^3. Defaults to 917 kg/m^3
        thermal_conductivity : optional, float
            thermal conductivity of ice in W/m/K. Defaults to 2.1
        latient_heat_of_freezing_ice : optional, float
            latient heat of freezing for ice in kJ/kg.
            Defaults to 3.35e5.
        """
        # define constants consistant with IceBlock
        self.ice_density = ice_density
        self.ki = thermal_conductivity
        self.Lf = latient_heat_of_freezing_ice
        self.kappa = thermal_diffusivity
        self.heat_capacity_intercept = 2115.3

        # geometry
        self.length = length
        self.ice_thickness = ice_thickness
        self.dt = dt_T
        self.dz = dz if self._ge(dz, 5) else 5
        self.dx = dx

        # NOTE: end of range = dx or dz to make end of array = 0
        self.z = np.arange(-self.ice_thickness, self.dz, self.dz) if isinstance(
            self.dz, int) else np.arange(-self.ice_thickness, self.dz, self.dz)
        # why start at -dx-self.length?
        self.x = x

        self.udef = udef  # defaults to 0, can be int/float/depth vector

        # crevasse info
        self.crevasses = crevasses
        self.crev_idx = self.find_crev_idx()
        # self.virtualblue = self.refreezing()

        # Boundary Conditions
        self.T_surface = T_surface if T_surface else 0
        self.T_bed = T_bed if T_bed else 0
        self.T_upglacier = self._set_upstream_bc(T_profile)
        # left = upglacier end, right = downglacier
        self.T = np.outer(self.T_upglacier, np.linspace(1, 0.99, self.x.size))
        self.T_downglacier = self.T[:, -1]
        self.T_crev = 0

        # initialize temperatures used for leap-frog advection
        # values only used in calculations, no need to allow user access
        self.T0 = None
        self.Tnm1 = None

        # For solver to consider horizontal ice velocity a udef term
        # needs to be added at some point
        # For sovler to consider vertical ice velocity need ablation

    def _diffusion_lengthscale(self):
        """calculate the horizontal diffusion of heat through ice, m"""
        return np.sqrt(self.kappa * self.dt)

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
                                      ].is_monotonic_increasing else (0, 1)
            t = Tprofile[Tprofile.columns[t]].values
            z = Tprofile[Tprofile.columns[z]].values
        elif isinstance(Tprofile, Tuple):
            t, z = Tprofile

        T = np.interp(self.z, z, t)

        # smooth temperature profile with a 25 m window
        win = self.dz*25+2
        T = pd.concat([pd.Series(T[:win]), pd.Series(T).rolling(
            win, min_periods=1, center=True).mean()[win:]])

        # apply basal temperature condition
        idx = 0 if z[0] < -1 else -1
        T[idx] = self.T_bed

        return T.values

    def df(self):
        """return pandas DataFrame object of Temperature Field"""
        return pd.DataFrame(data=self.T, index=self.z, columns=np.round(self.x))

    def t_resample(self, dz):
        """resample temperatures to match z resolution"""
        if dz % self.dz == 0:
            T = self.Tdf[self.Tdf.index.isin(self.z[::dz].tolist())].values
        # ToDo add else statement/another if statement
        return T

    # def _calc_thermal_diffusivity(self):

    #     return

    def A_matrix(self):
        """Physical coefficient matrix defining heat transfer properties

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
            within the ice block. matrix shape = nx*nz, nx*nz where
            nx is the number of points in the x direction of the
            ThermalModel domain and z is the number of points in the
            z direction.
        """
        nx = self.x.size
        # nz = self.z.size

        sx = self.kappa * self.dt / self.dx ** 2
        sz = self.kappa * self.dt / self.dz ** 2

        # Apply crevasse location boundary conditions to A
        crev_idx = self.find_crev_idx(nested=False)

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

    def find_crev_idx(self, nested=True):
        '''creating a list of T matrix indicies for crevasse locations

        Parameters
        ----------
        nested : bool, default True
            return a nested list of crevasse indicies. If True return a
            nested list such that there is one list of indicies for
            each crevasse. If False return a flattened list.
        inplace: bool, default True
            If False, return a copy. Otherwise, do operation inplace and
            return updated class attribute.

        Returns
        -------
            crev_idx : list of crevasse indicies within thermal model
        '''
        crev_idx = []

        for crev in self.crevasses:

            # crevasse index at ice surface
            crev_x = (self.x.size * self.z.size - 1) - \
                abs(round(crev[0]/self.dx))

            # index at crevasse tip (on thermal model grid)
            if abs(crev[1]) >= 2*self.dz:
                crevtip = crev_x-(abs(np.floor(crev[1]/self.dz))-1)*self.x.size
            else:
                crevtip = crev_x-(abs(np.floor(crev[1]/self.dz))) * self.x.size

            # TODO add in error if crev depth > ice thickness

            idx = np.arange(crevtip, crev_x + self.x.size,
                            self.x.size, dtype=int).tolist()

            crev_idx.append(idx)

        self.crev_idx = crev_idx

        return crev_idx if nested else flatten(crev_idx)

    def calc_temperature(self):
        """Solve for future temp w/ implicit finite difference scheme

        Solve for future temperatures while storing temperature fields
        for the the previous two timesteps

        """
        setattr(self, 'Tmn1', self.T0)
        setattr(self, 'T0', self.T)

        nx = self.x.size
        crev_idx = self.find_crev_idx(nested=False)

        A = self.A_matrix()

        # compute rhs
        rhs = self.T.flatten()

        # apply boundary conditions
        rhs[flatten(self.crev_idx)] = self.T_crev
        rhs[:nx] = self.T_bed
        rhs[-nx:] = self.T_surface
        rhs[np.arange(nx, self.T.size-nx, nx)] = self.T_upglacier[1:-1]
        rhs[np.arange(2*nx-1, self.T.size-nx, nx)] = self.T_downglacier[1:-1]

        # apply source term for latent heat near crevasses

        # downstream of creasse
        idx_surface = rhs.size-nx
        idx = [x+1 for x in crev_idx if x < idx_surface]
        rhs[idx] = rhs[idx] + (self.Lf/self.heat_capacity_intercept
                               * flatten(self.bluelayer_right)) / self.dx

        for num, crev in enumerate(self.crev_idx):
            # upstream of crevasse
            if num+1 < len(self.crev_idx) or max(crev) > rhs.size-nx:
                idx = [x-1 for x in crev if x < idx_surface]
                rhs[idx] = rhs[idx] + (self.Lf / self.heat_capacity_intercept
                                       * self.bluelayer_right) / self.dx

        # adjust for air in crevasse above water level
        # need to pass water depth from crevasse to crevasse field,
        # crevfield->iceblock->thermal model
        # then assign Tsurf to those points probably only matters for
        # alot of air in big crevasses

        # compute solution vector
        T = np.linalg.solve(A, rhs)
        return T.reshape(self.T.shape)

    def refreezing(self):
        """Find refrozen layer thickness at crevasse walls in meters.

        This funtion will evaluate and update `self.bluelayer` attrs
        and return virtualblue.



        The refreezing rate of meltwater depends on the horizontal
        temperature gradient in the ice of the crevasse walls dT/dx

        The refreezing rate of meltwater Vfrz(z) can be approximated
        as follows:

        V_frz(z)/dt = ki/Lf*rho_i [dT_L(x,z)/dx + dT_R(x,z)/dx]

        where
            ki is the thermal conductivity of ice
            Lf is the latent heat of freezing
            dt thermal model timestep
            TL and TR are the temperatures at the left and right crevasse
                walls respectively
            dx = the diffusion length over a year (~5 meters)

        Naming conventions:
            virtualblue - potential refreezing rate for all z points
                at crevasse location, therefore not limited to current
                crevasse depth. Values passed to `IceBlock` and used by
                `Crevasse` to find refreezing at crevasse walls as ice
                fractures deeper in subsequent timesteps.
            bluelayer - refreezing source term corresponding to actual
                crevasse depth and water level, used to solve iceblock
                temperatures in `.calc_temperature()`. Values internal
                to `ThermalModel`, not used externally.
            NOTE: In both cases values are indexed from -ice_thickness
                    (bed) to ice surface, and np.array values are
                    listed in the order they appear in `self.crevasses`

        Returns
        -------
        virtualblue_l, virtualblue_r : list of np.array
            virtualblue, refreezing rate, for all points in ice
            thickness corresponding to each crevasse location. Retruns
            list of np.arrays in the same order as `self.crevasses`.
            _l refers to left-hand (upglacier) side of crevasse wall and
            _r to the right-hand (downglacier) side.
            NOTE: to be usable by `Crevasse` values in virtualblue need
            to be divided by thermal frequency (e.g., current calc of
            refreezing rate is at the timestep of the thermal model) and
            interpolated to match the z resolution of IceBlock (e.g.,
            calculations are currently at z-resolution of thermal model)


        """
        # calculate refreezing for each crevasse

        # calculate how much volume will refreeze in a year
        # indicies diffusive lengthscale for 1 year = 5.6 meters

        # only make calculation if there is enough info in T
        # model must have advected more than 5.6 m
        # (1 year thermal diffusivity) from crevasse location

        ind = round(5.6/self.dx)

        # initalize
        bluelayer_l = bluelayer_r = []
        virtualblue_l = virtualblue_r = []

        for num, crev in enumerate(self.crevasses):

            # get indicies for all z cooresponding to crevasse
            ft_idx = np.arange(np.remainder(crev[-1], self.x.size),
                               crev[-1], self.x.size)

            # ice temp on downglacier side of crevasse (right)
            T_down = self.T.flatten()[[x + ind for x in ft_idx]]

            # use downglacier temperatures if domain is too small
            T_up = self.T.flatten()[[x - ind for x in self.ft_idx]
                                    ] if self.x[0] >= crev[0]-5.6 else -T_down

            virtualblue_l = self.calc_refreezing(T_up, self.T_crev)
            virtualblue_r = self.calc_refreezing(self.T_crev, T_down)

            # save np.arrays to list
            bluelayer_l.append(virtualblue_l[-len(self.crev_idx[num]):])
            bluelayer_r.append(virtualblue_r[-len(self.crev_idx[num]):])
            virtualblue_l.append(virtualblue_l)
            virtualblue_r.append(virtualblue_r)

        self.bluelayer_left = bluelayer_l
        self.bluelayer_right = bluelayer_r

        # save if you want to here
        # self.virtualblue = (virtualblue_l, virtualblue_r)

        return virtualblue_l, virtualblue_r

    def calc_refreezing(self,
                        T_crevasse,
                        T_ice,
                        # prevent_negatives=True
                        ):
        """Refreezing layer thickness at crevasse wall in meters.

        The refreezing rate of meltwater Vfrz(z) can be approximated
        as follows:

        V_frz(z)/dt = ki/(Lf*rho_i)  [dT_L(x,z)/dx + dT_R(x,z)/dx]

        where
        ki is the thermal conductivity of ice
        Lf is the latent heat of freezing
        dt thermal model timestep
        TL and TR are the temperatures at the left and right crevasse
            walls respectively
        dx = the diffusion length over a year (~5 meters)

        NOTE: refrozen layer thickness set to 0 if calculation yields a
        negative value.

        Parameters
        ----------
        T_crevasse : np.array
            Ice temperatures along crevasse wall
        T_ice : np.array
            Temperatures ~5.6m into the ice (horizontally) from crevasse
            walls. 5.6 meters is the diffusive lengthscale for one year
        # prevent_negatives : bool
        #     whether to prevent negative refreezing (i.e., melting). by
        #     default True

        Returns
        -------
        bluelayer : np.array
            refrozen layer thickness in meters along crevasse walls for
            thermal model timestep dt, accounts for expansion of water
            as it refreezes.

        """
        bluelayer = self.dt * (self.ki/self.Lf/self.ice_density) * (
            T_crevasse - T_ice) / (round(5.6/self.dx)*self.dx)

        # set to zero if any values are negative
        # if prevent_negatives:
        #     np.place(bluelayer, bluelayer < 0, 0)

        return bluelayer * (1000/self.ice_density)
