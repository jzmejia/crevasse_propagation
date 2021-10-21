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