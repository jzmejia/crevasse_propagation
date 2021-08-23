


SECONDS_IN_YEAR = 60 * 60 * 24 * 365

# PHYSICAL CONSTANTS
g = GRAVITATIONAL_ACCELERATION = 9.8
IDEAL_GAS_CONSTANT = 8.314
POISSONS_RATIO = 0.3
FRACTURE_TOUGHNESS = 0.1
DRIVING_STRESS = 100


DENSITY_ICE = 910
DENSITY_ROCK = 2700
DENSITY_WATER = 1000

MELTING_POINT = 273.15

THERMAL_CONDUCTIVITY_ICE = 2.1
THERMAL_CONDUCTIVITY_ROCK = 3.3

HEAT_CAPACITY_ICE = 2115.3
HEAT_CAPACITY_SLOPE_ICE = 7.79293
HEAT_CAPACITY_ROCK = 800

LATIENT_HEAT_OF_FREEZING = 3.35e5

THERMAL_DIFFUSIVITY = THERMAL_CONDUCTIVITY_ICE / DENSITY_ICE / HEAT_CAPACITY_ICE

units = {
    'density': 'kg/m^3',
    'thermal conductivity': 'J/m/K/s',
    'thermal diffusivity': 'm^2/s',
    'latient heat of freezing': 'J/kg',
    'heat capacity': 'J/kg/K',
    'melting point at 1 atm': 'K',
    'fracture toughness': 'MPa m^-1/2',
    'driving stress': 'kPa'
    
}

# Flow law parameter A(T)
# Huybrechet's paramaterization

