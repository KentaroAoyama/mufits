from constants import P_GRAD_ROCK

ORIGIN = (42.690559, 141.377357, 1041.0)

POS_SRC = (42.691521, 141.376839, -400.0)

PRES_SRC = 50.0  # MPa
SRC_TEMP = 700.0  # ℃
SRC_COMP1T = 0.0  # CO2 mole fraction

RAIN_AMOUNT = 1097.476 / 365.0 * 0.5  # mm/day
P_BOTTOM = P_GRAD_ROCK * 1050.0


# VTK
VLIM = {
    "PHST": (0.0, 1.0),
    "PRES (MPA)": (0.0, 3.0),
    "TEMPC (C)": (0.0, 700.0),
    "SAT#GAS": (0.0, 1.0),
    "SAT#LIQ": (0.0, 1.0),
}
