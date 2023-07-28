# PATH
DEM_PTH = "./dem"
SEADEM_PTH = "./seadem"
CRS_RECT = "epsg:6680"
RUNFILE_PTH = "tmp2.RUN"
ALIGN_CENTER = True
DXYZ = (
    [
        429.981696,
        358.31808,
        298.5984,
        248.832,
        207.36,
        172.8,
        144.0,
        120.0,
        100.0,
        86.4,
        72.0,
        60.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        60.0,
        72.0,
        86.4,
        100.0,
        120.0,
        144.0,
        172.79999999999998,
        207.35999999999999,
        248.83199999999997,
        298.59839999999997,
        358.31807999999995,
        429.98169599999994,
    ],
    [
        429.981696,
        358.31808,
        298.5984,
        248.832,
        207.36,
        172.8,
        144.0,
        120.0,
        100.0,
        86.4,
        72.0,
        60.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        60.0,
        72.0,
        86.4,
        100.0,
        120.0,
        144.0,
        172.79999999999998,
        207.35999999999999,
        248.83199999999997,
        298.59839999999997,
        358.31807999999995,
        429.98169599999994,
    ],
    [
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
    ],
)

ORIGIN = (42.690559, 141.377357, 1041.0)

POS_SRC = (42.691521, 141.376839, -400.0)

PRES_SRC = 50.0  # MPa
SRC_TEMP = 700.0  # ℃
SRC_COMP1T = 0.2  # CO2 mole fraction

# VTK
VLIM = {
    "PHST": (0.0, 1.0),
    "PRES (MPA)": (0.0, 3.0),
    "TEMPC (C)": (0.0, 700.0),
    "SAT#GAS": (0.0, 1.0),
    "SAT#LIQ": (0.0, 1.0),
}
