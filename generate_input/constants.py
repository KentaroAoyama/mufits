from pathlib import Path

ORIGIN = (42.690531, 141.376630, 1041.0)
POS_SRC = (42.691753, 141.375653, -400.0)

# lat0, lat1, lng0, lng1, Correction value (added to elevation value)
LAKE_BOUNDS = {
    "Shikotsu": (42.674796, 42.818721, 141.257462, 141.427571, -220.73 + 255.0)
}

# CRS of each toporogical data
CRS_WGS84 = "epsg:4326"
CRS_DEM = CRS_WGS84
CRS_SEA = CRS_WGS84
CRS_LAKE = CRS_WGS84

# resolution of each data
RES_DEM = 10.0 * 10.0
RES_SEA = 500.0 * 500.0  # https://www.jodc.go.jp/jodcweb/JDOSS/infoJEGG_j.html
RES_LAKE = (
    500.0
    * 500.0  # TODO: currently same as sea, but maybe better to re-compile contour data
)

# topology index
IDX_LAND = 0
IDX_SEA = 1
IDX_LAKE = 2
IDX_AIR = 3
IDX_VENT = 4

# PATH
DEM_PTH = "./dem"
SEADEM_PTH = "./seadem"
CRS_RECT = "epsg:6680"
RUNFILE_PTH = "tmp2.RUN"
CACHE_DIR = Path.cwd().joinpath("cache")
CACHE_DEM_FILENAME = "dem.pickle"
CACHE_SEA_FILENAME = "sea.pickle"
ALIGN_CENTER = True
DXYZ = (
    [
        743.00834,
        619.17362,
        515.97802,
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
        515.97802,
        619.17362,
        743.00834,
    ],
    [
        743.00834,
        619.17362,
        515.97802,
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
        515.97802,
        619.17362,
        743.00834,
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
        50.0,
        50.0,
        50.0,
        50.0,
    ],
)

# Conctsnt parameters
# NOTE: unit of "HCONDCF" is W/(m・K), "PERM" is mD
POROS = 0.2
PERM_HOST = 1.0e-16 / 9.869233 * 1.0e16
TOPO_CONST_PROPS = {
    IDX_LAND: {
        "HCONDCFX": 2.0,
        "HCONDCFY": 2.0,
        "HCONDCFZ": 2.0,
        "PORO": POROS,
        # "PERMX": PERM_HOST,
        # "PERMY": PERM_HOST,
        # "PERMZ": PERM_HOST,
        "DENS": 2900.0,
        "HC": 0.84,
        "TEMPC": 20.0,
        "COMP1T": 0.0001,
    },
    IDX_VENT: {
        "HCONDCFX": 2.0,
        "HCONDCFY": 2.0,
        "HCONDCFZ": 2.0,
        "PORO": POROS,
        # "PERMX": 1000,
        # "PERMY": 1000,
        # "PERMZ": 1000,
        "DENS": 2900.0,
        "HC": 0.84,
        "TEMPC": 20.0,
        "COMP1T": 0.0001,
    },
    IDX_SEA: {
        "HCONDCFX": 0.6,
        "HCONDCFY": 0.6,
        "HCONDCFZ": 0.6,
        "PORO": 0.9,
        "PERMX": 0.1,
        "PERMY": 0.1,
        "PERMZ": 0.1,
        "DENS": 1020.0,
        "HC": 3.9,
        "TEMPC": 20.0,
        "COMP1T": 0.0,
    },
    IDX_AIR: {
        "HCONDCFX": 0.0241,
        "HCONDCFY": 0.0241,
        "HCONDCFZ": 0.0241,
        "PORO": 1.0,
        "PERMX": 0.0,
        "PERMY": 0.0,
        "PERMZ": 0.0,
        "DENS": 1.293,
        "HC": 1.007,
        "TEMPC": 10.0,
        "COMP1T": 1.0,
    },
    IDX_LAKE: {
        "HCONDCFX": 0.6,
        "HCONDCFY": 0.6,
        "HCONDCFZ": 0.6,
        "PORO": 0.9,
        "PERMX": 0.1,
        "PERMY": 0.1,
        "PERMZ": 0.1,
        "DENS": 1000.0,
        "HC": 4.182,
        "TEMPC": 10.0,
        "COMP1T": 0.0,
    },
}

# Grain density (kg/m3)
DENS_ROCK = 2900.0

# Water density (kg/m3)
DENS_WATER = 1.0e3

# Heat capacity of grain
HC_ROCK = 0.84

# Atmospheric pressure (MPa)
P_GROUND = 1.013e-1

# Pressure gradient (MPa/m)
P_GRAD_AIR = 9.0e-6
P_GRAD_SEA = 1.02e-3
P_GRAD_LAKE = 1.0e-3
P_GRAD_ROCK = (DENS_ROCK * (1.0 - POROS) + DENS_WATER * POROS) * 1.0e-6

# Temperature gradient (℃)
T_GRAD_AIR = 0.0
T_GRAD_SEA = 0.0
T_GRAD_LAKE = 0.0
T_GRAD_ROCK = 0.02

# Time taken to reproduce steady state (in years)
TIME_SS = 500
# Initial time step (in days)
TSTEP_INIT = 0.0005
TSTEP_MAX = 100.0
TSTEP_MIN_MULT = 0.001
NDT = 1.0
TMULT = 1.2

OUTDIR = r"E:\tarumai"
