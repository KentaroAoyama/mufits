from pathlib import Path

# ORIGIN = (42.690531, 141.376630, 1041.0)
# A火口：315.9, -194.0
# B火口：-58.6, -143.9
ORIGIN = (42.690531, 141.376630, 955.0)
POS_SRC = (42.691753, 141.375653, -400.0)

POS_SINK = {"A": (42.688814, 141.380509, 955.6),
            "B": (42.689230, 141.375933, 981.0),
            "E": (42.690010,141.376491, 989.5),
            "G": (42.691461, 141.378561, 972.0)}

SINK_PARAMS = {"A": 200.0,
               "B": 34.29355281207131,
               "E": 17.55829903978051}

# lat0, lat1, lng0, lng1, Correction value (added to elevation value)
LAKE_BOUNDS = {
    "Shikotsu": (42.674796, 42.818721, 141.257462, 141.427571, -220.73 + 255.0)
}

# rainfall (m / year)
RAIN_AMOUNT = 1739.033333 * 1.0e-3

# evaporation rate (m /year)
EVAP_AMOUNT = 435.7482143 * 1.0e-3

# cross sectional area (m2), outflow height (m)
RIVERS = {"Tomakomai": (25.4 * 1.0e6, 1805.6666 * 1.0e-3),
          "Koitoi": (8.97 * 1.0e6, 2729.0 * 1.0e-3),
          "Nishitappu": (24.11* 1.0e6, 1765.0 * 1.0e-3),
          "Nishitappusawa": (8.65 * 1.0e6, 1166.6666 * 1.0e-3),
          "Kumanosawa": (13.66 * 1.0e6, 854.0 * 1.0e-3),
          "Oboppu": (9.79 * 1.0e6, 752.0 * 1.0e-3),
          "Tarumai": (24.23 * 1.0e6, 1755.6666 * 1.0e-3)}

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

# parameters of Manning & Ingebritsen (1998)
MIA = -14.0
MIB = -3.2

PERM_MAX: float = 1.0e-9

# topology index
IDX_LAND = 0
IDX_SEA = 1
IDX_LAKE = 2
IDX_AIR = 3
IDX_VENT = 4
IDX_CAP = 5
IDX_CAPVENT = 6

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
TEMPE_AIR = 10.0 # ℃
# Grain density (kg/m3)
DENS_ROCK = 2900.0
# Water density (kg/m3)
DENS_WATER = 1.0e3
TOPO_CONST_PROPS = {
    IDX_LAND: {
        "HCONDCFX": 2.0,
        "HCONDCFY": 2.0,
        "HCONDCFZ": 2.0,
        "PORO": POROS,
        # "PERMX": PERM_HOST,
        # "PERMY": PERM_HOST,
        # "PERMZ": PERM_HOST,
        "DENS": DENS_ROCK,
        "HC": 1.0,
        "TEMPC": 20.0,
        "COMP1T": 3.25e-7,
    },
    IDX_VENT: {
        "HCONDCFX": 2.0,
        "HCONDCFY": 2.0,
        "HCONDCFZ": 2.0,
        "PORO": POROS,
        # "PERMX": 1000,
        # "PERMY": 1000,
        # "PERMZ": 1000,
        "DENS": DENS_ROCK,
        "HC": 1.0,
        "TEMPC": 20.0,
        "COMP1T": 3.25e-7,
    },
    IDX_CAP: {
        "HCONDCFX": 2.0,
        "HCONDCFY": 2.0,
        "HCONDCFZ": 2.0,
        "PORO": POROS,
        # "PERMX": 1000,
        # "PERMY": 1000,
        # "PERMZ": 1000,
        "DENS": DENS_ROCK,
        "HC": 1.0,
        "TEMPC": 20.0,
        "COMP1T": 3.25e-7,
    },
    IDX_CAPVENT: {
        "HCONDCFX": 2.0,
        "HCONDCFY": 2.0,
        "HCONDCFZ": 2.0,
        "PORO": POROS,
        # "PERMX": 1000,
        # "PERMY": 1000,
        # "PERMZ": 1000,
        "DENS": DENS_ROCK,
        "HC": 1.0,
        "TEMPC": 20.0,
        "COMP1T": 3.25e-7,
    },
    IDX_SEA: {
        "HCONDCFX": 0.6,
        "HCONDCFY": 0.6,
        "HCONDCFZ": 0.6,
        "PORO": 1.0,
        "PERMX": 0.0,
        "PERMY": 0.0,
        "PERMZ": 0.0,
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
        "TEMPC": TEMPE_AIR,
        "COMP1T": 1.0,
    },
    IDX_LAKE: {
        "HCONDCFX": 0.6,
        "HCONDCFY": 0.6,
        "HCONDCFZ": 0.6,
        "PORO": 0.9,
        "PERMX": 0.0,
        "PERMY": 0.0,
        "PERMZ": 0.0,
        "DENS": 1000.0,
        "HC": 4.182,
        "TEMPC": 10.0,
        "COMP1T": 0.0,
    },
}

# Heat capacity of grain (Stissi et al., 2021; Hikcs et al., 2009 (10.1029/2008JB006198))
HC_ROCK = 1.0

# gravitational acceleration
G = 9.80665

# Atmospheric pressure (MPa)
P_TOP = 1.013e-1
P_GROUND = P_TOP + TOPO_CONST_PROPS[IDX_AIR]["DENS"] * G * ORIGIN[2] * 1.0e-6

# Henry's constant for CO2 gas to water
# https://www.eng-book.com/pdfs/879040e33a05a0e5f1cb85580ef77ad1.pdf
Kh = 0.104e4 * 1.013e-1 * 1.0e6  # NOTE: at 10℃, unit: Pa

# Pressure gradient (MPa/m)
P_GRAD_AIR = TOPO_CONST_PROPS[IDX_AIR]["DENS"] * G * 1.0e-6
P_GRAD_SEA = TOPO_CONST_PROPS[IDX_AIR]["DENS"] * G * 1.0e-6
P_GRAD_LAKE = DENS_WATER * G * 1.0e-6
P_GRAD_ROCK = DENS_WATER * G * 1.0e-6

# Temperature gradient (℃/m)
T_GRAD_AIR = 0.0
T_GRAD_SEA = 0.0
T_GRAD_LAKE = 0.0
T_GRAD_ROCK = 0.06

# Time taken to reproduce steady state (in years)
TIME_SS = 500 #!

# Initial time step (in days)
TSTEP_MIN = 1.0e-12
TSTEP_INIT = 1.0e-5
# Maximum time step (days)
TSTEP_MAX = 300.0 # not used
# number of iterations for each TSTEP_MAX
# 浸透率の異方性を入れる前：
# NDTFIRST = 10
# NDTEND = 10
# TMULT = 1.05
NDTFIRST = 400
NDTEND = 400
TMULT = 7.0 # 7 is optimum?

# for SS
TSTEP_UNREST = None
TRPT_UNREST = None # unrestに限らず, 途中から計算しなおすときにこの間隔にする
TEND_UNREST = None

# unrest or continue_from_latest
# 以下Noneでデフォルト値 (NOTE: unrestを計算しないときは, Noneに設定する)
# TSTEP_UNREST = None # Noneでデフォルト値, 設定すれば定数となる
# TRPT_UNREST = 30.0 # in days unrestに限らず, 途中から計算しなおすときにこの間隔にする
# TEND_UNREST = 30.0 # in years

OUTDIR = r"E:\tarumai2"
CONVERSION_CRITERIA = {"TEMPC": 1.0e-2,
                       "PRES": 1.0e-3,
                       "SAT#GAS": 1.0e-4,
                       "COMP1T": 1.0e-4,}
CONDS_PID_MAP_NAME = "pid.txt"

if __name__ == "__main__":
    pass