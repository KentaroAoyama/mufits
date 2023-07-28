from pathlib import Path

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

# PATH
CACHE_DIR = Path.cwd().joinpath("cache")
CACHE_DEM_FILENAME = "dem.pickle"
CACHE_SEA_FILENAME = "sea.pickle"

# Conctsnt parameters
# NOTE: unit of "HCONDCF" is W/(m・K), "PERM" is mD
POROS = 0.2
TOPO_CONST_PROPS = {
    IDX_LAND: {
        "HCONDCFX": 2.0,
        "HCONDCFY": 2.0,
        "HCONDCFZ": 2.0,
        "PORO": POROS,
        "PERMX": 100,
        "PERMY": 100,
        "PERMZ": 100,
        "DENS": 2900.0,
        "HC": 0.84,
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
    },
    IDX_AIR: {
        "HCONDCFX": 0.0241,
        "HCONDCFY": 0.0241,
        "HCONDCFZ": 0.0241,
        "PORO": 0.9,
        "PERMX": 0.1,
        "PERMY": 0.1,
        "PERMZ": 0.1,
        "DENS": 1.293,
        "HC": 1.007,
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
    },
}

# Initial parameters
TOPO_INIT_PROPS = {
    IDX_LAND: ("   TEMPC   60.0", "   COMP1T  0.2"),
    IDX_SEA: ("   TEMPC   20.0", "   COMP1T  0.0"),
    IDX_AIR: ("   TEMPC   20.0", "   COMP1T  0.0"),
    IDX_LAKE: ("   TEMPC   20.0", "   COMP1T  0.0"),
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

P_BOTTOM = P_GRAD_ROCK * 1000.0
