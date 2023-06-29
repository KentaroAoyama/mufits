from pathlib import Path

# lat0, lat1, lng0, lng1, Correction value (added to elevation value)
BOUNDS = {"Shikotsu": (42.674796, 42.818721, 141.257462, 141.427571, -220.73)}

# CRS of each toporogical data
CRS_WGS84 = "epsg:4326"
CRS_DEM = CRS_WGS84
CRS_SEA = CRS_WGS84  # WGS84
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
CACHE_DIR = Path.cwd()
CACHE_DEM_FILENAME = "dem.pickle"
CACHE_SEA_FILENAME = "sea.pickle"
