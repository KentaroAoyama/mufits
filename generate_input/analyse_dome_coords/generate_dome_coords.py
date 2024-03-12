import pickle
from pyproj import Transformer
from shapely import Polygon, Point
from pathlib import Path

import sys
sys.path.append('../')

from constants import ORIGIN, DXYZ, IDX_SEA, IDX_LAKE, IDX_AIR, CRS_WGS84, CRS_RECT, CACHE_DIR
from utils import calc_m, stack_from_center

COORDS = [(42.688567, 141.380857),
          (42.688723, 141.378352),
          (42.688486, 141.376947),
          (42.688703, 141.376131),
          (42.689054, 141.375348),
          (42.689456, 141.374801),
          (42.690068, 141.374457),
          (42.690631, 141.374345),
          (42.691073, 141.374484),
          (42.691597, 141.374790),
          (42.691925, 141.375149),
          (42.692268, 141.375740),
          (42.692477, 141.376389),
          (42.692489, 141.377344),
          (42.692429, 141.378143),
          (42.692220, 141.378856),
          (42.691633, 141.379532),
          (42.691156, 141.379806),
          (42.689356, 141.381115),
        ]

def get_dome_zone():
    rect_trans = Transformer.from_crs(CRS_WGS84, CRS_RECT, always_xy=True)
    lng_ls = [latlng[1] for latlng in COORDS]
    lat_ls = [latlng[0] for latlng in COORDS]
    x_ls, y_ls = rect_trans.transform(lng_ls, lat_ls)
    xy_ls = [(x, y) for x, y in zip(x_ls, y_ls)]
    return Polygon(xy_ls)

def get_m():
    rect_trans = Transformer.from_crs(CRS_WGS84, CRS_RECT, always_xy=True)
    x0, y0 = rect_trans.transform(ORIGIN[1], ORIGIN[0])
    xc_ls = stack_from_center(DXYZ[0])
    yc_ls = stack_from_center(DXYZ[1])
    xc_ls = [x0 + xc for xc in xc_ls]
    yc_ls = [y0 - yc for yc in yc_ls]
    dome: Polygon = get_dome_zone()
    with open(Path.cwd().parent.joinpath("cache").joinpath("topo_ls"), "rb") as pkf: #!
        topo_ls, _ = pickle.load(pkf)
    nx, ny = len(DXYZ[0]), len(DXYZ[1])
    m_ls = []
    for j, yc in enumerate(yc_ls):
        for i, xc in enumerate(xc_ls):
            m = calc_m(i, j, 0, nx, ny)
            if topo_ls[m] in (IDX_AIR, IDX_LAKE, IDX_SEA):
                continue
            if dome.contains(Point(xc, yc)):
                m_ls.append(m)
    with open(Path.cwd().joinpath("m_ls"), "wb") as pkf:
        pickle.dump(m_ls, pkf)
    return m_ls

from matplotlib import pyplot as plt
from utils import calc_ijk
if __name__ == "__main__":
    m_ls = get_m()
    i_ls, j_ls = [], []
    for m in m_ls:
        i, j, k = calc_ijk(m, 40, 40)
        i_ls.append(i)
        j_ls.append(j)
    plt.scatter(i_ls, j_ls)
    plt.show()
    pass