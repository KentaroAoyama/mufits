import pickle
from pyproj import Transformer
from shapely import Polygon, Point
from pathlib import Path

import sys
sys.path.append('../')

from constants import ORIGIN, DXYZ, IDX_SEA, IDX_LAKE, IDX_AIR, CRS_WGS84, CRS_RECT, CACHE_DIR
from utils import calc_m, stack_from_center

COORDS = [(141.372993,42.695787),
        (141.372628,42.695282),
        (141.372671,42.694383),
        (141.372607,42.694131),
        (141.372263,42.693263),
        (141.372113,42.692743),
        (141.372156,42.692443),
        (141.371834,42.692144),
        (141.371716,42.691925),
        (141.371555,42.691850),
        (141.371024,42.690489),
        (141.371191,42.690170),
        (141.371405,42.689965),
        (141.371368,42.689866),
        (141.371534,42.689389),
        (141.371550,42.689192),
        (141.371657,42.689046),
        (141.371663,42.688766),
        (141.371555,42.688550),
        (141.371577,42.688455),
        (141.37155,42.688230),
        (141.371630,42.688061),
        (141.371604,42.687926),
        (141.371663,42.687647),
        (141.371797,42.687449),
        (141.371829,42.687055),
        (141.373690,42.685081),
        (141.379967,42.684506),
        (141.380868,42.684719),
        (141.385063,42.686682),
        (141.386565,42.688173),
        (141.386737,42.688512),
        (141.387026,42.689466),
        (141.387155,42.690515),
        (141.387091,42.691075),
        (141.386940,42.691548),
        (141.386662,42.691958),
        (141.386232,42.692329),
        (141.383464,42.694261),
        (141.382391,42.694702),
        (141.380074,42.695459),
        (141.379699,42.695728),
        (141.379559,42.695720),
        (141.376469,42.696004),
        (141.376158,42.696059),
        (141.375836,42.695956),
        (141.373379,42.695696),
        (141.373250,42.695712),
        (141.373122,42.695814),
        (141.372993,42.695787),]
CRS_WGS84 = "epsg:4326"
CRS_RECT = "epsg:6680"

def wgs2rect():
    rect_trans = Transformer.from_crs(CRS_WGS84, CRS_RECT, always_xy=True)
    lng_ls = [latlng[0] for latlng in COORDS]
    lat_ls = [latlng[1] for latlng in COORDS]
    x_ls, y_ls = rect_trans.transform(lng_ls, lat_ls)
    xy_ls = [(x, y) for x, y in zip(x_ls, y_ls)]
    crator = Polygon(xy_ls)
    with open("./crator.pkl", "wb") as pkf:
        pickle.dump(crator, pkf, pickle.HIGHEST_PROTOCOL)
    return crator

def get_m():
    rect_trans = Transformer.from_crs(CRS_WGS84, CRS_RECT, always_xy=True)
    x0, y0 = rect_trans.transform(ORIGIN[1], ORIGIN[0])
    xc_ls = stack_from_center(DXYZ[0])
    yc_ls = stack_from_center(DXYZ[1])
    xc_ls = [x0 + xc for xc in xc_ls]
    yc_ls = [y0 - yc for yc in yc_ls]
    dome: Polygon = wgs2rect()
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