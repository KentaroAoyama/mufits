# calculate the magnetic field from tough2 output
# Butsuri tansa handbook p484
from typing import Tuple, List, Dict, Literal, Optional
from pathlib import Path
from os import PathLike, makedirs
from math import pi, cos, sin
from time import time

import numpy as np
import pickle
from pyproj import Transformer
from shapely.geometry.polygon import Polygon
from matplotlib import pyplot as plt

from constants import (ORIGIN,
                       DXYZ,
                       IDX_LAND,
                       IDX_VENT,
                       IDX_CAP,
                       IDX_CAPVENT,
                       CRS_DEM,
                       CRS_RECT)
from utils import (load_topo_ls,
                   calc_m,
                   calc_ijk,
                   stack_from_center,
                   stack_from_0,
                   load_snap,
                   get_fpth_in_timeseries)
from monitor import img2mov

# Constants
# https://www.s-yamaga.jp/nanimono/chikyu/chijiki-01.htm
# X axis: West to East
# Y axis: North to South

# Earth's magnetic field
I0 = 57.0  # inclination
D0 = -9.0  # declination
# 
I1 = 57.0  # inclination
D1 = -9.0  # declination

# Direction of magnetization
L0 = cos(I0*pi/180.0)*sin(D0*pi/180.0)
M0 = -cos(I0*pi/180.0)*cos(D0*pi/180.0)
N0 = sin(I0*pi/180.0)

L1= cos(I1*pi/180.0)*sin(D1*pi/180.0)
M1 = -cos(I1*pi/180.0)*cos(D1*pi/180.0)
N1 = sin(I1*pi/180.0)

E = [L0, M0, N0]
P = [L1, M1, N1]
C = [i*j for i,j in zip(E, P)]

Cxy = (E[0]*P[1] + E[1]*P[0])/2.0
Cyz = (E[1]*P[2] + E[2]*P[1])/2.0
Czx = (E[2]*P[0] + E[0]*P[2])/2.0

# obs points
IDX_SUBSURF = set([IDX_LAND, IDX_VENT, IDX_CAP, IDX_CAPVENT])
topo_ls, _ = load_topo_ls()
NX, NY, NZ = len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2])
xc_ls = stack_from_center(DXYZ[0])  # W to E
yc_ls = stack_from_center(DXYZ[1])  # N to S
XX_OBS, YY_OBS = np.meshgrid(xc_ls, yc_ls)
ZZ_OBS = np.zeros_like(XX_OBS)
for i in range(NX):
    for j in range(NY):
        z = 0.0
        for k in range(NZ):
            m = calc_m(i,j,k,NX,NY)
            if topo_ls[m] in IDX_SUBSURF:
                ZZ_OBS[j][i] = z - 1.5
                break
            z += DXYZ[2][k]

# mesh points
m_props: Dict[int, Dict[Literal["BD", "T0", "TDIFF"], List[float] | float]] = {}   # BD, TDIFF
xmesh_ls = stack_from_center(DXYZ[0], centroid=False)
xmesh_ls.append(xmesh_ls[-1]+DXYZ[0][-1])  # W to E
ymesh_ls = stack_from_center(DXYZ[1], centroid=False)
ymesh_ls.append(ymesh_ls[-1]+DXYZ[1][-1])  # N to S
zmesh_ls = stack_from_0(DXYZ[2], centroid=False)
zmesh_ls.append(zmesh_ls[-1]+DXYZ[2][-1])  # top to bottom
for m in range(NX*NY*NZ):
    i, j, k = calc_ijk(m, NX, NY)
    if topo_ls[m] not in IDX_SUBSURF:
        continue
    m_props.setdefault(m, {"BD": [[xmesh_ls[i], xmesh_ls[i+1]],
                                  [ymesh_ls[j], ymesh_ls[j+1]],
                                  [zmesh_ls[k], zmesh_ls[k+1]]]})

def load_t(sumpth: PathLike, cells_gindex: List[int]):
    # load pressure and temperature
    (time, t_ls)= load_snap(sumpth, ["TEMPC"])[0]
    return time, [t_ls[m] for m in cells_gindex]

def prism_kernel(i: int,
                 j: int,
                 k: int,
                 BD: List[List[float]],
                 ) -> np.ndarray:
    # https://staff.aist.go.jp/r-morijiri/research/research_amag/mag-model01.html
    xx = BD[0][i]-XX_OBS
    yy = BD[1][j]-YY_OBS
    zz = BD[2][k]-ZZ_OBS

    R = np.sqrt(np.square(xx) + np.square(yy) + np.square(zz))

    # Tbijk =  Cyz*np.log((R-xx)/(R+xx)) \
    #     + Czx*np.log((R-yy)/(R+yy)) \
    #     - 2.0*Cxy*np.log(R+zz) \
    #     - C[0]*np.atan2((xx*yy), (R*R + R*zz - yy*yy)) \
    #     - C[1]*np.atan2((xx*yy), (R*R + R*zz - xx*xx)) \
    #     + C[2]*np.atan2((xx*yy), (R*zz))

    Tbijk =  Cyz*np.log((R-xx)/(R+xx)) \
        + Czx*np.log((R-yy)/(R+yy)) \
        + Cxy*np.log((R-zz)/(R+zz)) \
        + C[0]*np.atan2((yy*zz), (R*xx)) \
        + C[1]*np.atan2((zz*xx), (R*yy)) \
        + C[2]*np.atan2((xx*yy), (R*zz))

    return Tbijk

def calc_magnetic_field(curpth: PathLike, savedir: PathLike, t0: float=0.0) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    global m_props
    m_ls = []
    for m, dct in m_props.items():
        assert dct.get("BD", None) is not None
        assert dct.get("T0", None) is not None
        m_ls.append(m)
    curpth = Path(curpth)
    savedir = Path(savedir)
    makedirs(savedir, exist_ok=True)
    
    time, t_ls = load_t(curpth, m_ls)
    time += t0
    for m, t in zip(m_ls, t_ls):
        t0 = m_props[m]["T0"]
        m_props[m]["TDIFF"] = t-t0
    savepth = Path(savedir).joinpath(str(time)+".pkl")

    FF = 0.0 * XX_OBS   # initialization of total field 
    for m, prop in m_props.items():
        bd = prop["BD"]
        tdiff = prop["TDIFF"]
        J = -0.01 * tdiff   # temperature dependency: -0.01 [A/m/K]
        Tb111 = prism_kernel(0, 0, 0, bd)
        Tb112 = prism_kernel(0, 0, 1, bd)
        Tb121 = prism_kernel(0, 1, 0, bd)
        Tb122 = prism_kernel(0, 1, 1, bd)
        Tb211 = prism_kernel(1, 0, 0, bd)
        Tb212 = prism_kernel(1, 0, 1, bd)
        Tb221 = prism_kernel(1, 1, 0, bd)
        Tb222 = prism_kernel(1, 1, 1, bd)
        F = Tb122 + Tb212 + Tb221 + Tb111 - Tb222 - Tb112 - Tb121 - Tb211
        F *= 100.0 * J  # conversion to nT unit
        FF += F
    obj = ((XX_OBS, YY_OBS, ZZ_OBS), FF)

    with open(savepth, "wb") as pkf:
        pickle.dump(obj, pkf, pickle.HIGHEST_PROTOCOL)

    return time, obj


def calc_magnetic_dir(dirpth: PathLike) -> None:
    # cache magnetic field on observation points
    dirpth = Path(dirpth)
    fpth_ls = get_fpth_in_timeseries(dirpth, ignore_first=False)
    refpth = fpth_ls.pop(0)
    global m_props
    m_ls = [m for m in m_props]
    _, t0_ls = load_t(refpth, m_ls)
    for m, t0 in zip(m_ls, t0_ls):
        m_props[m]["T0"] = t0

    savedir = dirpth.joinpath("magnetic")
    makedirs(savedir, exist_ok=True)
    time = 0.0
    t0 = 0.0
    parent_set = set()
    for fpth in fpth_ls:
        print(fpth)
        if fpth.parent not in parent_set:
            t0 = time
        time, _ = calc_magnetic_field(fpth, savedir, t0=t0)
        parent_set.add(fpth.parent)

def plt_surface_magnetic(cachepth: PathLike,
                         savepth: PathLike,
                         crator_coods: Optional[Tuple[List[float], List[float]]]=None,) -> None:
    # TODO: plot observation points
    cachepth = Path(cachepth)
    savepth = Path(savepth)
    with open(cachepth, "rb") as pkf:
        (XX_OBS, YY_OBS, _), FF = pickle.load(pkf)
    YY_OBS *= -1.0
    fig, ax = plt.subplots()
    mappable = ax.pcolormesh(XX_OBS, YY_OBS, FF, vmin=-800.0, vmax=300.0)
    if crator_coods is not None:
        ax.plot(crator_coods[0],
                crator_coods[1],
                color="black",
                alpha=0.5,
                linestyle="dashed")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    fig.colorbar(mappable, label="Total Magnetic Field Change (nT)")
    fig.savefig(savepth, dpi=200)
    plt.clf()
    plt.close()
    return

def plt_surface_magnetic_for_dir(cachedir: PathLike) -> None:
    cachedir = Path(cachedir)
    with open("./analyse_crator_coords/crator.pkl", "rb") as pkf:
        coords: Polygon = pickle.load(pkf)
    rect_trans = Transformer.from_crs(CRS_DEM, CRS_RECT, always_xy=True)
    x0, y0 = rect_trans.transform(ORIGIN[1], ORIGIN[0])
    x_crator, y_crator = coords.exterior.xy
    x_crator = [x-x0 for x in x_crator]
    y_crator = [y0-y for y in y_crator]
    xy_gnss: Dict[str, Tuple[float, float]] = {}
    # TODO: 観測点位置取得
    # for key, (lat,lng) in POS_GNSS.items():
    #     x, y = rect_trans.transform(lng, lat)
    #     xy_gnss.setdefault(key, (x-x0, y-y0))
    savedir = cachedir.parent.joinpath("tstep").joinpath("magnetic").joinpath("surface")
    makedirs(savedir, exist_ok=True)
    for fpth in cachedir.iterdir():
        if fpth.suffix != ".pkl":
            continue
        time = float(fpth.stem)
        plt_surface_magnetic(fpth,
                             savedir.joinpath(str(time)+".png"),
                             crator_coods=(x_crator, y_crator),
                            #  baselines=((xy_gnss["SW"],xy_gnss["NE"]),
                            #             (xy_gnss["SE"],xy_gnss["NW"]))
                             )

if __name__ == "__main__":
    dirpth_ls = [
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_100000.0_v/unrest/900.0_0.0_15000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_100000.0_v/unrest/900.0_0.0_20000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_100000.0_v/unrest/900.0_0.0_25000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_100000.0_v/unrest/900.0_0.0_30000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_100000.0_v/unrest/900.0_0.0_35000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_v/unrest/900.0_0.0_15000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_v/unrest/900.0_0.0_20000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_v/unrest/900.0_0.0_25000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_v/unrest/900.0_0.0_30000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_v/unrest/900.0_0.0_35000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_100000.0_v/unrest/900.0_0.0_10000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_100000.0_v/unrest/900.0_0.0_15000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_100000.0_v/unrest/900.0_0.0_20000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_100000.0_v/unrest/900.0_0.0_25000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_100000.0_v/unrest/900.0_0.0_30000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_100000.0_v/unrest/900.0_0.0_35000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_v/unrest/900.0_0.0_10000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_v/unrest/900.0_0.0_15000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_v/unrest/900.0_0.0_20000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_v/unrest/900.0_0.0_25000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_v/unrest/900.0_0.0_30000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_v/unrest/900.0_0.0_35000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_100000.0_v/unrest/900.0_0.0_15000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_100000.0_v/unrest/900.0_0.0_20000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_100000.0_v/unrest/900.0_0.0_25000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_100000.0_v/unrest/900.0_0.0_30000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_100000.0_v/unrest/900.0_0.0_35000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_v/unrest/900.0_0.0_15000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_v/unrest/900.0_0.0_20000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_v/unrest/900.0_0.0_25000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_v/unrest/900.0_0.0_30000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_v/unrest/900.0_0.0_35000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_100000.0_v/unrest/900.0_0.0_15000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_100000.0_v/unrest/900.0_0.0_20000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_100000.0_v/unrest/900.0_0.0_25000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_100000.0_v/unrest/900.0_0.0_30000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_100000.0_v/unrest/900.0_0.0_35000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_v/unrest/900.0_0.0_15000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_v/unrest/900.0_0.0_20000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_v/unrest/900.0_0.0_25000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_v/unrest/900.0_0.0_30000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_v/unrest/900.0_0.0_35000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_100000.0_v/unrest/900.0_0.1_15000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_100000.0_v/unrest/900.0_0.1_20000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_100000.0_v/unrest/900.0_0.1_25000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_100000.0_v/unrest/900.0_0.1_30000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_100000.0_v/unrest/900.0_0.1_35000.0_10.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_v/unrest/900.0_0.1_15000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_v/unrest/900.0_0.1_20000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_v/unrest/900.0_0.1_25000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_v/unrest/900.0_0.1_30000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_v/unrest/900.0_0.1_35000.0_10.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_100000.0_v/unrest/900.0_0.1_15000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_100000.0_v/unrest/900.0_0.1_20000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_100000.0_v/unrest/900.0_0.1_25000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_100000.0_v/unrest/900.0_0.1_30000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_100000.0_v/unrest/900.0_0.1_35000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_v/unrest/900.0_0.1_15000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_v/unrest/900.0_0.1_20000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_v/unrest/900.0_0.1_25000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_v/unrest/900.0_0.1_30000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_v/unrest/900.0_0.1_35000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_100000.0_v/unrest/900.0_0.1_15000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_100000.0_v/unrest/900.0_0.1_20000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_100000.0_v/unrest/900.0_0.1_25000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_100000.0_v/unrest/900.0_0.1_30000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_100000.0_v/unrest/900.0_0.1_35000.0_10000.0_100000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_v/unrest/900.0_0.1_15000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_v/unrest/900.0_0.1_20000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_v/unrest/900.0_0.1_25000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_v/unrest/900.0_0.1_30000.0_10000.0_v_d",
        "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_v/unrest/900.0_0.1_35000.0_10000.0_v_d",
        # TODO: brit条件
                 ]
    # for dirpth in dirpth_ls:
    #     calc_magnetic_dir(dirpth)
    #     plt_surface_magnetic_for_dir(dirpth + "/magnetic")
    #     img2mov(dirpth+"/tstep/magnetic/surface", ftype="magnetic")
  
    for m in m_props:
        m_props[m]["T0"] = 0.0
    calc_magnetic_field("tmp.0028.SUM", "tmp")
    plt_surface_magnetic_for_dir("./tmp")
    pass