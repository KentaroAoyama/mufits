#!/usr/bin/env python
# coding: utf-8

"""Generate input file for MUFITS (Afanasyev, 2012)
"""

from functools import partial
from typing import List, Tuple, TextIO, Callable
from pathlib import Path
from os import PathLike, makedirs
import re
from statistics import median
import pickle

import pandas as pd
from pyproj import Transformer

from constants import (
    BOUNDS,
    RES_DEM,
    RES_SEA,
    RES_LAKE,
    CRS_WGS84,
    CRS_DEM,
    CRS_SEA,
    CRS_LAKE,
    IDX_LAND,
    IDX_SEA,
    IDX_LAKE,
    IDX_AIR,
    CACHE_DIR,
    CACHE_DEM_FILENAME,
    CACHE_SEA_FILENAME,
)

nx, ny, nz = 10, 2, 5

# TODO: 天水の量をtimestepごとに調整する
# TODO: 入力：グリッド, モデルパラメータ（timestepごとの流入量, 浸透率, CO2分率とする）
# TODO: generate using 10km × 10km DEM (10m)

gx = [100.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 100.0]
gy = [10.0, 10.0]
gz = [35.0, 10.0, 10.0, 10.0, 35.0]
act_ls = []
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if k == 0:
                act_ls.append(2)
            else:
                act_ls.append(1)
assert len(act_ls) == nx * ny * nz


def __get_two_floatstring(_str: str) -> Tuple[float]:
    """Get two decimals in a row separated by whitespace
    (e.g., 40.0, 140.0)

    Args:
        _str (str): String

    Returns:
        Tuple[float]: Two floats
    """
    result = re.search("\d+(\.\d+)?\s+\d+(\.\d+)?", _str)
    lat_lng_str = result.group()
    _lat, _lng = lat_lng_str.split(" ")
    return float(_lat), float(_lng)


def __clip_xy(
    _x: List, _y: List, _elv: List, bounds: Tuple[float]
) -> Tuple[List, List, List]:
    _x_arr = np.array(_x)
    _y_arr = np.array(_y)
    _elv_arr = np.array(_elv)
    filt = (
        (bounds[0] < _x_arr)
        * (_x_arr < bounds[1])
        * (bounds[2] < _y_arr)
        * (_y_arr < bounds[3])
    )
    return _x_arr[filt].tolist(), _y_arr[filt].tolist(), _elv_arr.tolist()


def __clip_lake(_lat: List, _lng: List, _elv: List) -> Tuple[List, List, List]:
    lat_arr = np.array(_lat)
    lng_arr = np.array(_lng)
    elv_arr = np.array(_elv)
    lat_lake, lng_lake, elv_lake = [], [], []
    for _, (lat0, lat1, lng0, lng1, dh) in BOUNDS.items():
        filt = (lat0 < lat_arr) * (lat_arr < lat1) * (lng0 < lng_arr) * (lng_arr < lng1)
        lat_lake.extend(lat_arr[filt].tolist())
        lng_lake.extend(lng_arr[filt].tolist())
        elv_lake.extend((elv_arr[filt] + dh).tolist())
    return lat_lake, lng_lake, elv_lake


def __calc_ijk(m: int, nz: int, ny: int) -> Tuple[int]:
    k, q = divmod(m, nz)
    j, i = divmod(q, ny)
    return i, j, k


def load_dem(pth_dem: PathLike) -> Tuple[List, List, List]:
    base_dir = Path(pth_dem)
    lat_ls, lng_ls, elv_ls = [], [], []
    for pth in base_dir.glob("**/*"):
        if pth.suffix != ".xml":
            continue
        with open(pth, "r", encoding="utf-8") as f:
            # parameters of DEM
            flag_elv = False
            lat_range, lng_range = [None, None], [None, None]
            n_lat, n_lng = None, None
            lat_step, lng_step = None, None
            cou_max = None
            # increment
            cou: int = 0
            for line in f.readlines():
                if flag_elv:
                    result = re.search("\d+(\.\d+)?", line)
                    a, b = divmod(cou, n_lat)
                    lat_ls.append(lat_range[0] + a * lat_step)
                    lng_ls.append(lng_range[0] + b * lng_step)
                    elv_ls.append(float(result.group()))
                    cou += 1
                    if cou > cou_max:
                        break
                if "<gml:lowerCorner>" in line:
                    _lat, _lng = __get_two_floatstring(line)
                    lat_range[0] = _lat
                    lng_range[0] = _lng
                if "<gml:upperCorner>" in line:
                    _lat, _lng = __get_two_floatstring(line)
                    lat_range[1] = _lat
                    lng_range[1] = _lng
                if "<gml:high>" in line:
                    _nlat, _nlng = __get_two_floatstring(line)
                    n_lat = int(_nlat)
                    n_lng = int(_nlng)
                if "<gml:tupleList>" in line:
                    assert None not in lat_range
                    assert None not in lng_range
                    assert n_lat is not None
                    assert n_lng is not None
                    lat_step = (lat_range[1] - lat_range[0]) / n_lat
                    lng_step = (lng_range[1] - lng_range[0]) / n_lng
                    cou_max = n_lat * n_lng
                    flag_elv = True
    return lat_ls, lng_ls, elv_ls


def load_sea_dem(pth_seadem: PathLike) -> Tuple[List, List, List]:
    base_dir = Path(pth_seadem)
    lat_ls, lng_ls, elv_ls = [], [], []
    for pth in base_dir.glob("**/*"):
        _df = pd.read_csv(pth, sep="\s+", header=None)
        # first column indicates flag (interpolated value or not)
        lat_ls.extend(_df[1].to_list())
        lng_ls.extend(_df[2].to_list())
        elv_ls.extend((0.0 - _df[3]).to_list())
    return lat_ls, lng_ls, elv_ls


def __median(_ls: List) -> float:
    if len(_ls) == 0:
        return 0.0
    return median(_ls)


def generate_topo(
    pth_dem: PathLike,
    pth_seadem: PathLike,
    crs_rect: str,
    dxyz: Tuple[List, List, List],
    origin: Tuple[float, float, float],
) -> List[int]:
    # TODO: docstring & output type

    # generate empty array with mesh
    assert len(dxyz) == 3, dxyz
    dx_ls, dy_ls, dz_ls = dxyz

    if not CACHE_DIR.exists():
        makedirs(CACHE_DIR)

    # load DEM
    cache_dem = CACHE_DIR.joinpath(CACHE_DEM_FILENAME)
    lat_dem_ls, lng_dem_ls, elv_dem_ls = None, None, None
    if cache_dem.exists():
        with open(cache_dem, "rb") as pkf:
            lat_dem_ls, lng_dem_ls, elv_dem_ls = pickle.load(
                pkf, pickle.HIGHEST_PROTOCOL
            )
    else:
        lat_dem_ls, lng_dem_ls, elv_dem_ls = load_dem(pth_dem)
        with open(cache_dem, "wb") as pkf:
            pickle.dump(
                (lat_dem_ls, lng_dem_ls, elv_dem_ls), pkf, pickle.HIGHEST_PROTOCOL
            )

    # load marine & lake topology
    cache_sea = CACHE_DIR.joinpath(CACHE_SEA_FILENAME)
    lat_sea_ls, lng_sea_ls, elv_sea_ls = None, None, None
    if cache_sea.exists():
        with open(cache_sea, "rb") as pkf:
            lat_sea_ls, lng_sea_ls, elv_sea_ls = pickle.load(
                pkf, pickle.HIGHEST_PROTOCOL
            )
    else:
        lat_sea_ls, lng_sea_ls, elv_sea_ls = load_sea_dem(pth_seadem)
        with open(cache_sea, "wb") as pkf:
            pickle.dump(
                (lat_sea_ls, lng_sea_ls, elv_sea_ls), pkf, pickle.HIGHEST_PROTOCOL
            )

    # clip lake topography (TODO: 解像度が500mだと荒すぎるかも)
    lat_lake_ls, lng_lake_ls, elv_lake_ls = __clip_lake(
        lat_sea_ls, lng_sea_ls, elv_sea_ls
    )

    # convert to rect crs (WGS84 to crs_rect)
    # NOTE: Both DEM and bathymetry data use WGS84.
    transformer_wgs = Transformer.from_crs(CRS_WGS84, crs_rect, always_xy=True)
    transformer_dem = Transformer.from_crs(CRS_DEM, crs_rect, always_xy=True)
    transformer_sea = Transformer.from_crs(CRS_SEA, crs_rect, always_xy=True)
    transformer_lake = Transformer.from_crs(CRS_LAKE, crs_rect, always_xy=True)

    # calculate the coordinates of the grid center
    x_origin, y_origin = transformer_wgs.transform(origin[1], origin[0])
    elv_origin = origin[2]

    # convert DEM & seadem CRS
    x_dem_ls, y_dem_ls = transformer_dem.transform(lng_dem_ls, lat_dem_ls)
    x_sea_ls, y_sea_ls = transformer_sea.transform(lng_sea_ls, lat_sea_ls)
    x_lake_ls, y_lake_ls = transformer_lake.transform(lng_lake_ls, lat_lake_ls)

    # convert to numpy array
    x_dem_arr, y_dem_arr, elv_dem_arr = (
        np.array(x_dem_ls),
        np.array(y_dem_ls),
        np.array(elv_dem_ls),
    )
    x_sea_arr, y_sea_arr, elv_sea_arr = (
        np.array(x_sea_ls),
        np.array(y_sea_ls),
        np.array(elv_sea_ls),
    )
    x_lake_arr, y_lake_arr, elv_lake_arr = (
        np.array(x_lake_ls),
        np.array(y_lake_ls),
        np.array(elv_lake_ls),
    )

    # get center coordinates of grid
    nx, ny, nz = len(dx_ls), len(dy_ls), len(dz_ls)
    nxyz = nz * ny * nx
    x_ls, y_ls, z_ls = list(range(nxyz)), list(range(nxyz)), list(range(nxyz))
    cx, cy, cz = 0.0, 0.0, 0.0
    for dx in dx_ls:
        cx += 0.5 * dx
        for dy in dy_ls:
            cy += 0.5 * dy
            for dz in dz_ls:
                cz += 0.5 * dz
                x_ls.append(cx)
                y_ls.append(cy)
                z_ls.append(cz)

    # generate topology data
    topo_ls: List = [int]
    for m in range(nxyz):
        # get rectangular coordinates of each grid center
        xc = x_origin + x_ls[m]
        yc = y_origin - y_ls[m]
        elvc = elv_origin + z_ls[m]

        # get grid size δx and δy
        i, j, _ = __calc_ijk(m, nz, ny)
        dx, dy = dx_ls[i], dy_ls[j]

        # get DEM data within the grid size
        x_dem_ls, y_dem_ls, elv_dem_ls = __clip_xy(
            x_dem_arr, y_dem_arr, elv_dem_arr, (xc - dx, xc + dx, yc - dy, yc + dy)
        )

        # get sea data within the grid size
        x_sea_ls, y_sea_ls, elv_sea_ls = __clip_xy(
            x_sea_arr, y_sea_arr, elv_sea_arr, (xc - dx, xc + dx, yc - dy, yc + dy)
        )

        # get lake data within the grid size
        x_lake_ls, y_lake_ls, elv_lake_ls = __clip_xy(
            x_lake_arr, y_lake_arr, elv_lake_arr, (xc - dx, xc + dx, yc - dy, yc + dy)
        )

        # assign the topology with the largest area
        area_dem = RES_DEM * float(len(elv_dem_ls))
        area_sea = RES_SEA * float(len(elv_sea_ls))
        area_lake = RES_LAKE * float(len(elv_lake_ls))

        # if empty
        _idx: int = None
        if area_dem == area_sea == area_lake == 0.0:
            _idx = 0
        else:
            _idx = np.argmax([area_dem, area_sea, area_lake])
        assert _idx is not None, _idx

        # determine actnum
        topo_idx = None

        # refer DEM
        if _idx == 0:
            _median = __median(elv_dem_ls)
            # above DEM: AIR
            if _median > elvc:
                topo_idx = IDX_AIR
            # below DEM: LAND
            else:
                topo_idx = IDX_LAND
        # refer SEA
        elif _idx == 1:
            _median = __median(elv_sea_ls)
            # above 0m: AIR
            if elvc > 0.0:
                topo_idx = IDX_AIR
            # within the range of seabed and 0m: SEA
            elif _median < elvc <= 0.0:
                topo_idx = IDX_SEA
            # below seabed: LAND
            else:
                topo_idx = IDX_LAND
        # refer LAKE
        else:
            _median = __median(elv_lake_ls)
            _median_topo = __median(elv_dem_ls)
            # Above land topology: AIR
            if elvc > _median_topo:
                topo_idx = IDX_AIR
            # within the range of bottom of the lake and land topology
            elif _median < elvc <= _median_topo:
                topo_idx = IDX_LAKE
            # below bottom of the lake
            else:
                topo_idx = IDX_LAND

        topo_ls.append(topo_idx)

    return topo_ls


def write(_f: TextIO, _string: str):
    _string += "\n"
    _f.write(_string)


def generate_input(gx, gy, gz, act_ls, fpth: str):
    with open(fpth, "w", encoding="utf-8") as _f:
        __write: Callable = partial(write, _f)

        # RUNSPEC
        __write(
            "RUNSPEC   ################### RUNSPEC section begins here ######################"
        )
        __write("")  # \n

        # HCROCK
        __write("HCROCK                                  We enable heat conduction.")
        __write("")  # \n

        # GRIDUNIT
        __write("GRIDUNIT")
        __write("  'METRES' /")
        __write("")  # \n

        # GRID
        __write(
            "GRID      ##################### GRID section begins here #######################"
        )
        __write("")  # \n

        # MAKE
        __write("          The grid is specified within brackets MAKE-ENDMAKE   ")
        __write(
            "MAKE      <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )
        __write("-- cartesian                            We select Cartesia gridding")
        __write(
            "--    grid     nx  ny  nz               option and specify the number of"
        )
        __write(
            f"      CART     {len(gx)}   {len(gy)}   {len(gz)}  /            grid blocks along every axis."
        )
        __write("")  # \n

        # XYZBOUND
        __write("XYZBOUND")
        __write("-- xmin-xmax  ymin-ymax  zmin-zmax     we specify the domain extent.")
        __write(
            f"    0   {sum(gx)}   0    {sum(gy)}     0   {sum(gz)}   /  It is [0,{sum(gx)}]*[0,{sum(gy)}]*[0,{sum(gz)}] meters."
        )
        __write("")  # \n

        # DXV
        __write("DXV")
        _str = ""
        for _x in gx:
            _str += str(_x) + "  "
        _str += " /"
        __write(_str)
        __write("")

        # DYV
        __write("DYV")
        _str = ""
        for _y in gy:
            _str += str(_y) + "  "
        _str += " /"
        __write(_str)
        __write("")

        # DZV
        __write("DZV")
        _str = ""
        for _z in gz:
            _str += str(_z) + "  "
        _str += " /"
        __write(_str)
        __write("")

        # ACTNUM
        __write("ACTNUM")
        _str: str = ""
        for _a in act_ls:
            _str += str(_a) + "  "
        _str += "  /"
        __write(_str)
        __write("")  # \n

        # BOUNDARY
        # TODO
        __write("BOUNDARY                               We define the boundaries:")
        __write(
            "   102   1 10 1 2 5 5 'K+' 5* INFTHIN 4* 2 2 /    the bottom bound. marked as FLUXNUM=102/"
        )
        __write("/")
        __write("")

        # SRCSPECG
        # TODO
        __write("SRCSPECG")
        __write(" ’MAGMASRC’ 5 1 5 /")
        __write("/")
        __write("")

        # ENDMAKE
        __write(
            "ENDMAKE   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        )
        __write("")  # \n

        # EQUALREG
        # TODO
        __write(
            "EQUALREG                              We specify uniform distributions:"
        )
        __write("   PORO      0.25 ROCKNUM 1 /           porosity = 0.25")
        __write(
            "   PERMX     100  ROCKNUM 1 /                     X-permeability = 100 mD"
        )
        __write(
            "   PERMY     100  ROCKNUM 1 /                     Y-permeability = 100 mD"
        )
        __write(
            "   PERMZ     100  ROCKNUM 1 /                     Z-permeability = 100 mD"
        )
        __write(
            "   HCONDCFX  2.   ROCKNUM 1 /                     X-Heat cond. coeff. = 2 W/m/K"
        )
        __write(
            "   HCONDCFY  2.   ROCKNUM 1 /                     Y-Heat cond. coeff. = 2 W/m/K"
        )
        __write(
            "   HCONDCFZ  2.   ROCKNUM 1 /                     Z-Heat cond. coeff. = 2 W/m/K"
        )
        __write("/")
        __write("")  # \n

        # RPTGRID
        __write(
            "RPTGRID                               We define the output form the GRID sect."
        )
        __write("  PORO PERMX PERMZ /")
        __write("")  # \n

        # PROPS
        __write(
            "PROPS     ####################### PROPS section begins here ####################"
        )
        __write("")  # \n

        # ROCK
        __write("          Rock properties are specified within brackets ROCK-ENDROCK")
        __write(
            "ROCK      <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )
        __write("  1  /")
        __write("")  # \n

        # TODO
        __write("ROCKDH                                  We specify that")
        __write(
            "  2900  0.84 /                          rock density is 2900 kg/m3, rock heat capacity is 0.84 kJ/kg/K"
        )
        __write("")  # \n

        # ENDROCK
        __write(
            "ENDROCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        )
        __write("")  # \n

        # LOADEOS
        __write("LOADEOS                                 We load the EOS file.")
        __write("   './CO2H2O_V3.0.EOS' /")
        __write("")  # \n

        # SAT
        # TODO: confirm
        __write(
            "         The relative permeabilities are specified within brackets SAT-ENDSAT"
        )
        __write(
            "SAT      <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )
        __write("   /")
        __write("")  # \n

        # SATTAB
        __write("SATTAB")
        __write("    0.0    0.0   1.0  /           krliq=sliq^3")
        __write("    0.1    0.001 0.81 /           krgas=(1-sliq)^2")
        __write("    0.2    0.008 0.64 /")
        __write("    0.4    0.064 0.36 /")
        __write("    0.6    0.216 0.16 /")
        __write("    0.8    0.512 0.04 /")
        __write("    0.9    0.729 0.01 /")
        __write("    1.0    1.0   0.0  /")
        __write("/")
        __write("")  # \n

        # ENDSAT
        __write(
            "ENDSAT    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        )
        __write("")  # \n

        # PHASES
        """The keyword PHASES defines new MNEMONICS for data output. This keyword
        is followed by tabular data and every line of these data defines an
        individual phase. The 1st column is the phase name (not more
        than 4 characters). The following data items define typical
        thermodynamic parameters of the phase. The 2nd column is
        pressure, the 3rd is the enthalpy, the 4th and 5th are the molar
        composition of binary mixture.
        We define two phases. The 1st is the liquid water (LH2O). The 2nd is
        the supercritical CO2 phase.
        """
        __write("PHASES                               We define phases for output:")
        __write("-- name   pres  enth  CO2 H2O")
        __write("--       [MPa] [kJ/mol] CO2 H2O")
        __write("   LIQ    1.0    5.0  0.0 1.0 /        - liquid water")
        __write("   GAS    1.0   60.0  0.0 1.0 /        - water vapor")
        __write("/")
        __write("")  # \n

        # INIT
        __write(
            "INIT      ####################### INIT section begins here #####################"
        )
        __write("")  # \n

        # REGALL
        __write(
            "REGALL      We enable application of the following two keywords both to domain grid blocks and boundary grid blocks."
        )
        __write("")  # \n

        # ECHO
        __write("ECHO      Enables more information output in the LOG-file")
        __write("")  # \n

        # OPERAREG
        __write("OPERAREG")
        __write(
            "   PRES DEPTH  SATNUM  1 MULTA  0.2  0.0099 /          PRES=0.2+0.0099*DEPTH"
        )
        __write("/")
        __write("")  # \n

        # EQUALREG
        # TODO
        __write("EQUALREG")
        __write("   TEMPC   20  ROCKNUM 1   /     The initial temperature is 20 C  ")
        __write("   COMP1T  0.0             /     No CO2 is present ")
        __write(
            "   TEMPC   200 FLUXNUM 102 /     The temperature of the bottom boundary is 200 C"
        )
        __write("   PERMZ   0.0 FLUXNUM 102 /")
        __write("/")
        __write("")  # \n

        # EQUALNAM
        __write("EQUALNAM")
        __write("  PRES 10. ’MAGMASRC’ /")
        __write("  TEMPC 400. ’MAGMASRC’/")
        __write("  COMP1T 0.5 ’MAGMASRC’/")
        __write("/")
        __write("")  # \n
        __write("")  # \n

        # RPTSUM
        __write("RPTSUM")
        __write(
            "   PRES TEMPC PHST SAT#LIQ SAT#GAS /  We specify the properties saved at every report time."
        )
        __write("")  # \n

        # SCHEDULE
        __write(
            "SCHEDULE   #################### SCHEDULE section begins here ####################"
        )
        __write("")  # \n

        # SRCINJE
        __write("SRCINJE")
        __write("  ’MAGMASRC’ MASS 1* 25. 1* 500. /")
        __write("/")
        __write("")  # \n

        # TUNING
        # TODO:
        __write(
            "TUNING                        We specify that the maximal timestep is 1000 days and the initial timestep is 0.1 days."
        )
        __write("    1* 1000 0.1 /")
        __write("")  # \n

        # TSTEP
        # TODO:
        __write(
            "TSTEP                         We advance simulation to 100000 days reporting distributions every 1000 days."
        )
        __write("10*10000 /")
        __write("")  # \n

        # REPORTS
        __write("REPORTS")
        __write("   CONV MATBAL LINSOL  /")
        __write("")  # \n

        # POST
        __write(
            "POST      ####################### POST section begins here ######################"
        )
        __write("")  # \n

        # CONVERT
        __write(
            "CONVERT                                 We convert the output to ParaView compatible format."
        )
        __write("")  # \n

        # POSTSRC
        __write("POSTSRC")
        __write("  MAGMASRC /")
        __write("/")
        __write("")  # \n

        # END
        __write(
            "END       #######################################################################"
        )


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

if __name__ == "__main__":
    # lat, lng, elv = load_sea_dem("seadem")
    # lat_arr = np.array(lat)
    # lng_arr = np.array(lng)
    # elv_arr = np.array(elv)
    # filt = (
    #     (42.674796 < lat_arr)
    #     * (lat_arr < 42.818721)
    #     * (141.257462 < lng_arr)
    #     * (lng_arr < 141.427571)
    # )
    # lat_lake = lat_arr[filt]
    # lng_lake = lng_arr[filt]
    # elv_lake = elv_arr[filt] - 220.73
    # print(elv_lake.mean())
    # fig, ax = plt.subplots()
    # elv_min = elv_lake.min()
    # # print([i / elv_min for i in elv_lake])
    # # colors = [cm.jet(i / elv_min) for i in elv_lake]
    # mappable = ax.scatter(lng_lake, lat_lake, c=elv_lake, cmap="coolwarm")
    # fig.colorbar(mappable)
    # plt.show()

    #! INPUT
    pth_dem = "./dem"
    pth_sea = "./seadem"
    crs_rect = "epsg:6680"
    dxyz = (
        [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
    )
    origin = (42.688156, 141.379868, 1041.0)

    topo_ls = generate_topo(pth_dem, pth_sea, crs_rect, dxyz, origin)

    actnum_ls = []
    for _idx in topo_ls:
        actnum: int = None
        if _idx == IDX_LAND:
            actnum = 1
        else:
            actnum = 2
        actnum_ls.append(actnum)

    print("topo_ls")  #!
    print(topo_ls)  #!

    generate_input(dxyz[0], dxyz[1], dxyz[2], actnum_ls, "tmp2.RUN")
