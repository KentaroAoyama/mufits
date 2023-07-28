#!/usr/bin/env python
# coding: utf-8

"""Generate input file for MUFITS (Afanasyev, 2012)
"""

from functools import partial
from typing import List, Tuple, Dict, TextIO, Callable
from pathlib import Path
from os import PathLike, makedirs
import pickle

import pandas as pd
from pyproj import Transformer
import numpy as np
from tqdm import tqdm

from utils import calc_ijk, calc_m, plt_topo

from constants import (
    LAKE_BOUNDS,
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
    TOPO_CONST_PROPS,
    TOPO_INIT_PROPS,
    P_GROUND,
    P_GRAD_AIR,
    P_GRAD_SEA,
    P_GRAD_LAKE,
    P_GRAD_ROCK,
    P_BOTTOM,
)
from params import (
    DEM_PTH,
    SEADEM_PTH,
    CRS_RECT,
    RUNFILE_PTH,
    ALIGN_CENTER,
    DXYZ,
    ORIGIN,
    POS_SRC,
    PRES_SRC,
    SRC_TEMP,
    SRC_COMP1T,
)

# TODO: 天水のinput


def __clip_xy(
    _x: List, _y: List, _elv: List, bounds: Tuple[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _x_arr = np.array(_x)
    _y_arr = np.array(_y)
    _elv_arr = np.array(_elv)
    filt = (
        (bounds[0] < _x_arr)
        * (_x_arr < bounds[1])
        * (bounds[2] < _y_arr)
        * (_y_arr < bounds[3])
    )
    return _x_arr[filt], _y_arr[filt], _elv_arr[filt]


def __clip_lake(_lat: List, _lng: List, _elv: List) -> Tuple[List, List, List]:
    lat_arr = np.array(_lat)
    lng_arr = np.array(_lng)
    elv_arr = np.array(_elv)
    lat_lake, lng_lake, elv_lake = [], [], []
    lat_sea, lng_sea, elv_sea = [], [], []
    for _, (lat0, lat1, lng0, lng1, dh) in LAKE_BOUNDS.items():
        filt = (lat0 < lat_arr) * (lat_arr < lat1) * (lng0 < lng_arr) * (lng_arr < lng1)
        lat_lake.extend(lat_arr[filt].tolist())
        lng_lake.extend(lng_arr[filt].tolist())
        elv_lake.extend((elv_arr[filt] + dh).tolist())
        lat_sea.extend(lat_arr[np.logical_not(filt)].tolist())
        lng_sea.extend(lng_arr[np.logical_not(filt)].tolist())
        elv_sea.extend(elv_arr[np.logical_not(filt)].tolist())
    return lat_lake, lng_lake, elv_lake, lat_sea, lng_sea, elv_sea


def load_dem(pth_dem: PathLike) -> Tuple[List, List, List]:
    base_dir = Path(pth_dem)
    fpth = base_dir.joinpath("out.xyz")
    assert fpth.exists(), fpth
    _df = pd.read_csv(fpth, sep="\s+", header=None)
    return _df[1].tolist(), _df[0].tolist(), _df[2].tolist()


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
    return np.nanmedian(_ls)


def __stack_from_0(_ls: List[float]) -> List[float]:
    c_ls = []
    for i, _d in enumerate(_ls):
        if len(c_ls) == 0:
            c_ls.append(abs(_d) * 0.5)
            continue
        c_ls.append(c_ls[-1] + _ls[i - 1] * 0.5 + abs(_d) * 0.5)
    return c_ls


def __stack_from_center(_ls: List[float]) -> List[float]:
    n = len(_ls)
    if divmod(n, 2)[1] == 0:
        sum_left = -sum(_ls[: int(n * 0.5)])
    else:
        _lhalf = int(n * 0.5) + 1
        sum_left = -sum(_ls[:_lhalf]) + 0.5 * _ls[_lhalf]
    c_ls: List = []
    for _d in _ls:
        sum_left += _d * 0.5
        c_ls.append(sum_left)
        sum_left += _d * 0.5
    return c_ls


def generate_topo(
    pth_dem: PathLike,
    pth_seadem: PathLike,
    crs_rect: str,
    dxyz: Tuple[List, List, List],
    origin: Tuple[float, float, float],
    pos_src: Tuple[float, float, float],
    align_center: bool = True,
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
            lat_dem_ls, lng_dem_ls, elv_dem_ls = pickle.load(pkf)
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
            lat_sea_ls, lng_sea_ls, elv_sea_ls = pickle.load(pkf)
    else:
        lat_sea_ls, lng_sea_ls, elv_sea_ls = load_sea_dem(pth_seadem)
        with open(cache_sea, "wb") as pkf:
            pickle.dump(
                (lat_sea_ls, lng_sea_ls, elv_sea_ls), pkf, pickle.HIGHEST_PROTOCOL
            )

    # clip lake topography
    (
        lat_lake_ls,
        lng_lake_ls,
        elv_lake_ls,
        lat_sea_ls,
        lng_sea_ls,
        elv_sea_ls,
    ) = __clip_lake(lat_sea_ls, lng_sea_ls, elv_sea_ls)

    # convert to rect crs (WGS84 to crs_rect)
    transformer_wgs = Transformer.from_crs(CRS_WGS84, crs_rect, always_xy=True)
    transformer_dem = Transformer.from_crs(CRS_DEM, crs_rect, always_xy=True)
    transformer_sea = Transformer.from_crs(CRS_SEA, crs_rect, always_xy=True)
    transformer_lake = Transformer.from_crs(CRS_LAKE, crs_rect, always_xy=True)
    transformer_inv = Transformer.from_crs(crs_rect, CRS_WGS84, always_xy=True)

    # calculate the coordinates of the grid center
    x_origin, y_origin = transformer_wgs.transform(origin[1], origin[0])
    elv_origin = origin[2]
    xc_ls, yc_ls, zc_ls = None, None, None
    if align_center:
        xc_ls = __stack_from_center(dx_ls)
        yc_ls = __stack_from_center(dy_ls)
        print(xc_ls, yc_ls)
        zc_ls = __stack_from_0(dz_ls)
    else:
        xc_ls = __stack_from_0(dx_ls)
        yc_ls = __stack_from_0(dy_ls)
        zc_ls = __stack_from_0(dz_ls)

    # get src position
    latsrc, lngsrc, elvsrc = pos_src
    xsrc, ysrc = transformer_wgs.transform(lngsrc, latsrc)
    isrc = np.argmin(np.square(np.array(xc_ls) - (xsrc - x_origin)))
    jsrc = np.argmin(np.square(np.array(yc_ls) - (y_origin - ysrc)))
    ksrc = np.argmin(np.square(np.array(zc_ls) - (elv_origin - elvsrc)))

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
    # generate topology data
    topo_ls: List[int] = np.zeros(shape=nz * ny * nx).tolist()
    lat_2d: List = np.zeros(shape=(ny, nx)).tolist()
    lng_2d: List = np.zeros(shape=(ny, nx)).tolist()
    for i in tqdm(range(nx)):
        for j in range(ny):
            # get grid size δx and δy
            dx, dy = dx_ls[i] * 0.5, dy_ls[j] * 0.5
            xc = x_origin + xc_ls[i]
            yc = y_origin - yc_ls[j]
            lngc, latc = transformer_inv.transform(xc, yc)
            lat_2d[j][i] = latc
            lng_2d[j][i] = lngc

            # get DEM data within the grid size
            bounds = (xc - dx, xc + dx, yc - dy, yc + dy)
            _, _, elv_dem_tmp = __clip_xy(x_dem_arr, y_dem_arr, elv_dem_arr, bounds)

            # get sea data within the grid size
            _, _, elv_sea_tmp = __clip_xy(x_sea_arr, y_sea_arr, elv_sea_arr, bounds)

            # get lake data within the grid size
            _, _, elv_lake_tmp = __clip_xy(
                x_lake_arr,
                y_lake_arr,
                elv_lake_arr,
                bounds,
            )

            # assign the topology with the largest area
            area_dem = RES_DEM * float(len(elv_dem_tmp))
            area_sea = RES_SEA * float(len(elv_sea_tmp))
            area_lake = RES_LAKE * float(len(elv_lake_tmp))
            median_land = __median(elv_dem_tmp)
            median_sea = __median(elv_sea_tmp)
            median_lake = __median(elv_lake_tmp)
            # if empty
            _idx: int = None
            if area_dem == area_sea == area_lake == 0.0:
                _idx = 0
            else:
                _idx = np.argmax([area_dem, area_sea, area_lake])
            assert _idx is not None, _idx
            for k in range(nz):
                # get rectangular coordinates of each grid center
                elvc = elv_origin - zc_ls[k]
                # determine actnum
                topo_idx = None
                # refer DEM
                if _idx == 0:
                    # below DEM: LAND
                    if median_land > elvc:
                        topo_idx = IDX_LAND
                    # above DEM: AIR
                    else:
                        topo_idx = IDX_AIR
                # refer SEA
                elif _idx == 1:
                    # above 0m: AIR
                    if elvc > 0.0:
                        topo_idx = IDX_AIR
                    # within the range of seabed and 0m: SEA
                    elif median_sea < elvc <= 0.0:
                        topo_idx = IDX_SEA
                    # below seabed: LAND
                    else:
                        topo_idx = IDX_LAND
                # refer LAKE
                else:
                    # Above land topology: AIR
                    if elvc > median_land:
                        topo_idx = IDX_AIR
                    # within the range of bottom of the lake and land topology
                    elif median_lake < elvc <= median_land:
                        topo_idx = IDX_LAKE
                    # below bottom of the lake
                    else:
                        topo_idx = IDX_LAND
                topo_ls[calc_m(i, j, k, nx, ny)] = topo_idx

    return topo_ls, (lat_2d, lng_2d, isrc, jsrc, ksrc)


def generate_act_ls(topo_ls: List[int]) -> List[int]:
    actnum_ls = []
    for _idx in topo_ls:
        actnum: int = None
        if _idx == IDX_LAND:
            # activate
            actnum = 1
        else:
            # inactive
            actnum = 2
        actnum_ls.append(actnum)
    return actnum_ls


def generamte_rocknum_and_props(
    topo_ls: List[int], top: float, nxyz: Tuple, gz: List
) -> Tuple[List, Dict, Dict, Dict]:
    topo_unique = []
    for _idx in topo_ls:
        if _idx not in topo_unique:
            topo_unique.append(_idx)  # must maintain order
    topo_rocknum_map = {_idx: rocknum for rocknum, _idx in enumerate(topo_unique)}
    rocknum_ls = []
    rocknum_consts = {}
    rocknum_inits = {}
    for _idx in topo_ls:
        rocknum = topo_rocknum_map[_idx]
        rocknum_ls.append(rocknum)
        rocknum_consts.setdefault(rocknum, TOPO_CONST_PROPS[_idx])
        rocknum_inits.setdefault(rocknum, TOPO_INIT_PROPS[_idx])

    rocknum_pgrad = {}
    for _idx in topo_unique:
        _prop = rocknum_pgrad.setdefault(topo_rocknum_map[_idx], {})
        if _idx == IDX_LAND:
            _prop["a"] = P_GROUND
            _prop["b"] = P_GRAD_ROCK
        if _idx == IDX_AIR:
            _prop["a"] = P_GROUND - top * P_GRAD_AIR
            _prop["b"] = P_GRAD_AIR
        if _idx == IDX_LAKE:
            elv_lake: float = None
            for m, _idx_tmp in enumerate(topo_ls):
                if _idx_tmp == IDX_LAKE:
                    _, _, k = calc_ijk(m, nxyz[0], nxyz[1])
                    elv_lake = top - sum(gz[: k + 1])
                    break
            _prop["a"] = P_GROUND - elv_lake * P_GRAD_AIR
            _prop["b"] = P_GRAD_LAKE
        if _idx == IDX_SEA:
            _prop["a"] = P_GROUND
            _prop["b"] = P_GRAD_SEA
    return rocknum_ls, rocknum_consts, rocknum_inits, rocknum_pgrad


def calc_sattab(method="corey") -> Dict:
    satab: Tuple = None
    if method == "corey":
        slr = 0.33
        sgr = 0.05
        sl_ls = np.linspace(0.0, 1.0, 20).tolist()
        kl_ls, kg_ls = [], []
        for sl in sl_ls:
            s = (sl - slr) / (1.0 - slr - sgr)
            kl = s**4
            kg = (1.0 - s**2) * (1.0 - s) ** 2
            kl_ls.append(kl)
            kg_ls.append(kg)
        satab = (sl_ls, kl_ls, kg_ls)
    else:
        # default settings described in Manual
        # krliq=sliq^3, krgas=(1-sliq)^2
        sl_ls = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        kl_ls = [0.0, 0.001, 0.008, 0.064, 0.216, 0.512, 0.729, 1.0]
        kg_ls = [1.0, 0.81, 0.64, 0.36, 0.16, 0.04, 0.01, 0.0]
        satab = (sl_ls, kl_ls, kg_ls)
    return satab


def get_air_bounds(topo_ls, nxyz):
    nx, ny, _ = nxyz
    m_bounds: List = []
    for m, _idx in enumerate(topo_ls):
        if _idx != IDX_LAND:
            continue
        i, j, k = calc_ijk(m, nx, ny)
        m_above = calc_m(i, j, k - 1, nx, ny)
        if k == 0:
            m_bounds.append(m)
            continue
        if topo_ls[m_above] == IDX_AIR:
            m_bounds.append(m)
    return m_bounds


def write(_f: TextIO, _string: str):
    _string += "\n"
    _f.write(_string)


def generate_input(
    gx: List,
    gy: List,
    gz: List,
    act_ls: List,
    rocknum_ls,
    rocknum_consts: Dict,
    rocknum_params: Dict,
    rocknum_pres_grad: Dict,
    m_airbounds: List[int],
    sattab: Tuple,
    src_props: Dict,
    fpth: str,
):
    fluxnum = 100
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
        __write("BOUNDARY                               We define the boundaries:")
        # __write(f"   2   6* 'I-'  'I+'  'J-'  'J+'  2* INFTHIN /    <- Aquifer")
        # __write(
        #     f"   3   6* 'K-'  5*                   INFTHIN   4* 1* 2 /    <- Top boundary"
        # )
        __write(
            f"   102   6* 'K+'  5*                   INFTHIN   4* 2  2 /    <- Bottom boundary"
        )
        # AIR - LAND boundary
        # nx, ny, _ = nxyz
        # for m in m_airbounds:
        #     i, j, k = calc_ijk(m, nx, ny)
        #     i += 1
        #     j += 1
        #     k += 1
        #     __write(
        #         f"   {fluxnum}   {i}   {i+1}   {j}   {j+1}   {k}   {k}   6* 'K-'  5*                   INFTHIN   4* 1* 2 /"
        #     )
        # fluxnum += 1
        __write("/")
        __write("")

        # SRCSPECG
        __write("SRCSPECG")
        srci, srcj, srck = src_props["i"], src_props["j"], src_props["k"]
        __write(f" ’MAGMASRC’ {srci} {srcj} {srck} /")
        __write("/")
        __write("")

        # ENDMAKE
        __write(
            "ENDMAKE   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        )
        __write("")  # \n

        # EQUALREG
        __write(
            "EQUALREG                              We specify uniform distributions:"
        )
        for rocknum, props in rocknum_consts.items():
            for k, v in props.items():
                # DENS and HC should be written in ROCKDH section
                if k in ("DENS", "HC"):
                    continue
                __write(f"{k} {v}" + " " + f"ROCKNUM {rocknum}  /")
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
        for idx, prop in rocknum_consts.items():
            dens, hc = prop["DENS"], prop["HC"]
            __write(
                "          Rock properties are specified within brackets ROCK-ENDROCK"
            )
            __write(
                "ROCK      <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            )
            __write(f"  {idx} /")
            __write("ROCKDH                                  We specify that")
            __write(
                f"  {dens}  {hc} /                          rock density is {dens} kg/m3, rock heat capacity is {hc} kJ/kg/K"
            )
            __write("")  # \n

            # ENDROCK
            __write(
                "ENDROCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            )
            __write("")  # \n

        # ROCKNUM
        __write("ROCKNUM")
        _str: str = "  "
        for _r in rocknum_ls:
            _str += str(_r) + "  "
        _str += "  /"
        __write(_str)
        __write("")  # \n

        # LOADEOS
        __write("LOADEOS                                 We load the EOS file.")
        __write("   './CO2H2O_V3.0.EOS' /")
        __write("")  # \n

        # SAT
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
        if satab is None:
            __write("    0.0    0.0   1.0  /           krliq=sliq^3")
            __write("    0.1    0.001 0.81 /           krgas=(1-sliq)^2")
            __write("    0.2    0.008 0.64 /")
            __write("    0.4    0.064 0.36 /")
            __write("    0.6    0.216 0.16 /")
            __write("    0.8    0.512 0.04 /")
            __write("    0.9    0.729 0.01 /")
            __write("    1.0    1.0   0.0  /")
        else:
            for sl, kl, kg in zip(*sattab):
                __write(f"    {sl}    {kl}    {kg}  /")
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

        # OPERAREG (set initial value)
        __write("OPERAREG")
        for rocknum, prop in rocknum_pres_grad.items():
            a, b = prop["a"], prop["b"]
            __write(f"   PRES DEPTH  ROCKNUM  {rocknum} MULTA  {a}  {b}  /")
        __write("/")
        __write("")  # \n

        # EQUALREG
        __write("EQUALREG")
        for rocknum, params in rocknum_params.items():
            for _str in params:
                __write(_str + "  " + f"ROCKNUM  {rocknum}  /")

        __write(
            f"   PRES   {P_BOTTOM} FLUXNUM 102 /     The pressure of the bottom boundary"
        )
        __write("/")
        __write("")  # \n

        # EQUALNAM
        __write("EQUALNAM")
        pres, tempe, comp1t = src_props["pres"], src_props["tempe"], src_props["comp1t"]
        __write(f"  PRES {pres} ’MAGMASRC’ /")
        __write(f"  TEMPC {tempe} ’MAGMASRC’/")
        __write(f"  COMP1T {comp1t} ’MAGMASRC’/")
        __write("/")
        __write("")  # \n
        __write("")  # \n

        # RPTSUM
        __write("RPTSUM")
        __write(
            "   PRES TEMPC PHST SAT#LIQ SAT#GAS /  We specify the properties saved at every report time."
        )
        __write("")  # \n

        #  RPTSRC
        __write("RPTSRC")
        __write("  SMIR#1 SMIR#2 SMIT#1 SMIT#2 /")
        __write("")  # \n

        # SCHEDULE
        __write(
            "SCHEDULE   #################### SCHEDULE section begins here ####################"
        )
        __write("")  # \n

        # SRCINJE
        __write("SRCINJE")
        __write("  ’MAGMASRC’ MASS 1* 50. 1* 2000. /")
        __write("/")
        __write("")  # \n

        # TUNING
        __write(
            "TUNING                        We specify that the maximal timestep is 1000 days and the initial timestep is 0.01 days."
        )
        __write("    1* 1000 0.01 /")
        __write("")  # \n

        # TSTEP
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


if __name__ == "__main__":
    assert len(DXYZ[0]) * len(DXYZ[1]) * len(DXYZ[2]) < 25000, (
        len(DXYZ[0]) * DXYZ(DXYZ[1]) * DXYZ(DXYZ[2])
    )
    nxyz = (len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2]))

    topo_ls, (lat_2d, lng_2d, isrc, jsrc, ksrc) = generate_topo(
        DEM_PTH,
        SEADEM_PTH,
        CRS_RECT,
        DXYZ,
        ORIGIN,
        POS_SRC,
        align_center=ALIGN_CENTER,
    )

    # debug
    plt_topo(topo_ls, lat_2d, lng_2d, nxyz, "debug")

    actnum_ls = generate_act_ls(topo_ls)
    (
        rocknum_ls,
        rocknum_consts,
        rocknum_inits,
        rocknum_pres_grad,
    ) = generamte_rocknum_and_props(topo_ls, ORIGIN[2], nxyz, DXYZ[2])

    # get boundary region
    m_airbounds = get_air_bounds(topo_ls, nxyz)

    # relative permeability
    satab = calc_sattab(method="None")

    src_props = {
        "i": isrc + 1,
        "j": jsrc + 1,
        "k": ksrc + 1,
        "pres": PRES_SRC,
        "tempe": SRC_TEMP,
        "comp1t": SRC_COMP1T,
    }
    generate_input(
        DXYZ[0],
        DXYZ[1],
        DXYZ[2],
        actnum_ls,
        rocknum_ls,
        rocknum_consts,
        rocknum_inits,
        rocknum_pres_grad,
        m_airbounds,
        satab,
        src_props,
        RUNFILE_PTH,
    )
