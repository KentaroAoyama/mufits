#!/usr/bin/env python
# coding: utf-8
"""Generate input file for MUFITS (Afanasyev, 2012)
"""

from functools import partial
from typing import List, Tuple, Dict, TextIO, Callable
from pathlib import Path
from os import PathLike, makedirs, getcwd
import pickle
from math import exp, log10
from statistics import mean
from copy import deepcopy

import pandas as pd
from pyproj import Transformer
import numpy as np
from tqdm import tqdm
from shapely import Polygon, Point

from utils import (
    calc_ijk,
    calc_m,
    mdarcy2si,
    si2mdarcy,
    plt_topo,
    plt_airbounds,
    calc_k_z,
    plt_any_val,
    stack_from_0,
    stack_from_center,
    calc_press_air,
    calc_xco2_rain,
    calc_kh,
    calc_kv,
)

from constants import (
    ORIGIN,
    POS_SRC,
    POS_SINK,
    DEM_PTH,
    SEADEM_PTH,
    CRS_RECT,
    ALIGN_CENTER,
    DXYZ,
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
    IDX_VENT,
    IDX_CAP,
    IDX_CAPVENT,
    CACHE_DIR,
    CACHE_DEM_FILENAME,
    CACHE_SEA_FILENAME,
    TOPO_CONST_PROPS,
    TEMPE_AIR,
    G,
    P_GROUND,
    P_GRAD_AIR,
    P_GRAD_ROCK,
    P_GRAD_LAKE,
    P_GRAD_SEA,
    T_GRAD_ROCK,
    TIME_SS,
    TSTEP_INIT,
    TSTEP_MIN,
    TSTEP_MAX,
    NDTFIRST,
    NDTEND,
    TMULT,
    SINK_PARAMS,
    PERM_MAX,
    TSTEP_UNREST,
    TRPT_UNREST,
    TEND_UNREST,
)

from params import PARAMS, TUNING_PARAMS
from monitor import load_sum, get_v_ls


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


def __generate_rain_src_id(idx: int):
    return "R" + "{:07d}".format(idx)


def generate_simple_vent(
    topo_ls: List, xc_m: List, yc_m: List, elvc_m: List, vent_bounds: Dict
) -> List:
    topo_arr: np.ndarray = np.array(topo_ls)
    xc_m: np.ndarray = np.array(xc_m)
    yc_m: np.ndarray = np.array(yc_m)
    elvc_m: np.ndarray = np.array(elvc_m)
    zc_ls = stack_from_0(DXYZ[2])
    zc_arr = ORIGIN[2] - np.array(zc_ls)
    for elvc, bounds in vent_bounds.items():
        dz = DXYZ[2][np.argmin(np.square(zc_arr - elvc))]
        filt = (
            ((elvc - dz * 0.5) < elvc_m)
            & (elvc_m < (elvc + dz * 0.5))
            & (bounds[0] - 25.0 < xc_m)
            & (xc_m < bounds[1] + 25.0)
            & (bounds[2] - 25.0 < yc_m)
            & (yc_m < bounds[3] + 25.0)
        )
        topo_arr = np.where(filt, IDX_VENT, topo_arr)
    return topo_arr.tolist()


def generate_simple_cap(
    topo_ls: List,
    xc_m: List,
    yc_m: List,
    elv_cap: float,
    cap_bounds: Polygon,
) -> List:
    zc_ls = stack_from_0(DXYZ[2])
    zc_arr = ORIGIN[2] - np.array(zc_ls)
    k = np.argmin(np.square(zc_arr - elv_cap))
    transformer_wgs = Transformer.from_crs(CRS_WGS84, CRS_RECT, always_xy=True)
    x0, y0 = transformer_wgs.transform(ORIGIN[1], ORIGIN[0])
    nx, ny = len(DXYZ[0]), len(DXYZ[1])
    for i in range(nx):
        for j in range(ny):
            m = calc_m(i, j, k, nx, ny)
            xtmp = xc_m[m] + x0
            ytmp = yc_m[m] + y0
            if cap_bounds.contains(Point(xtmp, ytmp)):
                topo_ls[m] = IDX_CAP
                if topo_ls[calc_m(i, j, k - 1, nx, ny)] == IDX_VENT and k > 0:
                    topo_ls[m] = IDX_CAPVENT
    return topo_ls


def generate_topo(
    pth_dem: PathLike,
    pth_seadem: PathLike,
    crs_rect: str,
    dxyz: Tuple[List, List, List],
    origin: Tuple[float, float, float],
    pos_src: Tuple[float, float, float],
    pos_sink: Dict,
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
        xc_ls = stack_from_center(dx_ls)
        yc_ls = stack_from_center(dy_ls)
        zc_ls = stack_from_0(dz_ls)
    else:
        xc_ls = stack_from_0(dx_ls)
        yc_ls = stack_from_0(dy_ls)
        zc_ls = stack_from_0(dz_ls)

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
    # center of the coordinates sorted for indecies "m"
    xc_m: List[int] = np.zeros(shape=nz * ny * nx).tolist()
    yc_m: List[int] = np.zeros(shape=nz * ny * nx).tolist()
    zc_m: List[int] = np.zeros(shape=nz * ny * nx).tolist()
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
                m = calc_m(i, j, k, nx, ny)
                topo_ls[m] = topo_idx
                xc_m[m] = xc_ls[i]
                yc_m[m] = -yc_ls[j]
                zc_m[m] = elvc

    # get sink position
    pos_sink_ijk = {}
    for name, pos in pos_sink.items():
        latsink, lngsink, _ = pos
        xsink, ysink = transformer_wgs.transform(lngsink, latsink)
        isink = np.argmin(np.square(np.array(xc_ls) - (xsink - x_origin)))
        jsink = np.argmin(np.square(np.array(yc_ls) - (y_origin - ysink)))
        ksink = None
        for k in range(nz):
            m = calc_m(isink, jsink, k, nx, ny)
            if topo_ls[m] in (IDX_LAND, IDX_VENT, IDX_CAP, IDX_CAPVENT):
                ksink = k
                break
        pos_sink_ijk.setdefault(name, (isink, jsink, ksink))
    return topo_ls, (xc_m, yc_m, zc_m, lat_2d, lng_2d, (isrc, jsrc, ksrc), pos_sink_ijk)


def generate_act_ls(topo_ls: List[int]) -> List[int]:
    actnum_ls = []
    for _idx in topo_ls:
        actnum: int = None
        if _idx in (IDX_AIR, IDX_LAKE, IDX_SEA):
            # inactivate
            actnum = 2
        else:
            # activate
            actnum = 1
        actnum_ls.append(actnum)
    return actnum_ls


def fix_perm(v: float) -> float:
    """Correct the permeability to a reasonable value.

    Args:
        v (float): Permeability in mD

    Returns:
        float: Permeability in mD
    """
    if mdarcy2si(v) > PERM_MAX:
        v = si2mdarcy(PERM_MAX)
    if v < 0.0:
        v = si2mdarcy(PERM_MAX)
    return v


def generamte_rocknum_and_props(
    topo_ls: List[int],
    top: float,
    nxyz: Tuple,
    gz: List,
    topo_props: Dict,
    vent_scale: float,
    cap_scale: float,
    vk: bool,
) -> Tuple[List, Dict, Dict, Dict, Dict]:
    nx, ny, nz = nxyz
    topo_unique = []
    for _idx in topo_ls:
        if _idx not in topo_unique:
            topo_unique.append(_idx)  # must maintain order
    topo_rocknum_map = {_idx: rocknum for rocknum, _idx in enumerate(topo_unique)}
    rocknum_ls = []
    rocknum_params = {}
    for _idx in topo_ls:
        rocknum = topo_rocknum_map[_idx]
        rocknum_ls.append(rocknum)
        rocknum_params.setdefault(rocknum, topo_props[_idx])

    # permeability
    zeros: List = np.zeros(len(topo_ls)).tolist()
    permx_ls, permy_ls, permz_ls = deepcopy(zeros), deepcopy(zeros), deepcopy(zeros)
    for j in range(ny):
        for i in range(nx):
            z0, z1 = 0.0, 0.0
            for k in range(nz):
                m = calc_m(i, j, k, nx, ny)
                if z0 == 0.0:
                    z0 = 1.0
                z1 = z0 + gz[k]
                _idx = topo_ls[m]
                # depth from earth surface (m)
                if _idx not in (IDX_LAND, IDX_VENT, IDX_CAP, IDX_CAPVENT):
                    permx_ls[m] = topo_props[_idx]["PERMX"]
                    permy_ls[m] = topo_props[_idx]["PERMY"]
                    permz_ls[m] = topo_props[_idx]["PERMZ"]
                    continue
                if _idx == IDX_VENT:
                    _k = fix_perm(calc_k_z((z0 + z1) * 0.5))
                    kh, kv = None, None
                    if vk:
                        kh = _k
                        kv = fix_perm(vent_scale * _k)
                    else:
                        kh = fix_perm(vent_scale * _k)
                        kv = kh
                    permx_ls[m] = kh
                    permy_ls[m] = kh
                    permz_ls[m] = kv
                elif _idx == IDX_CAP:
                    v = si2mdarcy(1.0e-17)
                    permx_ls[m] = v
                    permy_ls[m] = v
                    permz_ls[m] = v
                elif _idx == IDX_CAPVENT:
                    v = fix_perm(si2mdarcy(1.0e-17 * cap_scale))
                    permx_ls[m] = v
                    permy_ls[m] = v
                    permz_ls[m] = v
                elif _idx == IDX_LAND:
                    kh = fix_perm(calc_k_z((z0 + z1) * 0.5))  #!
                    kv = kh  #!
                    permx_ls[m] = kh
                    permy_ls[m] = kh
                    permz_ls[m] = kv
                z0 = z1

    rocknum_ptgrad = {}
    for _idx in topo_unique:
        _prop = rocknum_ptgrad.setdefault(topo_rocknum_map[_idx], {})
        if _idx == IDX_AIR:
            _prop["a_p"] = P_GROUND - top * P_GRAD_AIR
            _prop["b_p"] = P_GRAD_AIR
            _prop["a_t"] = topo_props[IDX_AIR]["TEMPC"]
            _prop["b_t"] = 0.0
        elif _idx == IDX_LAKE:
            elv_lake: float = None
            for m, _idx_tmp in enumerate(topo_ls):
                if _idx_tmp == IDX_LAKE:
                    _, _, k = calc_ijk(m, nx, ny)
                    elv_lake = top - sum(gz[: k + 1])
                    break
            _prop["a_p"] = P_GROUND - elv_lake * P_GRAD_AIR
            _prop["b_p"] = P_GRAD_LAKE
            _prop["a_t"] = topo_props[IDX_LAKE]["TEMPC"]
            _prop["b_t"] = 0.0
        elif _idx == IDX_SEA:
            _prop["a_p"] = P_GROUND
            _prop["b_p"] = P_GRAD_SEA
            _prop["a_t"] = topo_props[IDX_SEA]["TEMPC"]
            _prop["b_t"] = 0.0
        else:
            _prop["a_p"] = P_GROUND
            _prop["b_p"] = P_GRAD_ROCK
            _prop["a_t"] = topo_props[IDX_LAND]["TEMPC"]
            _prop["b_t"] = T_GRAD_ROCK

    # T
    tempe_ls, pres_ls, xco2_ls = deepcopy(zeros), deepcopy(zeros), deepcopy(zeros)
    for j in range(ny):
        for i in range(nx):
            z = 0.0  # depth from top of the land
            for k in range(nz):
                m = calc_m(i, j, k, nx, ny)
                _idx = topo_ls[m]
                # depth from earth surface (m)
                if _idx not in (IDX_LAND, IDX_VENT, IDX_CAP):
                    tempe_ls[m] = topo_props[_idx]["TEMPC"]
                    continue
                z += gz[k] * 0.5
                tempe_ls[m] = TEMPE_AIR + T_GRAD_ROCK * z
                z += gz[k] * 0.5
    # P
    for j in range(ny):
        for i in range(nx):
            p_top = calc_press_air(top)
            for k in range(nz):
                m = calc_m(i, j, k, nx, ny)
                _idx = topo_ls[m]
                rho = None
                if _idx in (IDX_LAND, IDX_VENT, IDX_CAP):
                    rho = 1000.0
                else:
                    rho = TOPO_CONST_PROPS[_idx]["DENS"]
                dp = rho * G * (gz[k] * 0.5) * 1.0e-6
                p_top += dp
                pres_ls[m] = p_top
                p_top += dp

    # XCO2
    for j in range(ny):
        for i in range(nx):
            z = 0.0  # depth from top of the land
            for k in range(nz):
                m = calc_m(i, j, k, nx, ny)
                _idx = topo_ls[m]
                xco2_ls[m] = TOPO_CONST_PROPS[_idx]["COMP1T"]

    # lateral boundary
    lateral_props: Dict = {}
    fluxnum = 200
    for j in range(ny):
        for i in range(nx):
            if i not in (0, nx) and j not in (0, ny):
                continue
            # depth from earth surface (m)
            z = 0.0
            elvsurf = top
            for k in range(nz):
                m = calc_m(i, j, k, nx, ny)
                _idx = topo_ls[m]
                if _idx not in (IDX_LAND, IDX_VENT, IDX_CAP):
                    elvsurf -= gz[k]
                    continue
                z += gz[k] * 0.5
                prop: Dict = lateral_props.setdefault((i + 1, j + 1, k + 1), {})
                prop["FLUXNUM"] = fluxnum
                prop["TEMPC"] = TOPO_CONST_PROPS[IDX_AIR]["TEMPC"] + T_GRAD_ROCK * z
                prop["PRES"] = calc_press_air(elvsurf) + P_GRAD_ROCK * z
                prop["COMP1T"] = TOPO_CONST_PROPS[IDX_LAND]["COMP1T"]
                z += gz[k] * 0.5
                fluxnum += 1

    # bottom boundary
    bottom_props: Dict = {}
    for j in range(ny):
        for i in range(nx):
            # depth of air block
            dz = 0.0
            elv = top
            for k in range(nz):
                m = calc_m(i, j, k, nx, ny)
                _idx = topo_ls[m]
                if _idx in (IDX_LAND, IDX_VENT, IDX_CAP):
                    dz += gz[k]
                elif _idx == IDX_AIR:
                    elv -= gz[k]
            prop: Dict = bottom_props.setdefault((i + 1, j + 1, len(gz)), {})
            prop["FLUXNUM"] = fluxnum
            prop["TEMPC"] = TEMPE_AIR + dz * T_GRAD_ROCK
            prop["PRES"] = calc_press_air(elv) + dz * P_GRAD_ROCK
            prop["COMP1T"] = TOPO_CONST_PROPS[IDX_LAND]["COMP1T"]
            fluxnum += 1

    # RAINSRC
    m_airbounds = {}
    pres_top = calc_press_air(top)
    for m, _idx in enumerate(topo_ls):
        if _idx == IDX_AIR:
            continue
        i, j, k = calc_ijk(m, nx, ny)
        m_above = calc_m(i, j, k - 1, nx, ny)
        if k == 0:
            m_airbounds.setdefault(
                m,
                {
                    "FLUXNUM": fluxnum,
                    "TEMPC": TEMPE_AIR,
                    "PRES": pres_top,
                },
            )
            fluxnum += 1
            continue
        if topo_ls[m_above] == IDX_AIR:
            z = sum(gz[:k])
            m_airbounds.setdefault(
                m,
                {
                    "FLUXNUM": fluxnum,
                    "TEMPC": TEMPE_AIR,
                    "PRES": pres_top
                    + TOPO_CONST_PROPS[IDX_AIR]["DENS"] * G * z * 1.0e-6,
                },
            )
            fluxnum += 1
            continue

    # Lateral surface
    surf_lateral: Dict = {}
    for m in m_airbounds:
        i, j, k = calc_ijk(m, nx, ny)
        not_inc = True
        for itmp in (i - 1, i, i + 1):
            for jtmp in (j - 1, j, j + 1):
                if itmp in (-1, nx):
                    continue
                if jtmp in (-1, ny):
                    continue
                if i == itmp and j == jtmp:
                    continue
                mtmp = calc_m(itmp, jtmp, k, nx, ny)
                xco2: float = None
                if topo_ls[mtmp] == IDX_AIR:
                    b: str = None
                    if itmp == i - 1:
                        b = "I-"
                    elif itmp == i + 1:
                        b = "I+"
                    elif jtmp == j - 1:
                        b = "J-"
                    elif jtmp == j + 1:
                        b = "J+"
                    props: Dict = surf_lateral.setdefault(m, {})
                    props.setdefault("Direction", []).append(b)
                    # calc XCO2
                    if xco2 is None:
                        xco2 = calc_xco2_rain(
                            pres_ls[mtmp] * 1.0e6, 3.8e-4
                        )  # TODO: XCO2 should be loaded from constants.py
                    props.setdefault(
                        "prop",
                        {
                            "PRES": pres_ls[mtmp],
                            "TEMPC": tempe_ls[mtmp],
                            "COMP1T": xco2,
                        },
                    )
                    props.setdefault("FLUXNUM", fluxnum)
                    if not_inc:
                        fluxnum += 1
                        not_inc = False

    # TOP boudnary
    top_props: Dict = {}
    pres_top = calc_press_air(top)
    for i in range(nx):
        for j in range(ny):
            k = 0
            m = calc_m(i, j, k, nx, ny)
            if m not in m_airbounds:
                continue
            prop = {
                "FLUXNUM": fluxnum,
                "PRES": pres_top,
                "TEMPC": TEMPE_AIR,
                "COMP1T": calc_xco2_rain(pres_top * 1.0e6, 3.8e-4),
            }  # TODO: load from constant
            top_props.setdefault(m, prop)
            fluxnum += 1

    # #! fix permeability #!
    # for m in m_airbounds:
    #     i, j, k = calc_ijk(m, nx, ny)
    #     dz = DXYZ[2][k]
    #     if dz <= 10.0:
    #         px, py, pz = 1.0e-4, 1.0e-4
    #     else:
    #         porigh = mdarcy2si(fix_perm(calc_kh(10.0, dz)))
    #         porigv = mdarcy2si(fix_perm(calc_kv(10.0, dz)))
    #         px = (10.0 * 1.0e-4 + (dz - 10.0) * porigh) / dz
    #         py = px
    #         pz = dz / (10.0 / 1.0e-4 + (dz - 10.0) / porigv)
    #     permx_ls[m] = si2mdarcy(px)
    #     permy_ls[m] = si2mdarcy(py)
    #     permz_ls[m] = si2mdarcy(pz)

    return (
        rocknum_ls,
        (permx_ls, permy_ls, permz_ls),
        rocknum_params,
        rocknum_ptgrad,
        tempe_ls,
        pres_ls,
        xco2_ls,
        lateral_props,
        bottom_props,
        top_props,
        m_airbounds,
        surf_lateral,
    )


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
    # NOTE: Includes air over oceans, lakes
    nx, ny, _ = nxyz
    m_bounds: List = []
    for m, _idx in enumerate(topo_ls):
        if _idx == IDX_AIR:
            continue
        i, j, k = calc_ijk(m, nx, ny)
        m_above = calc_m(i, j, k - 1, nx, ny)
        if k == 0:
            m_bounds.append(m)
            continue
        if topo_ls[m_above] == IDX_AIR:
            m_bounds.append(m)
    return m_bounds


def generate_imperm_top_bc(m_airbounds: Dict) -> List[Tuple]:
    transformer = Transformer.from_crs(CRS_WGS84, CRS_RECT, always_xy=True)
    x0, y0 = transformer.transform(ORIGIN[1], ORIGIN[0])
    gx, gy, gz = DXYZ
    xc = np.array(stack_from_center(gx))
    yc = np.array(stack_from_center(gy))
    nx, ny = len(xc), len(yc)
    ijk_ls = []
    for _, pos in POS_SINK.items():
        x, y = transformer.transform(pos[1], pos[0])
        x -= x0
        y -= y0
        i = np.argmin(np.square(xc - x))
        j = np.argmin(np.square(yc - y))
        for k in range(len(gz)):
            if calc_m(i, j, k, nx, ny) in m_airbounds:
                ijk_ls.append((i, j, k))
                break
    return ijk_ls


# def calc_tstep(perm: float, q: float, A: float=0.0006743836062232225, B: float=0.4737982349312893) -> float:
def calc_tstep(perm: float, q: float, A: float = 0.0001, B: float = 0.2) -> float:
    # _max = 300.0 * (exp(-B * (log10(perm) - 1.0))) # before fix pressure gradient
    _max = 25.0 * (exp(-B * (log10(perm) - 1.0)))
    return _max * exp(-A * q)


def write(_f: TextIO, _string: str):
    _string += "\n"
    _f.write(_string)


def generate_input(
    gx: List,
    gy: List,
    gz: List,
    act_ls: List,
    rocknum_ls: List,
    perm_ls: Tuple[List, List, List],
    rocknum_params: Dict,
    lateral_bounds: Dict,
    bottom_bounds: Dict,
    top_bounds: Dict,
    rocknum_pt_grad: Dict,
    tempe_ls: List,
    pres_ls: List,
    xco2_ls: List,
    m_airbounds: Dict,
    surf_lateral: Dict,
    imperm_bounds: List[Tuple],
    sattab: Tuple,
    src_props: Dict,
    sink_props: Dict,
    params: PARAMS,
    tuning_params: Dict,
    fpth: PathLike,
):
    nx, ny = len(DXYZ[0]), len(DXYZ[1])
    with open(fpth, "w", encoding="utf-8") as _f:
        __write: Callable = partial(write, _f)

        # LISENCE
        __write("LICENSE")
        licpth = Path(getcwd(), "LICENSE.LIC")
        __write(f"  '{licpth}' /")
        __write("")  # \n

        # RUNSPEC
        __write(
            "RUNSPEC   ################### RUNSPEC section begins here ######################"
        )
        __write("")  # \n

        # Enable FAST option
        __write("FAST")
        __write("")

        # HCROCK
        __write("HCROCK                                  We enable heat conduction.")
        __write("")  # \n

        # #  DUALPORO
        # __write("DUALPORO")
        # __write("")

        # # GRAVDR
        # __write("GRAVDR                                  We enable heat gravity calculation.")
        # __write("")  # \n

        # GRAVIMET
        # __write("GRAVIMET")
        # __write("   ")
        # __write("")  # \n

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

        # # OBSSPECM #! 各軸4個間隔で置く
        # __write("OBSSPECM")
        # for m in m_airbounds:
        #     i, j, k = calc_ijk(m, nx, ny)
        #     if i % 4 == 0 and j % 4 == 0:
        #         # sum([]) .eq. 0
        #         xmap = sum(gx[:i])
        #         ymap = sum(gy[:j])
        #         depth = sum(gz[:k])
        #         __write(f"{m} {xmap} {ymap} {depth} /")
        # __write("/")
        # __write("")

        # ACTNUM
        __write("ACTNUM")
        _str: str = ""
        for _a in act_ls:
            _str += str(_a) + "  "
        _str += "  /"
        __write(_str)
        __write("")  # \n

        # BOUNDARY (fluxnum1 imin1 imax1 jmin1 jmax1 kmin1 kmax1 d1_1 d2_1 d3_1 d4_1 d5_1 d6_1
        # type_1 mode_1 nu1_1 nu2_1 nu3_1 typenum1 actnum1)
        # ※ typenum:  Cell type ID. Available values for grid blocks: 1 (default)- an ordinary block, 2 - an impermeable grid block in which only heat conduction equation is solved)
        # ※ actnum:  Activity flag. 0: cell is inactive (impermeable), 1 (default): cell is active, 2: cell is active but its parameters are fixed
        __write("BOUNDARY                               We define the boundaries:")
        srci, srcj, srck = src_props["i"], src_props["j"], src_props["k"]
        # __write(f"   2   6* 'I-'  'I+'  'J-'  'J+'  2* INFTHIN /    <- Aquifer")
        # __write(
        #     f"   3   6* 'K-'  5*                   INFTHIN   4* 1* 2 /    <- Top boundary"
        # )
        # __write(
        #     f"   102   6* 'K+'  5*                   INFTHIN   4* 2  2 /    <- Bottom boundary"
        # )
        # NOTE: Implicitly assumes the source is at the bottom.
        __write(
            f"   100   {srci} {srci} {srcj} {srcj} {srck} {srck} 'K+'  5*                   INFTHIN   4* 2  2 /    <- MAGMASRC"
        )

        # # Top boundary
        # for m, prop in top_bounds.items():
        #     i, j, k = calc_ijk(m, nx, ny)
        #     typenum: int = None
        #     actnum: int = None
        #     if (i, j, k) in imperm_bounds:
        #         typenum = 1
        #         actnum = 1
        #     else:
        #         typenum = 2
        #         actnum = 2 # TODO: actnum=1にすると計算が回らなくなったのでとりあえずこうしている
        #     i += 1
        #     j += 1
        #     k += 1
        #     fluxnum = prop["FLUXNUM"]
        #     __write(
        #     f"   {fluxnum}   {i} {i} {j} {j} {k} {k} 'K-'  5*                   INFTHIN   4* {typenum}  {actnum} /    <- Top boundary"
        # )

        # lateral boundary block locations
        for (i, j, k), prop in lateral_bounds.items():
            Ip, Im, Jp, Jm = None, None, None, None
            if i == 1:
                Ip = "'" + "I-" + "'"
            if i == nx:
                Im = "'" + "I+" + "'"
            if j == 1:
                Jp = "'" + "J-" + "'"
            if j == ny:
                Jm = "'" + "J+" + "'"
            dsum = ""
            cou_none = 2
            for _d in (Ip, Im, Jp, Jm):
                if _d is None:
                    cou_none += 1
                else:
                    dsum += _d + " "
            fluxnum = prop["FLUXNUM"]
            __write(
                f"   {fluxnum}   {i} {i} {j} {j} {k} {k} {dsum} {cou_none}*                   INFTHIN   4* 1  2 /    <- Lateral boundary"
            )

        # # Surafce boundary
        # for m, prop in m_airbounds.items():
        #     i, j, k = calc_ijk(m, nx, ny)
        #     i += 1
        #     j += 1
        #     k += 1
        #     fluxnum = prop["FLUXNUM"]
        #     __write(
        #     f"   {fluxnum}   {i} {i} {j} {j} {k} {k} 'K-'  5*                   INFTHIN   4* 1  2 /    <- Surface boundary (top)"
        # )

        # # Surface surrounded by air
        # for m, props in surf_lateral.items():
        #     i, j, k = calc_ijk(m, nx, ny)
        #     i += 1
        #     j += 1
        #     k += 1
        #     fluxnum = props["FLUXNUM"]
        #     directions = props["Direction"]
        #     Ip, Im, Jp, Jm = None, None, None, None
        #     if "I-" in directions:
        #         Ip = "'" + "I-" + "'"
        #     if "I+" in directions:
        #         Im = "'" + "I+" + "'"
        #     if "J-" in directions:
        #         Jp = "'" + "J-" + "'"
        #     if "J+" in directions:
        #         Jm = "'" + "J+" + "'"
        #     dsum = ""
        #     cou_none = 2
        #     for _d in (Ip, Im, Jp, Jm):
        #         if _d is None:
        #             cou_none += 1
        #         else:
        #             dsum += _d + " "
        #     __write(
        #         f"   {fluxnum}   {i} {i} {j} {j} {k} {k} {dsum} {cou_none}*                   INFTHIN   4* 2  2 /    <- Surface boundary (lateral)"
        #     )

        # bottom boundary
        for (i, j, k), prop in bottom_bounds.items():
            if i == srci and j == srcj and k == srck:
                continue
            fluxnum = prop["FLUXNUM"]
            __write(
                f"   {fluxnum}   {i} {i} {j} {j} {k} {k}  'K+'  5*                   INFTHIN   4* 2  2 /   <- Bottom boundary"
            )

        __write("/")
        __write("")

        # SRCSPECG (MAGMASRC and RAINSRC)
        __write("SRCSPECG")
        # MAGMASRC
        __write(f" ’MAGMASRC’ {srci} {srcj} {srck} /")
        # # FUMAROLE
        # for name, (sinki, sinkj, sinkk) in sink_props.items():
        #     __write(f" ’{name}’ {sinki + 1} {sinkj + 1} {sinkk + 1} /")
        # RAINSRC
        for idx, m in enumerate(m_airbounds):
            i, j, k = calc_ijk(m, nx, ny)
            __write(f" ’{idx}’ {i+1} {j+1} {k+1} /")
        __write("/")
        __write("")

        # ENDMAKE
        __write(
            "ENDMAKE   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        )
        __write("")  # \n

        # PERMX
        __write("PERMX")
        _str: str = ""
        for _p in perm_ls[0]:
            _str += str(_p) + "  "
        _str += "  /"
        __write(_str)
        __write("")  # \n

        # PERMY
        __write("PERMY")
        _str: str = ""
        for _p in perm_ls[1]:
            _str += str(_p) + "  "
        _str += "  /"
        __write(_str)
        __write("")  # \n

        # PERMZ
        __write("PERMZ")
        _str: str = ""
        for _p in perm_ls[2]:
            _str += str(_p) + "  "
        _str += "  /"
        __write(_str)
        __write("")  # \n

        # EQUALREG
        __write(
            "EQUALREG                              We specify uniform distributions:"
        )
        for rocknum, props in rocknum_params.items():
            for k, v in props.items():
                # DENS and HC should be written in ROCKDH section
                if k in ("DENS", "HC", "TEMPC", "COMP1T", "PERMX", "PERMY", "PERMZ"):
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
        for idx, prop in rocknum_params.items():
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
        eospth = Path(getcwd()).joinpath("CO2H2O_V3.0.EOS")
        __write(f"   '{eospth}' /")
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
        if sattab is None:
            __write("    0.2    0.000000   1.000000  /  The Brooks and Corey curves")
            __write("    0.3    0.000316   0.737758 /")
            __write("    0.4    0.005056   0.499535 /")
            __write("    0.5    0.025600   0.302400 /")
            __write("    0.6    0.080908   0.155832 /")
            __write("    0.7    0.197530   0.061728 /")
            __write("    0.8    0.409600   0.014400 /")
            __write("    0.9    0.758834   0.000572 /")
            __write("    1.0    1.000000   0.000000  /")
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
        NOTE: Used values are quated from Senario 10 in CourceC
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
        __write("REGALL")
        __write("")  # \n

        # ECHO
        __write("ECHO      Enables more information output in the LOG-file")
        __write("")  # \n

        # EQUALREG
        __write(
            "EQUALREG                              We specify uniform distributions:"
        )
        for rocknum, props in rocknum_params.items():
            for k, v in props.items():
                # DENS and HC should be written in ROCKDH section
                if k in (
                    "DENS",
                    "HC",
                    "HCONDCFX",
                    "HCONDCFY",
                    "HCONDCFZ",
                    "PORO",
                    "PERMX",
                    "PERMY",
                    "PERMZ",
                ):
                    continue
                __write(f"{k} {v}" + " " + f"ROCKNUM {rocknum}  /")
        # __write(
        #     f"   PRES   {P_BOTTOM} FLUXNUM 102 /     The pressure of the bottom boundary"
        # )
        __write(
            f"TEMPC   {params.SRC_TEMP} FLUXNUM 100 /     The temperature of the MAGMASRC"
        )

        # Top boundary conditions
        # for _, prop in top_bounds.items():
        #     FLUXNUM = prop["FLUXNUM"]
        #     T = prop["TEMPC"]
        #     P = prop["PRES"]
        #     COMP1T = prop["COMP1T"]
        #     __write(f"TEMPC   {T} FLUXNUM {FLUXNUM} /    <- Top boundary")
        #     __write(f"PRES   {P} FLUXNUM {FLUXNUM} /")
        #     __write(f"COMP1T   {COMP1T} FLUXNUM {FLUXNUM} /")

        # lateral boundary conditions
        for _, prop in lateral_bounds.items():
            FLUXNUM = prop["FLUXNUM"]
            T = prop["TEMPC"]
            P = prop["PRES"]
            COMP1T = prop["COMP1T"]
            __write(f"TEMPC   {T} FLUXNUM {FLUXNUM} /    <- Lateral boundary")
            __write(f"PRES   {P} FLUXNUM {FLUXNUM} /")
            __write(f"COMP1T   {COMP1T} FLUXNUM {FLUXNUM} /")

        # bottom boundary
        for _, prop in bottom_bounds.items():
            FLUXNUM = prop["FLUXNUM"]
            T = prop["TEMPC"]
            P = prop["PRES"]
            COMP1T = prop["COMP1T"]
            __write(f"TEMPC   {T} FLUXNUM {FLUXNUM} /    <- Bottom boundary")
            __write(f"PRES   {P} FLUXNUM {FLUXNUM} /")
            __write(f"COMP1T   {COMP1T} FLUXNUM {FLUXNUM} /")

        # # AIR-Land boundary (top)
        # for m, prop in m_airbounds.items():
        #     FLUXNUM = prop["FLUXNUM"]
        #     T = prop["TEMPC"]
        #     P = prop["PRES"] #!
        #     # P = 0.1 #!
        #     COMP1T = calc_xco2_rain(P * 1.0e6, params.XCO2_AIR)
        #     __write(f"TEMPC   {T} FLUXNUM {FLUXNUM} /    <- Surface boundary (top)")
        #     __write(f"PRES   {P} FLUXNUM {FLUXNUM} /")
        #     __write(f"COMP1T   {COMP1T} FLUXNUM {FLUXNUM} /")

        # # AIR-Land boundary (lateral)
        # for m, props in surf_lateral.items():
        #     FLUXNUM = props["FLUXNUM"]
        #     prop = props["prop"]
        #     T = prop["TEMPC"]
        #     P = prop["PRES"] #!
        #     # P = 0.1 #!
        #     COMP1T = prop["COMP1T"]
        #     __write(f"TEMPC   {T} FLUXNUM {FLUXNUM} /    <- Surface boundary (on lateral)")
        #     __write(f"PRES   {P} FLUXNUM {FLUXNUM} /")
        #     __write(f"COMP1T   {COMP1T} FLUXNUM {FLUXNUM} /")

        __write("/")
        __write("")  # \n

        # OPERAREG (set initial value)
        __write("OPERAREG")
        for rocknum, prop in rocknum_pt_grad.items():
            ap, bp = prop["a_p"], prop["b_p"]
            __write(f"   PRES DEPTH  ROCKNUM  {rocknum} MULTA  {ap}  {bp}  /")
            at, bt = prop["a_t"], prop["b_t"]
            __write(f"   TEMPC DEPTH  ROCKNUM  {rocknum} MULTA  {at}  {bt}  /")
        __write("/")
        __write("")  # \n

        # TEMPC
        if tempe_ls is not None:
            __write("TEMPC")
            _str: str = ""
            for _v in tempe_ls:
                _str += str(_v) + "  "
            _str += "  /"
            __write(_str)
            __write("")  # \n

        # PRES
        if pres_ls is not None:
            __write("PRES")
            _str: str = ""
            for _v in pres_ls:
                _str += str(_v) + "  "
            _str += "  /"
            __write(_str)
            __write("")  # \n

        # COMP1T
        if xco2_ls is not None:
            __write("COMP1T")
            _str: str = ""
            for _v in xco2_ls:
                _str += str(_v) + "  "
            _str += "  /"
            __write(_str)
            __write("")  # \n

        # OPERAREG (set initial value)
        # __write("OPERAREG")
        # for rocknum, prop in rocknum_pt_grad.items():
        #     ap, bp = prop["a_p"], prop["b_p"]
        #     __write(f"   PRES DEPTH  ROCKNUM  {rocknum} MULTA  {ap}  {bp}  /")
        #     at, bt = prop["a_t"], prop["b_t"]
        #     __write(f"   TEMPC DEPTH  ROCKNUM  {rocknum} MULTA  {at}  {bt}  /")
        # __write("/")
        # __write("")  # \n

        # EQUALNAM
        __write("EQUALNAM")
        # MAGMASRC
        pres, tempe, comp1t = src_props["pres"], src_props["tempe"], src_props["comp1t"]
        __write(f"  PRES {pres} ’MAGMASRC’ /")
        __write(f"  TEMPC {tempe} ’MAGMASRC’/")
        __write(f"  COMP1T {comp1t} ’MAGMASRC’/")

        # RAINSRC
        zc_ls = stack_from_0(gz)
        for idx, (m, prop) in enumerate(m_airbounds.items()):
            # calculate CO2 fraction of rain source
            i, j, k = calc_ijk(m, nx, ny)
            ptol = prop["PRES"] * 1.0e6
            xco2_rain = calc_xco2_rain(ptol, params.XCO2_AIR)
            __write(f"  PRES {ptol * 1.0e-6} ’{idx}’ /")
            # __write(f"  PRES {1.0} ’{idx}’ /") # upper limit
            __write(f"  TEMPC {params.TEMP_RAIN} ’{idx}’ /")
            __write(f"  COMP1T {xco2_rain} ’{idx}’ /")
        __write("/")
        __write("")  # \n
        __write("")  # \n

        # RPTSUM
        __write("RPTSUM")
        __write(
            "   PRES TEMPC PHST SAT#LIQ SAT#GAS FLUXK#E COMP1T COMP2T DENT/  We specify the properties saved at every report time."
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

        # Enalble WEEKTOL option
        __write("WEEKTOL")
        __write("")

        # NEWTON #!
        __write("NEWTON")
        # __write("1   2   2 /")  # 1, 3, 3 ?
        __write("1   5   5 /")
        __write("")

        # SRCINJE
        __write("SRCINJE")
        __write(f"  ’MAGMASRC’ MASS 1* 50. 1* {params.INJ_RATE} /")
        # FUMAROLE
        # for name, sink_rate in SINK_PARAMS.items():
        #     __write(f"  ’{name}’ MASS 1* 50. 1* -{sink_rate} /")
        # RAIN
        for idx, m in enumerate(m_airbounds):
            i, j, k = calc_ijk(m, nx, ny)
            mass_rain = 1.0e-3 * params.RAIN_AMOUNT * gx[i] * gy[j]  # t/day
            __write(f"  ’{idx}’ MASS 1* 10. 1* {mass_rain} /")
        __write("/")
        __write("")  # \n

        # TUNING
        years_total = 0.0
        ts = TSTEP_INIT
        if tuning_params is not None:
            ts = tuning_params[0]
        ts_max = calc_tstep(params.VENT_SCALE, params.INJ_RATE)
        #!
        if params.VENT_SCALE == 1000.0:
            ts_max = 2.0
        if params.VENT_SCALE == 10000.0:
            ts_max = 1.0
        if params.INJ_RATE > 10000.0:
            ts_max = 1.0
        if tuning_params is not None:
            ts_max = tuning_params[1]
        time_rpt = 0.0
        tend: float = TIME_SS
        if TEND_UNREST is not None:
            tend = TEND_UNREST
        while years_total < tend:
            if ts > ts_max:
                ts = ts_max
            if TRPT_UNREST is not None and ts == ts_max:
                tstep_rpt = TRPT_UNREST
            else:
                tstep_rpt = ts * (
                    NDTFIRST + years_total / tend * (NDTEND - NDTFIRST)
                )
            if tend - years_total < tstep_rpt / 365.25:
                tstep_rpt = (tend - years_total) * 365.25
            time_rpt += tstep_rpt
            # time_rpt += tstep_rpt * max(log10(params.VENT_SCALE), log10(params.INJ_RATE)) #!
            tsmax = ts
            __write("TUNING")
            __write(f"    1* {tsmax}   1* {TSTEP_MIN} /")
            __write("TIME")
            __write(f"    {time_rpt} /")
            __write("")
            years_total += tstep_rpt / 365.25
            tmult: float = TMULT
            if 1.0 <= time_rpt <= 180.0 and params.INJ_RATE > 10000.0:
                ts = 50.0 / (24 * 60 * 60)
                tmult = 1.0
            ts *= tmult

        # REPORTS
        __write("REPORTS")
        __write("   CONV MATBAL LINSOL  /")
        __write("")  # \n

        # VARS
        __write("VARS")
        __write("PRES     DMAX    1.0   / Maximum pressure change is set to be 1.0 MPa")
        __write("/")
        __write("")

        # ILUTFILL
        __write("ILUTFILL")
        __write("  4  /")
        __write("")
        __write(" ILUTDROP")
        __write("  1e-4  /")
        __write("")

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


def modify_file(refpth, tpth, tempe_ls, pres_ls, xco2_ls) -> None:

    nx, ny, nz = len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2])
    nxyz = nx * ny * nz
    assert len(tempe_ls) == nxyz, len(tempe_ls)
    assert len(pres_ls) == nxyz, len(pres_ls)
    assert len(xco2_ls) == nxyz, len(xco2_ls)

    with open(refpth, "r") as f:
        lines = f.readlines()

    ln_insert_props = None
    ln_insert_tuning = None
    for i, l in enumerate(lines):
        if "OPERAREG" in l:
            for j in range(i + 1, i + 1000):
                if "/\n" in lines[j] and lines[j + 1] == "/\n":
                    ln_insert_props = j + 2
                    break
        if "REPORTS" in l:
            ln_insert_tuning = i - 1
            break

    # find delete lines
    delete_index = set()
    for i, l in enumerate(lines):
        if "TUNING" in l:
            delete_index.update([i, i + 1, i + 2, i + 3])
            continue
        if "TEMPC\n" in l:
            delete_index.update([i, i + 1])
        if "PRES\n" in l:
            delete_index.update([i, i + 1])
        if "COMP1T\n" in l:
            delete_index.update([i, i + 1])
    # lines = [lines[i] for i in range(len(lines)) if i not in delete_index]
    with open(tpth, "w") as f:
        for ln, line in enumerate(lines):
            if ln in delete_index:
                continue
            if ln == ln_insert_props:
                # TEMPC
                f.write("\n")
                f.write("TEMPC\n")
                _str: str = ""
                for _v in tempe_ls:
                    _str += str(_v) + "  "
                _str += "  /\n"
                f.write(_str)
                f.write("\n")  # \n

                # PRES
                f.write("PRES\n")
                _str: str = ""
                for _v in pres_ls:
                    _str += str(_v) + "  "
                _str += "  /\n"
                f.write(_str)
                f.write("\n")  # \n

                # COMP1T
                f.write("COMP1T\n")
                _str: str = ""
                for _v in xco2_ls:
                    _str += str(_v) + "  "
                _str += "  /\n"
                f.write(_str)
                f.write("\n")  # \n
            
            if ln == ln_insert_tuning:
                time = 0.0
                while time < TEND_UNREST:
                    time += TRPT_UNREST
                    f.write(f"TUNING\n")
                    f.write(f"    1* {TSTEP_UNREST}   1* {TSTEP_MIN} /\n")
                    f.write(f"TIME\n")
                    f.write(f"    {time} /\n")
                    f.write(f"\n")

            f.write(line)


def generate_from_params(
    params: PARAMS,
    pth: PathLike,
    continue_from_latest: bool = False,
    refpth: PathLike = None,
    tuning_params: Tuple = None,
) -> None:
    """Generate RUN file from PARAMS class

    Args:
        params (PARAMS): Instance containing necessary parameters
        pth (PathLike): Path of RUN file.
        load_from_latest (bool): Whether load from latest .SUM file or not
            (NOTE: file path is assumed to be a subdirectory of the previous working directory)
    """
    nxyz = (len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2]))
    # modify tempe_ls, pres_ls, xco2_ls
    if continue_from_latest:
        pth = Path(pth)
        # file path is assumed to be a subdirectory of the previous working directory
        dirpth = pth.parent.parent
        for i in range(1000):
            iter_dir = dirpth.joinpath(f"ITER_{i}")
            if iter_dir.exists():
                dirpth = iter_dir
            else:
                break
        started: bool = False
        for i in range(100000):
            fn = str(i).zfill(4)
            fpth = dirpth.joinpath(f"tmp.{fn}.SUM")
            if i == 0 and not fpth.exists():
                started = False
                break
            elif i == 0 and fpth.exists():
                started = True
            elif not fpth.exists():
                fn = fn = str(i - 1).zfill(4)
                fpth = dirpth.joinpath(f"tmp.{fn}.SUM")
                break
        if started:
            print(f"refpth: {fpth}")  #!
            print(f"runpth: {pth}")  #!
            nxyz = nxyz[0] * nxyz[1] * nxyz[2]
            cellid_props, _, time = load_sum(fpth)
            tempe_ls = get_v_ls(cellid_props, "TEMPC")[:nxyz]
            pres_ls = get_v_ls(cellid_props, "PRES")[:nxyz]
            xco2_ls = get_v_ls(cellid_props, "COMP1T")[:nxyz]
            modify_file(
                dirpth.joinpath("tmp.RUN"),
                pth,
                tempe_ls,
                pres_ls,
                xco2_ls,
            )
            return

    cache_topo = CACHE_DIR.joinpath("topo_ls")
    topo_ls: List = None
    lat_2d, lng_2d, isrc, jsrc, ksrc = None, None, None, None, None

    if cache_topo.exists():
        with open(cache_topo, "rb") as pkf:
            topo_ls, (xc_m, yc_m, zc_m, lat_2d, lng_2d, srcpos, sinkpos) = pickle.load(
                pkf
            )
    else:
        topo_ls, (xc_m, yc_m, zc_m, lat_2d, lng_2d, srcpos, sinkpos,) = generate_topo(
            DEM_PTH,
            SEADEM_PTH,
            CRS_RECT,
            DXYZ,
            ORIGIN,
            POS_SRC,
            POS_SINK,
            align_center=ALIGN_CENTER,
        )
        with open(cache_topo, "wb") as pkf:
            pickle.dump(
                (topo_ls, (xc_m, yc_m, zc_m, lat_2d, lng_2d, srcpos, sinkpos)),
                pkf,
                pickle.HIGHEST_PROTOCOL,
            )

    # add vent structure
    with open("./analyze_magnetic_coords/elv_bounds_mufits.pkl", "rb") as pkf:
        elv_bounds_mufits: Dict = pickle.load(pkf)
    topo_ls = generate_simple_vent(topo_ls, xc_m, yc_m, zc_m, elv_bounds_mufits)
    if params.CAP_SCALE is not None:
        with open("./analyse_crator_coords/crator.pkl", "rb") as pkf:
            crator_coords: Polygon = pickle.load(pkf)
        generate_simple_cap(topo_ls, xc_m, yc_m, 700.0, crator_coords)
    # debug
    # plt_topo(topo_ls, lat_2d, lng_2d, nxyz, "debug")
    actnum_ls = generate_act_ls(topo_ls)
    (
        rocknum_ls,
        perm_ls,
        rocknum_params,
        rocknum_ptgrad,
        _,
        _,
        _,
        lateral_bounds,
        bottom_bounds,
        top_props,
        m_airbounds,
        surf_lateral,
    ) = generamte_rocknum_and_props(
        topo_ls,
        ORIGIN[2],
        nxyz,
        DXYZ[2],
        params.TOPO_PROPS,
        params.VENT_SCALE,
        params.CAP_SCALE,
        params.VK,
    )

    imperm_bounds: List[Tuple] = generate_imperm_top_bc(m_airbounds)

    if refpth is not None:
        nxyz = nxyz[0] * nxyz[1] * nxyz[2]
        cellid_props, _, time = load_sum(refpth)
        tempe_ls = get_v_ls(cellid_props, "TEMPC")[:nxyz]
        pres_ls = get_v_ls(cellid_props, "PRES")[:nxyz]
        xco2_ls = get_v_ls(cellid_props, "COMP1T")[:nxyz]
    else:
        tempe_ls, pres_ls, xco2_ls = None, None, None

    # debug
    # plt_topo(perm_ls, lat_2d, lng_2d, nxyz, "./debug/perm")
    # plt_airbounds(topo_ls, m_airbounds, lat_2d, lng_2d, nxyz, "./debug/airbounds")
    # relative permeability
    # sattab = calc_sattab(method="None")

    # permx_ls = [mdarcy2si(i) for i in perm_ls[0]] #!
    # permx_with_nan = np.where(np.array(permx_ls) <= 0, np.nan, np.array(permx_ls)) #!
    # permz_ls = [mdarcy2si(i) for i in perm_ls[2]]
    # permz_with_nan = np.where(np.array(permz_ls) <= 0, np.nan, np.array(permz_ls))
    # print(max(permx_ls), max(permz_ls))
    # plt_any_val(np.log10(permx_with_nan), stack_from_center(DXYZ[0]), [ORIGIN[2] - i for i in stack_from_0(DXYZ[2])], nxyz, "debug/permx3", r'Log $m^2$', _min=-14, _max=-9) #!
    # plt_any_val(np.log10(permz_with_nan), stack_from_center(DXYZ[0]), [ORIGIN[2] - i for i in stack_from_0(DXYZ[2])], nxyz, "debug/permz2", r'Log $m^2$',_min=-14, _max=-9)
    # for px, pv in zip(permx_ls, permz_ls):
    #     print(px, pv)

    src_props = {
        "i": srcpos[0] + 1,
        "j": srcpos[1] + 1,
        "k": srcpos[2] + 1,
        "pres": params.PRES_SRC,
        "tempe": params.SRC_TEMP,
        "comp1t": params.SRC_COMP1T,
    }

    # debug
    with open(Path(pth).parent.joinpath("debug.pkl"), "wb") as pkf:
        pickle.dump(
            (
                actnum_ls,
                rocknum_ls,
                perm_ls,
                rocknum_params,
                lateral_bounds,
                bottom_bounds,
                rocknum_ptgrad,
                tempe_ls,
                pres_ls,
                xco2_ls,
                m_airbounds,
                surf_lateral,
                imperm_bounds,
            ),
            pkf,
        )

    generate_input(
        DXYZ[0],
        DXYZ[1],
        DXYZ[2],
        actnum_ls,
        rocknum_ls,
        perm_ls,
        rocknum_params,
        lateral_bounds,
        bottom_bounds,
        top_props,
        rocknum_ptgrad,
        tempe_ls,
        pres_ls,
        xco2_ls,
        m_airbounds,
        surf_lateral,
        imperm_bounds,
        None,
        src_props,
        sinkpos,
        params,
        tuning_params,
        pth,
    )


from utils import condition_to_dir

if __name__ == "__main__":
    temp_src = 900.0
    comp1t = 0.1
    perm_vent = 1.0
    inj_rate = 10000.0
    cap_scale = 100000.0
    basedir = r"E:\tarumai_tmp16"
    from_latest = True
    dirpth: Path = condition_to_dir(
        basedir, temp_src, comp1t, inj_rate, perm_vent, cap_scale, from_latest
    )
    makedirs(dirpth, exist_ok=True)
    print(dirpth)
    generate_from_params(
        PARAMS(
            temp_src=temp_src,
            comp1t=comp1t,
            perm_vent=perm_vent,
            inj_rate=inj_rate,
            cap_scale=cap_scale,
        ),
        dirpth.joinpath("tmp.RUN"),
        from_latest,
    )
    # print(calc_tstep(10.0, 10000.0, ))
    pass
