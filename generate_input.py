#!/usr/bin/env python
# coding: utf-8
"""Generate input file for MUFITS (Afanasyev, 2012)
"""

from functools import partial
from typing import List, Tuple, Dict, TextIO, Callable, Set, Any, Optional
from collections import OrderedDict
from pathlib import Path
from os import PathLike, makedirs, getcwd
import pickle
from math import exp, log10
from statistics import mean
from copy import deepcopy

from pyproj import Transformer
from pyproj.enums import TransformDirection
import numpy as np
from tqdm import tqdm
from shapely import Polygon, Point
from LoopStructural import GeologicalModel

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
    load_dem,
)
from default_props import DEFAULT_ROCK_PROPS
from scenario import BaseSenario
from constants import (
    TOP,
    ORIGIN,
    RECT_CRS,
    DXYZ,
    CACHE_DIR,
    TEMPE_AIR,
    G,
    P_GROUND,
    P_GRAD_AIR,
    P_GRAD_ROCK,
    T_GRAD_ROCK,
    TIME_SS,
    TSTEP_INIT,
    TSTEP_MIN,
    TSTEP_MAX,
    NDTFIRST,
    NDTEND,
    TMULT,
    PERM_MAX,
    TRPT_UNREST,
    TEND_UNREST,
    DEM_CRS,
    RECT_CRS,
    XYZPTH,
    EOSPTH,
    RockType,
    SUBSURFACE,
    INNACTIVATE_ROCK_TYPES,
    Condition,
    WellProps,
    XCO2_AIR,
    TEMP_RAIN,
    RAIN_AMOUNT,
    EVAP_AMOUNT,
    COAL_LAYER_NAME,
    COAL_LAYER_DATA_PTH,
    COAL_VERTICAL_SECTIONS,
    LICENSE_PTH,
    D_WELL,
    COAL_HORIZONTAL_DATA_PTH,
    BHP_MAX
)
from interp_geology import load_vertical_data, interp_layer, load_horizontal_data
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
        topo_arr = np.where(filt, RockType.VENT, topo_arr)
    return topo_arr.tolist()



def generate_topo(loopmodel: GeologicalModel) -> Tuple[List[RockType], Tuple[Any]]:  # TODO: output type
    # TODO:
    # 上端がこの地域の標高の高いところになっているか確認

    # generate empty array with mesh
    assert len(DXYZ) == 3, DXYZ
    dx_ls, dy_ls, dz_ls = DXYZ

    # load DEM
    # easting, northing, elevation
    x_dem_ls, y_dem_ls, elv_dem_ls = load_dem(XYZPTH)

    # convert to rect crs
    rect_trans = Transformer.from_crs(DEM_CRS, RECT_CRS, always_xy=True)

    # calculate the coordinates of the grid center
    x_origin, y_origin = rect_trans.transform(ORIGIN[1], ORIGIN[0])
    elv_origin = ORIGIN[2]
    xc_ls = stack_from_center(dx_ls)
    yc_ls = stack_from_center(dy_ls)
    zc_ls = stack_from_0(dz_ls)

    # convert to numpy array
    x_dem_arr, y_dem_arr, elv_dem_arr = (
        np.array(x_dem_ls),
        np.array(y_dem_ls),
        np.array(elv_dem_ls),
    )

    # get center coordinates of grid
    nx, ny, nz = len(dx_ls), len(dy_ls), len(dz_ls)
    nxyz = nz*ny*nx
    # generate topology data
    topo_ls: List[int] = np.zeros(shape=nxyz).tolist()
    lat_2d: List = np.zeros(shape=(ny, nx)).tolist()
    lng_2d: List = np.zeros(shape=(ny, nx)).tolist()
    # center of the coordinates sorted for indecies "m"
    xc_m: List[int] = np.zeros(shape=nxyz).tolist()
    yc_m: List[int] = np.zeros(shape=nxyz).tolist()
    zc_m: List[int] = np.zeros(shape=nxyz).tolist()
    # LAND or AIR
    for i in tqdm(range(nx)):
        for j in range(ny):
            # get grid size δx and δy
            dx, dy = dx_ls[i] * 0.5, dy_ls[j] * 0.5
            xc = x_origin + xc_ls[i]
            yc = y_origin - yc_ls[j]
            lngc, latc = rect_trans.transform(xc, yc, direction=TransformDirection.INVERSE)
            lat_2d[j][i] = latc
            lng_2d[j][i] = lngc

            bounds = (xc - dx, xc + dx, yc - dy, yc + dy)
            _, _, elv_dem_tmp = __clip_xy(x_dem_arr, y_dem_arr, elv_dem_arr, bounds)
            median_land = __median(elv_dem_tmp)

            for k in range(nz):
                # get rectangular coordinates of each grid center
                elvc = elv_origin - zc_ls[k]
                # determine actnum
                rocktype = None
                # refer DEM
                # below DEM: LAND
                if median_land > elvc:
                    rocktype = RockType.LAND
                # above DEM: AIR
                else:
                    rocktype = RockType.AIR
                m = calc_m(i, j, k, nx, ny)
                topo_ls[m] = rocktype
                xc_m[m] = xc_ls[i]
                yc_m[m] = -yc_ls[j]
                zc_m[m] = elvc

        vals = loopmodel.evaluate_feature_value(COAL_LAYER_NAME,
                                         np.array([xc_m, [-1.0*yc for yc in yc_m], zc_m]).T,
                                         )
        for m, rt in enumerate(topo_ls):
            if rt is not RockType.LAND:
                continue
            # TODO: generalize for other geological layer
            if 0.0 <= vals[m] <= 1.0:
                topo_ls[m] = RockType.COAL
            elif vals[m] > 1.0:
                topo_ls[m] = RockType.CAP
    return topo_ls, (xc_m, yc_m, zc_m, lat_2d, lng_2d)


def generate_act_ls(topo_ls: List[int]) -> List[int]:
    actnum_ls = []
    for _idx in topo_ls:
        actnum: int = None
        if _idx in INNACTIVATE_ROCK_TYPES:
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
    topo_ls: List[RockType],
    condition: Condition):
    nx, ny, nz = len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2])
    topo_unique = list(set(topo_ls))
    topo_rocknum_map: Dict[RockType, int] = {rocktype: rocknum for rocknum, rocktype in enumerate(topo_unique)}
    rocknum_ls = []
    rocknum_params = {}
    for rocktype in topo_ls:
        rocknum = topo_rocknum_map[rocktype]
        rocknum_ls.append(rocknum)
        rocknum_params.setdefault(rocknum, condition["ROCK_PROPS"][rocktype])

    # permeability
    zeros: List[float] = np.zeros(len(topo_ls)).tolist()
    permx_ls, permy_ls, permz_ls = deepcopy(zeros), deepcopy(zeros), deepcopy(zeros)
    for j in range(ny):
        for i in range(nx):
            z0, z1 = 0.0, 0.0
            for k in range(nz):
                m = calc_m(i, j, k, nx, ny)
                if z0 == 0.0:
                    z0 = 1.0
                rocktype: RockType = topo_ls[m]
                if rocktype in SUBSURFACE:
                    z1 = z0 + DXYZ[2][k]
                # depth from earth surface (m)
                kx, ky, kz = None, None, None
                if rocktype is RockType.LAND:
                    kx = fix_perm(calc_k_z((z0 + z1) * 0.5))
                    ky = kx
                    kz = kx
                else:
                    kx = si2mdarcy(condition["ROCK_PROPS"][rocktype]["PERMX"])
                    ky = si2mdarcy(condition["ROCK_PROPS"][rocktype]["PERMY"])
                    kz = si2mdarcy(condition["ROCK_PROPS"][rocktype]["PERMZ"])
                permx_ls[m] = kx
                permy_ls[m] = ky
                permz_ls[m] = kz
                z0 = z1

    rocknum_ptgrad = {}
    for rocktype in topo_unique:
        _prop = rocknum_ptgrad.setdefault(topo_rocknum_map[rocktype], {})
        if rocktype is RockType.AIR:
            _prop["a_p"] = P_GROUND - TOP * P_GRAD_AIR
            _prop["b_p"] = P_GRAD_AIR
            _prop["a_t"] = TEMPE_AIR
            _prop["b_t"] = 0.0
        else:
            _prop["a_p"] = P_GROUND
            _prop["b_p"] = P_GRAD_ROCK
            _prop["a_t"] = TEMPE_AIR
            _prop["b_t"] = T_GRAD_ROCK

    # T
    tempe_ls = deepcopy(zeros)
    for j in range(ny):
        for i in range(nx):
            z = 0.0  # depth from top of the land
            for k in range(nz):
                m = calc_m(i, j, k, nx, ny)
                rocktype = topo_ls[m]
                # depth from earth surface (m)
                if rocktype not in SUBSURFACE:
                    tempc = condition["ROCK_PROPS"][rocktype]["TEMPC"]
                    assert isinstance(tempc, float) or tempc < 0.0, tempc
                    tempe_ls[m] = tempc
                    continue
                z += DXYZ[2][k] * 0.5
                tempe_ls[m] = TEMPE_AIR + T_GRAD_ROCK * z
                z += DXYZ[2][k] * 0.5
    # P
    pres_ls = deepcopy(zeros)
    for j in range(ny):
        for i in range(nx):
            p_top = calc_press_air(TOP)
            for k in range(nz):
                m = calc_m(i, j, k, nx, ny)
                rocktype = topo_ls[m]
                rho = None
                if rocktype in SUBSURFACE:
                    rho = 1000.0
                else:
                    rho = condition["ROCK_PROPS"][rocktype]["DENS"]
                dp = rho * G * (DXYZ[2][k] * 0.5) * 1.0e-6
                p_top += dp
                pres_ls[m] = p_top
                p_top += dp

    # XCO2
    xco2_ls = deepcopy(zeros)
    for j in range(ny):
        for i in range(nx):
            z = 0.0  # depth from top of the land
            for k in range(nz):
                m = calc_m(i, j, k, nx, ny)
                rocktype = topo_ls[m]
                xco2_ls[m] = condition["ROCK_PROPS"][rocktype]["COMP1T"]

    # lateral boundary
    lateral_props: Dict = {}
    fluxnum = 200
    for j in range(ny):
        for i in range(nx):
            if i not in (0, nx-1) and j not in (0, ny-1):
                continue
            # depth from earth surface (m)
            z = 0.0
            elvsurf = TOP
            for k in range(nz):
                m = calc_m(i, j, k, nx, ny)
                rocktype = topo_ls[m]
                if rocktype not in SUBSURFACE:
                    elvsurf -= DXYZ[2][k]
                    continue
                z += DXYZ[2][k] * 0.5
                prop: Dict = lateral_props.setdefault((i + 1, j + 1, k + 1), {})
                prop["FLUXNUM"] = fluxnum
                prop["TEMPC"] = condition["ROCK_PROPS"][RockType.AIR]["TEMPC"] + T_GRAD_ROCK * z
                prop["PRES"] = calc_press_air(elvsurf) + P_GRAD_ROCK * z
                prop["COMP1T"] = condition["ROCK_PROPS"][RockType.LAND]["COMP1T"]
                z += DXYZ[2][k] * 0.5
                fluxnum += 1

    # bottom boundary
    bottom_props: Dict = {}
    for j in range(ny):
        for i in range(nx):
            # depth of air block
            dz = 0.0
            elv = TOP
            for k in range(nz):
                m = calc_m(i, j, k, nx, ny)
                rocktype = topo_ls[m]
                if rocktype in SUBSURFACE:
                    dz += DXYZ[2][k]
                elif rocktype is RockType.AIR:
                    elv -= DXYZ[2][k]
            prop: Dict = bottom_props.setdefault((i + 1, j + 1, len(DXYZ[2])), {})
            prop["FLUXNUM"] = fluxnum
            prop["TEMPC"] = TEMPE_AIR + dz * T_GRAD_ROCK
            prop["PRES"] = calc_press_air(elv) + dz * P_GRAD_ROCK
            prop["COMP1T"] = condition["ROCK_PROPS"][RockType.LAND]["COMP1T"]
            fluxnum += 1

    # RAINSRC
    m_airbounds = {}
    pres_top = calc_press_air(TOP)
    for m, rocktype in enumerate(topo_ls):
        if rocktype is RockType.AIR:
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
        if topo_ls[m_above] is RockType.AIR:
            z = sum(DXYZ[2][:k])
            m_airbounds.setdefault(
                m,
                {
                    "FLUXNUM": fluxnum,
                    "TEMPC": TEMPE_AIR,
                    "PRES": pres_top
                    + condition["ROCK_PROPS"][RockType.AIR]["DENS"] * G * z * 1.0e-6,
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
                if topo_ls[mtmp] is RockType.AIR:
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
                            pres_ls[mtmp] * 1.0e6, condition["ROCK_PROPS"][RockType.AIR]["COMP1T"]
                        )
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
    pres_top = calc_press_air(TOP)
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
                "COMP1T": calc_xco2_rain(pres_top * 1.0e6, condition["ROCK_PROPS"][RockType.AIR]["COMP1T"]),
            }
            top_props.setdefault(m, prop)
            fluxnum += 1

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
        if _idx == RockType.AIR:
            continue
        i, j, k = calc_ijk(m, nx, ny)
        m_above = calc_m(i, j, k - 1, nx, ny)
        if k == 0:
            m_bounds.append(m)
            continue
        if topo_ls[m_above] == RockType.AIR:
            m_bounds.append(m)
    return m_bounds

def generate_well_specs(head_latlngs: Tuple[float, float],
                        well_id_ls: List[str],
                        ulim: float,
                        xc_m: List[float],
                        yc_m: List[float],
                        topo_ls: List[RockType],
                        transformer: Optional[Transformer]=None) -> Tuple[Dict, Dict]:
    nx, ny, nz = len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2])
    if transformer is None:
        transformer: Transformer = Transformer.from_crs(DEM_CRS, RECT_CRS, always_xy=True)
    well_specs = dict()
    well_compdat = dict()
    x0, y0 = transformer.transform(ORIGIN[1], ORIGIN[0])
    xc_arr, yc_arr = np.array(xc_m), np.array(yc_m)
    for idx, head_latlng in enumerate(head_latlngs):
        x, y = transformer.transform(head_latlng[1], head_latlng[0])
        mx = np.argmin(np.square(xc_arr-(x-x0)))
        my = np.argmin(np.square(yc_arr-(y-y0)))
        i,_,_=calc_ijk(mx,nx,ny)
        _,j,_=calc_ijk(my,nx,ny)
        # assuming vertical well
        bhp_dep = 0.0
        d_from_subsurf = 0.0
        compdat_ls: List[Tuple] = []
        hit_coal = False
        for k in range(nz-1):
            m = calc_m(i,j,k,nx,ny)
            rt = topo_ls[m]
            if rt in (RockType.LAND, RockType.CAP) and hit_coal:
                break
            bhp_dep += DXYZ[2][k]
            if rt is RockType.COAL:
                hit_coal = True
                d_from_subsurf += DXYZ[2][k]
                if d_from_subsurf >= ulim:
                    compdat_ls.append((i,j,k,k+1,"OPEN"))
                else:
                    compdat_ls.append((i,j,k,k+1,"SHUT"))
            elif rt in SUBSURFACE:
                d_from_subsurf += DXYZ[2][k]
                compdat_ls.append((i,j,k,k+1,"SHUT"))
        if ulim > d_from_subsurf:
            raise RuntimeError(f"ulim: {ulim}, bhp depth: {bhp_dep}")
        if not hit_coal:
            raise RuntimeError
        well_id = well_id_ls[idx]
        well_specs.setdefault(well_id, (i,j,bhp_dep))
        well_compdat.setdefault(well_id, compdat_ls)
    return well_specs, well_compdat

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
    sattab,
    tend: float,
    fpth: PathLike,
    well_specs: Optional[Dict],
    well_compdat: Optional[Dict],
    well_props: Optional[WellProps],
):
    
    ignore_well = well_specs is None and well_compdat is None and well_props is None
    nx, ny = len(DXYZ[0]), len(DXYZ[1])
    with open(fpth, "w", encoding="utf-8") as _f:
        __write: Callable = partial(write, _f)

        # LISENCE
        __write("LICENSE")
        licpth = Path(getcwd(), LICENSE_PTH)
        __write(f"  '{licpth}' /")
        __write("")  # \n

        # RUNSPEC
        __write(
            "RUNSPEC   ################### RUNSPEC section begins here ######################"
        )
        __write("")  # \n
        
        # CFLCTRL #! TODO: TUNING消すべき？
        # __write("CFLCTRL")
        # __write("ON    0.8")
        # __write("")  # \n

        # Enable FAST option
        __write("FAST")
        __write("")

        # HCROCK
        __write("HCROCK                                  We enable heat conduction.")
        __write("")  # \n

        # OPTIONS
        if not ignore_well:
            __write("OPTIONS                This switch influences the visualization of wells in ParaView")
            __write("85* 0 /")
            __write("")

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

        if not ignore_well:
            __write("WELSPECS")
            __write("-- name group i-idx j-idx bhp-depth")
            for name, spec in well_specs.items():
                __write(f" {name} 1*  {spec[0]+1}  {spec[1]+1}  {spec[2]}  /")
            __write("/")
            __write("")

            __write("COMPDAT")
            for name, props in well_compdat.items():
                for prop in props:
                    __write(f" {name} {prop[0]+1} {prop[1]+1} {prop[2]+1} {prop[3]+1} {prop[4]} 1* 1* {D_WELL}  /")
            __write("/")
            __write("")

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
        # magmasrc_idx: Set[Tuple] = set()
        # for i, (_, _dct) in enumerate(src_props.items()):
        #     srci, srcj, srck = _dct["i"], _dct["j"], _dct["k"]
        #     # NOTE: Implicitly assumes the source is at the bottom.
        #     __write(
        #         f"   {100 + i}   {srci} {srci} {srcj} {srcj} {srck} {srck} 'K+'  5*                   INFTHIN   4* 2  2 /    <- MAGMASRC"
        #     )
        #     magmasrc_idx.add((srci, srcj, srck))
        # __write(f"   2   6* 'I-'  'I+'  'J-'  'J+'  2* INFTHIN /    <- Aquifer")
        # __write(
        #     f"   3   6* 'K-'  5*                   INFTHIN   4* 1* 2 /    <- Top boundary"
        # )
        # __write(
        #     f"   102   6* 'K+'  5*                   INFTHIN   4* 2  2 /    <- Bottom boundary"
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


        # bottom boundary
        for (i, j, k), prop in bottom_bounds.items():
            # if (i, j, k) in magmasrc_idx: # TODO: WELL
            #     continue
            fluxnum = prop["FLUXNUM"]
            __write(
                f"   {fluxnum}   {i} {i} {j} {j} {k} {k}  'K+'  5*                   INFTHIN   4* 2  2 /   <- Bottom boundary"
            )

        __write("/")
        __write("")

        # SRCSPECG (MAGMASRC and RAINSRC)
        __write("SRCSPECG")
        # MAGMASRC TODO: WELL
        # for name, prop in src_props.items():
        #     srci, srcj, srck = prop["i"], prop["j"], prop["k"]
        #     __write(f" ’{name}’ {srci} {srcj} {srck} /")
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
        __write(f"   '{EOSPTH}' /")
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
        
        # for i, (_, prop) in enumerate(src_props.items()):
        #     srctempe = prop["tempe"]
        #     __write(
        #         f"TEMPC   {srctempe} FLUXNUM {i + 100} /     The temperature of the MAGMASRC"
        #     )

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
        if not ignore_well:
            __write(f"  PRES {well_props['FLUID']['PRES']} {well_props['WELL_ID']} /")
            __write(f"  TEMPC {well_props['FLUID']["TEMPC"]} {well_props['WELL_ID']} /")
            __write(f"  COMP1T {well_props['FLUID']["COMP1T"]} {well_props['WELL_ID']} /")
        # MAGMASRC
        # for name, prop in src_props.items():
        #     pres, tempe, comp1t = prop["pres"], prop["tempe"], prop["comp1t"]
        #     __write(f"  PRES {pres} ’{name}’ /")
        #     __write(f"  TEMPC {tempe} ’{name}’/")
        #     __write(f"  COMP1T {comp1t} ’{name}’/")

        # RAINSRC
        zc_ls = stack_from_0(gz)
        for idx, (m, prop) in enumerate(m_airbounds.items()):
            # calculate CO2 fraction of rain source
            i, j, k = calc_ijk(m, nx, ny)
            ptol = prop["PRES"] * 1.0e6
            xco2_rain = calc_xco2_rain(ptol, XCO2_AIR)
            __write(f"  PRES {ptol * 1.0e-6} ’{idx}’ /")
            # __write(f"  PRES {1.0} ’{idx}’ /") # upper limit
            __write(f"  TEMPC {TEMP_RAIN} ’{idx}’ /")
            __write(f"  COMP1T {xco2_rain} ’{idx}’ /")
        __write("/")
        __write("")  # \n
        __write("")  # \n

        # RPTSUM
        __write("RPTSUM")
        __write(
            "   PRES TEMPC PHST SAT#LIQ SAT#GAS FLUXK#E COMP1T COMP2T DENT /  We specify the properties saved at every report time."
        )
        __write("")  # \n
        
        if not ignore_well:
            __write("RPTWELL")
            __write("   WBHP WTHP WMIR#1 WMIRALL WMIT#1 WMITALL  /  We report these properties for every well.")
            __write("")

        #  RPTSRC
        __write("RPTSRC")
        __write("  SMIR#1 SMIR#2 SMIT#1 SMIT#2 /")
        __write("")  # \n

        # SCHEDULE
        __write(
            "SCHEDULE   #################### SCHEDULE section begins here ####################"
        )
        __write("")  # \n

        if not ignore_well:
            __write("WELLINJE")
            inj_type = well_props["INJE_UNIT"]
            if inj_type == "MASS":
                __write(f"  {well_props['WELL_ID']}  OPEN MASS 1* {well_props["INJE_RATE"]}  {BHP_MAX}  1E7 2*  100 1 /")
            else:
                __write(f"  {well_props['WELL_ID']}  OPEN RATE {well_props["INJE_RATE"]}  1*  {BHP_MAX}  1E7 2*  100 1 /")
            __write("/")
            __write("")

        # Enalble WEEKTOL option
        __write("WEEKTOL")
        __write("")

        # NEWTON #!
        __write("NEWTON")
        # __write("1   2   2 /")  # 1, 3, 3 ?
        __write("1   5   5 /")
        __write("")

        # SRCINJE TODO: WELL
        __write("SRCINJE")
        # for name, prop in src_props.items():
        #     injrate = prop["inj_rate"]
        #     __write(f"  ’{name}’ MASS 1* 50. 1* {injrate} /") # MAGMASRC
        # FUMAROLE
        # for name, sink_rate in SINK_PARAMS.items():
        #     __write(f"  ’{name}’ MASS 1* 50. 1* -{sink_rate} /")
        # RAIN
        for idx, m in enumerate(m_airbounds):
            i, j, k = calc_ijk(m, nx, ny)
            mass_rain = 1.0e-3 * (RAIN_AMOUNT - EVAP_AMOUNT) * gx[i] * gy[j]  # t/day
            __write(f"  ’{idx}’ MASS 1* 10. 1* {mass_rain} /")
        __write("/")
        __write("")  # \n

        # TUNING
        years_total = 0.0
        ts = TSTEP_INIT
        time_rpt = 0.0
        while years_total < TIME_SS:
            if ts > TSTEP_MAX:
                ts = TSTEP_MAX
            tstep_rpt = ts * (
                NDTFIRST + years_total / TIME_SS * (NDTEND - NDTFIRST)
            )
            if TIME_SS - years_total < tstep_rpt / 365.25:
                tstep_rpt = (TIME_SS - years_total) * 365.25
            time_rpt += tstep_rpt
            __write("TUNING")
            __write(f"    1* {ts}   1* {TSTEP_MIN} /")
            __write("TIME")
            __write(f"    {time_rpt} /")
            __write("")
            years_total += tstep_rpt / 365.25
            ts *= TMULT

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

        # # POSTSRC
        # __write("POSTSRC")
        # __write("  MAGMASRC /")
        # __write("/")
        # __write("")  # \n

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
                while time < TEND_UNREST * 365.25:
                    time += TRPT_UNREST
                    f.write(f"TUNING\n")
                    # f.write(f"    1* {TSTEP_UNREST}   1* {TSTEP_MIN} /\n")  #!
                    f.write(f"    1* {250.0/3600.0/24.0}   1* {TSTEP_MIN} /\n")  #!タイムステップを小さくするなら、TRPTも小さくする
                    f.write(f"TIME\n")
                    f.write(f"    {time} /\n")
                    f.write(f"\n")

            f.write(line)

def generate_mikasa_input(scenario: BaseSenario, runpth: PathLike, refpth: Optional[PathLike]=None):
    """Generate RUN file from PARAMS class

    Args:
        scenario (BaseScenario): Instance containing simulation condition
        runpth (PathLike): Path of RUN file.
    """
    condition: Condition = scenario.get_condition()

    assert DEFAULT_ROCK_PROPS[RockType.AIR]["TEMPC"] == TEMPE_AIR, (condition[RockType.AIR.name]["TEMPC"], TEMP_RAIN)
    makedirs(CACHE_DIR, exist_ok=True)
    cache_topo = CACHE_DIR.joinpath("topo_ls")
    topo_ls: List = None
    lat_2d, lng_2d = None, None

    if cache_topo.exists():
        print("Generate topo data from cache file")
        with open(cache_topo, "rb") as pkf:
            topo_ls, (xc_m, yc_m, zc_m, lat_2d, lng_2d) = pickle.load(
                pkf
            )
    else:
        data = load_vertical_data(COAL_LAYER_DATA_PTH,
                       COAL_VERTICAL_SECTIONS,
                       COAL_LAYER_NAME)
        # hdata = load_horizontal_data(COAL_HORIZONTAL_DATA_PTH[0],
        #                              COAL_HORIZONTAL_DATA_PTH[1],
        #                              COAL_LAYER_NAME)
        # data.merge(hdata)  # 水平を入れるとおかしくなったので、いったん無視
        model = interp_layer([data])
        topo_ls, (xc_m, yc_m, zc_m, lat_2d, lng_2d) = generate_topo(model)
        with open(cache_topo, "wb") as pkf:
            pickle.dump(
                (topo_ls, (xc_m, yc_m, zc_m, lat_2d, lng_2d)),
                pkf,
                pickle.HIGHEST_PROTOCOL,
            )

    # debug
    # nxyz = (len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2]))
    # plt_topo([v.value for v in topo_ls], lat_2d, lng_2d, nxyz, "debug")
    
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
        condition,
    )

    tend = TIME_SS
    if refpth is not None:
        nxyz = nxyz[0] * nxyz[1] * nxyz[2]
        cellid_props, _, _, time = load_sum(refpth)
        tempe_ls = get_v_ls(cellid_props, "TEMPC")[:nxyz]
        pres_ls = get_v_ls(cellid_props, "PRES")[:nxyz]
        xco2_ls = get_v_ls(cellid_props, "COMP1T")[:nxyz]
        tend = TEND_UNREST
    else:
        tempe_ls, pres_ls, xco2_ls = None, None, None

    # debug
    nxyz = (len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2]))
    plt_topo(perm_ls[0], lat_2d, lng_2d, nxyz, "./debug/perm")

    # relative permeability
    sattab = calc_sattab(method="None")

    # welldata
    if refpth is None:
        well_props, well_specs, well_compdat = None, None, None
    else:
        well_props = scenario.get_well_props()
        well_specs, well_compdat = generate_well_specs([well_props["LATLNG_HEAD"]],
                                                    [well_props["WELL_ID"]],
                                                    well_props["ULIM"],
                                                    xc_m,
                                                    yc_m,
                                                    topo_ls,)
    
    # src_props: OrderedDict = OrderedDict()
    # src_props.setdefault("MAGM0", {
    #     "i": srcpos[0] + 1,
    #     "j": srcpos[1] + 1,
    #     "k": srcpos[2] + 1,
    #     "pres": params.PRES_SRC,
    #     "tempe": params.SRC_TEMP,
    #     "comp1t": params.SRC_COMP1T,
    #     "inj_rate": params.INJ_RATE
    # })
    # if params.disperse_magmasrc:
    #     src_props["MAGM0"]["inj_rate"] /= 5.0
    #     for i in range(1, 5):
    #         dx, dy = 1, 1  # python index to MUFITS index
    #         if i == 1:
    #             dx += 1
    #         if i == 2:
    #             dy += 1
    #         if i == 3:
    #             dx -= 1
    #         if i == 4:
    #             dy -= 1
    #         src_props.setdefault(f"MAGM{i}", {
    #             "i": srcpos[0] + dx,
    #             "j": srcpos[1] + dy,
    #             "k": srcpos[2] + 1,
    #             "pres": params.PRES_SRC,
    #             "tempe": params.SRC_TEMP,
    #             "comp1t": params.SRC_COMP1T,
    #             "inj_rate": params.INJ_RATE / 5.0
    #         })

    # # debug
    # with open(Path(pth).parent.joinpath("debug.pkl"), "wb") as pkf:
    #     pickle.dump(
    #         (
    #             actnum_ls,
    #             rocknum_ls,
    #             perm_ls,
    #             rocknum_params,
    #             lateral_bounds,
    #             bottom_bounds,
    #             rocknum_ptgrad,
    #             tempe_ls,
    #             pres_ls,
    #             xco2_ls,
    #             m_airbounds,
    #             surf_lateral,
    #         ),
    #         pkf,
    #     )

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
        sattab,
        tend,
        runpth,
        well_specs,
        well_compdat,
        well_props,
    )
    return

if __name__ == "__main__":
    pass
