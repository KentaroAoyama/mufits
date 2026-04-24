"""Load .SUM file and monitor processes"""
from os import PathLike, access, R_OK, path, makedirs, kill, getcwd
import struct
from typing import List, Tuple, Dict, Any, Union, Optional, Literal
from pathlib import Path
from math import exp, log10
from time import sleep, time
from logging import Logger
import re
from statistics import median, mean
from subprocess import Popen

import numpy as np
import pandas as pd
from shapely import Polygon
from pyproj import Transformer
from matplotlib import pyplot as plt
import matplotlib as mpl
import cv2
import pickle
from tqdm import tqdm

OBSDIR = Path(getcwd()).joinpath("obsdata")

from constants import (
    CONVERSION_CRITERIA,
    DXYZ,
    ORIGIN,
    CACHE_DIR,
    IDX_AIR,
    IDX_CAP,
    IDX_CAPVENT,
    IDX_LAKE,
    IDX_SEA,
    CONDS_PID_MAP_NAME,
    OUTDIR,
    CRS_DEM,
    CRS_RECT,
    POS_SINK,
)
from utils import (
    calc_ijk,
    stack_from_center,
    stack_from_0,
    condition_to_dir,
    dir_to_condition,
    generate_simple_vent,
    generate_simple_cap,
    load_sum,
    load_snap,
    get_v_ls,
    get_fpth_in_timeseries,
    get_sumpth_time
)

NX, NY, NZ = len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2])

# mapping for matplotlib
PROP_NAME_MAP = {"TEMPC": "Temperature (℃)",
                 "PRES": "Pressure (MPa)",
                 "COMP1T": "CO$_2$ (Molar Fraction)",
                 "SAT#GAS": "Gas Saturation",
                 "DENT": "Fluid Density (kg/m$^{3}$)"
                 }

def _prepare_ticks(plt: plt, axes: List[plt.Axes], params: Tuple = (7, 5, 7, 5), labelsize=14, inner: bool = False) -> None:
    if inner:
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['axes.axisbelow'] = True
    for ax in axes:
        ax.tick_params(axis="x", which="major", length=params[0])
        ax.tick_params(axis="x", which="minor", length=params[1])
        ax.tick_params(axis="y", which="major", length=params[2])
        ax.tick_params(axis="y", which="minor", length=params[3])
        ax.tick_params(labelsize=labelsize)


def calc_prop_diff(props0: Dict, props1: Dict, prop_name: str) -> float:
    v1_ls, v2_ls = get_v_ls(props0, prop_name), get_v_ls(props1, prop_name)
    return np.sqrt(np.square(np.array(v1_ls) - np.array(v2_ls)).sum())


def load_props_ls(i_start: int, dirpth: PathLike) -> List[Tuple[Dict, Dict, float]]:
    props_ls: List[Tuple[Dict, Dict, float]] = []
    for i in range(i_start, 100000):
        fn = str(i).zfill(4)
        fpth = dirpth.joinpath(f"tmp.{fn}.SUM")
        if not (fpth.exists() and access(fpth, R_OK)):
            break
        if _is_writting(fpth) or not _is_enough_size(fpth):
            break
        cellprops1, srcprops1, time = load_sum(fpth)
        props_ls.append((cellprops1, srcprops1, time))
    return props_ls


def calc_change_rate(
    props_ls: List[Tuple[Dict, Dict, float]], prop_name: str
) -> Tuple[List, List]:
    time_ls, diff_ls = [], []
    time0: float = None
    cellprops0 = None
    for cellprops1, _, time in props_ls:
        if cellprops0 is None:
            time0 = time
            cellprops0 = cellprops1
            continue
        time_ls.append(time)
        diff_ls.append(
            calc_prop_diff(cellprops0, cellprops1, prop_name) / (time - time0)
        )
        cellprops0 = cellprops1
        time0 = time
    return time_ls, diff_ls


def plt_conv(time_ls, changerate_ls, fpth: PathLike):
    fig, ax = plt.subplots()
    ax.plot(time_ls, changerate_ls)
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("DAYS")
    ax.set_ylabel("Change Rate")
    fig.savefig(fpth, bbox_inches="tight", dpi=200)
    plt.clf()
    plt.close()


# TODO: fix or delete
def monitor_process(conds_dct: Dict[Tuple, Any]) -> Union[None, int]:
    # i, change_rate
    conds_status: Dict = {}
    conds_remain_ls = [i for i in conds_dct]
    # process_ls = process_ls.copy()
    while len(conds_remain_ls) > 0:
        if len(conds_dct) == 0:
            return None
        # for conds, process in zip(conds_remain_ls, process_ls):
        for conds in conds_remain_ls:
            prop = conds_dct[conds]
            SimDir: Path = prop["DirPth"]
            MonitorPth: Path = prop["MonitorPth"]
            logger: Logger = prop["Logger"]
            status: Dict = conds_status.setdefault(conds, {})
            i = status.get("i", 0)
            props_ls: List = load_props_ls(i, SimDir)
            if len(props_ls) < 2:
                continue
            i += len(props_ls)
            status["i"] = i
            _is_converged = True
            for cou, (metric, criteria) in enumerate(CONVERSION_CRITERIA.items()):
                time_ls, changerate_ls = calc_change_rate(props_ls, metric)
                # Confirm convergent or not
                if changerate_ls[-1] > criteria:
                    _is_converged = False
                # extend
                if cou == 0:
                    _extend(status, "time", time_ls)
                    logger.debug("=======================")
                    logger.debug(f"TIME: {time_ls[-1]} DAYS")
                _extend(status, metric, changerate_ls)
                logger.debug(f"{metric}: {changerate_ls[-1]}")

                # plot
                plt_conv(
                    status["time"], status[metric], MonitorPth.joinpath(f"{metric}.png")
                )

            if _is_converged:
                _str = str(condition_to_dir(OUTDIR, *conds))
                pid: int = None
                with open(
                    Path(SimDir).joinpath("tmp").joinpath(CONDS_PID_MAP_NAME), "r"
                ) as f:
                    for line in reversed(f.readlines()):
                        if _str in line:
                            line = line.replace("\n", "")
                            line = line.replace(f"{_str}, ", "")
                            pid = int(line)
                kill(pid, 15)
                conds_remain_ls.remove(conds)
                logger.debug("DONE")
                print(f"{MonitorPth} DONE")


def _extend(status: Dict, key: str, new_list: List) -> None:
    _ls: List = status.setdefault(key, [])
    _ls.extend(new_list)


def _is_writting(fpth: PathLike) -> bool:
    if time() - path.getmtime(fpth) < 180.0:
        return True
    else:
        return False


def _is_enough_size(fpth: PathLike, criteria: int = 2000000) -> bool:
    if path.getsize(fpth) > criteria:
        return True
    else:
        False


def generate_3darr(v_ls: List[float],
                   axis: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(v_ls)==NX*NY*NZ
    cache_topo = CACHE_DIR.joinpath("topo_ls")
    topo_ls: List[int] = None
    if cache_topo.exists():
        with open(cache_topo, "rb") as pkf:
            topo_ls, _ = pickle.load(pkf)
    # make array
    axis = axis.lower()
    val_3d = np.zeros(shape=(NZ, NY, NX))
    for m, v in enumerate(v_ls):
        i, j, k = calc_ijk(m, NX, NY)
        if topo_ls is None:
            val_3d[k][j][i] = v
        elif topo_ls[m] == IDX_AIR:
            val_3d[k][j][i] = np.nan
        else:
            val_3d[k][j][i] = v

    # transpose
    grid_x, grid_y = None, None
    if axis == "x":
        val_3d = np.transpose(val_3d, (2, 0, 1))
        # val_3d = np.flip(val_3d, 1)
        val_3d = np.flip(val_3d, 2)
        val_3d = np.flip(val_3d, 1)
        grid_x, grid_y = np.meshgrid(
            np.array(stack_from_center(DXYZ[1])),
            np.flip(ORIGIN[2] - np.array(stack_from_0(DXYZ[2]))),
        )
    if axis == "y":
        val_3d = np.transpose(val_3d, (1, 0, 2))
        val_3d = np.flip(val_3d, 0)
        val_3d = np.flip(val_3d, 1)
        grid_x, grid_y = np.meshgrid(
            np.array(stack_from_center(DXYZ[0])),
            np.flip(ORIGIN[2] - np.array(stack_from_0(DXYZ[2]))),
        )
    if axis == "z":
        val_3d = np.flip(val_3d, 0)
        val_3d = np.flip(val_3d, 1)
        grid_x, grid_y = np.meshgrid(
            np.array(stack_from_center(DXYZ[0])),
            np.flip(-np.array(stack_from_center(DXYZ[1])))
        )
    return grid_x, grid_y, val_3d


def generate_flux_arr(flux: Tuple[List[float],List[float],List[float]],
                      axis: str,) -> Tuple[np.ndarray, np.ndarray]:
    assert len(flux[0])==len(flux[1])==len(flux[2])==NX*NY*NZ
    topo_ls: List[int] = None
    cache_topo = CACHE_DIR.joinpath("topo_ls")
    if cache_topo.exists():
        with open(cache_topo, "rb") as pkf:
            topo_ls, _ = pickle.load(pkf)
    axis = axis.lower()
    u_ls: List[float] = None
    v_ls: List[float] = None
    if axis=="x":
        u_ls = flux[1]
        v_ls = flux[2]
        u_ls = [-u for u in u_ls]
        v_ls = [-v for v in v_ls]
    if axis=="y":
        u_ls = flux[0]
        v_ls = flux[2]
        v_ls = [-v for v in v_ls]
    if axis=="z":
        u_ls = flux[0]
        v_ls = flux[1]
        v_ls = [-v for v in v_ls]
    u_3d = np.zeros(shape=(NZ, NY, NX))
    v_3d = np.zeros(shape=(NZ, NY, NX))
    for m, (u,v) in enumerate(zip(u_ls, v_ls)):
        i, j, k = calc_ijk(m, NX, NY)
        if topo_ls is None:
            u_3d[k][j][i] = u
            v_3d[k][j][i] = v
        elif topo_ls[m] == IDX_AIR:
            u_3d[k][j][i] = np.nan
            v_3d[k][j][i] = np.nan
        else:
            u_3d[k][j][i] = u
            v_3d[k][j][i] = v
    
    # transpose
    if axis == "x":
        u_3d = np.transpose(u_3d, (2, 0, 1))
        u_3d = np.flip(u_3d, 2)
        u_3d = np.flip(u_3d, 1)
        v_3d = np.transpose(v_3d, (2, 0, 1))
        v_3d = np.flip(v_3d, 2)
        v_3d = np.flip(v_3d, 1)
    if axis == "y":
        u_3d = np.transpose(u_3d, (1, 0, 2))
        u_3d = np.flip(u_3d, 0)
        u_3d = np.flip(u_3d, 1)
        v_3d = np.transpose(v_3d, (1, 0, 2))
        v_3d = np.flip(v_3d, 0)
        v_3d = np.flip(v_3d, 1)
    if axis == "z":
        u_3d = np.flip(u_3d, 0)
        u_3d = np.flip(u_3d, 1)
        v_3d = np.flip(v_3d, 0)
        v_3d = np.flip(v_3d, 1)
    return (u_3d, v_3d)


def plt_single_cs(grid_x: np.ndarray,
                  grid_y: np.ndarray,
                  val_3d: np.ndarray,
                  idx: int,
                  prop_name: str,
                  vmin: float,
                  vmax: float,
                  fpth: PathLike,
                  flux: Optional[Tuple[np.ndarray, np.ndarray]]=None):

    # prop name to universal name (e.g., TEMPC → Temperature (℃))
    if PROP_NAME_MAP.get(prop_name, None):
        prop_name = PROP_NAME_MAP.get(prop_name)
    val2d = val_3d[idx]
    fig, ax = plt.subplots()
    mappable = ax.pcolormesh(
        grid_x,
        grid_y,
        val2d,
        vmin=vmin,
        vmax=vmax,
    )
    if flux is not None:
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.streamplot.html
        # https://stackoverflow.com/questions/51843313/flow-visualisation-in-python-using-curved-path-following-vectors
        U, V = flux[0][idx], flux[1][idx]
        norm = np.sqrt(np.square(U) + np.square(V))
        ax.quiver(grid_x,
                  grid_y,
                  U/norm*0.1,
                  V/norm*0.1,
                  scale=1.5e-3,
                  # norm,
                  angles="xy",
                  scale_units="xy",
                  color="w",
                  )
    pp = fig.colorbar(mappable, ax=ax, orientation="vertical")
    pp.set_label(prop_name)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    fig.savefig(fpth, dpi=500, bbox_inches="tight")
    plt.clf()
    plt.close()


def plot_values(
    v_ls: List[float],
    prop_name: str,
    savedir: PathLike,
    axis="Y",
    vmin: float = None,
    vmax: float = None,
    indexes: Tuple[int] = None,
    flux: Tuple[List[float], List[float], List[float]]=None,
) -> None:
    v_ls = v_ls[: NX*NY*NZ]
    grid_x, grid_y, val_3d = generate_3darr(v_ls, axis)
    if flux is not None:
        flux = generate_flux_arr(flux, axis)
    if indexes is None:
        indexes = list(range(len(val_3d)))
    dirpth = Path(savedir)
    makedirs(dirpth, exist_ok=True)
    for i in range(len(val_3d)):
        if indexes is not None:
            if i not in indexes:
                continue
        fpth = dirpth.joinpath(f"{i}.png")
        plt_single_cs(grid_x, 
                      grid_y,
                      val_3d,
                      i,
                      prop_name,
                      vmin,
                      vmax,
                      fpth,
                      flux=flux
                      )


def plot_results(
    fpth: PathLike,
    axis: Tuple[str] = ("X", "Y", "Z"),
    vmin: float = None,
    vmax: float = None,
    prop_ls: List[str] = list(CONVERSION_CRITERIA.keys()),
) -> None:
    fpth = Path(fpth)
    props = load_snap(fpth, prop_ls)
    for i, prop_name in enumerate(prop_ls):
        for ax in axis:
            savedir = fpth.parent.joinpath(prop_name).joinpath(ax)
            makedirs(savedir, exist_ok=True)
            plot_values(
                props[i][1],
                prop_name,
                savedir,
                axis=ax,
                vmin=vmin,
                vmax=vmax,
            )

def is_converged(cond_dir: Path) -> bool:
    # check if exists .vtu file
    for fpth in cond_dir.glob("**/*"):
        if ".vtu" in str(fpth):
            return True

    # check logfile in tmp dir
    logpth = cond_dir.joinpath("tmp").joinpath("log.txt")
    if not logpth.exists():
        return False
    with open(logpth, "r") as f:
        for line in reversed(f.readlines()):
            if "DONE" in line:
                return True

    day = 19.0 * 60.0 * 60.0
    time_ls: List = []
    for fpth in cond_dir.glob("**/*"):
        if ".SUM" in str(fpth):
            time_ls.append(fpth.stat().st_mtime)
    if len(time_ls) == 0:
        return False
    elif max(time_ls) - min(time_ls) > day:
        return True

    return False


# TODO: fix or delete
def optimize_tstep(sim_dir: PathLike):
    sim_dir: Path = Path(sim_dir)
    perm_dt: Dict = {}
    for conds_dir in sim_dir.iterdir():
        # if not _is_converged(conds_dir):
        #     continue
        # load logfile and get maximum time step
        logpth = conds_dir.joinpath("log.txt")
        if not logpth.exists():
            continue
        print(conds_dir)
        t, xco2, q, p = dir_to_condition(conds_dir)
        with open(logpth, "r") as f:
            lines: List[str] = f.readlines()
            dt_ls = []
            for i, line in enumerate(lines):
                if "WAR: RECALCULATION" in line:
                    dtline = lines[i + 3]
                    dt_ls.append(
                        float(
                            re.search(r"\d+.\d+ DAYS", dtline)
                            .group()
                            .replace(" DAYS", "")
                        )
                    )
            if len(dt_ls) > 0:
                _ls: List = perm_dt.setdefault(p, [[], []])
                _ls[0].append(q)
                _ls[1].append(median(dt_ls))

    # fit
    def _func(p_ls, q_ls, A: float = 0.0001, B: float = 0.2):
        v = []
        for _p, _q in zip(p_ls, q_ls):
            _max = 50.0 * (exp(-B * (log10(_p) - 1.0)))
            _max *= exp(-A * _q)
            v.append(_max)
        return v

    fig, ax = plt.subplots()
    for perm, results in perm_dt.items():
        q, ts = results[0], results[1]
        q = sorted(q)

        ax.plot(q, _func([perm for _ in range(len(q))], q))
        ax.scatter(results[0], results[1], label=perm)
    ax.legend()
    ax.set_xlabel("Mass Rate (t/day)")
    ax.set_ylabel("Maximum Time Step (day)")
    fig.savefig("tmp.png", dpi=300)
    # plt.show()
    plt.clf()
    plt.close()


def warning_tstep(dirpth: PathLike) -> List[List]:
    dirpth = Path(dirpth)
    logpth = dirpth.joinpath("log.txt")
    days, tsteps = [], []
    with open(logpth, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "WAR: RECALCULATION" in line:
                m = re.search(r"\d+\.\d+ DAYS", lines[i + 2])
                days.append(float(m.group().replace(" DAYS", "")))
                m = re.search(r"\d+\.\d+E.\d+ SEC", lines[i + 3])
                tsteps.append(float(m.group().replace(" SEC", "")))
    return days, tsteps


def plt_warning_tstep(dirpth: PathLike) -> None:
    dirpth = Path(dirpth)
    x, y = warning_tstep(dirpth)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("DAYS")
    ax.set_ylabel("TIMESTEP (in SEC)")
    fig.savefig(dirpth.joinpath("warning_tstep.png"), dpi=200, bbox_inches="tight")



def plt_progress_rate(dirpth: PathLike) -> None:
    dirpth = Path(dirpth)
    logpth = dirpth.joinpath("log.txt")
    days, line_ls = [], []
    with open(logpth, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "TIME REPORT. STEP" in line:
                m = re.search(r"\d+\.\d+ DAYS", lines[i + 1])
                days.append(float(m.group().replace(" DAYS", "")))
                line_ls.append(i)
    fig, ax = plt.subplots()
    ax.plot(line_ls, days)
    ax.set_xlabel("LINES")
    ax.set_ylabel("DAYS")
    fig.savefig(dirpth.joinpath("progress_rate.png"), dpi=200, bbox_inches="tight")


def plt_latests(
    cond_dir: Path,
    axes: Tuple[str] = ("Y",),
    show_time: bool = True,
    vmin: float = None,
    vmax: float = None,
    prop_ls: List[str] = list(CONVERSION_CRITERIA.keys()),
) -> None:
    cond_dir = Path(cond_dir)
    fpth_ls = []
    for i in range(10000):
        fn = str(i).zfill(4)
        fpth = cond_dir.joinpath(f"tmp.{fn}.SUM")
        if fpth.exists():
            fpth_ls.append(fpth)
    if len(fpth_ls) == 0:
        return
    plot_results(
        cond_dir.joinpath(fpth_ls[-1]),
        axes,
        show_time=show_time,
        vmin=vmin,
        vmax=vmax,
        prop_ls=prop_ls,
    )


def check_convergence(dirpth: PathLike, outpth: PathLike = None):
    dirpth = Path(dirpth)
    conv_dct: Dict = {}
    if outpth is None:
        outpth = dirpth.joinpath("conv.txt")
    f = open(outpth, "w")
    for cond_pth in dirpth.iterdir():
        print(cond_pth)
        sumpth_ls: List = []
        cou = 0
        for i in reversed(list(range(10000))):
            fn = str(i).zfill(4)
            fpth = cond_pth.joinpath(f"tmp.{fn}.SUM")
            if fpth.exists():
                sumpth_ls.append(fpth)
                cou += 1
            if cou == 2:
                continue
        if len(sumpth_ls) < 2:
            print(f"skip: {cond_pth}")
            continue
        props_ls = [load_sum(sumpth_ls[1]), load_sum(sumpth_ls[0])]  # TODO: refactor
        f.write(f"{cond_pth}:\n")
        for prop_name in CONVERSION_CRITERIA:
            time_ls, diff_ls = calc_change_rate(props_ls, prop_name)
            f.write(f"   {prop_name}: {diff_ls[-1]}\n")
    f.close()


def load_results_and_plt_conv(dirpth: PathLike):
    dirpth = Path(dirpth)
    sumpth_ls: List = []
    for i in range(10000):
        fn = str(i).zfill(4)
        fpth = dirpth.joinpath(f"tmp.{fn}.SUM")
        if fpth.exists():
            sumpth_ls.append(fpth)
        else:
            break
    # load properties
    props_ls = []
    for sumpth in sumpth_ls:
        props_ls.append(load_sum(sumpth))  # TODO: refactor
    # calc chagne rate
    props_change_rate_ls: Dict = {}
    for prop_name in CONVERSION_CRITERIA:
        time_ls, diff_ls = calc_change_rate(props_ls, prop_name)
        plt_conv(time_ls, diff_ls, dirpth.joinpath("tmp").joinpath(f"{prop_name}.png"))

# TODO: get_fumarole_propの呼び出し側
def get_fumarole_prop(v_ls: List[float], prop_name: str, calc_average: bool = True):
    # coordinates
    nx, ny, nz = len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2])
    x = np.array(stack_from_center(DXYZ[0]))
    y = -1.0 * np.array(stack_from_center(DXYZ[1]))

    transformer = Transformer.from_crs(CRS_DEM, CRS_RECT, always_xy=True)
    x0, y0 = transformer.transform(ORIGIN[1], ORIGIN[0])
    coords_fumarole: Dict = {}
    for name, pos in POS_SINK.items():
        xtmp, ytmp = transformer.transform(pos[1], pos[0])
        coords_fumarole.setdefault(name, (xtmp - x0, ytmp - y0))

    cache_topo = CACHE_DIR.joinpath("topo_ls")
    with open(cache_topo, "rb") as pkf:
        topo_ls, _ = pickle.load(pkf)

    props: Dict = {}
    for name, (xf, yf) in coords_fumarole.items():
        # get closest grid
        i = np.argmin(np.square(x - xf))
        j = np.argmin(np.square(y - yf))
        for k in range(nz):
            m = calc_m(i, j, k, nx, ny)
            if topo_ls[m] in (IDX_LAND, IDX_VENT, IDX_CAP, IDX_CAPVENT):
                v = v_ls[m]
                if prop_name == "FLUXK#E":
                    v *= -1
                props.setdefault(name, v)
                break
    # add highest temperature point
    props.setdefault("1819", v_ls[calc_m(18, 19, 0, nx, ny)])
    if calc_average:
        with open(
            Path.cwd().joinpath("analyse_dome_coords").joinpath("m_ls"), "rb"
        ) as pkf:
            mdome_ls = pickle.load(pkf)
        vtmp_ls = []
        for m in mdome_ls:
            vtmp_ls.append(v_ls[m])
        _ave = mean(vtmp_ls)
        if prop_name == "FLUXK#E":
            _ave *= -1.0
        props.setdefault("Average", _ave)
        if prop_name == "FLUXK#E":
            _sum = -1.0 * sum(vtmp_ls)
            props.setdefault("Sum", _sum)
    return props


def get_latest_fumarole_prop(
    cond_dir: PathLike,
    prop_name: str = "TEMPC",
    calc_average: bool = True
):
    cond_dir = Path(cond_dir)
    fpth_ls = []
    for i in range(10000):
        fn = str(i).zfill(4)
        fpth = cond_dir.joinpath(f"tmp.{fn}.SUM")
        if fpth.exists():
            fpth_ls.append(fpth)

    if len(fpth_ls) == 0:
        return
    _, v_ls = load_snap(fpth_ls[-1], [prop_name])[0]
    props: Dict = get_fumarole_prop(v_ls, prop_name, calc_average)
    with open(cond_dir.joinpath(f"fumarole_{prop_name}.txt"), "w") as f:
        for name, v in props.items():
            f.write(f"{name}: {v}\n")


def plot_sum_foreach_tstep(
    simdir: PathLike,
    axes=("X", "Y", "Z"),
    prop_names: List[str] = [
        "TEMPC",
    ],
    idx_ls=(
        [20,],
        [20,],
        [20,],
    ),
    showtime: bool = False,
    minmax: Optional[Tuple] = None,
    diff: bool = False,
):
    # TODO: 下のサブディレクトリを含めてプロットする仕様に変更する
    assert len(axes) == len(idx_ls)
    if minmax is not None:
        assert len(prop_names) == len(minmax)
    simdir = Path(simdir)
    fpth_ls = get_fpth_in_timeseries(simdir)
    nx, ny, nz = len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2])
    nxyz = nx * ny * nz
    time, time0 = 0.0, 0.0
    itern0: str = None
    v0_dct: Dict = {}
    for i, fpth in enumerate(fpth_ls):
        print(fpth)
        prop_ls = load_snap(fpth, prop_names)
        timetmp = prop_ls[0][0]
        m = re.search(r"ITER_\d+", str(fpth.parent))
        if m is not None:
            iterntmp = m.group()
        # fix time
        if m is None:
            time0 = timetmp
            time = timetmp
        elif itern0 is None and m is not None:
            time = time0 + timetmp
            itern0 = iterntmp
        elif itern0 != iterntmp:
            time0 = time
            itern0 = iterntmp
            time = time0 + timetmp
        else:
            time = time0 + timetmp
        for j, prop_name in enumerate(prop_names):
            v_ls = prop_ls[j][1][:nxyz]
            if diff:
                if i == 0:
                    v0_dct.setdefault(prop_name, v_ls)
                if v0_dct.get(prop_name, None) is not None:
                    v_ls = [v1 - v0 for v1, v0 in zip(v_ls, v0_dct.get(prop_name))]
            for k, ax in enumerate(axes):
                grid_x, grid_y, val_3d = generate_3darr(v_ls, ax)
                if prop_name == "FLUXK#E":
                    val_3d *= -1.0
                time_dir = simdir.joinpath("tstep").joinpath(prop_name).joinpath(ax)
                if diff:
                    time_dir = time_dir.joinpath("diff")
                makedirs(time_dir, exist_ok=True)
                for idx in idx_ls[k]:
                    fpth = time_dir.joinpath(f"{time}_{idx}.png")
                    if fpth.exists():
                        continue
                    vmin: Optional[float]
                    vmax: Optional[float]
                    if minmax is not None:
                        vmin, vmax = minmax[j]
                    else:
                        vmin, vmax = None, None
                    plt_single_cs(grid_x,
                                  grid_y,
                                  val_3d,
                                  idx,
                                  prop_name,
                                  showtime,
                                  vmin,
                                  vmax,
                                  fpth)


def get_obs_props(prop_names: List[str]) -> Dict:
    # lim, 物性値のキー, 火口のキー
    # suffix to mufits key in obsdata dir
    suffix_mkey = {"tempe": "TEMPC",
                   "height": "FLUXK#E",
                   "CO2": "COMP1T"}
    props: Dict[str, Dict] = {}
    for fpth in OBSDIR.glob("*.csv"):
        _name = fpth.name.replace(".csv", "")
        fumarore, suffix = _name.split("_")
        mkey = suffix_mkey[suffix]
        if mkey in prop_names:
            _dct: Dict = props.setdefault(mkey, {})
            __dct: Dict = _dct.setdefault(fumarore, {})
            _df = pd.read_csv(fpth)
            columns = _df.columns.tolist()
            __dct.setdefault(0, _df[columns[0]].tolist())
            __dct.setdefault(1, _df[columns[1]].tolist())
    return props


def plot_fumarole_props_foreach_tstep(
    simdir: PathLike,
    prop_names: List[str] = [
        "TEMPC", "FLUXK#E", "COMP1T"
    ],
    calc_average: bool = True,
    step: int = 1,
    with_obs: bool = True, 
    obs_props: Dict = None
):
    # TODO: 下のサブディレクトリを含めてプロットする仕様に変更する
    simdir = Path(simdir)
    fpth_ls = get_fpth_in_timeseries(simdir)

    if with_obs and obs_props is None:
        obs_props: Dict = get_obs_props(prop_names)
    
    props: Dict = {}
    time0, time = 0.0, 0.0
    itern0: str = None
    for fpth in tqdm(fpth_ls[::step]):
        prop_ls = load_snap(fpth, prop_names)
        timetmp = prop_ls[0][0]
        m = re.search(r"ITER_\d+", str(fpth.parent))
        if m is not None:
            iterntmp = m.group()
        # fix time
        if m is None:
            time0 = timetmp
            time = timetmp
        elif itern0 is None and m is not None:
            time = time0 + timetmp
            itern0 = iterntmp
        elif itern0 != iterntmp:
            time0 = time
            itern0 = iterntmp
            time = time0 + timetmp
        else:
            time = time0 + timetmp
        props_time: Dict = props.setdefault(time, {})
        for i, prop_name in enumerate(prop_names):
            _prop = get_fumarole_prop(prop_ls[i][1], prop_name, calc_average)
            props_time.setdefault(prop_name, _prop)
    
    # plot
    savedir = simdir.joinpath("tstep").joinpath("fumarole")
    makedirs(savedir, exist_ok=True)
    sim_dct: Dict = {}
    for prop_name in prop_names:
        time_ls = []
        name_v: Dict = {}
        for time, _props in props.items():
            time_ls.append(time)
            for name, v in _props[prop_name].items():
                name_v.setdefault(name, []).append(v)
        _dct: Dict = sim_dct.setdefault(prop_name, {})
        _dct.setdefault("time", time_ls)
        _dct.setdefault("name_v", name_v)
        for name, v_ls in name_v.items():
            fig, ax = plt.subplots()
            ax.plot(time_ls, v_ls)
            figpth = savedir.joinpath(f"{prop_name}_{name}.png")
            fig.savefig(figpth, dpi=200, bbox_inches="tight")
            plt.clf()
            plt.close()
            
    # with obs
    if obs_props is None:
        return
    cmap = mpl.colormaps['viridis']
    name_color: Dict = {}
    name_in_obs = []
    for prop_name, dct in obs_props.items():
        name_in_obs.extend(list(dct.keys()))
    name_in_obs = list(set(name_in_obs))
    fnames = ["A", "B", "1819"]
    for i, name in enumerate(fnames):
        name_color.setdefault(name, cmap(i/len(fnames)))
    for prop_name, fum_dct in obs_props.items():
        isflux = False
        if prop_name == "FLUXK#E":
            isflux = True
        fig, ax = plt.subplots()
        if isflux:
            ax2 = ax.twinx()
            _prepare_ticks(plt, [ax, ax2], inner=True)
        else:
            _prepare_ticks(plt, [ax,], inner=True)
        # obs
        t_tmp = []
        for name, values in fum_dct.items():
            if isflux:
                ax2.scatter(values[0], values[1], label=f"OBS: {name}", color=name_color[name])
            else:
                ax.scatter(values[0], values[1], label=f"OBS: {name}", color=name_color[name])
            t_tmp.extend(values[0]) # time
        # simulation results
        start = 1990.0 #!
        props = sim_dct[prop_name]
        for i, name in enumerate(fnames):
            if isflux:
                name = "Sum"
                name_color.setdefault(name, cmap(0))
            time_ls = [t / 365.25 + start for t in props["time"]]
            ax.plot(time_ls, props["name_v"][name], label=f"SIM: {name}", color=name_color[name])
            # before start
            if min(t_tmp) <= start:
                tls = np.linspace(min(t_tmp), start, 200)
                vls = np.full(tls.shape, props["name_v"][name][0])
                ax.plot(tls, vls, color=name_color[name], linestyle="dashed")
            if isflux:
                break
        labelsize = 14
        ax.set_xlim(min(tls) - 3.0, max((max(tls) + 30, max(time_ls))) + 3.0)
        ax.set_xlabel("Year", fontsize=labelsize)
        labelname_map = {"FLUXK#E": "Energy Flux (MW)",
                         "TEMPC": "Temperature (℃)",
                         "COMP1T": "CO$_2$ (Molar Fraction)"
                         }
        ax.set_ylabel(labelname_map[prop_name], fontsize=labelsize)
        if isflux:
            ax2.set_ylabel("Plume Height (m)", fontsize=labelsize)
            ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left',frameon=False)
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',frameon=False)
        fig.savefig(savedir.joinpath(f"{prop_name}_withobs.png"), dpi=200, bbox_inches="tight")
        plt.clf()
        plt.close()


def sanity_check(pth, prop_ls: List = ["TEMPC", "PRES", "COMP1T"]):
    cellid_props, srcprops, timetmp = load_sum(pth)
    nxyz = len(DXYZ[0]) * len(DXYZ[1]) * len(DXYZ[2])
    badconds: List = []
    for name in prop_ls:
        _ls = get_v_ls(cellid_props, name)
        _ls = _ls[:nxyz]
        if name == "TEMPC":
            if min(_ls) < 0.0:
                badconds.append(f"min {name} < 0℃: {min(_ls)}")
            if max(_ls) > 1000.0:
                badconds.append(f"max {name} > 1000℃: {max(_ls)}")
        if name == "PRES":
            if min(_ls) < 0.1:
                badconds.append(f"min {name} < 0.1 MPa: {min(_ls)}")
            if max(_ls) > 150.0:
                badconds.append(f"max {name} > 150 MPa: {max(_ls)}")
        if name == "COMP1T":
            if min(_ls) < 0.0:
                badconds.append(f"min {name} < 0: {min(_ls)}")
            if max(_ls) > 1.0:
                badconds.append(f"max {name} > 1: {max(_ls)}")
    print(badconds)
    

def img2mov(imgdir: PathLike, movdir: PathLike= None) -> None:
    imgdir = Path(imgdir)
    if movdir is None:
        movdir = Path(imgdir)
    movdir = Path(movdir)
    idxdct: Dict = {}
    for pth in imgdir.glob('**/*.png'):
        fname = pth.name.replace(".png", "")
        _ls: List = fname.split("_")
        idxdct.setdefault(int(_ls[1]), []).append([float(_ls[0]), pth])
        
    for idx, _ls in idxdct.items():
        time_ls = [_l[0] for _l in _ls]
        pth_ls = [_l[1] for _l in _ls]
        time_ls, pth_ls = zip(*sorted(zip(time_ls, pth_ls)))
        img_ls: List = []
        for j, (time, pth) in enumerate(zip(time_ls, pth_ls)):
            img = cv2.imread(str(pth))
            cv2.putText(img, str(time), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3, 4)
            height, width, layers = img.shape
            if j == 0:
                size = (width, height)
            img_ls.append(img)
        
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(movdir.joinpath(f"{idx}.mp4")), codec, 30000/1001, size, 1)
        for img in img_ls:
            writer.write(img)
        writer.release()

def progress_time(dirpth):
    fpth_ls = get_fpth_in_timeseries(dirpth)
    simdirs = set()
    simdirs.add(Path(dirpth))
    for pth in fpth_ls:
        if "ITER" in str(pth):
            simdirs.add(Path(pth.parent))
    time = 0.0
    for _dirpth in list(simdirs):
        for i in range(10000, -1, -1):
            fn = str(i).zfill(4)
            fpth = _dirpth.joinpath(f"tmp.{fn}.SUM")
            if fpth.exists():
                time += load_sum(fpth, only_time=True)
                break
    return time

Region = Literal["surface", "conduit", "capvent", "aquifer"]

m_in_surface: Optional[set[int]] = None
m_in_conduit: Optional[set[int]] = None
m_in_capvent: Optional[set[int]] = None
m_in_aquifer: Optional[set[int]] = None

def set_m_in_surface() -> None:
    gx, gy, gz = DXYZ
    nx = len(gx)
    ny = len(gy)
    xc = stack_from_center(gx)
    yc = stack_from_center(gy)
    nxyz = nx * ny * len(gz)
    with open(CACHE_DIR.joinpath("topo_ls"), "rb") as pkf:
        topo_ls, _ = pickle.load(pkf)
    global m_in_surface
    m_in_surface = set()
    for m in range(nxyz):
        i, j, k = calc_ijk(m, nx, ny)
        above_700 = ORIGIN[2]-sum(gz[:k])>=700.0
        isin_center = abs(xc[i])<=2000.0 and abs(yc[j])<=2000.0
        not_in_air = topo_ls[m] not in (IDX_AIR, IDX_LAKE, IDX_SEA)
        if above_700 and isin_center and not_in_air:
            m_in_surface.add(m)

def set_m_in_conduit():
    with open(CACHE_DIR.joinpath("topo_ls"), "rb") as pkf:
        topo_ls, (xc_m, yc_m, zc_m, _, _, _, _,) = pickle.load(pkf)
    with open("./analyze_magnetic_coords/elv_bounds_mufits.pkl", "rb") as pkf:
        elv_bounds_mufits: Dict = pickle.load(pkf)
    topo_ls = generate_simple_vent(topo_ls, xc_m, yc_m, zc_m, elv_bounds_mufits)
    global m_in_conduit
    m_in_conduit = set()
    for m, idx in enumerate(topo_ls):
        if idx == IDX_VENT:
            m_in_conduit.add(m)

def set_m_in_capvent():
    with open(CACHE_DIR.joinpath("topo_ls"), "rb") as pkf:
        topo_ls, (xc_m, yc_m, zc_m, _, _, _, _,) = pickle.load(pkf)
    with open("./analyze_magnetic_coords/elv_bounds_mufits.pkl", "rb") as pkf:
        elv_bounds_mufits: Dict = pickle.load(pkf)
    topo_ls = generate_simple_vent(topo_ls, xc_m, yc_m, zc_m, elv_bounds_mufits)
    with open("./analyse_crator_coords/crator.pkl", "rb") as pkf:
        crator_coords: Polygon = pickle.load(pkf)
    generate_simple_cap(topo_ls, xc_m, yc_m, 700.0, crator_coords)
    global m_in_capvent
    m_in_capvent = set()
    for m, idx in enumerate(topo_ls):
        if idx == IDX_CAPVENT:
            m_in_capvent.add(m)

def set_m_in_aquifer():
    if m_in_capvent is None:
        set_m_in_capvent()
    nx, ny, nz = len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2])
    global m_in_aquifer
    m_in_aquifer = set()
    for m in range(nx*ny*nz):
        i, j, k = calc_ijk(m, nx, ny)
        if k == nz-1:
            continue
        m_below = calc_m(i,j,k+1,nx,ny)
        if m_below in m_in_capvent:
            m_in_aquifer.add(m)

def clip_surface(v_ls: List[float]) -> List[float]:
    if m_in_surface is None:
        set_m_in_surface()
    v_clipped: List[float] = []
    for m, v in enumerate(v_ls):
        if m in m_in_surface:
            v_clipped.append(v)
    return v_clipped

def clip_conduit(v_ls: List[float]) -> List[float]:
    if m_in_conduit is None:
        set_m_in_conduit()
    v_clipped: List[float] = []
    for m, v in enumerate(v_ls):
        if m in m_in_conduit:
            v_clipped.append(v)
    return v_clipped

def clip_capvent(v_ls: List[float]) -> List[float]:
    if m_in_capvent is None:
        set_m_in_capvent()
    v_clipped: List[float] = []
    for m, v in enumerate(v_ls):
        if m in m_in_capvent:
            v_clipped.append(v)
    return v_clipped

def clip_aquifer(v_ls: List[float]) -> List[float]:
    if m_in_aquifer is None:
        set_m_in_aquifer()
    v_clipped: List[float] = []
    for m, v in enumerate(v_ls):
        if m in m_in_aquifer:
            v_clipped.append(v)
    return v_clipped

def clip_value(v_ls: List[float], region: Region) -> List[float]:
    assert region in ("surface", "conduit", "capvent", "aquifer")
    if region == "surface":
        v_clipped = clip_surface(v_ls)
    if region == "conduit":
        v_clipped = clip_conduit(v_ls)
    if region == "capvent":
        v_clipped = clip_capvent(v_ls)
    if region == "aquifer":
        v_clipped = clip_aquifer(v_ls)
    return v_clipped

def get_regional_timeseries(dirpth: PathLike,
                            regions: List[Region],
                            prop_names: List[str],
                            ) -> Dict[Region, Dict[str, List[Tuple[float, List[float]]]]]:
    sumpth_ls = get_fpth_in_timeseries(dirpth)
    regionnal_timeseries: Dict[Region, Dict[str, List[Tuple[float, List[float]]]]] = {}
    for sumpth in sumpth_ls:
        print(sumpth)
        props = load_snap(sumpth, prop_names)
        for region in regions:
            prop_timeseries: Dict[str, List[Tuple[float, List[float]]]] = regionnal_timeseries.setdefault(region, {})
            for i, prop_name in enumerate(prop_names):
                timeseries = prop_timeseries.setdefault(prop_name, [])
                timeseries.append((props[i][0], clip_value(props[i][1], region)))
    return regionnal_timeseries

def plt_regional_timeseries(dirpth: PathLike,
                            regions: List[Region],
                            prop_names: List[str]) -> None:
    regionnal_timeseries = get_regional_timeseries(dirpth,
                                                   regions,
                                                   prop_names)
    savedir_parent = Path(dirpth).joinpath("tstep")
    for region, prop_timeseries in regionnal_timeseries.items():
        for prop_name, timeseries in prop_timeseries.items():
            propdir = savedir_parent.joinpath(prop_name)
            makedirs(propdir, exist_ok=True)
            time_ls: List[float] = []
            ave_ls: List[float] = []
            max_ls: List[float] = []
            min_ls: List[float] = []
            iterated = False
            time0 = 0.0
            time0_tmp = 0.0
            for time, v_clipped_ls in timeseries:
                if time==0.0 and time0_tmp>0.0:
                    iterated = True
                    time0 = time0_tmp
                if iterated:
                    time += time0
                time0_tmp = time
                time_ls.append(time/365.25)
                ave_ls.append(mean(v_clipped_ls))
                max_ls.append(max(v_clipped_ls))
                min_ls.append(min(v_clipped_ls))
            fig, ax = plt.subplots()
            _prepare_ticks(plt, axes=[ax])
            ax.plot(time_ls, min_ls, label="Min.")
            ax.plot(time_ls, max_ls, label="Max.")
            ax.plot(time_ls, ave_ls, label="Ave.")
            ax.set_xscale("log")
            ax.set_xlabel("Year")
            ax.set_ylabel(prop_name)
            ax.legend(bbox_to_anchor=(1.05, 1))
            fig.savefig(propdir.joinpath(f"{region}.png"), bbox_inches="tight")
            plt.clf()
            plt.close()
    pass

def time_variation(dirpth: PathLike,
                   prop_names: List[str]):
    dirpth = Path(dirpth)
    sumpth_ls = get_fpth_in_timeseries(dirpth)
    pname_diff: Dict[str, List[List[float], List[float]]] = {}
    time, time0 = 0.0, 0.0
    itern0: str = None
    nx, ny, nz = len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2])
    nxyz = nx * ny * nz
    v_arr = np.zeros(nxyz)
    for i, dx in enumerate(DXYZ[0]):
        for j, dy in enumerate(DXYZ[1]):
            for k, dz in enumerate(DXYZ[2]):
                m = calc_m(i,j,k,nx,ny)
                v_arr[m] = dx*dy*dz
    for i, sumpth in enumerate(sumpth_ls):
        if i == 0:
            continue
        print(sumpth)
        props0 = load_snap(sumpth_ls[i-1], prop_names)
        props1 = load_snap(sumpth, prop_names)
        timetmp = props1[0][0]
        m = re.search(r"ITER_\d+", str(sumpth.parent))
        if m is not None:
            iterntmp = m.group()
        # fix time
        if m is None:
            time0 = timetmp
            time = timetmp
        elif itern0 is None and m is not None:
            time = time0 + timetmp
            itern0 = iterntmp
        elif itern0 != iterntmp:
            time0 = time
            itern0 = iterntmp
            time = time0 + timetmp
        else:
            time = time0 + timetmp
        for i, pname in enumerate(prop_names):
            ls: List = pname_diff.setdefault(pname, [[],[]])
            ls[0].append(time / 365.25)
            diff = np.array(props1[i][1]) - np.array(props0[i][1])
            pv = np.abs(diff[:nxyz])*v_arr
            ls[1].append(np.abs(pv).sum())
    savedir = dirpth.joinpath("tstep").joinpath("variation")
    makedirs(savedir, exist_ok=True)
    for pname in prop_names:
        ls = pname_diff[pname]
        fig, ax = plt.subplots()
        ax.plot(ls[0], ls[1])
        ax.set_xscale("log")
        fig.savefig(savedir.joinpath(f"{pname}.png"))
        plt.clf()
        plt.close()
    return

# TODO: move to top
from os import getcwd
from generate_input import modify_file
def simlation_for_flowplot(sumpth: PathLike,
                           simdir: PathLike,
                           add_props=["FLUXI#E",
                                      "FLUXJ#E",
                                      "FLUXK#E",
                                      "FLUXI#T",
                                      "FLUXJ#T",
                                      "FLUXK#T"]) -> Path:
    # calculate flow field
    sumpth = Path(sumpth)
    simdir = Path(simdir)
    condition = dir_to_condition(simdir.parent.parent)
    props = ["TEMPC", "PRES", "COMP1T"]
    if condition.get("permf_cap", None):
        props.append("TRANFRMT")

    values = load_snap(sumpth, props)

    makedirs(simdir, exist_ok=True)
    refpth = sumpth.parent.joinpath("tmp.RUN")
    tranfrmt_ls = None
    if condition.get("permf_cap", None):
        tranfrmt_ls = values[3][1]
    runpth = simdir.joinpath("tmp.RUN")
    nxyz = len(DXYZ[0])*len(DXYZ[1])*len(DXYZ[2])
    modify_file(refpth,
                runpth,
                values[0][1][:nxyz],
                values[1][1][:nxyz],
                values[2][1][:nxyz],
                tranfrmt_ls,
                tend=0.0,
                add_props=add_props
                )
    # run simulation
    exepth = Path(getcwd()).joinpath("H64.EXE")
    logpth = simdir.joinpath("log.txt")
    print(f"RUN: {runpth}")
    with open(logpth, "w") as outfile:
        outfile.write("")
        p = Popen(f"{exepth} {runpth}", stdout=outfile)
    p.wait()
    outpth = simdir.joinpath("tmp.0000.SUM")
    return outpth

def plt_stream_line(sumpth: PathLike,
                    flux_props: List[str]=["FLUXI#T",
                                           "FLUXJ#T",
                                           "FLUXK#T",
                                           ]) -> None:
    # 1. 質量フラックス・エネルギーフラックスをsumファイルを読み込み出力
    # 2. 質量フラックス・エネルギーフラックスを温度場上に保存する
    assert len(flux_props)==3
    sumpth = Path(sumpth)
    cond_dir = sumpth.parent
    while "ITER" in cond_dir.name:
        cond_dir = cond_dir.parent
    simdir = cond_dir.joinpath("flow").joinpath(str(get_sumpth_time(sumpth)))
    flux_ls = ["FLUXI#T",
               "FLUXJ#T",
               "FLUXK#T",
               "FLUXI#E",
               "FLUXJ#E",
               "FLUXK#E",]
    outpth = simlation_for_flowplot(sumpth, simdir, add_props=flux_ls)
    values = load_snap(outpth, flux_props)
    nxyz = NX*NY*NZ
    total_flux = np.log10(np.sqrt(np.square(np.array(values[flux_ls.index(flux_props[0])][1])) \
                                + np.square(np.array(values[flux_ls.index(flux_props[1])][1])) \
                                + np.square(np.array(values[flux_ls.index(flux_props[2])][1]))))
    plot_values(total_flux.tolist()[:nxyz],
                "Log Total Mass Flux (t/day)",
                simdir,
                axis="Y",
                vmin=-4.0,
                vmax=4.0,
                indexes=[20],
                flux=(values[0][1][:nxyz],
                      values[1][1][:nxyz],
                      values[2][1][:nxyz])
                )

# TODO:
def temperature_speed():
    return

def liq_sat_at500(sumpth: PathLike,) -> Dict[int, float]:
    sumpth = Path(sumpth)
    IDX_SUBSURF = set([IDX_LAND, IDX_VENT, IDX_CAP, IDX_CAPVENT])
    _, sg_ls = load_snap(sumpth, ["SAT#GAS"])[0]
    cache_topo = CACHE_DIR.joinpath("topo_ls")
    assert cache_topo.exists()
    with open(cache_topo, "rb") as pkf:
        topo_ls, _ = pickle.load(pkf)
    z_ls = stack_from_0(DXYZ[2])
    elv_ls = [ORIGIN[2]-z for z in z_ls]
    k500 = np.argmin(np.square(np.array(elv_ls)-500.0))

    sg500_dct: Dict[int, float] = {}
    for i in range(NX):
        for j in range(int(NY*0.5), NY):
            m = calc_m(i,j,k500,NX,NY)
            if topo_ls[m] not in IDX_SUBSURF:
                continue
            if i < NX-1:
                mtmp = calc_m(i+1,j,k500,NX,NY)
                if topo_ls[mtmp] not in IDX_SUBSURF:
                    sg500_dct.setdefault(m, sg_ls[m])
                    continue
            if i > 0:
                mtmp = calc_m(i-1,j,k500,NX,NY)
                if topo_ls[mtmp] not in IDX_SUBSURF:
                    sg500_dct.setdefault(m, sg_ls[m])
                    continue
            if j < NY-1:
                mtmp = calc_m(i,j+1,k500,NX,NY)
                if topo_ls[mtmp] not in IDX_SUBSURF:
                    sg500_dct.setdefault(m, sg_ls[m])
                    continue
            if j > 0:
                mtmp = calc_m(i,j-1,k500,NX,NY)
                if topo_ls[mtmp] not in IDX_SUBSURF:
                    sg500_dct.setdefault(m, sg_ls[m])
                    continue
    sg500_ls = [sg500_dct[m] for m in sg500_dct]

    with open(sumpth.parent.joinpath("liq_sat_500m.txt"), "w") as f:
        f.write(f"ave: {1.0-mean(sg500_ls)}\n")
        f.write(f"min: {1.0-min(sg500_ls)}\n")
        f.write(f"max: {1.0-max(sg500_ls)}\n")

    return sg500_dct



from utils import calc_m, calc_press_air
from constants import IDX_AIR, IDX_LAND, IDX_VENT, DXYZ

# TODO: 温度の上昇速度計算(定量的な基準で)
# → TODO: 変化点が分かれば、そこを1999/1/1としてプロット（論文用にきれいに直す）
# TODO: 流速ベクトルの計算・可視化（とりあえず準定常状態・キャップロックの浸透率を動的に変化させた場合か）
# TODO: 地盤膨張・全磁力計算結果と観測データの比較（平均などで比較したほうがよいか）

if __name__ == "__main__":
    # plt_stream_line(r"F:\tarumai2\900.0_0.0_1000.0_10.0_1.0_v\unrest\900.0_0.0_15000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7\tmp.0017.SUM",)
    # check_convergence(r"E:\tarumai4")
    # cellid_props, srcid_props, time = load_sum(r"E:\tarumai\200.0_0.0_100.0_10.0\tmp.0000.SUM")
    # for i, (_, prop) in enumerate(cellid_props.items()):
    #     if i == 0:
    #         print(prop)
    #     if isnan(prop["PRES"]):
    #         print(i)

    # dirpth = r"E:\tarumai2\900.0_0.1_10000.0_10.0_1.0_v\unrest\900.0_0.1_35000.0_10.0_1.0_v_d_dyn10000000.0_brit"
    # # fpth = dirpth + r"\tmp.0064.SUM"    
    # print(progress_time(dirpth) / 365.25)
    
    # # plot_results(
    # #     pth, ("Y"), False, None, None, ["FLUXK#E",]
    # # )
    # plot_results(
    #     fpth,
    #     ("Y"),
    #     False,
    #     10.0,
    #     500.0,
    #     [
    #         "TEMPC",
    #     ],
    # )
    # plot_results(
    #     fpth,
    #     ("Y"),
    #     False,
    #     0.0,
    #     1.0,
    #     [
    #         "SAT#GAS",
    #     ],
    # )
    # plot_results(
    #     fpth,
    #     ("Y"),
    #     False,
    #     0.0,
    #     10.0,
    #     [
    #         "PRES",
    #     ],
    # )
    
    # get_latest_fumarole_prop(dirpth, "TEMPC")
    # get_latest_fumarole_prop(dirpth, "FLUXK#E")

    # pth = r"E:\tarumai2\900.0_0.1_10000.0_10.0_1.0_v\unrest\900.0_0.1_15000.0_10.0_100000.0_v\ITER_3"
    # pth = r"E:\tarumai2\900.0_0.0_100.0_10000.0_v"
    # get_latest_fumarole_prop(pth, "TEMPC")
    # check_convergence_single(pth)
    # # get_latest_fumarole_prop(pth, "FLUXK#E")
    # # load_results_and_plt_conv(pth)
    
    dirpth = r"F:\tarumai2\900.0_0.1_10000.0_10000.0_1.0_v\unrest\900.0_0.1_30000.0_10000.0_1.0_v_d_dyn100000.0_brit_pf2.7"
    print(progress_time(dirpth) / 365.25)
    # plot_sum_foreach_tstep(dirpth,
    #                        ("Y",),
    #                        ["TEMPC",
    #                         "SAT#GAS",
    #                         "PRES",
    #                         "COMP1T",
    #                         "DENT",
    #                         "PRESFDYN",
    #                         "PFLDFACT",
    #                         "TRANFRMT",
    #                         ], 
    #                         ([20,],), False, 
    #                         ((0.0, 500.0), 
    #                          (0.0, 1.0), 
    #                          (0.0, 15.0),
    #                          (0.0, 0.1),
    #                          (0.0, 1500.0),
    #                          (0.0, 30.0), (-1.0, 1.0),(0.0, 0.4),
    #                          ))
    # plot_sum_foreach_tstep(dirpth, ("Y",), ["TEMPC", "SAT#GAS", "PRES",], ([20,],), False, ((-100.0, 100.0), (-1.0, 1.0), (-10.0, 10.0),), diff=True)
    # plot_fumarole_props_foreach_tstep(dirpth)
    # plt_regional_timeseries(dirpth, ["surface", "conduit", "capvent", "aquifer"], ["TEMPC", "SAT#GAS", "PRES", "COMP1T", "DENT", "TRANFRMT","PFLDFACT",])
    # plt_regional_timeseries(dirpth, ["surface", "conduit", "capvent", "aquifer"], ["TEMPC", "SAT#GAS", "PRES", "COMP1T", "DENT",])
    # img2mov(dirpth + r"\tstep\PRES\Y",)
    # img2mov(dirpth + r"\tstep\SAT#GAS\Y",)
    # img2mov(dirpth + r"\tstep\TEMPC\Y",)
    # img2mov(dirpth + r"\tstep\COMP1T\Y",)
    # plt_warning_tstep(dirpth)
    
    # time_variation(r"F:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_25000.0_10.0_100000.0_v_d",
    #                ["PRES"])

    # dirpth_ls = [r"E:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_15000.0_10.0_100000.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_20000.0_10.0_100000.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_25000.0_10.0_100000.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_30000.0_10.0_100000.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_35000.0_10.0_100000.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest\900.0_0.0_15000.0_10.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest\900.0_0.0_20000.0_10.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest\900.0_0.0_25000.0_10.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest\900.0_0.0_30000.0_10.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest\900.0_0.0_35000.0_10.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_15000.0_10.0_100000.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_20000.0_10.0_100000.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_25000.0_10.0_100000.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_30000.0_10.0_100000.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_35000.0_10.0_100000.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_10000.0_10.0_v\unrest\900.0_0.0_15000.0_10.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_10000.0_10.0_v\unrest\900.0_0.0_20000.0_10.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_10000.0_10.0_v\unrest\900.0_0.0_25000.0_10.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_10000.0_10.0_v\unrest\900.0_0.0_30000.0_10.0_v_d",
    #              r"E:\tarumai2\900.0_0.0_10000.0_10.0_v\unrest\900.0_0.0_35000.0_10.0_v_d",
    #              ]
    # for dirpth in dirpth_ls:
    #     print(dirpth)
    #     plot_sum_foreach_tstep(dirpth,
    #                        ("Y",),
    #                        ["TEMPC",
    #                         "SAT#GAS",
    #                         "PRES",
    #                         "COMP1T",
    #                         ], 
    #                         ([20,],), False, 
    #                         ((0.0, 500.0), 
    #                          (0.0, 1.0), 
    #                          (0.0, 15.0),
    #                          (0.0, 0.1),
    #                          ))

    # # dirpth += ""
    # plot_sum_foreach_tstep(dirpth, ("Y",), ["FLUXK#E",], ([20,],), False, )
    # img2mov(dirpth + r"\tstep\PRES\Y",)
    # img2mov(dirpth + r"\tstep\TEMPC\Y",)
    # img2mov(dirpth + r"\tstep\COMP1T\Y",)
    # img2mov(dirpth + r"\tstep\SAT#GAS\Y",)
    # img2mov(dirpth + r"\tstep\PFLDFACT\Y",)
    # img2mov(dirpth + r"\tstep\FLUXK#E\Y")
    # # print(progress_time(dirpth) / 365.25)
    # # get_latest_fumarole_prop(dirpth, "TEMPC")
    # plt_warning_tstep(dirpth)

    # kill(37348, 15)
    pass
