"""Load .SUM file and monitor processes"""
from os import PathLike, access, R_OK, path, makedirs, kill, getcwd
import struct
from typing import List, Tuple, Dict, BinaryIO, Any, OrderedDict, Union, Optional, Literal
from pathlib import Path
from math import isclose, isnan, exp, log10
from time import sleep, time
from logging import Logger
import re
from statistics import median, mean

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
    generate_simple_cap
)

ENCODING = "windows-1251"
NX, NY, NZ = len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2])

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


def read_Array(f: BinaryIO) -> Tuple[float, List]:
    props_ls: List = []
    b = f.read(8)
    b = f.read(8)
    # Record length
    b = f.read(8)
    # Number of properties
    b = f.read(4)
    np: int = struct.unpack("i", b)[0]
    # Number of objects
    b = f.read(4)
    no: int = struct.unpack("i", b)[0]
    # Property description
    for _ in range(np):
        # Tag
        b = f.read(8)
        mnemonic: str = b.decode(encoding=ENCODING)
        b = f.read(8)
        dimension: str = b.decode(encoding=ENCODING)
        b = f.read(8)
        tag_ls = []
        while "ENDITEM" not in b.decode(encoding=ENCODING):
            tag_ls.append(b.decode(encoding=ENCODING))
            b = f.read(8)  # ENDITEM
        props_ls.append((mnemonic, dimension, tag_ls))
    return no, props_ls


def read_DATA(f: BinaryIO, no: int, props_ls: List, cellid_props: Dict) -> Dict:
    for _ in range(no):
        cellid: int = None
        for prop in props_ls:
            prop_name, _, tag_ls = prop
            v = None
            flag_int1, flag_int2, flag_int4, flag_char8 = False, False, False, False
            for _s in tag_ls:
                if "INT1" in _s:
                    flag_int1 = True
                    continue
                if "INT2" in _s:
                    flag_int2 = True
                    continue
                if "INT4" in _s:
                    flag_int4 = True
                    continue
                if "CHAR8" in _s:
                    flag_char8 = True
            if flag_int1:
                b = f.read(1)
                v = struct.unpack("b", b)[0]
            elif flag_int2:
                b = f.read(2)
                v = struct.unpack("h", b)[0]
            elif flag_int4:
                b = f.read(4)
                v = struct.unpack("i", b)[0]
            elif flag_char8:
                b = f.read(8)
                v = b.decode(encoding=ENCODING)
            else:
                b = f.read(8)
                v = struct.unpack("d", b)[0]
            if "CELLID" in prop_name:
                # set id
                cellid = v
            elif "SRCNAME" in prop_name:
                # set id
                cellid = v
            else:
                # Set cellid_props
                _props: Dict = cellid_props.setdefault(cellid, {})
                prop_name = prop_name.replace(" ", "")
                _props.setdefault(prop_name, v)
    # ENDDATA
    b = f.read(8)
    b = f.read(8)  # 0
    return cellid_props


def dump_cache(prop: Tuple[float, List[float]], cache_pth: PathLike, update=False) -> None:
    cache_pth = Path(cache_pth)
    if cache_pth.exists() and not update:
        return
    makedirs(cache_pth.parent, exist_ok=True)
    with open(cache_pth, "wb") as pkf:
        pickle.dump(prop, pkf, pickle.HIGHEST_PROTOCOL)

def load_cache(cache_file: PathLike) -> Tuple[float, List[float]]:
    with open(cache_file, "rb") as pkf:
        data: Tuple[float, List[float]] = pickle.load(pkf)
    return data

def load_sum(fpth: PathLike, only_time=False) -> Tuple[Dict, Dict, float]:
    fpth = Path(fpth)
    with open(fpth, "rb") as f:
        cellid_props: Dict = {}
        srcid_props: Dict = {}  # not load for now
        time: float = None
        while f.readable():
            # get the name
            b = f.read(8)
            name = b.decode(encoding=ENCODING)
            if name in "BINARY":
                f.read(8)
                continue
            if name in "HMDSPEC":
                f.read(8)
                continue
            # Record TIME
            if name == "TIME    ":
                _ = f.read(8)  # 16 (int)
                # time value
                b = f.read(8)
                time = struct.unpack("d", b)[0]
                if only_time:
                    return time
                b = f.read(8)
                continue
            # Block CELLDATA
            # contains "ARRAYS" and "DATA"
            if name == "CELLDATA":
                # ARRAYS
                no, props_ls = read_Array(f)
                # DATA
                b = f.read(8)
                # Record length
                b = f.read(8)
                read_DATA(f, no, props_ls, cellid_props)
                break
            # Block SRCDATA
            # contains "ARRAYS" and "DATA"
            if "SRCDATA" in name:
                # ARRAYS
                no, props_ls = read_Array(f)
                # DATA
                b = f.read(8)
                # Record length
                b = f.read(8)
                read_DATA(f, no, props_ls, srcid_props)
                break
            if "ENDFILE" in name:
                break
    return cellid_props, srcid_props, time


def load_snap(sumpth: PathLike, prop_names: List[str]) -> List[Tuple[float, List[float]]]:
    sumpth = Path(sumpth)
    cache_dir_base = sumpth.parent.joinpath("cache")
    exist = True
    cache_file_ls: List[Path] = []
    for prop_name in prop_names:
        cache_file = cache_dir_base.joinpath(prop_name).joinpath(sumpth.stem)
        exist *= cache_file.exists()
        cache_file_ls.append(cache_file)
    props: List[Tuple[float, List[float]]] = []
    if exist:
        for cache_file in cache_file_ls:
            props.append(load_cache(cache_file))
    else:
        cellid_props, _, time = load_sum(sumpth)
        for prop_name, cache_file in zip(prop_names, cache_file_ls):
            v_ls = get_v_ls(cellid_props, prop_name)
            prop = (time, v_ls)
            dump_cache(prop, cache_file)
            props.append(prop)
    return props

def get_v_ls(props: Dict, prop_name: str) -> List[float]:
    v_ls: List[float] = list(range(len(props)))
    for i, (_, prop) in enumerate(props.items()):
        v = prop[prop_name]
        assert isinstance(v, float)
        if isnan(v):
            v = 0.0
        v_ls[i] = v
    return v_ls


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


def generate_3darr(v_ls, axis) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        grid_x, grid_y = np.meshgrid(
            np.array(stack_from_center(DXYZ[1])),
            ORIGIN[2] - np.array(stack_from_0(DXYZ[2])),
        )
    if axis == "y":
        val_3d = np.transpose(val_3d, (1, 0, 2))
        val_3d = np.flip(val_3d, 0)
        # val_3d = np.flip(val_3d, 1)
        grid_x, grid_y = np.meshgrid(
            np.array(stack_from_center(DXYZ[0])),
            ORIGIN[2] - np.array(stack_from_0(DXYZ[2])),
        )
    if axis == "z":
        val_3d = np.flip(val_3d, 0)
        val_3d = np.flip(val_3d, 1)
        grid_x, grid_y = np.meshgrid(
            np.array(stack_from_center(DXYZ[0])), np.array(stack_from_center(DXYZ[1]))
        )
    return grid_x, grid_y, val_3d


def plt_single_cs(grid_x, grid_y, val_3d, idx, prop_name, show_time, vmin, vmax, fpth):
    val2d = val_3d[idx]
    fig, ax = plt.subplots()
    mappable = ax.pcolormesh(
        grid_x,
        grid_y,
        val2d,
        vmin=vmin,
        vmax=vmax,
    )
    pp = fig.colorbar(mappable, ax=ax, orientation="vertical")
    pp.set_label(prop_name)
    ax.set_aspect("equal")
    plt.tick_params(labelsize=8)
    if show_time:
        ax.set_title("{:.3f}".format(time))
    fig.savefig(fpth, dpi=200, bbox_inches="tight")
    plt.clf()
    plt.close()


def plot_values(
    v_ls: List[float],
    prop_name: str,
    savedir: PathLike,
    axis="Y",
    show_time: bool = True,
    vmin: float = None,
    vmax: float = None,
    indexes: Tuple[int] = None,
) -> None:
    v_ls = v_ls[: NX * NY * NZ]
    grid_x, grid_y, val_3d = generate_3darr(v_ls, axis)
    if indexes is None:
        indexes = list(range(len(val_3d)))
    dirpth = Path(savedir)
    makedirs(dirpth, exist_ok=True)
    for i in range(len(val_3d)):
        if indexes is not None:
            if i not in indexes:
                continue
        fpth = dirpth.joinpath(f"{i}.png")
        plt_single_cs(grid_x, grid_y, val_3d, i, prop_name, show_time, vmin, vmax, fpth)


def plot_results(
    fpth: PathLike,
    axis: Tuple[str] = ("X", "Y", "Z"),
    show_time: bool = True,
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
                show_time=show_time,
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
    cond_dir: PathLike, prop_name: str = "TEMPC", calc_average: bool = True
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

def get_fpth_in_timeseries(simdir: PathLike, ignore_first: bool = False) -> List[Path]:
    def __get_fpth_in_singledir(__dir) -> List[Path]:
        __dir = Path(__dir)
        __fpth_ls = []
        for i in range(10000):
            if ignore_first and i == 0:
                continue
            fn = str(i).zfill(4)
            fpth = __dir.joinpath(f"tmp.{fn}.SUM")
            if fpth.exists():
                __fpth_ls.append(fpth)
            else:
                break
        return __fpth_ls

    simdir = Path(simdir)
    fpth_ls: List = __get_fpth_in_singledir(simdir)

    for i in range(1, 10000):
        _dirpth = simdir.joinpath(f"ITER_{i}")
        if _dirpth.exists():
            fpth_ls.extend(__get_fpth_in_singledir(_dirpth))
        else:
            break
    
    return fpth_ls


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
                    plt_single_cs(grid_x, grid_y, val_3d, idx, prop_name, showtime, vmin, vmax, fpth)


def get_obs_props(prop_names: List[str]) -> Dict:
    # lim, 物性値のキー, 火口のキー
    # suffix to mufits key in obsdata dir
    suffix_mkey = {"tempe": "TEMPC",
                   "height": "FLUXK#E"}
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
        "TEMPC", "FLUXK#E"
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
    fls = ("1819", "A", "B")
    name_color: Dict = {}
    name_in_obs = []
    for prop_name, dct in obs_props.items():
        name_in_obs.extend(list(dct.keys()))
    name_in_obs = list(set(name_in_obs))
    names = list(set([*name_in_obs, *fls])) # TODO: setで色
    for i, name in enumerate(names):
        name_color.setdefault(name, cmap(i/len(names)))
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
                ax2.scatter(values[0], values[1], label=f"Obs: {name}", color=name_color[name])
            else:
                ax.scatter(values[0], values[1], label=f"Obs: {name}", color=name_color[name])
            t_tmp.extend(values[0]) # time
        # simulation results
        strat = 1990.0 #!
        props = sim_dct[prop_name]
        for i, name in enumerate(fls):
            if isflux:
                name = "Sum"
                name_color.setdefault(name, cmap(0))
            time_ls = [t / 365.25 + strat for t in props["time"]]
            ax.plot(time_ls, props["name_v"][name], label=f"Sim: {name}", color=name_color[name])
            # before start
            tls = np.linspace(min(t_tmp), strat, 200)
            vls = np.full(tls.shape, props["name_v"][name][0])
            ax.plot(tls, vls, color=name_color[name], linestyle="dashed")
            if isflux:
                break
        # TODO: 目盛りとグラフとの間隔をあける
        labelsize = 14
        ax.set_xlim(min(tls) - 3.0, max((max(tls) + 30, max(time_ls))) + 3.0)
        ax.set_xlabel("Year", fontsize=labelsize)
        ax.set_ylabel(prop_name, fontsize=labelsize)
        if isflux:
            ax2.set_ylabel("Plume Height (m)", fontsize=labelsize)
        ax.legend()
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

from utils import calc_m, calc_press_air
from constants import IDX_AIR, IDX_LAND, IDX_VENT, DXYZ

if __name__ == "__main__":
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
    
    dirpth = r"E:\tarumai2\900.0_0.0_1000.0_10.0_1.0_v\unrest\900.0_0.0_35000.0_10.0_1.0_v_d_dyn100000.0_ibrit_pf2.7"
    print(progress_time(dirpth) / 365.25)
    # plot_sum_foreach_tstep(dirpth, 
    #                        ("Y",), 
    #                        ["TEMPC", 
    #                         "SAT#GAS", 
    #                         "PRES", 
    #                         "PRESFDYN", 
    #                         "PFLDFACT",
    #                         "TRANFRMT",
    #                         ], 
    #                         ([20,],), False, 
    #                         ((0.0, 500.0), 
    #                          (0.0, 1.0), 
    #                          (0.0, 15.0), 
    #                          (0.0, 30.0), (-1.0, 1.0),(0.0, 0.4),
    #                          ))
    # plot_sum_foreach_tstep(dirpth, ("Y",), ["TEMPC", "SAT#GAS", "PRES",], ([20,],), False, ((-100.0, 100.0), (-1.0, 1.0), (-10.0, 10.0),), diff=True)
    # plot_fumarole_props_foreach_tstep(dirpth)
    # plt_regional_timeseries(dirpth, ["surface", "conduit", "capvent", "aquifer"], ["TEMPC", "SAT#GAS", "PRES", "COMP1T","TRANFRMT","PFLDFACT",])
    # dirpth += ""
    # plot_sum_foreach_tstep(dirpth, ("Y",), ["FLUXK#E",], ([20,],), False, )
    # img2mov(dirpth + r"\tstep\PRES\Y\diff",)
    # img2mov(dirpth + r"\tstep\TEMPC\Y\diff",)
    # img2mov(dirpth + r"\tstep\SAT#GAS\Y",)
    # img2mov(dirpth + r"\tstep\PFLDFACT\Y",)
    # img2mov(dirpth + r"\tstep\FLUXK#E\Y")
    # print(progress_time(dirpth) / 365.25)
    # get_latest_fumarole_prop(dirpth, "TEMPC")
    plt_warning_tstep(dirpth)

    # dir_ls = [r"E:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_20000.0_10.0_100000.0_v_d",
    #  r"E:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_25000.0_10.0_100000.0_v_d",
    #  r"E:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_30000.0_10.0_100000.0_v_d",
    #  r"E:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_35000.0_10.0_100000.0_v_d",
    #  r"E:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest\900.0_0.0_35000.0_10.0_v_d",
    #  r"E:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_20000.0_10.0_100000.0_v_d",
    #  r"E:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_25000.0_10.0_100000.0_v_d",
    #  r"E:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_30000.0_10.0_100000.0_v_d",
    #  r"E:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_35000.0_10.0_100000.0_v_d",
    #  r"E:\tarumai2\900.0_0.1_1000.0_10.0_100000.0_v\unrest\900.0_0.1_20000.0_10.0_100000.0_v_d",
    #  r"E:\tarumai2\900.0_0.1_1000.0_10.0_100000.0_v\unrest\900.0_0.1_25000.0_10.0_100000.0_v_d",
    #  r"E:\tarumai2\900.0_0.1_1000.0_10.0_100000.0_v\unrest\900.0_0.1_30000.0_10.0_100000.0_v_d",
    #  r"E:\tarumai2\900.0_0.1_1000.0_10.0_100000.0_v\unrest\900.0_0.1_35000.0_10.0_100000.0_v_d",
    #  r"E:\tarumai2\900.0_0.1_1000.0_10.0_v\unrest\900.0_0.1_20000.0_10.0_v_d",
    #  r"E:\tarumai2\900.0_0.1_1000.0_10.0_v\unrest\900.0_0.1_25000.0_10.0_v_d",
    #  r"E:\tarumai2\900.0_0.1_1000.0_10.0_v\unrest\900.0_0.1_30000.0_10.0_v_d",
    #  r"E:\tarumai2\900.0_0.1_1000.0_10.0_v\unrest\900.0_0.1_35000.0_10.0_v_d",
    #  ]
    # for dirpth in dir_ls:
    #     print(dirpth)
    #     # plot_fumarole_props_foreach_tstep(dirpth)
    #     plt_regional_timeseries(dirpth, ["surface", "conduit"], ["TEMPC", "SAT#GAS", "PRES", "COMP1T"])
    # #     img2mov(dirpth + r"\tstep\TEMPC\Y",)
    # #     img2mov(dirpth + r"\tstep\SAT#GAS\Y",)

    # pth = r"E:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_20000.0_10.0_100000.0_v_d"
    # plot_sum_foreach_tstep(pth, ("Y",), ["TEMPC", "SAT#GAS"], ([20,],), False, ((0.0, 500.0), (0.0, 1.0)))
    # plot_fumarole_props_foreach_tstep(pth)

    # unrestの進捗の可視化
    # _ls = [r"E:\tarumai2\900.0_0.1_10000.0_10.0_1.0_v\unrest\900.0_0.1_15000.0_10.0_100000.0_v",
    #        r"E:\tarumai2\900.0_0.1_1000.0_10.0_100000.0_v\unrest\900.0_0.1_15000.0_10.0_100000.0_v",
    #        r"E:\tarumai2\900.0_0.1_1000.0_10.0_v\unrest\900.0_0.1_15000.0_10.0_v",
    #        r"E:\tarumai2\900.0_0.0_10000.0_10.0_1.0_v\unrest\900.0_0.0_15000.0_10.0_100000.0_v",
    #        r"E:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_15000.0_10.0_100000.0_v",
    #        r"E:\tarumai2\900.0_0.0_1000.0_10.0_1.0_v\unrest\900.0_0.0_15000.0_10.0_100000.0_v",
    #        r"E:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_15000.0_10.0_100000.0_v",
    #        r"E:\tarumai2\900.0_0.0_10000.0_10.0_v\unrest\900.0_0.0_15000.0_10.0_v",
    #        r"E:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest\900.0_0.0_15000.0_10.0_v"]
    # for pth in _ls:
    #     print(pth)
    #     # pth = r"E:\tarumai2\900.0_0.1_1000.0_10.0_100000.0_v\unrest\900.0_0.1_15000.0_10.0_100000.0_v\ITER_1"
    #     # TODO: plotがすでにある場合には途中から
    #     # plot_sum_foreach_tstep(pth, ("Y",), ["TEMPC", "SAT#GAS"], ([20,],), False, ((0.0, 500.0), (0.0, 1.0)))
    #     # plot_fumarole_props_foreach_tstep(pth)
    #     plot_sum_foreach_tstep(pth, ("Y",), ["TEMPC", "SAT#GAS", "PRES"], ([20,],), False, ((-100.0, 100.0), (-1.0, 1.0), (-5.0, 5.0)), True)
    #     # plt_progress_rate(pth)
    
    # sanity_check(r"E:\tarumai2\900.0_0.1_10000.0_10.0_1.0_v\unrest\900.0_0.1_15000.0_10.0_100000.0_v\ITER_2\tmp.0019.SUM")

    # img2mov(r"E:\tarumai2\900.0_0.0_1000.0_10.0_1.0_v\unrest\900.0_0.0_15000.0_10.0_100000.0_v\tstep\SAT#GAS\Y\diff",)

    # target_ls = (
    #     r"E:\tarumai4\900.0_0.001_1000.0_10.0",
    #     r"E:\tarumai4\900.0_0.001_1000.0_100.0",
    #     r"E:\tarumai4\900.0_0.001_1000.0_1000.0",
    #     r"E:\tarumai4\900.0_0.001_1000.0_10000.0",
    #     r"E:\tarumai4\900.0_0.001_10000.0_10.0",
    #     r"E:\tarumai4\900.0_0.001_10000.0_100.0",
    #     r"E:\tarumai4\900.0_0.001_10000.0_1000.0",
    #     r"E:\tarumai4\900.0_0.001_10000.0_10000.0",
    #     r"E:\tarumai4\900.0_0.01_100.0_10.0",
    #     r"E:\tarumai4\900.0_0.01_100.0_100.0",
    #     r"E:\tarumai4\900.0_0.01_100.0_1000.0",
    #     r"E:\tarumai4\900.0_0.01_100.0_10000.0",
    # )
    # target_ls = [i for i in Path(r"E:\tarumai4").iterdir()]
    # for fpth in reversed(target_ls):
    #     print(fpth)
    #     plt_latests(fpth, show_time=False, vmin=10.0, vmax=100.0, prop_ls=["TEMPC"])

    # 900.0_0.01_1000.0_1000.0から
    # props_ls: List = load_props_ls(0, Path(r"E:\tarumai4\700.0_0.0_10000.0_1000.0"))
    # for cou, (metric, criteria) in enumerate(CONVERSION_CRITERIA.items()):
    #     time_ls, changerate_ls = calc_change_rate(props_ls, metric)
    #     plt_conv(time_ls, changerate_ls, rf"E:\tarumai4\700.0_0.0_10000.0_1000.0\tmp\{metric}.png")
    # print(calc_ijk(2142, 40, 40))

    # load_results_and_plt_conv(r"E:\tarumai\200.0_0.1_10000.0_10000.0_1.0")
    # kill(32324, 15)
    # load_results_and_plt_conv(r"E:\tarumai_tmp11\900.0_0.1_10000.0_10000.0")

    # TODO: 等方的な浸透率でもう一度 E:\tarumai4\200.0_0.1_100.0_1000.0

    # _, _, time = load_sum(r"E:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest\900.0_0.0_15000.0_10.0_v\tmp.0363.SUM")
    # print(time)

    pass
