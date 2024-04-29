"""Load .SUM file and monitor processes"""
from os import PathLike, access, R_OK, path, makedirs, kill
import struct
from typing import List, Tuple, Dict, BinaryIO, Any, OrderedDict, Union
from pathlib import Path
from math import isclose, isnan, exp, log10
from time import sleep, time
from logging import Logger
import re
from statistics import median, mean

import numpy as np
from pyproj import Transformer
from matplotlib import pyplot as plt
import cv2
import pickle


from constants import (
    CONVERSION_CRITERIA,
    DXYZ,
    ORIGIN,
    CACHE_DIR,
    IDX_AIR,
    IDX_CAP,
    IDX_CAPVENT,
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
)

ENCODING = "windows-1251"
NX, NY, NZ = len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2])


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


def load_sum(fpth: PathLike) -> Tuple[Dict, Dict, float]:
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
        print(fpth)
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


def generate_3darr(v_ls, axis) -> np.ndarray:
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


def plot_sumfile(
    fpth: PathLike,
    prop_name: str,
    savedir: PathLike,
    use_cache: bool = True,
    axis="Y",
    show_time: bool = True,
    vmin: float = None,
    vmax: float = None,
    indexes: Tuple[int] = None,
) -> None:

    fpth = Path(fpth)
    cachepth = fpth.parent.joinpath(f"{fpth.stem}.pkl")
    # load cellid_props
    cellid_props: Dict = None
    if use_cache and cachepth.exists():
        with open(cachepth, "rb") as pkf:
            cellid_props = pickle.load(pkf)
    else:
        cellid_props, _, time = load_sum(fpth)
    v_ls = get_v_ls(cellid_props, prop_name)
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
    fpth,
    axis: Tuple[str] = ("X", "Y", "Z"),
    show_time: bool = True,
    vmin: float = None,
    vmax: float = None,
    prop_ls: List[str] = list(CONVERSION_CRITERIA.keys()),
) -> None:
    fpth = Path(fpth)
    for prop_name in prop_ls:
        for ax in axis:
            savedir = fpth.parent.joinpath(prop_name).joinpath(ax)
            makedirs(savedir, exist_ok=True)
            plot_sumfile(
                fpth,
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
    plt.show()
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
        props_ls = [load_sum(sumpth_ls[1]), load_sum(sumpth_ls[0])]
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
        props_ls.append(load_sum(sumpth))
    # calc chagne rate
    props_change_rate_ls: Dict = {}
    for prop_name in CONVERSION_CRITERIA:
        time_ls, diff_ls = calc_change_rate(props_ls, prop_name)
        plt_conv(time_ls, diff_ls, dirpth.joinpath("tmp").joinpath(f"{prop_name}.png"))

def get_fumarole_prop(fpth: PathLike, prop_name: str, calc_average: bool = True, sumprops: Tuple = None):
    if sumprops is not None:
        cellid_props, _, time = sumprops
    else:
        cellid_props, _, time = load_sum(fpth)
    v_ls = get_v_ls(cellid_props, prop_name)
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
    props: Dict = get_fumarole_prop(fpth_ls[-1], prop_name, calc_average)
    with open(cond_dir.joinpath(f"fumarole_{prop_name}.txt"), "w") as f:
        for name, v in props.items():
            f.write(f"{name}: {v}\n")

def get_fpth_in_timeseries(simdir, ignore_first: bool = False) -> List[Path]:
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
    minmax: Tuple = ((20.0, 500.0),),
    diff: bool = False,
):
    assert len(axes) == len(idx_ls)
    assert len(prop_names) == len(minmax)
    simdir = Path(simdir)
    fpth_ls = get_fpth_in_timeseries(simdir)
    nx, ny, nz = len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2])
    nxyz = nx * ny * nz
    time, time0 = 0.0, 0.0
    itern0: str = None
    v0_dct: Dict = {}
    for i, fpth in enumerate(fpth_ls):
        cellid_props, _, timetmp = load_sum(fpth)
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
            v_ls = get_v_ls(cellid_props, prop_name)[:nxyz]
            if diff:
                if i == 0:
                    v0_dct.setdefault(prop_name, v_ls)
                if v0_dct.get(prop_name, None) is not None:
                    v_ls = [v1 - v0 for v1, v0 in zip(v_ls, v0_dct.get(prop_name))]
            for k, ax in enumerate(axes):
                grid_x, grid_y, val_3d = generate_3darr(v_ls, ax)
                time_dir = simdir.joinpath("tstep").joinpath(prop_name).joinpath(ax)
                if diff:
                    time_dir = time_dir.joinpath("diff")
                makedirs(time_dir, exist_ok=True)
                for idx in idx_ls[k]:
                    fpth = time_dir.joinpath(f"{time}_{idx}.png")
                    if fpth.exists():
                        continue
                    plt_single_cs(grid_x, grid_y, val_3d, idx, prop_name, showtime, minmax[j][0], minmax[j][1], fpth)


def plot_fumarole_props_foreach_tstep(
    simdir: PathLike,
    prop_names: List[str] = [
        "TEMPC", "FLUXK#E"
    ],
    calc_average: bool = True
):
    simdir = Path(simdir)
    # fpth_ls = []
    # for i in range(10000):
    #     fn = str(i).zfill(4)
    #     fpth = simdir.joinpath(f"tmp.{fn}.SUM")
    #     if fpth.exists():
    #         fpth_ls.append(fpth)
    fpth_ls = get_fpth_in_timeseries(simdir)

    props: Dict = {}
    time0, time = 0.0, 0.0
    itern0: str = None
    for fpth in fpth_ls:
        cellid_props, srcprops, timetmp = load_sum(fpth)
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
        for prop_name in prop_names:
            _prop = get_fumarole_prop(None, prop_name, calc_average, (cellid_props, srcprops, time))
            props_time.setdefault(prop_name, _prop)
    
    # plot
    savedir = simdir.joinpath("tstep").joinpath("fumarole")
    makedirs(savedir, exist_ok=True)
    for prop_name in prop_names:
        time_ls = []
        name_v: Dict = {}
        for time, _props in props.items():
            time_ls.append(time)
            for name, v in _props[prop_name].items():
                name_v.setdefault(name, []).append(v)
        for name, v_ls in name_v.items():
            fig, ax = plt.subplots()
            ax.plot(time_ls, v_ls)
            figpth = savedir.joinpath(f"{prop_name}_{name}.png")
            fig.savefig(figpth, dpi=200, bbox_inches="tight")
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

    pth = r"E:\tarumai2\900.0_0.0_1000.0_10.0_1.0_v\tmp.0028.SUM"
    # plot_results(
    #     pth, ("Y"), False, None, None, ["FLUXK#E",]
    # )
    # plot_results(
    #     pth,
    #     ("Y"),
    #     False,
    #     10.0,
    #     500.0,
    #     [
    #         "TEMPC",
    #     ],
    # )
    # plot_results(
    #     pth,
    #     ("Y"),
    #     False,
    #     0.0,
    #     1.0,
    #     [
    #         "SAT#GAS",
    #     ],
    # )
    plot_results(
        pth,
        ("Y"),
        False,
        0.0,
        10.0,
        [
            "PRES",
        ],
    )  # ← ここから

    # pth = r"E:\tarumai2\900.0_0.1_10000.0_10.0_1.0_v\unrest\900.0_0.1_15000.0_10.0_100000.0_v\ITER_3"
    # get_latest_fumarole_prop(pth, "TEMPC")
    # # get_latest_fumarole_prop(pth, "FLUXK#E")
    # # load_results_and_plt_conv(pth)
    # plot_sum_foreach_tstep(pth, ("Y",), ["TEMPC", "SAT#GAS"], ([20,],), False, ((0.0, 500.0), (0.0, 1.0)))
    # plot_fumarole_props_foreach_tstep(pth)

    # pth = r"E:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_15000.0_10.0_100000.0_v\ITER_1"
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
    # kill(19676, 15)
    # load_results_and_plt_conv(r"E:\tarumai_tmp11\900.0_0.1_10000.0_10000.0")

    # TODO: 等方的な浸透率でもう一度 E:\tarumai4\200.0_0.1_100.0_1000.0

    # _, _, time = load_sum(r"E:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest\900.0_0.0_15000.0_10.0_v\tmp.0363.SUM")
    # print(time)
    pass
