"""Load .SUM file and monitor processes"""
from os import PathLike, access, R_OK, path, makedirs, kill
import struct
from typing import List, Tuple, Dict, BinaryIO, Any, OrderedDict
from pathlib import Path
from math import isnan
from time import sleep, time
from logging import Logger

import numpy as np
from matplotlib import pyplot as plt
import pickle

from constants import CONVERSION_CRITERIA, DXYZ, CACHE_DIR, IDX_AIR, CONDS_PID_MAP_NAME, OUTDIR
from utils import calc_ijk, stack_from_center, stack_from_0, condition_to_dir

ENCODING = 'windows-1251'
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
            b = f.read(8) # ENDITEM
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
    b = f.read(8) # 0
    return cellid_props

def load_sum(fpth: PathLike) -> Tuple[Dict, Dict, float]:
    with open(fpth, "rb") as f:
        cellid_props: Dict = {}
        srcid_props: Dict = {} # not load for now
        time: float = None
        while f.readable():
            # get the name
            b = f.read(8)
            name= b.decode(encoding=ENCODING)
            if name in "BINARY":
                f.read(8)
                continue
            if name in "HMDSPEC":
                f.read(8)
                continue
            # Record TIME
            if name == "TIME    ":
                _ = f.read(8) # 16 (int)
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
    for i in range(i_start, 1000):
        fn = str(i).zfill(4)
        fpth = dirpth.joinpath(f"tmp.{fn}.SUM")
        if not (fpth.exists() and access(fpth, R_OK)):
            break
        if _is_writting(fpth) or not _is_enough_size(fpth):
            break
        cellprops1, srcprops1, time = load_sum(fpth)
        props_ls.append((cellprops1, srcprops1, time))
    return props_ls

def calc_change_rate(props_ls: List[Tuple[Dict, Dict, float]], prop_name: str) -> Tuple[List, List]:
    time_ls, diff_ls = [], []
    time0: float = None
    cellprops0 = None
    for cellprops1, _, time in props_ls:
        if cellprops0 is None:
            time0 = time
            cellprops0 = cellprops1
            continue
        time_ls.append(time)
        diff_ls.append(calc_prop_diff(cellprops0, cellprops1, prop_name) / (time - time0))
        cellprops0 = cellprops1
        time0 = time
    return time_ls, diff_ls

def plt_conv(time_ls, changerate_ls, fpth: PathLike):
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xlabel("DAYS")
    ax.set_ylabel("Change Rate")
    ax.plot(time_ls, changerate_ls)
    fig.savefig(fpth, bbox_inches="tight", dpi=200)
    plt.clf()
    plt.close()

def monitor_process(conds_dct: Dict[Tuple, Any]) -> None or int:
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
            is_converged = True
            for cou, (metric, criteria) in enumerate(CONVERSION_CRITERIA.items()):
                time_ls, changerate_ls = calc_change_rate(props_ls, metric)
                # Confirm convergent or not
                if changerate_ls[-1] > criteria:
                    is_converged = False
                # extend
                if cou == 0:
                    _extend(status, "time", time_ls)
                    logger.debug("=======================")
                    logger.debug(f"TIME: {time_ls[-1]} DAYS")
                _extend(status, metric, changerate_ls)
                logger.debug(f"{metric}: {changerate_ls[-1]}")
                
                # plot
                plt_conv(status["time"], status[metric], MonitorPth.joinpath(f"{metric}.png"))

            if is_converged:
                _str = str(condition_to_dir(OUTDIR, *conds))
                pid: int = None
                with open(Path(SimDir).joinpath("tmp").joinpath(CONDS_PID_MAP_NAME), "r") as f:
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
    if time() - path.getmtime(fpth) < 120.0:
        return True
    else:
        return False

def _is_enough_size(fpth: PathLike, criteria: int=2000000) -> bool:
    if path.getsize(fpth) > criteria:
        return True
    else:
        False

def plot_sum(fpth: PathLike, prop_name: str, savedir: PathLike, use_cache: bool=True, axis="Y") -> None:
    
    cache_topo = CACHE_DIR.joinpath("topo_ls")
    topo_ls: List[int] = None
    if cache_topo.exists():
        with open(cache_topo, "rb") as pkf:
            topo_ls, _ = pickle.load(
                pkf
            )
    
    fpth = Path(fpth)
    cachepth = fpth.parent.joinpath(f"{fpth.stem}.pkl")
    # load cellid_props
    cellid_props: Dict = None
    if use_cache and cachepth.exists():
        with open(cachepth, "rb") as pkf:
            cellid_props = pickle.load(pkf)
    else:
        cellid_props, _, _ = load_sum(fpth)
    v_ls = get_v_ls(cellid_props, prop_name)
    v_ls = v_ls[:NX * NY * NZ]

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
        val_3d = np.flip(val_3d, 1)
        val_3d = np.flip(val_3d, 2)
        grid_x, grid_y = np.meshgrid(np.array(stack_from_center(DXYZ[1])), np.array(stack_from_0(DXYZ[2])))
    if axis == "y":
        val_3d = np.transpose(val_3d, (1, 0, 2))
        val_3d = np.flip(val_3d, 0)
        val_3d = np.flip(val_3d, 1)
        grid_x, grid_y = np.meshgrid(np.array(stack_from_center(DXYZ[0])), np.array(stack_from_0(DXYZ[2])))
    if axis == "z":
        val_3d = np.flip(val_3d, 0)
        val_3d = np.flip(val_3d, 1)
        grid_x, grid_y = np.meshgrid(np.array(stack_from_center(DXYZ[0])), np.array(stack_from_center(DXYZ[1])))

    dirpth = Path(savedir)
    makedirs(dirpth, exist_ok=True)
    for i, val2d in enumerate(val_3d):
        fpth = dirpth.joinpath(f"{i}.png")
        fig, ax = plt.subplots()
        mappable = ax.pcolormesh(grid_x, grid_y, val2d)
        pp = fig.colorbar(mappable, ax=ax, orientation="vertical")
        pp.set_label(prop_name)
        ax.set_aspect("equal")
        fig.savefig(fpth, dpi=200, bbox_inches="tight")
        plt.clf()
        plt.close()

def plot_results(fpth) -> None:
    fpth = Path(fpth)
    for prop_name in CONVERSION_CRITERIA:
        for axis in ("X", "Y", "Z"):
            savedir = fpth.parent.joinpath(prop_name).joinpath(axis)
            makedirs(savedir, exist_ok=True)
            plot_sum(fpth, prop_name, savedir, True, axis)

if __name__ == "__main__":
    # cellid_props, srcid_props, time = load_sum(r"E:\tarumai\200.0_0.0_100.0_10.0\tmp.0000.SUM")
    # for i, (_, prop) in enumerate(cellid_props.items()):
    #     if i == 0:
    #         print(prop)
    #     if isnan(prop["PRES"]):
    #         print(i)
    plot_results(r"E:\tarumai\200.0_0.0_100.0_10.0\tmp.0159.SUM")
    # plt_conv(r"E:\tarumai\200.0_0.0_100.0_10.0", "TEMPC")
    # plt_conv(r"E:\tarumai\200.0_0.0_100.0_10.0", "PRES")
    # plt_conv(r"E:\tarumai\200.0_0.0_100.0_10.0", "SAT#GAS")
    # plt_conv(r"E:\tarumai\200.0_0.0_100.0_10.0", "COMP1T")
