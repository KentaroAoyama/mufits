# TODO: should not call command in this directory

from typing import Dict, OrderedDict, List, Any
from os import PathLike, makedirs, getcwd, cpu_count, getpid
import subprocess
from pathlib import Path
from concurrent import futures
from logging import getLogger, FileHandler, Formatter, DEBUG, Logger
from collections import OrderedDict
from copy import copy

import yaml
from generate_input import generate_from_params
from params import PARAMS, TUNING_PARAMS
from constants import OUTDIR, CONDS_PID_MAP_NAME
from monitor import monitor_process, is_converged
from utils import condition_to_dir, dir_to_condition, calc_infiltration, unrest_dir

Future = futures.Future

cur_dir = Path(getcwd())
base_dir = Path(OUTDIR)


def generate_logger(i, fpth) -> Logger:
    # create logger
    logger = getLogger(str(i))
    logger.setLevel(DEBUG)
    file_handler = FileHandler(fpth, mode="a", encoding="utf-8")
    handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(handler_format)
    logger.addHandler(file_handler)
    return logger


def run_single_condition(
    temp: float,
    comp1t: float,
    inj_rate: float,
    perm_vent: float,
    cap_scale: float,
    base_dir: PathLike,
    from_latest: bool = False,
    ignore_convergence: bool = False,
    vk: bool = False,
) -> None:
    sim_dir: Path = condition_to_dir(
        base_dir,
        temp,
        comp1t,
        inj_rate,
        perm_vent,
        cap_scale,
        from_latest,
        vk
    )
    makedirs(sim_dir, exist_ok=True)
    if is_converged(sim_dir) and not ignore_convergence:
        return None

    # make directory for each condition, generate run file, and run
    params = PARAMS(
        temp_src=temp,
        comp1t=comp1t,
        inj_rate=inj_rate,
        perm_vent=perm_vent,
        cap_scale=cap_scale,
        rain_unit=calc_infiltration() * 1000.0,
        vk=vk,
    )

    runpth = sim_dir.joinpath("tmp.RUN")
    generate_from_params(params, runpth, from_latest)
    exepth = cur_dir.joinpath("H64.EXE")
    logpth = sim_dir.joinpath("log.txt")

    print(f"RUN: {runpth}")
    with open(logpth, "w") as outfile:
        outfile.write("")
        p = subprocess.Popen(f"{exepth} {runpth}", stdout=outfile)

    _str = f"{str(sim_dir)}, {p.pid}\n"
    makedirs(Path(sim_dir).joinpath("tmp"), exist_ok=True)
    with open(Path(sim_dir).joinpath("tmp").joinpath(CONDS_PID_MAP_NAME), "w") as f:
        f.write(_str)
    p.wait()
    return


def search_conditions(
    max_workers: int = cpu_count() - 5,
    from_latest: bool = False,
    with_cap: bool = False,
    ignore_convergence: bool = False,
):
    assert max_workers < cpu_count() - 1, max_workers
    with open(Path("./conditions.yml"), "r") as ymf:
        conditions: Dict = yaml.safe_load(ymf)
    # NOTE: 1 for monitor
    pool = futures.ProcessPoolExecutor(max_workers=max_workers)
    conds_dct: OrderedDict[Future] = OrderedDict()
    cou = 0
    makedirs(base_dir, exist_ok=True)
    for temp in reversed(sorted(conditions["tempe"])):
        for comp1t in sorted(conditions["comp1t"]):
            for inj_rate in sorted(conditions["inj_rate"]):
                for perm_vent in sorted(conditions["perm"]):
                    for i, cap_scale in enumerate(sorted(conditions["cap_scale"])):
                        if not with_cap and i != 0:
                            continue
                        elif not with_cap:
                            cap_scale = None
                        for vk in conditions["vk"]:
                            sim_dir: Path = condition_to_dir(
                                base_dir,
                                temp,
                                comp1t,
                                inj_rate,
                                perm_vent,
                                cap_scale,
                                from_latest,
                                vk
                            )
                            makedirs(sim_dir, exist_ok=True)
                            # run_single_condition(temp=temp, comp1t=comp1t, inj_rate=inj_rate, perm_vent=perm_vent, cap_scale=cap_scale, base_dir=base_dir, from_latest=from_latest, ignore_convergence=ignore_convergence, vk=vk) #!
                            pool.submit(
                                run_single_condition,
                                temp=temp,
                                comp1t=comp1t,
                                inj_rate=inj_rate,
                                perm_vent=perm_vent,
                                cap_scale=cap_scale,
                                base_dir=base_dir,
                                from_latest=from_latest,
                                ignore_convergence=ignore_convergence,
                                vk=vk,
                            )

                            # monitor
                            monitor_pth = sim_dir.joinpath("tmp")
                            makedirs(monitor_pth, exist_ok=True)
                            logger = generate_logger(cou, monitor_pth.joinpath("log.txt"))
                            props: Dict = conds_dct.setdefault(
                                (temp, comp1t, inj_rate, perm_vent), {}
                            )
                            props.setdefault("DirPth", sim_dir)
                            props.setdefault("MonitorPth", monitor_pth)
                            props.setdefault("Logger", logger)
                            cou += 1

    # monitor_process(conds_dct) #!
    pool.shutdown(wait=True)


def run_single_unrest(
    base_dir: PathLike,
    refpth: PathLike,
    temp: float,
    comp1t: float,
    q: float,
    a: float,
    c: float,
    vk: bool,
):
    simdir: Path = condition_to_dir(base_dir, temp, comp1t, q, a, c, vk=vk)
    makedirs(simdir, exist_ok=True)
    params = PARAMS(
        temp_src=temp,
        comp1t=comp1t,
        inj_rate=q,
        perm_vent=a,
        cap_scale=c,
        vk=vk,
        rain_unit=calc_infiltration() * 1000.0,
    )
    runpth = simdir.joinpath("tmp.RUN")
    # min, max, rptstep
    generate_from_params(
        params, runpth, refpth=refpth, # tuning_params=(1.0e-4, 10.0 / 400.0, 10.0)
    )
    exepth = cur_dir.joinpath("H64.EXE")
    logpth = simdir.joinpath("log.txt")

    print(f"RUN: {runpth}")
    with open(logpth, "w") as outfile:
        outfile.write("")
        p = subprocess.Popen(f"{exepth} {runpth}", stdout=outfile)

    _str = f"{str(simdir)}, {p.pid}\n"
    makedirs(Path(simdir).joinpath("tmp"), exist_ok=True)
    with open(Path(simdir).joinpath("tmp").joinpath(CONDS_PID_MAP_NAME), "w") as f:
        f.write(_str)
    p.wait()
    print(runpth)
    return


def run_unrest(
    simdir: PathLike,
    max_workers: int = cpu_count() - 5,
) -> None:
    simdir = Path(simdir)
    base_dir = unrest_dir(simdir)
    with open(Path("./conditions_unrest.yml"), "r") as ymf:
        conditions: Dict = yaml.safe_load(ymf)
    refpth = None
    for i in range(100000):
        fn = str(i).zfill(4)
        fpth = simdir.joinpath(f"tmp.{fn}.SUM")
        if i == 0 and not fpth.exists():
            raise
        elif fpth.exists():
            fn = str(i - 1).zfill(4)
            refpth = simdir.joinpath(f"tmp.{fn}.SUM")
        else:
            break
    print(f"refpth: {refpth}")
    # obtain base condition
    cond_ls = dir_to_condition(simdir)
    cap_scale, vk = None, False
    if len(cond_ls) == 6:
        cap_scale = cond_ls[4]
        vk = True
    elif len(cond_ls) == 5 and isinstance(cond_ls[4], bool):
        vk = cond_ls[4]
    elif len(cond_ls) == 5:
        cap_scale = cond_ls[4]
    default_cond = {
        "tempe": cond_ls[0],
        "comp1t": cond_ls[1],
        "inj_rate": cond_ls[2],
        "perm": cond_ls[3],
        "cap_scale": cap_scale,
        "vk": vk,
    }
    pool = futures.ProcessPoolExecutor(max_workers=max_workers)
    for tempe in sorted(conditions["tempe"]):
        if tempe == "same":
            tempe = default_cond["tempe"]
        for comp1t in sorted(conditions["comp1t"]):
            if comp1t == "same":
                comp1t = default_cond["comp1t"]
            for inj_rate in sorted(conditions["inj_rate"]):
                if inj_rate == "same":
                    inj_rate = default_cond["inj_rate"]
                for perm_vent in sorted(conditions["perm"]):
                    if perm_vent == "same":
                        perm_vent = default_cond["perm"]
                    for cap_scale in sorted(conditions["cap_scale"]):
                        if cap_scale == "same":
                            cap_scale = default_cond["cap_scale"]
                        for vk in conditions["vk"]:
                            if (
                                tempe == default_cond["tempe"]
                                and comp1t == default_cond["comp1t"]
                                and inj_rate == default_cond["inj_rate"]
                                and perm_vent == default_cond["perm"]
                                and cap_scale == default_cond["cap_scale"]
                                and vk == default_cond["vk"]
                            ):
                                continue
                            # run_single_unrest(base_dir,
                            #     refpth,
                            #     tempe,
                            #     comp1t,
                            #     inj_rate,
                            #     perm_vent,
                            #     cap_scale,
                            #     vk)
                            pool.submit(
                                run_single_unrest,
                                base_dir,
                                refpth,
                                tempe,
                                comp1t,
                                inj_rate,
                                perm_vent,
                                cap_scale,
                                vk
                            )
    pool.shutdown(wait=True)


if __name__ == "__main__":
    search_conditions(8, False, True)
    # run_single_condition(
    #     900.0, 0.1, 1000.0, 10.0, 1.0, r"E:\tarumai2", False, True, True
    # )
    # run_single_condition(
    #     900.0, 0.1, 1000.0, 10.0, None, r"E:\tarumai2", False, True, True
    # )
    # run_single_condition(
    #     900.0, 0.1, 1000.0, 10000.0, 1.0, r"E:\tarumai2", False, False, True
    # )
    # キャップなし
    # run_single_condition(
    #     900.0, 0.1, 10000.0, 10.0, None, r"E:\tarumai2", False, False, True
    # )
    # run_single_condition(
    #     900.0, 0.1, 10000.0, 10000.0, None, r"E:\tarumai2", False, False, True
    # )
    # run_single_condition(
    #     900.0, 0.1, 1000.0, 10000.0, None, r"E:\tarumai2", False, False, True
    # )

    # run_unrest(r"E:\tarumai2\900.0_0.1_10000.0_10.0_1.0_v", 1)
    # run_unrest(r"E:\tarumai2\900.0_0.1_10000.0_10000.0_1.0_v", 1)
    # run_unrest(r"E:\tarumai2\900.0_0.1_1000.0_10000.0_100000.0_v", 2)
    # run_unrest(r"E:\tarumai2\900.0_0.1_10000.0_10000.0_v", 1)
    # run_unrest(r"E:\tarumai2\900.0_0.1_1000.0_10000.0_v", 2)
    # run_unrest(r"E:\tarumai2\900.0_0.0_10000.0_10.0_1.0_v", 2)
    # run_unrest(r"E:\tarumai2\900.0_0.0_10000.0_10000.0_100000.0_v", 1)
    pass
