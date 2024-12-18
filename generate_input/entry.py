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
    disperse_magmasrc: bool = False,
) -> None:
    """Run MUFITS simulator in single condition 

    Args:
        temp (float): Source temperature (℃)
        comp1t (float): Molar fraction of CO2 in source
        inj_rate (float): Injected fluid rate from bottom (t/day)
        perm_vent (float): Permeability ratio: conduit / host rock
        cap_scale (float): Permeability ratio: (cap rock top) / (default cap rock: 10^-17)
            If None, cap rock is not set.
        base_dir (PathLike): Parent directory of each condition (.../{base_dir}/{condition})
        from_latest (bool, optional): Controls whether to compute from latest 
            SUM file or not (True: compute from latest SUM file)
        ignore_convergence (bool, optional): By default, directory that contains
            convergenced result is skipped, but you can ignore it by setting this
            parameter to True.
        vk (bool, optional): Controls wheter to set high permeability only on 
            Z-axis or not (True: high permeability will only be set on Z-axis).
        disperse_magmasrc (bool, optional): Controls wheter to inject magmatic
            fluid from multiple bottom blocks (True: inject from multiple blocks)
    """
    sim_dir: Path = condition_to_dir(
        base_dir,
        temp,
        comp1t,
        inj_rate,
        perm_vent,
        cap_scale,
        from_latest,
        vk=vk,
        disperse_magmasrc=disperse_magmasrc,
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
        disperse_magmasrc=disperse_magmasrc,
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
    """Simulate MUFITS on multiple conditions written in ./conditions.yml

    Args:
        max_workers (int): Number of CPUs used for parallel computing
        from_latest (bool): Controls whether to compute from latest 
            SUM file or not (True: compute from latest SUM file)
        with_cap (bool): Controls whether to consider cap rock or not 
            (True: consider cap rock) 
        ignore_convergence (bool): Controls whether to ignore directory
            that already converged or not (True: ignore converged condition) 
    """
    assert max_workers < cpu_count() - 1, max_workers
    with open(Path("./conditions.yml"), "r") as ymf:
        conditions: Dict = yaml.safe_load(ymf)
    # NOTE: 1 for monitor
    pool = futures.ProcessPoolExecutor(max_workers=max_workers)
    conds_dct: OrderedDict[Future] = OrderedDict()
    disp: bool = conditions["disperse_magmasrc"][0]
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
                                vk,
                                disp,
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
                                disperse_magmasrc=disp,
                            )

                            # monitor
                            monitor_pth = sim_dir.joinpath("tmp")
                            makedirs(monitor_pth, exist_ok=True)
                            logger = generate_logger(
                                cou, monitor_pth.joinpath("log.txt")
                            )
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
    disperse_magmasrc: bool,
):
    """Calculate unrest phase

    Args:
        base_dir (PathLike): Parent directory of each condition 
            (.../{base_dir}/{condition})
        refpth (PathLike): Path of SUM file to be referenced. Parameters in
            this file will be initial parameter of this unrest calculation. 
        temp (float): Source temperature (℃)
        comp1t (float): Molar fraction of CO2 in source
        q (float): Injected fluid rate from bottom (t/day)
        a (float): Permeability ratio: conduit / host rock
        c (float): Permeability ratio: (cap rock top) / (default cap rock: 10^-17)
        vk (bool): Controls wheter to set high permeability only on 
            Z-axis or not (True: high permeability will only be set on Z-axis).
        disperse_magmasrc (bool): Controls wheter to inject magmatic
            fluid from multiple bottom blocks (True: inject from multiple blocks)
    """
    simdir: Path = condition_to_dir(
        base_dir, temp, comp1t, q, a, c, vk=vk, disperse_magmasrc=disperse_magmasrc
    )
    makedirs(simdir, exist_ok=True)
    params = PARAMS(
        temp_src=temp,
        comp1t=comp1t,
        inj_rate=q,
        perm_vent=a,
        cap_scale=c,
        vk=vk,
        rain_unit=calc_infiltration() * 1000.0,
        disperse_magmasrc=disperse_magmasrc,
    )
    runpth = simdir.joinpath("tmp.RUN")
    # min, max, rptstep
    generate_from_params(
        params,
        runpth,
        refpth=refpth,  # tuning_params=(1.0e-4, 10.0 / 400.0, 10.0)
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
    """Simulate MUFITS on multiple conditions written in ./conditions_unrest.yml

    Args:
        simdir (PathLike): Directory where results of steady state exist 
            (e.g., .../900.0_0.0_100.0_10.0/)
        max_workers (int): Number of CPUs used for parallel computing
    """
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
            refpth = simdir.joinpath(f"tmp.{fn}.SUM")
        else:
            break
    print(f"refpth: {refpth}")
    # obtain base condition
    conds: Dict = dir_to_condition(simdir)

    default_cond = {
        "tempe": conds["tempe"],
        "comp1t": conds["comp1t"],
        "inj_rate": conds["inj_rate"],
        "perm": conds["perm"],
        "cap_scale": conds["cap_scale"],
        "vk": conds["vk"],
        "disperse_magmasrc": conds["d"],
    }
    disperse_magmasrc = conditions["disperse_magmasrc"]
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
                                and disperse_magmasrc
                                == default_cond["disperse_magmasrc"]
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
                                vk,
                                disperse_magmasrc,
                            )
    pool.shutdown(wait=True)


if __name__ == "__main__":
    # search_conditions(4, False, False, True)
    run_single_condition(900.0,
                         0.0,
                         100.0,
                         10.0,
                         None,
                         base_dir,
                         False,
                         True,
                         True,
                         False
                         )
    
    # run_single_unrest(r"E:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest",
    #                   r"E:\tarumai2\900.0_0.0_1000.0_10.0_v\tmp.0028.SUM",
    #                   900.0,
    #                   0.0,
    #                   15000.0,
    #                   10.0,
    #                   None,
    #                   True,
    #                   True)
    pass
