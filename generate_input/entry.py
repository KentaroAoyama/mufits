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
from utils import condition_to_dir

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
    temp: float, comp1t: float, inj_rate: float, perm_vent: float, cap_scale: float, sim_dir: PathLike, from_latest: bool=False,
) -> None:
    sim_dir = Path(sim_dir)
    if is_converged(sim_dir):
        return None

    # make directory for each condition, generate run file, and run
    params = PARAMS(
        temp_src=temp,
        comp1t=comp1t,
        inj_rate=inj_rate,
        perm_vent=perm_vent,
        cap_scale=cap_scale
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


def search_conditions(max_workers: int = cpu_count() - 5, from_latest: bool = False, with_cap: bool = False):
    assert max_workers < cpu_count() - 1, max_workers
    with open(Path("./conditions.yml"), "r") as ymf:
        conditions: Dict = yaml.safe_load(ymf)
    # NOTE: 1 for monitor
    pool = futures.ProcessPoolExecutor(max_workers=max_workers)
    conds_dct: OrderedDict[Future] = OrderedDict()
    cou = 0
    makedirs(base_dir, exist_ok=True)
    for temp in sorted(conditions["tempe"]):
        for comp1t in sorted(conditions["comp1t"]):
            for inj_rate in sorted(conditions["inj_rate"]):
                for perm_vent in sorted(conditions["pearm"]):
                    for cap_scale in sorted(conditions["cap_scale"]):
                        if not with_cap:
                            cap_scale = None
                        if from_latest:
                            if (temp, comp1t, inj_rate, perm_vent) not in TUNING_PARAMS:
                                continue
                        sim_dir: Path = condition_to_dir(base_dir, temp, comp1t, inj_rate, perm_vent, cap_scale, from_latest,)
                        makedirs(sim_dir, exist_ok=True)
                        # run_single_condition(temp=temp, comp1t=comp1t, inj_rate=inj_rate, perm_vent=perm_vent, cap_scale=cap_scale, sim_dir=sim_dir, from_latest=from_latest) #!
                        pool.submit(run_single_condition, temp=temp, comp1t=comp1t, inj_rate=inj_rate, perm_vent=perm_vent, cap_scale=cap_scale, sim_dir=sim_dir, from_latest=from_latest,)

                        # monitor
                        monitor_pth = sim_dir.joinpath("tmp")
                        makedirs(monitor_pth, exist_ok=True)
                        logger = generate_logger(cou, monitor_pth.joinpath("log.txt"))
                        props: Dict = conds_dct.setdefault((temp, comp1t, inj_rate, perm_vent), {})
                        props.setdefault("DirPth", sim_dir)
                        props.setdefault("MonitorPth", monitor_pth)
                        props.setdefault("Logger", logger)
                        cou += 1

    # monitor_process(conds_dct) #!
    pool.shutdown(wait=True)

if __name__ == "__main__":
    search_conditions(12, False, True)
    # run_single_condition(200.0, 0.1, 100.0, 1000.0, None, r"E:\tarumai4\200.0_0.1_100.0_1000.0", False)
    pass
