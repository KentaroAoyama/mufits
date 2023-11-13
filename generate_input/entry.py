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
from params import PARAMS
from constants import OUTDIR, CONDS_PID_MAP_NAME
from monitor import monitor_process
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
    temp: float, comp1t: float, inj_rate: float, perm_vent: float, sim_dir: PathLike
) -> None:

    # make directory for each condition, generate run file, and run
    params = PARAMS(
        temp_src=temp,
        comp1t=comp1t,
        inj_rate=inj_rate,
        perm_vent=perm_vent,
    )

    runpth = sim_dir.joinpath("tmp.RUN")
    generate_from_params(params, runpth)

    exepth = cur_dir.joinpath("H64.EXE")
    logpth = sim_dir.joinpath("log.txt")

    print(f"RUN: {runpth}")
    with open(logpth, "w") as outfile:
        outfile.write("")
        p = subprocess.Popen(f"{exepth} {runpth}", stdout=outfile)
    
    _str = f"{str(condition_to_dir(base_dir, temp, comp1t, inj_rate, perm_vent))}, {p.pid}\n"
    with open(Path(sim_dir).joinpath("tmp").joinpath(CONDS_PID_MAP_NAME), "w") as f:
        f.write(_str)
    p.wait()
    return


def main(max_workers: int = cpu_count() - 5):
    assert max_workers < cpu_count() - 1, max_workers
    with open(Path("./conditions.yml"), "r") as ymf:
        conditions: Dict = yaml.safe_load(ymf)
    # NOTE: 1 for monitor
    pool = futures.ProcessPoolExecutor(max_workers=max_workers)
    conds_dct: OrderedDict[Future] = OrderedDict()
    cou = 0
    makedirs(base_dir, exist_ok=True)
    # process_ls: List[Process] = []
    for temp in conditions["tempe"]:
        for comp1t in conditions["comp1t"]:
            for inj_rate in conditions["inj_rate"]:
                for perm_vent in conditions["pearm"]:
                    sim_dir: Path = condition_to_dir(base_dir, temp, comp1t, inj_rate, perm_vent)
                    makedirs(sim_dir, exist_ok=True)
                    # process = Process(temp, comp1t, inj_rate, perm_vent, sim_dir)
                    pool.submit(run_single_condition, temp=temp, comp1t=comp1t, inj_rate=inj_rate, perm_vent=perm_vent, sim_dir=sim_dir)
                    # process_ls.append(process)

                    # monitor
                    monitor_pth = sim_dir.joinpath("tmp")
                    makedirs(monitor_pth, exist_ok=True)
                    logger = generate_logger(cou, monitor_pth.joinpath("log.txt"))
                    props: Dict = conds_dct.setdefault((temp, comp1t, inj_rate, perm_vent), {})
                    props.setdefault("DirPth", sim_dir)
                    props.setdefault("MonitorPth", monitor_pth)
                    props.setdefault("Logger", logger)
                    cou += 1

    monitor_process(conds_dct)
    pool.shutdown(wait=True)


if __name__ == "__main__":
    main(10)
    pass
