from typing import Union
from os import PathLike, makedirs
import subprocess
from pathlib import Path
from warnings import warn

from generate_input import generate_mikasa_input
from constants import CONDS_PID_MAP_NAME, RUNFILENAME, H64PTH, LOGFILENAME
from monitor import is_converged
from utils import simulation_dir

from scenario import *

def run_single_condition(
    scenario: BaseSenario,
    ignore_convergence: bool = False,
) -> None:
    """Run MUFITS simulator in single condition 
    """
    sim_dir: Path = simulation_dir(scenario.get_sim_id())
    makedirs(sim_dir, exist_ok=True)
    if is_converged(sim_dir) and not ignore_convergence:
        warn(f"{sim_dir.name} is already converged. Exit.")
        return
    if scenario is Shutdown:
        pass
    runpth = sim_dir.joinpath(RUNFILENAME)
    generate_mikasa_input(scenario, runpth)
    exepth = H64PTH
    logpth = sim_dir.joinpath(LOGFILENAME)

    print(f"RUN: {runpth}")
    with open(logpth, "w") as outfile:
        outfile.write("")
        p = subprocess.Popen(f"{exepth} {runpth}", stdout=outfile)
    _str = f"{str(sim_dir)}, {p.pid}\n"
    makedirs(Path(sim_dir).joinpath("tmp"), exist_ok=True)
    with open(Path(sim_dir).joinpath("tmp").joinpath(CONDS_PID_MAP_NAME), "w") as f:
        f.write(_str)
    p.wait()

if __name__ == "__main__":
    run_single_condition(PombetsuHighCapMB())
    pass
