from typing import List, Dict
from os import PathLike, makedirs, getcwd
import subprocess
import psutil
from time import sleep
from pathlib import Path
import yaml
from generate_input import generate_from_params
from params import PARAMS
from constants import PREFIX

base_dir = Path(getcwd()).joinpath(PREFIX)


def condition_to_dir(
    tempe_src: float, comp1t: float, inj_rate: float, pearm: float
) -> PathLike:
    return base_dir.joinpath(f"{tempe_src}_{comp1t}_{inj_rate}_{pearm}")


def main():
    # make directory for each condition, generate run file, and run
    with open(Path("./conditions.yml"), "r") as ymf:
        conditions: Dict = yaml.safe_load(ymf)
    for pearm in conditions["pearm"]:
        for temp in conditions["tempe"]:
            for comp1t in conditions["comp1t"]:
                for inj_rate in conditions["inj_rate"]:
                    while True:
                        if (
                            psutil.virtual_memory().percent < 60.0
                            and psutil.cpu_percent(interval=1) < 60.0
                        ):
                            params = PARAMS(
                                temp_src=temp,
                                comp1t=comp1t,
                                inj_rate=inj_rate,
                                perm_vent=pearm,
                            )
                            sim_dir: Path = condition_to_dir(
                                temp, comp1t, inj_rate, pearm
                            )
                            makedirs(sim_dir, exist_ok=True)

                            runpth = sim_dir.joinpath("tmp.RUN")
                            generate_from_params(params, runpth)

                            exepth = base_dir.joinpath("H64.EXE")
                            logpth = sim_dir.joinpath("log.txt")
                            with open(logpth, "w") as outfile:
                                outfile.write("")
                                subprocess.run(f"{exepth} {runpth}", stdout=outfile)
                            break
                        else:
                            sleep(5)


if __name__ == "__main__":
    main()
    pass
