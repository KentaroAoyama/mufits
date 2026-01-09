from typing import Optional
from default_props import (DEFAULT_ROCK_PROPS,
                           DEFAULT_WELL_PROPS,
                           PERM_CAP_HIGH)
from copy import deepcopy
from os import PathLike
from pathlib import Path
from utils import simulation_dir
from constants import (Condition,
                       RockType,
                       WellProps,
                       RockProps,
                       TEMPC_INJ_FLUID,
                       INJ_RATE_MB,
                       INJ_UNIT_MB,
                       P_MB,
                       XCO2_MB,
                       INJ_RATE_SC,
                       INJ_UNIT_SC,
                       P_SC,
                       XCO2_SC,
                       HEAD_LATLNG_HORONAI_MB,
                       HEAD_LATLNG_HORONAI_SC,
                       HEAD_LATLNG_POMBETSU_MB,
                       HEAD_LATLNG_POMBETSU_SC,
                       DEPTH_ULIM_MB,
                       DEPTH_ULIM_SC
                        )

# 変更する条件：capの浸透率、坑跡の座標
# TODO: 流速分布

class BaseSenario:
    def __init__(self, sim_id: Optional[str]=None) -> None:
        self.condition: Condition = self.set_condition(sim_id)

    def set_condition(self, sim_id: str) -> Condition:
        if sim_id is None:
            sim_id: str = self.__class__.__name__
        cnd: Condition = dict()
        cnd["SIM_ID"] = sim_id
        cnd["ROCK_PROPS"] = deepcopy(DEFAULT_ROCK_PROPS)
        cnd["WELL_PROPS"] = deepcopy(DEFAULT_WELL_PROPS)
        self.condition = cnd
        return self.condition

    def get_sim_id(self) -> str:
        return self.condition["SIM_ID"]
    
    def get_rock_props(self) -> RockProps:
        return deepcopy(self.condition["ROCK_PROPS"])
    
    def get_well_props(self) -> WellProps:
        return deepcopy(self.condition["WELL_PROPS"])

    def get_condition(self) -> dict:
        return deepcopy(self.condition)

class HoronaiBase(BaseSenario):
    """幌内炭鉱ベースシナリオ (MB)
    """
    def __init__(self) -> None:
        super().__init__()
        # well props
        well_props = deepcopy(DEFAULT_WELL_PROPS)
        well_props["WELL_ID"] = "H1"
        well_props["FLUID"]["TEMPC"] = TEMPC_INJ_FLUID
        well_props["FLUID"]["PRES"] = P_MB
        well_props["FLUID"]["COMP1T"] = XCO2_MB
        well_props["INJE_RATE"] = INJ_RATE_MB
        well_props["INJE_UNIT"] = INJ_UNIT_MB
        well_props["LATLNG_HEAD"] = HEAD_LATLNG_HORONAI_MB
        well_props["ULIM"] = DEPTH_ULIM_MB
        self.condition["WELL_PROPS"] = well_props

class HoronaiSC(HoronaiBase):
    """
    Horonai, SC CO2
    """
    def __init__(self) -> None:
        super().__init__()
        # well props
        well_props = self.condition["WELL_PROPS"]
        well_props["FLUID"]["PRES"] = P_SC
        well_props["FLUID"]["COMP1T"] = XCO2_SC
        well_props["INJE_RATE"] = INJ_RATE_SC
        well_props["INJE_UNIT"] = INJ_UNIT_SC
        well_props["LATLNG_HEAD"] = HEAD_LATLNG_HORONAI_SC
        well_props["ULIM"] = DEPTH_ULIM_SC
        self.condition["WELL_PROPS"] = well_props

class HoronaiHighCapMB(HoronaiBase):
    """
    Horonai, High cap permeability, MB
    """
    def __init__(self) -> None:
        super().__init__()
        # rock props
        rock_props = self.condition["ROCK_PROPS"]
        rock_props[RockType.CAP]["PERMX"] = PERM_CAP_HIGH
        rock_props[RockType.CAP]["PERMY"] = PERM_CAP_HIGH
        rock_props[RockType.CAP]["PERMZ"] = PERM_CAP_HIGH
        self.condition["ROCK_PROPS"] = rock_props

class HoronaiHighCapSC(HoronaiSC):
    """
    Horonai, High cap permeability, SC
    """
    def __init__(self) -> None:
        super().__init__()
        # rock props
        rock_props = self.condition["ROCK_PROPS"]
        rock_props[RockType.CAP]["PERMX"] = PERM_CAP_HIGH
        rock_props[RockType.CAP]["PERMY"] = PERM_CAP_HIGH
        rock_props[RockType.CAP]["PERMZ"] = PERM_CAP_HIGH
        self.condition["ROCK_PROPS"] = rock_props

class PombetsuBase(BaseSenario):
    """奔別炭鉱ベースシナリオ (MB)
    """
    def __init__(self) -> None:
        super().__init__()
        # well props
        well_props = deepcopy(DEFAULT_WELL_PROPS)
        well_props["WELL_ID"] = "P1"
        well_props["FLUID"]["TEMPC"] = TEMPC_INJ_FLUID
        well_props["FLUID"]["PRES"] = P_MB
        well_props["FLUID"]["COMP1T"] = XCO2_MB
        well_props["INJE_RATE"] = INJ_RATE_MB
        well_props["INJE_UNIT"] = INJ_UNIT_MB
        well_props["LATLNG_HEAD"] = HEAD_LATLNG_POMBETSU_MB
        well_props["ULIM"] = DEPTH_ULIM_MB
        self.condition["WELL_PROPS"] = well_props

class PombetsuSC(PombetsuBase):
    """奔別炭鉱 SC CO2
    """
    def __init__(self) -> None:
        super().__init__()
        # well props
        well_props = self.condition["WELL_PROPS"]
        well_props["FLUID"]["PRES"] = P_SC
        well_props["FLUID"]["COMP1T"] = XCO2_SC
        well_props["INJE_RATE"] = INJ_RATE_SC
        well_props["INJE_UNIT"] = INJ_UNIT_SC
        well_props["LATLNG_HEAD"] = HEAD_LATLNG_POMBETSU_SC
        well_props["ULIM"] = DEPTH_ULIM_SC
        self.condition["WELL_PROPS"] = well_props

class PombetsuHighCapMB(PombetsuBase):
    """
    Pombetsu, High cap permeability, MB
    """
    def __init__(self) -> None:
        super().__init__()
        # rock props
        rock_props = self.condition["ROCK_PROPS"]
        rock_props[RockType.CAP]["PERMX"] = PERM_CAP_HIGH
        rock_props[RockType.CAP]["PERMY"] = PERM_CAP_HIGH
        rock_props[RockType.CAP]["PERMZ"] = PERM_CAP_HIGH
        self.condition["ROCK_PROPS"] = rock_props

class PombetsuHighCapSC(PombetsuSC):
    """
    Pombetsu, High cap permeability, SC
    """
    def __init__(self) -> None:
        super().__init__()
        # rock props
        rock_props = self.condition["ROCK_PROPS"]
        rock_props[RockType.CAP]["PERMX"] = PERM_CAP_HIGH
        rock_props[RockType.CAP]["PERMY"] = PERM_CAP_HIGH
        rock_props[RockType.CAP]["PERMZ"] = PERM_CAP_HIGH
        self.condition["ROCK_PROPS"] = rock_props

class Shutdown(BaseSenario):
    def __init__(self, refpth: PathLike) -> None:
        super().__init__()
        self.refpth: Path = Path(refpth)

if __name__ == "__main__":
    pass
