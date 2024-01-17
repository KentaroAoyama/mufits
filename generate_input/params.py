from copy import deepcopy
from constants import TOPO_CONST_PROPS, IDX_AIR


class PARAMS:
    def __init__(
        self,
        pres_src: float = 50.0,
        temp_src: float = 700.0,
        comp1t: float = 0.005,
        rain_unit: float = 3.53,
        xco2_air: float = 3.8e-4, # TODO: move to constant
        temp_rain: float = TOPO_CONST_PROPS[IDX_AIR]["TEMPC"],
        perm_vent: float = 10.0,
        inj_rate: float = 2000.0,
        cap_scale: float = None
    ) -> None:
        """Parameters

        Args:
            pres_src (float): Pressure of magmatic source (MPa).
            temp_src (float): Temperature of magmatic source (℃).
            comp1t (float): CO2 mole fraction of magmatic source.
            rain_unit (float): Amount of rain sources (mm/day).
            xco2_air (float): CO2 mole fraction of rain.
            temp_rain (float): Temperature of rain sources (℃).
            perm_vent (float): Factor multiplied by the permeability of the host rock.
            inj_rate (float): Injection rate (t/day)
        """
        # Source properties
        self.PRES_SRC = pres_src
        self.SRC_TEMP = temp_src
        self.SRC_COMP1T = comp1t

        # Rain properties
        self.RAIN_AMOUNT = rain_unit  # mm/day
        self.XCO2_AIR = xco2_air
        self.TEMP_RAIN = temp_rain

        # Vent properties
        self.VENT_SCALE = perm_vent
        self.TOPO_PROPS = deepcopy(TOPO_CONST_PROPS)
        self.INJ_RATE = inj_rate

        # cap properties
        self.CAP_SCALE: float = cap_scale



class PARAMS_VTK:
    VLIM = {
        "PHST": (0.0, 1.0),
        "PRES (MPA)": (0.0, 10.0),
        "TEMPC (C)": (0.0, 500.0),
        "SAT#GAS": (0.0, 1.0),
        "SAT#LIQ": (0.0, 1.0),
        "COMP1T": (0.0, 1.0),
        "COMP2T": (0.0, 1.0),
    }
    XLIM = (2426.45, 6426.45)
    YLIM = (-6426.45, -2054.94)
    ZLIM = (-1200.0, 0.0)

TUNING_PARAMS = {(200.0, 0.0, 100.0, 10.0): 4.0 / (24.0 * 60.0 * 60),
                (200.0, 0.0, 100.0, 100.0): 4.0 / (24.0 * 60.0 * 60),
                (200.0, 0.0, 100.0, 1000.0): 4.0 / (24.0 * 60.0 * 60),
                (200.0, 0.0, 100.0, 10000.0): 4.0 / (24.0 * 60.0 * 60),
                (200.0, 0.0, 1000.0, 10.0): 4.0 / (24.0 * 60.0 * 60),
                (200.0, 0.0, 1000.0, 100.0): 4.0 / (24.0 * 60.0 * 60),
                (200.0, 0.0, 1000.0, 1000.0): 4.0 / (24.0 * 60.0 * 60),
                (200.0, 0.0, 1000.0, 10000.0): 4.0 / (24.0 * 60.0 * 60),
                (200.0, 0.0, 10000.0, 10.0): 4.0 / (24.0 * 60.0 * 60),
                (200.0, 0.0, 10000.0, 100.0): 4.0 / (24.0 * 60.0 * 60),
                (200.0, 0.0, 10000.0, 1000.0): 4.0 / (24.0 * 60.0 * 60),
                (200.0, 0.0, 10000.0, 10000.0): 4.0 / (24.0 * 60.0 * 60),
          }
