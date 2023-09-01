from copy import deepcopy
from constants import TOPO_CONST_PROPS, IDX_VENT, IDX_AIR


class PARAMS:
    def __init__(
        self,
        pres_src: float = 50.0,
        temp_src: float = 700.0,
        comp1t: float = 0.005,
        rain_unit: float = 1.50339178,
        xco2_rain: float = 3.8e-4,
        temp_rain: float = TOPO_CONST_PROPS[IDX_AIR]["TEMPC"],
        perm_vent: float = 1.0e-12,
        inj_rate: float = 2000.0,
    ) -> None:
        """Parameters

        Args:
            pres_src (float): Pressure of magmatic source (MPa).
            temp_src (float): Temperature of magmatic source (℃).
            comp1t (float): CO2 mole fraction of magmatic source.
            rain_unit (float): Amount of rain sources (mm/day).
            xco2_rain (float): CO2 mole fraction of rain.
            temp_rain (float): Temperature of rain sources (℃).
            perm_vent (float): Permeability of vent (m^2).
            inj_rate (float): Injection rate (t/day)
        """
        # Source properties
        self.PRES_SRC = pres_src
        self.SRC_TEMP = temp_src
        self.SRC_COMP1T = comp1t

        # Rain properties
        self.RAIN_AMOUNT = rain_unit  # mm/day
        self.XCO2_RAIN = xco2_rain
        self.TEMP_RAIN = temp_rain

        # Vent properties
        # convert SI unit to mD (mili darcy)
        # 1 darcy is equivalent to 9.869233×10−13 m²
        self.PEAM_VENT = perm_vent / 9.869233 * 1.0e16
        self.TOPO_PROPS = deepcopy(TOPO_CONST_PROPS)
        self.TOPO_PROPS[IDX_VENT]["PERMX"] = self.PEAM_VENT
        self.TOPO_PROPS[IDX_VENT]["PERMY"] = self.PEAM_VENT
        self.TOPO_PROPS[IDX_VENT]["PERMZ"] = self.PEAM_VENT
        self.INJ_RATE = inj_rate


class PARAMS_VTK:
    VLIM = {
        "PHST": (0.0, 1.0),
        "PRES (MPA)": (0.0, 3.0),
        "TEMPC (C)": (0.0, 500.0),
        "SAT#GAS": (0.0, 1.0),
        "SAT#LIQ": (0.0, 1.0),
        "COMP1T": (0.0, 1.0),
        "COMP2T": (0.0, 1.0),
    }
