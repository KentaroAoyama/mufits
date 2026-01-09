from constants import (RockType,
                       POROS,
                       DENS_ROCK,
                       TEMPE_AIR,
                       TEMPC_INJ_FLUID,
                       XCO2_MB,
                       P_MB,
                       INJ_RATE_MB,
                       INJ_UNIT_MB,
                       HEAD_LATLNG_HORONAI_MB,
                       DEPTH_ULIM_MB,
                       RockProps,
                       FluidProps,
                       WellProps,
                       )

def kc2k(kc: float):
    # convert hydraulic conductivity (m/s) to permeability (m2) 
    mu = 1.3064e-3 # pure water, 10℃
    rho = 0.999694e3 # pure water, 10℃
    g = 9.80665
    return kc * mu / (rho * g)

def calc_khv(k1: float, k2: float, d1: float, d2: float) -> tuple[float, float]:
    kh = (k1 * d1 + k2 * d2) / (d1 + d2)
    kv = (d1 + d2) / (d1/k1 + d2/k2)
    return kh, kv

kch, kcv = calc_khv(3.0e-3, 3.0e-4, 3.0, 12.0)  # 三笠市CO2地下固定研究業務受託コンソーシアム (2022), 三笠市 CO2 地下固定研究業務, pp61
PERM_COAL_H, PERM_COAL_V = kc2k(kch), kc2k(kcv)
PERM_CAP = kc2k(3.0e-11)  # 三笠市CO2地下固定研究業務受託コンソーシアム (2022), 三笠市 CO2 地下固定研究業務, pp62

PERM_CAP_HIGH = 1.0e-9

### Rock properties
# HCONDCFX: Bulk thermal conductivity in X-axis (W/(m・K))
# HCONDCFY: Bulk thermal conductivity in Y-axis (W/(m・K))
# HCONDCFZ: Bulk thermal conductivity in Z-axis (W/(m・K))
# PORO: Effective porosity (0―1)
# PERMX: Intrinsic permeability in X-axis (m^2)
# PERMY: Intrinsic permeability in Y-axis (m^2)
# PERMZ: Intrinsic permeability in Z-axis (m^2)
# DENS: Grain density (kg/m^3)
# HC: Heat capacity of grain (kJ/(kg・℃))
# TEMPC: Initial temperature (℃) (OUTDATED)
# COMP1T: Initial molar fraction of CO2 (OUTDATED)
# TODO: properties to ENUM
DEFAULT_ROCK_PROPS: RockProps = {
    # 幾春別層でない領域
    RockType.LAND: {
        "HCONDCFX": 2.0,
        "HCONDCFY": 2.0,
        "HCONDCFZ": 2.0,
        "PORO": POROS,
        # "PERMX": PERM_HOST,
        # "PERMY": PERM_HOST,
        # "PERMZ": PERM_HOST,
        "DENS": DENS_ROCK,
        "HC": 1.0,
        "TEMPC": 20.0,
        "COMP1T": 3.25e-7,
    },
    RockType.CAP: {
        "HCONDCFX": 2.0,
        "HCONDCFY": 2.0,
        "HCONDCFZ": 2.0,
        "PORO": POROS,
        "PERMX": PERM_CAP,
        "PERMY": PERM_CAP,
        "PERMZ": PERM_CAP,
        "DENS": DENS_ROCK,
        "HC": 1.0,
        "TEMPC": 20.0,
        "COMP1T": 3.25e-7,
    },
    RockType.AIR: {
        "HCONDCFX": 0.0241,
        "HCONDCFY": 0.0241,
        "HCONDCFZ": 0.0241,
        "PORO": 1.0,
        "PERMX": 0.0,
        "PERMY": 0.0,
        "PERMZ": 0.0,
        "DENS": 1.293,
        "HC": 1.007,
        "TEMPC": TEMPE_AIR,  # OUTDATED
        "COMP1T": 1.0,       # OUTDATED
    },
    RockType.COAL: {
        "HCONDCFX": 2.0,
        "HCONDCFY": 2.0,
        "HCONDCFZ": 2.0,
        "PORO": 0.2,
        "PERMX": PERM_COAL_H,
        "PERMY": PERM_COAL_H,
        "PERMZ": PERM_COAL_V,
        "DENS": DENS_ROCK,
        "HC": 1.0,
        "TEMPC": 20.0,     # OUTDATED
        "COMP1T": 3.25e-7  # OUTDATED
    },
}

DEFAULT_FLUID_PROPS: FluidProps = {"TEMPC": TEMPC_INJ_FLUID,
                                   "PRES": P_MB,
                                   "COMP1T": XCO2_MB}

DEFAULT_WELL_PROPS: WellProps = {
                                 "WELL_ID": "WA",
                                 "FLUID": DEFAULT_FLUID_PROPS,
                                 "INJE_RATE": INJ_RATE_MB,
                                 "INJE_UNIT": INJ_UNIT_MB,
                                 "LATLNG_HEAD": HEAD_LATLNG_HORONAI_MB,
                                 "ULIM": DEPTH_ULIM_MB
                                 }

if __name__ == "__main__":
    pass