from typing import TypedDict, Optional, Iterator
from dataclasses import dataclass
from pathlib import Path
from enum import Enum, auto


RUNFILENAME_PREFIX = "MIKASA"
RUNFILENAME = RUNFILENAME_PREFIX + ".RUN"
H64PTH = "H64.EXE"
LICENSE_PTH = "LICENSE.LIC"
EOSPTH = "CO2H2O_V3.0.EOS"
LOGFILENAME = "log.txt"

DEM_CRS = "EPSG:6668"  # JGD2011（日本の測地系）
GEO_CRS = "EPSG:6668"  # 地質図幅のCRS
RECT_CRS = "EPSG:6680"
XYZPTH = r"E:\mikasa_ccs\data\xyz\out.xyz"
DEMPTH = r"E:\mikasa_ccs\data\dem"

TOP = 400.0
ORIGIN = (43.2574705, 141.9383815, TOP)
class RockType(Enum):
    LAND = auto()
    AIR = auto()
    CAP = auto()
    COAL = auto()

SUBSURFACE = set((RockType.LAND,
                  RockType.CAP,
                  RockType.COAL,))

INNACTIVATE_ROCK_TYPES = set((RockType.AIR,))

COAL_LAYER_NAME = "ikushumbetsu"
COAL_LAYER_DATA_PTH = "./geology/ikushumbetsu_vertical"
COAL_VERTICAL_SECTIONS = ("ABC", "DE", "KL", "MN", "OP", "QR")
COAL_HORIZONTAL_DATA_PTH = ("./geology/ic0_1.csv","./geology/ic1_1.csv",)

# rainfall (m / year)
# https://www.hro.or.jp/upload/3673/kenpo25-4.pdf
# Last ten years average of total amount: https://www.data.jma.go.jp/stats/etrn/view/annually_s.php?prec_no=15&block_no=47413&year=&month=&day=&view=
RAIN_AMOUNT = 1329.7 * 1.0e-3

# evaporation rate (80% of rainfall for now)(m /year)
EVAP_AMOUNT = 1063.76 * 1.0e-3

# resolution of each data
RES_DEM = 10.0 * 10.0

# parameters of Manning & Ingebritsen (1998)
MIA = -14.0
MIB = -3.2

# Upper limit of permeability
PERM_MAX: float = 1.0e-9

# PATH
DEM_PTH = "./dem"
SEADEM_PTH = "./seadem"
CACHE_DIR = Path.cwd().joinpath("cache")
CACHE_DEM_FILENAME = "dem.pickle" # need not to be changed
CACHE_SEA_FILENAME = "sea.pickle" # need not to be changed

ALIGN_CENTER = True
XCO2_AIR = 3.8e-4
TEMP_RAIN = 10.0

DXYZ = ([743.00834,619.17362,515.97802, 429.981696, 358.31808, 298.5984, 248.832, 207.36, 172.8, 144.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 144.0, 172.79999999999998, 207.35999999999999, 248.83199999999997, 298.59839999999997, 358.31807999999995, 429.98169599999994, 515.97802, 619.17362, 743.00834,],
    [743.00834,619.17362,515.97802, 429.981696, 358.31808, 298.5984, 248.832, 207.36, 172.8, 144.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 144.0, 172.79999999999998, 207.35999999999999, 248.83199999999997, 298.59839999999997, 358.31807999999995, 429.98169599999994, 515.97802, 619.17362, 743.00834,],
    [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
    ],
)

DXYZ = ([743.00834,619.17362,515.97802, 429.981696, 358.31808, 298.5984, 248.832, 207.36, 172.8, 144.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 144.0, 172.79999999999998, 207.35999999999999, 248.83199999999997, 298.59839999999997, 358.31807999999995, 429.98169599999994, 515.97802, 619.17362, 743.00834,],
    [743.00834,619.17362,515.97802, 429.981696, 358.31808, 298.5984, 248.832, 207.36, 172.8, 144.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 144.0, 172.79999999999998, 207.35999999999999, 248.83199999999997, 298.59839999999997, 358.31807999999995, 429.98169599999994, 515.97802, 619.17362, 743.00834,],
    [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
    ],
)

# Constant/Default parameters
# NOTE: unit of "HCONDCF" is W/(m・K), "PERM" is mD
POROS = 0.1    # 三笠市 CO2地下固定研究事業受託コンソーシアム (2025) pp.29
PERM_HOST = 1.0e-16 / 9.869233 * 1.0e16
TEMPE_AIR = 10.0 # ℃  NOTE: must be same as conditions.yml
# Grain density (kg/m3)
DENS_ROCK = 2700.0    # 三笠市 CO2地下固定研究事業受託コンソーシアム (2025) pp.29
# Water density (kg/m3)
DENS_WATER = 1.0e3
DENS_AIR = 1.293

# Heat capacity of grain (Stissi et al., 2021; Hikcs et al., 2009 (10.1029/2008JB006198))
HC_ROCK = 1.0  # OUTDATED

# gravitational acceleration
G = 9.80665

# Atmospheric pressure (MPa)
P_TOP = 1.013e-1
P_GROUND = P_TOP + DENS_AIR * G * ORIGIN[2] * 1.0e-6

# Henry's constant for CO2 gas to water
# https://www.eng-book.com/pdfs/879040e33a05a0e5f1cb85580ef77ad1.pdf
Kh = 0.104e4 * 1.013e-1 * 1.0e6  # NOTE: at 10℃, unit: Pa

# Pressure gradient (MPa/m)
P_GRAD_AIR = DENS_AIR * 1.0e-6
P_GRAD_ROCK = DENS_WATER * G * 1.0e-6

# Temperature gradient (℃/m)
T_GRAD_AIR = 0.0
T_GRAD_ROCK = 0.03

# Simulation time
TIME_SS = 50.0

# Initial time step (in days)
TSTEP_MIN = 1.0e-12
TSTEP_INIT = 1.0e-3
# Maximum time step (days)
TSTEP_MAX = 10.0
# number of iterations for each TSTEP_MAX
NDTFIRST = 100
NDTEND = 7
TMULT = 2.0 # 7 is optimum?

# for SS (in days)
TRPT_UNREST = 2000.0*400 / 24.0 / 3600.0 # unrestに限らず, 途中から計算しなおすときにこの間隔にする
# TRPT_UNREST = 1.0
TEND_UNREST = 150.0

# unrest or continue_from_latest
# 以下Noneでデフォルト値 (NOTE: unrestを計算しないときは, Noneに設定する)
# TSTEP_UNREST = None # Noneでデフォルト値, 設定すれば定数となる
# TRPT_UNREST = 30.0 # in days unrestに限らず, 途中から計算しなおすときにこの間隔にする
# TEND_UNREST = 30.0 # in years

BASEDIR = r"E:\mikasa_ccs_sim"
CONVERSION_CRITERIA = {"TEMPC": 1.0e-2,
                       "PRES": 1.0e-3,
                       "SAT#GAS": 1.0e-4,
                       "COMP1T": 1.0e-4,}
CONDS_PID_MAP_NAME = "pid.txt"

@dataclass
class XY:
    X: list[float]
    Y: list[float]
    def __iter__(self) -> Iterator[tuple[float, float]]:
        assert len(self.X)==len(self.Y)
        return ((self.X[i], self.Y[i]) for i in range(len(self.X)))
    
@dataclass
class XYZ:
    X: list[float]
    Y: list[float]
    Z: list[float]
    def merge(self, xyz: 'XYZ') -> None:
        self.X.extend(xyz.X)
        self.Y.extend(xyz.Y)
        self.Z.extend(xyz.Z)
    def __iter__(self) -> Iterator[tuple[float, float]]:
        assert len(self.X)==len(self.Y)==len(self.Z)
        return ((self.X[i], self.Y[i], self.Z[i]) for i in range(len(self.X)))

class RockProp(TypedDict):
    PERMX: float
    PERMY: float
    PERMZ: float
    HCONDCFX: float
    HCONDCFY: float
    HCONDCFZ: float
    PORO: float
    DENS: float
    HC: float
    TEMPC: Optional[float]
    COMP1T: Optional[float]

class RockProps(TypedDict):
    NAME: RockProp

class FluidProps(TypedDict):
    TEMPC: float
    PRES: float
    COMP1T: float

class WellProps(TypedDict):
    WELL_ID: str
    FLUID: FluidProps
    INJE_RATE: float
    INJE_UNIT: str
    LATLNG_HEAD: tuple[float, float]
    ULIM: float

class Condition(TypedDict):
    """
    Simulation condition
    """
    SIM_ID: str                  # Simulation ID
    ROCK_PROPS: RockProps         # Properties of rock
    WELL_PROPS: WellProps        # Properties of well

@dataclass
class GeologicalData:
    LAYER_NAME: str
    BOTTOM: XYZ
    TOP: XYZ
    def merge(self, data: 'GeologicalData') -> None:
        assert self.LAYER_NAME == data.LAYER_NAME, (self.LAYER_NAME, data.LAYER_NAME)
        self.BOTTOM.merge(data.BOTTOM)
        self.TOP.merge(data.TOP)


# 石炭地質図, 地質図幅に引かれた地質断面の線がある範囲 (MIN, MAX)
GEOLOGICAL_BBOX = ((141.80837563451777,43.02268009430514),
                   (142.13762905524112,43.33337464625808))
# 地質断面図から読み取った座標は(0,1)に規格化してあると想定しているが、その0が標高何メートルに対応するかを指定
BOTTOM_VERTICAL_SECTION = 0.0

# Diameter of well (in ft)
D_WELL = 0.1968503937  # 6cm

def massper2molper(xmass: float) -> float:
    # xmol: Mass% of CO2
    mco2 = 44.0
    mh2o = 18.0
    return mh2o*xmass / (mco2-(mco2-mh2o)*xmass)

BHP_MAX = 1200

TEMPC_INJ_FLUID = 10.0  # ℃

# microbubble storage
P_MB = 6.0  # MPa
XCO2_MB = massper2molper(0.09)  # mol%
INJ_RATE_MB = 24.0 * 24.0  # 24m3/h=576m3/day 
INJ_UNIT_MB = "RATE"

# super-critical state
P_SC = 9.0     # MPa
XCO2_SC = 1.0  # mol%
INJ_RATE_SC = 2.0 * 24.0  # 2t/h=48t/day
INJ_UNIT_SC = "MASS"

# 幌内CCS基地
HEAD_LATLNG_HORONAI_MB = (43.246391, 141.926022)
HEAD_LATLNG_HORONAI_SC = (43.246328, 141.908813)
# 奔別CCS基地
HEAD_LATLNG_POMBETSU_MB = (43.268550, 141.950741)
HEAD_LATLNG_POMBETSU_SC = (43.268550, 141.950741)

# upper limit of well completion
DEPTH_ULIM_MB = 400.0
DEPTH_ULIM_SC = 800.0

if __name__ == "__main__":
    pass