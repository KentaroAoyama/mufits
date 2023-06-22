#!/usr/bin/env python
# coding: utf-8

"""Generate input file for MUFITS (Afanasyev, 2012)
"""
from functools import partial
from typing import List, Tuple, TextIO, Callable
from pathlib import Path
from os import PathLike
import re

import pandas as pd

nx, ny, nz = 10, 2, 5

# TODO: 天水の量をtimestepごとに調整する
# TODO: 入力：グリッド, モデルパラメータ（timestepごとの流入量, 浸透率, CO2分率とする）
# TODO: generate using 10km × 10km DEM (10m)

gx = [100., 50., 50., 50., 50., 50., 50., 50., 50., 100.]
gy = [10., 10.]
gz = [35., 10., 10., 10., 35.]

def __get_two_floatstring(_str: str) -> Tuple[float]:
    result = re.search("\d+(\.\d+)?\s+\d+(\.\d+)?", _str)
    lat_lng_str = result.group()
    _lat, _lng = lat_lng_str.split(" ")
    return float(_lat), float(_lng)

act_ls = []
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if k == 0:
                act_ls.append(2)
            else:
                act_ls.append(1)
assert len(act_ls) == nx * ny * nz


def load_dem(pth_dem: PathLike) -> Tuple[List, List, List]:
    base_dir = Path(pth_dem)
    lat_ls, lng_ls, elv_ls = [], [], []
    for pth in base_dir.glob("**/*"):
        if pth.suffix != ".xml":
            continue
        with open(pth, "r", encoding="utf-8") as f:
            # parameters of DEM
            flag_elv = False
            lat_range, lng_range = [None, None], [None, None]
            n_lat, n_lng = None, None
            lat_step, lng_step = None, None
            cou_max = None
            # increment
            cou: int = 0
            for line in f.readlines():
                if flag_elv:
                    result = re.search("\d+(\.\d+)?", line)
                    a, b = divmod(cou, n_lat)
                    lat_ls.append(lat_range[0] + a * lat_step)
                    lng_ls.append(lng_range[0] + b * lng_step)
                    elv_ls.append(float(result.group()))
                    cou += 1
                    if cou > cou_max:
                        break
                if "<gml:lowerCorner>" in line:
                    _lat, _lng = __get_two_floatstring(line)
                    lat_range[0] = _lat
                    lng_range[0] = _lng
                if "<gml:upperCorner>" in line:
                    _lat, _lng = __get_two_floatstring(line)
                    lat_range[1] = _lat
                    lng_range[1] = _lng
                if "<gml:high>" in line:
                    _nlat, _nlng = __get_two_floatstring(line)
                    n_lat = int(_nlat)
                    n_lng = int(_nlng)
                if "<gml:tupleList>" in line:
                    assert None not in lat_range
                    assert None not in lng_range
                    assert n_lat is not None
                    assert n_lng is not None
                    lat_step = (lat_range[1] - lat_range[0]) / n_lat
                    lng_step = (lng_range[1] - lng_range[0]) / n_lng
                    cou_max = n_lat * n_lng
                    flag_elv = True
    return lat_ls, lng_ls, elv_ls


def load_sea_dem(pth_seadem: PathLike) -> Tuple[List, List, List]:
    base_dir = Path(pth_seadem)
    lat_ls, lng_ls, elv_ls = [], [], []
    for pth in base_dir.glob("**/*"):
        _df = pd.read_csv(pth, sep='\s+', header=None)
        lat_ls.extend(_df[1].to_list())
        lng_ls.extend(_df[2].to_list())
        elv_ls.extend(_df[3].to_list())
    return lat_ls, lng_ls, elv_ls


def generate_mesh(pth_dem: PathLike, pth_seadem: PathLike, crs_rect: str, dxv: List[float], dyv: List[float], dzv: List[float], latlng_center: Tuple[float, float], top: float):
    # TODO: docstring & output type

    # load DEM
    # lat_topo_ls, lng_topo_ls, elv_topo_ls = load_dem(pth_dem)
    
    # load marine & lake topology
    
    # TODO:
    # convert to rect crs

    # グリッド中心の座標を計算する

    # グリッドごとに、平均の標高、海底地形、湖底地形を計算

    # gx, gy, gz, actnumを生成する
    pass


def write(_f: TextIO, _string: str):
    _string += "\n"
    _f.write(_string)


def generate_input(gx, gy, gz, act_ls, fpth: str):

    with open(fpth, "w", encoding="utf-8") as _f:
        
        __write: Callable = partial(write, _f)

        # RUNSPEC
        __write("RUNSPEC   ################### RUNSPEC section begins here ######################")
        __write("") #\n
        
        # HCROCK
        __write("HCROCK                                  We enable heat conduction.")
        __write("") #\n
        
        # GRIDUNIT
        __write("GRIDUNIT")
        __write("  'METRES' /")
        __write("") #\n
        
        # GRID
        __write("GRID      ##################### GRID section begins here #######################")
        __write("") #\n
        
        # MAKE
        __write("          The grid is specified within brackets MAKE-ENDMAKE   ")
        __write("MAKE      <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        __write("-- cartesian                            We select Cartesia gridding")
        __write("--    grid     nx  ny  nz               option and specify the number of")
        __write(f"      CART     {len(gx)}   {len(gy)}   {len(gz)}  /            grid blocks along every axis.")
        __write("") #\n
        
        # XYZBOUND
        __write("XYZBOUND")
        __write("-- xmin-xmax  ymin-ymax  zmin-zmax     we specify the domain extent.")
        __write(f"    0   {sum(gx)}   0    {sum(gy)}     0   {sum(gz)}   /  It is [0,{sum(gx)}]*[0,{sum(gy)}]*[0,{sum(gz)}] meters.")
        __write("") #\n
        
        # DXV
        __write("DXV")
        _str = ""
        for _x in gx:
            _str += str(_x) + "  "
        _str += " /"
        __write(_str)
        __write("")

        # DYV
        __write("DYV")
        _str = ""
        for _y in gy:
            _str += str(_y) + "  "
        _str += " /"
        __write(_str)
        __write("")

        # DZV
        __write("DZV")
        _str = ""
        for _z in gz:
            _str += str(_z) + "  "
        _str += " /"
        __write(_str)
        __write("")
        
        # ACTNUM
        __write("ACTNUM")
        _str: str = ""
        for _a in act_ls:
            _str += str(_a) + "  "
        _str += "  /"
        __write(_str)
        __write("") #\n

        # BOUNDARY
        # TODO
        __write("BOUNDARY                               We define the boundaries:")
        __write("   102   1 10 1 2 5 5 'K+' 5* INFTHIN 4* 2 2 /    the bottom bound. marked as FLUXNUM=102/")
        __write("/")
        __write("")

        # SRCSPECG
        # TODO
        __write("SRCSPECG")
        __write(" ’MAGMASRC’ 5 1 5 /")
        __write("/")
        __write("")

        # ENDMAKE
        __write("ENDMAKE   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        __write("") #\n

        # EQUALREG
        # TODO
        __write("EQUALREG                              We specify uniform distributions:")
        __write("   PORO      0.25 ROCKNUM 1 /           porosity = 0.25")
        __write("   PERMX     100  ROCKNUM 1 /                     X-permeability = 100 mD")
        __write("   PERMY     100  ROCKNUM 1 /                     Y-permeability = 100 mD")
        __write("   PERMZ     100  ROCKNUM 1 /                     Z-permeability = 100 mD")
        __write("   HCONDCFX  2.   ROCKNUM 1 /                     X-Heat cond. coeff. = 2 W/m/K")
        __write("   HCONDCFY  2.   ROCKNUM 1 /                     Y-Heat cond. coeff. = 2 W/m/K")
        __write("   HCONDCFZ  2.   ROCKNUM 1 /                     Z-Heat cond. coeff. = 2 W/m/K")
        __write("/")
        __write("") #\n

        # RPTGRID
        __write("RPTGRID                               We define the output form the GRID sect.")
        __write("  PORO PERMX PERMZ /")
        __write("") #\n

        # PROPS
        __write("PROPS     ####################### PROPS section begins here ####################")
        __write("") #\n

        # ROCK
        __write("          Rock properties are specified within brackets ROCK-ENDROCK")
        __write("ROCK      <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        __write("  1  /")
        __write("") #\n

        # TODO
        __write("ROCKDH                                  We specify that")
        __write("  2900  0.84 /                          rock density is 2900 kg/m3, rock heat capacity is 0.84 kJ/kg/K")
        __write("") #\n

        # ENDROCK
        __write("ENDROCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        __write("") #\n

        # LOADEOS
        __write("LOADEOS                                 We load the EOS file.")
        __write("   './CO2H2O_V3.0.EOS' /")
        __write("") #\n

        # SAT
        # TODO: confirm
        __write("         The relative permeabilities are specified within brackets SAT-ENDSAT")
        __write("SAT      <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        __write("   /")
        __write("") #\n

        # SATTAB
        __write("SATTAB")
        __write("    0.0    0.0   1.0  /           krliq=sliq^3")
        __write("    0.1    0.001 0.81 /           krgas=(1-sliq)^2")
        __write("    0.2    0.008 0.64 /")
        __write("    0.4    0.064 0.36 /")
        __write("    0.6    0.216 0.16 /")
        __write("    0.8    0.512 0.04 /")
        __write("    0.9    0.729 0.01 /")
        __write("    1.0    1.0   0.0  /")
        __write("/")
        __write("") #\n

        # ENDSAT
        __write("ENDSAT    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        __write("") #\n

        # PHASES
        """The keyword PHASES defines new MNEMONICS for data output. This keyword
        is followed by tabular data and every line of these data defines an
        individual phase. The 1st column is the phase name (not more
        than 4 characters). The following data items define typical
        thermodynamic parameters of the phase. The 2nd column is
        pressure, the 3rd is the enthalpy, the 4th and 5th are the molar
        composition of binary mixture.
        We define two phases. The 1st is the liquid water (LH2O). The 2nd is
        the supercritical CO2 phase.
        """
        __write("PHASES                               We define phases for output:")
        __write("-- name   pres  enth  CO2 H2O")
        __write("--       [MPa] [kJ/mol] CO2 H2O")
        __write("   LIQ    1.0    5.0  0.0 1.0 /        - liquid water")
        __write("   GAS    1.0   60.0  0.0 1.0 /        - water vapor")
        __write("/")
        __write("") #\n

        # INIT
        __write("INIT      ####################### INIT section begins here #####################")
        __write("") #\n

        # REGALL
        __write("REGALL      We enable application of the following two keywords both to domain grid blocks and boundary grid blocks.")
        __write("") #\n

        # ECHO
        __write("ECHO      Enables more information output in the LOG-file")
        __write("") #\n

        # OPERAREG
        __write("OPERAREG")
        __write("   PRES DEPTH  SATNUM  1 MULTA  0.2  0.0099 /          PRES=0.2+0.0099*DEPTH")
        __write("/")
        __write("") #\n

        # EQUALREG
        # TODO
        __write("EQUALREG")
        __write("   TEMPC   20  ROCKNUM 1   /     The initial temperature is 20 C  ")
        __write("   COMP1T  0.0             /     No CO2 is present ")
        __write("   TEMPC   200 FLUXNUM 102 /     The temperature of the bottom boundary is 200 C")
        __write("   PERMZ   0.0 FLUXNUM 102 /")
        __write("/")
        __write("") #\n

        # EQUALNAM
        __write("EQUALNAM")
        __write("  PRES 10. ’MAGMASRC’ /")
        __write("  TEMPC 400. ’MAGMASRC’/")
        __write("  COMP1T 0.5 ’MAGMASRC’/")
        __write("/")
        __write("") #\n
        __write("") #\n

        # RPTSUM
        __write("RPTSUM")
        __write("   PRES TEMPC PHST SAT#LIQ SAT#GAS /  We specify the properties saved at every report time.")
        __write("") #\n

        # SCHEDULE
        __write("SCHEDULE   #################### SCHEDULE section begins here ####################")
        __write("") #\n

        # SRCINJE
        __write("SRCINJE")
        __write("  ’MAGMASRC’ MASS 1* 25. 1* 500. /")
        __write("/")
        __write("") #\n

        # TUNING
        # TODO:
        __write("TUNING                        We specify that the maximal timestep is 1000 days and the initial timestep is 0.1 days.")
        __write("    1* 1000 0.1 /")
        __write("") #\n

        # TSTEP
        # TODO:
        __write("TSTEP                         We advance simulation to 100000 days reporting distributions every 1000 days.")
        __write("10*10000 /")
        __write("") #\n
        
        # REPORTS
        __write("REPORTS")
        __write("   CONV MATBAL LINSOL  /")
        __write("") #\n

        # POST
        __write("POST      ####################### POST section begins here ######################")
        __write("") #\n

        # CONVERT
        __write("CONVERT                                 We convert the output to ParaView compatible format.")
        __write("") #\n

        # POSTSRC
        __write("POSTSRC")
        __write("  MAGMASRC /")
        __write("/")
        __write("") #\n

        # END
        __write("END       #######################################################################")

if __name__ == "__main__":
    # load_dem("./dem")
    load_sea_dem("./seadem")
