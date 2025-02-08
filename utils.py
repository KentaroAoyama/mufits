from typing import List, Tuple, Dict
from pathlib import Path
from os import PathLike, makedirs
from math import log10, log
from copy import deepcopy

import numpy as np
from scipy.integrate import quad
import pickle
from matplotlib import pyplot as plt
# import vtk

from constants import (
    DXYZ,
    P_GROUND,
    P_GRAD_AIR,
    Kh,
    DXYZ,
    RAIN_AMOUNT,
    EVAP_AMOUNT,
    RIVERS,
    MIA,
    MIB,
)


def calc_ijk(m: int, nx: int, ny: int) -> Tuple[int]:
    """Convert global index m to indecies in X-, Y-, and Z-direction (i, j, k)

    Args:
        m (int): Global index which indicates block id in 3d space.
        nx (int): Total grid number in X-axis
        ny (int): Total grid number in Y-axis

    Returns:
        Tuple[int]: Indecies in X-, Y-, and Z-direction (i, j, k)
    """
    q, i = divmod(m, nx)
    k, j = divmod(q, ny)
    return i, j, k


def calc_m(i: int, j: int, k: int, nx: int, ny: int) -> int:
    """Calculate global index m from indeciex in X-, Y-, and Z-direction (i, j, k)

    Args:
        i (int): Index in X-direction
        j (int): Index in Y-direction
        k (int): Index in Z-direction
        nx (int): Total grid number in X-axis
        ny (int): Total grid number in Y-axis

    Returns:
        int: Global index which indicates block id in 3d space.
    """
    return nx * ny * k + nx * j + i


def stack_from_0(_ls: List[float]) -> List[float]:
    """Generate a list containing the center coordinates of the grid from
    the list containing the grid spacing. The element with index 0 is the origin.

    Args:
        _ls (List[float]): 1d list containing the grid spacing

    Returns:
        List[float]: 1d list containing the center coordinates of the grid
    """
    c_ls = []
    for i, _d in enumerate(_ls):
        if len(c_ls) == 0:
            c_ls.append(abs(_d) * 0.5)
            continue
        c_ls.append(c_ls[-1] + _ls[i - 1] * 0.5 + abs(_d) * 0.5)
    return c_ls


def stack_from_center(_ls: List[float]) -> List[float]:
    """Generate a list containing the center coordinates of the grid from
    the list containing the grid spacing. The element whose index is in 
    the center is assumed to be the origin.

    Args:
        _ls (List[float]): 1d list containing the grid spacing

    Returns:
        List[float]: 1d list containing the center coordinates of the grid
    """
    n = len(_ls)
    if divmod(n, 2)[1] == 0:
        sum_left = -sum(_ls[: int(n * 0.5)])
    else:
        _lhalf = int(n * 0.5) + 1
        sum_left = -sum(_ls[:_lhalf]) + 0.5 * _ls[_lhalf]
    c_ls: List = []
    for _d in _ls:
        sum_left += _d * 0.5
        c_ls.append(sum_left)
        sum_left += _d * 0.5
    return c_ls


def calc_k_z(z: float) -> float:
    """Permeability with depth dependence

    Reference:
        Manning and Ingebritsen (1999) https://doi.org/10.1029/1998RG900002

    Args:
        z (float): Depth from earth surface (m)

    Returns:
        float: Permeability (mD)
    """
    return 10.0 ** (MIA + MIB * log10(z / 1000.0)) / 9.869233 * 1.0e16


def _kh_i(z: float, A: float, B: float) -> float:
    return 0.5 * A * B * z**2


def calc_kh(z0: float, z1: float) -> float:
    """Calculate combined permeability in horizontal axis assuming the
    depth dependence of permeability reported in Manning and Ingebritsen (1999)

    Args:
        z0 (float): Elevation of lower limit of the layer (m)
        z1 (float): Elevation of upper limit of the layer (m)

    Returns:
        float: Combined permeability (m^2)
    """
    assert z1 > z0
    # A = 10.0**MIA
    # B = 10.0**MIB
    # return (_kh_i(z1, A, B) - _kh_i(z0, A, B)) / (z1 - z0) / 9.869233 * 1.0e16
    return quad(calc_k_z, z0, z1)[0] / (z1 - z0)


def _kz_inv(z: float) -> float:
    assert z > 0.0
    return 1.0 / calc_k_z(z)


def _kv_i(z: float, A: float, B: float) -> float:
    return log(z) / A * B


def calc_kv(z0: float, z1: float) -> float:
    """Calculate combined permeability in vertical axis assuming the
    depth dependence of permeability reported in Manning and Ingebritsen (1999)

    Args:
        z0 (float): Elevation of lower limit of the layer (m)
        z1 (float): Elevation of upper limit of the layer (m)

    Returns:
        float: Combined permeability (m^2)
    """
    assert z1 > z0
    # A = 10.0**MIA
    # B = 10.0**MIB
    # bottom = (_kv_i(z1, A, B) - _kv_i(z0, A, B)) / (z1 - z0)
    # return 1.0 / bottom /  9.869233 * 1.0e16
    return (z1 - z0) / quad(_kz_inv, z0, z1)[0]


def condition_to_dir(
    base_dir: PathLike,
    tempe_src: float,
    comp1t: float,
    inj_rate: float,
    perm: float,
    cap_scale: float = None,
    from_latest: bool = False,
    vk: bool = False,
    disperse_magmasrc: bool = False,
) -> PathLike:
    """Convert simulation condition to directory name

    Args:
        base_dir (PathLike): Parent directory of each condition 
            (.../{base_dir}/{condition})
        tempe_src (float): Source temperature (℃)
        comp1t (float): Molar fraction of CO2 in source
        inj_rate (float): Injected fluid rate from bottom (t/day)
        perm (float): Permeability ratio: conduit / host rock
        cap_scale (float): Permeability ratio: (cap rock top) / (default cap rock: 10^-17)
            If None, cap rock is not set.
        from_latest (bool): Controls whether to compute from latest 
            SUM file or not (True: compute from latest SUM file)
        vk (bool): Controls wheter to set high permeability only on 
            Z-axis or not (True: high permeability will only be set on Z-axis).
        disperse_magmasrc (bool): Controls wheter to inject magmatic
            fluid from multiple bottom blocks (True: inject from multiple blocks)

    Returns:
        PathLike: Directory path containing single simulation condition
    """
    base_dir = Path(base_dir)
    name = f"{tempe_src}_{comp1t}_{inj_rate}_{perm}"
    if cap_scale is not None:
        name += f"_{cap_scale}"
    if vk:
        name += "_v"
    if disperse_magmasrc:
        name += "_d"
    sim_dir = base_dir.joinpath(name)
    if not from_latest:
        return sim_dir
    for i in range(1, 1000):
        sim_dir_tmp = sim_dir.joinpath(f"ITER_{i}")
        if not sim_dir_tmp.exists():
            break
    return sim_dir_tmp


def unrest_dir(sim_dir: PathLike) -> Path:
    """Generate directory containing unrest results

    Args:
        sim_dir (PathLike): Directory containing steady state results

    Returns:
        Path: Directory containing unrest results
    """
    unrest_dir = Path(sim_dir).joinpath("unrest")
    makedirs(unrest_dir, exist_ok=True)
    return unrest_dir


def dir_to_condition(cond_dir: PathLike) -> Dict[str, float]:
    """Convert directory name to simulation condition
    (assumed directory name was generated by condition_to_dir)

    Args:
        cond_dir (PathLike): Directory containing simulation results

    Returns:
        Dict[str, float]: Dictionary whose keys are condition name and 
            values are parameter values.
    """
    conds_str = str(Path(cond_dir).name).split("_")
    _dct: Dict = {}
    _dct.setdefault("temp", conds_str[0])
    _dct.setdefault("comp1t", conds_str[1])
    _dct.setdefault("inj_rate", conds_str[2])
    _dct.setdefault("perm", conds_str[3])
    # cap scale
    _dct.setdefault("cap_scale", None)
    if len(conds_str) > 4:
        if isinstance(conds_str[4], float):
            _dct["cap_scale"] = conds_str[4]
    if "vk" in conds_str:
        _dct.setdefault("vk", True)
    else:
        _dct.setdefault("vk", False)
    if "d" in conds_str:
        _dct.setdefault("d", True)
    else:
        _dct.setdefault("d", False)
    return _dct


def calc_press_air(elv: float) -> float:
    """Calculate air pressure in Pa

    Args:
        elv (float): Elevation (m)

    Returns:
        float: Air pressure (MPa)
    """
    return P_GROUND - P_GRAD_AIR * elv


def calc_xco2_rain(ptol: float, xco2_air: float) -> float:
    """Calculate mole fraction of CO2 dissolved in rain by Henry's law

    Args:
        ptol (float): Total pressure in Pa (at 0m a.s.l, about 1e5 Pa)
        xco2_air (float): Mole fraction of CO2 in the air

    Returns:
        float: Mole fraction of CO2 dissolved in rain
    """
    pco2 = ptol * xco2_air
    return pco2 / Kh


def calc_infiltration(
    rain_amount: float = None, evap_amount: float = None, rivers: Dict = None
) -> float:
    """Calculate infiltration (precipitation) rate

    Args:
        rain_amount (float, optional): Total rain fall (m/year). Defaults to None.
        evap_amount (float, optional): Total evaporation (m/year). Defaults to None.
        rivers (Dict, optional): River properties. Defaults to None.

    Returns:
        float: Infiltration rate (m/days)
    """
    if rain_amount is None:
        rain_amount = deepcopy(RAIN_AMOUNT)
    if evap_amount is None:
        evap_amount = EVAP_AMOUNT
    if rivers is None:
        rivers = deepcopy(RIVERS)
    # return rain_amount * 0.0 / 365.25 #! set 0
    days = 365.25
    area = sum(DXYZ[0]) * sum(DXYZ[1])
    rain_total = area * rain_amount / days
    evap_total = area * evap_amount / days

    with open("./analyse_river/river_inout_areas.pkl", "rb") as pkf:
        rivers_inout: Dict = pickle.load(pkf)

    rivers_total: float = 0.0
    for river_name, (rarea, h) in rivers.items():
        if river_name in rivers_inout:
            inout = rivers_inout[river_name]
            in_area, out_area = inout[0], inout[1]
            rivers_total += h * in_area
            # print(river_name, (rarea - (in_area + out_area)) / rarea)
    rivers_total /= days
    return (rain_total - evap_total - rivers_total) / area


def plt_topo(
    topo_ls: List, latc_ls: List, lngc_ls: List, nxyz: Tuple[int], savedir: PathLike
):
    """Plot topology (elevation map) for debugging.

    Args:
        topo_ls (List): 1d list whose index is global index (m) and
            value is rock type index (i.e., topology index in constants.py)
        latc_ls (List): 2d list containing lalitute of grid center 
        lngc_ls (List): 2d list containing longitude of grid center 
        nxyz (Tuple[int]): Tuple containing total grid numbers for each axis 
        savedir (PathLike): Save directory
    """
    nx, ny, nz = nxyz
    topo_3d = np.zeros(shape=(nz, ny, nx)).tolist()
    for m, idx in enumerate(topo_ls):
        i, j, k = calc_ijk(m, nx, ny)
        topo_3d[k][j][i] = idx
    basedir = Path(savedir)
    if not basedir.exists():
        makedirs(basedir)
    for k, val in enumerate(topo_3d):
        fpth = basedir.joinpath(str(k))
        fig, ax = plt.subplots()
        mappable = ax.pcolormesh(lngc_ls, latc_ls, np.array(val), cmap="jet")
        fig.colorbar(mappable=mappable)
        fig.savefig(fpth, dpi=200)
        plt.clf()
        plt.close()


def plt_any_val(
    val_ls: List,
    x_ls: List,
    y_ls: List,
    nxyz: Tuple[int],
    savedir: PathLike,
    label_name: str,
    ax="Y",
    _min: float = None,
    _max: float = None,
) -> None:
    nx, ny, nz = nxyz
    ax = ax.lower()
    dirpth = Path(savedir)
    val_3d = np.zeros(shape=(nz, ny, nx))
    for m, idx in enumerate(val_ls):
        i, j, k = calc_ijk(m, nx, ny)
        val_3d[k][j][i] = idx
    # transpose
    if ax == "x":
        val_3d = np.transpose(val_3d, (2, 0, 1))
    if ax == "y":
        val_3d = np.transpose(val_3d, (1, 0, 2))
        # val_3d = np.flip(val_3d, axis=1)

    grid_x, grid_y = np.meshgrid(
        np.array(x_ls),
        np.array(y_ls),
    )
    makedirs(dirpth, exist_ok=True)
    for i, val2d in enumerate(val_3d):
        fpth = dirpth.joinpath(f"{i}.png")
        fig, ax = plt.subplots()
        mappable = ax.pcolormesh(grid_x, grid_y, val2d, vmin=_min, vmax=_max)
        pp = fig.colorbar(
            mappable,
            ax=ax,
            orientation="vertical",
        )
        pp.set_label(label_name)
        ax.set_aspect("equal")
        fig.savefig(fpth, dpi=200, bbox_inches="tight")
        plt.clf()
        plt.close()


def plt_airbounds(
    topo_ls: List,
    m_airbounds,
    latc_ls: List,
    lngc_ls: List,
    nxyz: Tuple[int],
    savedir: PathLike,
):
    nx, ny, nz = nxyz
    topo_3d = np.zeros(shape=(nz, ny, nx)).tolist()
    for m, idx in enumerate(topo_ls):
        i, j, k = calc_ijk(m, nx, ny)
        topo_3d[k][j][i] = idx

    for m in m_airbounds:
        i, j, k = calc_ijk(m, nx, ny)
        topo_3d[k][j][i] = 100

    basedir = Path(savedir)
    if not basedir.exists():
        makedirs(basedir)
    for k, val in enumerate(topo_3d):
        fpth = basedir.joinpath(str(k))
        fig, ax = plt.subplots()
        mappable = ax.pcolormesh(lngc_ls, latc_ls, np.array(val), cmap="jet")
        fig.colorbar(mappable=mappable)
        fig.savefig(fpth, dpi=200)
        plt.clf()
        plt.close()


def vtu_to_numpy(vtu_file_path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_file_path)
    reader.Update()

    cell2point = vtk.vtkCellDataToPointData()
    cell2point.SetInputData(reader.GetOutput())
    cell2point.Update()

    data = cell2point.GetOutput()
    points = data.GetPoints()
    num_points = points.GetNumberOfPoints()
    num_arrays = data.GetPointData().GetNumberOfArrays()

    coordinates = np.array([points.GetPoint(i) for i in range(num_points)])

    arrays = {}
    for i in range(num_arrays):
        array = data.GetPointData().GetArray(i)
        array_name = array.GetName()
        array_data = np.array([array.GetTuple(i) for i in range(num_points)])
        arrays[array_name] = array_data
    return coordinates, arrays


def plt_result(values, coordinates, vmin, vmax, xlim, ylim, zlim, outdir):
    x0 = 0.0
    x_ls = []
    for dx in DXYZ[0]:
        x_ls.append(x0)
        x0 += dx
    y0 = 0.0
    y_ls = []
    for dy in DXYZ[1]:
        y_ls.append(y0)
        y0 -= dy
    z0 = 0.0
    z_ls = []
    for dz in DXYZ[2]:
        z_ls.append(z0)
        z0 -= dz

    pth_outdir = Path(outdir)
    makedirs(pth_outdir, exist_ok=True)

    xc, yc, zc = coordinates.T

    # plot z
    zdir = pth_outdir.joinpath("z")
    makedirs(zdir, exist_ok=True)
    xx, yy = np.meshgrid(np.array(x_ls), np.array(y_ls))
    for z in z_ls:
        # generate value
        vv = np.zeros(shape=(len(y_ls), len(x_ls)))
        for i, x in enumerate(x_ls):
            for j, y in enumerate(y_ls):
                vv[j][i] = values[
                    np.argmin(np.square(xc - x) + np.square(yc - y) + np.square(zc - z))
                ]
        fig, ax = plt.subplots()
        mappable = ax.pcolormesh(xx, yy, vv, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
        fig.colorbar(mappable)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        fig.savefig(zdir.joinpath(f"{str(z)}.png"), dpi=200, bbox_inches="tight")
        plt.clf()
        plt.close()

    # plot x
    xdir = pth_outdir.joinpath("x")
    makedirs(xdir, exist_ok=True)
    yy, zz = np.meshgrid(np.array(y_ls), np.array(z_ls))
    for x in x_ls:
        vv = np.zeros(shape=(len(z_ls), len(y_ls)))
        for j, y in enumerate(y_ls):
            for k, z in enumerate(z_ls):
                vv[k][j] = values[
                    np.argmin(np.square(xc - x) + np.square(yc - y) + np.square(zc - z))
                ]
        fig, ax = plt.subplots()
        mappable = ax.pcolormesh(yy, zz, vv, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
        fig.colorbar(mappable)
        ax.set_xlim(*ylim)
        ax.set_ylim(*zlim)
        fig.savefig(xdir.joinpath(f"{str(x)}.png"), dpi=200, bbox_inches="tight")
        plt.clf()
        plt.close()

    # plot y
    ydir = pth_outdir.joinpath("y")
    makedirs(ydir, exist_ok=True)
    xx, zz = np.meshgrid(np.array(x_ls), np.array(z_ls))
    for y in y_ls:
        vv = np.zeros(shape=(len(z_ls), len(x_ls)))
        for i, x in enumerate(x_ls):
            for k, z in enumerate(z_ls):
                vv[k][i] = values[
                    np.argmin(np.square(xc - x) + np.square(yc - y) + np.square(zc - z))
                ]
        fig, ax = plt.subplots()
        mappable = ax.pcolormesh(xx, zz, vv, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
        fig.colorbar(mappable)
        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)
        fig.savefig(ydir.joinpath(f"{str(y)}.png"), dpi=200, bbox_inches="tight")
        plt.clf()
        plt.close()


def si2mdarcy(perm: float) -> float:
    """Convert unit of permeability in SI to darcy

    Args:
        perm (float): Permeability in SI unit

    Returns:
        float: Permeability in darcy
    """
    return perm / 9.869233 * 1.0e16


def mdarcy2si(perm: float) -> float:
    """Convert unit of permeability in darcy to SI

    Args:
        perm (float): Permeability in darcy

    Returns:
        float:  Permeability in SI unit
    """
    return perm * 9.869233 * 1.0e-16


if __name__ == "__main__":
    # print(calc_ijk(14569  - 1, 40, 40))
    # path = Path(r"E:\tarumai\200.0_0.0_1000.0_10.0").joinpath("tmp.0358.vtu")
    # coordinates, arrays = vtu_to_numpy(str(path))
    # result_dir = Path("./result/200.0_0.0_1000.0_10.0")
    # for key, val in arrays.items():
    #     if key in ("PHST", "SAT#GAS", "COMP2T"):
    #         continue
    #     print("===")
    #     print(key)
    #     outdir = result_dir.joinpath(key)
    #     vlim = PARAMS_VTK.VLIM[key]
    #     xlim = PARAMS_VTK.XLIM
    #     ylim = PARAMS_VTK.YLIM
    #     zlim = PARAMS_VTK.ZLIM
    #     plt_result(val, coordinates, vlim[0], vlim[1], xlim, ylim, zlim, outdir)
    # print(calc_k_z(25.0))
    # print(calc_xco2_rain(1.0e5, 3.8e-4))
    # print(calc_ijk(2811, 40, 40))
    # print(mdarcy2si(1.0e9))
    # print(P_GRAD_AIR)
    # print(calc_infiltration() / RAIN_AMOUNT * 365.25)
    # print(mdarcy2si(1013249.9658281449))
    # q0 = 500.0 * 500.0 * 3.53 * 1.0e-3
    # q1 = 10000.0
    # print(q0)
    # print((q0 * 10.0 + q1 * 900.0) / (q0 + q1))
    # print(mdarcy2si(calc_kv(0.1, 1150.0)))
    print(dir_to_condition("./900.0_0.1_100.0_100.0_100.0"))
    pass
