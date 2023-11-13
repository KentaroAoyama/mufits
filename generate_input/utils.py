from typing import List, Tuple
from pathlib import Path
from os import PathLike, makedirs
from math import log10

import numpy as np
from matplotlib import pyplot as plt
import vtk

from constants import DXYZ, P_GROUND, P_GRAD_AIR, Kh
from params import PARAMS_VTK


def calc_ijk(m: int, nx: int, ny: int) -> Tuple[int]:
    q, i = divmod(m, nx)
    k, j = divmod(q, ny)
    return i, j, k


def calc_m(i: int, j: int, k: int, nx: int, ny: int) -> int:
    return nx * ny * k + nx * j + i

def stack_from_0(_ls: List[float]) -> List[float]:
    c_ls = []
    for i, _d in enumerate(_ls):
        if len(c_ls) == 0:
            c_ls.append(abs(_d) * 0.5)
            continue
        c_ls.append(c_ls[-1] + _ls[i - 1] * 0.5 + abs(_d) * 0.5)
    return c_ls


def stack_from_center(_ls: List[float]) -> List[float]:
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
    return 10.0 ** (-14.0 - 3.2 * log10(z / 1000.0)) / 9.869233 * 1.0e16

def condition_to_dir(
    base_dir: PathLike, tempe_src: float, comp1t: float, inj_rate: float, pearm: float
) -> PathLike:
    base_dir = Path(base_dir)
    return base_dir.joinpath(f"{tempe_src}_{comp1t}_{inj_rate}_{pearm}")

def calc_press_air(elv: float) -> float:
    """Calculate air pressure in Pa

    Args:
        elv (float): Elevation (m)

    Returns:
        float: Air pressure (Pa)
    """
    return (P_GROUND - P_GRAD_AIR * elv) * 1.0e6

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


def plt_topo(
    topo_ls: List, latc_ls: List, lngc_ls: List, nxyz: Tuple[int], savedir: PathLike
):
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

def plt_any_val(val_ls: List, x_ls: List, y_ls: List, nxyz: Tuple[int], savedir: PathLike, label_name: str, ax="Y") -> None:
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
        np.array(x_ls), np.array(y_ls),
    )
    makedirs(dirpth, exist_ok=True)
    for i, val2d in enumerate(val_3d):
        fpth = dirpth.joinpath(f"{i}.png")
        fig, ax = plt.subplots()
        mappable = ax.pcolormesh(grid_x, grid_y, val2d)
        pp = fig.colorbar(mappable, ax=ax, orientation="vertical")
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
    return perm / 9.869233 * 1.0e16

def mdarcy2si(perm: float) -> float:
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
    print(calc_xco2_rain(1.0e5, 3.8e-4))
    pass
