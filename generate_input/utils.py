from typing import List, Tuple
from pathlib import Path
from os import PathLike, makedirs

import numpy as np
from matplotlib import pyplot as plt
import vtk

from constants import DXYZ
from params import PARAMS_VTK


def calc_ijk(m: int, nx: int, ny: int) -> Tuple[int]:
    q, i = divmod(m, nx)
    k, j = divmod(q, ny)
    return i, j, k


def calc_m(i, j, k, nx, ny):
    return nx * ny * k + nx * j + i


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


def plt_result(values, coordinates, vmin, vmax, outdir):
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
        fig.savefig(ydir.joinpath(f"{str(y)}.png"), dpi=200, bbox_inches="tight")
        plt.clf()
        plt.close()


if __name__ == "__main__":
    coordinates, arrays = vtu_to_numpy("./test/tmp2.0010.vtu")
    print(coordinates.shape)
    result_dir = Path("./result/10_modified2")
    print(arrays.keys())
    for key, val in arrays.items():
        if key == "PHST":
            continue
        #!
        if key != "TEMPC (C)":
            continue
        print("===")
        print(key)
        outdir = result_dir.joinpath(key)
        vlim = PARAMS_VTK.VLIM[key]
        plt_result(val, coordinates, vlim[0], vlim[1], outdir)
