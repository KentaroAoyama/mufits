from typing import List, Tuple
from pathlib import Path
from os import PathLike, makedirs

import numpy as np
from matplotlib import pyplot as plt
import vtk

from params import DXYZ, VLIM


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


def plt_vtk(
    vtkpth: PathLike, xlim: Tuple = None, ylim: Tuple = None, zlim: Tuple = None
) -> None:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtkpth)
    reader.Update()  # Needed because of GetScalarRange
    vtk_data = reader.GetOutput()
    cell2point = vtk.vtkCellDataToPointData()
    cell2point.SetInputData(reader.GetOutput())
    cell2point.Update()
    coord = vtk.utils.numpy_support.vtk_to_numpy(
        cell2point.GetOutput().GetPoints().GetData()
    )
    # vtk_array = vtk_data.GetPointData().GetArray("PHST")
    # # vtk_array_shape = vtk_array.GetDimensions()[::-1]
    # numpy_array = np.array(vtk_array)
    # print(numpy_array)
    # # numpy_array = numpy_array.reshape(vtk_array_shape, order="F")v


import xml.etree.ElementTree as ET


def parse_vtu_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # データ配列を格納するリスト
    data_arrays = []

    # データ配列の名前を取得
    for data_array in root.findall(".//DataArray"):
        name = data_array.attrib.get("Name")
        data = data_array.text.strip().split()
        data = np.array(data, dtype=float)
        data_arrays.append((name, data))
    print(data_arrays)
    return data_arrays


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
    coordinates, arrays = vtu_to_numpy("./test/tmp2.0040.vtu")
    print(coordinates.shape)
    result_dir = Path("./result/40")
    print(arrays.keys())
    for key, val in arrays.items():
        print("===")
        print(key)
        outdir = result_dir.joinpath(key)
        vlim = VLIM[key]
        plt_result(val, coordinates, vlim[0], vlim[1], outdir)
