from typing import List, Tuple
from pathlib import Path
from os import PathLike, makedirs

import numpy as np
from matplotlib import pyplot as plt
import vtk


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
    # .vtuファイルを読み込む
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_file_path)
    reader.Update()

    cell2point = vtk.vtkCellDataToPointData()
    cell2point.SetInputData(reader.GetOutput())
    cell2point.Update()

    # データを取得
    data = cell2point.GetOutput()
    points = data.GetPoints()
    num_points = points.GetNumberOfPoints()
    num_arrays = data.GetPointData().GetNumberOfArrays()

    # 座標情報をNumPy配列に変換
    coordinates = np.array([points.GetPoint(i) for i in range(num_points)])

    # 各配列をNumPy配列に変換
    arrays = {}
    for i in range(num_arrays):
        array = data.GetPointData().GetArray(i)
        array_name = array.GetName()
        array_data = np.array([array.GetTuple(i) for i in range(num_points)])
        arrays[array_name] = array_data
    return coordinates, arrays


if __name__ == "__main__":
    coordinates, arrays = vtu_to_numpy("./test/tmp2.0000.vtu")
    # parse_vtu_file("./test/tmp2.0000.vtu")
    # plt_vtk("./test/tmp2.0000.vtu")
