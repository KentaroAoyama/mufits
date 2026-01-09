from typing import Optional, Dict
from os import PathLike
from math import sqrt, pow
import numpy as np
import pandas as pd
from LoopStructural import GeologicalModel, BoundingBox
from pyproj import Transformer
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from constants import (GEO_CRS,
                       RECT_CRS,
                       ORIGIN,
                       GeologicalData,
                       XY,
                       XYZ,
                       BOTTOM_VERTICAL_SECTION,
                       XYZPTH)

from utils import load_dem

def interp_layer(geodata: list[GeologicalData]) -> GeologicalModel:
    # interpolate geological layer by LoopStructual3D
    # generate BBOX
    xmin = min([min([min(layer.BOTTOM.X), min(layer.TOP.X)]) for layer in geodata])
    xmax = max([max([max(layer.BOTTOM.X), max(layer.TOP.X)]) for layer in geodata])
    ymin = min([min([min(layer.BOTTOM.Y), min(layer.TOP.Y)]) for layer in geodata])
    ymax = max([max([max(layer.BOTTOM.Y), max(layer.TOP.Y)]) for layer in geodata])
    zmin = min([min([min(layer.BOTTOM.Z), min(layer.TOP.Z)]) for layer in geodata])
    zmax = max([max([max(layer.BOTTOM.Z), max(layer.TOP.Z)]) for layer in geodata])
    bbox = BoundingBox(np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax]))
    model = GeologicalModel(bbox)

    # generate data
    dct = dict()
    val = 0
    name_ls = []
    for layer in geodata:
        dct.setdefault("X", []).extend(layer.BOTTOM.X)
        dct.setdefault("Y", []).extend(layer.BOTTOM.Y)
        dct.setdefault("Z", []).extend(layer.BOTTOM.Z)
        dct.setdefault("val", []).extend([val]*len(layer.BOTTOM.X))
        dct.setdefault("feature_name", []).extend([layer.LAYER_NAME]*len(layer.BOTTOM.X))
        val += 1
        dct.setdefault("X", []).extend(layer.TOP.X)
        dct.setdefault("Y", []).extend(layer.TOP.Y)
        dct.setdefault("Z", []).extend(layer.TOP.Z)
        dct.setdefault("val", []).extend([val]*len(layer.TOP.X))
        dct.setdefault("feature_name", []).extend([layer.LAYER_NAME]*len(layer.TOP.X))
        name_ls.append(layer.LAYER_NAME)
        val += 1

    data = pd.DataFrame(data=dct)
    model.data = data
    for name in name_ls:
        model.create_and_add_foliation(name)

    # # plot
    # x = np.linspace(xmin, xmax, 100)
    # z = np.linspace(zmin, zmax, 100)
    # xx, zz = np.meshgrid(x, z)
    # yy = np.zeros_like(xx)
    # for y in np.linspace(ymin, ymax, 20):
    #     fig, ax = plt.subplots()
    #     yy[:] = y
    #     for name in name_ls:
    #         vals = model.evaluate_feature_value(
    #         name, np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    #         )
    #         mappable = ax.contourf(vals.reshape(100,100),extent=(xmin,xmax,zmin,zmax))
    #         ax.contour(vals.reshape((100, 100)), [0, 1], extent=(xmin,xmax,zmin,zmax))
    #         fig.colorbar(mappable=mappable)
    #         fig.savefig(f"{name}_{y}.png")

    return model

def calc_total_distance(xy: XY) -> float:
    dist_tot = 0.0
    x0, y0 = None, None
    for _xy in xy:
        x1,y1 = _xy
        if x0 is None and y0 is None:
            x0 = x1
            y0 = y1
            continue
        d = sqrt(pow(x1-x0,2)+pow(y1-y0,2))
        dist_tot += d
        x0 = x1
        y0 = y1
    return dist_tot

def calc_layer_xyz(
        cross_pts: XY,
        zbounds: tuple[float, float],
        vertical_data: XY,
        ) -> XYZ:
    # calculate (x,y,z) coordinates from geological sectional data
    # lineとvertical_dataのマッピングを作成 (e.g., (0,0.5,1)↔ ((140,40), (140.5,40.5), (141,41)) )
    mapping: Dict[tuple[float, float], 
                  tuple[tuple[float, float], tuple[float, float]]]=dict()
    dist_tot = calc_total_distance(cross_pts)
    x0, y0 = None, None
    d0, d1 = 0.0, 0.0
    for xy in cross_pts:
        x1, y1 = xy
        if x0 is None and y0 is None:
            x0 = x1
            y0 = y1
            continue
        d1 += sqrt(pow(x1-x0,2)+pow(y1-y0,2))
        mapping.setdefault((d0/dist_tot, d1/dist_tot),
                           ((x0, y0),
                           (x1-x0,y1-y0)))
        x0 = x1
        y0 = y1
        d0 = d1
    assert len(mapping) != 0

    # convert vertical sectional data to XYZ
    x_ls: list[float] = []
    y_ls: list[float] = []
    z_ls: list[float] = []
    for xzn in vertical_data:
        xn, zn = xzn
        # find look-up-table
        find_lookup = False
        for xn_range, start_scale in mapping.items():
            if xn_range[0] <= xn <= xn_range[1]:
                find_lookup = True
                break
        assert find_lookup
        # regularize xn in [0,1]
        xn /= xn_range[1] - xn_range[0]
        x0, y0 = start_scale[0]
        sx, sy = start_scale[1]
        x_ls.append(x0+sx*xn)
        y_ls.append(y0+sy*xn)
        z_ls.append((zbounds[1]-zbounds[0])*zn)

    return XYZ(X=x_ls,Y=y_ls,Z=z_ls)

def split_csv_line(s: str) -> list[float]:
    s = s.replace("\n", "")
    s = s.replace(" ", "")
    str_ls = s.split(",")
    float_ls = [float(v) for v in str_ls]
    return float_ls

def load_cross_points(fpth: PathLike,
                      origin: tuple[float, float],
                      transformer: Optional[Transformer]=None,
                      ) -> XY:
    x_ls: list[float] = []
    y_ls: list[float] = []
    with open(fpth, "r") as f:
        for l in f.readlines():
            float_ls = split_csv_line(l)
            x_ls.append(float_ls[0])
            y_ls.append(float_ls[1])
    if transformer is not None:
        x_ls, y_ls = transformer.transform(x_ls, y_ls)
    x_ls = [x-origin[0] for x in x_ls]
    y_ls = [y-origin[1] for y in y_ls]
    return XY(X=x_ls, Y=y_ls)

def load_highest_elevation(fpth: PathLike) -> float:
    with open(fpth, "r") as f:
        l = f.readline()
        l = l.replace("\n", "")
    return float(l)

def load_vertical_section_data(fpth: PathLike) -> XY:
    x_ls: list[float] = []
    y_ls: list[float] = []
    with open(fpth, "r") as f:
        for l in f.readlines():
            float_ls = split_csv_line(l)
            x_ls.append(float_ls[0])
            y_ls.append(float_ls[1])
    return XY(X=x_ls,Y=y_ls)

def load_vertical_data(datadir: PathLike,
                       line_names: list[str],
                       layer_name: str) -> GeologicalData:
    # 石炭地質図, 地質図幅をアノテーションしたデータを読み込むと想定
    transformer = Transformer.from_crs(GEO_CRS, RECT_CRS, always_xy=True)
    xy0 = transformer.transform(ORIGIN[1], ORIGIN[0]) # origin
    datadir = Path(datadir)
    bottom = XYZ(X=[],Y=[],Z=[])
    top = XYZ(X=[],Y=[],Z=[])
    for line_name in line_names:
        file_num = 0
        cp_xy: XY = None
        elv_highest: float = None
        bottom_xy: XY = None
        top_xy: XY = None
        for fpth in datadir.glob(f"**/{line_name}*"):
            file_num += 1
            assert file_num <= 4
            if "cross" in fpth.name:
                cp_xy = load_cross_points(fpth, xy0, transformer)
            if "highest" in fpth.name:
                elv_highest = load_highest_elevation(fpth)
            if "0.csv" in fpth.name:
                bottom_xy = load_vertical_section_data(fpth)
            if "1.csv" in fpth.name:
                top_xy = load_vertical_section_data(fpth)
        assert None not in (cp_xy, elv_highest, bottom_xy, top_xy)
        bottom_tmp = calc_layer_xyz(cp_xy,
                                    (BOTTOM_VERTICAL_SECTION, elv_highest),
                                    bottom_xy)
        top_tmp = calc_layer_xyz(cp_xy,
                                 (BOTTOM_VERTICAL_SECTION, elv_highest),
                                 top_xy)
        bottom.merge(bottom_tmp)
        top.merge(top_tmp)

    return GeologicalData(LAYER_NAME=layer_name,
                          BOTTOM=bottom,
                          TOP=top)

def get_elevation(terrain_data, coordinates):
    """
    terrain_data: numpy array of shape (N, 3) where each row is (X, Y, Z)
    coordinates: numpy array of shape (M, 2) where each row is (X, Y)
    
    Returns: numpy array of shape (M,) with the interpolated elevations.
    """
    # terrain_dataをX, Y, Zに分ける
    points = terrain_data[:, :2]
    elevations = terrain_data[:, 2]
    
    # 座標に基づいて標高を補間する
    interpolated_elevations = griddata(points, elevations, coordinates, method='nearest')
    return interpolated_elevations

def load_horizontal_data(bottom_pth: PathLike,
                         top_pth: PathLike,
                         layer_name: str) -> GeologicalData:
    transformer = Transformer.from_crs(GEO_CRS, RECT_CRS, always_xy=True)
    x0, y0 = transformer.transform(ORIGIN[1], ORIGIN[0]) # origin
    # bottom
    with open(bottom_pth, "r") as f:
        lng_ls, lat_ls = [], []
        for l in f.readlines():
            lng, lat = split_csv_line(l)
            lng_ls.append(lng)
            lat_ls.append(lat)
        xb, yb = transformer.transform(lng_ls, lat_ls)
        xyb = np.stack((xb, yb), axis=1)
    # top
    with open(top_pth, "r") as f:
        lng_ls, lat_ls = [], []
        for l in f.readlines():
            lng, lat = split_csv_line(l)
            lng_ls.append(lng)
            lat_ls.append(lat)
        xt, yt = transformer.transform(lng_ls, lat_ls)
        xyt = np.stack((xt, yt), axis=1)
    
    xtopo_ls, ytopo_ls, ztopo_ls = load_dem(XYZPTH)
    xyztopo_arr = np.stack((xtopo_ls, ytopo_ls, ztopo_ls), axis=1)
    elvb = get_elevation(xyztopo_arr, xyb)
    elvt = get_elevation(xyztopo_arr, xyt)

    bottom_xyz = XYZ(X=[x-x0 for x in xb],
                     Y=[y-y0 for y in yb],
                     Z=elvb.tolist())
    top_xyz = XYZ(X=[x-x0 for x in xt],
                  Y=[y-y0 for y in yt],
                  Z=elvt.tolist())
    data = GeologicalData(LAYER_NAME=layer_name,
                   BOTTOM=bottom_xyz,
                   TOP=top_xyz)
    return data

from pathlib import Path
def plt_scatter():
    df: pd.DataFrame = None
    for fpth in Path("./geology").iterdir():
        if not fpth.is_file():
            continue
        _df = pd.read_csv(fpth, header=None)
        if df is None:
            df = _df
        else:
            df = pd.concat([df, _df])
    fig, ax = plt.subplots()
    mappable = ax.scatter(df[0], df[1], s=0.1)
    fig.colorbar(mappable=mappable)
    fig.savefig("tmp.png")

def plt_data(pth_ls):
    fig, ax = plt.subplots()
    for pth in pth_ls:
        pth = Path(pth)
        x_ls, y_ls = [], []
        with open(pth, "r") as f:
            for l in f.readlines():
                x, y = split_csv_line(l)
                x_ls.append(x)
                y_ls.append(y)
        ax.scatter(x_ls, y_ls)
    ax.set_aspect("equal")
    fig.savefig("tmp.png")

if __name__ == "__main__":
    # plt_data([
    #     "./geology/ic0_1.csv",
    #     "./geology/ic1_1.csv",
    #     ])
    pass