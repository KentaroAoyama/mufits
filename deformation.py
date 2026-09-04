# https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity_code.html
# https://jsdokken.com/dolfinx-tutorial/chapter3/neumann_dirichlet_code.html
# https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html

from typing import Tuple, List, Dict, Callable, Optional, Union
from os import PathLike, makedirs
from pathlib import Path
import pathlib
from enum import Enum, auto
from copy import deepcopy
from time import time

import numpy as np
from scipy.interpolate import interpn
import pandas as pd
import pickle
from pyproj import Transformer
from shapely.geometry.polygon import Polygon
from tqdm import tqdm

from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem import (
    Function,
    Constant,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
    form,
    assemble_scalar
)
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import basix.ufl
import ufl
from ufl import (
    Mesh,
    sym,
    grad,
    inner,
    nabla_div,
    Identity,
    Measure,
    TestFunction,
    TrialFunction,
    dot,
    dx,
    grad,
    inner,
    lhs,
    rhs,
    SpatialCoordinate,
    as_vector,
    div
)
from petsc4py import PETSc
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from constants import CACHE_DIR, DXYZ, IDX_LAND, IDX_VENT, IDX_CAP, IDX_CAPVENT, CRS_DEM, CRS_RECT, ORIGIN, POS_GNSS, OUTDIR
from utils import (stack_from_center,
                   stack_from_0,
                   load_topo_ls,
                   calc_m,
                   calc_ijk,
                   load_snap,
                   get_fpth_in_timeseries,
                   dir_to_condition)
from monitor import img2mov
from params import PARAMS

class BOUNDS(Enum):
    TOP = auto()
    LATERAL = auto()
    BOTTOM = auto()

mu = 4.0e9  # 剛性率: 気象研技術報告書 第53号 (2008) 2.5.1.2樽前山における繰り返しGPS観測
nu = 0.25   # ポアソン比: 気象研技術報告書 第53号 (2008) 2.5.1.2樽前山における繰り返しGPS観測

E = 2.0*mu*(1.0+nu)
Ks = 30.0*1.0e9 # solid bulk moduli
alpha = 1.0e-5  # thermal expansion coefficient
lambda_ = E*nu/((1.0+nu)*(1.0-2.0*nu)) # 1st Lame's constant
Kd = E / (3.0*(1.0-2.0*nu))
beta = 1.0 - Kd/Ks

IDX_SUBSURF = set([IDX_LAND, IDX_VENT, IDX_CAP, IDX_CAPVENT])
BOUNDARIES = List[Tuple[int, Callable]]

def u_d(x):
    return 0.0 * x[0]

def load_domain_info() -> Tuple[np.ndarray, np.ndarray, BOUNDARIES]:
    with open(CACHE_DIR.joinpath("domain.pkl"), "rb") as pkf:
        obj: Tuple[mesh.Mesh, BOUNDARIES] = pickle.load(pkf)
    return obj

def bounds(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return (x >= vmin) & (x <= vmax)

def generate_domain_info() -> Tuple[np.ndarray, np.ndarray, BOUNDARIES, List[int]]:
    topo_ls, _ = load_topo_ls()
    nx, ny, nz = len(DXYZ[0]), len(DXYZ[1]), len(DXYZ[2])
    # nodes
    x_ls = stack_from_center(DXYZ[0], centroid=False)
    x_ls.append(x_ls[-1]+DXYZ[0][-1])  # W to E
    y_ls = stack_from_center(DXYZ[1], centroid=False)
    y_ls.append(y_ls[-1]+DXYZ[1][-1])  # N to S
    z_ls = stack_from_0(DXYZ[2], centroid=False)
    z_ls.append(z_ls[-1]+DXYZ[2][-1])  # top to bottom
    # points
    points = []
    point_index = {}
    cells = []
    boundaries: List[Tuple[int, Callable]] = []
    boundaries.append(("lateral",BOUNDS.LATERAL.value, lambda u: np.isclose(u[1], y_ls[0])|np.isclose(u[1], y_ls[-1])|np.isclose(u[0], x_ls[0])|np.isclose(u[0], x_ls[-1])))
    boundaries.append(("bottom",BOUNDS.BOTTOM.value, lambda u: np.isclose(u[2], z_ls[-1])))
    boundaries.append(("k-",BOUNDS.TOP.value, lambda u: np.isclose(u[2], z_ls[0])))
    cells_gindex: List[int] = []  # contains global index m
    for i in range(len(x_ls)-1):
        for j in range(len(y_ls)-1):
            for k in range(len(z_ls)-1):
                m = calc_m(i,j,k,nx,ny)
                if topo_ls[m] not in IDX_SUBSURF:
                    if i >= 1:
                        mtmp = calc_m(i-1,j,k,nx,ny)
                        if topo_ls[mtmp] in IDX_SUBSURF:
                            boundaries.append(("i-", BOUNDS.TOP.value, lambda u,i=i,j=j,k=k: np.isclose(u[0], x_ls[i]) & bounds(u[1], y_ls[j], y_ls[j+1]) & bounds(u[2], z_ls[k], z_ls[k+1])))
                    if i < nx-1:
                        mtmp = calc_m(i+1,j,k,nx,ny)
                        if topo_ls[mtmp] in IDX_SUBSURF:
                            boundaries.append(("i+", BOUNDS.TOP.value, lambda u,i=i,j=j,k=k: np.isclose(u[0], x_ls[i+1]) & bounds(u[1], y_ls[j], y_ls[j+1]) & bounds(u[2], z_ls[k], z_ls[k+1])))
                    if  j >= 1:
                        mtmp = calc_m(i,j-1,k,nx,ny)
                        if topo_ls[mtmp] in IDX_SUBSURF:
                            boundaries.append(("j-",BOUNDS.TOP.value, lambda u,i=i,j=j,k=k: np.isclose(u[1], y_ls[j]) & bounds(u[0], x_ls[i], x_ls[i+1]) & bounds(u[2], z_ls[k], z_ls[k+1])))
                    if j < ny-1:
                        mtmp = calc_m(i,j+1,k,nx,ny)
                        if topo_ls[mtmp] in IDX_SUBSURF:
                            boundaries.append(("j+",BOUNDS.TOP.value, lambda u,i=i,j=j,k=k: np.isclose(u[1], y_ls[j+1]) & bounds(u[0], x_ls[i], x_ls[i+1]) & bounds(u[2], z_ls[k], z_ls[k+1])))
                    if k < nz-1:
                        mtmp = calc_m(i,j,k+1,nx,ny)
                        if topo_ls[mtmp] in IDX_SUBSURF:
                            boundaries.append(("k+",BOUNDS.TOP.value, lambda u,i=i,j=j,k=k: np.isclose(u[2], z_ls[k+1]) & bounds(u[0], x_ls[i], x_ls[i+1]) & bounds(u[1], y_ls[j], y_ls[j+1])))
                if topo_ls[m] not in IDX_SUBSURF:
                    continue
                # https://docs.fenicsproject.org/basix/main/
                points_neighbor = [[x_ls[i],y_ls[j],z_ls[k]],
                                   [x_ls[i+1],y_ls[j],z_ls[k]],
                                   [x_ls[i],y_ls[j+1],z_ls[k]],
                                   [x_ls[i+1],y_ls[j+1],z_ls[k]],
                                   [x_ls[i],y_ls[j],z_ls[k+1]],
                                   [x_ls[i+1],y_ls[j],z_ls[k+1]],
                                   [x_ls[i],y_ls[j+1],z_ls[k+1]],
                                   [x_ls[i+1],y_ls[j+1],z_ls[k+1]],]
                idx_element = []
                for p in points_neighbor:
                    key = tuple(p)
                    if key in point_index:
                        idx_element.append(point_index[key])
                    else:
                        idx = len(points)
                        point_index.setdefault(key, idx)
                        idx_element.append(idx)
                        points.append(p)
                cells.append(idx_element)
                cells_gindex.append(m)

    points = np.array(points, dtype=np.float64)
    cells = np.array(cells, dtype=np.int64)
    
    return points, cells, boundaries, cells_gindex

def generate_domain_info_test():
    dx_ls = [100.0]*100
    dy_ls = deepcopy(dx_ls)
    dz_ls = deepcopy(dx_ls)
    x_ls = stack_from_center(dx_ls, centroid=False)
    x_ls.append(x_ls[-1]+dx_ls[-1])  # W to E
    y_ls = stack_from_center(dy_ls, centroid=False)
    y_ls.append(y_ls[-1]+dy_ls[-1])  # N to S
    z_ls = stack_from_0(dz_ls, centroid=False)
    z_ls.append(z_ls[-1]+dz_ls[-1])  # top to bottom
    nx, ny = len(dx_ls), len(dy_ls)
    # points
    points = []
    point_index = {}
    cells = []
    boundaries: List[Tuple[int, Callable]] = []
    boundaries.append(("lateral",BOUNDS.LATERAL.value, lambda u: np.isclose(u[1], y_ls[0])|np.isclose(u[1], y_ls[-1])|np.isclose(u[0], x_ls[0])|np.isclose(u[0], x_ls[-1])))
    boundaries.append(("bottom",BOUNDS.BOTTOM.value, lambda u: np.isclose(u[2], z_ls[-1])))
    boundaries.append(("k-",BOUNDS.TOP.value, lambda u: np.isclose(u[2], z_ls[0])))
    cells_gindex: List[int] = []  # contains global index m
    for i in range(len(x_ls)-1):
        for j in range(len(y_ls)-1):
            for k in range(len(z_ls)-1):
                # https://docs.fenicsproject.org/basix/main/
                points_neighbor = [[x_ls[i],y_ls[j],z_ls[k]],
                                   [x_ls[i+1],y_ls[j],z_ls[k]],
                                   [x_ls[i],y_ls[j+1],z_ls[k]],
                                   [x_ls[i+1],y_ls[j+1],z_ls[k]],
                                   [x_ls[i],y_ls[j],z_ls[k+1]],
                                   [x_ls[i+1],y_ls[j],z_ls[k+1]],
                                   [x_ls[i],y_ls[j+1],z_ls[k+1]],
                                   [x_ls[i+1],y_ls[j+1],z_ls[k+1]],]
                idx_element = []
                for p in points_neighbor:
                    key = tuple(p)
                    if key in point_index:
                        idx_element.append(point_index[key])
                    else:
                        idx = len(points)
                        point_index.setdefault(key, idx)
                        idx_element.append(idx)
                        points.append(p)
                cells.append(idx_element)
                cells_gindex.append(calc_m(i,j,k,nx,ny))

    points = np.array(points, dtype=np.float64)
    cells = np.array(cells, dtype=np.int64)
    
    return points, cells, boundaries, cells_gindex

def generate_domain() -> Tuple[mesh.Mesh, BOUNDARIES, List[int]]:
    points, cells, boundaries, cells_gindex = generate_domain_info()
    ufl_quad = Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,)))
    domain = mesh.create_mesh(MPI.COMM_WORLD,
                              cells=cells,
                              x=points,
                              e=ufl_quad)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    return domain, boundaries, cells_gindex

# ε: 3×3
def epsilon(u):
    # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
    return sym(grad(u))

# σ: 3×3
def calc_sigma(u, dp, dt):
    return lambda_ * nabla_div(u) * Identity(len(u)) + 2.0 * mu * epsilon(u) - alpha * Kd * dt * Identity(len(u)) - beta * dp * Identity(len(u))

def calc_sigma_test(u, dp, dt):
    return lambda_ * nabla_div(u) * Identity(len(u)) + 2.0 * mu * epsilon(u) - alpha * Kd * dt * Identity(len(u)) - dp * Identity(len(u))

def zeros(u: np.ndarray):
    return 0.0 * u

def load_pt(sumpth: PathLike, cells_gindex: List[int]):
    # load pressure and temperature
    (time, t_ls), (_, p_ls) = load_snap(sumpth, ["TEMPC", "PRES"])
    return (time, [p_ls[m] for m in cells_gindex], [t_ls[m] for m in cells_gindex])

def calc_displacement(ref: PathLike | Tuple[float, np.ndarray, np.ndarray],
                      curpth: PathLike,
                      savedir: PathLike,
                      t0: float=0.0) -> List[List[float]]:
    # TOP: σ・n=0
    # LATERAL & BOTTOM: u=0
    # pressure and temperature
    
    domain, boundaries, cells_gindex = generate_domain()
    
    if isinstance(ref, tuple):
        _, p0_ls, t0_ls = ref
    else:
        _, p0_ls, t0_ls = load_pt(ref, cells_gindex)
    time, p1_ls, t1_ls = load_pt(curpth, cells_gindex)
    time += t0
    
    savepth = Path(savedir).joinpath(str(time)+".pkl")
    if savepth.exists():
        return time, ()
    makedirs(savedir, exist_ok=True)

    V = functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    u, v = TrialFunction(V), TestFunction(V)
    dp_array = np.array(np.array(p1_ls) - np.array(p0_ls)) * 1.0e6
    dt_array = np.array(np.array(t1_ls) - np.array(t0_ls))
    Q = fem.functionspace(domain, ("DG", 0))
    dp = fem.Function(Q)
    dp.x.array[:] = dp_array
    dt = fem.Function(Q)
    dt.x.array[:] = dt_array

    f = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    sigma = calc_sigma(u, dp, dt)
    
    facet_indices, facet_markers = [], []
    fdim = domain.topology.dim - 1
    for d, marker, locator in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        assert facets.shape[0] > 0, (d, marker, facets.shape)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(
        domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    ds = Measure("ds", domain=domain, subdomain_data=facet_tag)
    F = inner(sigma, epsilon(v)) * dx - dot(f, v) * dx - inner(Constant(domain, default_scalar_type((0, 0, 0))), v) * ds(BOUNDS.TOP.value)

    bcs = []
    for marker in (BOUNDS.LATERAL.value, BOUNDS.BOTTOM.value):
        u_D = Function(V)
        u_D.interpolate(zeros)
        facets = facet_tag.find(marker)
        dofs = locate_dofs_topological(V, fdim, facets)
        bcs.append(dirichletbc(u_D, dofs))

    # Solve linear variational problem
    a = lhs(F)
    L = rhs(F)

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={"ksp_type": "cg",
                       "pc_type": "hypre",
                       "pc_hypre_type": "boomeramg",
                       "ksp_rtol": 1e-8},  # {"ksp_type": "preonly", "pc_type": "lu"}
        petsc_options_prefix="neumann_dirichlet_",
    )

    uh = problem.solve()

    # save results
    coords_dof = uh.function_space.tabulate_dof_coordinates()
    uh_3d = uh.x.array.reshape(coords_dof.shape[0], 3)

    with open(savepth, "wb") as pkf:
        pickle.dump((coords_dof, uh_3d), pkf, pickle.HIGHEST_PROTOCOL)

    return time, (coords_dof, uh_3d)

    # # Create plotter and pyvista grid
    # p = pyvista.Plotter()
    # topology, cell_types, geometry = plot.vtk_mesh(V)
    # grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # # Attach vector values to grid and warp grid by vector
    # grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
    # actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    # warped = grid.warp_by_vector("u", factor=1.5)
    # actor_1 = p.add_mesh(warped, show_edges=True)
    # p.show_axes()
    # if not pyvista.OFF_SCREEN:
    #     p.show()
    # else:
    #     figure_as_array = p.screenshot("deflection.png")
    # grid.save("tmp.vtk")

    # with XDMFFile(domain.comm, "deformation.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(domain)
    #     uh.name = "Deformation"
    #     xdmf.write_function(uh)

    # s = calc_sigma(uh, dp, dt) - 1.0 / 3 * tr(calc_sigma(uh, dp, dt)) * Identity(len(uh))
    # von_Mises = sqrt(3.0 / 2 * inner(s, s))

    # V_von_mises = fem.functionspace(domain, ("DG", 0))
    # stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points)
    # stresses = fem.Function(V_von_mises)
    # stresses.interpolate(stress_expr)

    # warped.cell_data["VonMises"] = stresses.x.petsc_vec.array
    # warped.set_active_scalars("VonMises")
    # p = pyvista.Plotter()
    # p.add_mesh(warped)
    # p.show_axes()
    # if not pyvista.OFF_SCREEN:
    #     p.show()
    # else:
    #     stress_figure = p.screenshot("stresses.png")

    # return

def get_surface_value(coords: np.ndarray,
                      uh_3d: np.ndarray
                      ) -> Tuple[List[float], List[float], np.ndarray]:
    threshold = 1.0e-3
    x_ls = stack_from_center(DXYZ[0], centroid=False)
    x_ls.append(x_ls[-1]+DXYZ[0][-1])  # W to E
    y_ls = stack_from_center(DXYZ[1], centroid=False)
    y_ls.append(y_ls[-1]+DXYZ[1][-1])  # N to S
    vv = np.zeros((len(y_ls), len(x_ls), 3))
    for i, x in enumerate(x_ls):
        for j, y in enumerate(y_ls):
            filt = np.square(coords[:,[0,1]]-np.array([x,y])).sum(axis=1) < threshold
            vv[j][i] = uh_3d[filt][np.argmin(coords[filt][:,2])]
    return x_ls, y_ls, vv

def plt_surface_uh(cachepth: PathLike,
                   savedir: PathLike,
                   time: Optional[float]=None,
                   crator_coods: Optional[Tuple[List[float], List[float]]]=None,
                   baselines: Optional[Tuple[Tuple[float,float],Tuple[float,float]]]=None) -> None:
    cachepth = Path(cachepth)
    savedir = Path(savedir)
    with open(cachepth, "rb") as pkf:
        coords_dof, uh_3d = pickle.load(pkf)

    x_ls, y_ls, vv = get_surface_value(coords_dof, uh_3d)
    
    x_fine = np.linspace(min(x_ls), max(x_ls), 1000)
    y_fine = np.linspace(min(y_ls), max(y_ls), 1000)

    xx_fine, yy_fine = np.meshgrid(x_fine, y_fine)
    xx_fine1d = xx_fine.flatten()
    yy_fine1d = yy_fine.flatten()
    eval_points = np.stack([xx_fine1d, yy_fine1d], axis=1)
    values: np.ndarray = interpn((np.array(x_ls), np.array(y_ls)),
                                  vv,
                                  eval_points,
                                  method="linear"
                                )

    values = values.flatten().reshape((*xx_fine.shape, 3))
    values *= 100.0

    fname = cachepth.stem + ".png"
    if time is not None:
        fname = str(time) + ".png"
    xdir = savedir.joinpath("X")
    makedirs(xdir, exist_ok=True)
    ydir = savedir.joinpath("Y")
    makedirs(ydir, exist_ok=True)
    zdir = savedir.joinpath("Z")
    makedirs(zdir, exist_ok=True)
    magdir = savedir.joinpath("MAG")
    makedirs(magdir, exist_ok=True)

    # X
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    mappable = ax.pcolormesh(xx_fine, yy_fine, values[:,:,0], vmin=-100.0, vmax=100.0)
    if crator_coods is not None:
        ax.plot(crator_coods[0],
                crator_coods[1],
                color="black",
                alpha=0.25,
                linestyle="dashed",
                )
    if baselines is not None:
        for (x0,y0), (x1,y1) in baselines:
            ax.scatter([x0,x1],
                       [y0,y1],
                       s=15,
                       c="black",
                       alpha=0.25,
                       edgecolors='none')
            ax.plot([x0,x1],
                    [y0,y1],
                    color="black",
                    alpha=0.5,
                    linestyle="dashed")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    fig.colorbar(mappable,label="Displacement (cm)")
    fig.savefig(xdir.joinpath(fname), dpi=200)
    plt.clf()
    plt.close()

    # Y
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    mappable = ax.pcolormesh(xx_fine, yy_fine, -values[:,:,1], vmin=-100.0, vmax=100.0)
    if crator_coods is not None:
        ax.plot(crator_coods[0],
                crator_coods[1],
                color="black",
                alpha=0.5,
                linestyle="dashed")
    if baselines is not None:
        for (x0,y0), (x1,y1) in baselines:
            ax.scatter([x0,x1],
                       [y0,y1],
                       s=15,
                       c="black",
                       alpha=0.25,
                       edgecolors='none')
            ax.plot([x0,x1],
                    [y0,y1],
                    color="black",
                    alpha=0.5,
                    linestyle="dashed")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    fig.colorbar(mappable,label="Displacement (cm)")
    fig.savefig(ydir.joinpath(fname), dpi=200)
    plt.clf()
    plt.close()

    # Z
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    mappable = ax.pcolormesh(xx_fine, yy_fine, -values[:,:,2], vmin=10.0, vmax=100.0)
    if crator_coods is not None:
        ax.plot(crator_coods[0],
                crator_coods[1],
                color="black",
                alpha=0.5,
                linestyle="dashed")
    if baselines is not None:
        for (x0,y0), (x1,y1) in baselines:
            ax.scatter([x0,x1],
                       [y0,y1],
                       s=15,
                       c="black",
                       alpha=0.25,
                       edgecolors='none')
            ax.plot([x0,x1],
                    [y0,y1],
                    color="black",
                    alpha=0.5,
                    linestyle="dashed")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    fig.colorbar(mappable,label="Displacement (cm)")
    fig.savefig(zdir.joinpath(fname), dpi=200)
    plt.clf()
    plt.close()

    # magnitude
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    mag = np.sqrt(np.square(values).sum(axis=2))
    mappable = ax.pcolormesh(xx_fine, yy_fine, mag, vmin=0.0, vmax=100.0)
    if crator_coods is not None:
        ax.plot(crator_coods[0],
                crator_coods[1],
                color="black",
                alpha=0.5,
                linestyle="dashed")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    fig.colorbar(mappable, label="Displacement (cm)")
    fig.savefig(magdir.joinpath(fname), dpi=200)
    plt.clf()
    plt.close()

def plt_surface_uh_for_dir(cachedir: PathLike) -> None:
    cachedir = Path(cachedir)
    with open("./analyse_crator_coords/crator.pkl", "rb") as pkf:
        coords: Polygon = pickle.load(pkf)
    rect_trans = Transformer.from_crs(CRS_DEM, CRS_RECT, always_xy=True)
    x0, y0 = rect_trans.transform(ORIGIN[1], ORIGIN[0])
    x_crator, y_crator = coords.exterior.xy
    x_crator = [x-x0 for x in x_crator]
    y_crator = [y0-y for y in y_crator]
    xy_gnss: Dict[str, Tuple[float, float]] = {}
    for key, (lat,lng) in POS_GNSS.items():
        x, y = rect_trans.transform(lng, lat)
        xy_gnss.setdefault(key, (x-x0, y-y0))
    for fpth in cachedir.iterdir():
        if fpth.suffix != ".pkl":
            continue
        time = float(fpth.stem)
        plt_surface_uh(fpth,
                       cachedir.parent.joinpath("tstep").joinpath("displacement"),
                       time=time,
                       crator_coods=(x_crator, y_crator),
                       baselines=((xy_gnss["SW"],xy_gnss["NE"]),
                                  (xy_gnss["SE"],xy_gnss["NW"]))
                       )


def calc_distance(uh0: np.ndarray, uh1: np.ndarray) -> float:
    return np.sqrt(np.square(uh1 - uh0).sum())
    

def calc_baseline(xyv: PathLike | Tuple[np.ndarray, np.ndarray],
                  locations: Dict[str, Tuple[float, float]],) -> Tuple[float, float]:
    
    if isinstance(xyv, tuple):
        coords, uh_3d = xyv
    else:
        with open(xyv, "rb") as pkf:
            coords, uh_3d = pickle.load(pkf)
    x_ls, y_ls, vv = get_surface_value(coords, uh_3d)
    x_arr, y_arr = np.array(x_ls), np.array(y_ls)
    direction_uh: Dict[str, np.ndarray] = {}
    for direction, (xobs, yobs) in locations.items():
        i = np.argmin(np.square(x_arr-xobs))
        j = np.argmin(np.square(y_arr-yobs))
        direction_uh.setdefault(direction, vv[j][i])
    
    d_nw_se = calc_distance(direction_uh["NW"], direction_uh["SE"])
    d_ne_sw = calc_distance(direction_uh["NE"], direction_uh["SW"])

    return (d_nw_se, d_ne_sw)


def get_cachepth_in_timeseries(cachedir: PathLike) -> List[Path]:
    cachedir = Path(cachedir)
    time_cachepth: Dict[float, Path] = {}
    for fpth in cachedir.iterdir():
        if fpth.suffix != ".pkl":
            continue
        time = float(fpth.stem)
        time_cachepth.setdefault(time, fpth)
    return [time_cachepth[time] for time in sorted(list(time_cachepth.keys()))]

def calc_baseline_from_singledir(simdir: PathLike):
    print(simdir)
    simdir = Path(simdir)
    cachedir = simdir.joinpath("displacement")
    # load temperature change
    start = 1999.0 # constant
    shift = 0.0
    cachepth = Path(OUTDIR).joinpath("summary").joinpath("unrest").joinpath("param_fumarole_times.pkl")
    if cachepth.exists():
        # TODO: test
        print("exists")
        with open(cachepth, "rb") as pkf:
            params_fumarole_times: Dict[Path, Dict[str, Dict[Union[float, str], float]]] = pickle.load(pkf)
        
        fprops = None
        condition = dir_to_condition(simdir)
        for param, value in params_fumarole_times.items():
            if not (param.SRC_TEMP == condition["temp"] and
                    param.SRC_COMP1T == condition["comp1t"] and
                    param.INJ_RATE == condition["inj_rate"] and
                    param.VENT_SCALE == condition["perm"] and
                    param.CAP_SCALE == condition["cap_scale"] and
                    param.permf_cap == condition["permf_cap"] and
                    param.VK == condition["vk"] and
                    param.disperse_magmasrc == condition["d"] and
                    param.db == condition["db"] and
                    param.pfail == condition["pfail"]
                    ):
                continue
        if value.get("Sim.", None) is not None:
            fprops = value.get("Sim.", None)
        elif value.get("1819", None) is not None:
            fprops = value.get("1819", None)
        if fprops is not None:
            if fprops.get(300.0, None) is not None:
                shift = fprops[300.0]
    
    rect_trans = Transformer.from_crs(CRS_DEM, CRS_RECT, always_xy=True)
    x0, y0 = rect_trans.transform(ORIGIN[1], ORIGIN[0])
    locations: Dict[str, Tuple[float, float]] = {}
    for direction, (lat, lng) in POS_GNSS.items():
        x, y = rect_trans.transform(lng, lat)
        x -= x0
        y = y0 - y
        locations.setdefault(direction, (x, y))
    
    fpth_ls = get_cachepth_in_timeseries(cachedir)
    time_ls: List[float] = []
    d_nw_se_ls: List[float] = []
    d_ne_sw_ls: List[float] = []
    for fpth in tqdm(fpth_ls):
        time_ls.append((float(fpth.stem)-shift)/365.25+start)  # day to year
        d_nw_se, d_ne_sw = calc_baseline(fpth, locations)
        d_nw_se_ls.append(d_nw_se*100.0)  # m to cm
        d_ne_sw_ls.append(d_ne_sw*100.0)  # m to cm
    
    return (shift, time_ls, d_nw_se_ls, d_ne_sw_ls)

def plt_baseline_graph(simdir: PathLike) -> Tuple[Figure, Axes]:
    
    shift, time_ls, d_nw_se_ls, d_ne_sw_ls = calc_baseline_from_singledir(simdir)
    obsdir = Path("obsdata")
    obs_ne_sw = pd.read_csv(obsdir.joinpath("気象台 2016_E1-W1.csv"), header=None)
    obs_nw_se = pd.read_csv(obsdir.joinpath("気象台 2016_N1-S1_annotation.csv"), header=None)


    # plot
    outdir = simdir.joinpath("tstep").joinpath("displacement")
    makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(time_ls,
            d_nw_se_ls,
            color="#808080",
            label="Sim.")
    ax.scatter(obs_nw_se[0].tolist(),
               ((obs_nw_se[1]-obs_nw_se[1][0])*100.0).tolist(),
               color="#808080",
               label="Obs.")
    ax.set_xlim(obs_nw_se[0].min()-shift/365.25-3.0,
                min((obs_nw_se[0].max()+1.0,
                     time_ls[-1]))+1.0)
    # ax.set_xscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel("Baseline change (cm)")
    ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left',frameon=False)
    fig.savefig(outdir.joinpath("d_nw_se.png"), dpi=200)
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(time_ls,
            d_ne_sw_ls,
            color="#808080",
            label="Sim.")
    ax.scatter(obs_ne_sw[0].tolist(),
               ((obs_ne_sw[1]-obs_ne_sw[1][0])*100.0).tolist(),
               color="#808080",
               label="Obs.")
    ax.set_xlim(obs_ne_sw[0].min()-shift-3.0,
                min((obs_ne_sw[0].max()+1,
                     time_ls[-1]))+1.0)
    # ax.set_xscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel("Baseline change (cm)")
    fig.savefig(outdir.joinpath("d_ne_sw.png"), dpi=200)
    plt.clf()
    plt.close()
    return

def plt_baseline_compare_graph(simdirs: List[PathLike]) -> Tuple[Figure, Axes]:


    assert len(simdirs)==2
    shift1, time_ls1, d_nw_se_ls1, d_ne_sw_ls1 = calc_baseline_from_singledir(simdirs[0])
    shift2, time_ls2, d_nw_se_ls2, d_ne_sw_ls2 = calc_baseline_from_singledir(simdirs[1])

    shift1 = 7124.284000000001/365.25  #!
    shift2 = 2144.284/365.25  #!

    obsdir = Path("obsdata")
    obs_ne_sw = pd.read_csv(obsdir.joinpath("気象台 2016_E1-W1.csv"), header=None)
    obs_nw_se = pd.read_csv(obsdir.joinpath("気象台 2016_N1-S1_annotation.csv"), header=None)

    # plot
    outdir = Path(simdirs[0]).joinpath("tstep").joinpath("displacement")
    makedirs(outdir, exist_ok=True)
    savepth = outdir.joinpath("d_ne_sw_compare.png")
    print(savepth)

    fig, axes = plt.subplots(1,2)
    _prepare_ticks(plt, axes)
    axes[0].plot(time_ls1,
                 d_nw_se_ls1,
                 color="#808080",
                 linestyle="dashed",
                 label="Sim1")
    axes[0].plot(time_ls2,
                 d_nw_se_ls2,
                 color="#696969",
                 label="Sim2")
    axes[0].scatter(obs_nw_se[0].tolist(),
               ((obs_nw_se[1]-obs_nw_se[1][0])*100.0).tolist(),
               color="#696969",
               label="Obs.")
    axes[0].set_xlim(obs_ne_sw[0].min()-1.0,
                    obs_ne_sw[0].max()+1.0)
    # ax.set_xscale("log")
    axes[0].set_xlabel("Year")

    axes[1].plot(time_ls1,
            d_ne_sw_ls1,
            color="#808080",
            linestyle="dashed",
            label="Sim1")
    axes[1].plot(time_ls2,
                 d_ne_sw_ls2,
                 color="#696969",
                 label="Sim2",
                 )
    axes[1].scatter(obs_ne_sw[0].tolist(),
               ((obs_ne_sw[1]-obs_ne_sw[1][0])*100.0).tolist(),
               color="#696969",
               label="Obs.")
    axes[1].set_xlim(obs_ne_sw[0].min()-1.0,
                    obs_ne_sw[0].max()+1.0)
    # ax.set_xscale("log")
    axes[1].set_xlabel("Year")
    plt.subplots_adjust(wspace=0.3)
    fig.savefig(outdir.joinpath("d_ne_sw_compare.png"), dpi=200)
    plt.clf()
    plt.close()
    return

def calc_displacement_dir(dirpth: PathLike) -> None:
    dirpth = Path(dirpth)
    fpth_ls = get_fpth_in_timeseries(dirpth, ignore_first=False)
    refpth = fpth_ls.pop(0)
    savedir = dirpth.joinpath("displacement")
    makedirs(savedir, exist_ok=True)
    time = 0.0
    t0 = 0.0
    parent_set = set()
    for fpth in fpth_ls:
        print(fpth)
        if fpth.parent not in parent_set:
            t0 = time
        time, _ = calc_displacement(refpth, fpth, savedir, t0=t0)
        parent_set.add(fpth.parent)
    return

def mogi_h(a,nu,r,d,dp):
    # 青木 (2016), eq(23)
    return (1.0-nu)*a**3*dp/mu*\
        (1.0+
         (a/d)**3*((1.0+nu)/(2.0*(-7.0+5.0*nu))+
                   15.0*d**2*(-2.0+nu)/(4.0*(-7.0+5*nu)*(r**2+d**2))))*\
        r/(r**2+d**2)**1.5

def mogi_v(a,nu,r,d,dp):
    # 青木 (2016), eq(22)
    return (1.0-nu)*a**3*dp/mu*\
        (1.0+
         (a/d)**3*((1.0+nu)/(2.0*(-7.0+5.0*nu))+
                   15.0*d**2*(-2.0+nu)/(4.0*(-7.0+5.0*nu)*(r**2+d**2))))*\
        d/(r**2+d**2)**1.5


def test2():
    # Method of manufactured solution
    def u_ufl(x):
        return ufl.as_vector([x[0]**2, x[1]**2, x[2]**2])

    def u_numpy(x):
        return np.array([x[0]**2, x[1]**2, x[2]**2])

    points, cells, boundaries, cells_gindex = generate_domain_info()
    ufl_quad = Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,)))
    domain = mesh.create_mesh(MPI.COMM_WORLD,
                              cells=cells,
                              x=points,
                              e=ufl_quad)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    
    x = SpatialCoordinate(domain)
    Q = fem.functionspace(domain, ("DG", 0))
    dp = fem.Function(Q)
    dp.x.array[:] = 0.0
    dt = fem.Function(Q)
    dt.x.array[:] = 0.0
    u_ex = u_ufl(x)
    f = -div(calc_sigma_test(u_ex,dp,dt))
    
    facet_indices, facet_markers = [], []
    fdim = domain.topology.dim - 1
    for d, marker, locator in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        assert facets.shape[0] > 0, (d, marker, facets.shape)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(
        domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    V = functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    u, v = TrialFunction(V), TestFunction(V)
    ds = Measure("ds", domain=domain, subdomain_data=facet_tag)
    sigma = calc_sigma_test(u,dp,dt)
    F = inner(sigma, epsilon(v)) * dx - dot(f, v) * dx #  - inner(Constant(domain, default_scalar_type((0, 0, 0))), v) * ds(BOUNDS.TOP.value)

    u_bc = Function(V)
    u_bc.interpolate(u_numpy)
    bcs = []
    for marker in (BOUNDS.LATERAL.value, BOUNDS.BOTTOM.value, BOUNDS.TOP.value):
        facets = facet_tag.find(marker)
        dofs = locate_dofs_topological(V, fdim, facets)
        bcs.append(dirichletbc(u_bc, dofs))

    # Solve linear variational problem
    a = lhs(F)
    L = rhs(F)

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={"ksp_type": "cg",
                       "pc_type": "hypre",
                       "pc_hypre_type": "boomeramg",
                       "ksp_rtol": 1e-8},  # {"ksp_type": "preonly", "pc_type": "lu"}
        petsc_options_prefix="neumann_dirichlet_",
    )

    uh = problem.solve()

    lu_solver = problem.solver
    viewer = PETSc.Viewer().createASCII("lu_output.txt")
    lu_solver.view(viewer)
    solver_output = open("lu_output.txt", "r")
    for line in solver_output.readlines():
        print(line)

    # Error
    comm = uh.function_space.mesh.comm
    error = form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)
    E = np.sqrt(comm.allreduce(assemble_scalar(error), MPI.SUM))
    if comm.rank == 0:
        print(f"L2-error: {E:.2e}")

    # save results
    coords_dof = uh.function_space.tabulate_dof_coordinates()
    uh_3d = uh.x.array.reshape(coords_dof.shape[0], 3)

    with open("u2dp0dt0.pkl", "wb") as pkf:
        pickle.dump((coords_dof, uh_3d), pkf, pickle.HIGHEST_PROTOCOL)

    return time, (coords_dof, uh_3d)

def test3():
    # Method of manufactured solution
    # dp = x^2
    def u_ufl(x):
        return ufl.as_vector([x[0]**2, x[1]**2, x[2]**2])

    def u_numpy(x):
        return np.array([x[0]**2, x[1]**2, x[2]**2])

    points, cells, boundaries, cells_gindex = generate_domain_info()
    ufl_quad = Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,)))
    domain = mesh.create_mesh(MPI.COMM_WORLD,
                              cells=cells,
                              x=points,
                              e=ufl_quad)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    
    x = SpatialCoordinate(domain)
    Q = fem.functionspace(domain, ("DG", 0))
    dp = fem.Function(Q)
    nx, ny = len(DXYZ[0]), len(DXYZ[1])
    xc_ls = stack_from_center(DXYZ[0])
    dp_ls = []
    for m in cells_gindex:
        i,j,k = calc_ijk(m, nx, ny)
        dp_ls.append(xc_ls[i]**2)

    dp.x.array[:] = np.array(dp_ls)
    dt = fem.Function(Q)
    dt.x.array[:] = 0.0
    u_ex = u_ufl(x)
    f = -div(calc_sigma_test(u_ex,dp,dt))
    
    facet_indices, facet_markers = [], []
    fdim = domain.topology.dim - 1
    for d, marker, locator in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        assert facets.shape[0] > 0, (d, marker, facets.shape)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(
        domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    V = functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    u, v = TrialFunction(V), TestFunction(V)
    ds = Measure("ds", domain=domain, subdomain_data=facet_tag)
    sigma = calc_sigma_test(u,dp,dt)
    F = inner(sigma, epsilon(v)) * dx - dot(f, v) * dx #  - inner(Constant(domain, default_scalar_type((0, 0, 0))), v) * ds(BOUNDS.TOP.value)

    u_bc = Function(V)
    u_bc.interpolate(u_numpy)
    bcs = []
    for marker in (BOUNDS.LATERAL.value, BOUNDS.BOTTOM.value, BOUNDS.TOP.value):
        facets = facet_tag.find(marker)
        dofs = locate_dofs_topological(V, fdim, facets)
        bcs.append(dirichletbc(u_bc, dofs))

    # Solve linear variational problem
    a = lhs(F)
    L = rhs(F)

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={"ksp_type": "cg",
                       "pc_type": "hypre",
                       "pc_hypre_type": "boomeramg",
                       "ksp_rtol": 1e-8},  # {"ksp_type": "preonly", "pc_type": "lu"}
        petsc_options_prefix="neumann_dirichlet_",
    )

    uh = problem.solve()

    lu_solver = problem.solver
    viewer = PETSc.Viewer().createASCII("lu_output.txt")
    lu_solver.view(viewer)
    solver_output = open("lu_output.txt", "r")
    for line in solver_output.readlines():
        print(line)

    # Error
    comm = uh.function_space.mesh.comm
    error = form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)
    E = np.sqrt(comm.allreduce(assemble_scalar(error), MPI.SUM))
    if comm.rank == 0:
        print(f"L2-error: {E:.2e}")

    # save results
    coords_dof = uh.function_space.tabulate_dof_coordinates()
    uh_3d = uh.x.array.reshape(coords_dof.shape[0], 3)

    with open("u2dp2dt0.pkl", "wb") as pkf:
        pickle.dump((coords_dof, uh_3d), pkf, pickle.HIGHEST_PROTOCOL)

    return time, (coords_dof, uh_3d)

def test4():
    # Method of manufactured solution
    # dt = x^2
    def u_ufl(x):
        return ufl.as_vector([x[0]**2, x[1]**2, x[2]**2])

    def u_numpy(x):
        return np.array([x[0]**2, x[1]**2, x[2]**2])

    points, cells, boundaries, cells_gindex = generate_domain_info()
    ufl_quad = Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,)))
    domain = mesh.create_mesh(MPI.COMM_WORLD,
                              cells=cells,
                              x=points,
                              e=ufl_quad)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    
    x = SpatialCoordinate(domain)
    Q = fem.functionspace(domain, ("DG", 0))
    dp = fem.Function(Q)
    dp.x.array[:] = 0.0
    dt = fem.Function(Q)
    nx, ny = len(DXYZ[0]), len(DXYZ[1])
    xc_ls = stack_from_center(DXYZ[0])
    dt_ls = []
    for m in cells_gindex:
        i,j,k = calc_ijk(m, nx, ny)
        dt_ls.append(xc_ls[i]**2)
    dt.x.array[:] = np.array(dt_ls)
    u_ex = u_ufl(x)
    f = -div(calc_sigma_test(u_ex,dp,dt))
    
    facet_indices, facet_markers = [], []
    fdim = domain.topology.dim - 1
    for d, marker, locator in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        assert facets.shape[0] > 0, (d, marker, facets.shape)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(
        domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    V = functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    u, v = TrialFunction(V), TestFunction(V)
    ds = Measure("ds", domain=domain, subdomain_data=facet_tag)
    sigma = calc_sigma_test(u,dp,dt)
    F = inner(sigma, epsilon(v)) * dx - dot(f, v) * dx #  - inner(Constant(domain, default_scalar_type((0, 0, 0))), v) * ds(BOUNDS.TOP.value)

    u_bc = Function(V)
    u_bc.interpolate(u_numpy)
    bcs = []
    for marker in (BOUNDS.LATERAL.value, BOUNDS.BOTTOM.value, BOUNDS.TOP.value):
        facets = facet_tag.find(marker)
        dofs = locate_dofs_topological(V, fdim, facets)
        bcs.append(dirichletbc(u_bc, dofs))

    # Solve linear variational problem
    a = lhs(F)
    L = rhs(F)

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={"ksp_type": "cg",
                       "pc_type": "hypre",
                       "pc_hypre_type": "boomeramg",
                       "ksp_rtol": 1e-8},  # {"ksp_type": "preonly", "pc_type": "lu"}
        petsc_options_prefix="neumann_dirichlet_",
    )

    uh = problem.solve()

    lu_solver = problem.solver
    viewer = PETSc.Viewer().createASCII("lu_output.txt")
    lu_solver.view(viewer)
    solver_output = open("lu_output.txt", "r")
    for line in solver_output.readlines():
        print(line)

    # Error
    comm = uh.function_space.mesh.comm
    error = form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)
    E = np.sqrt(comm.allreduce(assemble_scalar(error), MPI.SUM))
    if comm.rank == 0:
        print(f"L2-error: {E:.2e}")

    # save results
    coords_dof = uh.function_space.tabulate_dof_coordinates()
    uh_3d = uh.x.array.reshape(coords_dof.shape[0], 3)

    with open("u2dp0dt2.pkl", "wb") as pkf:
        pickle.dump((coords_dof, uh_3d), pkf, pickle.HIGHEST_PROTOCOL)

    return time, (coords_dof, uh_3d)

def _prepare_ticks(plt: plt, axes: List[plt.Axes], params: Tuple = (7, 5, 7, 5), labelsize=14, inner: bool = False) -> None:
    if inner:
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['axes.axisbelow'] = True
    for ax in axes:
        ax.tick_params(axis="x", which="major", length=params[0])
        ax.tick_params(axis="x", which="minor", length=params[1])
        ax.tick_params(axis="y", which="major", length=params[2])
        ax.tick_params(axis="y", which="minor", length=params[3])
        ax.tick_params(labelsize=labelsize)

def plt_dtdp0():
    # L2-error: 1.75e+10
    with open("u2dp0dt0.pkl", "rb") as pkf:
        coords_dof, uh_3d = pickle.load(pkf)
    x_fem_ls = []
    ux_fem_ls = []
    y_fem_ls = []
    uy_fem_ls = []
    z_fem_ls = []
    uz_fem_ls = []

    for u, xyz in zip(uh_3d.tolist(), coords_dof.tolist()):
        x_fem_ls.append(xyz[0])
        ux_fem_ls.append(u[0])
        y_fem_ls.append(xyz[1])
        uy_fem_ls.append(u[1])
        z_fem_ls.append(xyz[2])
        uz_fem_ls.append(u[2])

    # x
    fig, axes = plt.subplots(1,2)
    _prepare_ticks(plt,axes)
    axes[0].scatter(x_fem_ls, ux_fem_ls, s=10, label="Ux",c="#7f7f7f")
    x_ls = np.linspace(-0.5*sum(DXYZ[0]), 0.5*sum(DXYZ[0]), 1000)
    axes[0].plot(x_ls, [x**2 for x in x_ls],c="#7f7f7f")
    
    axes[1].scatter(z_fem_ls, uz_fem_ls, s=10, label="Uz",c="#7f7f7f")
    z_ls = np.linspace(0.0, sum(DXYZ[2]), 1000)
    axes[1].plot(z_ls, [z**2 for z in z_ls],c="#7f7f7f")
    fig.legend()
    plt.subplots_adjust(wspace=0.3)
    fig.savefig("uxuz.png", dpi=200, bbox_inches="tight")
    plt.clf()
    plt.close()

    # y
    fig, ax = plt.subplots()
    ax.scatter(y_fem_ls, uy_fem_ls, s=10, label="Uy",c="#7f7f7f")
    y_ls = np.linspace(-0.5*sum(DXYZ[1]), 0.5*sum(DXYZ[1]), 1000)
    ax.plot(y_ls, [y**2 for y in y_ls],c="#7f7f7f")
    fig.legend()
    fig.savefig("uy.png")
    plt.clf()
    plt.close()

    # z
    fig, ax = plt.subplots()
    ax.scatter(z_fem_ls, uz_fem_ls, s=10, label="Uz",c="#7f7f7f")
    z_ls = np.linspace(0.0, sum(DXYZ[2]), 1000)
    ax.plot(z_ls, [z**2 for z in z_ls],c="#7f7f7f")
    fig.legend()
    fig.savefig("uz.png")
    plt.clf()
    plt.close()
    return

def plt_dp2dt0():
    # L2-error: 1.75e+10
    with open("u2dp2dt0.pkl", "rb") as pkf:
        coords_dof, uh_3d = pickle.load(pkf)
    x_fem_ls = []
    ux_fem_ls = []
    y_fem_ls = []
    uy_fem_ls = []
    z_fem_ls = []
    uz_fem_ls = []

    for u, xyz in zip(uh_3d.tolist(), coords_dof.tolist()):
        x_fem_ls.append(xyz[0])
        ux_fem_ls.append(u[0])
        y_fem_ls.append(xyz[1])
        uy_fem_ls.append(u[1])
        z_fem_ls.append(xyz[2])
        uz_fem_ls.append(u[2])

    # x
    fig, axes = plt.subplots(1,2)
    _prepare_ticks(plt,axes)
    axes[0].scatter(x_fem_ls, ux_fem_ls, s=10, label="Ux",c="#7f7f7f")
    x_ls = np.linspace(-0.5*sum(DXYZ[0]), 0.5*sum(DXYZ[0]), 1000)
    axes[0].plot(x_ls, [x**2 for x in x_ls],c="#7f7f7f")
    
    axes[1].scatter(z_fem_ls, uz_fem_ls, s=10, label="Uz",c="#7f7f7f")
    z_ls = np.linspace(0.0, sum(DXYZ[2]), 1000)
    axes[1].plot(z_ls, [z**2 for z in z_ls],c="#7f7f7f")
    fig.legend()
    plt.subplots_adjust(wspace=0.3)
    fig.savefig("uxuzdp2.png", dpi=200, bbox_inches="tight")
    plt.clf()
    plt.close()

    # y
    fig, ax = plt.subplots()
    ax.scatter(y_fem_ls, uy_fem_ls, s=10, label="Uy",c="#7f7f7f")
    y_ls = np.linspace(-0.5*sum(DXYZ[1]), 0.5*sum(DXYZ[1]), 1000)
    ax.plot(y_ls, [y**2 for y in y_ls],c="#7f7f7f")
    fig.legend()
    fig.savefig("uydp2.png")
    plt.clf()
    plt.close()

    # z
    fig, ax = plt.subplots()
    ax.scatter(z_fem_ls, uz_fem_ls, s=10, label="Uz",c="#7f7f7f")
    z_ls = np.linspace(0.0, sum(DXYZ[2]), 1000)
    ax.plot(z_ls, [z**2 for z in z_ls],c="#7f7f7f")
    fig.legend()
    fig.savefig("uzdp2.png")
    plt.clf()
    plt.close()
    return

def plt_dp0dt2():
    # L2-error: 1.76e+10
    with open("u2dp0dt2.pkl", "rb") as pkf:
        coords_dof, uh_3d = pickle.load(pkf)
    x_fem_ls = []
    ux_fem_ls = []
    y_fem_ls = []
    uy_fem_ls = []
    z_fem_ls = []
    uz_fem_ls = []

    for u, xyz in zip(uh_3d.tolist(), coords_dof.tolist()):
        x_fem_ls.append(xyz[0])
        ux_fem_ls.append(u[0])
        y_fem_ls.append(xyz[1])
        uy_fem_ls.append(u[1])
        z_fem_ls.append(xyz[2])
        uz_fem_ls.append(u[2])

    # x
    fig, axes = plt.subplots(1,2)
    _prepare_ticks(plt,axes)
    axes[0].scatter(x_fem_ls, ux_fem_ls, s=10, label="Ux",c="#7f7f7f")
    x_ls = np.linspace(-0.5*sum(DXYZ[0]), 0.5*sum(DXYZ[0]), 1000)
    axes[0].plot(x_ls, [x**2 for x in x_ls],c="#7f7f7f")
    
    axes[1].scatter(z_fem_ls, uz_fem_ls, s=10, label="Uz",c="#7f7f7f")
    z_ls = np.linspace(0.0, sum(DXYZ[2]), 1000)
    axes[1].plot(z_ls, [z**2 for z in z_ls],c="#7f7f7f")
    fig.legend()
    plt.subplots_adjust(wspace=0.3)
    fig.savefig("uxuzdt2.png", dpi=200, bbox_inches="tight")
    plt.clf()
    plt.close()

    # y
    fig, ax = plt.subplots()
    ax.scatter(y_fem_ls, uy_fem_ls, s=10, label="Uy",c="#7f7f7f")
    y_ls = np.linspace(-0.5*sum(DXYZ[1]), 0.5*sum(DXYZ[1]), 1000)
    ax.plot(y_ls, [y**2 for y in y_ls],c="#7f7f7f")
    fig.legend()
    fig.savefig("uydt2.png")
    plt.clf()
    plt.close()

    # z
    fig, ax = plt.subplots()
    ax.scatter(z_fem_ls, uz_fem_ls, s=10, label="Uz",c="#7f7f7f")
    z_ls = np.linspace(0.0, sum(DXYZ[2]), 1000)
    ax.plot(z_ls, [z**2 for z in z_ls],c="#7f7f7f")
    
    fig.legend()
    fig.savefig("uzdt2.png")
    plt.clf()
    plt.close()
    return

if __name__ == "__main__":
    plt_baseline_compare_graph(["/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_100000.0_v/unrest/900.0_0.0_35000.0_10.0_100000.0_v_d",
                                "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_100000.0_v/unrest/900.0_0.0_35000.0_10.0_100000.0_v_d",
                                ])
    # test4()
    # plt_dp0dt2()
    # start = time()
    # calc_displacement("/mnt/f/tarumai2/900.0_0.1_10000.0_10.0_1.0_v/tmp.0057.SUM",
    #                   "/mnt/f/tarumai2/900.0_0.1_10000.0_10.0_1.0_v/unrest/900.0_0.1_15000.0_10.0_100000.0_v_d/ITER_1/tmp.1096.SUM",
                      
    #                   )
    # end = time()
    # print(f"elapsed time: {end-start} s")
    # calc_uh_timeseries("/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_100000.0_v/unrest/900.0_0.0_15000.0_10.0_100000.0_v_d")
    dirpth_ls = [
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_100000.0_v/unrest/900.0_0.0_15000.0_10.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_100000.0_v/unrest/900.0_0.0_20000.0_10.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_100000.0_v/unrest/900.0_0.0_25000.0_10.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_100000.0_v/unrest/900.0_0.0_30000.0_10.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_100000.0_v/unrest/900.0_0.0_35000.0_10.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_v/unrest/900.0_0.0_15000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_v/unrest/900.0_0.0_20000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_v/unrest/900.0_0.0_25000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_v/unrest/900.0_0.0_30000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_v/unrest/900.0_0.0_35000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_100000.0_v/unrest/900.0_0.0_10000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_100000.0_v/unrest/900.0_0.0_15000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_100000.0_v/unrest/900.0_0.0_20000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_100000.0_v/unrest/900.0_0.0_25000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_100000.0_v/unrest/900.0_0.0_30000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_100000.0_v/unrest/900.0_0.0_35000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_v/unrest/900.0_0.0_15000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_v/unrest/900.0_0.0_20000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_v/unrest/900.0_0.0_25000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_v/unrest/900.0_0.0_30000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_1000.0_10000.0_v/unrest/900.0_0.0_35000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_100000.0_v/unrest/900.0_0.0_15000.0_10.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_100000.0_v/unrest/900.0_0.0_20000.0_10.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_100000.0_v/unrest/900.0_0.0_25000.0_10.0_100000.0_v_d",         #! これより下向き
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_100000.0_v/unrest/900.0_0.0_30000.0_10.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_100000.0_v/unrest/900.0_0.0_35000.0_10.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_v/unrest/900.0_0.0_15000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_v/unrest/900.0_0.0_20000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_v/unrest/900.0_0.0_25000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_v/unrest/900.0_0.0_30000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10.0_v/unrest/900.0_0.0_35000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_100000.0_v/unrest/900.0_0.0_15000.0_10000.0_100000.0_v_d",  #! これより上向き
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_100000.0_v/unrest/900.0_0.0_20000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_100000.0_v/unrest/900.0_0.0_25000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_100000.0_v/unrest/900.0_0.0_30000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_100000.0_v/unrest/900.0_0.0_35000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_v/unrest/900.0_0.0_15000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_v/unrest/900.0_0.0_20000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_v/unrest/900.0_0.0_25000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_v/unrest/900.0_0.0_30000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.0_10000.0_10000.0_v/unrest/900.0_0.0_35000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_100000.0_v/unrest/900.0_0.1_15000.0_10.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_100000.0_v/unrest/900.0_0.1_20000.0_10.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_100000.0_v/unrest/900.0_0.1_25000.0_10.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_100000.0_v/unrest/900.0_0.1_30000.0_10.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_100000.0_v/unrest/900.0_0.1_35000.0_10.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_v/unrest/900.0_0.1_15000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_v/unrest/900.0_0.1_20000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_v/unrest/900.0_0.1_25000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_v/unrest/900.0_0.1_30000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10.0_v/unrest/900.0_0.1_35000.0_10.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_100000.0_v/unrest/900.0_0.1_15000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_100000.0_v/unrest/900.0_0.1_20000.0_10000.0_100000.0_v_d",  #!
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_100000.0_v/unrest/900.0_0.1_25000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_100000.0_v/unrest/900.0_0.1_30000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_100000.0_v/unrest/900.0_0.1_35000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_v/unrest/900.0_0.1_15000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_v/unrest/900.0_0.1_20000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_v/unrest/900.0_0.1_25000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_v/unrest/900.0_0.1_30000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_1000.0_10000.0_v/unrest/900.0_0.1_35000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_100000.0_v/unrest/900.0_0.1_15000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_100000.0_v/unrest/900.0_0.1_20000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_100000.0_v/unrest/900.0_0.1_25000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_100000.0_v/unrest/900.0_0.1_30000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_100000.0_v/unrest/900.0_0.1_35000.0_10000.0_100000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_v/unrest/900.0_0.1_15000.0_10000.0_v_d",  #! ↓
        # "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_v/unrest/900.0_0.1_20000.0_10000.0_v_d",     #! ↑
        # "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_v/unrest/900.0_0.1_25000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_v/unrest/900.0_0.1_30000.0_10000.0_v_d",
        # "/mnt/f/tarumai2/900.0_0.1_10000.0_10000.0_v/unrest/900.0_0.1_35000.0_10000.0_v_d",
        # TODO: brit条件
                 ]
    # for dirpth in dirpth_ls:
        # print(dirpth)
        # calc_displacement_dir(dirpth)
        # plt_surface_uh_for_dir(dirpth + "/displacement")
        # img2mov(dirpth+"/tstep/displacement/X/", ftype="displacement")
        # img2mov(dirpth+"/tstep/displacement/Y/", ftype="displacement")
        # img2mov(dirpth+"/tstep/displacement/Z/", ftype="displacement")
        # plt_baseline_graph(dirpth)
    # plt_surface_uh_for_dir("/mnt/f/tarumai2/900.0_0.0_1000.0_10.0_100000.0_v/unrest/900.0_0.0_15000.0_10.0_100000.0_v_d"+"/displacement_lu")
    pass