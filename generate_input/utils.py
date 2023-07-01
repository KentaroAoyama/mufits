from typing import Tuple


def calc_ijk(m: int, nx: int, ny: int) -> Tuple[int]:
    q, i = divmod(m, nx)
    k, j = divmod(q, ny)
    return i, j, k


def calc_m(i, j, k, nx, ny):
    return nx * ny * k + nx * j + i
