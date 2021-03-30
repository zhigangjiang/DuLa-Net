'''
Reimplement post optimization code from https://github.com/zouchuhang/LayoutNet
'''
import numpy as np


def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi


def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi


def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5


def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5


def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y

def calc_height(coor, floor_z=-1.68, w=1024, h=512):
    heights = []
    for i in range(0, len(coor), 2):
        y_c = coor[i][1]
        y_f = coor[i+1][1]

        assert y_f > y_c and y_f > h * 0.5

        s_c = (h * 0.5 - y_c) / (y_f - h * 0.5)
        assert s_c > 0
        heights.append(-floor_z * (1 + s_c))
    return np.array(heights).mean()

def pano_connect_points(p1, p2, z=-50, w=1024, h=512):
    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    if abs(p1[0] - p2[0]) < w / 2:
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)
    coorxs = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    vx = x2 - x1
    vy = y2 - y1
    us = coorx2u(coorxs, w)
    ps = (np.tan(us) * x1 - y1) / (vy - np.tan(us) * vx)
    cs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    vs = np.arctan2(z, cs)
    coorys = v2coory(vs)

    return np.stack([coorxs, coorys], axis=-1)
