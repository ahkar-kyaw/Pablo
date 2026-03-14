# kinematics_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import math
import time

import numpy as np


@dataclass(frozen=True)
class PlanarWaypoint:
    x_m: float
    y_m: float
    pen_down: bool


@dataclass(frozen=True)
class CartesianWaypoint:
    x_m: float
    y_m: float
    z_m: float
    pen_down: bool


@dataclass
class DrawingPlane:
    origin_base_m: np.ndarray
    x_axis_base: np.ndarray
    y_axis_base: np.ndarray
    z_pen_down_m: float = 0.050
    z_pen_up_m: float = 0.120

    def __post_init__(self) -> None:
        self.origin_base_m = np.array(self.origin_base_m, dtype=float).reshape(3)
        self.x_axis_base = np.array(self.x_axis_base, dtype=float).reshape(3)
        self.y_axis_base = np.array(self.y_axis_base, dtype=float).reshape(3)

        self.x_axis_base = self.x_axis_base / (np.linalg.norm(self.x_axis_base) + 1e-12)
        self.y_axis_base = self.y_axis_base / (np.linalg.norm(self.y_axis_base) + 1e-12)

        if abs(float(np.dot(self.x_axis_base, self.y_axis_base))) > 0.2:
            raise ValueError("DrawingPlane axes are not close to orthogonal.")


class KinematicsEngine:
    def __init__(self):
        try:
            from hal.products.qarm import QArmUtilities  # type: ignore
        except Exception:
            from Libraries.hal.products.qarm import QArmUtilities  # type: ignore

        self._qarm_utils = QArmUtilities()

    def planar_to_cartesian(
        self,
        plane: DrawingPlane,
        planar: Iterable[PlanarWaypoint],
        max_step_m: float = 0.003,
    ) -> List[CartesianWaypoint]:
        pts = list(planar)
        if not pts:
            return []

        base_xy: List[Tuple[float, float, bool]] = []
        for p in pts:
            base_xyz = plane.origin_base_m + p.x_m * plane.x_axis_base + p.y_m * plane.y_axis_base
            base_xy.append((float(base_xyz[0]), float(base_xyz[1]), bool(p.pen_down)))

        resampled_xy: List[Tuple[float, float, bool]] = [base_xy[0]]
        for (x0, y0, down0), (x1, y1, down1) in zip(base_xy, base_xy[1:]):
            dx = x1 - x0
            dy = y1 - y0
            dist = math.hypot(dx, dy)
            n = max(1, int(math.ceil(dist / max_step_m)))

            for k in range(1, n + 1):
                a = k / n
                x = x0 + a * dx
                y = y0 + a * dy
                pen_down = down1 if k == n else down0
                resampled_xy.append((x, y, pen_down))

        out: List[CartesianWaypoint] = []
        prev_down = False
        for i, (x, y, pen_down) in enumerate(resampled_xy):
            if i == 0:
                out.append(CartesianWaypoint(x, y, plane.z_pen_up_m, False))
                if pen_down:
                    out.append(CartesianWaypoint(x, y, plane.z_pen_down_m, True))
                prev_down = pen_down
                continue

            if pen_down != prev_down:
                if pen_down:
                    out.append(CartesianWaypoint(x, y, plane.z_pen_up_m, False))
                    out.append(CartesianWaypoint(x, y, plane.z_pen_down_m, True))
                else:
                    out.append(CartesianWaypoint(x, y, plane.z_pen_up_m, False))
                prev_down = pen_down
            else:
                z = plane.z_pen_down_m if pen_down else plane.z_pen_up_m
                out.append(CartesianWaypoint(x, y, z, pen_down))

        last = out[-1]
        if last.pen_down:
            out.append(CartesianWaypoint(last.x_m, last.y_m, plane.z_pen_up_m, False))

        return out

    def ik(self, position_xyz_m: np.ndarray, phi_prev: np.ndarray, gamma_rad: float) -> np.ndarray:
        pos = np.array(position_xyz_m, dtype=float).reshape(3)
        phi_prev = np.array(phi_prev, dtype=float).reshape(4)

        all_phi, phi_cmd = self._qarm_utils.qarm_inverse_kinematics(pos, float(gamma_rad), phi_prev)
        _ = all_phi
        return np.array(phi_cmd, dtype=float).reshape(4)


class TrajectoryTimer:
    def __init__(self, rate_hz: float):
        self.rate_hz = float(rate_hz)
        self.dt = 1.0 / self.rate_hz
        self._t_next = time.time()

    def sleep_to_next(self) -> None:
        self._t_next += self.dt
        t_now = time.time()
        if self._t_next > t_now:
            time.sleep(self._t_next - t_now)
        else:
            self._t_next = t_now