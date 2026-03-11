# kinematics_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import math
import time

import numpy as np


@dataclass(frozen=True)
class PlanarWaypoint:
    """2D waypoint on the drawing plane (paper frame)."""
    x_m: float
    y_m: float
    pen_down: bool  # True means contact with paper


@dataclass(frozen=True)
class CartesianWaypoint:
    """3D waypoint in the robot base frame."""
    x_m: float
    y_m: float
    z_m: float
    pen_down: bool


@dataclass
class DrawingPlane:
    """
    Defines a 2D drawing plane embedded in the robot base frame.

    origin_base_m is the base-frame XYZ location of the paper frame origin.
    x_axis_base and y_axis_base are unit vectors in base frame.
    """
    origin_base_m: np.ndarray  # shape (3,)
    x_axis_base: np.ndarray    # shape (3,)
    y_axis_base: np.ndarray    # shape (3,)

    z_pen_down_m: float = 0.050  # absolute Z in base frame for pen contact
    z_pen_up_m: float = 0.120    # absolute Z in base frame for safe hover

    def __post_init__(self) -> None:
        self.origin_base_m = np.array(self.origin_base_m, dtype=float).reshape(3)
        self.x_axis_base = np.array(self.x_axis_base, dtype=float).reshape(3)
        self.y_axis_base = np.array(self.y_axis_base, dtype=float).reshape(3)

        # Normalize axes defensively
        self.x_axis_base = self.x_axis_base / (np.linalg.norm(self.x_axis_base) + 1e-12)
        self.y_axis_base = self.y_axis_base / (np.linalg.norm(self.y_axis_base) + 1e-12)

        # Basic orthogonality check (not enforced, but useful during debugging)
        if abs(float(np.dot(self.x_axis_base, self.y_axis_base))) > 0.2:
            raise ValueError("DrawingPlane axes are not close to orthogonal. Recheck calibration.")


class KinematicsEngine:
    """
    Wraps IK calls and simple trajectory utilities.

    Uses QArmUtilities.qarm_inverse_kinematics(positionCmd, gamma, phi_prev)
    when available (the same convention used in your repo). If the library
    is not importable, it raises at runtime.
    """

    def __init__(self, gamma_rad: float = 0.0):
        self.gamma_rad = float(gamma_rad)

        # Match the import pattern in your repo
        try:
            from hal.products.qarm import QArmUtilities  # type: ignore
        except Exception:
            try:
                from Libraries.hal.products.qarm import QArmUtilities  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Could not import QArmUtilities. Ensure Quanser libraries are on PYTHONPATH."
                ) from e

        self._qarm_utils = QArmUtilities()

    def planar_to_cartesian(
        self,
        plane: DrawingPlane,
        planar: Iterable[PlanarWaypoint],
        max_step_m: float = 0.003,
    ) -> List[CartesianWaypoint]:
        """
        Map 2D planar waypoints into 3D base-frame waypoints and resample to enforce max step.
        """
        pts = list(planar)
        if not pts:
            return []

        # Convert planar points to base XY positions (Z handled separately)
        base_xy: List[Tuple[float, float, bool]] = []
        for p in pts:
            base_xyz = plane.origin_base_m + p.x_m * plane.x_axis_base + p.y_m * plane.y_axis_base
            base_xy.append((float(base_xyz[0]), float(base_xyz[1]), bool(p.pen_down)))

        # Resample segments
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

                # If pen state changes, prefer to apply the new state at the end of the segment
                pen_down = down1 if k == n else down0
                resampled_xy.append((x, y, pen_down))

        # Add Z with pen-up and pen-down transitions
        out: List[CartesianWaypoint] = []
        prev_down = False
        for i, (x, y, pen_down) in enumerate(resampled_xy):
            if i == 0:
                # Always start hovering
                out.append(CartesianWaypoint(x, y, plane.z_pen_up_m, False))
                if pen_down:
                    out.append(CartesianWaypoint(x, y, plane.z_pen_down_m, True))
                prev_down = pen_down
                continue

            if pen_down != prev_down:
                # Transition at this XY
                if pen_down:
                    out.append(CartesianWaypoint(x, y, plane.z_pen_up_m, False))
                    out.append(CartesianWaypoint(x, y, plane.z_pen_down_m, True))
                else:
                    out.append(CartesianWaypoint(x, y, plane.z_pen_up_m, False))
                prev_down = pen_down
            else:
                z = plane.z_pen_down_m if pen_down else plane.z_pen_up_m
                out.append(CartesianWaypoint(x, y, z, pen_down))

        # Ensure final is pen-up
        last = out[-1]
        if last.pen_down:
            out.append(CartesianWaypoint(last.x_m, last.y_m, plane.z_pen_up_m, False))

        return out

    def ik(self, position_xyz_m: np.ndarray, phi_prev: np.ndarray) -> np.ndarray:
        """
        Compute joint command (phiCmd) for desired XYZ and current previous joints.

        Returns phiCmd shape (4,) in radians.
        """
        pos = np.array(position_xyz_m, dtype=float).reshape(3)
        phi_prev = np.array(phi_prev, dtype=float).reshape(4)

        all_phi, phi_cmd = self._qarm_utils.qarm_inverse_kinematics(pos, self.gamma_rad, phi_prev)
        _ = all_phi
        return np.array(phi_cmd, dtype=float).reshape(4)


class TrajectoryTimer:
    """Simple fixed-rate timer helper for main loops."""
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
            # We are behind, resync
            self._t_next = t_now