# vision_processor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
from pathlib import Path

import numpy as np

from kinematics_engine import PlanarWaypoint


@dataclass
class DrawingJobSpec:
    mode: str = "circle"  # "circle" or "image"

    # Image settings
    image_filename: Optional[str] = None  # file in the same folder as this script
    drawing_size_m: Tuple[float, float] = (0.18, 0.18)  # width, height in plane frame
    center_xy_m: Tuple[float, float] = (0.10, 0.10)
    ink_is_dark: bool = True
    max_points: int = 600

    # Circle params used in circle mode only
    radius_m: float = 0.05
    points: int = 240


class VisionProcessor:
    def build_planar_waypoints(self, job: DrawingJobSpec) -> List[PlanarWaypoint]:
        mode = (job.mode or "").lower().strip()
        if mode == "circle":
            return self._circle(job.center_xy_m, job.radius_m, job.points)

        if mode == "image":
            if not job.image_filename:
                raise ValueError("image_filename is not set. Use: load <filename> in the UI.")
            pts_px = self._extract_contour_pixels_from_image(job.image_filename, job.ink_is_dark)
            pts_xy_m = self._pixels_to_plane(
                pts_px,
                center_xy_m=job.center_xy_m,
                drawing_size_m=job.drawing_size_m,
                max_points=job.max_points,
            )
            return self._polyline_waypoints(pts_xy_m)

        raise ValueError(f"Unknown job.mode={job.mode!r}")

    def _resolve_local_image_path(self, image_filename: str) -> Path:
        here = Path(__file__).resolve().parent
        return (here / image_filename).resolve()

    def _extract_contour_pixels_from_image(self, image_filename: str, ink_is_dark: bool) -> np.ndarray:
        path = self._resolve_local_image_path(image_filename)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Try OpenCV first
        try:
            import cv2  # type: ignore

            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError("cv2 failed to load the image")

            # Otsu threshold
            thresh_flag = cv2.THRESH_BINARY_INV if ink_is_dark else cv2.THRESH_BINARY
            _, bw = cv2.threshold(img, 0, 255, thresh_flag + cv2.THRESH_OTSU)

            # External contours only
            found = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = found[0] if len(found) == 2 else found[1]
            if not contours:
                raise RuntimeError("No contours found in image")

            # Pick the largest contour
            contour = max(contours, key=cv2.contourArea)
            pts = contour[:, 0, :]  # Nx2 as (x, y) pixels
            return pts.astype(float)

        except ImportError:
            pass
        except Exception:
            # Fall back if OpenCV is present but contour extraction fails
            pass

        # Pillow fallback
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise ImportError("Neither OpenCV nor Pillow is available. Install pillow or opencv-python.") from e

        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.uint8)

        t = self._otsu_threshold(arr)
        if ink_is_dark:
            mask = arr < t
        else:
            mask = arr > t

        edge = self._binary_edge(mask)
        coords = np.argwhere(edge)  # (row, col)
        if coords.shape[0] < 20:
            raise RuntimeError("Edge extraction produced too few points. Use a higher contrast image.")

        # Convert to (x, y) pixels with a simple angular ordering
        rows = coords[:, 0].astype(float)
        cols = coords[:, 1].astype(float)
        cy = float(np.mean(rows))
        cx = float(np.mean(cols))
        ang = np.arctan2(rows - cy, cols - cx)
        order = np.argsort(ang)
        pts = np.stack([cols[order], rows[order]], axis=1)  # Nx2 (x, y)
        return pts

    def _otsu_threshold(self, gray: np.ndarray) -> int:
        hist = np.bincount(gray.ravel(), minlength=256).astype(float)
        total = gray.size
        sum_total = np.dot(np.arange(256), hist)

        sum_b = 0.0
        w_b = 0.0
        var_max = -1.0
        thr = 128

        for i in range(256):
            w_b += hist[i]
            if w_b <= 0:
                continue
            w_f = total - w_b
            if w_f <= 0:
                break

            sum_b += i * hist[i]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = w_b * w_f * (m_b - m_f) ** 2

            if var_between > var_max:
                var_max = var_between
                thr = i

        return int(thr)

    def _binary_edge(self, mask: np.ndarray) -> np.ndarray:
        # 8-neighbor erosion then edge = mask & ~eroded
        h, w = mask.shape
        p = np.pad(mask, 1, constant_values=False)

        er = mask.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                er &= p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]

        edge = mask & (~er)
        return edge

    def _pixels_to_plane(
        self,
        pts_xy_px: np.ndarray,
        center_xy_m: Tuple[float, float],
        drawing_size_m: Tuple[float, float],
        max_points: int,
    ) -> np.ndarray:
        pts = np.array(pts_xy_px, dtype=float).reshape(-1, 2)

        # Downsample early if needed
        if pts.shape[0] > max_points:
            idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(int)
            pts = pts[idx]

        x_px = pts[:, 0]
        y_px = pts[:, 1]

        min_x, max_x = float(np.min(x_px)), float(np.max(x_px))
        min_y, max_y = float(np.min(y_px)), float(np.max(y_px))

        w_px = max(1.0, (max_x - min_x))
        h_px = max(1.0, (max_y - min_y))

        w_m, h_m = drawing_size_m
        scale = min(w_m / w_px, h_m / h_px)

        cx_px = 0.5 * (min_x + max_x)
        cy_px = 0.5 * (min_y + max_y)

        cx_m, cy_m = center_xy_m

        # Pixel y increases downward, plane y is defined upward, so invert y
        x_m = cx_m + (x_px - cx_px) * scale
        y_m = cy_m - (y_px - cy_px) * scale

        return np.stack([x_m, y_m], axis=1)

    def _polyline_waypoints(self, pts_xy_m: np.ndarray) -> List[PlanarWaypoint]:
        pts = np.array(pts_xy_m, dtype=float).reshape(-1, 2)
        if pts.shape[0] < 3:
            raise RuntimeError("Polyline too small")

        wps: List[PlanarWaypoint] = []
        x0, y0 = float(pts[0, 0]), float(pts[0, 1])

        wps.append(PlanarWaypoint(x0, y0, False))
        wps.append(PlanarWaypoint(x0, y0, True))

        for k in range(1, pts.shape[0]):
            wps.append(PlanarWaypoint(float(pts[k, 0]), float(pts[k, 1]), True))

        wps.append(PlanarWaypoint(x0, y0, True))
        wps.append(PlanarWaypoint(x0, y0, False))
        return wps

    def _circle(self, center_xy_m: Tuple[float, float], radius_m: float, points: int) -> List[PlanarWaypoint]:
        cx, cy = center_xy_m
        r = float(radius_m)
        n = max(12, int(points))

        wps: List[PlanarWaypoint] = []
        x0 = cx + r
        y0 = cy

        wps.append(PlanarWaypoint(x0, y0, False))
        wps.append(PlanarWaypoint(x0, y0, True))

        for k in range(1, n + 1):
            th = (2.0 * math.pi) * (k / n)
            x = cx + r * math.cos(th)
            y = cy + r * math.sin(th)
            wps.append(PlanarWaypoint(x, y, True))

        wps.append(PlanarWaypoint(x0, y0, False))
        return wps