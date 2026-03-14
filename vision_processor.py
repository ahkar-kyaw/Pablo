# vision_processor.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from kinematics_engine import PlanarWaypoint


@dataclass
class DrawingJobSpec:
    image_filename: Optional[str] = None
    drawing_size_m: Tuple[float, float] = (0.18, 0.18)
    center_xy_m: Tuple[float, float] = (0.10, 0.10)
    ink_is_dark: bool = True
    max_points: int = 900


class VisionProcessor:
    def build_planar_waypoints(self, job: DrawingJobSpec) -> List[PlanarWaypoint]:
        if not job.image_filename:
            raise ValueError("No image loaded. Use load <filename> first.")

        pts_px = self._extract_contour_pixels(job.image_filename, job.ink_is_dark)
        pts_xy_m = self._pixels_to_plane(
            pts_px,
            center_xy_m=job.center_xy_m,
            drawing_size_m=job.drawing_size_m,
            max_points=job.max_points,
        )
        return self._polyline_waypoints(pts_xy_m)

    def _local_path(self, filename: str) -> Path:
        here = Path(__file__).resolve().parent
        return (here / filename).resolve()

    def _extract_contour_pixels(self, filename: str, ink_is_dark: bool) -> np.ndarray:
        path = self._local_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Prefer OpenCV
        try:
            import cv2  # type: ignore

            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError("OpenCV could not load image")

            thresh_flag = cv2.THRESH_BINARY_INV if ink_is_dark else cv2.THRESH_BINARY
            _, bw = cv2.threshold(img, 0, 255, thresh_flag + cv2.THRESH_OTSU)

            found = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = found[0] if len(found) == 2 else found[1]
            if not contours:
                raise RuntimeError("No contours found")

            contour = max(contours, key=cv2.contourArea)
            pts = contour[:, 0, :]
            return pts.astype(float)

        except ImportError:
            pass
        except Exception:
            pass

        # Pillow fallback
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise ImportError("Install opencv-python or pillow to load images") from e

        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.uint8)

        t = self._otsu_threshold(arr)
        mask = arr < t if ink_is_dark else arr > t

        edge = self._binary_edge(mask)
        coords = np.argwhere(edge)
        if coords.shape[0] < 30:
            raise RuntimeError("Too few edge points. Use a higher contrast image.")

        rows = coords[:, 0].astype(float)
        cols = coords[:, 1].astype(float)
        cy = float(np.mean(rows))
        cx = float(np.mean(cols))
        ang = np.arctan2(rows - cy, cols - cx)
        order = np.argsort(ang)
        pts = np.stack([cols[order], rows[order]], axis=1)
        return pts

    def _otsu_threshold(self, gray: np.ndarray) -> int:
        hist = np.bincount(gray.ravel(), minlength=256).astype(float)
        total = gray.size
        sum_total = float(np.dot(np.arange(256), hist))

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
        h, w = mask.shape
        p = np.pad(mask, 1, constant_values=False)

        er = mask.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                er &= p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]

        return mask & (~er)

    def _pixels_to_plane(
        self,
        pts_xy_px: np.ndarray,
        center_xy_m: Tuple[float, float],
        drawing_size_m: Tuple[float, float],
        max_points: int,
    ) -> np.ndarray:
        pts = np.array(pts_xy_px, dtype=float).reshape(-1, 2)

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

        # invert y (pixel y down, plane y up)
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