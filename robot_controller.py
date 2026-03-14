# robot_controller.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from kinematics_engine import CartesianWaypoint, DrawingPlane, KinematicsEngine
from ui_handler import UIEvent, UIEventType
from vision_processor import DrawingJobSpec, VisionProcessor


class RobotState(Enum):
    IDLE = auto()
    PREVIEW = auto()
    CALIBRATING = auto()
    HOMING = auto()
    DRAWING = auto()
    GRIP_OPENING = auto()
    GRIP_CLOSING = auto()
    ROTATE_EE = auto()
    EMERGENCY_STOP = auto()


@dataclass
class DrawingJobRuntime:
    job: DrawingJobSpec
    waypoints: List[CartesianWaypoint]
    index: int = 0


class QArmIO:
    """
    Wraps pal.products.qarm.QArm.
    """

    def __init__(self, hardware: int = 1):
        try:
            from pal.products.qarm import QArm  # type: ignore
        except Exception:
            from Libraries.pal.products.qarm import QArm  # type: ignore

        self.arm = QArm(hardware=hardware)
        self.status_ok = True

        try:
            self.arm.read_std()
        except Exception:
            pass

    def get_joint_position(self) -> np.ndarray:
        return np.array(self.arm.measJointPosition[0:4], dtype=float).reshape(4)

    def get_joint_currents(self) -> Optional[np.ndarray]:
        cur = getattr(self.arm, "measJointCurrent", None)
        if cur is None or len(cur) < 4:
            return None
        return np.array(cur[0:4], dtype=float).reshape(4)

    def get_gripper_current(self) -> Optional[float]:
        cur = getattr(self.arm, "measJointCurrent", None)
        if cur is None or len(cur) < 5:
            return None
        return float(cur[4])

    def send(self, phi_cmd: np.ndarray, gripper_cmd: float, led_rgb: Optional[np.ndarray] = None) -> None:
        if led_rgb is None:
            led_rgb = np.array([0.0, 0.0, 0.0], dtype=float)

        phi_cmd = np.array(phi_cmd, dtype=float).reshape(4)
        self.arm.read_write_std(phiCMD=phi_cmd, grpCMD=float(gripper_cmd), baseLED=led_rgb)
        self.status_ok = bool(getattr(self.arm, "status", True))

    def terminate(self) -> None:
        try:
            self.arm.terminate()
        except Exception:
            pass


class RobotController:
    # Gripper commands
    GRIP_OPEN_CMD = 0.1
    GRIP_MAX_CLOSE_CMD = 0.85

    # Gripper closing behavior
    GRIP_STEP_PER_TICK = 0.01
    GRIP_CURRENT_LIMIT_A = 1.0
    GRIP_OVERCURRENT_COUNT = 5
    GRIP_BACKOFF = 0.02
    GRIP_CLOSE_TIMEOUT_S = 2.0

    # Wrist rotate smoothing
    WRIST_MAX_STEP_DEG = 5.0

    # Z compliance
    Z_COMPLIANCE_ENABLED = True

    # Contact detect uses joint currents as a proxy for contact force
    CONTACT_CURRENT_RISE_A = 0.35
    CONTACT_HYSTERESIS_A = 0.10

    # Descent step and compliance gain
    Z_DESCENT_STEP_M = 0.0008
    Z_ADMITTANCE_M_PER_A = 0.0009

    # Clamp around contact Z to avoid digging into the surface
    Z_ABOVE_CONTACT_MAX_M = 0.006
    Z_BELOW_CONTACT_MAX_M = 0.001

    # Filter to avoid noisy thresholds
    CURRENT_LPF_ALPHA = 0.25

    def __init__(
        self,
        io: QArmIO,
        kinematics: KinematicsEngine,
        vision: VisionProcessor,
        plane: DrawingPlane,
        sample_rate_hz: float = 50.0,
    ):
        self.io = io
        self.kin = kinematics
        self.vision = vision
        self.plane = plane

        self.state = RobotState.IDLE
        self.sample_rate_hz = float(sample_rate_hz)

        self._job_spec = DrawingJobSpec()
        self._job_runtime: Optional[DrawingJobRuntime] = None
        self._job_dirty = True

        q0 = self.io.get_joint_position()
        self._last_phi_cmd = q0.copy()
        self._last_grip_cmd = self.GRIP_OPEN_CMD

        # Wrist gamma used for IK during home and drawing
        self._wrist_gamma_rad = float(self._last_phi_cmd[3])

        self._running = True

        # Home pose in task space
        self._home_xyz = np.array([0.30, -0.10, 0.20], dtype=float)
        self._home_cycles_remaining = 0
        self._after_home_state = RobotState.IDLE

        # Preview
        self._preview_fig = None
        self._preview_anim = None

        # Gripper close
        self._grip_overcurrent = 0
        self._grip_timeout_cycles = 0

        # Rotate
        self._wrist_target_rad: Optional[float] = None

        # Z compliance runtime
        self._contact_active = False
        self._contact_z_m: Optional[float] = None
        self._z_cmd_m = float(self.plane.z_pen_up_m)

        self._baseline_current: Optional[float] = None
        self._filtered_current = 0.0

    @property
    def running(self) -> bool:
        return self._running and self.io.status_ok

    def handle_event(self, ev: UIEvent) -> None:
        if ev.event_type == UIEventType.QUIT:
            self._running = False
            return

        if ev.event_type == UIEventType.HELP:
            return

        if ev.event_type == UIEventType.LOAD_IMAGE_FILENAME:
            self._job_spec.image_filename = ev.payload
            self._job_dirty = True
            print(f"Loaded: {ev.payload}")
            return

        if ev.event_type == UIEventType.STOP:
            self._transition(RobotState.EMERGENCY_STOP)
            return

        if ev.event_type == UIEventType.RESET:
            if self.state == RobotState.EMERGENCY_STOP:
                self._transition(RobotState.IDLE)
            return

        if self.state == RobotState.EMERGENCY_STOP:
            return

        if ev.event_type == UIEventType.PREVIEW:
            if self.state != RobotState.IDLE:
                print("Preview is available from IDLE only")
                return
            self._transition(RobotState.PREVIEW)
            return

        if ev.event_type == UIEventType.START:
            if self.state != RobotState.IDLE:
                print("Start is available from IDLE only")
                return
            self._transition(RobotState.CALIBRATING)
            return

        if ev.event_type == UIEventType.HOME:
            if self.state != RobotState.IDLE:
                print("Home is available from IDLE only")
                return
            self._after_home_state = RobotState.IDLE
            self._home_cycles_remaining = int(2.0 * self.sample_rate_hz)
            self._transition(RobotState.HOMING)
            return

        if ev.event_type == UIEventType.GRIP:
            if self.state != RobotState.IDLE:
                print("Grip is available from IDLE only")
                return
            if ev.payload == "open":
                self._transition(RobotState.GRIP_OPENING)
            elif ev.payload == "close":
                self._grip_overcurrent = 0
                self._grip_timeout_cycles = int(self.GRIP_CLOSE_TIMEOUT_S * self.sample_rate_hz)
                self._transition(RobotState.GRIP_CLOSING)
            return

        if ev.event_type == UIEventType.ROTATE_WRIST_DEG:
            if self.state != RobotState.IDLE:
                print("Rotate is available from IDLE only")
                return
            tgt = self._parse_rotate_payload(ev.payload or "")
            if tgt is None:
                return
            self._wrist_target_rad = tgt
            self._transition(RobotState.ROTATE_EE)
            return

    def update(self) -> None:
        if self.state == RobotState.IDLE:
            self._idle_step()
        elif self.state == RobotState.PREVIEW:
            self._preview_step()
        elif self.state == RobotState.CALIBRATING:
            self._calibrating_step()
        elif self.state == RobotState.HOMING:
            self._homing_step()
        elif self.state == RobotState.DRAWING:
            self._drawing_step()
        elif self.state == RobotState.GRIP_OPENING:
            self._grip_open_step()
        elif self.state == RobotState.GRIP_CLOSING:
            self._grip_close_step()
        elif self.state == RobotState.ROTATE_EE:
            self._rotate_step()
        elif self.state == RobotState.EMERGENCY_STOP:
            self._estop_step()

    def _transition(self, new_state: RobotState) -> None:
        if new_state == self.state:
            return

        print(f"STATE {self.state.name} -> {new_state.name}")

        if self.state == RobotState.PREVIEW and new_state != RobotState.PREVIEW:
            self._stop_preview()

        if new_state in (RobotState.IDLE, RobotState.EMERGENCY_STOP):
            self._reset_contact_state()

        self.state = new_state

    def _reset_contact_state(self) -> None:
        self._contact_active = False
        self._contact_z_m = None
        self._z_cmd_m = float(self.plane.z_pen_up_m)
        self._baseline_current = None
        self._filtered_current = 0.0

    def _send_hold(self, led: np.ndarray) -> None:
        self.io.send(self._last_phi_cmd, self._last_grip_cmd, led_rgb=led)

    def _idle_step(self) -> None:
        self._send_hold(np.array([0.0, 1.0, 0.0], dtype=float))

    def _calibrating_step(self) -> None:
        if not getattr(self._job_spec, "image_filename", None):
            print("No image loaded. Use load <filename> first")
            self._transition(RobotState.IDLE)
            return

        try:
            self._prepare_job_if_needed()
        except Exception as e:
            print(f"Failed to build drawing job: {e}")
            self._transition(RobotState.EMERGENCY_STOP)
            return

        self._after_home_state = RobotState.DRAWING
        self._home_cycles_remaining = int(2.0 * self.sample_rate_hz)
        self._transition(RobotState.HOMING)

    def _homing_step(self) -> None:
        led = np.array([0.0, 0.5, 1.0], dtype=float)
        q_prev = self.io.get_joint_position()
        try:
            phi_cmd = self.kin.ik(self._home_xyz, q_prev, gamma_rad=self._wrist_gamma_rad)
        except Exception as e:
            print(f"Homing IK failed: {e}")
            self._transition(RobotState.EMERGENCY_STOP)
            return

        self._last_phi_cmd = phi_cmd
        self.io.send(self._last_phi_cmd, self._last_grip_cmd, led_rgb=led)

        self._home_cycles_remaining = max(0, self._home_cycles_remaining - 1)
        if self._home_cycles_remaining == 0:
            self._transition(self._after_home_state)

    def _preview_step(self) -> None:
        led = np.array([0.6, 0.6, 0.0], dtype=float)
        self._send_hold(led)

        if self._preview_fig is None:
            try:
                self._prepare_job_if_needed()
                assert self._job_runtime is not None
                self._start_preview(self._job_runtime.waypoints)
                print("Preview running. Close the preview window to return to IDLE.")
            except Exception as e:
                print(f"Preview failed: {e}")
                self._transition(RobotState.IDLE)
                return

        try:
            plt.pause(0.001)
        except Exception:
            pass

        try:
            if self._preview_fig is not None and not plt.fignum_exists(self._preview_fig.number):
                self._stop_preview()
                self._transition(RobotState.IDLE)
        except Exception:
            self._stop_preview()
            self._transition(RobotState.IDLE)

    def _start_preview(self, waypoints: List[CartesianWaypoint]) -> None:
        wp = np.array([[w.x_m, w.y_m, 0.0 if w.pen_down else 1.0] for w in waypoints], dtype=float)
        x, y, up = wp[:, 0], wp[:, 1], wp[:, 2].astype(int)

        fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x.min() - 0.05, x.max() + 0.05)
        ax.set_ylim(y.min() - 0.05, y.max() + 0.05)
        # ax.invert_xaxis()
        # ax.invert_yaxis()
        ax.set_title("Preview: pen-down path")

        down_line, = ax.plot([], [], lw=1)
        pen_dot, = ax.plot([], [], marker="o", markersize=4)

        down_x: List[float] = []
        down_y: List[float] = []
        frames = list(range(1, len(wp)))

        def init():
            down_line.set_data([], [])
            pen_dot.set_data([], [])
            return down_line, pen_dot

        def update(k):
            nonlocal down_x, down_y
            x0, y0 = x[k - 1], y[k - 1]
            x1, y1 = x[k], y[k]
            pen_dot.set_data([x1], [y1])

            if up[k] == 0:
                if not down_x:
                    down_x.extend([x0, x1])
                    down_y.extend([y0, y1])
                else:
                    down_x.append(x1)
                    down_y.append(y1)

            down_line.set_data(down_x, down_y)
            return down_line, pen_dot

        anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval=10, blit=True, repeat=False)

        self._preview_fig = fig
        self._preview_anim = anim
        plt.show(block=False)

    def _stop_preview(self) -> None:
        if self._preview_fig is not None:
            try:
                plt.close(self._preview_fig)
            except Exception:
                pass
        self._preview_fig = None
        self._preview_anim = None

    def _drawing_step(self) -> None:
        led = np.array([1.0, 1.0, 1.0], dtype=float)

        if self._job_runtime is None or not self._job_runtime.waypoints:
            print("No job loaded. Returning to IDLE.")
            self._transition(RobotState.IDLE)
            return

        if self._job_runtime.index >= len(self._job_runtime.waypoints):
            print("Drawing complete.")
            self._job_runtime = None
            self._reset_contact_state()
            self._transition(RobotState.IDLE)
            return

        wp = self._job_runtime.waypoints[self._job_runtime.index]

        if not wp.pen_down or not self.Z_COMPLIANCE_ENABLED:
            if not wp.pen_down:
                self._reset_contact_state()

            q_prev = self.io.get_joint_position()
            try:
                phi_cmd = self.kin.ik(
                    np.array([wp.x_m, wp.y_m, wp.z_m], dtype=float),
                    q_prev,
                    gamma_rad=self._wrist_gamma_rad,
                )
            except Exception as e:
                print(f"Drawing IK failed at index {self._job_runtime.index}: {e}")
                self._transition(RobotState.EMERGENCY_STOP)
                return

            self._last_phi_cmd = phi_cmd
            self.io.send(self._last_phi_cmd, self._last_grip_cmd, led_rgb=led)
            self._job_runtime.index += 1
            return

        # Pen down with compliance
        metric = self._read_contact_metric()
        if self._baseline_current is None:
            self._baseline_current = metric

        if not self._contact_active:
            # Touch down phase
            self._z_cmd_m = min(self._z_cmd_m, float(self.plane.z_pen_up_m))
            next_z = self._z_cmd_m - self.Z_DESCENT_STEP_M
            z_min = float(self.plane.z_pen_down_m)
            if next_z < z_min:
                next_z = z_min

            q_prev = self.io.get_joint_position()
            try:
                phi_cmd = self.kin.ik(
                    np.array([wp.x_m, wp.y_m, next_z], dtype=float),
                    q_prev,
                    gamma_rad=self._wrist_gamma_rad,
                )
            except Exception as e:
                print(f"Touch down IK failed: {e}")
                self._transition(RobotState.EMERGENCY_STOP)
                return

            self._last_phi_cmd = phi_cmd
            self._z_cmd_m = float(next_z)
            self.io.send(self._last_phi_cmd, self._last_grip_cmd, led_rgb=led)

            metric_after = self._read_contact_metric()
            rise = metric_after - float(self._baseline_current)

            if rise > self.CONTACT_CURRENT_RISE_A or abs(self._z_cmd_m - z_min) < 1e-6:
                self._contact_active = True
                self._contact_z_m = float(self._z_cmd_m)

            return

        # Compliant draw phase
        assert self._contact_z_m is not None
        metric = self._read_contact_metric()
        rise = metric - float(self._baseline_current)

        target_rise = self.CONTACT_CURRENT_RISE_A
        error = rise - target_rise

        z_adjust = self.Z_ADMITTANCE_M_PER_A * error
        z_next = self._z_cmd_m + z_adjust

        z_upper = self._contact_z_m + self.Z_ABOVE_CONTACT_MAX_M
        z_lower = self._contact_z_m - self.Z_BELOW_CONTACT_MAX_M
        z_next = float(np.clip(z_next, z_lower, z_upper))

        q_prev = self.io.get_joint_position()
        try:
            phi_cmd = self.kin.ik(
                np.array([wp.x_m, wp.y_m, z_next], dtype=float),
                q_prev,
                gamma_rad=self._wrist_gamma_rad,
            )
        except Exception as e:
            print(f"Compliant draw IK failed at index {self._job_runtime.index}: {e}")
            self._transition(RobotState.EMERGENCY_STOP)
            return

        self._last_phi_cmd = phi_cmd
        self._z_cmd_m = float(z_next)
        self.io.send(self._last_phi_cmd, self._last_grip_cmd, led_rgb=led)

        # If contact is lost, re enter touch down
        if rise < (target_rise - self.CONTACT_HYSTERESIS_A):
            self._contact_active = False

        self._job_runtime.index += 1

    def _read_contact_metric(self) -> float:
        """
        Contact proxy metric from joint currents.
        Uses joints 2 and 3 since they react strongly to vertical contact at the tip.
        """
        cur = self.io.get_joint_currents()
        if cur is None:
            return 0.0

        metric_raw = float(abs(cur[1]) + abs(cur[2]))
        self._filtered_current = (
            (1.0 - self.CURRENT_LPF_ALPHA) * self._filtered_current
            + self.CURRENT_LPF_ALPHA * metric_raw
        )
        return float(self._filtered_current)

    def _grip_open_step(self) -> None:
        led = np.array([0.8, 0.2, 0.8], dtype=float)
        self._last_grip_cmd = self.GRIP_OPEN_CMD
        self.io.send(self._last_phi_cmd, self._last_grip_cmd, led_rgb=led)
        self._transition(RobotState.IDLE)

    def _grip_close_step(self) -> None:
        led = np.array([0.8, 0.2, 0.8], dtype=float)

        self._last_grip_cmd = min(self.GRIP_MAX_CLOSE_CMD, self._last_grip_cmd + self.GRIP_STEP_PER_TICK)
        self.io.send(self._last_phi_cmd, self._last_grip_cmd, led_rgb=led)

        cur = self.io.get_gripper_current()
        if cur is not None and cur > self.GRIP_CURRENT_LIMIT_A:
            self._grip_overcurrent += 1
        else:
            self._grip_overcurrent = 0

        self._grip_timeout_cycles = max(0, self._grip_timeout_cycles - 1)

        if self._grip_overcurrent >= self.GRIP_OVERCURRENT_COUNT or self._grip_timeout_cycles == 0:
            self._last_grip_cmd = max(self.GRIP_OPEN_CMD, self._last_grip_cmd - self.GRIP_BACKOFF)
            self.io.send(self._last_phi_cmd, self._last_grip_cmd, led_rgb=led)
            self._transition(RobotState.IDLE)

    def _rotate_step(self) -> None:
        led = np.array([0.2, 0.8, 0.8], dtype=float)

        if self._wrist_target_rad is None:
            self._transition(RobotState.IDLE)
            return

        target = float(np.arctan2(np.sin(self._wrist_target_rad), np.cos(self._wrist_target_rad)))
        cur = float(self._last_phi_cmd[3])

        max_step = float(np.deg2rad(self.WRIST_MAX_STEP_DEG))
        step = float(np.clip(target - cur, -max_step, max_step))

        q_cmd = self._last_phi_cmd.copy()
        q_cmd[3] = cur + step

        self._last_phi_cmd = q_cmd
        self.io.send(self._last_phi_cmd, self._last_grip_cmd, led_rgb=led)

        if abs(target - float(self._last_phi_cmd[3])) < float(np.deg2rad(1.0)):
            self._wrist_gamma_rad = float(self._last_phi_cmd[3])
            self._transition(RobotState.IDLE)

    def _estop_step(self) -> None:
        self._send_hold(np.array([1.0, 0.0, 0.0], dtype=float))

    def _prepare_job_if_needed(self) -> None:
        if self._job_runtime is not None and not self._job_dirty:
            return

        planar = self.vision.build_planar_waypoints(self._job_spec)
        cart = self.kin.planar_to_cartesian(self.plane, planar)
        self._job_runtime = DrawingJobRuntime(job=self._job_spec, waypoints=cart, index=0)
        self._job_dirty = False

        self._reset_contact_state()

    def _parse_rotate_payload(self, s: str) -> Optional[float]:
        s = s.strip()
        if not s:
            return None
        try:
            deg = float(s)
        except Exception:
            print("Invalid rotate value")
            return None
        return float(np.deg2rad(deg))