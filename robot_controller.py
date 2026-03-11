# robot_controller.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import numpy as np

from kinematics_engine import CartesianWaypoint, DrawingPlane, KinematicsEngine
from tool_axis_link import ToolAxisLink
from vision_processor import DrawingJobSpec, VisionProcessor
from ui_handler import UIEvent, UIEventType


class RobotState(Enum):
    IDLE = auto()
    CALIBRATING = auto()
    DRAWING = auto()
    PAUSED = auto()
    EMERGENCY_STOP = auto()


@dataclass
class DrawingJobRuntime:
    job: DrawingJobSpec
    waypoints: List[CartesianWaypoint]
    index: int = 0


class QArmIO:
    def __init__(self, hardware: int = 0):
        try:
            from pal.products.qarm import QArm  # type: ignore
        except Exception:
            try:
                from Libraries.pal.products.qarm import QArm  # type: ignore
            except Exception as e:
                raise ImportError("Could not import QArm. Ensure Quanser libraries are on PYTHONPATH.") from e

        self.arm = QArm(hardware=hardware)
        self.status_ok = True

    def get_joint_position(self) -> np.ndarray:
        return np.array(self.arm.measJointPosition[0:4], dtype=float).reshape(4)

    def send(self, phi_cmd: np.ndarray, gripper_cmd: float, led_rgb: Optional[np.ndarray] = None) -> None:
        if led_rgb is None:
            led_rgb = np.array([0, 0, 0], dtype=float)
        phi_cmd = np.array(phi_cmd, dtype=float).reshape(4)

        self.arm.read_write_std(phiCMD=phi_cmd, grpCMD=float(gripper_cmd), baseLED=led_rgb)
        self.status_ok = bool(getattr(self.arm, "status", True))

    def terminate(self) -> None:
        try:
            self.arm.terminate()
        except Exception:
            pass


class RobotController:
    def __init__(
        self,
        io: QArmIO,
        kinematics: KinematicsEngine,
        vision: VisionProcessor,
        plane: DrawingPlane,
        tool_axis: Optional[ToolAxisLink] = None,
        sample_rate_hz: float = 50.0,
    ):
        self.io = io
        self.kin = kinematics
        self.vision = vision
        self.plane = plane
        self.tool_axis = tool_axis

        self.state = RobotState.IDLE

        self.sample_rate_hz = float(sample_rate_hz)
        self.dt = 1.0 / self.sample_rate_hz

        self._job_spec = DrawingJobSpec(mode="circle")
        self._job_runtime: Optional[DrawingJobRuntime] = None

        self._last_phi_cmd = np.zeros(4, dtype=float)
        self._last_grip_cmd = 0.1

        self._running = True

        self._home_xyz = np.array([0.30, -0.10, 0.20], dtype=float)

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
            print(f"Image filename set to {ev.payload}")
            return

        if ev.event_type == UIEventType.SET_MODE_CIRCLE:
            self._job_spec.mode = "circle"
            print("Mode set to circle")
            return

        if ev.event_type == UIEventType.SET_MODE_IMAGE:
            self._job_spec.mode = "image"
            print("Mode set to image")
            return

        if ev.event_type == UIEventType.EMERGENCY_STOP:
            self._transition(RobotState.EMERGENCY_STOP)
            return

        if ev.event_type == UIEventType.RESET:
            if self.state in (RobotState.EMERGENCY_STOP, RobotState.PAUSED):
                self._job_runtime = None
                self._transition(RobotState.IDLE)
            return

        if ev.event_type == UIEventType.PAUSE_TOGGLE:
            if self.state == RobotState.DRAWING:
                self._transition(RobotState.PAUSED)
            elif self.state == RobotState.PAUSED:
                self._transition(RobotState.DRAWING)
            return

        if ev.event_type == UIEventType.START:
            if self.state == RobotState.IDLE:
                self._transition(RobotState.CALIBRATING)
            return

    def _transition(self, new_state: RobotState) -> None:
        if new_state == self.state:
            return
        print(f"STATE {self.state.name} -> {new_state.name}")

        # Safety action on entry to EMERGENCY_STOP
        if new_state == RobotState.EMERGENCY_STOP and self.tool_axis is not None:
            self.tool_axis.force_up()

        self.state = new_state

    def update(self) -> None:
        if self.state == RobotState.IDLE:
            self._idle_step()
        elif self.state == RobotState.CALIBRATING:
            self._calibrating_step()
        elif self.state == RobotState.DRAWING:
            self._drawing_step()
        elif self.state == RobotState.PAUSED:
            self._paused_step()
        elif self.state == RobotState.EMERGENCY_STOP:
            self._estop_step()

    def _idle_step(self) -> None:
        led = np.array([0, 1, 0], dtype=float)
        self.io.send(self._last_phi_cmd, self._last_grip_cmd, led_rgb=led)

    def _calibrating_step(self) -> None:
        led = np.array([0, 0, 1], dtype=float)

        q_prev = self.io.get_joint_position()
        try:
            phi_cmd = self.kin.ik(self._home_xyz, q_prev)
        except Exception as e:
            print(f"Calibration IK failed: {e}")
            self._transition(RobotState.EMERGENCY_STOP)
            return

        self._last_phi_cmd = phi_cmd
        self._last_grip_cmd = 0.1
        self.io.send(phi_cmd, self._last_grip_cmd, led_rgb=led)

        try:
            planar = self.vision.build_planar_waypoints(self._job_spec)
            cart = self.kin.planar_to_cartesian(self.plane, planar)
            self._job_runtime = DrawingJobRuntime(job=self._job_spec, waypoints=cart, index=0)
        except Exception as e:
            print(f"Failed to build drawing job: {e}")
            self._transition(RobotState.EMERGENCY_STOP)
            return

        self._transition(RobotState.DRAWING)

    def _drawing_step(self) -> None:
        led = np.array([1, 1, 1], dtype=float)

        if self._job_runtime is None or not self._job_runtime.waypoints:
            print("No job loaded. Returning to IDLE.")
            self._transition(RobotState.IDLE)
            return

        if self._job_runtime.index >= len(self._job_runtime.waypoints):
            print("Drawing complete.")
            self._transition(RobotState.IDLE)
            self._job_runtime = None
            return

        wp = self._job_runtime.waypoints[self._job_runtime.index]
        q_prev = self.io.get_joint_position()

        try:
            phi_cmd = self.kin.ik(np.array([wp.x_m, wp.y_m, wp.z_m], dtype=float), q_prev)
        except Exception as e:
            print(f"Drawing IK failed at index {self._job_runtime.index}: {e}")
            self._transition(RobotState.EMERGENCY_STOP)
            return

        grip = 0.1
        self._last_phi_cmd = phi_cmd
        self._last_grip_cmd = grip
        self.io.send(phi_cmd, grip, led_rgb=led)

        # Optional tool axis command tied to pen state
        if self.tool_axis is not None:
            self.tool_axis.send_pen_state(wp.pen_down)

        self._job_runtime.index += 1

    def _paused_step(self) -> None:
        led = np.array([1, 1, 0], dtype=float)
        self.io.send(self._last_phi_cmd, self._last_grip_cmd, led_rgb=led)

    def _estop_step(self) -> None:
        led = np.array([1, 0, 0], dtype=float)
        self.io.send(self._last_phi_cmd, self._last_grip_cmd, led_rgb=led)