# main.py
from __future__ import annotations

import numpy as np

from kinematics_engine import DrawingPlane, KinematicsEngine, TrajectoryTimer
from robot_controller import QArmIO, RobotController
from ui_handler import TerminalUI
from vision_processor import VisionProcessor


def main() -> None:
    plane = DrawingPlane(
        origin_base_m=np.array([0.25, -0.15, 0.0]),
        x_axis_base=np.array([1.0, 0.0, 0.0]),
        y_axis_base=np.array([0.0, 1.0, 0.0]),
        z_pen_down_m=0.050,
        z_pen_up_m=0.120,
    )

    io = QArmIO(hardware=1)  # set 0 for virtual
    kin = KinematicsEngine()
    vision = VisionProcessor()

    controller = RobotController(
        io=io,
        kinematics=kin,
        vision=vision,
        plane=plane,
        sample_rate_hz=50.0,
    )

    ui = TerminalUI()
    ui.start()

    timer = TrajectoryTimer(rate_hz=50.0)

    try:
        while controller.running:
            while not ui.events.empty():
                controller.handle_event(ui.events.get_nowait())

            controller.update()
            timer.sleep_to_next()
    finally:
        io.terminate()


if __name__ == "__main__":
    main()