# main.py
from __future__ import annotations

import numpy as np

from kinematics_engine import DrawingPlane, KinematicsEngine, TrajectoryTimer
from robot_controller import QArmIO, RobotController
from tool_axis_link import ToolAxisConfig, ToolAxisLink
from ui_handler import TerminalUI
from vision_processor import VisionProcessor


def main() -> None:
    # Toggle this to enable the microcontroller link for the extra tool axis
    # Stream.connect(uri, non_blocking, send_buffer_size, receive_buffer_size) is the standard Python API. :contentReference[oaicite:4]{index=4}
    ENABLE_TOOL_AXIS_LINK = False

    # Serial URI syntax is serial://hostname:port?option=value,... :contentReference[oaicite:5]{index=5}
    # On Windows, port 3 maps to COM3. :contentReference[oaicite:6]{index=6}
    TOOL_AXIS_URI = "serial://tool:3?baud=115200"

    tool_axis = ToolAxisLink(
        ToolAxisConfig(
            enabled=ENABLE_TOOL_AXIS_LINK,
            uri=TOOL_AXIS_URI,
            non_blocking=True,
            up_value=0.0,
            down_value=1.0,
        )
    )

    plane = DrawingPlane(
        origin_base_m=np.array([0.25, -0.15, 0.0]),
        x_axis_base=np.array([1.0, 0.0, 0.0]),
        y_axis_base=np.array([0.0, 1.0, 0.0]),
        z_pen_down_m=0.050,
        z_pen_up_m=0.120,
    )

    ui = TerminalUI()
    ui.start()

    io = QArmIO(hardware=1)
    kin = KinematicsEngine(gamma_rad=0.0)
    vision = VisionProcessor()

    controller = RobotController(
        io=io,
        kinematics=kin,
        vision=vision,
        plane=plane,
        tool_axis=tool_axis,
        sample_rate_hz=50.0,
    )

    timer = TrajectoryTimer(rate_hz=50.0)

    try:
        while controller.running:
            while not ui.events.empty():
                ev = ui.events.get_nowait()
                controller.handle_event(ev)

            controller.update()
            timer.sleep_to_next()
    finally:
        tool_axis.close()
        io.terminate()


if __name__ == "__main__":
    main()