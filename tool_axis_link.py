# tool_axis_link.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolAxisConfig:
    enabled: bool = False

    # Serial URI syntax follows QUARC conventions
    # Example for Windows COM3 with baud option
    # serial://tool:3?baud=115200
    uri: str = "serial://tool:3?baud=115200"

    non_blocking: bool = True
    send_buffer_size: int = 64
    receive_buffer_size: int = 64

    up_value: float = 0.0
    down_value: float = 1.0


class ToolAxisLink:
    """
    Optional microcontroller link for an extra tool axis using Quanser Stream.

    When disabled, all methods become no-ops.

    The Stream.connect signature in Quanser Python API is
    Stream.connect(uri, non_blocking=False, send_buffer_size=..., receive_buffer_size=...). :contentReference[oaicite:2]{index=2}
    """

    def __init__(self, config: ToolAxisConfig):
        self.cfg = config
        self._stream = None
        self._pollflag = None
        self._connected = False

        if self.cfg.enabled:
            self._connect()

    @property
    def connected(self) -> bool:
        return self._connected

    def _connect(self) -> None:
        try:
            from quanser.communications import Stream, PollFlag  # type: ignore
        except Exception as e:
            raise ImportError("Quanser communications API not available. Install Quanser Python packages.") from e

        self._stream = Stream()
        self._pollflag = PollFlag

        # Non-blocking is recommended so the robot loop does not stall. :contentReference[oaicite:3]{index=3}
        self._stream.connect(
            self.cfg.uri,
            self.cfg.non_blocking,
            self.cfg.send_buffer_size,
            self.cfg.receive_buffer_size,
        )
        self._connected = True

    def close(self) -> None:
        if not self.cfg.enabled or self._stream is None:
            return
        try:
            self._stream.close()
        except Exception:
            pass
        self._connected = False

    def send_value(self, value: float) -> None:
        if not self.cfg.enabled or self._stream is None or not self._connected:
            return

        try:
            # send_double exists in Quanser communications Python API
            self._stream.send_double(float(value))
            # Flush if possible on non-blocking streams
            if self._pollflag is not None:
                try:
                    if self._stream.poll(0, self._pollflag.FLUSH) > 0:
                        self._stream.flush()
                except Exception:
                    # If poll/flush fails, keep running and try again next cycle
                    pass
        except Exception:
            # If the link fails mid-run, degrade gracefully
            self._connected = False

    def send_pen_state(self, pen_down: bool) -> None:
        self.send_value(self.cfg.down_value if pen_down else self.cfg.up_value)

    def force_up(self) -> None:
        self.send_value(self.cfg.up_value)