# tool_axis_link.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolAxisConfig:
    enabled: bool = False

    uri: str = "serial://tool:3?baud=115200"
    non_blocking: bool = True
    send_buffer_size: int = 64
    receive_buffer_size: int = 64

    up_value: float = 0.0
    down_value: float = 1.0


class ToolAxisLink:
    def __init__(self, config: ToolAxisConfig):
        self.cfg = config
        self._stream = None
        self._connected = False

        if self.cfg.enabled:
            self._connect()

    @property
    def connected(self) -> bool:
        return self._connected

    def _connect(self) -> None:
        from quanser.communications import Stream  # type: ignore

        self._stream = Stream()
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
            self._stream.send_double(float(value))
        except Exception:
            self._connected = False

    def send_pen_state(self, pen_down: bool) -> None:
        self.send_value(self.cfg.down_value if pen_down else self.cfg.up_value)

    def force_up(self) -> None:
        self.send_value(self.cfg.up_value)