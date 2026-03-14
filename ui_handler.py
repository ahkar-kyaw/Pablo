# ui_handler.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue
from threading import Thread
from typing import Optional


class UIEventType(Enum):
    START = auto()
    PAUSE_TOGGLE = auto()
    EMERGENCY_STOP = auto()
    RESET = auto()
    SET_MODE_CIRCLE = auto()
    SET_MODE_IMAGE = auto()
    LOAD_IMAGE_FILENAME = auto()
    QUIT = auto()
    HELP = auto()


@dataclass(frozen=True)
class UIEvent:
    event_type: UIEventType
    payload: Optional[str] = None


class TerminalUI:
    def __init__(self):
        self.events: "Queue[UIEvent]" = Queue()
        self._thread = Thread(target=self._reader_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()
        self._print_help()

    def _print_help(self) -> None:
        print(
            "\nCommands\n"
            "  start             -> IDLE to CALIBRATING to DRAWING\n"
            "  pause             -> toggle PAUSED during DRAWING\n"
            "  stop              -> EMERGENCY_STOP from any state\n"
            "  reset             -> return to IDLE from PAUSED or EMERGENCY_STOP\n"
            "  mode circle       -> draw a generated circle\n"
            "  mode image        -> draw from an image file in this folder\n"
            "  load <filename>   -> set image filename, example load circle.png\n"
            "  help\n"
            "  quit\n"
        )

    def _reader_loop(self) -> None:
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.events.put(UIEvent(UIEventType.QUIT))
                return

            if not line:
                continue

            cmd, *rest = line.split(maxsplit=1)
            cmd = cmd.lower()

            if cmd == "start":
                self.events.put(UIEvent(UIEventType.START))
            elif cmd == "pause":
                self.events.put(UIEvent(UIEventType.PAUSE_TOGGLE))
            elif cmd == "stop":
                self.events.put(UIEvent(UIEventType.EMERGENCY_STOP))
            elif cmd == "reset":
                self.events.put(UIEvent(UIEventType.RESET))
            elif cmd == "mode":
                arg = rest[0].strip().lower() if rest else ""
                if arg == "circle":
                    self.events.put(UIEvent(UIEventType.SET_MODE_CIRCLE))
                elif arg == "image":
                    self.events.put(UIEvent(UIEventType.SET_MODE_IMAGE))
                else:
                    print("Unknown mode. Use mode circle or mode image")
            elif cmd == "load":
                name = rest[0].strip() if rest else ""
                if not name:
                    print("Usage load <filename>")
                else:
                    self.events.put(UIEvent(UIEventType.LOAD_IMAGE_FILENAME, payload=name))
            elif cmd == "help":
                self._print_help()
                self.events.put(UIEvent(UIEventType.HELP))
            elif cmd == "quit":
                self.events.put(UIEvent(UIEventType.QUIT))
                return
            else:
                print("Unknown command. Type help.")