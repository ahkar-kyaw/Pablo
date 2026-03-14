# ui_handler.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue
from threading import Thread
from typing import Optional


class UIEventType(Enum):
    LOAD_IMAGE_FILENAME = auto()
    PREVIEW = auto()
    START = auto()
    STOP = auto()
    RESET = auto()
    HOME = auto()
    GRIP = auto()
    ROTATE_WRIST_DEG = auto()
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
            "  load <filename>      load image from this folder, example load circle.png\n"
            "  preview              live preview of interpreted path, no robot motion\n"
            "  start                execute drawing\n"
            "  stop                 emergency stop\n"
            "  reset                return to idle from emergency stop\n"
            "  home                 move robot to home pose\n"
            "  grip open            open gripper\n"
            "  grip close           close gripper with current limiting\n"
            "  rotate <deg>         set wrist absolute, example rotate 90\n"
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
            arg = rest[0].strip() if rest else ""

            if cmd == "load":
                if not arg:
                    print("Usage load <filename>")
                else:
                    self.events.put(UIEvent(UIEventType.LOAD_IMAGE_FILENAME, payload=arg))

            elif cmd == "preview":
                self.events.put(UIEvent(UIEventType.PREVIEW))

            elif cmd == "start":
                self.events.put(UIEvent(UIEventType.START))

            elif cmd == "stop":
                self.events.put(UIEvent(UIEventType.STOP))

            elif cmd == "reset":
                self.events.put(UIEvent(UIEventType.RESET))

            elif cmd == "home":
                self.events.put(UIEvent(UIEventType.HOME))

            elif cmd == "grip":
                a = arg.lower()
                if a not in ("open", "close"):
                    print("Usage grip open|close")
                else:
                    self.events.put(UIEvent(UIEventType.GRIP, payload=a))

            elif cmd == "rotate":
                if not arg:
                    print("Usage rotate <deg>")
                else:
                    self.events.put(UIEvent(UIEventType.ROTATE_WRIST_DEG, payload=arg))

            elif cmd == "help":
                self._print_help()
                self.events.put(UIEvent(UIEventType.HELP))

            elif cmd == "quit":
                self.events.put(UIEvent(UIEventType.QUIT))
                return

            else:
                print("Unknown command. Type help.")