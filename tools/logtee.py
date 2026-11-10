"""Run a command, show its output live, and keep a copy in a log file.

    python tools/logtee.py logs/runs/<ts>/agent.log -- python src/agent.py dev

Colour codes are stripped in the file and every line gets a wall-clock timestamp, so the
logs of the agent, the bridge and ngrok can be lined up. Exits with the command's code.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) < 3 or argv[1] != "--":
        print(__doc__)
        return 2
    log_path, cmd = Path(argv[0]), argv[2:]
    if cmd[0] == "python":
        cmd[0] = sys.executable  # the project's own Python
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S} $ {' '.join(argv[2:])}\n")
        log.flush()
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        except FileNotFoundError:
            msg = f"command not found: {cmd[0]}"
            print(msg)
            log.write(msg + "\n")
            return 127
        assert proc.stdout is not None
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace")
                sys.stdout.write(line)
                sys.stdout.flush()
                clean = ANSI.sub("", line).replace("\r", "").rstrip("\n")
                log.write(f"{dt.datetime.now():%H:%M:%S.%f}"[:-3] + f" | {clean}\n")
                log.flush()
        except KeyboardInterrupt:
            proc.terminate()
        code = proc.wait()
        log.write(f"===== exited with code {code}\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
