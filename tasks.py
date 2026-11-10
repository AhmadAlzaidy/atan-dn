"""Cross-platform task runner (the Makefile calls it; on Windows without make run ``py -3 tasks.py <task>``).

Tasks: install, test, lint, prepare, tune, train, evaluate, complexity, pipeline.
Variables (command line ``KEY=value`` or environment): MISSION (m1|m2|ops_sat, default m1), CONFIG
(default configs/<MISSION>.yaml), ARGS (extra arguments forwarded to the script).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = {"prepare": "prepare_data.py", "tune": "tune.py", "train": "train.py", "evaluate": "evaluate.py",
           "complexity": "complexity.py", "pipeline": "run_pipeline.py"}


def _run(cmd: list[str]) -> int:
    sys.stderr.write("+ " + " ".join(cmd) + "\n")
    return subprocess.call(cmd, cwd=ROOT)


def main(argv: list[str]) -> int:
    """Run the requested task and return its exit code."""
    opts = {k: v for k, v in (a.split("=", 1) for a in argv if "=" in a)}
    tasks = [a for a in argv if "=" not in a]
    if not tasks or tasks[0] in {"-h", "--help", "help"}:
        sys.stderr.write(__doc__ + "\n")
        return 0
    task = tasks[0]
    mission = opts.get("MISSION", os.environ.get("MISSION", "m1"))
    config = opts.get("CONFIG", os.environ.get("CONFIG", f"configs/{mission}.yaml"))
    extra = shlex.split(opts.get("ARGS", os.environ.get("ARGS", "")))
    py = sys.executable
    if task == "install":
        for f, cmd in (("pyproject.toml", [py, "-m", "pip", "install", "-e", ".[dev]"]),
                       ("requirements.txt", [py, "-m", "pip", "install", "-r", "requirements.txt"])):
            if (ROOT / f).is_file():
                return _run(cmd)
        sys.stderr.write("neither pyproject.toml nor requirements.txt found\n")
        return 1
    if task == "test":
        return _run([py, "-m", "pytest", "tests", "-q", *extra])
    if task == "lint":
        if subprocess.call([py, "-m", "ruff", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            return _run([py, "-m", "ruff", "check", "atandn", "scripts", "tests", "tasks.py"])
        sys.stderr.write("ruff not installed; falling back to a byte-compile syntax check\n")
        return _run([py, "-m", "compileall", "-q", "atandn", "scripts", "tests", "tasks.py"])
    if task in SCRIPTS:
        return _run([py, str(Path("scripts") / SCRIPTS[task]), "--config", config, *extra])
    sys.stderr.write(f"unknown task '{task}'\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
