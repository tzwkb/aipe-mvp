"""Run the standard AIPE verification commands.

The default path runs inside docker compose so the checks use the same runtime
as the API container.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build_docker_check_command(root: Path, pytest_args: list[str]) -> list[str]:
    pytest_tail = " ".join(pytest_args) if pytest_args else ""
    shell = "python -m pip install pytest -q && python -m compileall -q app scripts tests && python -m pytest -q"
    if pytest_tail:
        shell = f"{shell} {pytest_tail}"
    return [
        "docker",
        "compose",
        "run",
        "--rm",
        "-v",
        f"{root}:/work",
        "-w",
        "/work",
        "--entrypoint",
        "sh",
        "api",
        "-c",
        shell,
    ]


def build_local_check_commands(pytest_args: list[str]) -> list[list[str]]:
    pytest_cmd = ["python", "-m", "pytest", "-q", *pytest_args]
    return [["python", "-m", "compileall", "-q", "app", "scripts", "tests"], pytest_cmd]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compileall and pytest for AIPE.")
    parser.add_argument("pytest_args", nargs="*", help="Optional pytest target(s), e.g. tests/test_pipeline.py")
    parser.add_argument("--local", action="store_true", help="Run in the current Python environment")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.local:
        for cmd in build_local_check_commands(args.pytest_args):
            print("+ " + " ".join(cmd))
            subprocess.run(cmd, cwd=root, check=True)
    else:
        cmd = build_docker_check_command(root, args.pytest_args)
        print("+ " + " ".join(cmd))
        subprocess.run(cmd, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
