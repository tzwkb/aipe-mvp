"""Conservative Docker cleanup for the AIPE dev stack.

Default mode is dry-run. It removes only stopped compose containers and dangling
images when committed. Qdrant volumes are never touched unless
``--include-volumes`` is explicitly provided.
"""

from __future__ import annotations

import argparse
import subprocess


def build_cleanup_commands(
    *,
    include_build_cache: bool = False,
    include_volumes: bool = False,
) -> list[list[str]]:
    commands: list[list[str]] = [
        ["docker", "compose", "rm", "-f"],
        ["docker", "image", "prune", "-f"],
    ]
    if include_build_cache:
        commands.append(["docker", "builder", "prune", "-f"])
    if include_volumes:
        commands.append(["docker", "compose", "down", "--volumes", "--remove-orphans"])
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean conservative Docker leftovers for this dev stack.")
    parser.add_argument("--commit", action="store_true", help="Actually execute cleanup commands")
    parser.add_argument("--include-build-cache", action="store_true", help="Also prune Docker build cache")
    parser.add_argument(
        "--include-volumes",
        action="store_true",
        help="Also remove compose volumes. This deletes Qdrant local data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    commands = build_cleanup_commands(
        include_build_cache=args.include_build_cache,
        include_volumes=args.include_volumes,
    )

    if not args.commit:
        print("dry-run: add --commit to execute")
    if args.include_volumes:
        print("warning: --include-volumes will remove compose volumes, including Qdrant data")

    for cmd in commands:
        print("+ " + " ".join(cmd))
        if args.commit:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
