"""Command dispatcher for Phase-I ML tools."""
from __future__ import annotations
import argparse


def main() -> int:
    parser = argparse.ArgumentParser(prog="phasei_ml")
    parser.add_argument("command", choices=("convert", "train", "evaluate", "experiment"))
    args, rest = parser.parse_known_args()
    if args.command == "convert":
        from .convert import main as run
    elif args.command == "train":
        from .train import main as run
    elif args.command == "evaluate":
        from .evaluate import main as run
    else:
        from .experiment import main as run
    return run(rest)


if __name__ == "__main__":
    raise SystemExit(main())
