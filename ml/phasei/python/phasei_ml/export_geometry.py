"""Run the opt-in C++ Phase-I geometry exporter and bind its output to YAML."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_geometry(executable: str | Path, geometry_yaml: str | Path, output: str | Path) -> dict:
    executable, geometry_yaml, output = map(lambda p: Path(p).expanduser().resolve(),
                                             (executable, geometry_yaml, output))
    proc = subprocess.run([str(executable), str(geometry_yaml)], check=True, capture_output=True, text=True)
    marker = proc.stdout.find('{"schema_version":1')
    if marker < 0:
        raise RuntimeError(f"geometry exporter produced no JSON payload: {proc.stdout[-500:]}")
    data = json.loads(proc.stdout[marker:])
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    output.write_text(payload, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "geometry_yaml": os.path.relpath(geometry_yaml, output.parent),
        "geometry_yaml_sha256": sha256(geometry_yaml),
        "geometry_export": output.name,
        "geometry_export_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "counts": {name: len(item["channels"]) for name, item in data["detectors"].items()},
    }
    output.with_name("geometry_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exporter", required=True, type=Path)
    parser.add_argument("--geometry-yaml", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(export_geometry(args.exporter, args.geometry_yaml, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
