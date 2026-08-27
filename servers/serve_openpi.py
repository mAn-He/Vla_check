"""π0.5 (openpi) server launcher.

openpi already ships a LIBERO-ready policy server speaking exactly the protocol
this harness uses (we adopted openpi's protocol, not the other way around), so
this script does NOT reimplement anything: it exec's the official
`scripts/serve_policy.py` inside the openpi checkout via uv.

Official command (verified against openpi README + scripts/serve_policy.py):
    uv run scripts/serve_policy.py --env LIBERO --port 8000
which for --env LIBERO defaults to
    Checkpoint(config="pi05_libero", dir="gs://openpi-assets/checkpoints/pi05_libero")

The one client-side difference vs. the other models: openpi's LIBERO example
resizes images to 224x224 with padding BEFORE sending (examples/libero/main.py),
so configs/eval.yaml sets `resize_with_pad: 224` for this model and the eval
client replicates that preprocessing exactly.

Usage:
    OPENPI_DIR=third_party/openpi python servers/serve_openpi.py --port 8000
    # or with an explicit checkpoint:
    python servers/serve_openpi.py --config pi05_libero --checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero
"""

import argparse
import os
import pathlib
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-dir", default=os.environ.get("OPENPI_DIR", "third_party/openpi"),
                        help="Path to the openpi checkout (default: $OPENPI_DIR or third_party/openpi)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=None,
                        help="openpi training config name (default: env LIBERO default = pi05_libero)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="checkpoint dir (default: gs://openpi-assets/checkpoints/pi05_libero via --env LIBERO)")
    args = parser.parse_args()

    openpi_dir = pathlib.Path(args.openpi_dir).resolve()
    if not (openpi_dir / "scripts" / "serve_policy.py").exists():
        sys.exit(f"openpi checkout not found at {openpi_dir} (run setup/20_openpi.sh, or set OPENPI_DIR)")

    cmd = ["uv", "run", "scripts/serve_policy.py", "--port", str(args.port)]
    if args.config or args.checkpoint_dir:
        if not (args.config and args.checkpoint_dir):
            sys.exit("--config and --checkpoint-dir must be given together (tyro subcommand policy:checkpoint)")
        cmd += ["policy:checkpoint", f"--policy.config={args.config}", f"--policy.dir={args.checkpoint_dir}"]
    else:
        cmd += ["--env", "LIBERO"]

    os.chdir(openpi_dir)
    print("exec:", " ".join(cmd), f"(cwd={openpi_dir})", flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
