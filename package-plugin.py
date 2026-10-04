"""Build the Windows/Linux plugin ZIP using only the Python standard library."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--version", default="plugin-dev")
parser.add_argument("--out-dir", type=Path, default=root / "dist")
args = parser.parse_args()
if not all(c.isalnum() or c in "-._" for c in args.version):
    parser.error("version must contain only letters, numbers, dots, hyphens and underscores")
name = "darask-paint-ai-diffusion-" + args.version
files = {
    "darask-plugin.bat": "darask-plugin.bat",
    "darask-plugin.sh": "darask-plugin.sh",
    "darask-plugin.json": "darask-plugin.json",
    "flake.nix": "flake.nix",
    "flake.lock": "flake.lock",
    "LICENSE": "LICENSE",
    "README_darask.md": "README.md",
    "darask_server.py": "darask_server.py",
}
args.out_dir.mkdir(parents=True, exist_ok=True)
destination = args.out_dir / (name + ".zip")
with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
    for source, target in files.items():
        archive.write(root / source, name + "/" + target)
print(destination)
