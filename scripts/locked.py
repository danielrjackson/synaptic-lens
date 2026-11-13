"""Hash-enforced Windows CPU install and validation, without changing shared environments."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements"
UV = str(Path(sys.executable).with_name("uv.exe")) if Path(sys.executable).with_name("uv.exe").is_file() else "uv"


def check():
    manifest = json.loads((REQ / "integrity.json").read_text())
    for name, expected in manifest["sha256"].items():
        data = (ROOT / name).read_bytes()
        if name == "pyproject.toml":
            data = data.replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Lock input/output integrity mismatch: {name}")
    pins = {}
    for line in (REQ / "windows-cpu.in").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            pins[name] = version
    locked = {}
    for record in (REQ / "windows-cpu.txt").read_text().replace("\\\n", " ").splitlines():
        match = re.fullmatch(r"([a-z0-9-]+)==([^\s]+)\s+(.*)", record)
        if not match or not re.fullmatch(r"(?:--hash=sha256:[a-f0-9]{64}\s*)+", match[3]):
            raise ValueError("Every resolved dependency must have an exact version and SHA-256")
        locked[match[1]] = match[2]
    if locked != pins:
        raise ValueError("Input pins and resolved lock differ")
    print(f"Verified {len(locked)} exact transitive dependency pins and lock byte hashes")


def run(*command, cwd=ROOT):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run([str(x) for x in command], cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["check", "install", "test"])
    parser.add_argument("--environment", type=Path, default=ROOT / ".venv-locked")
    args = parser.parse_args()
    check()
    if args.action == "check":
        return
    if sys.platform != "win32" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("This locked profile requires Windows x64 CPython 3.13")
    if sys.maxsize <= 2**32:
        raise RuntimeError("This locked profile requires x64")
    env = args.environment.resolve()
    python = env / "Scripts" / "python.exe"
    if args.action == "install":
        if env.exists():
            raise FileExistsError("Use a new environment path; existing environments are preserved")
        uv_version = subprocess.check_output([UV, "--version"], text=True).split()[1]
        if uv_version != "0.7.4":
            raise RuntimeError("Install the hash-locked uv 0.7.4 bootstrap first")
        run(UV, "venv", "--python", sys.executable, env)
        run(UV, "pip", "sync", "--python", python, "--require-hashes", "--only-binary", ":all:", REQ / "windows-cpu.txt")
        # The locked backend builds local Git source; no unpinned PEP 517 isolation download.
        with tempfile.TemporaryDirectory(prefix="sl-locked-wheel-") as temp:
            run(python, "-I", "-c", "from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])", temp)
            wheels = list(Path(temp).glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("Expected exactly one local project wheel")
            run(UV, "pip", "install", "--python", python, "--no-deps", wheels[0])
        run(UV, "pip", "check", "--python", python)
    else:
        profile = json.loads((REQ / "integrity.json").read_text())
        package = profile["import_name"]
        run(python, "-I", "-c", "import importlib.metadata as m,sys; pins=[line.strip().split('==') for line in open(sys.argv[1]) if line.strip() and not line.startswith('#')]; assert all(m.version(name)==version for name,version in pins); print('Installed dependency versions match lock')", REQ / "windows-cpu.in")
        run(python, "-I", "-c", "import importlib, pathlib, sys; m=importlib.import_module(sys.argv[1]); assert pathlib.Path(m.__file__).is_relative_to(pathlib.Path(sys.prefix)); print(m.__file__)", package)
        os.environ["SL_LOCKED_PACKAGE"] = package
        run(python, "-I", "-m", "pytest", "-q", "-o", "pythonpath=", profile["tests"])


if __name__ == "__main__":
    main()
