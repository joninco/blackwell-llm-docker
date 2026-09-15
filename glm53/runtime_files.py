"""Export pinned native supplements and verify image source inventories without GPUs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys


def canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    if path.is_symlink():
        return digest(os.readlink(path).encode())
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path, prefix: str = "") -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink() or path.is_file():
            rows.append({"path": str(Path(prefix) / relative), "sha256": file_digest(path)})
    return rows


def inventory_bytes(rows: list[dict]) -> bytes:
    return "".join(f"{row['sha256']}  {row['path']}\n" for row in rows).encode()


def export_native() -> None:
    recipe = Path("/recipe")
    lock = json.loads((recipe / "sources.lock.json").read_text())
    data = (recipe / "native-vllm.sha256").read_bytes()
    if digest(data) != lock["native_inventory_sha256"]:
        raise RuntimeError("Native supplement inventory differs from sources.lock.json")
    root = Path(lock["native_package_root"])
    output = Path("/output/native")
    output.mkdir()
    for line in data.decode().splitlines():
        expected, name = line.split("  ", 1)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Invalid native supplement path: {name}")
        source = root / relative
        if source.is_symlink() or not source.is_file() or file_digest(source) != expected:
            raise RuntimeError(f"Runtime native supplement differs from the lock: {name}")
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if inventory_bytes(inventory(output)) != data:
        raise RuntimeError("Exported native supplement inventory does not match the lock")
    print(f"Exported and verified {len(data.splitlines())} native/generated files", flush=True)


def verify() -> None:
    context = Path("/output/context")
    runtime = Path("/opt/glm53-flash")
    manifest_data = (context / "materialization-manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    if (runtime / "qualification/materialization-manifest.json").read_bytes() != manifest_data:
        raise RuntimeError("Image source manifest differs from the build context")
    rows = inventory(runtime / "b12x", "b12x") + inventory(runtime / "vllm", "vllm")
    actual = inventory_bytes(rows)
    if (actual != (context / "runtime-files.sha256").read_bytes() or
            digest(actual) != manifest["runtime_files"]["content_sha256"]):
        raise RuntimeError("Image source/native files differ from the build inventory")

    import torch
    import b12x
    import vllm

    if torch.cuda.is_available():
        raise RuntimeError("Run source verification without GPU access")
    for name, module in (("b12x", b12x), ("vllm", vllm)):
        if not Path(module.__file__).is_relative_to(runtime / name):
            raise RuntimeError(f"{name} is not imported from its committed source tree")
    result = {
        "status": "passed", "scope": "Source/native identities and CPU imports; no GPU inference",
        "runtime_files": len(rows), "runtime_inventory_sha256": digest(actual),
        "native_libraries": {row["path"]: row["sha256"] for row in rows if row["path"].endswith(".so")},
        "package_versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "vllm", "b12x", "triton", "nvidia-cutlass-dsl", "flashinfer-python")},
        "torch_cuda": torch.version.cuda,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    {"export-native": export_native, "verify": verify}[sys.argv[1]]()
