#!/usr/bin/env python3
"""Build committed joninco GLM sources on an immutable CUDA 13.3 runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

from runtime_files import canonical, digest, inventory, inventory_bytes

RECIPE = Path(__file__).resolve().parent


def run(*args: str, **kwargs) -> str:
    return subprocess.check_output(args, text=True, **kwargs).strip()


def git(root: Path, *args: str) -> str:
    return run("git", "-C", str(root), *args)


def allows_native_reuse(repository: str, path: str) -> bool:
    """Allow Python payload edits while keeping native/dependency inputs fixed."""
    relative = Path(path)
    return relative.suffix == ".py" and relative.parts[0] in {
        repository, "tests", "benchmarks", "examples"
    }


def checkout(name: str, spec: dict, output: Path, branch_heads: bool) -> dict:
    revision = spec["commit"]
    if branch_heads:
        ref = "refs/heads/" + spec["branch"]
        rows = run("git", "ls-remote", "--exit-code", spec["url"], ref).splitlines()
        if len(rows) != 1 or rows[0].split()[1] != ref:
            raise RuntimeError(f"Could not resolve {spec['url']} {ref}")
        revision = rows[0].split()[0]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"Expected a full Git commit for {name}: {revision}")

    root = output / "sources" / name
    root.mkdir(parents=True)
    git(root, "init", "--quiet")
    git(root, "remote", "add", "origin", spec["url"])
    git(root, "fetch", "--quiet", "--depth=1", "origin", revision,
        spec["native_compatible_commit"])
    git(root, "checkout", "--quiet", "--detach", revision)
    tree = git(root, "rev-parse", "HEAD^{tree}")
    if not branch_heads and tree != spec["tree"]:
        raise RuntimeError(f"Pinned {name} tree does not match sources.lock.json")
    if any(row.startswith("160000 ") for row in git(root, "ls-files", "--stage").splitlines()):
        raise RuntimeError(f"{name} contains submodules; recursive export is unsupported")
    changed = git(root, "diff", "--no-renames", "--name-only", "-z",
                  spec["native_compatible_commit"], revision).split("\0")
    changed = [path for path in changed if path]
    blocked = [path for path in changed if not allows_native_reuse(name, path)]
    if blocked:
        raise RuntimeError(
            f"{name} changes files outside the Python payload allowed with the pinned "
            "native libraries. Rebuild and validate the runtime before using these "
            "revisions:\n" + "\n".join(blocked)
        )
    return {
        **spec, "commit": revision, "tree": tree,
        "changed_from_native_compatible_commit": changed,
    }


def export_commit(repository: Path, revision: str, destination: Path) -> None:
    destination.mkdir()
    process = subprocess.Popen(
        ["git", "-C", str(repository), "archive", "--format=tar", revision],
        stdout=subprocess.PIPE,
    )
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            archive.extractall(destination, filter="data")
    finally:
        process.stdout.close()
        result = process.wait()
    if result:
        raise RuntimeError(f"git archive failed for {repository}")


def docker_python(image: str, command: str, output: Path, *, writable: bool) -> list[str]:
    output_mount = f"type=bind,src={output},dst=/output"
    if not writable:
        output_mount += ",readonly"
    return [
        "docker", "run", "--rm", "--network=none", "--runtime=runc",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "TMPDIR=/tmp", "--env", "XDG_CACHE_HOME=/tmp/cache",
        "--tmpfs", "/cache:rw,mode=1777",
        "--mount", f"type=bind,src={RECIPE},dst=/recipe,readonly",
        "--mount", output_mount,
        "--entrypoint", "/opt/venv/bin/python", image,
        "/recipe/runtime_files.py", command,
    ]


def materialize(lock: dict, sources: dict, output: Path) -> tuple[str, Path]:
    base = lock["base_image"]
    if subprocess.run(["docker", "image", "inspect", base],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        subprocess.run(["docker", "pull", "--platform=linux/amd64", base], check=True)
    base_info = json.loads(run("docker", "image", "inspect", base))[0]
    if (base_info["Os"], base_info["Architecture"]) != ("linux", "amd64"):
        raise RuntimeError("The pinned runtime requires linux/amd64")
    subprocess.run(docker_python(base, "export-native", output, writable=True), check=True)

    inputs = {
        "schema_version": 3,
        **{name: {key: sources[name][key] for key in ("commit", "tree")}
           for name in ("b12x", "vllm")},
        "native_vllm_content_sha256": lock["native_inventory_sha256"],
        "native_image_id": base.split("@", 1)[1],
        "native_image_package_root": lock["native_package_root"],
        "generated_file_policy": "exclude Python bytecode and __pycache__",
    }
    source_key = digest(canonical(inputs))
    context = output / "context"
    context.mkdir()
    for name in ("b12x", "vllm"):
        export_commit(output / "sources" / name, sources[name]["commit"], context / name)
    for row in inventory(output / "native"):
        destination = context / "vllm" / "vllm" / row["path"]
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                raise RuntimeError(f"Native supplement conflicts with directory: {destination}")
            continue  # Committed files take precedence over generated/native supplements.
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "native" / row["path"], destination)

    rows = inventory(context / "b12x", "b12x") + inventory(context / "vllm", "vllm")
    inventory_data = inventory_bytes(rows)
    pinned = all(sources[name]["commit"] == lock["sources"][name]["commit"]
                 for name in sources)
    if pinned and (source_key != lock["published_source_key"] or
                   digest(inventory_data) != lock["published_runtime_inventory_sha256"]):
        raise RuntimeError("Materialized source/native bytes differ from the published image")
    (context / "runtime-files.sha256").write_bytes(inventory_data)
    manifest = {
        "schema_version": 1,
        "status": "implemented; source verification is not serving qualification",
        "source_key": source_key, "source_key_inputs": inputs,
        "base_image": base, "sources": sources,
        "native_policy": "Verified runtime supplements fill paths absent from committed vLLM source",
        "runtime_files": {"file_count": len(rows), "content_sha256": digest(inventory_data)},
    }
    (context / "materialization-manifest.json").write_bytes(canonical(manifest))
    return source_key, context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-heads", action="store_true",
                        help="Resolve b12x/master and vllm/dev/jovian-judgement once at build start")
    parser.add_argument("--output", type=Path, default=RECIPE / "build",
                        help="An absent directory for checkouts, context, logs, and build records")
    parser.add_argument("--tag", help="Local result tag; default includes both commits and ends in -local")
    args = parser.parse_args()
    output = args.output.resolve()
    # A distinct output directory prevents reusing a partial or edited source export.
    output.mkdir(parents=True, exist_ok=False)
    lock = json.loads((RECIPE / "sources.lock.json").read_text())
    sources = {name: checkout(name, spec, output, args.branch_heads)
               for name, spec in lock["sources"].items()}
    (output / "resolved-sources.json").write_bytes(canonical(sources))
    source_key, context = materialize(lock, sources, output)
    tag = args.tag or (
        f"joninco/vllm:glm53-b12x{sources['b12x']['commit'][:8]}-"
        f"vllm{sources['vllm']['commit'][:10]}-r27-local"
    )
    manifest_sha = digest((context / "materialization-manifest.json").read_bytes())
    build_args = {
        "BASE_IMAGE": lock["base_image"], "BASE_IMAGE_DIGEST": lock["base_image"].split("@")[1],
        "SOURCE_HASH": source_key, "MATERIALIZATION_MANIFEST_SHA256": manifest_sha,
        "IMAGE_TAG": tag, "LAUNCHER_SHA256": digest((context / "vllm" / "serve-glm53.sh").read_bytes()),
    }
    for name, source in sources.items():
        for key in ("commit", "tree", "upstream_base"):
            build_args[f"{name.upper()}_{key.upper()}"] = source[key]
    (output / "build-args.json").write_bytes(canonical(build_args))
    command = ["docker", "build", "--platform=linux/amd64", "--pull=false",
               "--network=none", "--progress=plain", "--file", str(RECIPE / "Dockerfile")]
    for name, value in build_args.items():
        command.extend(["--build-arg", f"{name}={value}"])
    command.extend(["--tag", tag, str(context)])
    print(f"Building {tag}; Docker output: {output / 'build.log'}", flush=True)
    with (output / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    built = json.loads(run("docker", "image", "inspect", tag))[0]
    with (output / "verification.json").open("w") as log:
        subprocess.run(docker_python(tag, "verify", output, writable=False), stdout=log, check=True)
    verification = json.loads((output / "verification.json").read_text())
    result = {"status": "implemented", "image_tag": tag, "image_id": built["Id"],
              "source_key": source_key, "sources": sources, "base_image": lock["base_image"],
              "verification": verification}
    (output / "BUILD.json").write_bytes(canonical(result))
    print(f"Verified {verification['runtime_files']} source/native files in {tag}", flush=True)
    print(f"Build record: {output / 'BUILD.json'}", flush=True)


if __name__ == "__main__":
    main()
