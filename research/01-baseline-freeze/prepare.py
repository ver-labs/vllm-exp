"""Recreate the pinned source copies without building or starting vLLM."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def git(repository, *args):
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cpu-native-from", type=Path)
    args = parser.parse_args()
    lock = json.loads((HERE / "baseline-lock.json").read_text())
    patch = HERE / "candidate.patch"
    assert (
        hashlib.sha256(patch.read_bytes()).hexdigest()
        == lock["candidate"]["patch_sha256"]
    )
    repository = args.repository.resolve()
    destination = args.destination.resolve()
    names = ("stock-v0.31.0", "candidate-v0.31.0", "upstream-20261024")
    for name in names:
        if (destination / name).exists():
            raise FileExistsError(f"Refusing existing worktree: {destination / name}")
    pins = (
        lock["base"]["commit"],
        lock["base"]["commit"],
        lock["upstream_reference"]["commit"],
    )
    for commit in set(pins):
        git(repository, "cat-file", "-e", f"{commit}^{{commit}}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, commit in zip(names, pins):
        tree = destination / name
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "worktree",
                "add",
                "--detach",
                str(tree),
                commit,
            ],
            check=True,
        )
        if name.startswith("candidate"):
            subprocess.run(
                ["git", "-C", str(tree), "apply", "--check", str(patch)], check=True
            )
            subprocess.run(["git", "-C", str(tree), "apply", str(patch)], check=True)
        if args.cpu_native_from and name != "upstream-20261024":
            source = args.cpu_native_from.resolve()
            assert (
                git(source, "describe", "--tags", "--always", lock["base"]["commit"])
                == "v0.31.0"
            )
            for item in lock["local_cpu_native_bootstrap"]:
                original = source / item["relative_path"]
                assert (
                    hashlib.sha256(original.read_bytes()).hexdigest() == item["sha256"]
                )
                target = tree / item["relative_path"]
                if target.exists():
                    raise FileExistsError(target)
                target.symlink_to(original)
        print(name, git(tree, "rev-parse", "HEAD"), flush=True)
    print("Source preparation complete. CUDA build and execution remain unvalidated.")


if __name__ == "__main__":
    main()
