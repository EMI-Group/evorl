"""Validate release metadata and package contents without importing EvoRL."""

import argparse
import ast
from email.parser import Parser
from pathlib import Path
import re
import tarfile
import zipfile


def source_version(root):
    """Read the literal version without loading runtime dependencies."""
    module = ast.parse((root / "evorl/__init__.py").read_text())
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise SystemExit("Missing literal evorl.__version__")


def check_metadata(text, expected_version):
    """Require the intended distribution name and version."""
    metadata = Parser().parsestr(text)
    if metadata["Name"] != "evorl-jax" or metadata["Version"] != expected_version:
        raise SystemExit(
            f"Unexpected distribution: {metadata['Name']} {metadata['Version']}"
        )
    if not metadata.get_all("License-File"):
        raise SystemExit("Missing license metadata")


def main():
    """Check the artifacts built by CI or a local release build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = source_version(root)
    if args.tag and (
        not re.fullmatch(r"v\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", args.tag)
        or args.tag[1:] != version
    ):
        raise SystemExit(f"Tag {args.tag!r} must match v{version}")

    wheels = list(args.dist_dir.glob("*.whl"))
    sdists = list(args.dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("Expected exactly one wheel and one sdist in dist/")
    expected_files = {
        path.relative_to(root).as_posix() for path in (root / "evorl").rglob("*.py")
    }
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        check_metadata(wheel.read(metadata_name).decode(), version)
        if not expected_files <= names:
            raise SystemExit(f"Wheel missing modules: {sorted(expected_files - names)}")
        if not any(name.endswith("/LICENSE") for name in names):
            raise SystemExit("Wheel missing LICENSE")
    with tarfile.open(sdists[0], "r:gz") as sdist:
        members = sdist.getnames()
        prefix = members[0].split("/")[0] + "/"
        names = {name.removeprefix(prefix) for name in members}
        metadata_file = sdist.extractfile(prefix + "PKG-INFO")
        if metadata_file is None:
            raise SystemExit("sdist missing PKG-INFO")
        check_metadata(metadata_file.read().decode(), version)
        if not expected_files <= names or "LICENSE" not in names:
            raise SystemExit("sdist missing package modules or LICENSE")
    print(f"Validated evorl-jax {version}: wheel, sdist, license, and package modules")


if __name__ == "__main__":
    main()
