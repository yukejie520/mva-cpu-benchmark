"""Small, dependency-free readers for frozen benchmark manifests.

The main-platform measurement path intentionally keeps using ``list_images``
from ``eval_common``.  This module is only used when a second platform is
explicitly asked to consume the recorded image list or artifact map.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def _entries(path: Path) -> list[str]:
    """Read one non-empty, non-comment entry per line."""
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            entries.append(value)
    return entries


def image_paths_from_manifest(manifest: str | Path, image_dir: str | Path,
                              expected_count: int | None = None) -> list[Path]:
    """Resolve a recorded image-name list below ``image_dir``.

    The manifest stores names rather than machine-specific absolute paths, so
    the same file can be copied to a Pi.  Absolute entries are accepted for
    diagnostics, but relative entries are always resolved below ``image_dir``.
    """
    manifest_path = Path(manifest)
    entries = _entries(manifest_path)
    if expected_count is not None and len(entries) != expected_count:
        raise ValueError(
            f"{manifest_path} contains {len(entries)} images; "
            f"expected {expected_count}"
        )
    if len(set(entries)) != len(entries):
        raise ValueError(f"{manifest_path} contains duplicate image names")

    root = Path(image_dir)
    paths: list[Path] = []
    for entry in entries:
        raw = Path(entry)
        candidate = raw if raw.is_absolute() else root / raw
        if not candidate.is_file():
            raise FileNotFoundError(
                f"manifest entry does not exist: {entry!r} -> {candidate}"
            )
        if candidate.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError(f"manifest entry is not a JPEG image: {entry!r}")
        paths.append(candidate)
    return paths


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_int8_triplet(manifest: str | Path, root: str | Path,
                      model: str) -> dict[str, Path]:
    """Load and verify the explicit FP32/QDQ/QOperator artifact mapping.

    The two INT8 rows must share the recipe metadata recorded by S1.  Every
    file must exist and match its recorded SHA-256.  This is deliberately
    strict: a missing or changed artifact stops the cross-ISA probe rather
    than silently substituting a newly exported graph.
    """
    manifest_path = Path(manifest)
    root_path = Path(root)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row.get("model") == model]
    if not rows:
        raise ValueError(f"no artifact rows for model {model!r} in {manifest_path}")

    by_format: dict[str, dict[str, str]] = {}
    for row in rows:
        fmt = (row.get("format") or "").strip()
        if fmt in by_format:
            raise ValueError(f"duplicate {model}/{fmt} row in {manifest_path}")
        by_format[fmt] = row

    required = {"FP32", "QDQ", "QOperator"}
    missing = required - set(by_format)
    extra = set(by_format) - required
    if missing or extra:
        raise ValueError(
            f"artifact map for {model} must contain exactly FP32/QDQ/QOperator; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )

    int8_rows = [by_format[fmt] for fmt in ("QDQ", "QOperator")]
    for field in ("recipe_family", "head_preserving", "calibration"):
        values = {row.get(field, "").strip() for row in int8_rows}
        if len(values) != 1:
            raise ValueError(
                f"QDQ/QOperator {field} metadata disagree for {model}: {values}"
            )
    if by_format["QDQ"].get("head_preserving", "").strip().lower() != "yes":
        raise ValueError("S2 requires the head-preserving recipe")

    result: dict[str, Path] = {}
    for fmt in ("FP32", "QDQ", "QOperator"):
        row = by_format[fmt]
        rel = (row.get("file") or "").strip()
        recorded = (row.get("sha256") or "").strip().upper()
        if not rel or len(recorded) != 64:
            raise ValueError(f"incomplete artifact row for {model}/{fmt}")
        path = Path(rel)
        if not path.is_absolute():
            path = root_path / path
        if not path.is_file():
            raise FileNotFoundError(
                f"recorded {model}/{fmt} artifact is missing: {path}"
            )
        actual = sha256_file(path)
        if actual != recorded:
            raise ValueError(
                f"SHA-256 mismatch for {model}/{fmt}: recorded {recorded}, "
                f"actual {actual}; S2 is stopped"
            )
        result[fmt] = path
    return result
