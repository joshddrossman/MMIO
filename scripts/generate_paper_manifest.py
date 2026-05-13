#!/usr/bin/env python3
"""Emit a reproducibility manifest for workshop / paper tables (benchmark v2, etc.).

Run from repo root after any config change that affects scoring or headline success:

    python scripts/generate_paper_manifest.py
    python scripts/generate_paper_manifest.py --benchmark-version v2 \\
        --out paper/paper_results_manifest.json

Captures git identity, dirty state, mirrored success thresholds / guard defaults,
and SHA-256 of key source files so result JSONs can be tied to a code revision
without re-running APIs.

``git.commit`` is ``git rev-parse HEAD`` at generation time (the revision you had
checked out when you ran this script). After you commit the manifest, that hash
will usually be the *parent* of the new commit; reproducibility is primarily
from ``key_file_sha256`` plus ``generated_at_utc``.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except OSError:
        return None


def _import_benchmark_constants() -> Dict[str, Any]:
    """Load live thresholds from the same code reviewers will use."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from analysis.sweep import OFFICIAL_FRACTION_THRESHOLDS
        from queries import default_guards

        guard_public = []
        for g in default_guards():
            d: Dict[str, Any] = {"name": g.name}
            if g.max_pct_increase is not None:
                d["bound_kind"] = "max_pct_increase"
                d["max_pct_increase"] = float(g.max_pct_increase)
            elif g.max_abs_increase is not None:
                d["bound_kind"] = "max_abs_increase"
                d["max_abs_increase"] = float(g.max_abs_increase)
            elif g.max_abs_value is not None:
                d["bound_kind"] = "max_abs_value"
                d["max_abs_value"] = float(g.max_abs_value)
            guard_public.append(d)
        return {
            "official_fraction_thresholds": dict(OFFICIAL_FRACTION_THRESHOLDS),
            "default_guard_specs": guard_public,
        }
    finally:
        if sys.path and sys.path[0] == str(REPO_ROOT):
            sys.path.pop(0)


def _shape_success_fraction_default() -> float:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import queries as queries_mod

        fn = getattr(queries_mod, "make_shape_niceness_query_from_metadata", None)
        if fn is None:
            return float("nan")
        sig = inspect.signature(fn)
        p = sig.parameters.get("success_fraction")
        if p is None or p.default is inspect.Parameter.empty:
            return float("nan")
        return float(p.default)
    finally:
        if sys.path and sys.path[0] == str(REPO_ROOT):
            sys.path.pop(0)


def build_manifest(
    *,
    benchmark_version: str,
    results_roots: List[str],
    extra_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    key_files = [
        REPO_ROOT / "queries.py",
        REPO_ROOT / "analysis" / "sweep.py",
        REPO_ROOT / "run_dataset.py",
        REPO_ROOT / "test_agent.py",
    ]
    if extra_paths:
        for rel in extra_paths:
            p = (REPO_ROOT / rel).resolve()
            if not str(p).startswith(str(REPO_ROOT.resolve())):
                raise ValueError(f"extra path escapes repo: {rel}")
            key_files.append(p)

    file_digests: Dict[str, str] = {}
    for p in key_files:
        if p.is_file():
            file_digests[str(p.relative_to(REPO_ROOT))] = _sha256_file(p)

    bm = _import_benchmark_constants()
    dirty_out = _git("status", "--porcelain")
    manifest: Dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_version": benchmark_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "is_dirty": bool(dirty_out),
            "porcelain": dirty_out.splitlines()[:200] if dirty_out else [],
        },
        "shape_niceness_default_success_fraction_from_queries_py": (
            _shape_success_fraction_default()
        ),
        "official_fraction_thresholds": bm["official_fraction_thresholds"],
        "default_guard_specs": bm["default_guard_specs"],
        "key_file_sha256": file_digests,
        "results_roots_documented": results_roots,
        "notes": (
            "Headline success in JSON is run-time score.success; pair metadata "
            "may include success_threshold_fraction_improved. Regenerate after "
            "changing queries.py, analysis/sweep.py, or agent scoring paths. "
            "Field git.commit is HEAD at generation time (often the parent of "
            "the commit that adds this file); use key_file_sha256 for exact code."
        ),
    }
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "paper" / "paper_results_manifest.json",
        help="Output JSON path (default: paper/paper_results_manifest.json)",
    )
    ap.add_argument(
        "--benchmark-version",
        default="v2",
        help='Tag for this benchmark/protocol freeze (e.g. "v2", "v3")',
    )
    ap.add_argument(
        "--results-root",
        action="append",
        default=[],
        help=(
            "Path(s) to result trees cited in the paper (repeatable), "
            "repo-relative. Defaults to existing full_dataset and out/runs."
        ),
    )
    ap.add_argument(
        "--extra-key-file",
        action="append",
        default=[],
        help="Additional repo-relative files to include in key_file_sha256",
    )
    args = ap.parse_args()

    roots = list(args.results_root)
    if not roots:
        for rel in ("full_dataset", Path("out") / "runs"):
            p = REPO_ROOT / rel
            if p.is_dir():
                roots.append(str(p.relative_to(REPO_ROOT)))

    manifest = build_manifest(
        benchmark_version=args.benchmark_version,
        results_roots=roots,
        extra_paths=args.extra_key_file or None,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
