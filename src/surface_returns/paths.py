from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = Path(os.environ.get("SURFACE_RETURNS_LOCAL_ROOT", str(PACKAGE_ROOT))).expanduser()
REMOTE_ROOT_ENV = os.environ.get("SURFACE_RETURNS_REMOTE_ROOT")
REMOTE_ROOT = Path(REMOTE_ROOT_ENV).expanduser() if REMOTE_ROOT_ENV else None
APPROVED_SLURM_JOB_ID = os.environ.get("SURFACE_RETURNS_SLURM_JOB_ID")


def project_root() -> Path:
    override = os.environ.get("SURFACE_RETURNS_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path.cwd().resolve()


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


def assert_approved_root(root: Path | None = None) -> Path:
    root = (root or project_root()).resolve()
    approved = [LOCAL_ROOT.resolve()]
    if REMOTE_ROOT is not None:
        approved.append(REMOTE_ROOT.resolve())
    if not any(_same_path(root, item) for item in approved):
        raise RuntimeError(
            f"Refusing to operate outside approved workspaces. Got {root}; "
            f"approved roots are {approved}."
        )
    return root


def assert_approved_slurm_job() -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    root = Path.cwd().resolve()
    if REMOTE_ROOT is not None and _same_path(root, REMOTE_ROOT.resolve()) and not job_id:
        raise RuntimeError(
            f"Refusing to run compute/WRDS code on the Amarel login node. "
            "Use an approved SLURM allocation."
        )
    if APPROVED_SLURM_JOB_ID and job_id and job_id != APPROVED_SLURM_JOB_ID:
        raise RuntimeError(f"Refusing SLURM job {job_id}; approved job is {APPROVED_SLURM_JOB_ID}.")


def ensure_project_dirs(root: Path) -> dict[str, Path]:
    assert_approved_root(root)
    dirs = {
        "logs": root / "logs",
        "manifests": root / "manifests",
        "metadata": root / "data" / "metadata",
        "raw_smoke": root / "data" / "raw" / "smoke",
        "processed_smoke": root / "data" / "processed" / "smoke",
        "figures_smoke": root / "outputs" / "figures" / "smoke",
        "reports": root / "outputs" / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
