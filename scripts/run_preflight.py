from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from surface_returns.manifest import write_json_atomic
from surface_returns.paths import (
    APPROVED_SLURM_JOB_ID,
    assert_approved_root,
    assert_approved_slurm_job,
    ensure_project_dirs,
)


def main() -> int:
    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    dirs = ensure_project_dirs(root)
    checks: dict[str, object] = {
        "root": str(root),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "approved_job": (
            True
            if not APPROVED_SLURM_JOB_ID
            else os.environ.get("SLURM_JOB_ID") == APPROVED_SLURM_JOB_ID
        ),
    }
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
            text=True,
            capture_output=True,
        )
        checks["nvidia_smi_returncode"] = result.returncode
        checks["gpus"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    else:
        checks["nvidia_smi_returncode"] = None
        checks["gpus"] = []

    modules = {}
    for module in ["numpy", "pandas", "pyarrow", "sklearn", "matplotlib", "wrds", "yaml"]:
        try:
            imported = __import__(module)
            modules[module] = getattr(imported, "__version__", "ok")
        except Exception as exc:  # pragma: no cover - only used in runtime manifest
            modules[module] = f"MISSING:{type(exc).__name__}"
    checks["modules"] = modules
    checks["status"] = "PASS" if checks["approved_job"] and len(checks["gpus"]) >= 2 else "FAIL"
    write_json_atomic(dirs["manifests"] / "preflight.json", checks)
    print(f"preflight_status={checks['status']}")
    print(f"manifest={dirs['manifests'] / 'preflight.json'}")
    return 0 if checks["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
