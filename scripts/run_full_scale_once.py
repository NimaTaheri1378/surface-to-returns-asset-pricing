from __future__ import annotations

from pathlib import Path

from surface_returns.manifest import read_json
from surface_returns.paths import assert_approved_root, assert_approved_slurm_job


def main() -> int:
    root = assert_approved_root(Path.cwd())
    assert_approved_slurm_job()
    manifest_path = root / "manifests" / "wrds_smoke_manifest.json"
    if not manifest_path.exists():
        print("full_scale_status=BLOCKED_NO_SMOKE_MANIFEST")
        return 10
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS":
        print(f"full_scale_status=BLOCKED_SMOKE_STATUS_{manifest.get('status')}")
        return 11
    print("full_scale_status=READY")
    print("Launch with: python scripts/run_full_feature_extract.py --start-year 1996 --end-year 2025 --resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
