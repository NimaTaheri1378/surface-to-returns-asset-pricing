from __future__ import annotations

from scripts.public_safety_scan import iter_files


def test_iter_files_skips_ignored_env_files(tmp_path):
    env_file = tmp_path / ".env.local"
    env_file.write_text("FRED_" + "API" + "_KEY=secret-value-that-is-private\n", encoding="utf-8")
    visible = tmp_path / "README.md"
    visible.write_text("public text", encoding="utf-8")

    files = {path.name for path in iter_files(tmp_path)}

    assert files == {"README.md"}
