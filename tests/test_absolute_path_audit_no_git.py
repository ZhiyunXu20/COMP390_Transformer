"""Verify check_absolute_path_caveats.py works in fallback mode (no .git)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_fallback_mode_works(tmp_path: Path) -> None:
    (tmp_path / "small_try").mkdir()
    (tmp_path / "small_try" / "config.py").write_text(
        "DATA_PATH = 'data/splits/en_fr_50k_seed42'\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "check_absolute_path_caveats.py").write_bytes(
        Path(__file__).resolve().parent.parent.joinpath(
            "scripts",
            "check_absolute_path_caveats.py",
        ).read_bytes(),
    )
    result = subprocess.run(
        [sys.executable, "scripts/check_absolute_path_caveats.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Expected exit 0, got {result.returncode}: {result.stderr}\n{result.stdout}"
    )


def test_fallback_fails_on_active_config_with_absolute_path(tmp_path: Path) -> None:
    (tmp_path / "small_try").mkdir()
    (tmp_path / "small_try" / "config.py").write_text(
        "DATA_PATH = '/root/autodl-tmp/data/splits/en_fr_50k_seed42'\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "check_absolute_path_caveats.py").write_bytes(
        Path(__file__).resolve().parent.parent.joinpath(
            "scripts",
            "check_absolute_path_caveats.py",
        ).read_bytes(),
    )
    result = subprocess.run(
        [sys.executable, "scripts/check_absolute_path_caveats.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (
        f"Expected fail on active config, got {result.returncode}: {result.stderr}"
    )
