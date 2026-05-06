"""Verify ablation_lib monitoring helper name and deprecation alias (A28)."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_ablation_lib():
    path = _ROOT / "scripts" / "ablation_lib.py"
    spec = importlib.util.spec_from_file_location("ablation_lib_a28", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_renamed_helper_exists() -> None:
    mod = _load_ablation_lib()
    assert hasattr(mod, "evaluate_checkpoint_on_sampled_test_monitoring")


def test_deprecation_alias_still_works() -> None:
    mod = _load_ablation_lib()
    assert mod.evaluate_checkpoint_on_test is mod.evaluate_checkpoint_on_sampled_test_monitoring


def test_no_misleading_full_test_helper() -> None:
    mod = _load_ablation_lib()
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if name == "evaluate_checkpoint_on_test":
            continue
        src = inspect.getsource(obj)
        if (
            "max_samples=bleu_sample_size" in src
            and "test" in name.lower()
            and "sampled" not in name.lower()
            and "monitoring" not in name.lower()
        ):
            raise AssertionError(
                f"{name} uses sampled BLEU but its name suggests full test eval."
            )
