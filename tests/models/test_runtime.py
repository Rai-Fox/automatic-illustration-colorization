from __future__ import annotations

import sys

from illustration_colorizer.models.runtime import isolated_vendor_imports


def test_isolated_vendor_imports_restores_existing_module(tmp_path) -> None:
    original_models = object()
    sys.modules["models"] = original_models  # type: ignore[assignment]

    with isolated_vendor_imports(tmp_path):
        assert "models" not in sys.modules
        sys.modules["models"] = object()  # type: ignore[assignment]

    assert sys.modules["models"] is original_models
