from __future__ import annotations

import asyncio

from services.bot.bot.services.settings_store import (
    InMemoryUserSettingsStore,
    UserSettings,
    apply_setting,
)


def test_in_memory_user_settings_store_persists_reference_and_settings() -> None:
    store = InMemoryUserSettingsStore()

    settings = UserSettings(
        chat_id=12345,
        model_id="colorcomic_reference",
        seed=7,
        options={"size": 512},
        reference_image=b"reference",
    )
    asyncio.run(store.save(settings))

    loaded = asyncio.run(store.get(12345))

    assert loaded.model_id == "colorcomic_reference"
    assert loaded.seed == 7
    assert loaded.options == {"size": 512}
    assert loaded.reference_image == b"reference"


def test_apply_setting_keeps_reference_when_model_changes() -> None:
    settings = UserSettings(
        chat_id=12345,
        model_id="ddcolor",
        seed=7,
        options={"size": 512},
        reference_image=b"reference",
    )

    apply_setting(
        settings,
        param="model_id",
        value="cgan_reference",
        clear_values={"none", "null", "off", "clear"},
    )

    assert settings.model_id == "cgan_reference"
    assert settings.seed == 7
    assert settings.options == {"size": 512}
    assert settings.reference_image == b"reference"
