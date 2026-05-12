from __future__ import annotations

import asyncio
from io import BytesIO

from PIL import Image

from services.bot.bot.main import (
    HELP_COMPARISON_PANEL_PATH,
    HELP_TEXT,
    MODEL_CALLBACK_PREFIX,
    START_TEXT,
    _build_model_keyboard,
    _enabled_models,
    _format_models_list,
    _send_help_comparison_panel,
    _validate_image_upload,
)


def _png_bytes(size: tuple[int, int] = (4, 4)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_validate_image_upload_accepts_valid_image() -> None:
    assert (
        _validate_image_upload(
            _png_bytes(),
            max_bytes=1024,
            max_side=16,
        )
        is None
    )


def test_validate_image_upload_rejects_invalid_or_large_images() -> None:
    assert "слишком большое" in (
        _validate_image_upload(
            _png_bytes(),
            max_bytes=1,
            max_side=16,
        )
        or ""
    )
    assert "Сторона изображения" in (
        _validate_image_upload(
            _png_bytes(size=(32, 4)),
            max_bytes=1024,
            max_side=16,
        )
        or ""
    )
    assert "Не удалось прочитать" in (
        _validate_image_upload(
            b"not an image",
            max_bytes=1024,
            max_side=16,
        )
        or ""
    )


def test_enabled_models_filters_disabled_models() -> None:
    models = [
        {"model_id": "ddcolor", "enabled": True},
        {"model_id": "passthrough", "enabled": False},
    ]

    assert _enabled_models(models) == [{"model_id": "ddcolor", "enabled": True}]


def test_model_list_and_keyboard_mark_current_and_reference_models() -> None:
    models = [
        {
            "model_id": "cgan_reference",
            "enabled": True,
            "requires_reference": True,
        },
        {
            "model_id": "ddcolor",
            "enabled": True,
            "requires_reference": False,
        },
    ]

    text = _format_models_list(models, current_model_id="ddcolor")
    keyboard = _build_model_keyboard(models, current_model_id="ddcolor")

    assert "cgan_reference" in text
    assert "reference" in text
    assert "ddcolor" in text
    assert "выбрана" in text
    assert keyboard.inline_keyboard[0][0].callback_data == (
        f"{MODEL_CALLBACK_PREFIX}cgan_reference"
    )
    assert keyboard.inline_keyboard[1][0].callback_data == (
        f"{MODEL_CALLBACK_PREFIX}ddcolor"
    )


def test_start_text_describes_async_colorization_flow() -> None:
    assert "Illustration Autocolorizer" in START_TEXT
    assert "/models" in START_TEXT
    assert "/set_reference" in START_TEXT
    assert "/colorize" in START_TEXT
    assert "worker" in START_TEXT
    assert "автоматически" in START_TEXT


def test_help_text_contains_all_current_commands() -> None:
    for command in (
        "/start",
        "/help",
        "/settings",
        "/models",
        "/model",
        "/set_settings <param> <value>",
        "/set_reference",
        "/colorize",
    ):
        assert command in HELP_TEXT


def test_help_comparison_panel_asset_is_sent() -> None:
    class FakeMessage:
        def __init__(self) -> None:
            self.photo = None
            self.caption = None

        async def answer_photo(self, photo, caption: str | None = None) -> None:
            self.photo = photo
            self.caption = caption

    message = FakeMessage()

    assert HELP_COMPARISON_PANEL_PATH.exists()
    asyncio.run(_send_help_comparison_panel(message))  # type: ignore[arg-type]

    assert str(message.photo.path) == str(HELP_COMPARISON_PANEL_PATH)
    assert message.caption == "Model comparison panel: sample 71."
