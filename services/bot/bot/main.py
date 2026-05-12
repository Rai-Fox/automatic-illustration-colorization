from __future__ import annotations

import asyncio
import contextlib
import io
import logging
from typing import Any

import httpx
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.filters.command import CommandObject
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from PIL import Image

from services.bot.bot.config import load_settings
from services.bot.bot.services.api_client import ColorizationApiClient
from services.bot.bot.services.events import JobEventListener
from services.bot.bot.services.settings_store import (
    PostgresUserSettingsStore,
    UserSettings,
    UserSettingsStore,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_IMAGE_BYTES = 10_485_760
DEFAULT_MAX_IMAGE_SIDE = 4096
MODEL_CALLBACK_PREFIX = "select_model:"


async def _download_photo_from_message(message: Message, bot: Bot) -> bytes | None:
    source = message
    if not source.photo and message.reply_to_message is not None:
        source = message.reply_to_message
    if not source.photo:
        return None

    photo = source.photo[-1]
    file = await bot.get_file(photo.file_id)
    buffer = io.BytesIO()
    await bot.download_file(file.file_path, destination=buffer)
    return buffer.getvalue()


def _model_by_id(models: list[dict[str, Any]], model_id: str) -> dict[str, Any] | None:
    return next((model for model in models if model["model_id"] == model_id), None)


def _enabled_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [model for model in models if model.get("enabled")]


def _normalize_model_id(model_id: str) -> str:
    normalized = model_id.strip().lower()
    if normalized == "cgan":
        return "cgan_reference"
    return normalized


def _format_model_label(model: dict[str, Any], *, current_model_id: str) -> str:
    labels: list[str] = []
    if model["model_id"] == current_model_id:
        labels.append("выбрана")
    if model.get("requires_reference"):
        labels.append("reference")
    suffix = f" ({', '.join(labels)})" if labels else ""
    return f"{model['model_id']}{suffix}"


def _format_models_list(
    models: list[dict[str, Any]],
    *,
    current_model_id: str,
) -> str:
    if not models:
        return "Нет доступных моделей."
    lines = ["Доступные модели:"]
    lines.extend(
        f"- {_format_model_label(model, current_model_id=current_model_id)}"
        for model in models
    )
    return "\n".join(lines)


def _build_model_keyboard(
    models: list[dict[str, Any]],
    *,
    current_model_id: str,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=_format_model_label(model, current_model_id=current_model_id),
                    callback_data=f"{MODEL_CALLBACK_PREFIX}{model['model_id']}",
                )
            ]
            for model in models
        ]
    )


def _parse_setting_value(value: str) -> Any:
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "on"}:
        return True
    if normalized in {"false", "no", "off"}:
        return False
    with contextlib.suppress(ValueError):
        return int(value)
    with contextlib.suppress(ValueError):
        return float(value)
    return value


def _validate_image_upload(
    image_bytes: bytes,
    *,
    max_bytes: int,
    max_side: int,
) -> str | None:
    if len(image_bytes) > max_bytes:
        return f"Изображение слишком большое. Максимум: {max_bytes} bytes."
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_size = image.size
            image.verify()
            if max(image_size) > max_side:
                return (
                    f"Сторона изображения слишком большая: {image_size}. "
                    f"Максимум: {max_side}."
                )
    except (OSError, ValueError):
        return "Не удалось прочитать изображение. Отправьте корректный PNG/JPEG."
    return None


def _log_command(message: Message, command: str) -> None:
    user_id = message.from_user.id if message.from_user is not None else None
    LOGGER.info(
        "telegram command received command=%s chat_id=%s user_id=%s",
        command,
        message.chat.id,
        user_id,
    )


def _format_settings(settings: UserSettings) -> str:
    reference_status = (
        "задан" if settings.reference_image is not None else "не задан"
    )
    seed = settings.seed if settings.seed is not None else "не задан"
    options = settings.options if settings.options else "нет"
    return (
        f"model_id: {settings.model_id}\n"
        f"seed: {seed}\n"
        f"options: {options}\n"
        f"reference: {reference_status}"
    )


async def deliver_job_events(
    bot: Bot,
    *,
    api_client: ColorizationApiClient,
    event_listener: JobEventListener,
) -> None:
    while True:
        try:
            async for event in event_listener.listen():
                LOGGER.info(
                    "telegram job event received job_id=%s status=%s chat_id=%s",
                    event.get("job_id"),
                    event.get("status"),
                    event.get("chat_id"),
                )
                chat_id = event.get("chat_id")
                if chat_id is None:
                    continue

                job_id = str(event["job_id"])
                if event.get("status") == "succeeded":
                    try:
                        result = await api_client.get_job_result(job_id)
                        await bot.send_photo(
                            int(chat_id),
                            BufferedInputFile(result, filename="colorized.png"),
                            caption=f"Готово: {job_id}",
                        )
                        LOGGER.info(
                            "telegram job result delivered job_id=%s chat_id=%s",
                            job_id,
                            chat_id,
                        )
                    except Exception:
                        LOGGER.exception("Could not deliver job %s result", job_id)
                        await bot.send_message(
                            int(chat_id),
                            (
                                f"Задача {job_id} завершена, "
                                "но результат не удалось отправить."
                            ),
                        )
                elif event.get("status") == "failed":
                    await bot.send_message(
                        int(chat_id),
                        f"Задача {job_id} завершилась с ошибкой: {event.get('error')}",
                    )
                    LOGGER.info(
                        "telegram job failure delivered job_id=%s chat_id=%s",
                        job_id,
                        chat_id,
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Job event listener failed; reconnecting")
            await asyncio.sleep(5)


def create_router(
    api_client: ColorizationApiClient,
    settings_store: UserSettingsStore,
    *,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
) -> Router:
    router = Router()

    @router.message(Command("start"))
    async def start(message: Message) -> None:
        _log_command(message, "start")
        await message.answer(
            "Отправьте /colorize с изображением, чтобы поставить задачу "
            "колоризации в очередь. Результат придет автоматически."
        )

    @router.message(Command("help"))
    async def help_message(message: Message) -> None:
        _log_command(message, "help")
        await message.answer(
            "/settings - показать текущие настройки\n"
            "/models - показать доступные модели и выбрать модель кнопкой\n"
            "/set_settings <param> <value> - изменить настройку\n"
            "/set_reference + изображение - сохранить reference image\n"
            "/colorize + изображение - запустить колоризацию\n\n"
            "Поддерживаемые настройки: model_id, seed и параметры модели "
            "в options, например /set_settings size 576."
        )

    @router.message(Command("settings"))
    async def settings(message: Message) -> None:
        _log_command(message, "settings")
        try:
            user_settings = await settings_store.get(message.chat.id)
        except Exception:
            LOGGER.exception(
                "Could not load user settings for chat %s",
                message.chat.id,
            )
            await message.answer("Не удалось загрузить настройки. Попробуйте позже.")
            return
        await message.answer(_format_settings(user_settings))

    @router.message(Command("models", "model"))
    async def models(message: Message) -> None:
        _log_command(message, "models")
        try:
            user_settings = await settings_store.get(message.chat.id)
            available_models = _enabled_models(await api_client.list_models())
        except httpx.HTTPError:
            LOGGER.exception("Could not load available models")
            await message.answer(
                "Не удалось получить список моделей. Попробуйте позже."
            )
            return
        except Exception:
            LOGGER.exception(
                "Could not load model selection state for chat %s",
                message.chat.id,
            )
            await message.answer(
                "Не удалось загрузить настройки. Попробуйте позже."
            )
            return

        LOGGER.info(
            "telegram model list sent chat_id=%s models_count=%s current_model_id=%s",
            message.chat.id,
            len(available_models),
            user_settings.model_id,
        )
        await message.answer(
            _format_models_list(
                available_models,
                current_model_id=user_settings.model_id,
            ),
            reply_markup=_build_model_keyboard(
                available_models,
                current_model_id=user_settings.model_id,
            ),
        )

    @router.callback_query(F.data.startswith(MODEL_CALLBACK_PREFIX))
    async def select_model(callback: CallbackQuery) -> None:
        selected_model_id = (callback.data or "").removeprefix(MODEL_CALLBACK_PREFIX)
        message = callback.message
        chat_id = message.chat.id if message is not None else callback.from_user.id
        LOGGER.info(
            "telegram model selected chat_id=%s user_id=%s model_id=%s",
            chat_id,
            callback.from_user.id,
            selected_model_id,
        )
        try:
            available_models = _enabled_models(await api_client.list_models())
        except httpx.HTTPError:
            LOGGER.exception("Could not load available models")
            await callback.answer(
                "Не удалось получить список моделей. Попробуйте позже.",
                show_alert=True,
            )
            return
        if _model_by_id(available_models, selected_model_id) is None:
            await callback.answer("Модель недоступна.", show_alert=True)
            return

        try:
            user_settings = await settings_store.get(chat_id)
            if user_settings.model_id == selected_model_id:
                await callback.answer("Эта модель уже выбрана.")
                return
            user_settings.model_id = selected_model_id
            await settings_store.save(user_settings)
        except Exception:
            LOGGER.exception("Could not save selected model for chat %s", chat_id)
            await callback.answer(
                "Не удалось сохранить настройки. Попробуйте позже.",
                show_alert=True,
            )
            return

        await callback.answer("Модель выбрана.")
        models_text = _format_models_list(
            available_models,
            current_model_id=selected_model_id,
        )
        response_text = (
            "Настройки обновлены:\n"
            f"{_format_settings(user_settings)}\n\n"
            f"{models_text}"
        )
        keyboard = _build_model_keyboard(
            available_models,
            current_model_id=selected_model_id,
        )
        if message is not None:
            await message.edit_text(response_text, reply_markup=keyboard)
        LOGGER.info(
            "telegram selected model saved chat_id=%s model_id=%s reference_set=%s",
            chat_id,
            selected_model_id,
            user_settings.reference_image is not None,
        )

    @router.message(Command("set_settings"))
    async def set_settings(message: Message, command: CommandObject) -> None:
        _log_command(message, "set_settings")
        try:
            user_settings = await settings_store.get(message.chat.id)
        except Exception:
            LOGGER.exception(
                "Could not load user settings for chat %s",
                message.chat.id,
            )
            await message.answer("Не удалось загрузить настройки. Попробуйте позже.")
            return

        args = (command.args or "").split(maxsplit=1)
        if len(args) != 2:
            await message.answer("Формат: /set_settings <param> <value>")
            return

        param, raw_value = args[0].strip(), args[1].strip()
        clear_values = {"none", "null", "off", "clear"}
        if param in {"model", "model_id"}:
            selected_model_id = _normalize_model_id(raw_value)
            LOGGER.info(
                "telegram setting update requested chat_id=%s param=%s value=%s",
                message.chat.id,
                param,
                selected_model_id,
            )
            try:
                models = [
                    model
                    for model in await api_client.list_models()
                    if model["enabled"]
                ]
            except httpx.HTTPError:
                LOGGER.exception("Could not load available models")
                await message.answer(
                    "Не удалось получить список моделей. Попробуйте позже."
                )
                return
            if _model_by_id(models, selected_model_id) is None:
                available_models = ", ".join(model["model_id"] for model in models)
                await message.answer(
                    f"Модель недоступна. Доступные модели: {available_models}"
                )
                return
            user_settings.model_id = selected_model_id
        elif param == "seed":
            LOGGER.info(
                "telegram setting update requested chat_id=%s param=%s value=%s",
                message.chat.id,
                param,
                raw_value,
            )
            if raw_value.lower() in clear_values:
                user_settings.seed = None
            else:
                try:
                    user_settings.seed = int(raw_value)
                except ValueError:
                    await message.answer("seed должен быть целым числом.")
                    return
        else:
            LOGGER.info(
                "telegram option update requested chat_id=%s option=%s value=%s",
                message.chat.id,
                param,
                raw_value,
            )
            if raw_value.lower() in clear_values:
                user_settings.options.pop(param, None)
            else:
                user_settings.options[param] = _parse_setting_value(raw_value)

        try:
            await settings_store.save(user_settings)
        except Exception:
            LOGGER.exception(
                "Could not save user settings for chat %s",
                message.chat.id,
            )
            await message.answer("Не удалось сохранить настройки. Попробуйте позже.")
            return
        await message.answer(
            "Настройки обновлены:\n" + _format_settings(user_settings)
        )
        LOGGER.info(
            "telegram settings updated chat_id=%s model_id=%s seed_set=%s "
            "reference_set=%s options_keys=%s",
            message.chat.id,
            user_settings.model_id,
            user_settings.seed is not None,
            user_settings.reference_image is not None,
            sorted(user_settings.options),
        )

    @router.message(Command("set_reference"))
    async def set_reference(message: Message, bot: Bot) -> None:
        _log_command(message, "set_reference")
        image_bytes = await _download_photo_from_message(message, bot)
        if image_bytes is None:
            await message.answer(
                "Отправьте /set_reference вместе с изображением "
                "или ответьте этой командой на изображение."
            )
            return
        validation_error = _validate_image_upload(
            image_bytes,
            max_bytes=max_image_bytes,
            max_side=max_image_side,
        )
        if validation_error is not None:
            await message.answer(validation_error)
            return
        try:
            user_settings = await settings_store.get(message.chat.id)
        except Exception:
            LOGGER.exception(
                "Could not load user settings for chat %s",
                message.chat.id,
            )
            await message.answer("Не удалось загрузить настройки. Попробуйте позже.")
            return
        user_settings.reference_image = image_bytes
        try:
            await settings_store.save(user_settings)
        except Exception:
            LOGGER.exception("Could not save reference for chat %s", message.chat.id)
            await message.answer(
                "Не удалось сохранить reference image. Попробуйте позже."
            )
            return
        await message.answer("Reference image сохранен.")
        LOGGER.info(
            "telegram reference saved chat_id=%s size_bytes=%s model_id=%s",
            message.chat.id,
            len(image_bytes),
            user_settings.model_id,
        )

    @router.message(Command("colorize"))
    async def colorize(message: Message, bot: Bot) -> None:
        _log_command(message, "colorize")
        try:
            user_settings = await settings_store.get(message.chat.id)
        except Exception:
            LOGGER.exception(
                "Could not load user settings for chat %s",
                message.chat.id,
            )
            await message.answer("Не удалось загрузить настройки. Попробуйте позже.")
            return
        image_bytes = await _download_photo_from_message(message, bot)
        if image_bytes is None:
            await message.answer(
                "Отправьте /colorize вместе с изображением "
                "или ответьте этой командой на изображение."
            )
            return
        validation_error = _validate_image_upload(
            image_bytes,
            max_bytes=max_image_bytes,
            max_side=max_image_side,
        )
        if validation_error is not None:
            await message.answer(validation_error)
            return

        try:
            models = await api_client.list_models()
        except httpx.HTTPError:
            LOGGER.exception("Could not load available models")
            await message.answer(
                "Не удалось получить список моделей. Попробуйте позже."
            )
            return
        model_info = _model_by_id(models, user_settings.model_id)
        if model_info is None or not model_info["enabled"]:
            await message.answer(
                "Выбранная модель недоступна. "
                "Используйте /set_settings model_id <model_id>."
            )
            return
        if model_info["requires_reference"] and user_settings.reference_image is None:
            await message.answer(
                "Этой модели нужен reference image. "
                "Сначала отправьте /set_reference с изображением."
            )
            return

        try:
            job = await api_client.create_colorization_job(
                image_bytes,
                model_id=user_settings.model_id,
                chat_id=message.chat.id,
                reference_bytes=(
                    user_settings.reference_image
                    if model_info["requires_reference"]
                    else None
                ),
                seed=user_settings.seed,
                options=user_settings.options,
            )
        except httpx.HTTPError:
            LOGGER.exception("Could not create colorization job")
            await message.answer("Не удалось создать задачу. Попробуйте позже.")
            return
        await message.answer(
            f"Задача поставлена в очередь: {job['job_id']}\n"
            "Результат придет автоматически после обработки."
        )
        LOGGER.info(
            "telegram colorize job created chat_id=%s job_id=%s model_id=%s "
            "input_size_bytes=%s reference_sent=%s",
            message.chat.id,
            job["job_id"],
            user_settings.model_id,
            len(image_bytes),
            model_info["requires_reference"],
        )

    @router.message(F.photo)
    async def photo_without_command(message: Message) -> None:
        LOGGER.info("telegram photo without command chat_id=%s", message.chat.id)
        await message.answer(
            "Для запуска используйте /colorize + изображение. "
            "Для reference image используйте /set_reference + изображение."
        )

    return router


async def main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bot = Bot(token=settings.token)
    api_client = ColorizationApiClient(settings.api_url)
    event_listener = JobEventListener(settings.redis_url)
    settings_store = PostgresUserSettingsStore(settings.database_url)
    await settings_store.initialize()
    dispatcher = Dispatcher()
    dispatcher.include_router(
        create_router(
            api_client,
            settings_store,
            max_image_bytes=settings.max_reference_bytes,
            max_image_side=settings.max_reference_side,
        )
    )
    listener_task = asyncio.create_task(
        deliver_job_events(
            bot,
            api_client=api_client,
            event_listener=event_listener,
        )
    )
    try:
        await dispatcher.start_polling(bot)
    finally:
        listener_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await listener_task
        await api_client.close()
        await settings_store.close()


if __name__ == "__main__":
    asyncio.run(main())
