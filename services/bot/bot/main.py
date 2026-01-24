import asyncio
import io

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, Message

from services.bot.bot.config import load_settings
from services.bot.bot.services.api_client import colorize_via_api


def create_router(api_url: str) -> Router:
    router = Router()

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer("Send an illustration and I will colorize it.")

    @router.message(F.photo)
    async def handle_photo(message: Message, bot: Bot) -> None:
        if not message.photo:
            return
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        buffer = io.BytesIO()
        await bot.download_file(file.file_path, destination=buffer)
        colorized = await colorize_via_api(api_url, buffer.getvalue())
        await message.reply_photo(
            BufferedInputFile(colorized, filename="colorized.png")
        )

    return router


async def main() -> None:
    settings = load_settings()
    bot = Bot(token=settings.token)
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router(settings.api_url))
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
