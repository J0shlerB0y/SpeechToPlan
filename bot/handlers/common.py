from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

common_router = Router(name="common")

WELCOME = (
    "это ассистент-планировщик\n\n"
    "* пришли *текст* — превращу в задачу\n"
    "* пришли *голосовое* — расшифрую и сделаю задачу\n\n"
)


@common_router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(WELCOME, parse_mode="Markdown")


@common_router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(WELCOME, parse_mode="Markdown")


@common_router.message(Command("ping"))
async def cmd_ping(message: Message) -> None:
    await message.answer("pong")
