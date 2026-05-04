"""Команды /start, /help, /ping."""
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

common_router = Router(name="common")

WELCOME = (
    "Привет! Я локальный ассистент-планировщик.\n\n"
    "• Пришли *текст* — превращу в задачу.\n"
    "• Пришли *голосовое* — расшифрую, оценю эмоцию и тоже сделаю задачу.\n\n"
    "Все модели работают локально, ничего во внешний мир не уходит."
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
