@echo off
rem Chat with the WhatsApp bot in this window: no WhatsApp or Meta needed. /new starts a new chat.
cd /d "%~dp0"
set PYTHONUTF8=1
uv run --frozen python src/whatsapp_bot.py chat
pause
