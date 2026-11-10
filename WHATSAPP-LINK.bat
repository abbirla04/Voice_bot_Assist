@echo off
rem Links the WhatsApp Business Account to the Meta app so incoming WhatsApp messages reach the bot.
rem Safe to run any time (START-PHONE-TEST also does it when the WhatsApp bot starts).
cd /d "%~dp0"
set PYTHONUTF8=1
uv run --frozen python tools/logtee.py logs\whatsapp-link.log -- python src/whatsapp_bot.py link
pause
