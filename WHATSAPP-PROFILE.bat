@echo off
rem Sets the WhatsApp number's profile photo (assets\whatsapp-dp.jpg) and its "About" line.
rem Not included in the repo: put your clinic's logo at assets\whatsapp-dp.jpg first (640x640 JPG).
cd /d "%~dp0"
set PYTHONUTF8=1
uv run --frozen python tools/logtee.py logs\whatsapp-profile.log -- python src/whatsapp_bot.py profile assets\whatsapp-dp.jpg
pause
