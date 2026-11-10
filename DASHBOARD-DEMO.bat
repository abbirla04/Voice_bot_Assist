@echo off
rem The dashboard with sample calls and chats: see it working without a real call.
rem Runs on its own port (8768), so it never mixes with the real dashboard. Close this window to stop.
cd /d "%~dp0"
set PYTHONUTF8=1
set DASHBOARD_PORT=8768
start "" cmd /c "timeout /t 5 >nul & start http://127.0.0.1:8768"
uv run --frozen python src/dashboard_hub.py demo
pause
