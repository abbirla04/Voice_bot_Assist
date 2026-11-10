@echo off
rem Samples of calm Indian female voices saying Clara's real lines, next to the current voice.
rem Saves them to voice-samples\ and opens the folder. Pick one by number.
cd /d "%~dp0"
set PYTHONUTF8=1
uv run --frozen python tools/voice_samples.py
pause
