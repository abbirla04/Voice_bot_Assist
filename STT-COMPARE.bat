@echo off
rem Re-runs your recorded test calls through every speech recogniser: which one heard you best?
cd /d "%~dp0"
uv run python tools/logtee.py logs\stt-compare.log -- python bench/stt_compare.py --calls 3
pause
