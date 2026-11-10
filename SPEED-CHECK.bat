@echo off
rem LLM + voice + speech-recogniser speed from this PC, kept in logs\speed-check.log
cd /d "%~dp0"
uv run python tools/logtee.py logs\speed-check.log -- python bench/latency.py --only llm tts stt --n 3
pause
