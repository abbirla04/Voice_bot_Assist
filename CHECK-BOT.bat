@echo off
rem Checks the bot still behaves like the known-good version. See check-bot.ps1
powershell -ExecutionPolicy Bypass -File "%~dp0check-bot.ps1"
