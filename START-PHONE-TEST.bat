@echo off
rem Starts the whole Demo Clinic front desk in this one window: voice bot, Exotel bridge, WhatsApp bot
rem and ngrok, checked end to end. Closing this window stops everything.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0phone-test.ps1"
