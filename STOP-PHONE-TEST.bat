@echo off
rem Stops everything START-PHONE-TEST.bat started (also done by pressing Q or closing its window).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-phone-test.ps1"
pause
