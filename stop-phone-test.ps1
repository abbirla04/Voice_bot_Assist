# Stops everything START-PHONE-TEST.bat started: its window, the voice bot, the Exotel bridge,
# the WhatsApp bot and ngrok (and a dashboard, if one was started by hand).
$PidFile = Join-Path $PSScriptRoot "logs\phone-test.pids"
if (Test-Path $PidFile) {
    foreach ($id in Get-Content $PidFile) {
        if (-not $id) { continue }
        $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
        # only ours: a PID from an old run may belong to another program by now
        if ($proc -and $proc.ProcessName -eq "powershell") { taskkill /PID $id /T /F | Out-Null }
    }
    Remove-Item $PidFile -Force
}
Get-Process ngrok -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "Stopped the Demo Clinic voice bot, bridge, WhatsApp bot and ngrok." -ForegroundColor Green
