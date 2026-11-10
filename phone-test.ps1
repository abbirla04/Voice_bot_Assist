# Demo Clinic voicebot (Clara): everything in one window.
# Installs what is missing, starts the voice bot (agent), the Exotel bridge, the WhatsApp bot
# and ngrok in the background (no window each), records the greetings, and places fake Exotel
# calls (greeting + one caller turn) locally and through ngrok. No dashboard (set DASHBOARD=1
# and run src/dashboard_hub.py to bring it back). This window is the bot: closing it (or
# pressing Q) stops everything.
# Run again any time: it restarts everything cleanly (calls drop for about a minute).
# Every run keeps its logs in logs\runs\<date-time>\ : agent.log, bridge.log, ngrok.log,
# warmup.log, sim-*.log, starter.log. Call reports: logs\calls\*.json
# Double-click START-PHONE-TEST.bat (or: powershell -ExecutionPolicy Bypass -File phone-test.ps1)

$ErrorActionPreference = "Continue"  # native tools write progress to stderr; errors are checked explicitly
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root
# Your ngrok domain and test numbers live in .env.local (not committed):
#   NGROK_DOMAIN=your-name.ngrok-free.dev
#   PHONE_TEST_NUMBERS=011-xxx-xxxxx (pin xxxx-xxxx-xx)
function EnvLocal($name) {
    $f = Join-Path $Root ".env.local"
    if (-not (Test-Path $f)) { return $null }
    $line = Get-Content -LiteralPath $f | Where-Object { $_ -match "^\s*$name\s*=" } | Select-Object -Last 1
    if ($line) { return ($line -replace "^\s*$name\s*=\s*", "").Trim() }
    return $null
}
$Domain = EnvLocal "NGROK_DOMAIN"
if (-not $Domain) { $Domain = "your-name.ngrok-free.dev" }
$Numbers = EnvLocal "PHONE_TEST_NUMBERS"
if (-not $Numbers) { $Numbers = "your ExoPhone numbers (set PHONE_TEST_NUMBERS in .env.local)" }
$Port = 8765
$PidFile = Join-Path $Root "logs\phone-test.pids"
$RunDir = Join-Path $Root ("logs\runs\" + (Get-Date -Format "yyyyMMdd-HHmmss"))
$CallerWav = "bench\results\caller-sample.wav"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
# test calls are recorded (caller + bot WAVs) so missed speech can be heard and re-tested:
# STT-COMPARE.bat runs them through every speech recogniser
$env:EXOTEL_RECORD_DIR = Join-Path $Root "logs\audio"
# One bot (Deepgram -> Claude -> ElevenLabs) answers every number, the 095 line included.
New-Item -ItemType Directory -Force $RunDir | Out-Null
Set-Content -Path (Join-Path $Root "logs\runs\latest.txt") -Value $RunDir
$host.UI.RawUI.WindowTitle = "Demo Clinic front desk - starting"
try { Start-Transcript -Path (Join-Path $RunDir "starter.log") | Out-Null } catch { }

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "   OK  $text" -ForegroundColor Green }
function Warn($text) { Write-Host "   !!  $text" -ForegroundColor Yellow }
function Fail($text) {
    Write-Host "`nSTOPPED: $text" -ForegroundColor Red
    Write-Host "Logs of this run: $RunDir"
    try { Stop-Transcript | Out-Null } catch { }
    Read-Host "Press Enter to close"
    exit 1
}
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}
function Ensure($cmd, $wingetId) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) { Ok "$cmd found"; return }
    Step "Installing $cmd with winget ($wingetId)"
    winget install --id $wingetId -e
    Refresh-Path
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Fail "$cmd is still missing. Install it (winget install $wingetId), then run this again."
    }
    Ok "$cmd installed"
}
# run a command here, showing its output and keeping it in $RunDir\<name>.log
function Logged($name, $command) {
    $log = Join-Path $RunDir "$name.log"
    $uvArgs = @("run", "python", "tools/logtee.py", $log, "--") + $command
    & uv @uvArgs | Out-Host  # Out-Host: the output is shown, never returned as the result
    return $LASTEXITCODE
}
# The parts of the bot run with no window of their own, tied to this window by a Windows
# job object: when this window closes, however it closes, Windows stops all of them.
$JobCode = @"
using System;
using System.Runtime.InteropServices;
public static class FrontDeskJob {
    [StructLayout(LayoutKind.Sequential)]
    public struct BasicLimit {
        public long PerProcessUserTimeLimit; public long PerJobUserTimeLimit; public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize; public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit; public UIntPtr Affinity; public uint PriorityClass;
        public uint SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct IoCounters { public ulong R, W, O, RB, WB, OB; }
    [StructLayout(LayoutKind.Sequential)]
    public struct ExtendedLimit {
        public BasicLimit Basic; public IoCounters Io; public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit; public UIntPtr PeakProcessMemoryUsed; public UIntPtr PeakJobMemoryUsed;
    }
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref ExtendedLimit info, uint size);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    // the handle is never closed: it closes when this window's process ends, and then
    // KILL_ON_JOB_CLOSE stops every process in the job
    public static IntPtr Create() {
        IntPtr job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) return IntPtr.Zero;
        ExtendedLimit info = new ExtendedLimit();
        info.Basic.LimitFlags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (!SetInformationJobObject(job, 9, ref info, (uint)Marshal.SizeOf(typeof(ExtendedLimit)))) {
            return IntPtr.Zero;
        }
        return job;
    }
    public static bool Add(IntPtr job, IntPtr process) { return AssignProcessToJobObject(job, process); }
}
"@
$Job = [IntPtr]::Zero
try {
    Add-Type -TypeDefinition $JobCode -ErrorAction Stop
    $Job = [FrontDeskJob]::Create()
} catch { $Job = [IntPtr]::Zero }
$Parts = [ordered]@{}  # name -> label, process, log
# start a part of the bot in the background; its output is kept in $RunDir\<name>.log
function Start-Part($label, $name, $command) {
    $log = Join-Path $RunDir "$name.log"
    $full = "Set-Location -LiteralPath '$Root'; " +
            "`$env:PYTHONUTF8 = '1'; `$env:PYTHONUNBUFFERED = '1'; " +
            "uv run python tools/logtee.py '$log' -- $command"
    $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($full))
    # no console window at all (CREATE_NO_WINDOW): with Windows Terminal as the default
    # terminal, a "hidden" Start-Process still opens a tab
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "powershell.exe"
    $psi.Arguments = "-NoProfile -EncodedCommand $enc"
    $psi.WorkingDirectory = $Root
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    if ($Job -ne [IntPtr]::Zero) {
        try { [void][FrontDeskJob]::Add($Job, $p.Handle) } catch { }
    }
    Add-Content -Path $PidFile -Value $p.Id
    $Parts[$name] = @{ Label = $label; Process = $p; Log = $log; Down = $false }
}
# the last lines a part wrote, to see why it stopped
function Show-Log($name, $lines = 12) {
    $log = Join-Path $RunDir "$name.log"
    Write-Host "   last lines of $log :" -ForegroundColor Yellow
    Get-Content -LiteralPath $log -Tail $lines -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
}
# stop what the last run started (its window too, if it is still open)
function Stop-Previous {
    if (Test-Path $PidFile) {
        foreach ($id in Get-Content $PidFile) {
            if (-not $id -or [int]$id -eq $PID) { continue }
            $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
            # only ours: a PID from an old run may belong to another program by now
            if ($proc -and $proc.ProcessName -eq "powershell") { taskkill /PID $id /T /F | Out-Null }
        }
        Remove-Item $PidFile -Force
    }
}
function Wait-Http($url, $seconds) {
    for ($i = 0; $i -lt $seconds; $i++) {
        try { return Invoke-RestMethod -Uri $url -TimeoutSec 2 } catch { Start-Sleep -Seconds 1 }
    }
    return $null
}
function Get-Tunnel {
    try {
        $t = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 2
        return ($t.tunnels | Where-Object { $_.public_url -like "*$Domain*" } | Select-Object -First 1)
    } catch { return $null }
}
function Sim($name, $url, $to = "01140000000") {
    $cmd = @("python", "bench/exotel_sim.py", "--url", $url, "--to", $to)
    if (Test-Path (Join-Path $Root $CallerWav)) { $cmd += @("--wav", $CallerWav) }
    return (Logged $name $cmd)
}

Write-Host "Logs of this run: $RunDir"

Step "1/7  Tools"
Ensure "uv" "astral-sh.uv"
Ensure "ngrok" "Ngrok.Ngrok"
if (-not (Test-Path (Join-Path $Root ".env.local"))) { Fail ".env.local (your API keys) is missing from $Root" }
Ok ".env.local found"

Step "2/7  Stopping the running bot (calls DROP from now until step 5 says the line is up)"
Stop-Previous
$old = Get-Process ngrok -ErrorAction SilentlyContinue
if ($old) {
    $old | Stop-Process -Force
    Ok "stopped the ngrok that was already running (only one program can use $Domain)"
}
Start-Sleep -Seconds 1
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    $owner = Get-Process -Id $busy[0].OwningProcess -ErrorAction SilentlyContinue
    Fail "port $Port is in use by $($owner.ProcessName) (PID $($busy[0].OwningProcess)). Close it, or set EXOTEL_BRIDGE_PORT in .env.local and use that port for ngrok."
}
Ok "port $Port is free"
$downAt = Get-Date

Step "3/7  Python packages (first run takes a few minutes)"
uv sync --frozen  # exact package versions from uv.lock: nothing updates behind your back
if ($LASTEXITCODE -ne 0) { Fail "uv sync failed (see above)" }
if ((Logged "download" @("python", "src/agent.py", "download-files")) -ne 0) { Warn "model download failed; the agent will try again on start" }

Step "4/7  Keys, config, and recording the greetings"
if ((Logged "preflight" @("python", "src/agent.py", "preflight")) -ne 0) { Fail "preflight is not READY (see the report above)" }
Ok "preflight READY"
if ((Logged "warmup" @("python", "src/agent.py", "warmup")) -ne 0) { Warn "could not record the greetings; calls will speak them live (slower start). See $RunDir\warmup.log" }

Step "5/7  Starting the voice bot, the Exotel bridge, the WhatsApp bot and ngrok"
Add-Content -Path $PidFile -Value $PID  # a new run closes this window (and everything in it)
if ($Job -eq [IntPtr]::Zero) { Warn "closing this window will not stop the bot: press Q, or use STOP-PHONE-TEST.bat" }
Start-Part "Voice bot" "agent" "python src/agent.py dev"
Start-Part "Phone line (Exotel bridge)" "bridge" "python src/exotel_bridge.py"
if (-not (Wait-Http "http://127.0.0.1:$Port/health" 90)) { Show-Log "bridge"; Fail "the bridge did not start (log above)" }
Ok "bridge is up on port $Port"
# WhatsApp bot: Meta reaches it through the bridge at /whatsapp/webhook
Start-Part "WhatsApp bot" "whatsapp" "python src/whatsapp_bot.py serve"
Start-Part "Internet tunnel (ngrok)" "ngrok" "ngrok http --url=$Domain $Port --log stdout"
$tunnel = $null
for ($i = 0; $i -lt 20 -and -not $tunnel; $i++) { Start-Sleep -Seconds 1; $tunnel = Get-Tunnel }
if (-not $tunnel) {
    Warn "retrying with the older ngrok flag (--domain)"
    Get-Process ngrok -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Part "Internet tunnel (ngrok)" "ngrok" "ngrok http --domain=$Domain $Port --log stdout"
    for ($i = 0; $i -lt 20 -and -not $tunnel; $i++) { Start-Sleep -Seconds 1; $tunnel = Get-Tunnel }
}
if (-not $tunnel) { Show-Log "ngrok"; Fail "ngrok did not open $Domain (log above; ERR_NGROK_334 = the domain is still in use somewhere else)" }
$down = [int]((Get-Date) - $downAt).TotalSeconds
Ok "LINE IS UP: $($tunnel.public_url) -> localhost:$Port (calls were dropping for $down s)"
Write-Host "   waiting 20 s for the agent to register with LiveKit..."
Start-Sleep -Seconds 20

Step "6/7  Fake Exotel call to the bridge: greeting + one caller sentence"
$passed = $false
for ($try = 1; $try -le 3 -and -not $passed; $try++) {
    if ((Sim "sim-local" "ws://127.0.0.1:$Port/exotel/stream") -eq 0) { $passed = $true } else {
        Warn "no full answer yet (try $try of 3); the agent may still be warming up"
        Start-Sleep -Seconds 10
    }
}
if (-not $passed) { Show-Log "agent"; Show-Log "bridge"; Fail "the bot did not answer the fake call (logs above)" }
Ok "the bot answered the fake call"

Step "7/7  The same fake call through ngrok, the way Exotel connects"
if ((Sim "sim-ngrok" "wss://$Domain/exotel/stream") -ne 0) { Show-Log "ngrok"; Fail "the fake call through ngrok failed (the local one worked; log above)" }
Ok "the bot answered a fake call through ngrok"
if (-not (Wait-Http "http://127.0.0.1:8766/whatsapp/health" 30)) { Warn "the WhatsApp bot is not answering yet: see $RunDir\whatsapp.log" }

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host " READY. Call the bot (use a phone other than your Exotel-registered one):" -ForegroundColor Green
Write-Host "   $Numbers`: Claude + ElevenLabs voice" -ForegroundColor Green
Write-Host "   WhatsApp: webhook https://$Domain/whatsapp/webhook  (no WhatsApp needed to try it: WHATSAPP-TEST-CHAT.bat)" -ForegroundColor Green
Write-Host " Logs (kept): $RunDir" -ForegroundColor Green
Write-Host " Calls are recorded to logs\audio (test only)." -ForegroundColor Green
Write-Host " Which speech recogniser hears you best: STT-COMPARE.bat (after a call)" -ForegroundColor Green
Write-Host " LLM + voice + recogniser speed check (optional, 2 min): SPEED-CHECK.bat" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green

# -- running: this window watches the parts and is the off switch -----------------------------
$host.UI.RawUI.WindowTitle = "Demo Clinic front desk - RUNNING (close this window to stop)"
Write-Host ""
foreach ($s in $Parts.Values) {
    $state = if ($s.Process.HasExited) { "STOPPED" } else { "running" }
    Write-Host ("   {0,-28} {1}" -f $s.Label, $state) -ForegroundColor $(if ($state -eq "running") { "Green" } else { "Red" })
}
Write-Host ""
Write-Host " Everything runs in this one window. Keep it open (minimising is fine)." -ForegroundColor Cyan
Write-Host " Keys:  L = open this run's logs   Q = stop everything" -ForegroundColor Cyan
Write-Host " Closing this window stops everything too." -ForegroundColor Cyan
try { Stop-Transcript | Out-Null } catch { }
$watching = $true
while ($watching) {
    foreach ($s in $Parts.Values) {
        if (-not $s.Down -and $s.Process.HasExited) {
            $s.Down = $true
            Write-Host ""
            Warn "$($s.Label) STOPPED at $(Get-Date -Format 'HH:mm:ss'). Run START-PHONE-TEST.bat again to restart everything."
            Get-Content -LiteralPath $s.Log -Tail 8 -ErrorAction SilentlyContinue |
                ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
        }
    }
    $key = $null
    try { if ([Console]::KeyAvailable) { $key = [Console]::ReadKey($true).Key } } catch { }
    if ($key -eq "L") { try { Start-Process explorer.exe $RunDir } catch { } }
    elseif ($key -eq "Q") { $watching = $false }
    Start-Sleep -Milliseconds 300
}
Step "Stopping everything"
foreach ($s in $Parts.Values) {
    if (-not $s.Process.HasExited) { taskkill /PID $s.Process.Id /T /F | Out-Null }
}
Get-Process ngrok -ErrorAction SilentlyContinue | Stop-Process -Force
if (Test-Path $PidFile) { Remove-Item $PidFile -Force }
Ok "stopped"
Start-Sleep -Seconds 2
