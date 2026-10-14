# Start the API and the web app detached from this terminal.
#
# Both were being run as children of a terminal tab, which is why they kept
# dying mid-job: closing the tab — or the app replacing it between sessions —
# takes the child with it, and a transcription that was 76% done simply
# vanished. Start-Process with its own window detaches them, so they outlive
# whatever started them and only `-Stop` or a reboot ends them.
#
#   .\scripts\serve.ps1            start both, print the URLs
#   .\scripts\serve.ps1 -Stop      stop both
#   .\scripts\serve.ps1 -Status    what is running
#   .\scripts\serve.ps1 -Public    also open a public URL over Cloudflare
#
# Logs land in logs/api.log and logs/web.log.

param(
    [switch]$Stop,
    [switch]$Status,
    [switch]$Public
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Logs = Join-Path $Root "logs"
$Pids = Join-Path $Logs "pids.json"

function Get-Running {
    if (-not (Test-Path $Pids)) { return @{} }
    try { return (Get-Content $Pids -Raw | ConvertFrom-Json) } catch { return @{} }
}

function Show-Status {
    $saved = Get-Running
    foreach ($name in @("api", "web", "tunnel")) {
        $procId = $saved.$name
        $alive = $false
        if ($procId) { $alive = $null -ne (Get-Process -Id $procId -ErrorAction SilentlyContinue) }
        $state = if ($alive) { "running (pid $procId)" } else { "not running" }
        "{0,-8} {1}" -f $name, $state
    }
}

if ($Status) { Show-Status; exit 0 }

if ($Stop) {
    $saved = Get-Running
    foreach ($name in @("api", "web", "tunnel")) {
        $procId = $saved.$name
        if ($procId) {
            $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
            if ($p) { Stop-Process -Id $procId -Force; "stopped $name (pid $procId)" }
        }
    }
    if (Test-Path $Pids) { Remove-Item $Pids }
    exit 0
}

if (-not (Test-Path $Logs)) { New-Item -ItemType Directory $Logs | Out-Null }

# A second copy of either server would bind the same port and serve stale
# code from an older checkout, which has happened and is hard to spot.
.\scripts\serve.ps1 -Stop | Out-Null

$python = Join-Path $Root ".venv\Scripts\python.exe"
$api = Start-Process -FilePath $python -ArgumentList "-m", "server.main" `
    -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $Logs "api.log") `
    -RedirectStandardError (Join-Path $Logs "api.err.log")

$web = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "npm run dev" `
    -WorkingDirectory (Join-Path $Root "web") -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $Logs "web.log") `
    -RedirectStandardError (Join-Path $Logs "web.err.log")

$state = @{ api = $api.Id; web = $web.Id }

if ($Public) {
    $cf = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
    if (Test-Path $cf) {
        # Only port 3000 is exposed: Next.js proxies /api/* to the backend
        # server-side, so the API never needs its own public address.
        $tunnel = Start-Process -FilePath $cf `
            -ArgumentList "tunnel", "--url", "http://localhost:3000", "--no-autoupdate" `
            -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $Logs "tunnel.log") `
            -RedirectStandardError (Join-Path $Logs "tunnel.err.log")
        $state.tunnel = $tunnel.Id
    } else {
        Write-Warning "cloudflared not found at $cf - skipping the public URL"
    }
}

$state | ConvertTo-Json | Set-Content $Pids -Encoding utf8

"started detached - these survive this terminal closing"
"  api  http://localhost:8000   (log: logs\api.log)"
"  web  http://localhost:3000   (log: logs\web.log)"
if ($Public) { "  public URL appears in logs\tunnel.err.log within ~10s" }
""
"stop with:  .\scripts\serve.ps1 -Stop"
