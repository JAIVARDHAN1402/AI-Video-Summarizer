# Keep the public ngrok URL alive.
#
# A free ngrok session is not guaranteed: the agent can be dropped with
# nothing written to its own log, and the endpoint then answers
# ERR_NGROK_3200 until somebody notices. That is fine while developing and
# not fine ten minutes before a review, so this watches the agent and brings
# it back when it goes.
#
#   .\scripts\keep-tunnel.ps1              watch in this window, Ctrl+C to stop
#   .\scripts\keep-tunnel.ps1 -Detached    watch in the background
#   .\scripts\keep-tunnel.ps1 -Stop        stop watching
#
# It never starts a second agent while one is serving, because a free account
# holds one session at a time and two agents fight over the domain.

param(
    [switch]$Detached,
    [switch]$Stop,
    [int]$EverySeconds = 20
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Logs = Join-Path $Root "logs"
$PidFile = Join-Path $Logs "tunnel-watch.pid"
$LogFile = Join-Path $Logs "tunnel-watch.log"

New-Item -ItemType Directory -Force -Path $Logs | Out-Null

function Write-Line([string]$text) {
    # Add-Content with an explicit encoding, not Tee-Object: in Windows
    # PowerShell 5.1 Tee writes UTF-16, which makes the log unreadable in
    # anything that assumes text.
    $line = "{0}  {1}" -f (Get-Date -Format "HH:mm:ss"), $text
    Add-Content -Path $LogFile -Value $line -Encoding utf8
    Write-Output $line
}

if ($Stop) {
    if (Test-Path $PidFile) {
        $watchPid = (Get-Content $PidFile -Raw).Trim()
        try {
            Stop-Process -Id ([int]$watchPid) -Force -ErrorAction Stop
            "stopped the tunnel watcher (pid $watchPid)"
        } catch {
            "no watcher running (stale pid $watchPid)"
        }
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    } else {
        "no watcher running"
    }
    return
}

if ($Detached) {
    $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath,
              "-EverySeconds", $EverySeconds)
    $proc = Start-Process -FilePath "powershell" -ArgumentList $args `
        -WindowStyle Hidden -PassThru
    Set-Content -Path $PidFile -Value $proc.Id -Encoding ascii
    "watching the tunnel in the background (pid $($proc.Id)); log: logs\tunnel-watch.log"
    "stop it with: .\scripts\keep-tunnel.ps1 -Stop"
    return
}

Set-Content -Path $PidFile -Value $PID -Encoding ascii

# The domain the account owns; without it a restart would hand out a new
# random hostname and every link already written down would be wrong.
$domain = ""
$envFile = Join-Path $Root ".env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*VS_NGROK_DOMAIN\s*=\s*(.+?)\s*$') { $domain = $Matches[1] }
    }
}
if (-not $domain) {
    Write-Line "VS_NGROK_DOMAIN is not set in .env - refusing to watch, because a"
    Write-Line "restart would change the public URL."
    return
}

function Get-TunnelUrl {
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" `
            -TimeoutSec 4 -ErrorAction Stop
        if ($r.tunnels -and $r.tunnels.Count -gt 0) { return $r.tunnels[0].public_url }
    } catch { }
    return $null
}

function Start-Agent {
    # Clear any half-dead agent first: a process that is alive but no longer
    # has a session still occupies the account's single session slot.
    Get-Process -Name "ngrok" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Start-Process -FilePath "ngrok" `
        -ArgumentList @("http", "3000", "--log", "stdout", "--domain", $domain) `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $Logs "ngrok.log") `
        -RedirectStandardError (Join-Path $Logs "ngrok.err.log") | Out-Null
}

Write-Line "watching https://$domain every ${EverySeconds}s"
$wasUp = $null

while ($true) {
    $url = Get-TunnelUrl
    $up = $null -ne $url

    if ($up -ne $wasUp) {
        if ($up) { Write-Line "tunnel up: $url" }
        else { Write-Line "tunnel down" }
        $wasUp = $up
    }

    if (-not $up) {
        # Only restart when the web app is actually there to serve; otherwise
        # the agent would come up pointing at nothing and flap.
        $webUp = $null -ne (Get-NetTCPConnection -LocalPort 3000 -State Listen `
                            -ErrorAction SilentlyContinue)
        if ($webUp) {
            Write-Line "restarting the ngrok agent"
            try { Start-Agent } catch { Write-Line "restart failed: $_" }
            Start-Sleep -Seconds 6
        } else {
            Write-Line "web app is not listening on 3000 - waiting rather than restarting"
        }
    }

    Start-Sleep -Seconds $EverySeconds
}
