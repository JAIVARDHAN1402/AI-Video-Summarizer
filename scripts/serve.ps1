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
#   .\scripts\serve.ps1 -Public    a throwaway Cloudflare URL, no account
#   .\scripts\serve.ps1 -Ngrok     the same URL every time (needs an account)
#
# -Public is the quickest way to show someone the app: no account, no setup,
# and a new random hostname every run. -Ngrok is for when the address has to
# keep working - a report, a slide, somebody checking back tomorrow - and
# reads VS_NGROK_DOMAIN from .env so the hostname is the account's own.
#
# Logs land in logs/api.log and logs/web.log.

param(
    [switch]$Stop,
    [switch]$Status,
    [switch]$Public,
    [switch]$Ngrok
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Logs = Join-Path $Root "logs"
$Pids = Join-Path $Logs "pids.json"

function Get-Running {
    if (-not (Test-Path $Pids)) { return @{} }
    try { return (Get-Content $Pids -Raw | ConvertFrom-Json) } catch { return @{} }
}

function Test-Port([int]$Port) {
    # `npm run dev` is launched through cmd.exe, which hands off to node and
    # exits, so the pid recorded at launch stops existing while the server is
    # still serving. What is actually being asked is whether something is
    # listening, so ask that instead.
    $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Show-Status {
    $saved = Get-Running
    foreach ($entry in @(
        @{ name = "api";    port = 8000 },
        @{ name = "web";    port = 3000 },
        @{ name = "tunnel"; port = 0 },
        @{ name = "ngrok";  port = 0 }
    )) {
        $name = $entry.name
        $procId = $saved.$name
        if ($entry.port -gt 0) {
            $state = if (Test-Port $entry.port) {
                "listening on $($entry.port)"
            } else { "not running" }
        } elseif ($name -eq "ngrok") {
            # Ask the agent, not the pid we recorded. The watcher in
            # keep-tunnel.ps1 starts its own agent when a session drops, so
            # the pid here goes stale and -Status would report "not running"
            # about a tunnel that is serving.
            $url = $null
            try {
                $r = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" `
                    -TimeoutSec 3 -ErrorAction Stop
                if ($r.tunnels -and $r.tunnels.Count -gt 0) { $url = $r.tunnels[0].public_url }
            } catch { }
            $state = if ($url) { "serving $url" } else { "not running" }
        } else {
            $alive = $procId -and ($null -ne (Get-Process -Id $procId -ErrorAction SilentlyContinue))
            $state = if ($alive) { "running (pid $procId)" } else { "not running" }
        }
        "{0,-8} {1}" -f $name, $state
    }
    if ($saved.url) { "public   $($saved.url)" }
}

if ($Status) { Show-Status; exit 0 }

if ($Stop) {
    $saved = Get-Running
    foreach ($name in @("api", "web", "tunnel", "ngrok")) {
        $procId = $saved.$name
        if ($procId) {
            $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
            if ($p) { Stop-Process -Id $procId -Force; "stopped $name (pid $procId)" }
        }
    }
    # And whatever is still holding the ports, which for the web server is a
    # node process cmd.exe left behind.
    foreach ($port in @(8000, 3000)) {
        foreach ($conn in @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) {
            $owner = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
            if ($owner) {
                Stop-Process -Id $owner.Id -Force -ErrorAction SilentlyContinue
                "stopped whatever held port $port (pid $($owner.Id))"
            }
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

if ($Ngrok) {
    # A free ngrok account may hold one agent session at a time. Starting a
    # second one while the first is alive makes them fight over the domain and
    # one of them loses, which is how the public URL kept answering
    # ERR_NGROK_3200 a few seconds after a restart. So: if an agent is already
    # serving, leave it alone.
    $agent = $null
    try {
        $agent = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" `
            -TimeoutSec 3 -ErrorAction Stop
    } catch { $agent = $null }

    $alreadyServing = ($agent -and $agent.tunnels -and $agent.tunnels.Count -gt 0)
    if ($alreadyServing) {
        $state.url = $agent.tunnels[0].public_url
        "ngrok is already serving $($state.url) - leaving it alone"
        "  (run -Stop first if you want a fresh agent)"
    }

    # The domain belongs to the account, so it lives in .env beside the other
    # deployment settings rather than being baked in here.
    $domain = ""
    $envFile = Join-Path $Root ".env"
    if (Test-Path $envFile) {
        foreach ($line in Get-Content $envFile) {
            if ($line -match '^\s*VS_NGROK_DOMAIN\s*=\s*(.+?)\s*$') { $domain = $Matches[1] }
        }
    }
    $ngrokArgs = @("http", "3000", "--log", "stdout")
    if ($domain) {
        $ngrokArgs += @("--domain", $domain)
    } else {
        Write-Warning "VS_NGROK_DOMAIN is not set in .env - starting with a random ngrok URL"
    }
    if (-not $alreadyServing) {
        $ng = Start-Process -FilePath "ngrok" -ArgumentList $ngrokArgs `
            -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $Logs "ngrok.log") `
            -RedirectStandardError (Join-Path $Logs "ngrok.err.log")
        $state.ngrok = $ng.Id
        if ($domain) { $state.url = "https://$domain" }
    }
}

$state | ConvertTo-Json | Set-Content $Pids -Encoding utf8

"started detached - these survive this terminal closing"
"  api  http://localhost:8000   (log: logs\api.log)"
"  web  http://localhost:3000   (log: logs\web.log)"
if ($Public) { "  public URL appears in logs\tunnel.err.log within ~10s" }
if ($Ngrok) {
    if ($state.url) { "  public  $($state.url)   (stable - same every run)" }
    else { "  ngrok URL appears in logs\ngrok.log within ~10s" }
}
""
"stop with:  .\scripts\serve.ps1 -Stop"
