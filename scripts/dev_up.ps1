param(
  [string]$ComposeFile = 'docker-compose.dev.yml',
  [switch]$NoBuild
)

$ErrorActionPreference = 'Stop'

function Info($msg){ Write-Host "[dev_up] $msg" -ForegroundColor Cyan }
function Fail($msg){ Write-Host "[dev_up] $msg" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $ComposeFile)) { Fail "Compose file not found: $ComposeFile" }

$composeArgs = @('-f', $ComposeFile, 'up', '-d')
if (-not $NoBuild) { $composeArgs += '--build' }

Info 'Starting services (backend, ml-inference, kafka-ui, redisinsight, prometheus, grafana)'
& docker compose $composeArgs 'backend' 'ml-inference' 'kafka-ui' 'redisinsight' 'prometheus' 'grafana' | Write-Host

# Wait for backend health
$healthUrl = 'http://localhost:8000/admin/health'
Info "Waiting for backend health: $healthUrl"
$deadline = (Get-Date).AddMinutes(3)
while ((Get-Date) -lt $deadline) {
  try {
    $resp = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 5
    if ($resp.StatusCode -eq 200 -and $resp.Content -match '"status":"ok"') {
      Info 'Backend is healthy.'
      break
    }
  } catch {}
  Start-Sleep -Seconds 2
}
if ((Get-Date) -ge $deadline) { Fail 'Backend did not become healthy in time.' }

Write-Host '[dev_up] Done.' -ForegroundColor Green
