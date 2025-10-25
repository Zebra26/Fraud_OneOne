param(
  [int]$Total = 5000,
  [int]$Concurrency = 400,
  [int]$BackendWorkers = 8,
  [int]$InferenceWorkers = 8,
  [int]$TimeoutSec = 20,
  [switch]$NoBuild
)

$ErrorActionPreference = 'Stop'

function Info($msg){ Write-Host "[stress_up] $msg" -ForegroundColor Cyan }
function Warn($msg){ Write-Host "[stress_up] $msg" -ForegroundColor Yellow }
function Fail($msg){ Write-Host "[stress_up] $msg" -ForegroundColor Red; exit 1 }

if (-not (Test-Path 'docker-compose.dev.yml')) { Fail 'docker-compose.dev.yml not found. Run from repo root.' }

# Prepare env override file
$envFile = Join-Path $PWD '.env.stress'
@(
  "USE_GUNICORN=true",
  "GUNICORN_WORKERS=$BackendWorkers",
  "GUNICORN_TIMEOUT=30",
  "UVICORN_WORKERS=$InferenceWorkers",
  "PERF_MODE=true",
  "USE_LOCAL_INFERENCE=false"
) | Set-Content -Path $envFile -Encoding ASCII
Info "Wrote env overrides to $envFile"

# Bring up services with overrides
$composeArgs = @('-f','docker-compose.dev.yml','--env-file', $envFile, 'up','-d')
if (-not $NoBuild) { $composeArgs += '--build' }
Info "Starting services with BackendWorkers=$BackendWorkers, InferenceWorkers=$InferenceWorkers"
& docker compose $composeArgs 'backend' 'ml-inference' 'kafka-ui' 'redisinsight' 'prometheus' 'grafana' | Write-Host

# Wait for backend health
$healthUrl = 'http://localhost:8000/admin/health'
Info "Waiting for backend health: $healthUrl"
$deadline = (Get-Date).AddMinutes(5)
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

# Run stress test
$python = 'python'
try { & $python -V | Out-Null } catch { Fail 'Python not found in PATH. Activate your venv.' }
Info "Running stress: Total=$Total, Concurrency=$Concurrency, Timeout=$TimeoutSec s"
& $python -m scripts.stress_send_scores --total $Total --concurrency $Concurrency --timeout $TimeoutSec --vary-devices

# Cleanup env file
try {
  Remove-Item -Force $envFile
} catch {
  Warn ("Could not delete {0}: {1}" -f $envFile, $_)
}

Write-Host '[stress_up] Done.' -ForegroundColor Green
