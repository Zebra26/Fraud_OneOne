param(
  [string]$ComposeFile = 'docker-compose.dev.yml',
  [switch]$CleanVolumes
)

$ErrorActionPreference = 'Stop'

function Info($msg){ Write-Host "[dev_down] $msg" -ForegroundColor Cyan }
function Fail($msg){ Write-Host "[dev_down] $msg" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $ComposeFile)) { Fail "Compose file not found: $ComposeFile" }

$args = @('-f', $ComposeFile, 'down')
if ($CleanVolumes) { $args += '-v' }

Info 'Stopping services'
& docker compose $args | Write-Host

Write-Host '[dev_down] Done.' -ForegroundColor Green
