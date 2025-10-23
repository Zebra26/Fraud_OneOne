param(
  [switch]$Sync,
  [switch]$Upgrade
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Ensure-PipTools {
  python -m pip install --upgrade pip setuptools wheel | Out-Null
  python -m pip install --upgrade pip-tools | Out-Null
}

function Compile-RootLock {
  $args = @()
  if ($Upgrade) { $args += '--upgrade' }
  $args += @('-o','requirements.lock.txt','requirements.in')
  python -m piptools compile @args
}

Ensure-PipTools
Compile-RootLock

if ($Sync) {
  python -m piptools sync requirements.lock.txt
}

Write-Host "pip-tools done. Updated requirements.lock.txt" -ForegroundColor Green
