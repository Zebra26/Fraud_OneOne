param(
  [int]$Count = 50,
  [int]$DelayMs = 200,
  [switch]$VerboseOutput
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path 'scripts/send_synthetic_scores.py')) {
  Write-Error 'Run this script from the repo root (contains scripts\send_synthetic_scores.py).'
}

$python = 'python'
try {
  $ver = & $python -c "import sys; print(sys.version)" 2>$null
} catch {
  Write-Error 'Python not found in PATH. Activate your venv first.'
}

$argsList = @('scripts\send_synthetic_scores.py', '--count', $Count, '--sleep-ms', $DelayMs, '--machine')
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $python
# Build one string of arguments (PS 5.1 compatible)
$quoted = $argsList | ForEach-Object { if ($_ -is [string] -and $_.Contains(' ')) { '"' + $_ + '"' } else { $_ } }
$psi.Arguments = ($quoted -join ' ')
$psi.WorkingDirectory = (Get-Location).Path
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false

$proc = [System.Diagnostics.Process]::Start($psi)
if ($null -eq $proc) { Write-Error 'Failed to start Python process.' }
$stdout = @()
while (-not $proc.HasExited) {
  $line = $proc.StandardOutput.ReadLine()
  if ($null -ne $line) { $stdout += $line; if ($VerboseOutput) { Write-Host $line } }
}
while (-not $proc.StandardOutput.EndOfStream) {
  $stdout += $proc.StandardOutput.ReadLine()
}
$stderr = $proc.StandardError.ReadToEnd()
if ($stderr) { Write-Verbose $stderr }

$summary = @{
  FRAUD  = 0
  NORMAL = 0
  ERROR  = 0
}

foreach ($line in $stdout) {
  try {
    $obj = $line | ConvertFrom-Json -ErrorAction Stop
    if ($obj.status -eq 200 -and $obj.decision) {
      $summary[$obj.decision]++
    } else {
      $summary['ERROR']++
    }
  } catch {
    $summary['ERROR']++
  }
}

Write-Host ("Total: {0} | FRAUD: {1} | NORMAL: {2} | ERROR: {3}" -f ($Count), $summary.FRAUD, $summary.NORMAL, $summary.ERROR)
