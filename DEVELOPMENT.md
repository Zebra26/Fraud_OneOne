Prerequisites

- Docker Desktop
- PowerShell (Windows 10/11)
- Python 3.11+ installed and on PATH

Bootstrap (venv + requirements)

- Run: `./scripts/bootstrap.ps1`
  - Creates `.venv` if missing
  - Installs backend requirements with OS markers (Windows-safe)
  - Installs sender requirements (`scripts/requirements-sender.txt`)

Start the stack

- Run: `./scripts/dev_up.ps1`
  - Builds and starts services from `docker-compose.dev.yml`
  - Waits for `http://localhost:8000/admin/health` to be OK

Send synthetic scores

- Activate venv: `./.venv/Scripts/Activate.ps1`
- Quick run: `python -m scripts.send_synthetic_scores --count 10 --sleep-ms 50`
- Or with summary: `./scripts/send_scores.ps1 -Count 100 -DelayMs 100`

Stop the stack

- Run: `./scripts/dev_down.ps1`
  - Add `-CleanVolumes` to remove volumes

Stress test (multi-workers)

- Start with scaled workers and run a standard stress:
  - `./scripts/stress_up.ps1 -Total 5000 -Concurrency 400 -BackendWorkers 8 -InferenceWorkers 8`
  - Overrides env for workers, brings up services, waits for health, runs stress sender, then cleans env override.

