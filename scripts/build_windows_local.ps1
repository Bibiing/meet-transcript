# Build Windows lokal — CERMINAN PERSIS langkah Nuitka di .github/workflows/build-nuitika.yaml.
#
# Tujuan: menemukan bug packaging di mesin sendiri, TANPA menunggu CI. Kelas bug seperti
# data file paket yang hilang (soundcard .h) dan UCRT yang salah bundel hanya muncul pada
# artifact terpaket, bukan saat `python -m src.app`.
#
# Flag WAJIB identik dengan CI. Bila CI berubah, perbarui file ini bersamaan.
#
# Pakai:
#   pwsh -File scripts/build_windows_local.ps1
#   pwsh -File scripts/build_windows_local.ps1 -SkipSmoke

param(
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# PARITAS CI: CI memakai Python 3.11. Bila versi lokal berbeda, hasil smoke test
# tidak dapat dipercaya di dua arah (gagal palsu maupun lolos palsu). Utamakan
# .venv311 bila ada; buat dengan: uv venv .venv311 --python 3.11
$python = Join-Path $repo ".venv311\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = Join-Path $repo ".venv\Scripts\python.exe" }
if (-not (Test-Path $python)) { $python = "python" }
$ver = & $python -c "import sys;print('.'.join(map(str,sys.version_info[:2])))"
if ($ver -ne "3.11") { Write-Warning "Python lokal $ver != CI 3.11 - hasil smoke test kurang dapat dipercaya" }

$output = "ListenPLN-Windows.exe"

Write-Host "== Build Nuitka (flag identik CI) ==" -ForegroundColor Cyan
& $python -m nuitka `
    --standalone `
    --onefile `
    --plugin-enable=pyside6 `
    --include-data-dir=src/icon=src/icon `
    --windows-icon-from-ico=src/icon/app.ico `
    --include-data-dir=config=config `
    --include-package=src `
    --include-module=sounddevice `
    --include-module=soundcard `
    --include-package-data=soundcard `
    --nofollow-import-to=whisper,numba,torch,torchaudio `
    --include-windows-runtime-dlls=no `
    --windows-console-mode=disable `
    --remove-output `
    --assume-yes-for-downloads `
    src/app.py
if ($LASTEXITCODE -ne 0) { throw "Nuitka build gagal" }

if (Test-Path $output) { Remove-Item $output -Force }
Move-Item app.exe $output
Write-Host "Artifact: $output" -ForegroundColor Green

if (-not $SkipSmoke) {
    Write-Host "== Smoke test jalur CLI (sama dengan CI) ==" -ForegroundColor Cyan
    & $python scripts/ci_smoke_packaged_cli.py ".\$output"
    if ($LASTEXITCODE -ne 0) { throw "Smoke test gagal" }

    Write-Host "== Verifikasi UCRT TIDAK ikut dibundel ==" -ForegroundColor Cyan
    # Regresi "Bad Image" di mesin target: apiset UCRT hasil panen mesin build
    # tidak boleh ikut ke dalam payload onefile.
    Get-ChildItem "$env:TEMP\onefile_*" -Recurse -Filter "api-ms-win-crt-*.dll" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty Name -Unique | ForEach-Object { Write-Warning "UCRT ikut terbundel: $_" }
}

Write-Host "SELESAI" -ForegroundColor Green
