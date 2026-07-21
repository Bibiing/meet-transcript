"""CI smoke: buktikan executable packaged benar-benar menjalankan jalur CLI.

Bukan sekadar `--help`. Skrip ini men-spawn exe packaged PERSIS seperti GUI
men-spawn subprocess live-nya (stdout=PIPE), dengan mode CLI ringan (`preprocess`,
tanpa server/audio), lalu membuktikan:

  1. exe men-dispatch ke jalur CLI (bukan GUI) — output CLI muncul.
  2. stdout jalur CLI SAMPAI ke parent via PIPE — kritis untuk protokol sentinel
     (status/level/transcript) pada build GUI-subsystem (--windows-console-mode=disable).

Pakai: python scripts/ci_smoke_packaged_cli.py <path-ke-exe>
Exit 0 bila lolos; non-zero + pesan bila gagal.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED_MARKERS = ("Detected OS", "preprocessing")  # hanya dicetak jalur CLI, bukan GUI


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: ci_smoke_packaged_cli.py <exe>", file=sys.stderr)
        return 2
    exe = Path(sys.argv[1]).resolve()
    if not exe.exists():
        print(f"[FAIL] exe tidak ditemukan: {exe}", file=sys.stderr)
        return 2

    out_dir = Path(tempfile.mkdtemp(prefix="pln_smoke_"))
    # Spawn seperti GUI: stdout=PIPE, mode CLI ringan tanpa server/audio.
    proc = subprocess.run(
        [str(exe), "--mode", "preprocess", "--output-dir", str(out_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    output = proc.stdout or ""
    print("--- captured packaged CLI output (via PIPE) ---")
    print(output)
    print("--- exit code:", proc.returncode, "---")

    missing = [m for m in REQUIRED_MARKERS if m not in output]
    if missing:
        print(
            f"[FAIL] penanda jalur CLI hilang dari stdout: {missing}. "
            f"Kemungkinan: exe meluncurkan GUI (dispatch rusak) ATAU stdout GUI-subsystem "
            f"tidak sampai ke PIPE (protokol sentinel akan rusak di paket).",
            file=sys.stderr,
        )
        return 1

    print("[OK] exe packaged men-dispatch ke jalur CLI dan stdout sampai ke parent via PIPE.")

    # 3. Jalur SELF-SPAWN: exe -> spawn subprocess -> subprocess berjalan.
    # Smoke di atas memanggil exe dari LUAR, sehingga tidak pernah melewati
    # `app_executable()`. Jalur itulah yang gagal dengan WinError 2 karena memakai
    # `sys.executable` (python.exe fiktif di direktori ekstraksi onefile).
    print("--- self-spawn check ---")
    spawn = subprocess.run(
        [str(exe), "--selftest-spawn"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    spawn_output = spawn.stdout or ""
    print(spawn_output)
    if spawn.returncode != 0 or "selftest: OK" not in spawn_output:
        print(
            "[FAIL] exe packaged TIDAK dapat men-spawn dirinya sendiri. "
            "Ini regresi WinError 2: executable untuk self-spawn salah di-resolve.",
            file=sys.stderr,
        )
        return 1

    print("[OK] exe packaged berhasil men-spawn subprocess dan subprocess berjalan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
