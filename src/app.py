"""Single entry dispatch untuk GUI maupun CLI (Milestone 3).

Ini SATU-SATUNYA jalur startup aplikasi. Baik binari packaged (Nuitka onefile)
maupun invokasi dev diarahkan ke sini, lalu di-dispatch:

  - Ada argumen CLI mode (--mode / --replay-file)  -> jalur CLI (src.main.main)
  - Tanpa argumen tersebut                          -> GUI (src.qt_client.run_gui)

Alasan: di binari packaged, `sys.executable` = exe aplikasi dan Nuitka TIDAK
mendukung `python -m modul`. Maka subprocess live di-spawn sebagai
`exe --mode live ...` yang kembali masuk ke dispatcher ini dan menjalankan CLI.

Import GUI/CLI dilakukan lazy agar modul ini ringan (dipakai juga oleh engine
untuk deteksi frozen tanpa menarik PySide6).
"""
from __future__ import annotations

import sys


def is_frozen() -> bool:
    """True bila berjalan sebagai binari terkompilasi (Nuitka) atau frozen."""
    return bool(getattr(sys, "frozen", False)) or ("__compiled__" in globals())


def is_cli_invocation(argv: list[str]) -> bool:
    """CLI bila ada penanda mode kerja non-GUI (record/preprocess/live/replay)."""
    for arg in argv:
        if arg == "--mode" or arg.startswith("--mode=") or arg == "--replay-file" or arg.startswith("--replay-file="):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if is_cli_invocation(argv):
        from src.main import main as cli_main

        return int(cli_main(argv) or 0)
    from src.qt_client import run_gui

    return int(run_gui() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
