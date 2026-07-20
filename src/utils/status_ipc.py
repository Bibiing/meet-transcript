from __future__ import annotations

import json

# Protokol status terstruktur lintas-proses (BUG-002).
#
# Subprocess live (`src.main`) mengirim status koneksi ke parent desktop
# (`src.core.engine`) lewat stdout yang sama dengan prosa human-readable dan
# transcript. Agar parent tidak perlu menebak status dari substring prosa
# (yang menabrak teks transcript, mis. kata "error"), status dikirim sebagai
# satu baris ber-sentinel yang tidak mungkin bertabrakan dengan baris lain.
#
# Modul ini hanya mendefinisikan *mechanism* (format wire). Pemetaan kode
# status -> state UI (CONNECTING/CONNECTED/ERROR/DISCONNECTED) adalah *policy*
# milik parent dan tidak berada di sini.

STATUS_SENTINEL = "##PLN-STATUS##"
LEVEL_SENTINEL = "##PLN-LEVEL##"
# Kanal MASUK (parent -> subprocess) lewat stdin, untuk kontrol runtime seperti
# true mute: mengubah perilaku sesi tanpa restart subprocess/koneksi.
COMMAND_SENTINEL = "##PLN-CMD##"


def format_status_line(source: str, status: str, **details: object) -> str:
    """Bentuk satu baris status ber-sentinel (termasuk newline penutup).

    Ditulis sebagai satu baris utuh agar parser parent dapat memisahkannya
    dari prosa/transcript hanya dengan memeriksa prefix dan mem-parse JSON.
    `details` opsional membawa data pendamping kode status (mis. `min_version`
    pada penolakan versi); mekanisme ini tetap netral terhadap policy.
    """
    payload = json.dumps({"source": source, "status": status, **details}, ensure_ascii=False)
    return f"{STATUS_SENTINEL}{payload}\n"


def parse_status_line(line: str) -> tuple[str, str, dict] | None:
    """Ekstrak (source, status, details) dari baris stdout, atau None bila bukan sinyal.

    Defensif terhadap baris yang terpotong/tercampur: hanya mengembalikan hasil
    bila baris benar-benar diawali sentinel dan sisanya JSON valid dengan field
    `source` dan `status` bertipe string. Selain itu None (diperlakukan sebagai
    log biasa oleh parent, tidak pernah memengaruhi status).
    """
    stripped = line.strip()
    if not stripped.startswith(STATUS_SENTINEL):
        return None
    raw = stripped[len(STATUS_SENTINEL):]
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    source = payload.get("source")
    status = payload.get("status")
    if not isinstance(source, str) or not isinstance(status, str):
        return None
    details = {k: v for k, v in payload.items() if k not in {"source", "status"}}
    return source, status, details


def format_level_line(source: str, rms_db: float) -> str:
    """Bentuk satu baris level audio ber-sentinel (RMS dB per source).

    Dipakai untuk indikator Mic/Speaker real-time di GUI. Nilai rms_db harus
    berupa float terhingga (pemanggil meng-clamp -inf saat hening).
    """
    payload = json.dumps({"source": source, "rms_db": rms_db})
    return f"{LEVEL_SENTINEL}{payload}\n"


def format_command_line(cmd: str, **fields: object) -> str:
    """Bentuk satu baris perintah ber-sentinel untuk stdin subprocess."""
    payload = json.dumps({"cmd": cmd, **fields})
    return f"{COMMAND_SENTINEL}{payload}\n"


def parse_command_line(line: str) -> tuple[str, dict] | None:
    """Ekstrak (cmd, payload) dari baris perintah, atau None bila bukan perintah.

    Defensif: baris stdin apa pun yang bukan sentinel/JSON valid diabaikan,
    sehingga input tak terduga tidak pernah mengubah perilaku sesi.
    """
    stripped = line.strip()
    if not stripped.startswith(COMMAND_SENTINEL):
        return None
    raw = stripped[len(COMMAND_SENTINEL):]
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    cmd = payload.get("cmd")
    if not isinstance(cmd, str):
        return None
    return cmd, payload


def parse_level_line(line: str) -> tuple[str, float] | None:
    """Ekstrak (source, rms_db) dari baris level, atau None bila bukan sinyal level.

    Defensif seperti parse_status_line: hanya bertindak bila prefix cocok dan JSON
    valid dengan field bertipe benar.
    """
    stripped = line.strip()
    if not stripped.startswith(LEVEL_SENTINEL):
        return None
    raw = stripped[len(LEVEL_SENTINEL):]
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    source = payload.get("source")
    rms_db = payload.get("rms_db")
    if not isinstance(source, str) or not isinstance(rms_db, (int, float)) or isinstance(rms_db, bool):
        return None
    return source, float(rms_db)
