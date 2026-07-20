"""FEASIBILITY SPIKE (throwaway) — ScreenCaptureKit system-audio via PyObjC.

Tujuan: menjawab SATU pertanyaan secara definitif di Mac nyata —
  "Apakah delegate audio SCStream benar-benar dipanggil di PyObjC?"

Konteks: PyObjC issue #647 melaporkan pada macOS 15 capture audio SCStream gagal
(-3805 connectionInvalid) ATAU callback audio tak pernah muncul. Swift native
bekerja. Dugaan kuat: delegate tidak di-retain oleh PyObjC. Spike ini me-retain
delegate secara eksplisit dan meng-instrumentasi tiap tahap agar kegagalan dapat
diatribusikan (getShareableContent / createStream / startCapture / callbacks).

BUKAN kode produksi. Hapus setelah keputusan backend M1 diambil.

Cara pakai (di Mac):
  1) uv sync   (atau: pip install pyobjc-framework-ScreenCaptureKit)
  2) Putar audio apa pun (YouTube/musik) — HARUS ada audio sistem berjalan.
  3) python scripts/spike_screencapturekit_audio.py
  4) Saat pertama kali: macOS akan meminta izin "Screen Recording". Beri izin,
     lalu JALANKAN ULANG (izin TCC baru berlaku setelah restart proses).
  5) Amati ringkasan di akhir (10 detik).

Interpretasi hasil:
  - "AUDIO CALLBACKS: N>0" + format terbaca  -> Opsi A (pyobjc) LAYAK.
  - "AUDIO CALLBACKS: 0" atau error -3805     -> Opsi A tidak layak di pyobjc
                                                 -> evidence untuk pivot ke helper Swift.
"""
from __future__ import annotations

import sys
import threading
import time

REQUIRED = ["ScreenCaptureKit", "CoreMedia", "Foundation", "objc"]
missing = []
for mod in REQUIRED:
    try:
        __import__(mod)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{mod}: {exc}")
if missing:
    print("[FATAL] modul native tidak tersedia (jalankan di macOS dengan pyobjc):")
    for m in missing:
        print("   -", m)
    sys.exit(2)

import objc  # noqa: E402
import ScreenCaptureKit as SCK  # noqa: E402
import CoreMedia as CM  # noqa: E402
from Foundation import NSObject, NSRunLoop, NSDate  # noqa: E402

# libdispatch queue untuk sample handler. Nama modul pyobjc bisa berbeda antar versi.
try:
    import dispatch  # type: ignore
except Exception:  # noqa: BLE001
    try:
        import Dispatch as dispatch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] modul libdispatch tidak tersedia ({exc}). "
              f"Coba: pip install pyobjc-framework-libdispatch")
        sys.exit(2)


CAPTURE_SECONDS = 10.0

# Counter bersama antara delegate (thread dispatch) dan main thread.
_stats = {"audio_cb": 0, "other_cb": 0, "error": None, "started": False, "asbd": None}
_lock = threading.Lock()


class SpikeDelegate(NSObject):
    """SCStreamDelegate + SCStreamOutput. HARUS di-retain (disimpan di variabel)."""

    def stream_didStopWithError_(self, stream, error):  # SCStreamDelegate
        with _lock:
            _stats["error"] = str(error)
        print(f"[delegate] stream stopped with error: {error}")

    def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, out_type):  # SCStreamOutput
        # out_type: 0=screen, 1=audio, 2=microphone (SCStreamOutputType)
        if out_type == SCK.SCStreamOutputTypeAudio:
            with _lock:
                _stats["audio_cb"] += 1
                if _stats["asbd"] is None:
                    _stats["asbd"] = _read_audio_format(sample_buffer)
        else:
            with _lock:
                _stats["other_cb"] += 1


def _read_audio_format(sample_buffer):
    """Baca AudioStreamBasicDescription (sampleRate/channels/format) — bukti format."""
    try:
        fmt = CM.CMSampleBufferGetFormatDescription(sample_buffer)
        if fmt is None:
            return "format-desc None"
        asbd = CM.CMAudioFormatDescriptionGetStreamBasicDescription(fmt)
        if asbd is None:
            return "asbd None"
        return f"sampleRate={getattr(asbd, 'mSampleRate', '?')} channels={getattr(asbd, 'mChannelsPerFrame', '?')} formatID={getattr(asbd, 'mFormatID', '?')}"
    except Exception as exc:  # noqa: BLE001
        return f"format read error: {exc}"


def _get_shareable_content(timeout=10.0):
    """Async SCShareableContent -> sync via event; pompa runloop."""
    result = {"content": None, "error": None}
    done = threading.Event()

    def handler(content, error):
        result["content"] = content
        result["error"] = error
        done.set()

    SCK.SCShareableContent.getShareableContentWithCompletionHandler_(handler)
    deadline = time.time() + timeout
    while not done.is_set() and time.time() < deadline:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
    return result["content"], result["error"]


def main():
    print("=== SPIKE ScreenCaptureKit audio (pyobjc) ===")
    print(f"pyobjc: {objc.__version__ if hasattr(objc,'__version__') else '?'}  python: {sys.version.split()[0]}")

    print("[1] getShareableContent ...")
    content, err = _get_shareable_content()
    if err is not None or content is None:
        print(f"[FATAL] getShareableContent gagal: {err}. "
              f"Biasanya izin 'Screen Recording' belum diberikan -> beri izin di "
              f"System Settings > Privacy & Security > Screen Recording, lalu jalankan ulang.")
        sys.exit(3)
    displays = content.displays()
    if not displays:
        print("[FATAL] tidak ada display.")
        sys.exit(3)
    display = displays[0]
    print(f"    OK — displays={len(displays)}, using display {display.displayID()}")

    print("[2] build content filter + config (audio-only intent) ...")
    # SCStream tetap butuh konten layar; kita minimalkan video, aktifkan audio.
    scfilter = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])
    config = SCK.SCStreamConfiguration.alloc().init()
    config.setCapturesAudio_(True)
    config.setExcludesCurrentProcessAudio_(False)
    config.setSampleRate_(48000)
    config.setChannelCount_(2)
    # video minimal (SCStream mensyaratkan config video walau kita hanya butuh audio)
    config.setWidth_(2)
    config.setHeight_(2)
    print("    OK — capturesAudio=True, sampleRate=48000, channels=2")

    print("[3] create SCStream + register audio output (delegate DI-RETAIN) ...")
    delegate = SpikeDelegate.alloc().init()  # disimpan -> di-retain
    stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(scfilter, config, delegate)
    queue = dispatch.dispatch_queue_create(b"spike.audio", None)
    ok, add_err = stream.addStreamOutput_type_sampleHandlerQueue_error_(
        delegate, SCK.SCStreamOutputTypeAudio, queue, None
    )
    if not ok:
        print(f"[FATAL] addStreamOutput(audio) gagal: {add_err}")
        sys.exit(4)
    print("    OK — audio output terdaftar")

    print("[4] startCapture ...")
    start_done = threading.Event()
    start_res = {"error": None}

    def start_handler(error):
        start_res["error"] = error
        start_done.set()

    stream.startCaptureWithCompletionHandler_(start_handler)
    deadline = time.time() + 8.0
    while not start_done.is_set() and time.time() < deadline:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
    if start_res["error"] is not None:
        print(f"[FATAL] startCapture gagal: {start_res['error']}")
        sys.exit(5)
    with _lock:
        _stats["started"] = True
    print(f"    OK — capturing {CAPTURE_SECONDS:.0f}s. PUTAR AUDIO SEKARANG bila belum.")

    t_end = time.time() + CAPTURE_SECONDS
    while time.time() < t_end:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
        with _lock:
            if _stats["error"] and "-3805" in str(_stats["error"]):
                break

    stop_done = threading.Event()
    stream.stopCaptureWithCompletionHandler_(lambda e: stop_done.set())
    d2 = time.time() + 3.0
    while not stop_done.is_set() and time.time() < d2:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))

    with _lock:
        print("\n===================== HASIL SPIKE =====================")
        print(f"AUDIO CALLBACKS : {_stats['audio_cb']}")
        print(f"OTHER CALLBACKS : {_stats['other_cb']}")
        print(f"AUDIO FORMAT    : {_stats['asbd']}")
        print(f"STREAM ERROR    : {_stats['error']}")
        verdict = "LAYAK (pyobjc bisa capture audio)" if _stats["audio_cb"] > 0 else \
                  "TIDAK LAYAK di pyobjc (nol callback / error) -> pertimbangkan helper Swift"
        print(f"VERDICT         : {verdict}")
        print("======================================================")


if __name__ == "__main__":
    main()
