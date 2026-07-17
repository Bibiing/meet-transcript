from __future__ import annotations

import bisect
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class TimelineCheckpoint:
    sent_position: float
    real_position: float


class SentTimeline:
    """
    Kompensasi drift timestamp akibat VAD drop.
    Memetakan waktu relatif yang dilihat server (byte audio yang ia terima)
    kembali ke waktu asli (wall-clock) di klien.
    """

    def __init__(self):
        self._checkpoints: list[TimelineCheckpoint] = []
        # List paralel posisi sent untuk bisect, dijaga sinkron dengan
        # _checkpoints agar map_to_real tidak membangun ulang list tiap panggilan.
        self._sent_positions: list[float] = []
        self._last_offset = 0.0
        self._lock = threading.RLock()

    def record_sent(self, sent_duration: float, real_start: float) -> None:
        """
        Catat waktu pengiriman chunk.
        
        Args:
            sent_duration: Total durasi audio yang telah dikirim SEBELUM chunk ini.
            real_start: Waktu asli (wall-clock/PreprocessedAudioChunk.start_seconds) dari chunk ini.
        """
        with self._lock:
            offset = real_start - sent_duration
            
            # Hanya simpan checkpoint baru jika offset berubah signifikan (karena drop)
            if abs(offset - self._last_offset) > 0.001 or not self._checkpoints:
                self._checkpoints.append(TimelineCheckpoint(sent_duration, real_start))
                self._sent_positions.append(sent_duration)
                self._last_offset = offset

    def map_to_real(self, sent_time: float) -> float:
        """
        Petakan waktu dari server kembali ke waktu asli.
        """
        with self._lock:
            if not self._checkpoints:
                return sent_time

            idx = bisect.bisect_right(self._sent_positions, sent_time)

            if idx == 0:
                cp = self._checkpoints[0]
            else:
                cp = self._checkpoints[idx - 1]
                
            offset = cp.real_position - cp.sent_position
            return sent_time + offset

    def reset(self) -> None:
        """Reset timeline, biasa digunakan saat reconnect karena server reset counter ke 0."""
        with self._lock:
            self._checkpoints.clear()
            self._sent_positions.clear()
            self._last_offset = 0.0
