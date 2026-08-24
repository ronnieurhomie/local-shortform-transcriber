from __future__ import annotations
import os
import threading

import numpy as np
from faster_whisper import WhisperModel

class LocalTranscriber:
    def __init__(self):
        model_name = os.getenv("WHISPER_MODEL", "small")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> dict:
        segments, info = self.model.transcribe(
            audio,
            language=language or None,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True,
        )

        rows = []
        parts = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            rows.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": text,
            })
            parts.append(text)

        text = " ".join(parts).strip()
        words = len(text.split())
        media_duration = len(audio) / 16000
        wpm = round(words / (media_duration / 60), 1) if media_duration else 0

        return {
            "text": text,
            "segments": rows,
            "language": getattr(info, "language", None),
            "duration_seconds": round(media_duration, 1),
            "word_count": words,
            "words_per_minute": wpm,
        }

_instance = None
_lock = threading.Lock()

def get_transcriber():
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = LocalTranscriber()
    return _instance
