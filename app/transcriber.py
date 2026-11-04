from __future__ import annotations
import os
import threading
from collections.abc import Callable

import numpy as np
from faster_whisper import WhisperModel


def transcript_result(audio: np.ndarray, rows: list[dict], language: str | None) -> dict:
    text = " ".join(row["text"] for row in rows).strip()
    words = len(text.split())
    media_duration = len(audio) / 16000
    return {
        "text": text,
        "segments": rows,
        "language": language,
        "duration_seconds": round(media_duration, 1),
        "word_count": words,
        "words_per_minute": round(words / (media_duration / 60), 1) if media_duration else 0,
    }


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

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        on_segment: Callable[[dict], None] | None = None,
    ) -> dict:
        segments, info = self.model.transcribe(
            audio,
            language=language or None,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True,
        )

        rows = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            row = {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": text,
            }
            rows.append(row)
            if on_segment is not None:
                on_segment(row)

        return transcript_result(audio, rows, getattr(info, "language", None))

_instance = None
_lock = threading.Lock()

def get_transcriber():
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                engine = os.getenv("WHISPER_ENGINE", "faster-whisper")
                if engine == "metal":
                    from .metal_transcriber import MetalTranscriber
                    _instance = MetalTranscriber()
                elif engine == "faster-whisper":
                    _instance = LocalTranscriber()
                else:
                    raise ValueError("WHISPER_ENGINE must be faster-whisper or metal.")
    return _instance
