"""Optional Apple Silicon acceleration using the same multilingual small model."""

import os
import platform
import threading

import numpy as np
from faster_whisper.vad import SpeechTimestampsMap, VadOptions, get_speech_timestamps
from huggingface_hub import hf_hub_download

from .transcriber import transcript_result


class MetalTranscriber:
    def __init__(self):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Mac acceleration requires an Apple Silicon Mac.")
        if os.getenv("WHISPER_MODEL", "small") != "small":
            raise RuntimeError("Mac acceleration currently supports the multilingual small model.")
        try:
            from pywhispercpp.model import Model
        except ImportError as error:
            raise RuntimeError("Install requirements-metal.txt to use Mac acceleration.") from error

        model_path = hf_hub_download(
            repo_id="ggerganov/whisper.cpp", filename="ggml-small.bin", token=False,
        )
        self.model = Model(
            model_path,
            params_sampling_strategy=1,
            beam_search={"beam_size": 5, "patience": 1.0},
            n_threads=4,
            context_params={"use_gpu": True, "flash_attn": True},
            print_progress=False,
            print_realtime=False,
            print_timestamps=False,
        )
        self._lock = threading.Lock()

    def transcribe(self, audio, language=None, on_segment=None):
        # Use precisely the same speech filter as the standard engine.
        chunks = get_speech_timestamps(audio, VadOptions())
        if not chunks:
            return transcript_result(audio, [], language)
        speech = np.concatenate([audio[chunk["start"]:chunk["end"]] for chunk in chunks])
        timestamps = SpeechTimestampsMap(chunks, 16000)
        rows = []
        duration = len(audio) / 16000

        def completed(segment):
            text = segment.text.strip()
            if not text:
                return
            row = {
                "start": min(duration, timestamps.get_original_time(segment.t0 / 100)),
                "end": min(duration, timestamps.get_original_time(segment.t1 / 100, is_end=True)),
                "text": text,
            }
            rows.append(row)
            if on_segment is not None:
                on_segment(row)

        # whisper.cpp retains decoder state between calls unless no_context is
        # set. Clear previous jobs while retaining context within this file.
        with self._lock:
            self.model.transcribe(
                speech, language=language or "", no_context=True,
                new_segment_callback=completed,
            )
            import _pywhispercpp as native
            detected = native.whisper_lang_str(native.whisper_full_lang_id(self.model._ctx))
        return transcript_result(audio, rows, detected)
