import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app import metal_transcriber as metal


class MetalTranscriberTests(unittest.TestCase):
    def make_transcriber(self, decode):
        transcriber = metal.MetalTranscriber.__new__(metal.MetalTranscriber)
        transcriber.model = SimpleNamespace(transcribe=decode, _ctx="test-context")
        transcriber._lock = threading.Lock()
        return transcriber

    def native_language(self, language="en"):
        return SimpleNamespace(
            whisper_full_lang_id=Mock(return_value=0),
            whisper_lang_str=Mock(return_value=language),
        )

    def test_silence_mapping_restores_segment_boundaries_and_full_media_metrics(self):
        audio = np.arange(16000 * 12, dtype=np.float32)
        chunks = [
            {"start": 16000, "end": 48000},
            {"start": 112000, "end": 144000},
        ]
        events = []
        native = self.native_language()

        def decode(speech, **kwargs):
            np.testing.assert_array_equal(speech, np.concatenate([audio[16000:48000], audio[112000:144000]]))
            self.assertTrue(transcriber._lock.locked())
            kwargs["new_segment_callback"](SimpleNamespace(t0=0, t1=200, text=" First phrase. "))
            events.append("between-segments")
            kwargs["new_segment_callback"](SimpleNamespace(t0=200, t1=400, text="Second phrase."))
            events.append("decode-finished")

        transcriber = self.make_transcriber(decode)
        with (
            patch.object(metal, "get_speech_timestamps", return_value=chunks) as vad,
            patch.dict(sys.modules, {"_pywhispercpp": native}),
        ):
            result = transcriber.transcribe(audio, on_segment=lambda row: events.append(row.copy()))
        events.append("result-returned")

        self.assertEqual(vad.call_args.args[1], metal.VadOptions())
        self.assertEqual(result["segments"], [
            {"start": 1.0, "end": 3.0, "text": "First phrase."},
            {"start": 7.0, "end": 9.0, "text": "Second phrase."},
        ])
        self.assertEqual(result["text"], "First phrase. Second phrase.")
        self.assertEqual(result["duration_seconds"], 12.0)
        self.assertEqual(result["word_count"], 4)
        self.assertEqual(result["words_per_minute"], 20.0)
        self.assertEqual(result["language"], "en")
        self.assertEqual(events, [
            result["segments"][0], "between-segments", result["segments"][1],
            "decode-finished", "result-returned",
        ])
        native.whisper_full_lang_id.assert_called_once_with("test-context")

    def test_no_speech_skips_native_decode_and_preserves_duration(self):
        decode = Mock(side_effect=AssertionError("Empty speech must not be decoded"))
        callback = Mock()
        transcriber = self.make_transcriber(decode)
        audio = np.zeros(16000 * 8, dtype=np.float32)
        with patch.object(metal, "get_speech_timestamps", return_value=[]):
            result = transcriber.transcribe(audio, language="zh", on_segment=callback)

        decode.assert_not_called()
        callback.assert_not_called()
        self.assertEqual(result["text"], "")
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["duration_seconds"], 8.0)
        self.assertEqual(result["language"], "zh")
        self.assertEqual(result["words_per_minute"], 0)

    def test_every_job_clears_previous_decoder_context_and_passes_language(self):
        decode = Mock()
        transcriber = self.make_transcriber(decode)
        audio = np.zeros(32000, dtype=np.float32)
        native = self.native_language("zh")
        with (
            patch.object(metal, "get_speech_timestamps", return_value=[{"start": 0, "end": len(audio)}]),
            patch.dict(sys.modules, {"_pywhispercpp": native}),
        ):
            transcriber.transcribe(audio, language="zh")
            transcriber.transcribe(audio)

        self.assertEqual(decode.call_count, 2)
        for call in decode.call_args_list:
            self.assertIs(call.kwargs["no_context"], True)
            self.assertTrue(callable(call.kwargs["new_segment_callback"]))
        self.assertEqual(decode.call_args_list[0].kwargs["language"], "zh")
        self.assertEqual(decode.call_args_list[1].kwargs["language"], "")

    def test_non_apple_silicon_is_rejected_before_model_download(self):
        for system, machine in (("Linux", "arm64"), ("Darwin", "x86_64")):
            with (
                self.subTest(system=system, machine=machine),
                patch.object(metal.platform, "system", return_value=system),
                patch.object(metal.platform, "machine", return_value=machine),
                patch.object(metal, "hf_hub_download") as download,
            ):
                with self.assertRaisesRegex(RuntimeError, "Apple Silicon"):
                    metal.MetalTranscriber()
                download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
