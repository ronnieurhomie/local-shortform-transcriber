import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.transcriber import LocalTranscriber
from app import transcriber as engines


class TranscriberCallbackTests(unittest.TestCase):
    def test_completed_segments_arrive_while_decoding_and_final_result_is_unchanged(self):
        events = []

        def decoded_segments():
            yield SimpleNamespace(start=1.234, end=2.345, text=" First phrase. ")
            events.append("decode-next")
            yield SimpleNamespace(start=3.0, end=4.0, text="  ")
            yield SimpleNamespace(start=7.89, end=9.99, text="Second phrase.")
            events.append("decode-done")

        decode = Mock(side_effect=lambda *_args, **_kwargs: (decoded_segments(), SimpleNamespace(language="en")))
        transcriber = LocalTranscriber.__new__(LocalTranscriber)
        transcriber.model = SimpleNamespace(transcribe=decode)
        audio = np.zeros(16000 * 12, dtype=np.float32)

        streamed = transcriber.transcribe(audio, on_segment=lambda row: events.append(row.copy()))
        self.assertEqual(events, [
            {"start": 1.23, "end": 2.35, "text": "First phrase."},
            "decode-next",
            {"start": 7.89, "end": 9.99, "text": "Second phrase."},
            "decode-done",
        ])
        self.assertEqual(streamed, transcriber.transcribe(audio))
        self.assertEqual(streamed["duration_seconds"], 12.0)
        self.assertEqual(streamed["word_count"], 4)
        self.assertEqual(streamed["words_per_minute"], 20.0)
        for call in decode.call_args_list:
            self.assertEqual(call.kwargs["beam_size"], 5)
            self.assertIs(call.kwargs["vad_filter"], True)
            self.assertIs(call.kwargs["condition_on_previous_text"], True)


class EngineSelectionTests(unittest.TestCase):
    def test_standard_engine_remains_default_and_is_reused(self):
        model = object()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(engines, "_instance", None),
            patch.object(engines, "LocalTranscriber", return_value=model) as load,
        ):
            self.assertIs(engines.get_transcriber(), model)
            self.assertIs(engines.get_transcriber(), model)
        load.assert_called_once_with()

    def test_metal_engine_requires_explicit_selection(self):
        model = object()
        with (
            patch.dict(os.environ, {"WHISPER_ENGINE": "metal"}),
            patch.object(engines, "_instance", None),
            patch("app.metal_transcriber.MetalTranscriber", return_value=model) as load,
            patch.object(engines, "LocalTranscriber") as standard,
        ):
            self.assertIs(engines.get_transcriber(), model)
        load.assert_called_once_with()
        standard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
