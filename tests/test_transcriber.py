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
    def start_patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def setUp(self):
        self.start_patch(patch.dict(os.environ, {}, clear=True))
        self.start_patch(patch.object(engines, "_instances", {}))

    def apple_silicon(self):
        self.start_patch(patch.object(engines.platform, "system", return_value="Darwin"))
        self.start_patch(patch.object(engines.platform, "machine", return_value="arm64"))
        return self.start_patch(patch.object(engines.importlib.util, "find_spec", return_value=object()))

    def test_english_uses_mac_gpu_and_reuses_each_engine_across_requests(self):
        self.apple_silicon()
        accelerated, standard = object(), object()
        with (
            patch("app.metal_transcriber.MetalTranscriber", return_value=accelerated) as metal_load,
            patch.object(engines, "LocalTranscriber", return_value=standard) as cpu_load,
        ):
            self.assertIs(engines.get_transcriber("en"), accelerated)
            self.assertIs(engines.get_transcriber("en"), accelerated)
            self.assertIs(engines.get_transcriber("zh"), standard)
            self.assertIs(engines.get_transcriber(None), standard)
        metal_load.assert_called_once_with()
        cpu_load.assert_called_once_with()

    def test_other_hardware_and_languages_use_standard_engine(self):
        for system, machine, language in (
            ("Darwin", "x86_64", "en"),
            ("Linux", "x86_64", "en"),
            ("Windows", "AMD64", "en"),
            ("Darwin", "arm64", "zh"),
            ("Darwin", "arm64", None),
        ):
            with (
                self.subTest(system=system, machine=machine, language=language),
                patch.object(engines, "_instances", {}),
                patch.object(engines.platform, "system", return_value=system),
                patch.object(engines.platform, "machine", return_value=machine),
                patch.object(engines.importlib.util, "find_spec", return_value=object()),
                patch.object(engines, "LocalTranscriber", return_value=object()) as cpu_load,
                patch("app.metal_transcriber.MetalTranscriber") as metal_load,
            ):
                self.assertIs(engines.get_transcriber(language), cpu_load.return_value)
                metal_load.assert_not_called()
                cpu_load.assert_called_once_with()

    def test_missing_native_binding_keeps_existing_install_usable(self):
        binding = self.apple_silicon()
        binding.return_value = None
        with (
            patch.object(engines, "LocalTranscriber", return_value=object()) as cpu_load,
            patch("app.metal_transcriber.MetalTranscriber") as metal_load,
        ):
            self.assertIs(engines.get_transcriber("en"), cpu_load.return_value)
        cpu_load.assert_called_once_with()
        metal_load.assert_not_called()

    def test_gpu_initialization_failure_reuses_standard_engine_and_does_not_retry(self):
        self.apple_silicon()
        with (
            patch.object(engines, "LocalTranscriber", return_value=object()) as cpu_load,
            patch("app.metal_transcriber.MetalTranscriber", side_effect=RuntimeError("No GPU")) as metal_load,
        ):
            existing = engines.get_transcriber(None)
            with self.assertLogs(engines.__name__, level="WARNING") as logs:
                self.assertIs(engines.get_transcriber("en"), existing)
            self.assertIs(engines.get_transcriber("en"), existing)
            self.assertIs(engines.get_transcriber(None), existing)
        self.assertIn("using the standard engine", logs.output[0])
        cpu_load.assert_called_once_with()
        metal_load.assert_called_once_with()

    def test_gpu_failure_on_first_request_constructs_a_reusable_standard_fallback(self):
        self.apple_silicon()
        with (
            patch.object(engines, "LocalTranscriber", return_value=object()) as cpu_load,
            patch("app.metal_transcriber.MetalTranscriber", side_effect=RuntimeError("Download unavailable")),
            self.assertLogs(engines.__name__, level="WARNING"),
        ):
            self.assertIs(engines.get_transcriber("en"), cpu_load.return_value)
            self.assertIs(engines.get_transcriber(None), cpu_load.return_value)
        cpu_load.assert_called_once_with()

    def test_explicit_standard_override_is_respected_on_compatible_mac(self):
        self.apple_silicon()
        with (
            patch.dict(os.environ, {"WHISPER_ENGINE": "faster-whisper"}),
            patch.object(engines, "LocalTranscriber", return_value=object()) as cpu_load,
            patch("app.metal_transcriber.MetalTranscriber") as metal_load,
        ):
            self.assertIs(engines.get_transcriber("en"), cpu_load.return_value)
        metal_load.assert_not_called()

    def test_explicit_metal_override_does_not_silently_hide_initialization_failure(self):
        with (
            patch.dict(os.environ, {"WHISPER_ENGINE": "metal"}),
            patch("app.metal_transcriber.MetalTranscriber", side_effect=RuntimeError("Native engine unavailable")),
            patch.object(engines, "LocalTranscriber") as standard,
        ):
            with self.assertRaisesRegex(RuntimeError, "Native engine unavailable"):
                engines.get_transcriber("en")
        standard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
