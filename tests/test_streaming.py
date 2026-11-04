import asyncio
import io
import itertools
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException, UploadFile

from app import main


ROW = {"start": 0.0, "end": 1.0, "text": "A completed sentence."}


def result():
    return {
        "text": ROW["text"],
        "segments": [ROW.copy()],
        "language": "en",
        "duration_seconds": 2.0,
        "word_count": 3,
        "words_per_minute": 90.0,
    }


def transcribe(_audio, _language, on_segment=None):
    if on_segment:
        on_segment(ROW.copy())
    return result()


async def collect(response):
    return [json.loads(line) async for line in response.body_iterator]


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.audio = np.zeros(32000, dtype=np.float32)
        self.model = SimpleNamespace(transcribe=transcribe)

    async def test_url_delivers_completed_segment_before_transcription_finishes(self):
        release = threading.Event()
        finished = threading.Event()

        def blocked_transcribe(audio, language, on_segment=None):
            on_segment(ROW.copy())
            if not release.wait(3):
                raise RuntimeError("Test worker timed out")
            finished.set()
            return result()

        with (
            patch.object(main, "audio_from_url", return_value=self.audio),
            patch.object(main, "get_transcriber", return_value=SimpleNamespace(transcribe=blocked_transcribe)),
        ):
            response = await main.transcribe_url_stream(main.URLRequest(url="https://youtu.be/example"))
            iterator = response.body_iterator
            before = []
            try:
                while not before or before[-1]["type"] != "segment":
                    before.append(json.loads(await asyncio.wait_for(anext(iterator), 2)))
                self.assertFalse(finished.is_set())
                self.assertEqual(before[-1]["data"], ROW)
                self.assertEqual(before[0]["type"], "audio")
                self.assertEqual(before[0]["duration_seconds"], 2.0)
                release.set()
                after = [json.loads(line) async for line in iterator]
            finally:
                release.set()
                await iterator.aclose()

        self.assertEqual(after[-1]["type"], "result")
        final = after[-1]["data"]
        for key, expected in result().items():
            self.assertEqual(final[key], expected)
        self.assertIn("Preparing transcription model", final["processing_times"])
        self.assertIn("Transcribing locally", final["processing_times"])

    async def test_upload_stream_returns_result_and_deletes_temporary_media(self):
        paths = []

        def load(path):
            paths.append(path)
            self.assertEqual(path.read_bytes(), b"test-media")
            return self.audio

        upload = UploadFile(filename="screen.mov", file=io.BytesIO(b"test-media"))
        with (
            patch.object(main, "audio_from_file", side_effect=load),
            patch.object(main, "get_transcriber", return_value=self.model),
        ):
            events = await collect(await main.transcribe_upload_stream(upload, None))

        self.assertEqual(events[0]["name"], "Converting the audio")
        self.assertIn("segment", [event["type"] for event in events])
        self.assertEqual(events[-1]["type"], "result")
        self.assertTrue(upload.file.closed)
        self.assertFalse(paths[0].exists())

    async def test_failure_after_partial_text_emits_error_and_cleans_upload(self):
        paths = []

        def load(path):
            paths.append(path)
            return self.audio

        def fail(_audio, _language, on_segment=None):
            on_segment(ROW.copy())
            raise RuntimeError("inference stopped")

        upload = UploadFile(filename="clip.mp4", file=io.BytesIO(b"media"))
        with (
            patch.object(main, "audio_from_file", side_effect=load),
            patch.object(main, "get_transcriber", return_value=SimpleNamespace(transcribe=fail)),
        ):
            events = await collect(await main.transcribe_upload_stream(upload, None))

        self.assertEqual(events[-2]["type"], "segment")
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("inference stopped", events[-1]["message"])
        self.assertNotIn("result", [event["type"] for event in events])
        self.assertFalse(paths[0].exists())

    async def test_disconnected_upload_keeps_media_until_worker_finishes_then_deletes_it(self):
        release = threading.Event()
        paths = []

        def load(path):
            paths.append(path)
            return self.audio

        def blocked(_audio, _language, on_segment=None):
            on_segment(ROW.copy())
            if not release.wait(3):
                raise RuntimeError("Test worker timed out")
            self.assertTrue(paths[0].exists())
            return result()

        upload = UploadFile(filename="clip.mp4", file=io.BytesIO(b"media"))
        with (
            patch.object(main, "audio_from_file", side_effect=load),
            patch.object(main, "get_transcriber", return_value=SimpleNamespace(transcribe=blocked)),
        ):
            response = await main.transcribe_upload_stream(upload, None)
            iterator = response.body_iterator
            try:
                while True:
                    event = json.loads(await asyncio.wait_for(anext(iterator), 2))
                    if event["type"] == "segment":
                        break
                self.assertTrue(paths[0].exists())
                closing = asyncio.create_task(iterator.aclose())
                await asyncio.sleep(0)
                release.set()
                await asyncio.wait_for(closing, 2)
            finally:
                release.set()
                await iterator.aclose()

        self.assertFalse(paths[0].exists())

    async def test_oversized_stream_upload_is_rejected_and_partial_file_deleted(self):
        paths = []
        original = tempfile.NamedTemporaryFile

        def tracked(*args, **kwargs):
            tmp = original(*args, **kwargs)
            paths.append(Path(tmp.name))
            return tmp

        upload = UploadFile(filename="clip.mp4", file=io.BytesIO(b"123456"))
        with (
            patch.object(main, "MAX_UPLOAD_BYTES", 5),
            patch.object(main.tempfile, "NamedTemporaryFile", side_effect=tracked),
        ):
            with self.assertRaises(HTTPException) as caught:
                await main.transcribe_upload_stream(upload, None)

        self.assertEqual(caught.exception.status_code, 413)
        self.assertTrue(upload.file.closed)
        self.assertFalse(paths[0].exists())

    async def test_json_endpoints_keep_result_schema_and_load_model_on_worker(self):
        loop_thread = threading.get_ident()
        model_threads = []
        paths = []

        def get_model():
            model_threads.append(threading.get_ident())
            return self.model

        def load_file(path):
            paths.append(path)
            return self.audio

        with (
            patch.object(main, "audio_from_url", return_value=self.audio),
            patch.object(main, "audio_from_file", side_effect=load_file),
            patch.object(main, "get_transcriber", side_effect=get_model),
        ):
            url_result = await main.transcribe_url(main.URLRequest(url="https://youtu.be/example"))
            upload = UploadFile(filename="clip.wav", file=io.BytesIO(b"media"))
            upload_result = await main.transcribe_upload(upload, None)

        self.assertEqual(url_result, result())
        self.assertEqual(upload_result, result())
        self.assertEqual(len(model_threads), 2)
        self.assertTrue(all(thread != loop_thread for thread in model_threads))
        self.assertFalse(paths[0].exists())

    async def test_timings_use_worker_clock_and_accumulate_repeated_phases(self):
        ticks = itertools.count(10)

        def load(progress):
            progress("Resolving")  # 10
            progress("Converting")  # 11
            progress("Fallback")  # 12
            progress("Converting")  # 13
            return self.audio  # Audio event: 14

        with (
            patch.object(main, "time", SimpleNamespace(monotonic=lambda: next(ticks))),
            patch.object(main, "get_transcriber", return_value=self.model),
        ):
            events = await collect(main._stream_transcription(load, None))

        self.assertEqual(
            events[-1]["data"]["processing_times"],
            {
                "Resolving": 1,
                "Converting": 3,
                "Fallback": 1,
                "Preparing transcription model": 1,
                "Transcribing locally": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
