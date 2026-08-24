import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException, UploadFile

from app.extractor import ExtractionError, _validate_social_url
from app.main import transcribe_upload
from app.transcriber import LocalTranscriber


class URLValidationTests(unittest.TestCase):
    def test_accepts_https_tiktok_and_instagram(self):
        self.assertEqual(
            _validate_social_url("https://www.tiktok.com/@creator/video/123")[1],
            "tiktok",
        )
        self.assertEqual(
            _validate_social_url("https://www.instagram.com/reel/example/")[1],
            "instagram",
        )
        self.assertEqual(_validate_social_url("https://youtu.be/example")[1], "youtube")
        self.assertEqual(_validate_social_url("https://x.com/user/status/123")[1], "x")

    def test_rejects_non_https(self):
        with self.assertRaisesRegex(ExtractionError, "Only HTTPS"):
            _validate_social_url("http://www.tiktok.com/@creator/video/123")

    def test_rejects_unapproved_host_and_lookalike(self):
        for url in (
            "https://example.com/video",
            "https://tiktok.com.evil.example/video",
        ):
            with self.subTest(url=url), self.assertRaisesRegex(
                ExtractionError, "only public TikTok, Instagram, YouTube, or X"
            ):
                _validate_social_url(url)

    def test_rejects_credentials(self):
        with self.assertRaisesRegex(ExtractionError, "credentials"):
            _validate_social_url("https://user:pass@www.instagram.com/reel/example/")


class UploadCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_oversized_partial_upload_is_deleted(self):
        created = []
        real_named_temporary_file = tempfile.NamedTemporaryFile

        def tracked_tempfile(*args, **kwargs):
            tmp = real_named_temporary_file(*args, **kwargs)
            created.append(Path(tmp.name))
            return tmp

        upload = UploadFile(filename="clip.mp4", file=io.BytesIO(b"123456"))
        with (
            patch("app.main.MAX_UPLOAD_BYTES", 5),
            patch("app.main.tempfile.NamedTemporaryFile", side_effect=tracked_tempfile),
        ):
            with self.assertRaises(HTTPException) as raised:
                await transcribe_upload(upload)

        self.assertEqual(raised.exception.status_code, 413)
        self.assertTrue(created)
        self.assertFalse(created[0].exists())


class TranscriptMetricTests(unittest.TestCase):
    def test_duration_and_wpm_use_full_media_not_last_spoken_word(self):
        transcriber = LocalTranscriber.__new__(LocalTranscriber)
        transcriber.model = SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: (
                iter([SimpleNamespace(start=0.0, end=5.0, text="ten words are counted across this short spoken sample now")]),
                SimpleNamespace(language="en"),
            )
        )
        audio = np.zeros(16000 * 10, dtype=np.float32)

        result = transcriber.transcribe(audio)

        self.assertEqual(result["duration_seconds"], 10.0)
        self.assertEqual(result["word_count"], 10)
        self.assertEqual(result["words_per_minute"], 60.0)


if __name__ == "__main__":
    unittest.main()
