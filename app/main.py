from __future__ import annotations
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

from .extractor import ExtractionError, audio_from_file, audio_from_url
from .transcriber import get_transcriber

APP_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 250 * 1024 * 1024  # 250 MB

app = FastAPI(
    title="Local Shortform Transcriber",
    version="0.6.1",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Protect the localhost app against DNS-rebinding / unexpected Host headers.
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost"],
)

class URLRequest(BaseModel):
    url: str
    language: Optional[str] = "en"

@app.get("/")
def index():
    return FileResponse(
        APP_DIR / "static" / "index.html",
        headers={"Cache-Control": "no-store"},
    )

@app.get("/api/health")
def health():
    return {"ok": True, "model": os.getenv("WHISPER_MODEL", "small")}

@app.post("/api/transcribe-url")
async def transcribe_url(req: URLRequest):
    def run():
        audio = audio_from_url(req.url)
        return _transcribe_audio(audio, req.language)

    try:
        return await asyncio.shield(asyncio.create_task(asyncio.to_thread(run)))
    except ExtractionError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")

def _transcribe_audio(audio, language, progress=None, on_segment=None):
    if progress:
        progress("Preparing transcription model")
    transcriber = get_transcriber(language)
    if progress:
        progress("Transcribing locally")
    return transcriber.transcribe(audio, language, on_segment=on_segment)


def _stream_transcription(load_audio, language, cleanup=None):
    """Run extraction and inference off the event loop; stream completed segments."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def publish(event):
        event["at"] = time.monotonic()
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def publish_phase(name):
        publish({"type": "phase", "name": name})

    def run():
        try:
            audio = load_audio(publish_phase)
            publish({"type": "audio", "duration_seconds": round(len(audio) / 16000, 1)})
            result = _transcribe_audio(
                audio,
                language,
                progress=publish_phase,
                on_segment=lambda row: publish({"type": "segment", "data": row}),
            )
            publish({"type": "result", "data": result})
        except ExtractionError as e:
            publish({"type": "error", "message": str(e)})
        except Exception as e:
            publish({"type": "error", "message": f"Transcription failed: {e}"})
        finally:
            if cleanup:
                cleanup()

    # Start the independent worker before returning the response. Its finally
    # owns uploaded-file cleanup even if the response is disconnected/cancelled.
    task = asyncio.create_task(asyncio.to_thread(run))

    async def events():
        phase_name = None
        phase_started = None
        timings = {}
        try:
            while True:
                event = await queue.get()
                now = event["at"]
                if event["type"] in {"phase", "result", "error"} and phase_name is not None:
                    timings[phase_name] = timings.get(phase_name, 0) + now - phase_started
                if event["type"] == "phase":
                    phase_name = event["name"]
                    phase_started = now

                if event["type"] == "result":
                    event["data"]["processing_times"] = {
                        name: round(value, 1) for name, value in timings.items()
                    }
                yield json.dumps(event) + "\n"
                if event["type"] in {"result", "error"}:
                    break
        finally:
            await asyncio.shield(task)

    return StreamingResponse(
        events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"}
    )


@app.post("/api/transcribe-url-stream")
async def transcribe_url_stream(req: URLRequest):
    return _stream_transcription(
        lambda progress: audio_from_url(req.url, progress=progress), req.language
    )


async def _save_upload(file):
    suffix = Path(file.filename or "upload.mp4").suffix.lower() or ".mp4"
    allowed = {".mp4", ".mov", ".m4v", ".webm", ".mp3", ".m4a", ".wav", ".aac", ".ogg"}
    if suffix not in allowed:
        await file.close()
        raise HTTPException(400, "Unsupported file type.")

    total = 0
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            path = Path(tmp.name)
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File is larger than the 250 MB local safety limit.")
                tmp.write(chunk)
        return path
    except BaseException:
        if path is not None:
            path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.post("/api/transcribe-upload")
async def transcribe_upload(
    file: UploadFile = File(...),
    language: Optional[str] = Form("en"),
):
    path = await _save_upload(file)

    def run():
        try:
            return _transcribe_audio(audio_from_file(path), language)
        finally:
            path.unlink(missing_ok=True)

    try:
        return await asyncio.shield(asyncio.create_task(asyncio.to_thread(run)))
    except ExtractionError as e:
        raise HTTPException(422, str(e))


@app.post("/api/transcribe-upload-stream")
async def transcribe_upload_stream(
    file: UploadFile = File(...),
    language: Optional[str] = Form("en"),
):
    path = await _save_upload(file)

    def load_audio(progress):
        progress("Converting the audio")
        return audio_from_file(path)

    return _stream_transcription(
        load_audio, language, cleanup=lambda: path.unlink(missing_ok=True)
    )
