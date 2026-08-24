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
    version="0.4.0",
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
    language: Optional[str] = None

@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")

@app.get("/api/health")
def health():
    return {"ok": True, "model": os.getenv("WHISPER_MODEL", "small")}

@app.post("/api/transcribe-url")
async def transcribe_url(req: URLRequest):
    try:
        audio = await asyncio.to_thread(audio_from_url, req.url)
        return await asyncio.to_thread(get_transcriber().transcribe, audio, req.language)
    except ExtractionError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")

@app.post("/api/transcribe-url-stream")
async def transcribe_url_stream(req: URLRequest):
    """Stream truthful processing phases, followed by the normal transcript result."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    async def events():
        phase_name = None
        phase_started = time.monotonic()
        timings = {}

        def publish_phase(name: str):
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "phase", "name": name})

        def run():
            try:
                audio = audio_from_url(req.url, progress=publish_phase)
                publish_phase("Transcribing locally")
                result = get_transcriber().transcribe(audio, req.language)
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "result", "data": result})
            except ExtractionError as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": f"Transcription failed: {e}"})

        task = asyncio.create_task(asyncio.to_thread(run))
        try:
            while True:
                event = await queue.get()
                now = time.monotonic()
                if event["type"] == "phase":
                    if phase_name is not None:
                        timings[phase_name] = round(now - phase_started, 1)
                    phase_name = event["name"]
                    phase_started = now
                elif phase_name is not None:
                    timings[phase_name] = round(now - phase_started, 1)

                if event["type"] == "result":
                    event["data"]["processing_times"] = timings
                yield json.dumps(event) + "\n"
                if event["type"] in {"result", "error"}:
                    break
        finally:
            await task

    return StreamingResponse(events(), media_type="application/x-ndjson")

@app.post("/api/transcribe-upload")
async def transcribe_upload(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    suffix = Path(file.filename or "upload.mp4").suffix.lower() or ".mp4"
    allowed = {".mp4", ".mov", ".m4v", ".webm", ".mp3", ".m4a", ".wav", ".aac", ".ogg"}
    if suffix not in allowed:
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
        audio = await asyncio.to_thread(audio_from_file, path)
        return await asyncio.to_thread(get_transcriber().transcribe, audio, language)
    except ExtractionError as e:
        raise HTTPException(422, str(e))
    finally:
        await file.close()
        if path is not None:
            path.unlink(missing_ok=True)
