from __future__ import annotations
import ipaddress
import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse, urlencode
from urllib.request import Request, urlopen

import numpy as np
import yt_dlp

class ExtractionError(RuntimeError):
    pass

def _clean_error(error: Exception) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", str(error))

ALLOWED_SOCIAL_HOSTS = {
    "tiktok.com",
    "www.tiktok.com",
    "vt.tiktok.com",
    "vm.tiktok.com",
    "m.tiktok.com",
    "instagram.com",
    "www.instagram.com",
    "instagr.am",
    "www.instagr.am",
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "x.com",
    "www.x.com",
    "twitter.com",
    "www.twitter.com",
    "mobile.twitter.com",
}

MAX_AUDIO_SECONDS = 15 * 60

def _validate_social_url(url: str) -> tuple[str, str]:
    try:
        parsed = urlparse(url.strip())
    except Exception:
        raise ExtractionError("Invalid URL.")

    if parsed.scheme != "https":
        raise ExtractionError("Only HTTPS TikTok, Instagram, YouTube, or X URLs are allowed.")

    host = (parsed.hostname or "").lower().rstrip(".")
    if host not in ALLOWED_SOCIAL_HOSTS:
        raise ExtractionError("For safety, only public TikTok, Instagram, YouTube, or X URLs are accepted.")

    if parsed.username or parsed.password:
        raise ExtractionError("URLs containing credentials are not allowed.")

    if "tiktok" in host:
        platform = "tiktok"
    elif "instagram" in host or "instagr.am" in host:
        platform = "instagram"
    elif "youtube" in host or host == "youtu.be":
        platform = "youtube"
    else:
        platform = "x"
    return url.strip(), platform

def _validate_media_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        raise ExtractionError("Resolver returned an invalid media URL.")

    if parsed.scheme != "https" or not parsed.hostname:
        raise ExtractionError("Resolver returned a non-HTTPS media URL.")

    if parsed.username or parsed.password:
        raise ExtractionError("Resolver returned a media URL containing credentials.")

    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"}:
        raise ExtractionError("Resolver returned a local address, which was blocked.")

    # Block literal private/loopback/link-local IPs.
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ExtractionError("Resolver returned a private/local IP address, which was blocked.")
    except ValueError:
        pass

    return url

def _ffmpeg_to_audio(source: str, headers: dict[str, str] | None = None) -> np.ndarray:
    source = _validate_media_url(source) if source.startswith(("http://", "https://")) else source

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

    if headers:
        safe_headers = {}
        for key in ("User-Agent", "Referer", "Origin", "Cookie"):
            if key in headers and isinstance(headers[key], str):
                safe_headers[key] = headers[key]
        if safe_headers:
            header_blob = "".join(f"{k}: {v}\r\n" for k, v in safe_headers.items())
            cmd += ["-headers", header_blob]

    cmd += [
        "-i", source,
        "-t", str(MAX_AUDIO_SECONDS),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "s16le",
        "pipe:1",
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(os.getenv("FFMPEG_TIMEOUT", "180")),
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise ExtractionError("Audio processing exceeded the local safety timeout.")

    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise ExtractionError(f"Could not read the audio stream. {err[-500:]}")

    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size < 16000:
        raise ExtractionError("Less than one second of usable audio was returned.")

    max_samples = 16000 * MAX_AUDIO_SECONDS
    return audio[:max_samples]

def _pick_media_url(info: dict):
    if info.get("url"):
        return info["url"], info.get("http_headers") or {}

    candidates = info.get("requested_formats") or info.get("formats") or []
    for fmt in reversed(candidates):
        if fmt.get("url") and fmt.get("acodec") not in (None, "none"):
            return fmt["url"], fmt.get("http_headers") or {}

    raise ExtractionError("The page resolved, but no playable audio stream was found.")

def _yt_dlp_media(url: str, youtube_client: str | None = None) -> tuple[str, dict]:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "format": "bestaudio/best",
        "socket_timeout": 20,
        "retries": 1,
        "extractor_retries": 1,
    }
    if youtube_client:
        opts["extractor_args"] = {"youtube": {"player_client": [youtube_client]}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not isinstance(info, dict):
        raise ExtractionError("The URL did not resolve to a video.")
    return _pick_media_url(info)

def _tikwm_media(url: str) -> tuple[str, dict]:
    """
    Fallback for PUBLIC TikTok links only.
    Sends the public TikTok URL to TikWM; no cookies, credentials, or browser data.
    """
    body = urlencode({"url": url, "hd": "1"}).encode("utf-8")
    req = Request(
        "https://www.tikwm.com/api/",
        data=body,
        method="POST",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=20) as resp:
            raw = resp.read(2 * 1024 * 1024)
    except Exception as e:
        raise ExtractionError(f"TikWM fallback could not be reached: {str(e)[-250:]}")

    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        raise ExtractionError("TikWM fallback returned an unreadable response.")

    # Common TikWM response: {"code":0,"data":{"hdplay":"...","play":"..."}}
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        msg = payload.get("msg") if isinstance(payload, dict) else None
        raise ExtractionError(f"TikWM fallback returned no video data{': ' + str(msg) if msg else ''}.")

    candidates = [
        data.get("hdplay"),
        data.get("play"),
        data.get("wmplay"),
        data.get("download_url"),
        data.get("video"),
    ]
    media_url = next((x for x in candidates if isinstance(x, str) and x.startswith("https://")), None)
    if not media_url:
        raise ExtractionError("TikWM fallback returned no playable HTTPS video URL.")

    return _validate_media_url(media_url), {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.tikwm.com/",
    }

def audio_from_url(url: str, progress: Callable[[str], None] | None = None) -> np.ndarray:
    url, platform = _validate_social_url(url)
    errors = []

    if progress:
        progress("Resolving the public video")

    # Attempt 1: direct extraction with yt-dlp.
    try:
        media_url, headers = _yt_dlp_media(url)
        if progress:
            progress("Converting the audio")
        return _ffmpeg_to_audio(media_url, headers)
    except Exception as e:
        errors.append(f"Direct extractor: {_clean_error(e)[-220:]}")

    # Some public YouTube pages reject the default web client even though their
    # audio remains publicly available. Retry with yt-dlp's cookie-free Android
    # VR client before giving up; this does not read browser data or credentials.
    if platform == "youtube":
        try:
            if progress:
                progress("Trying a YouTube-compatible client")
            media_url, headers = _yt_dlp_media(url, youtube_client="android_vr")
            if progress:
                progress("Converting the audio")
            return _ffmpeg_to_audio(media_url, headers)
        except Exception as e:
            errors.append(f"YouTube fallback: {_clean_error(e)[-220:]}")

    # Attempt 2: TikTok-only public resolver fallback.
    if platform == "tiktok":
        try:
            if progress:
                progress("Trying the TikTok fallback")
            media_url, headers = _tikwm_media(url)
            if progress:
                progress("Converting the audio")
            return _ffmpeg_to_audio(media_url, headers)
        except Exception as e:
            errors.append(f"TikWM fallback: {_clean_error(e)[-220:]}")

    detail = " | ".join(errors)
    method_summary = "Direct extraction and the TikTok fallback failed" if platform == "tiktok" else "Direct extraction failed"
    raise ExtractionError(
        f"{method_summary}. The video may be unavailable, private, unsupported, or temporarily blocked. "
        "No login or browser cookies were used. "
        f"Details: {detail}"
    )

def audio_from_file(path: Path) -> np.ndarray:
    # Local file path is created by the app itself, not user-supplied URL text.
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-t", str(MAX_AUDIO_SECONDS),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "s16le", "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=int(os.getenv("FFMPEG_TIMEOUT", "180")), check=False
        )
    except subprocess.TimeoutExpired:
        raise ExtractionError("Audio processing exceeded the local safety timeout.")

    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise ExtractionError(f"Could not read the uploaded media. {err[-500:]}")

    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size < 16000:
        raise ExtractionError("Less than one second of usable audio was returned.")
    return audio[:16000 * MAX_AUDIO_SECONDS]
