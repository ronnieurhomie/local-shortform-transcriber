# Security notes — hardened local MVP

This is a small local-only tool, not a public web service.

## Deliberate protections
- Server binds only to `127.0.0.1`; it is not exposed to your LAN or internet.
- Trusted Host middleware accepts only `localhost` and `127.0.0.1`.
- URL input is restricted to HTTPS URLs on TikTok, Instagram, YouTube, X, and official Xiaohongshu / RedNote domains.
- No shell execution with user-provided text.
- No browser cookies, login credentials, API keys, or browser profiles are read.
- URL transcription does not intentionally save the source video.
- Uploaded files are limited to 250 MB and are deleted after processing.
- Audio processing is capped at 15 minutes and has a timeout.
- API documentation endpoints are disabled.
- The startup script does not install system software automatically.
- Python dependencies are installed into this folder's isolated `.venv` only on first setup.

## External network access
The tool still needs internet access for:
1. TikTok, Instagram, YouTube, X, or Xiaohongshu media retrieval.
2. Installing Python dependencies during first setup.
3. Downloading the faster-whisper model on first transcription.

Those are the only expected network uses.

The optional **Copy for ChatGPT Analysis** action only writes the transcript and
fixed analysis instructions to the local clipboard. It does not make a network
request or send the transcript to an AI service automatically.

## Residual risk
No software that installs third-party packages or processes untrusted media can be guaranteed risk-free.
The main residual risks are:
- supply-chain compromise of a dependency downloaded from PyPI/model hosting;
- vulnerabilities in FFmpeg/yt-dlp when parsing malicious media;
- platform changes causing extraction failures.

For a personal local MVP, the mitigations above substantially reduce exposure, but they do not make the risk mathematically zero.


## V2 TikTok fallback
If direct TikTok extraction fails, V2 may send the public TikTok URL to `https://www.tikwm.com/api/` to obtain a playable public media URL.
It does NOT send TikTok credentials, cookies, session IDs, browser profiles, or local files to TikWM.
This is a third-party service, so its availability and privacy policy are outside this app's control. If you do not want a public URL sent to TikWM, do not use the URL route; use the local upload fallback instead.
