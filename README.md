# Local Shortform Transcriber

A lightweight, localhost-only Mac app for turning public short-form video URLs
into verbatim transcripts. Transcription runs locally with FFmpeg and
faster-whisper—no paid transcription API, accounts, database, analytics, browser
cookies, or API keys.

## What it does

1. Paste a public HTTPS TikTok, Instagram, YouTube, or X URL.
2. Click **Get transcript**.
3. Follow real extraction, conversion, and transcription progress.
4. Receive the verbatim transcript with duration, word count, and WPM.
5. Optionally copy the timestamped transcript with a compact Pareto-analysis
   prompt for ChatGPT.

The app is optimized for talking-head, interview, and presentation-style speech.
Songs, laughter, overlapping voices, and heavily processed audio may not be fully
transcribed.

## Platform support

| Platform | Extraction path | Verified status |
| --- | --- | --- |
| TikTok | yt-dlp, then TikWM public-link fallback | End-to-end tested |
| Instagram | yt-dlp | End-to-end tested on a public Reel |
| YouTube | yt-dlp with a cookie-free compatibility fallback | Public audio extraction tested |
| X / Twitter | yt-dlp | End-to-end tested on a public video |

Platform websites change frequently, so a public URL that works today can later
require a newer yt-dlp release. Private, age-restricted, login-only, or otherwise
restricted media is intentionally unsupported.

## Requirements

- macOS
- Python 3.10 or newer (Python 3.13 is preferred when available)
- FFmpeg
- Internet access for public-media retrieval, dependency installation, and the
  first faster-whisper model download

The default accuracy-first configuration uses the faster-whisper `small` model on
CPU with int8 computation and beam size 5.

## Run locally

1. Install current Python and FFmpeg yourself. With Homebrew:

   ```bash
   brew install python@3.13 ffmpeg
   ```

2. Double-click `START.command`.
3. Approve creation of the isolated `.venv` on first run.
4. Keep the Terminal window open and use `http://127.0.0.1:8000`.

Do not open `app/static/index.html` directly; the interface requires the local
Python server.

## Tests

After setup:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The regression suite covers URL restrictions, credential rejection, oversized
upload cleanup, and full-media duration/WPM metrics.

## Privacy and security

- Binds only to `127.0.0.1`.
- Accepts only approved HTTPS social-media domains.
- Does not request browser cookies, profiles, credentials, or API keys.
- Caps uploads at 250 MB and processing at 15 minutes.
- Deletes temporary uploads after processing, including oversized partial files.
- Sends a public TikTok URL to TikWM only when direct TikTok extraction fails.
- Makes no automatic LLM request; the analysis action only writes to the local
  clipboard.

See [SECURITY.md](SECURITY.md) for the detailed security model and residual risks.

## Cost

There is no paid transcription or analysis API. Compute runs on the user's Mac.

## Scope

This is a deliberately focused local MVP. It does not include accounts, history,
payments, analytics, cloud hosting, automatic analysis, or private-media access.
