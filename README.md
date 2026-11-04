# Local Shortform Transcriber

A lightweight, localhost-only Mac app for turning public short-form video URLs
into verbatim transcripts, optimized for English speech. Transcription runs
locally with FFmpeg and Whisper—no paid transcription API, accounts, database, analytics, browser
cookies, or API keys.

## What it does

1. Paste a public HTTPS TikTok, Instagram, YouTube, X, or Xiaohongshu URL.
2. Click **Transcribe**.
3. Follow extraction, model preparation, and transcription progress, with text
   appearing as each segment finishes. This works for URLs and uploads.
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
| Xiaohongshu / RedNote | yt-dlp with a RedNote-domain compatibility adapter | Beta: `xiaohongshu.com`, `rednote.com`, and `xhslink.com` URLs accepted; anti-bot/CAPTCHA may block extraction |

Platform websites change frequently, so a public URL that works today can later
require a newer yt-dlp release. Private, age-restricted, login-only, or otherwise
restricted media is intentionally unsupported. Xiaohongshu is especially prone
to rejecting cookie-free automated requests; use the saved-video upload fallback
when a public URL is blocked.

## Requirements

- macOS
- Python 3.10 or newer (Python 3.13 is preferred when available)
- FFmpeg
- Internet access for public-media retrieval, dependency installation, and the
  first faster-whisper model download

The app uses the multilingual `small` Whisper model and specifies English by
default. Compatible Apple Silicon Macs automatically use GPU acceleration.
Other devices, existing installations without the GPU binding, and requests
for other languages use the standard CPU engine. See
[performance notes](docs/PERFORMANCE.md) for the measured speed and accuracy.

## Run locally

1. Install current Python and FFmpeg yourself. With Homebrew:

   ```bash
   brew install python@3.13 ffmpeg
   ```

2. Double-click `START.command`.
3. Approve creation of the isolated `.venv` on first run.
4. Keep the Terminal window open and use `http://127.0.0.1:8000`.

For convenient repeat use on macOS, create a Finder alias (right-click
`START.command` → **Make Alias**) and place that alias on your Desktop or Dock.
The alias is only a local shortcut; it is not part of the repository.

Do not open `app/static/index.html` directly; the interface requires the local
Python server.

## Tests

After setup:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The regression suite covers URL restrictions, credential rejection, upload
cleanup, streaming results and failures, full-media duration/WPM metrics, and
Mac acceleration's speech filtering and timestamp restoration.

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
