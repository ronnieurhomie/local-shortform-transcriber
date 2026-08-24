#!/bin/bash
set -e
cd "$(dirname "$0")"

echo ""
echo "Local Shortform Transcriber — hardened local build"
echo "=========================================="
echo ""
echo "This app binds ONLY to 127.0.0.1 (your Mac)."
echo "It accepts ONLY TikTok, Instagram, YouTube, and X HTTPS URLs."
echo "It does not ask for passwords, cookies, browser profiles, or API keys."
echo ""

if command -v python3.13 >/dev/null 2>&1; then
  TRANSCRIPT_PYTHON="$(command -v python3.13)"
elif command -v python3 >/dev/null 2>&1; then
  TRANSCRIPT_PYTHON="$(command -v python3)"
else
  echo "Python 3 is missing."
  echo "Install it yourself from python.org, then run this again."
  read -n 1 -s -r -p "Press any key to close..."
  exit 1
fi

if ! "$TRANSCRIPT_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10 or newer is required for current YouTube support."
  echo "Install a current Python yourself, then run this again."
  read -n 1 -s -r -p "Press any key to close..."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "FFmpeg is required but is not installed."
  echo ""
  echo "For safety this script will NOT install system software automatically."
  echo "If you already use Homebrew, run this yourself in Terminal:"
  echo ""
  echo "  brew install ffmpeg"
  echo ""
  read -n 1 -s -r -p "Press any key to close..."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "First-time setup will create an isolated .venv inside this folder."
  echo "It will install only the Python packages listed in requirements.txt."
  echo ""
  read -r -p "Continue with the isolated Python setup? [y/N] " answer
  case "$answer" in
    [yY][eE][sS]|[yY]) ;;
    *) echo "Cancelled."; exit 0 ;;
  esac

  "$TRANSCRIPT_PYTHON" -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  touch .setup-complete
else
  source .venv/bin/activate
  if [ ! -f ".setup-complete" ]; then
    echo "The local environment exists but setup is incomplete."
    echo "Delete the .venv folder and run START.command again."
    exit 1
  fi
  if ! python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "This app's existing .venv uses an outdated Python version."
    echo "Move or delete only the .venv folder, then run START.command again."
    exit 1
  fi
fi

echo ""
echo "Starting ONLY on http://127.0.0.1:8000"
echo "Keep this Terminal window open while using the tool."
echo "Press Control+C in Terminal to stop it."
echo ""

(sleep 2; open "http://127.0.0.1:8000") &
exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
