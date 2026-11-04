# English speed and accuracy

English transcription automatically uses Apple GPU acceleration on compatible
Apple Silicon Macs. It keeps the multilingual `small` Whisper model and the
existing Silero speech filter, with greedy decoding selected by the comparison
below. Other devices, missing GPU dependencies, custom models/devices, and
requests for other languages use faster-whisper (CPU int8 by default) and beam size 5.
If automatic GPU initialization fails, the app falls back to the standard engine.

The web app defaults to English. API clients can supply another language code,
or `null` for automatic detection, to retain the original multilingual engine.
Model instances are reused while the server runs. Completed segments appear
before the full result for both URLs and uploads.

## Setup

New installations use the normal `START.command` setup. The GPU binding is a
prebuilt Python package installed only on Apple Silicon macOS; no separate
Whisper app or native command-line tools are needed.

For an existing installation, update dependencies once from the project folder:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

The first GPU job downloads about 466 MiB of model weights to the user's Hugging
Face cache and prepares GPU shaders. Subsequent jobs reuse the loaded model.
There are no account requirements, paid APIs, or per-transcription charges.

To deliberately use the previous engine, stop the server and launch:

```bash
WHISPER_ENGINE=faster-whisper ./START.command
```

`WHISPER_ENGINE=metal` still explicitly selects the GPU engine. The legacy
`requirements-metal.txt` entry point remains compatible with the main requirements.

## Focused release comparison

Tested on an Apple M1 MacBook Air with 16 GB RAM, faster-whisper 1.2.1 /
CTranslate2 4.8.1, and pywhispercpp 1.5.1. Nine English recordings contained
95.565 seconds of speech and 274 normalized reference words. The eight
LibriSpeech clips have different speakers and were selected before inference;
the ninth is the JFK sample containing negation. References include proper
names and spoken numbers. Both engines use the same speech selection.

| Configuration | Total inference | Word errors |
| --- | ---: | ---: |
| Previous CPU, auto language, beam 5 | 40.167 s | 3 / 274 (1.09%) |
| GPU, explicit English, beam 5 | 12.518 s | 4 / 274 (1.46%) |
| GPU, English-only small.en model, beam 5 | 15.749 s | 4 / 274 (1.46%) |
| **Selected: GPU, explicit English, greedy** | **11.069 s** | **3 / 274 (1.09%)** |

The selected setting was about 3.6 times faster than the previous CPU baseline
and made the same three normalized word errors, involving names/spelling.
The checked numbers and negation were retained; no phrases were omitted or
inserted. The English-only model was not selected because it was slower and
offered no accuracy improvement on this set.

Inference times include speech filtering but exclude model loading, downloads,
and social-media retrieval. CPU weights are int8; GPU weights are FP16. WER is
Levenshtein substitutions + deletions + insertions divided by reference words.
Normalization ignores case/punctuation, expands common contractions, and treats
numerals below 100 as their spoken words. Each shortlisted configuration ran
once on this fixed set. These clean recordings are a release comparison, not
proof of accuracy on noisy conversations, all accents, or specialist terminology.
Do not interpret 1.09% WER as a universal accuracy guarantee.

Sources: [LibriSpeech CC BY 4.0](https://www.openslr.org/12/),
[dataset revision 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1](https://huggingface.co/datasets/openslr/librispeech_asr/tree/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1),
and [whisper.cpp JFK sample](https://github.com/ggml-org/whisper.cpp/tree/master/samples).
LibriSpeech row indices: 3, 101, 302, 601, 904, 1202, 1803, 2404 (clean/test).

## Unchanged limits

This release does not add history or change duration limits. Processing remains
capped at the first 15 minutes, uploads at 250 MiB, and audio conversion at a
180-second timeout. Longer recordings need a separate update.
