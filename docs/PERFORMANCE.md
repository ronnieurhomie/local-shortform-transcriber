# Transcription performance

The standard engine remains faster-whisper with multilingual `small`, CPU int8,
beam size 5, and Silero speech filtering. Models are reused while the app runs.
Both URL and upload flows show completed transcript segments before the full
result. This reduces waiting for readable text; it does not itself reduce total
inference time. Source-clock timings separate extraction/conversion, model
preparation, and inference, and include repeated fallback attempts.

## Optional Apple Silicon acceleration

The optional Metal engine uses whisper.cpp through `pywhispercpp`. It uses the
same multilingual `small` model size, beam size 5, and the standard engine's
Silero speech filter. Timestamps are mapped back to the original audio, and
duration/WPM use the whole input. Decoder state is cleared between separate jobs.

After normal setup, install the optional binding once from the project folder:

```bash
.venv/bin/python -m pip install -r requirements-metal.txt
```

Stop the running server, then launch with:

```bash
WHISPER_ENGINE=metal ./START.command
```

Launching `START.command` normally uses the original engine. Metal currently
supports Apple Silicon macOS and the default `small` model. Other models/devices
continue to use the standard engine. There is no paid service or per-use fee.

The first Metal launch downloads an additional model-format file of about
466 MiB to the user's Hugging Face cache. It also prepares GPU shaders, which
can take extra time. The model is kept in memory for subsequent jobs. The
optional binding ships as a prebuilt Python wheel; no separate native app or
command-line Whisper installation is needed.

## Measured results

Measured on an Apple M1 MacBook Air with 16 GB RAM, using faster-whisper 1.2.1 /
CTranslate2 4.8.1 and pywhispercpp 1.5.1. CPU used four threads. Both engines used
automatic language detection, multilingual `small`, beam size 5, and the same
Silero speech selection. CPU uses int8 weights; Metal uses the FP16 GGML model.
Each inference measurement is the mean of two runs
with the model loaded; media retrieval and first-time setup are excluded.

| Sample | CPU | Metal | Inference speedup |
| --- | ---: | ---: | ---: |
| English speech, 45 seconds | 13.54 s | 5.80 s | 2.33x |
| Natural Mandarin, 4.28 seconds | 3.81 s | 1.69 s | 2.26x |
| Synthetic Mandarin, 4 seconds | 4.58 s | 1.67 s | 2.73x |

On the English clip, the first segment appeared at about 7.68 s on CPU versus
3.73 s on Metal. Initial Metal model/shader preparation took about 20.6 s;
loading after shader preparation took about 1.1 s. Eight CPU threads were
slower than four on the tested machine, so the default was not increased.

These are small-sample results, not a guarantee for other videos or devices.
English wording differed in one repeat. Both engines made the same mistake
on the natural Mandarin fixture, and both misrecognized the opening of the
synthetic fixture. The optional engine is faster on these clips, but this does
not establish identical or universally better accuracy. It therefore remains
opt-in. No production model was made smaller and beam search was not reduced.

Public fixtures and references:

- [WeNet speech fixtures](https://github.com/wenet-e2e/wenet/tree/main/test/resources)
- [WeNet reference transcripts](https://github.com/wenet-e2e/wenet/blob/main/test/resources/dataset/text)
- [Hugging Face audio fixtures](https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples)
- [whisper.cpp JFK sample](https://github.com/ggml-org/whisper.cpp/tree/master/samples)

The 45-second English fixture is a timing and baseline comparison; it was not
scored against an independently verified reference. The short Mandarin clips
do not establish performance or accuracy on long conversations.

## Current limits

This speed update does not raise the limits: processing is capped at the first
15 minutes and uploads at 250 MiB. Longer recordings require separate duration,
memory, timeout, and completeness handling before the cap can be raised safely.
Audio conversion currently has a 180-second timeout, separate from inference.
