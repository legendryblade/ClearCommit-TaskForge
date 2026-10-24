# Attribution & pre-existing work disclosure

## Reused from the author's pre-existing ClearCommit project

The following files in `engine/` are copied **unmodified** from the author's
earlier, private ClearCommit repository (the "Live Interpreter" desktop app)
and predate this event:

- `stream_pipeline.py` — capture → STT → translate → TTS pipeline
- `transcriber.py` — Groq Whisper / Deepgram Nova-3 STT backends
- `audio_capture.py`, `loopback_capture.py` — mic and loopback capture
- `translator.py`, `nvidia_stream.py`, `nvidia_translator.py` — LLM translation
- `deepgram_tts.py`, `elevenlabs_ws.py` — TTS backends
- `config.py`, `logger.py`, `utils.py` — configuration/logging helpers
- `ai_service.py` — bounded async AI execution (queue/timeout/circuit breaker)
- `ai_providers.py` — OpenAI-compatible provider client

Everything else — the native-window web UI (`ui/`, `app.py`,
`bridge_payloads.py`), the Conversation Clarity feature
(`conversation_clarity.py`), the bridge, tests, and documentation — was
created for this TaskForge submission.

## Third-party

- pywebview (BSD) — native window with WebView2
- OpenAI Groq, Deepgram, ElevenLabs — commercial APIs used via the author's
  own keys; no keys are included in this repository
