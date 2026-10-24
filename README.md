# ClearCommit — TaskForge submission

A Windows desktop app for **two-way live speech interpretation** with a new
**Conversation Clarity** feature, rendered in a modern HTML/CSS/JS interface
inside a native desktop window (pywebview / WebView2). Python owns all audio,
AI, and state; the frontend only displays what Python sends.

## Run it

```
pip install -r requirements.txt
copy .env.example .env   # paste your keys
python app.py
```

Keys are read from the environment, a local `.env`, or
`%APPDATA%\LiveInterpreter\.env`. **No key is ever sent to the frontend.**

| Key | Used for |
|-----|----------|
| `GROQ_API_KEY` | STT (Whisper), LLM translation, Clarity provider (default) |
| `DEEPGRAM_API_KEY` | STT (Nova-3) + TTS (Aura) |
| `ELEVENLABS_API_KEY` | TTS fallback for languages Deepgram Aura lacks |
| `TASKFORGE_AI_PRESET` | Clarity provider: `groq` (default) / `openrouter` / `custom` |

For Discord calls: install VB-Cable, route the app's TTS output to
**CABLE Input**, and set Discord's input to **CABLE Output**.

## What works (honest status)

- **Two-way live interpretation**: Person 1 (system/loopback audio) → English
  captions; your microphone → their language as text and TTS. This is the
  proven engine from the pre-existing ClearCommit project (see ATTRIBUTION).
- **Device selection**: real enumeration via sounddevice; default devices
  preselected; understandable errors when enumeration or startup fails.
- **Recognizer choice**: cloud Groq Whisper (needs key) or **local offline
  Whisper** (`pip install faster-whisper`, no key, no network, private —
  downloads the model on first use).
- **Auto-detect**: Person 1's language can be fixed or auto-detected per
  utterance. The AI provider for Clarity is auto-detected too
  (`TASKFORGE_AI_PRESET=auto` picks the first provider that has a key).
- **Conversation Clarity** (new, built for TaskForge): after each finalized
  utterance, the configured LLM is asked to identify at most one materially
  plausible ambiguity. Output is validated deterministically — a "quote" that
  doesn't appear in the utterance is rejected, confidence is clamped, and no
  ambiguity produces no alert. You can edit the suggested clarification
  question and approve it. Approved questions are **displayed only** — never
  auto-sent, auto-spoken, or treated as agreement.
- **Honest UI**: status pill reflects real session state; timestamps are
  message arrival times; there are no simulated transcripts, latency numbers,
  or decorative controls.

## Known limitations

- "Pause" freezes the transcript **view**; audio keeps flowing (a live call
  must not be interrupted).
- Language changes apply on the next session start.
- Clarity analyzes finalized utterances one at a time and is rate-limited by
  the provider; free-tier models may throttle.
- STT is file-based (per-utterance), so per-sentence latency has a floor of
  roughly 0.5–2 s.
- One language per side; code-switching is not modeled.

## Architecture

```
mic / loopback ──► StreamingInterpreter (engine/) ──► pop_captions() ─┐
                        │                    │                        ▼
                        │                    └─ on_final_utterance ─► event pump ──► window.__push()
                        │                                     (Clarity via AIService)
                        ▼
                 TTS (Deepgram Aura / ElevenLabs) → selected output device
```

- `app.py` — webview window, whitelisted `js_api` bridge, event pump.
- `bridge_payloads.py` — schema for every Python→JS message (unit-tested).
- `conversation_clarity.py` — bounded ambiguity detection + validation.
- `engine/` — the interpreter pipeline, copied unmodified from the
  pre-existing ClearCommit project (see ATTRIBUTION).
- `ui/` — HTML/CSS/JS frontend. No frameworks, no CDNs, works offline.

## Tests

```
python -m pytest tests/ -q
python app.py --selftest   # checks deps/devices/credentials incl. frozen builds
```

## Troubleshooting

- **No audio devices listed** — check the OS input device; run
  `python app.py --selftest` to see what the app can see.
- **Pipeline fails to start** — the error is shown in the app banner AND as a
  toast from any view; `--selftest` pinpoints missing modules/credentials.
- **STT errors with the local recognizer** — install `faster-whisper`
  (`pip install -r requirements.txt` covers it) or switch to Cloud.

## TaskForge integrity notes

Pre-existing components (the `engine/` directory and the Clarity provider
infrastructure concepts) come from the author's earlier ClearCommit project
and are disclosed in ATTRIBUTION.md. The web UI, the clarity feature, and the
desktop-window integration were built for this event. **Confirm with the
TaskForge rules** whether pre-existing code reuse and dual-event submissions
are permitted before submitting.
