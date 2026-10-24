# Demo script (~90 seconds)

1. **Window opens (0:00)** — dark, calm UI. Header shows real state "Idle" and
   the configured language pair. Sidebar: Conversation / Audio & languages /
   Clarity. Transcript area shows the honest empty state.
2. **Audio & languages (0:10)** — open the section: real device lists
   (sounddevice), defaults preselected, credentials check marks from actual
   environment keys. Select mic and TTS output (CABLE Input for Discord).
3. **Start (0:25)** — press Start. Status pill → "Listening". Speak in your
   language: finalized utterances appear with speaker label, source text,
   arrival timestamp — then the translation arrives and fills in. The other
   side of the call is captured from system/loopback audio and captioned.
4. **TTS (0:45)** — your speech is spoken in the other language through the
   selected output device (into the call).
5. **Conversation Clarity (0:55)** — say something genuinely ambiguous
   (e.g. an idiom, or "let's sync on the deliverable next week" without a
   time). A Clarity card appears: the quoted phrase (verified to exist in the
   utterance), why it may be misread, alternative readings, a model-estimated
   confidence bar, and an editable suggested clarification question.
6. **Edit & approve (1:10)** — reword the question, press Approve. The card
   turns green and says clearly: "Approved — shown here only, not sent or
   spoken." Press Dismiss on another card to remove it.
7. **Stop (1:25)** — stop the session; state returns to "Stopped".
