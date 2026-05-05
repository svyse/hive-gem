# Speech dictation fix (persistent mic + smart formatting)

## What this patch does
- Keeps dictation listening **until you press Stop**.
- Auto-restarts SpeechRecognition sessions when the browser ends them (common in Chrome/Edge after pauses).
- Adds light "smart formatting":
  - Say **"following points"** / **"bullet points"** to enter list mode.
  - Each subsequent final utterance becomes a bullet: `- ...`
  - Say **"end list"** / **"stop list"** to exit list mode.
  - Converts common spoken punctuation: "comma", "period", "question mark", "new line", etc.
  - Merges spelled letters: "s h e r a" -> "shera".

## How to apply
Replace this file in your repo:

`frontend/src/components/PromptPanel.jsx`

with the one included in this patch.

Then restart the frontend dev server:

```bash
cd frontend
npm run dev
```

## Notes / browser limitations
- Web Speech API behavior varies by browser.
- Chrome/Edge frequently triggers `onend` after silence; this patch auto-restarts so you don't have to re-click.
- If you see "not-allowed" or "audio-capture" errors, the browser blocked mic access; you must re-allow permissions.
