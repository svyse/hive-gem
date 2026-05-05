Voice answers default patch

What changed
- Spoken replies now play for Q&A answers even when the user typed the query.
- Code-run narration also follows the same global voice-answer preference, regardless of whether the original prompt was typed or dictated.
- Turning voice answers off now cancels any in-progress browser speech synthesis.
- The UI text was updated so the toggle clearly states that it applies to both typed and voice queries.

Files changed
- frontend/src/App.jsx
- frontend/src/components/PromptPanel.jsx

Behavior
- Default remains ON via localStorage key: speakVoiceResponses
- Users can still disable spoken replies with the existing checkbox in the prompt panel.
