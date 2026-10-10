# M3 Frontend Smoke Tests

The repository does not use a browser test framework. `m3-smoke.html` is a lightweight browser harness that loads the real frontend modules and uses deterministic Web Speech API stubs.

## Run

Start the backend from the project directory:

```powershell
uvicorn app:app --reload
```

Open `http://localhost:8000/m3-smoke.html` and select **Run automated smoke tests**.

## Coverage

### Automated browser tests with mocked browser APIs

The harness executes the real `voice.js`, `speech.js`, `chat.js`, and `transparency.js` modules and covers:

- Voice input start, accumulated live transcript, editable text, stop/restart, unsupported browsers, permission/device/network/no-speech errors, and missing start events.
- TTS play/pause/resume/stop, delayed start events, stale callbacks, language/voice selection, empty voice lists, and unsupported speechSynthesis.
- Transparency citation-to-chunk/document matching, comparative two-source mapping, answer/retrieval status, score and location metadata, low-confidence and empty/error states, keyboard disclosure, and unsupported/mismatched citation warnings.

The Web Speech APIs are mocked because microphone and speech synthesis cannot be deterministic in automated browser execution.

### Manual verification against the running application

The harness lists these checks because they require the real API/session lifecycle:

1. Submit an ambiguous query and verify the clarification response appears.
2. Answer the clarification and verify the same conversation continues through `/chat`.
3. Switch between two conversations and verify messages do not leak across sessions.
4. Start a new conversation and verify the previous conversation is no longer active.

Use the main application at `http://localhost:8000/` for these checks.

## Regression suite

The frontend harness does not replace the Python suite:

```powershell
python -m pytest -q
```
