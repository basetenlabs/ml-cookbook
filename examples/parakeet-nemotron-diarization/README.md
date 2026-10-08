# Parakeet + Nemotron 3 Diarization

Speaker-attributed transcription in English: who said what, in real time or from a recording. [NVIDIA Nemotron 3 Diarization](https://www.baseten.co/library/nemotron-3-diarization/) tracks who is speaking on every 80 ms frame, and NVIDIA's multitalker Parakeet 0.6B runs one ASR instance per speaker on the same mixed audio, so fully overlapped speech is transcribed per speaker. Both presets run on one NVIDIA RTX PRO 6000 and deploy from the Model Library.

| | Real-time | Batch |
| --- | --- | --- |
| Deploy | [nemotron-3-diarized-transcription-streaming](https://app.baseten.co/deploy/baseten/nemotron-3-diarized-transcription-streaming) | [nemotron-3-diarized-transcription-batch](https://app.baseten.co/deploy/baseten/nemotron-3-diarized-transcription-batch) |
| Transport | WebSocket, one connection per stream | HTTP, one request per file |
| Capacity per GPU | 190 hour-long streams | 61 six-minute files per minute |
| Latency | words a median 0.7 s after they're spoken | p50 about 29 s per file under full load |
| Cost per audio hour | about $0.022 | about $0.012 |

For other languages, see [Qwen3-ASR + Nemotron 3 Diarization](../qwen3-asr-nemotron-diarization/).

## Setup

Deploy from a link above, wait for the deployment to become active, and copy its model ID.

```bash
export BASETEN_API_KEY="YOUR_API_KEY"
export BASETEN_MODEL_ID="YOUR_MODEL_ID"
```

To target a specific deployment instead of production, replace `environments/production` in the URLs with `deployment/{deployment_id}`.

## Real-time

```
wss://model-{model_id}.api.baseten.co/environments/production/websocket
Authorization: Bearer {api_key}
```

All messages are JSON text frames; audio travels base64-encoded inside JSON, not as binary frames.

### Client messages

Send an optional handshake first, then audio, then a commit:

```json
{"session_id": "call-42", "max_speakers": 8}
{"type": "input_audio_buffer.append", "audio": "<base64 PCM16 16 kHz mono>"}
{"type": "input_audio_buffer.commit"}
```

| Handshake field | Default | What it does |
| --- | --- | --- |
| `session_id` | random | Echoed in every server frame. Use it to correlate logs. |
| `max_speakers` | 8 | 1 to 8. Caps the per-speaker ASR instances. Speakers beyond the cap are merged into existing labels. |
| `words` | 1 | Per-word timings in `segments`. Set 0 to cut frame size by about 3×. |
| `overlap` | 1 | Let turns of different speakers overlap in time. Set 0 for a single running tail where simultaneous speech alternates word by word. |
| `turn_segments` | 1 | Cut `segments` into turns. Set 0 for NeMo's native output: one running block per speaker, without `partial` or `words`. |
| `partials` | 1 | Send the `partial` array. |

The handshake is optional and must be the first frame when sent. Audio frames carry raw little-endian 16-bit mono PCM at 16 kHz with no WAV header; 100 ms frames (3,200 bytes) at real-time pace work well. The commit flushes the remaining audio, sends the final frame, and closes the socket.

### Server messages

The server sends a frame on every ASR chunk, about every 1.12 s. Every frame carries the full current list of closed turns, so replace your view with it; a client that falls behind can skip frames and lose nothing. A real frame from a single-speaker recording, with `words` trimmed to two per turn:

```json
{"type": "transcription", "is_final": false, "session_id": "demo", "processed_s": 12.32, "num_speakers": 1,
 "segments": [
   {"speaker": "speaker_0", "start": 1.12, "end": 8.08,
    "text": "The birch canoe slid on the smooth planks, glue the sheet to the dark blue background.",
    "overlap": false, "overlaps_with": [],
    "words": [{"w": "The", "start": 1.12, "end": 1.2}, {"w": "birch", "start": 1.28, "end": 1.68}]},
   {"speaker": "speaker_0", "start": 8.24, "end": 11.28, "text": "It is easy to tell the depth over well.",
    "overlap": false, "overlaps_with": [],
    "words": [{"w": "It", "start": 8.24, "end": 8.32}, {"w": "is", "start": 8.4, "end": 8.48}]}],
 "partial": [{"speaker": "speaker_0", "text": "These days a chicken leg", "start": 11.36, "end": 12.24}]}
```

| Field | Description |
| --- | --- |
| `type` | `transcription`, or `error`. |
| `is_final` | `true` only on the last frame, after the commit. The final frame carries every closed turn and an empty `partial`. |
| `session_id` | From the handshake, or generated. |
| `processed_s` | Seconds of audio processed so far. |
| `num_speakers` | Distinct speakers seen so far. |
| `segments` | All closed turns so far, ordered by start. Each has `speaker`, `start`, `end`, punctuated and cased `text`, `overlap`, `overlaps_with`, and `words` as `{"w", "start", "end"}`. Word boundaries sit on the ASR's 80 ms frame grid. |
| `partial` | One entry per speaker currently talking: `speaker`, `text`, `start`, `end`. Re-sent every chunk until it closes into `segments`. |

Labels run `speaker_0` to `speaker_7`, ordered by first arrival and local to the session. A closed turn is immutable except that a trailing word piece can be appended if a word straddled a chunk boundary, and `overlap` can flip to `true` when a later-closing parallel turn touches it.

Each speaker has their own open tail. It closes into a turn when that speaker pauses for more than 1.2 s, at sentence-final punctuation, or when others have talked for at least 1 s or 3 words since their last word. A one- or two-word back-channel never splits the running turn; it becomes its own short overlapping turn. Because tails are per speaker, two people talking at once produce two parallel turns rather than alternating fragments.

### Errors

The server sends one `{"type": "error", "error": "..."}` frame and closes the socket:

| Error | Cause |
| --- | --- |
| `max_speakers must be in [1, 8]`, `max_speakers must be an integer` | Bad handshake value. |
| `frame must be a JSON object`, or a base64 decode error | Malformed client frame. |
| `internal error: <ExceptionName>` | Server fault. |

### Client

[`transcribe_live.py`](transcribe_live.py) streams a 16 kHz mono PCM16 WAV at real-time pace and prints each closed turn once plus each speaker's live partial:

```bash
curl -fL https://docs.baseten.co/assets/audio/jfk.wav -o jfk.wav
uv run --python 3.11 transcribe_live.py jfk.wav
```

## Batch

```
POST https://model-{model_id}.api.baseten.co/environments/production/predict
Authorization: Bearer {api_key}
Content-Type: application/json
```

### Request

```json
{"transcription_input": {"audio": {"url": "https://example.com/meeting.wav"}, "max_speakers": 8}}
```

| Field | Default | What it does |
| --- | --- | --- |
| `transcription_input` | required | Wrapper object. |
| `transcription_input.audio` | required | `{"url": ...}` or `{"audio_b64": ...}`. Any format ffmpeg decodes, normalized to 16 kHz mono on the server. For long files send a URL or a compressed format such as FLAC. |
| `transcription_input.max_speakers` | 8 | 1 to 8. Caps the per-speaker ASR instances. |

Requests that arrive close together run as one batch on the GPU, so throughput rises with concurrency. A request must finish within the gateway's 300 s limit.

### Response

```json
{"segments": [{"speaker": "speaker_0", "start": 0.51, "end": 4.2, "text": "So the agenda today."}],
 "speakers": 3,
 "text_by_speaker": {"speaker_0": "…", "speaker_1": "…", "speaker_2": "…"},
 "compute_s": 21.6, "peak_gpu_gb": 4.6, "batch_n": 1}
```

| Field | Description |
| --- | --- |
| `segments` | Speaker turns sorted by start: `speaker`, `start`, `end` in seconds, and `text`. Turns of different speakers can overlap. No per-word timings. |
| `speakers` | Number of speakers with at least one word. Labels are local to the file and ordered by arrival. |
| `text_by_speaker` | Each speaker's full text in order. |
| `compute_s` | GPU compute for the batch this request ran in. |
| `peak_gpu_gb` | Peak GPU memory during that batch. |
| `batch_n` | How many requests shared that batch. |

### Errors

Client errors return HTTP 400 with `{"detail": "..."}`:

| `detail` | Cause |
| --- | --- |
| `request requires 'transcription_input'` | Missing wrapper. |
| `transcription_input.audio must be an object with 'url' or 'audio_b64'`, `audio requires 'url' or 'audio_b64'` | Missing or malformed audio. |
| `max_speakers must be an integer`, `max_speakers must be in [1, 8]` | Bad `max_speakers`. |
| `could not decode audio: ...` | ffmpeg rejected the bytes. |
| `audio is empty` | Decoded audio has no samples. |

### Client

[`transcribe_file.py`](transcribe_file.py) sends a URL and prints one line per segment:

```bash
uv run --python 3.11 transcribe_file.py https://example.com/meeting.wav
```

## Accuracy

cpWER, lower is better, scored with the Whisper normalizer and meeteval on the same sessions, references, and scorer for every row.

| Dataset | Parakeet real-time | Parakeet batch | Qwen3-ASR real-time, for reference |
| --- | --- | --- | --- |
| NOTSOFAR-1 eval-30 (30 far-field meetings) | 29.49 | 28.58 | 27.44 |
| NOTSOFAR-1, all 129 meetings | 26.16 | 28.94 | 25.95 |
| AMI headset mix, 16 meetings | 13.26 | 13.27 | 19.51 |

This pairing is strongest on clean far-field English rooms like AMI and has the lowest real-time latency and highest stream density. The Qwen3-ASR pairing recovers more overlapped speech on dense meetings and covers 18 languages.

## Limits

- English only.
- 8 speakers per connection or file.
- Real-time: algorithmic latency is 1.12 s, the ASR chunk, and speaker activity is known about 1.04 s after the audio.
- Real-time: a connection that sends nothing for 120 s is closed by the server.
- Real-time: each replica admits up to 190 streams; extra connections wait at the gateway rather than degrading live ones.
- Batch: word timings are not returned.
- Costs use the $4.25 per hour RTX PRO 6000 list price.
