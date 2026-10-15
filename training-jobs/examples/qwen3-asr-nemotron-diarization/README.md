# Qwen3-ASR + Nemotron 3 Diarization

Speaker-attributed transcription in 18 languages: who said what, with word timings. [NVIDIA Nemotron 3 Diarization](https://www.baseten.co/library/nemotron-3-diarization/) decides who is speaking on every 80 ms frame, and a Qwen3-ASR-1.7B that Baseten fine-tuned for speaker-sliced audio transcribes each speaker. Both presets run on one NVIDIA RTX PRO 6000 and deploy from the Model Library.

| | Real-time | Batch |
| --- | --- | --- |
| Deploy | [nemotron-3-td-qwen-streaming](https://app.baseten.co/deploy/baseten/nemotron-3-td-qwen-streaming) | [nemotron-3-td-qwen-batch](https://app.baseten.co/deploy/baseten/nemotron-3-td-qwen-batch) |
| Transport | WebSocket, one connection per stream | HTTP, one request per file |
| Capacity per GPU | 64 real-time streams | 107 six-minute files per minute |
| Cost per audio hour | about $0.066 | about $0.0066 |

Weights: [`baseten-admin/qwen-baseten-mtg-notes`](https://huggingface.co/baseten-admin/qwen-baseten-mtg-notes) (Apache 2.0). For English only, with lower real-time latency and higher stream density, see [Parakeet + Nemotron 3 Diarization](../parakeet-nemotron-diarization/).

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

All messages are JSON text frames. Send an optional handshake first, then base64 PCM16 16 kHz mono audio in 100 ms frames at real-time pace, then a commit:

```json
{"session_id": "call-42", "max_speakers": 8, "language": "en", "mode": "meeting"}
{"type": "input_audio_buffer.append", "audio": "<base64 PCM16 16 kHz mono>"}
{"type": "input_audio_buffer.commit"}
```

| Handshake field | Default | What it does |
| --- | --- | --- |
| `session_id` | random | Echoed in every server frame. |
| `max_speakers` | 8 | 1 to 8 speaker slots for the connection. |
| `language` | `auto` | `auto` or one of `en zh de pl nl pt es id fr it ja ko ru ar hi tr vi th`. Set it when you know the language. A code outside the list falls back to `auto`. |
| `mode` | `meeting` | `meeting` suits far-field multi-party audio. `call` suits close-mic two-party calls: no cross-speaker leak, at the cost of some overlapped words. |
| `horizon_s`, `context_s` | 5, 10 | Final cadence in seconds of a speaker's activity, and the same-speaker context decoded with it. |
| `partial_s`, `partial_context_s` | 0.5, 3 | Partial cadence (0 = finals only) and its context. |
| `pause_s`, `max_wait_s`, `max_turn_s` | 0.8, 6, 12 | Silence that closes a turn, wall-clock cap on uncommitted audio, longest open turn. |
| `overlap_mode`, `dominance_margin` | `argmax`, 0.25 | The slicing gate. Explicit values override `mode`. |
| `min_slice_s` | 0.5 | Defer slices shorter than this. 0 = off. |
| `max_wps` | 6 | Drop sub-second finals denser than this many words per second. 0 = off. |
| `loop_retry` | 2 | Re-decode degenerate windows up to this many times. 0 = off. |
| `dup_merge_jaccard` | 0.5 | Merge two labels that are one voice when their activity overlap reaches this. 0 = off. |
| `loop_cap`, `loop_cap_backchannel` | 3, 1 | Collapse a 1- to 3-gram repeated more than this many times in a row; back-channels such as "yeah", "ok", "mm-hmm" are exempt while `loop_cap_backchannel` is 1. |
| `merge_relabel` | 1 | After a duplicate-label merge, relabel the merged label's earlier turns to the survivor and delete echoes, announced once in `relabels`. |

Numeric fields are bounded to [0, 60]. The defaults are tuned; change the guards only when debugging.

Audio frames carry raw little-endian 16-bit mono PCM at 16 kHz with no WAV header. The commit flushes the remaining audio, sends the final frame, and closes the socket.

Every server frame carries the full current list of closed turns, so a client that falls behind can skip frames and lose nothing:

```json
{"type": "transcription", "is_final": false, "session_id": "call-42", "processed_s": 12.32, "num_speakers": 2,
 "segments": [
   {"speaker": "speaker_0", "start": 0.51, "end": 4.2, "text": "So the agenda today.",
    "overlap": true, "overlaps_with": ["speaker_1"],
    "words": [{"w": "So", "start": 0.51, "end": 0.59}, {"w": "the", "start": 0.59, "end": 0.7}]},
   {"speaker": "speaker_1", "start": 2.9, "end": 3.3, "text": "Mm-hmm.",
    "overlap": true, "overlaps_with": ["speaker_0"], "words": [{"w": "Mm-hmm.", "start": 2.9, "end": 3.3}]}],
 "partial": [{"speaker": "speaker_0", "text": "and then the", "start": 4.4, "end": 5.1, "committed_text": ""}]}
```

| Field | Description |
| --- | --- |
| `type` | `transcription`, or `error`. |
| `is_final` | `true` only on the last frame, after the commit. |
| `session_id` | From the handshake, or generated. |
| `processed_s` | Seconds of audio processed so far. |
| `num_speakers` | Distinct speakers seen so far. |
| `segments` | All closed turns so far, ordered by start. Each has `speaker`, `start`, `end`, `text`, `overlap`, `overlaps_with`, and `words` as `{"w", "start", "end"}`. Turns of different speakers can overlap. |
| `partial` | One entry per speaker with an open tail: `text` is the uncommitted tail and `committed_text` the part already fixed. Empty on the final frame. |
| `relabels` | Present once, on the frame after a duplicate-label merge: `[{"from", "new", "span", "text", "action"}]`, where `action` is `relabel` or `delete`. Clients that key on spans apply it; clients that redraw from `segments` need nothing. |
| `stats` | Final frame only. |

Labels run `speaker_0` to `speaker_7`, ordered by first arrival and local to the session. A final frame's `stats` looks like this:

```json
{"audio_s": 363.4, "activity_s": 291.2, "commits": 118, "partials": 540,
 "fixes": {"loop_splits": 1, "skipped_implausible": 0, "deferred_tiny": 3, "dup_merged": {}},
 "commit_lat_s": {"n": 118, "p50": 0.9, "p95": 1.7, "max": 4.2},
 "language": "auto", "lid": {"lang": "en"}}
```

Errors arrive as one `{"type": "error", "error": "..."}` frame, then the socket closes: a field out of range, a malformed frame, or `capacity: ...` when the replica is at its session budget. On `capacity`, reconnect and the gateway routes to another replica when one exists.

[`transcribe_live.py`](transcribe_live.py) streams a WAV at real-time pace and prints each closed turn once plus each speaker's live partial. Output on the 11-second JFK sample against a live deployment, partials trimmed:

```bash
curl -fL https://docs.baseten.co/assets/audio/jfk.wav -o jfk.wav
uv run --python 3.11 transcribe_live.py jfk.wav en
```

```
    … speaker_0: And so my fellow Americans ask not what your country
    … speaker_0: And so my fellow Americans ask not what your country can do for you
    … speaker_0: And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
[speaker_0   0.31- 10.63] And so my fellow Americans ask not what your country can do for you ask what you can do for your country.

1 speaker(s), 1 turns, 11.0 s processed
```

## Batch

```
POST https://model-{model_id}.api.baseten.co/environments/production/predict
Authorization: Bearer {api_key}
```

```json
{"transcription_input": {"audio": {"url": "https://example.com/meeting.wav"}, "latency": "offline", "language": "en", "max_speakers": 8}}
```

Nemotron 3 labels the speakers, each speaker's turns are packed into windows of at most 28 s of their own speech, and each window is transcribed once.

| Request field | Default | What it does |
| --- | --- | --- |
| `transcription_input` | required | Wrapper object. `diarization_input` is accepted as an alias. |
| `audio` | required | `{"url": ...}` or `{"audio_b64": ...}`. Any format ffmpeg decodes, resampled to 16 kHz mono on the server. For files over a few minutes send FLAC or a URL; a request must finish within the gateway's 300 s limit. |
| `latency` | `offline` | Diarizer profile: `offline` (30.4 s buffer, best accuracy), `low` (1.04 s), or `ultralow` (0.32 s). |
| `language` | `auto` | The same 18 codes. `auto` makes one decision per file from the longest window, so set it when you know it. The hint steers transcription; it isn't translation. |
| `max_speakers` | 8 | 1 to 8. Extra diarizer speakers are dropped by total speech. |
| `pack_s`, `context_s` | 28, 0 | Seconds of one speaker's speech per window (2 to 30), and same-speaker context per window (0 to 20). |
| `loop_retry`, `dup_merge_jaccard` | 2, 0.5 | Re-decode depth for degenerate windows, and the activity overlap above which duplicated speaker labels merge. 0 = off. |
| `debug` | false | Top-level flag beside `transcription_input`. Adds the raw diarizer turns and every window to the response. |

```json
{
  "segments": [
    {"speaker": "speaker_0", "start": 5.19, "end": 16.55,
     "text": "Hello, my friends. We're going to start the meeting now.",
     "words": [{"w": "Hello,", "start": 5.19, "end": 5.61}]},
    {"speaker": "speaker_1", "start": 17.28, "end": 23.52, "text": "Okay, I want to start.", "words": []}
  ],
  "speakers": 3,
  "num_speakers": 3,
  "text_by_speaker": {"speaker_0": "…", "speaker_1": "…", "speaker_2": "…"},
  "language": {"requested": "auto", "used": "en"},
  "latency": "offline",
  "timing": {"fetch_ms": 14, "diar_ms": 610, "pack_ms": 3, "qwen_ms": 2210, "qwen_calls": 9, "qwen_decoded_s": 191.4, "audio_s": 363.4, "total_ms": 2840},
  "packing": {"pack_s": 28, "gap_s": 0.1, "loop_retry": 2, "dup_merge_jaccard": 0.5, "n_windows": 9, "loop_splits": 0, "dup_merged": []}
}
```

| Response field | Description |
| --- | --- |
| `segments` | Sorted by start. A segment is a run of one speaker's turns closer than 2.5 s; segments of different speakers can overlap. `words` timings are placed proportionally inside each speaker span, not forced-aligned. |
| `speakers`, `num_speakers` | Speakers with at least one word. Labels are local to the file and ordered by arrival. |
| `text_by_speaker` | Each speaker's full text in order. |
| `language` | `requested` echoes the request; `used` is the language prefilled for transcription, or `auto` when no confident detection happened. |
| `latency` | The diarizer profile that ran. |
| `timing` | Server timings in milliseconds (`fetch_ms`, `diar_ms`, `pack_ms`, `qwen_ms`, `total_ms`), the number of transcription calls, and decoded seconds. |
| `packing` | The packing settings used, the window count, re-decodes, and label merges. |

A malformed request, unknown `latency`, out-of-range field, or undecodable audio returns HTTP 400 with `{"detail": "..."}`. A transcription failure on one window drops that window's words, never the whole request. [`transcribe_file.py`](transcribe_file.py) sends a URL and prints one line per segment:

```bash
uv run --python 3.11 transcribe_file.py https://example.com/meeting.wav en
```

## Language

Set `language` whenever you know it. Under `language: en`, 398,000 words of English output contained no non-Latin characters. `auto` is weaker for Mandarin: it translated Mandarin to English in 9 of 12 AISHELL-4 recordings, while `language: zh` transcribes Mandarin correctly.

## Accuracy

cpWER, lower is better, scored with the Whisper normalizer and meeteval. N is the number of full repeats.

| Dataset | Real-time | Batch | Parakeet real-time | Parakeet batch |
| --- | --- | --- | --- | --- |
| NOTSOFAR-1 eval-30 (30 far-field meetings) | 27.44 ± 0.34 (N=3) | 21.40 ± 0.07 (N=3) | 29.49 | 28.58 |
| NOTSOFAR-1, all 129 meetings | 25.95 ± 0.04 (N=3) | 20.23 | 26.16 | 28.94 |
| AMI headset mix, 16 meetings | 19.51 ± 0.06 (N=3) | 15.15 ± 0.05 (N=3) | 13.26 | 13.27 |
| AISHELL-4 Mandarin, character level | 19.21 (`language: zh`) | 18.36 | not supported | not supported |

This pairing is stronger on overlapped far-field meetings and in other languages. The Parakeet pairing is stronger on clean far-field English rooms like AMI.

## Known limitations

- Occasional short repeats of a 1- to 4-word phrase, about 1 per 1,000 words. No runaway loops.
- Real-time, dense meetings: a phrase can occasionally show under two labels within a second or two.
- Real-time, close-mic two-person calls: one voice can carry two labels for the first few seconds until the merge fires, and `relabels` then fixes the history. `mode: call` avoids it.
- A real-time connection that sends nothing for 120 s is closed by the server.
- Costs use the $4.25 per hour RTX PRO 6000 list price.
