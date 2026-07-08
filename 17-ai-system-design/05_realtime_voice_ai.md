# Design 5 — Real-Time Voice AI

## Requirements
- Voice-in, voice-out conversation
- End-to-end latency: p95 < 800ms
- Interruptible (user can talk over the AI)
- Multi-language

## The tight latency budget

```
User speaks final syllable
        │
        ▼    Speech-to-Text (Whisper / Deepgram)    150ms
        ▼    LLM think + start streaming (TTFT)     250ms
        ▼    First TTS chunk (ElevenLabs streaming) 150ms
        ▼    Audio playback to user                  50ms
                                                     ────
                                                     600ms  ← total round-trip
```

That's tight. 3 tricks make it work:

## Trick 1: Streaming everywhere

```
STT streaming ─▶ LLM starts as soon as ~2 tokens land, doesn't wait for end-of-speech
LLM streaming ─▶ TTS starts on first sentence, not on full response
TTS streaming ─▶ audio bytes flow to speaker as generated
```

## Trick 2: VAD + speculative execution

Voice Activity Detection detects likely-end-of-speech at 100ms silence.
LLM starts speculatively on partial transcript. If user resumes, cancel.

## Trick 3: The "phatic filler" trick

If LLM TTFT is going to exceed budget, immediately play a natural-sounding
filler ("Sure, one moment...") while the real response generates. Users
perceive lower latency than actual.

## Architecture

```
                   ┌─────────────┐
      Mic ─▶ WS ─▶│  Audio       │──▶  STT stream (Deepgram/Whisper)
                   │  ingestion   │            │
                   └─────────────┘            ▼
                                    ┌──────────────────┐
                                    │  Turn manager   │
                                    │  (VAD + prompt   │
                                    │   assembly)     │
                                    └────────┬────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │   LLM (gpt-4o    │
                                    │   or Claude      │
                                    │   Haiku)         │
                                    │   streaming     │
                                    └────────┬────────┘
                                             │  tokens
                                             ▼
                                    ┌──────────────────┐
                                    │  TTS stream     │
                                    │(ElevenLabs)     │
                                    └────────┬────────┘
                                             │  audio bytes
                                             ▼
                                       ← WS ← Speaker
```

## Barge-in (interruption)

- STT keeps running even while TTS plays
- If STT detects new speech >200ms while assistant speaks:
  - Cancel current LLM generation (drop pending tokens)
  - Cancel current TTS
  - Start new turn

## Model choice

- **Latency-first LLMs**: Claude Haiku, gpt-4o-mini, or self-hosted Llama with vLLM
- Avoid gpt-4o for real-time; use for follow-up analysis, not conversation turns

## Cost per minute of conversation

```
STT     :  ~$0.006/min
LLM     :  ~$0.015/min (avg 500 tokens/min per party)
TTS     :  ~$0.10/min (ElevenLabs)
       ─────
Total   :  ~$0.12/min

Optimize: self-hosted TTS (Coqui/OpenVoice) drops to ~$0.02/min.
```

## Interview follow-ups

- How do you handle > 300ms network jitter? → local buffering + adaptive bitrate
- How do you support 20 languages? → language detection on STT → route to matching TTS voice
- What happens when the LLM says something inappropriate? → output guardrail also runs on token stream, can cancel + play "hmm let me rephrase..."
- How do you evaluate voice agents? → transcript-based evals (Phase 14) + separate audio-quality metrics (WER, latency histograms)
