"""Voice mode: Sprout listens and speaks, fully offline, with the same chat model behind it.

A cascaded pipeline, as in Kyutai's Unmute: the user's speech -> text -> the usual chat model -> speech.
(End-to-end speech models such as Moshi, arXiv 2410.00037, answer faster but are separate 7B GPU
models; they would not fit beside the 26B chat model, and the user wants Sprout's own model.)
All three parts run on the CPU through sherpa-onnx (k2-fsa, Apache-2.0), measured on this laptop:

- Silero VAD (MIT): finds where speech starts and ends (well under 1 ms per 32 ms chunk).
- NVIDIA Parakeet TDT 110M (English), int8 (CC-BY-4.0): speech to text in 0.06-0.10 s for a short
  sentence, 166 MB. Compared on the same clips (American, British and Indian-accented English, a Hindi
  sentence): Parakeet 0.6B v3 was as accurate but used 669 MB, which pushed the 26B chat model into
  SSD paging (replies 5-11 s instead of ~2 s); Moonshine base invented a sentence for the Hindi clip;
  Whisper-large-v3-turbo was 5x slower than Parakeet 0.6B, used 1.3 GB and garbled Hindi.
- Kokoro-82M v1.0 (Apache-2.0): text to speech, 24 kHz; 6.3 s of speech in 2.4 s on 4 threads
  (fp32). The int8 export was slower than real time on this CPU, so it isn't used.

The browser captures the microphone with echo cancellation, so the user can interrupt Sprout while
it speaks; it sends 16 kHz audio here and plays back what speak() returns. Models load when voice
mode starts and are freed when it stops (~0.7 GB of RAM).
"""
import io
import os
import queue
import re
import threading
import wave
from pathlib import Path

import numpy as np

LLM = Path(os.environ.get("LLM_HOME") or Path.home() / "llm")
VOICE_DIR = LLM / "models" / "voice"
KOKORO = VOICE_DIR / "kokoro-multi-lang-v1_0"
PARAKEET = VOICE_DIR / "sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8"
VAD_MODEL = VOICE_DIR / "silero_vad.onnx"
RATE = 16000  # microphone audio, as the VAD and recogniser expect
WINDOW = 512  # samples per VAD step (32 ms)
DEFAULT_VOICE = "am_michael"
# Kokoro's English voices graded C or better by its author (hexgrad/Kokoro-82M VOICES.md, "Overall Grade",
# an estimate of the quality and quantity of each voice's training data), best first. D-graded voices
# (Adam, Onyx, Echo, Eric, Liam, Alice, Lily, ...) sound noticeably less natural and are left out.
VOICES = [
    {"id": "am_michael", "name": "Michael", "gender": "male", "accent": "American", "grade": "C+"},
    {"id": "am_fenrir", "name": "Fenrir", "gender": "male", "accent": "American", "grade": "C+"},
    {"id": "am_puck", "name": "Puck", "gender": "male", "accent": "American", "grade": "C+"},
    {"id": "bm_george", "name": "George", "gender": "male", "accent": "British", "grade": "C"},
    {"id": "bm_fable", "name": "Fable", "gender": "male", "accent": "British", "grade": "C"},
    {"id": "af_heart", "name": "Heart", "gender": "female", "accent": "American", "grade": "A"},
    {"id": "af_bella", "name": "Bella", "gender": "female", "accent": "American", "grade": "A-"},
    {"id": "af_nicole", "name": "Nicole", "gender": "female", "accent": "American", "grade": "B-"},
    {"id": "bf_emma", "name": "Emma", "gender": "female", "accent": "British", "grade": "B-"},
    {"id": "af_aoede", "name": "Aoede", "gender": "female", "accent": "American", "grade": "C+"},
    {"id": "af_kore", "name": "Kore", "gender": "female", "accent": "American", "grade": "C+"},
    {"id": "af_sarah", "name": "Sarah", "gender": "female", "accent": "American", "grade": "C+"},
]
VOICE_IDS = [v["id"] for v in VOICES]

_lock = threading.Lock()
_M = {"tts": None, "stt": None, "speakers": {}, "lexicon": None}


def available():
    """Whether the voice models are installed (they live outside the repository, like the chat models)."""
    return all(p.exists() for p in (KOKORO / "model.onnx", PARAKEET / "encoder.int8.onnx", VAD_MODEL))


def _speaker_ids():
    """Kokoro's voice names, read from the model's own metadata (speaker_names, in id order)."""
    data = (KOKORO / "model.onnx").read_bytes()
    i = data.find(b"speaker_names")
    m = re.search(rb"speaker_names.{1,8}?([a-z_,]{100,})", data[i:i + 2000], re.S) if i >= 0 else None
    names = m.group(1).decode().split(",") if m else []
    return {n: k for k, n in enumerate(names)}


def load_tts(voice=DEFAULT_VOICE):
    """Load the voice with the pronunciation dictionary of its accent (American or British): with both,
    the first dictionary wins on every shared word, so British voices would get American pronunciations."""
    import sherpa_onnx
    lexicon = "lexicon-gb-en.txt" if str(voice).startswith("b") else "lexicon-us-en.txt"
    with _lock:
        if _M["tts"] is None or _M["lexicon"] != lexicon:
            _M["tts"] = None
            _M["tts"] = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=str(KOKORO / "model.onnx"), voices=str(KOKORO / "voices.bin"),
                        tokens=str(KOKORO / "tokens.txt"), data_dir=str(KOKORO / "espeak-ng-data"),
                        lexicon=str(KOKORO / lexicon)),
                    num_threads=4, provider="cpu"),
                max_num_sentences=1))
            _M["lexicon"] = lexicon
            _M["speakers"] = _M["speakers"] or _speaker_ids()


def load(voice=DEFAULT_VOICE):
    """Load the recogniser and the voice (once; a few seconds)."""
    import sherpa_onnx
    load_tts(voice)
    with _lock:
        if _M["stt"] is None:
            _M["stt"] = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(PARAKEET / "encoder.int8.onnx"), decoder=str(PARAKEET / "decoder.int8.onnx"),
                joiner=str(PARAKEET / "joiner.int8.onnx"), tokens=str(PARAKEET / "tokens.txt"),
                num_threads=4, model_type="nemo_transducer")


def unload():
    with _lock:
        _M["tts"] = _M["stt"] = _M["lexicon"] = None


def loaded():
    return _M["tts"] is not None and _M["stt"] is not None


def transcribe(samples):
    """Float32 16 kHz mono -> text."""
    rec = _M["stt"]
    s = rec.create_stream()
    s.accept_waveform(RATE, samples)
    rec.decode_stream(s)
    return s.result.text.strip()


# ---- speaking: text as a person would say it
_MD = [(r"```.*?```", " "), (r"`([^`]*)`", r"\1"), (r"\[([^\]]+)\]\([^)]+\)", r"\1"), (r"https?://\S+", "a link"),
       (r"^\s*#+\s*", ""), (r"^\s*[-*+]\s+", ""), (r"^\s*\d+\.\s+", ""), (r"[*_~]{1,3}", ""), (r"\$+", ""),
       (r"\s*\|\s*", ", "), (r"[<>{}\[\]\\^]", " ")]


def speakable(text):
    """Markdown, code, links and symbols out; what remains reads naturally aloud."""
    t = str(text or "")
    for pat, rep in _MD:
        t = re.sub(pat, rep, t, flags=re.S | re.M)
    t = t.replace("&", " and ").replace("%", " percent").replace("°C", " degrees Celsius").replace("°F", " degrees Fahrenheit")
    return re.sub(r"\s+", " ", t).strip()


def speak(text, voice=DEFAULT_VOICE, speed=1.0):
    """Text -> WAV bytes (24 kHz mono, 16-bit), or None if there's nothing to say."""
    t = speakable(text)
    if not t or not re.search(r"\w", t):
        return None
    load_tts(voice)  # no-op unless the voice's accent needs the other dictionary
    sid = _M["speakers"].get(voice, _M["speakers"].get(DEFAULT_VOICE, 0))
    with _lock:
        audio = _M["tts"].generate(t, sid=sid, speed=float(speed))
    pcm = (np.clip(np.asarray(audio.samples, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(audio.sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class Listener:
    """One voice conversation: microphone audio in, events out ("speech_start" for interrupting, "transcript")."""

    def __init__(self):
        import sherpa_onnx
        cfg = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(VAD_MODEL), threshold=0.5, min_silence_duration=0.6, min_speech_duration=0.25,
                max_speech_duration=25, window_size=WINDOW),
            sample_rate=RATE, num_threads=1, provider="cpu")
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
        self.events = queue.Queue()
        self.pending = np.zeros(0, dtype=np.float32)
        self.in_speech = False
        self.closed = False
        self.work = queue.Queue()
        threading.Thread(target=self._transcriber, daemon=True).start()

    def feed(self, pcm16):
        """Raw 16-bit little-endian mono 16 kHz audio from the browser."""
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768
        self.pending = np.concatenate([self.pending, x])
        while len(self.pending) >= WINDOW:
            self.vad.accept_waveform(self.pending[:WINDOW])
            self.pending = self.pending[WINDOW:]
            speaking = self.vad.is_speech_detected()
            if speaking and not self.in_speech:
                self.events.put({"type": "speech_start"})
            self.in_speech = speaking
            while not self.vad.empty():
                self.work.put(np.array(self.vad.front.samples, dtype=np.float32))
                self.vad.pop()

    def _transcriber(self):
        while not self.closed:
            try:
                seg = self.work.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                text = transcribe(seg)
            except Exception as e:  # report, keep listening
                self.events.put({"type": "error", "error": f"speech recognition failed: {e}"})
                continue
            if re.search(r"\w", text):
                self.events.put({"type": "transcript", "text": text, "seconds": round(len(seg) / RATE, 2)})
            else:
                self.events.put({"type": "speech_end"})  # a cough or noise: nothing to answer

    def close(self):
        self.closed = True
        self.events.put({"type": "closed"})
