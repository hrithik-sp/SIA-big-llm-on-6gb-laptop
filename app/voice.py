"""Voice mode: Sia listens and speaks, fully offline, with the same chat model behind it.

A cascaded pipeline, as in Kyutai's Unmute: the user's speech -> text -> the usual chat model -> speech.
(End-to-end speech models such as Moshi, arXiv 2410.00037, answer faster but are separate 7B GPU
models; they would not fit beside the 26B chat model, and the user wants Sia's own model.)
All three parts run on the CPU through sherpa-onnx (k2-fsa, Apache-2.0), measured on this laptop:

- Silero VAD (MIT): finds where speech starts and ends (well under 1 ms per 32 ms chunk).
- NVIDIA Parakeet TDT 110M (English), int8 (CC-BY-4.0): speech to text in 0.06-0.10 s for a short
  sentence, 166 MB. Compared on the same clips (American, British and Indian-accented English, a Hindi
  sentence): Parakeet 0.6B v3 was as accurate but used 669 MB, which pushed the 26B chat model into
  SSD paging (replies 5-11 s instead of ~2 s); Moonshine base invented a sentence for the Hindi clip;
  Whisper-large-v3-turbo was 5x slower than Parakeet 0.6B, used 1.3 GB and garbled Hindi.
- Kokoro-82M v1.0 (Apache-2.0): text to speech, 24 kHz; 6.3 s of speech in 2.4 s on 4 threads
  (fp32). The int8 export was slower than real time on this CPU, so it isn't used.

The browser captures the microphone with echo cancellation, so the user can interrupt Sia while
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
PREROLL = int(0.3 * RATE)  # audio kept from just before detected speech
HEARD_MAX = 30 * RATE  # recent audio kept for that
KWS = VOICE_DIR / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
WAKE_FILE = Path(__file__).resolve().parent / "data" / "wake_words.txt"  # Hey/Hi/OK/Okay/Hello Sia, as BPE pieces
HOTWORDS = Path(__file__).resolve().parent / "data" / "hotwords.txt"  # words the recogniser is nudged towards
HOTWORD_SCORE = 1.5
WAKE_LEAD = 2 * RATE  # a wake hit this long after a segment ends still counts for it (decoding lags a little)
# "Hey Sia" as the recogniser writes it. "Sia" is short and sounds like "see ya", so the keyword spotter alone was
# weak (10/30 caught) or, with more spellings, fired on "Hey, see you tomorrow" (16/66). With the recogniser nudged
# towards "sia" (HOTWORDS), this rule on its transcript caught 40/48 spoken wake phrases in six voices, with 1 false
# alarm in 66 other phrases ("Okay, see ya", which sounds the same). "Sir" counts only when the spotter also heard it.
_GREET = r"^\W*(?:(?:hey|hi|hay|he|ok(?:ay)?|hello|yo)\W+)"
WAKE_TEXT = re.compile(
    rf"(?:{_GREET}(?:sia|siya|siah|sier|siera|sierra(?!\W+leone)|cia|xia|zia)|^\W*(?:sia|siya|siah))\b(?!\W+(?:is|was)\b)"
    r"[\s,.!?]*", re.I)
WAKE_SIR = re.compile(rf"{_GREET}sir\b[\s,.!?]*", re.I)


def wake_split(text, spotted=False):
    """(was the wake phrase said, the request after it). spotted: the keyword spotter heard it in this segment."""
    m = WAKE_TEXT.match(text) or (spotted and WAKE_SIR.match(text))
    return (True, text[m.end():].strip()) if m else (False, text.strip())
FOLLOW_UP_S = 8  # after the wake phrase, or after Sia answers: speech without the wake phrase is still for Sia
DEFAULT_VOICE = "af_heart"  # the owner's choice; also Kokoro's top-graded voice (A)
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


NAMES = {"Sia": "sˈiə", "Hrithik": "ɹˈɪθɪk"}  # misaki phonemes for names (the H in Hrithik is silent)


class _Spoken:
    def __init__(self, samples, sample_rate):
        self.samples, self.sample_rate = samples, sample_rate


class _MisakiKokoro:
    """Kokoro, with its pronunciation chosen by misaki (hexgrad/misaki, Kokoro's own G2P, Apache-2.0), which
    tags each word's part of speech with spaCy before picking a pronunciation. The sherpa-onnx path looks each
    word up in a dictionary with one pronunciation, so heteronyms were a coin toss: "I live here" was read
    "laiv", "I read it yesterday" "reed". On tools/heteronym_cases.py the dictionary got 17/32 right and misaki
    29/32. The same model file and voices run directly in onnxruntime; it is also faster (RTF 0.28 vs 0.38)."""

    def __init__(self, british):
        import onnxruntime as ort
        from misaki import en, espeak
        self.g2p = en.G2P(trf=False, british=british, fallback=espeak.EspeakFallback(british=british))
        for word, sounds in NAMES.items():  # names it has never seen ("Sia" came out as "zhuh")
            for form in (word, word.lower(), word.upper()):
                self.g2p.lexicon.golds[form] = sounds
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("SIA_TTS_THREADS", "4"))
        self.sess = ort.InferenceSession(str(KOKORO / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self.vocab = {}
        for line in (KOKORO / "tokens.txt").read_text(encoding="utf-8").splitlines():
            sym, _, idx = line.rpartition(" ")
            if idx.isdigit():
                self.vocab[sym or " "] = int(idx)
        self.styles = np.fromfile(KOKORO / "voices.bin", dtype=np.float32).reshape(-1, 510, 1, 256)

    def phonemes(self, text):
        return self.g2p(text)[0]

    def generate(self, text, sid=0, speed=1.0):
        ids = [self.vocab[c] for c in self.phonemes(text) if c in self.vocab]
        out = []
        for i in range(0, len(ids), 500):  # the model takes up to 510 tokens at a time
            part = ids[i:i + 500]
            audio = self.sess.run(None, {"tokens": np.array([[0, *part, 0]], dtype=np.int64),
                                         "style": self.styles[sid, min(len(part), 509)],
                                         "speed": np.array([speed], dtype=np.float32)})[0]
            out.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        return _Spoken(np.concatenate(out) if out else np.zeros(0, np.float32), 24000)


def phonemes(text, voice=DEFAULT_VOICE):
    """How Sia will pronounce the text (misaki phonemes), or None without misaki. For tests."""
    load_tts(voice)
    return _M["tts"].phonemes(text) if isinstance(_M["tts"], _MisakiKokoro) else None


def load_tts(voice=DEFAULT_VOICE):
    """Load the voice for its accent (American or British). misaki when it's installed; otherwise sherpa-onnx with
    the accent's pronunciation dictionary (with both dictionaries, the first wins on every shared word)."""
    import sherpa_onnx
    lexicon = "lexicon-gb-en.txt" if str(voice).startswith("b") else "lexicon-us-en.txt"
    with _lock:
        if _M["tts"] is None or _M["lexicon"] != lexicon:
            _M["tts"] = None
            if os.environ.get("SIA_G2P", "misaki") == "misaki":
                try:
                    _M["tts"] = _MisakiKokoro(british=lexicon.startswith("lexicon-gb"))
                    _M["lexicon"] = lexicon
                    _M["speakers"] = _M["speakers"] or _speaker_ids()
                    _M["tts"].generate("Okay.", sid=0)  # the first synthesis is slow (set-up): do it now
                    return
                except Exception:  # misaki or onnxruntime missing or broken: the dictionary voice still works
                    _M["tts"] = None
            _M["tts"] = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=str(KOKORO / "model.onnx"), voices=str(KOKORO / "voices.bin"),
                        tokens=str(KOKORO / "tokens.txt"), data_dir=str(KOKORO / "espeak-ng-data"),
                        lexicon=str(KOKORO / lexicon)),
                    # 4 threads: the first sentence is ready ~0.3 s sooner than with 2. Fewer threads didn't make the
                    # model write faster while Sia speaks (finding 64), and it writes ~6x faster than speech anyway.
                    num_threads=int(os.environ.get("SIA_TTS_THREADS", "4")), provider="cpu"),
                max_num_sentences=1))
            _M["lexicon"] = lexicon
            _M["speakers"] = _M["speakers"] or _speaker_ids()
            _M["tts"].generate("Okay.", sid=0, speed=1.0)  # the first synthesis is slow (engine set-up): do it now


def kws_available():
    return (KWS / "tokens.txt").exists() and WAKE_FILE.exists()


def _spotter():
    """The wake-phrase spotter (sherpa-onnx Zipformer KWS, GigaSpeech, 3.3M parameters, ~1.5% of one CPU core).
    It reacts within the phrase, so Sia's voice can pause at once; the transcript then decides (WAKE_TEXT), because
    on its own the spotter caught only 10 of 30 spoken "Hey Sia"s, and with more spellings it fired on "Hey, see you
    tomorrow". (For the earlier name it caught 29/30 "Hey Sprout"s with no false alarms: longer names spot better.)"""
    import sherpa_onnx
    with _lock:
        if _M.get("kws") is None:
            _M["kws"] = sherpa_onnx.KeywordSpotter(
                tokens=str(KWS / "tokens.txt"), encoder=str(KWS / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
                decoder=str(KWS / "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
                joiner=str(KWS / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"), num_threads=1,
                keywords_file=str(WAKE_FILE), keywords_score=2.0, keywords_threshold=0.2, max_active_paths=4,
                num_trailing_blanks=1, provider="cpu")
        return _M["kws"]


def load(voice=DEFAULT_VOICE):
    """Load the recogniser and the voice (once; a few seconds)."""
    import sherpa_onnx
    load_tts(voice)
    with _lock:
        if _M["stt"] is None:
            _M["stt"] = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(PARAKEET / "encoder.int8.onnx"), decoder=str(PARAKEET / "decoder.int8.onnx"),
                joiner=str(PARAKEET / "joiner.int8.onnx"), tokens=str(PARAKEET / "tokens.txt"),
                num_threads=4, model_type="nemo_transducer", **_hotword_args())


def _hotword_args():
    """Contextual biasing (sherpa-onnx hotwords: modified beam search plus an Aho-Corasick graph of the hotwords'
    pieces) so the recogniser writes "Sia" instead of "see ya" / "sir". It needs a BPE vocabulary; Parakeet ships
    only tokens.txt, so one is written from it (piece, then a score that favours earlier, more frequent pieces).
    +12 ms per 3.8 s sentence."""
    vocab = PARAKEET / "bpe.vocab"
    try:
        if not vocab.exists():
            with open(PARAKEET / "tokens.txt", encoding="utf-8") as f:
                rows = [line.rstrip("\n").rsplit(" ", 1) for line in f if line.strip()]
            vocab.write_text("".join(f"{tok}\t{-float(i)}\n" for tok, i in rows), encoding="utf-8")
    except OSError:
        return {}
    return dict(decoding_method="modified_beam_search", max_active_paths=4, hotwords_file=str(HOTWORDS),
                hotwords_score=HOTWORD_SCORE, modeling_unit="bpe", bpe_vocab=str(vocab))


def unload():
    with _lock:
        _M["tts"] = _M["stt"] = _M["lexicon"] = _M["kws"] = None


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
    """One voice conversation: microphone audio in, events out.

    Events: "wake" (the keyword spotter heard the wake phrase: a quick guess), "speech_start" (speech probably
    addressed to Sia began: used for pausing its voice), "transcript" (what was said, with "addressed": whether it
    was meant for Sia, "wake" and "request": the words after the wake phrase).

    mode "wake": only speech that starts with the wake phrase ("Hey Sia", read from the transcript, see WAKE_TEXT)
    or starts in the follow-up window is addressed to Sia.
    Other people talking nearby is still transcribed, so the page can recognise "stop", but never answered.
    mode "always": everything heard is addressed to Sia (the earlier behaviour; picks up nearby conversations).
    """

    def __init__(self, mode="wake"):
        import sherpa_onnx
        cfg = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(VAD_MODEL), threshold=0.5, min_silence_duration=0.6, min_speech_duration=0.25,
                max_speech_duration=25, window_size=WINDOW),
            sample_rate=RATE, num_threads=1, provider="cpu")
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
        self.mode = mode if mode == "always" or kws_available() else "always"
        self.kws = self.kws_stream = None
        if self.mode == "wake":
            self.kws = _spotter()
            self.kws_stream = self.kws.create_stream()
        self.events = queue.Queue()
        self.pending = np.zeros(0, dtype=np.float32)
        self.heard, self.heard_from = np.zeros(0, dtype=np.float32), 0  # recent audio (for pre-roll) and its offset
        self.fed = 0  # samples received so far: the clock for wake hits and the follow-up window
        self.wake_hits = []  # sample positions where the wake phrase was heard
        self.awake_until = 0  # follow-up window: speech starting before this sample is addressed to Sia
        self.in_speech = False
        self.closed = False
        self.work = queue.Queue()
        threading.Thread(target=self._transcriber, daemon=True).start()

    def awake(self, seconds):
        """Open (or close, with 0) the follow-up window: speech starting in it needs no wake phrase."""
        self.awake_until = self.fed + int(seconds * RATE) if seconds > 0 else 0

    def _spotted(self, start, end):
        return any(start - RATE <= h <= end + WAKE_LEAD for h in self.wake_hits)

    def _addressed(self, start, end):
        """A first guess, while the speech is still going (to pause Sia's voice): follow-up window or spotter hit.
        The transcript decides (_transcriber)."""
        return self.mode == "always" or start < self.awake_until or self._spotted(start, end)

    def feed(self, pcm16):
        """Raw 16-bit little-endian mono 16 kHz audio from the browser."""
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768
        self.pending = np.concatenate([self.pending, x])
        self.heard = np.concatenate([self.heard, x])
        if len(self.heard) > HEARD_MAX:
            self.heard_from += len(self.heard) - HEARD_MAX
            self.heard = self.heard[-HEARD_MAX:]
        start_of_chunk = self.fed
        self.fed += len(x)
        if self.kws:
            self.kws_stream.accept_waveform(RATE, x)
            while self.kws.is_ready(self.kws_stream):
                self.kws.decode_stream(self.kws_stream)
                if self.kws.get_result(self.kws_stream):
                    self.kws.reset_stream(self.kws_stream)
                    self.wake_hits = [h for h in self.wake_hits if h > self.fed - 60 * RATE] + [self.fed]
                    self.events.put({"type": "wake"})  # tentative: Sia's voice pauses until the transcript decides
        done = 0
        while len(self.pending) >= WINDOW:
            self.vad.accept_waveform(self.pending[:WINDOW])
            self.pending = self.pending[WINDOW:]
            done += WINDOW
            speaking = self.vad.is_speech_detected()
            now = start_of_chunk + done
            if speaking and not self.in_speech and self._addressed(now, now):
                self.events.put({"type": "speech_start"})
            self.in_speech = speaking
            while not self.vad.empty():
                seg = self.vad.front
                samples = np.array(seg.samples, dtype=np.float32)
                end = seg.start + len(samples)
                self.work.put((self._with_preroll(seg.start, samples),
                               self.mode == "always" or seg.start < self.awake_until, self._spotted(seg.start, end)))
                self.vad.pop()

    def _with_preroll(self, start, samples):
        """The segment plus the PREROLL before it: the detector marks speech a little late, and without this the
        first syllable was lost ("What's a good way to start my morning?" was heard as "That's a good way ...")."""
        a = max(start - PREROLL, self.heard_from)
        if a >= start or start - self.heard_from > len(self.heard):
            return samples
        return np.concatenate([self.heard[a - self.heard_from:start - self.heard_from], samples])

    def _transcriber(self):
        while not self.closed:
            try:
                seg, in_window, spotted = self.work.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                text = transcribe(seg)
            except Exception as e:  # report, keep listening
                self.events.put({"type": "error", "error": f"speech recognition failed: {e}"})
                continue
            if re.search(r"\w", text):
                woke, request = wake_split(text, spotted) if self.mode == "wake" else (False, text)
                if woke and not request:
                    self.awake(FOLLOW_UP_S)  # just "Hey Sia": the request follows after a pause
                self.events.put({"type": "transcript", "text": text, "addressed": in_window or woke, "wake": woke,
                                 "request": request, "seconds": round(len(seg) / RATE, 2)})
            else:
                self.events.put({"type": "speech_end"})  # a cough or noise: nothing to answer

    def close(self):
        self.closed = True
        self.events.put({"type": "closed"})
