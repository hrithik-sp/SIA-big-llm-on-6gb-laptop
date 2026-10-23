# Changelog

## v1.1.0: Sprout knows your laptop, and works for you

Sprout v1.0 could chat, but in daily use it fell short: it made up the time, couldn't say where
the laptop was, felt like a chatty friend instead of an assistant, took 12–16 s per reply, and
played word games badly. This update fixes that. Every change was checked with the real model:

- a ten-question laptop test: **3 → 10 correct**;
- an eight-conversation behaviour test: **3/8 → 8/8 in three runs**.

### New
- **It knows the time.** Every message carries the laptop's date, time, time zone and region.
  "What time is it?" is answered from the clock, never guessed.
- **It knows where you are.** A location tool uses Windows location (±150 m) and names the place
  offline from a built-in list of 34,179 places (GeoNames), e.g. "Bengaluru, about 10 km
  north-east of the city centre". Nothing is sent anywhere by Sprout.
- **It reads your devices.** One tool covers:
  - battery health (75% here: design 70.1 Wh, now 52.8 Wh) and time left;
  - display (resolution, refresh rate, brightness);
  - volume and mute;
  - Bluetooth devices and whether they're connected;
  - USB devices and printers;
  - apps that start with Windows;
  - internet connection, Wi-Fi name and signal;
  - the laptop model.
- **Recently opened files**, not just recent downloads.
- **Fair word games.** Wordle and Hangman run on a real referee: Sprout's code picks and keeps
  the secret word and scores every guess with the official rules, so feedback is always
  consistent and the word never leaks.

### Better
- **An assistant, not a buddy.** Chat mode now behaves like a sharp personal assistant:
  - it works for you (you're the boss);
  - replies are brief and natural, with no filler laughter, no gushing and no long apologies;
  - it knows your name but doesn't greet you with it every time.
- **Gets things done in one go.** Asked to find a file, it searches once with every likely
  spelling (`aadhaar|aadhar`), looks in your personal folders first, uses your real folder
  paths, and reports what it found instead of asking "should I?".
- **Thinks when it needs to.** Questions with numbers, plans or comparisons get a short burst of
  reasoning (about 10 s); everyday chat stays instant. The Think switch still forces full
  reasoning.
- **No more invented facts.** It's told never to guess a fact about the laptop: it checks with a
  tool or says it can't see it.

### Fixed
- **Slow replies (12–16 s → 1.5–2.5 s).** A clock line at the top of the instructions changed
  every minute and forced the whole conversation to be re-read. The time now rides on the newest
  message, so the cached conversation is reused.
- **Empty answers.** After several tool lookups, an answer could come back blank. The last step
  now always produces an answer.
- **Wi-Fi questions** take one lookup instead of two.

### Under the hood
- New: `app/devices.ps1` (read-only Windows device queries), `app/games.py` (the referee),
  `app/data/places.tsv` (GeoNames, CC BY 4.0).
- New checks:
  - `tools/sprout_behaviour_check.py`: behaviour, 8 scripted conversations;
  - `tools/sprout_latency_check.py`: reply speed and cache reuse;
  - `tools/sprout_tools_check.py` now takes your own questions.
- 65 unit tests (was 57); they also pass on machines without a GPU.
- Lab notebook findings 55–56; REPORT problems 27–32.

### Known limits
- The first message after Sprout loads its model still takes about 40 s (the model is paged in
  from the SSD). Later messages are fast.
- Location needs Windows location turned on (Settings > Privacy & security > Location).

## v1.0.0: first public release

- The research: Gemma 4 26B and gpt-oss-20b on an RTX 3050 6 GB laptop. Released-precision
  models, output-preserving speed-ups, controlled benchmarks and paired accuracy tests
  (54 findings).
- Sprout, the offline desktop app: 8 modes, a standalone window, the `Ctrl+Alt+Space` hotkey,
  read-only laptop tools with per-call permission.
