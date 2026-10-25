# Changelog

## v1.2.0 (prepared, not yet released): Sia acts on your laptop

### New: Sia runs your laptop, with your approval
- **Open** apps by name (anything in the Start menu), documents, folders, websites and Settings pages.
- **Control** volume and mute, brightness, Bluetooth, Wi-Fi, dark/light theme, and music or video.
  "Make it a bit louder" and "dim the screen a little" work too.
- **Windows:** bring to front, minimise, maximise, close.
- **Power:** lock; sleep, restart and shutdown (asked every time; restart and shutdown wait 60 s and
  can be cancelled).
- **Screenshots, clipboard, notes, reminders and timers** (they pop up as Windows notifications with a
  sound; "10 seconds" means 10 seconds).
- **Exact calculator:** every calculation, from `11*2` to a 49-digit product, compound interest, doses or
  engineering formulas, goes through an exact calculator instead of the model's guess. Whole numbers of
  any length are exact, decimals have no rounding error, and maths functions use 40 digits.
- **Faster first reply:** the pre-read instructions are saved and restored instead of re-read at every
  start or wake: a message sent right away now gets its first word in about 21 s instead of 47 s.
- **New name: Sia** (formerly Sprout): window, shortcuts, Explorer menu ("Ask Sia") and wake phrase.
- **Says words the right way by meaning:** "share your live location" vs "I live here", "read" now and
  yesterday, "close the door" vs "close to home", and names like Sia and Hrithik.
- **"Hey Sia":** in voice mode Sia answers only when spoken to (wake phrase, then a short follow-up
  window), ignores other people in the room, and has a pause button.
- **Knows the world again:** general-knowledge questions (famous people, history, sport) are answered
  instead of refused.
- **Voice mode:** talk to Sia and it answers aloud in the same window, with both sides in the chat.
  It runs fully offline on the CPU beside the same model. There are 12 natural male and female voices
  with Preview in Settings, and you can interrupt it by talking.
- **Sia knows who made it:** "Who created you?" now answers with the owner's name, not Google.
- **Python sandbox:** for anything that needs a little program (dates, durations, conversions,
  sorting, counting, text), Sia writes and runs Python in a locked-down sandbox (Pydantic Monty):
  no files, internet or programs, 5 seconds at most.
- **Plug-ins and memory:** Sia can use MCP plug-ins, the common plug-in format for AI tools, with
  its own approval rules on each. The first is **Memory**: Sia remembers lasting facts you tell
  it across chats. Each save is approved, and "forget …" shows exactly what will go.
- **Undo and cancel:** "undo that", "put the volume back", "cancel the tea timer", "never mind the
  shutdown" or "clear all my reminders" reverse what Sia did earlier. The card says exactly what will be
  undone. Files are never deleted; for a screenshot or note Sia gives its path.
- **Security:**
  - an approval card for every action, saying exactly what will happen;
  - fixed, checked actions, with no scripts, shells or programs run from files;
  - nothing deleted or overwritten;
  - a separate Settings switch, and a log of every action.
- **Tested like a user:** 32 real requests, each verified against the laptop's real state and then
  restored (32/32).

### Fixed
- **Bluetooth status was wrong:** it reported "on" whenever an adapter existed. It now reads the radio
  itself.
- **Announced actions without doing them** ("Bluetooth is now off" with no action). Commands now always
  go through a tool.
- **A declined action reported as done** ("Shutdown cancelled." after you declined the card). When you
  decline, Sia now says "Okay, not done" in fixed words.
- **Timers:** a 10-second timer became 10 minutes. The duration you type now wins.
- **Wrong arithmetic:** 876 × 88894 was answered 77,868,972 (it's 77,871,144). Now calculated exactly;
  simple sums no longer pause to "think" (11 × 2: 23 s → 3 s).
- **Reads more:** what's open in each window (e.g. the file in VS Code), installed games, and Sia's
  own memory use labelled as Sia.

### Tested on everyday use

A wider test of 33 everyday requests found eleven more problems, and all are fixed. The final full
run passed 32 of 33; its one failure (a game guess the model made up) was then fixed and re-tested.

### Fixed
- **No more invented actions.** Sia sometimes said "I checked your files" or "I'll keep
  searching" without doing anything. It now searches for real or says it doesn't know
  (0 of 15 invented searches in a targeted test).
- **Remembers what you said** earlier in the chat.
- **Long chats no longer break.** Beyond the model's 16k-token memory, the oldest messages are
  left out instead of the request failing; a single message that's too long gets a clear
  explanation.
- **Games keep their state:** no restart on the first guess, real hints, "play again" starts a
  fresh game, and every letter's feedback is reported.
- **Knows what it is:** "a Gemma 4 26B (QAT) model running entirely offline on this laptop".
- **Drafted emails are signed with your name**, not "[Your Name]".
- **Lookups stay fast:** "how many tabs?" no longer triggers reasoning.
- **Laptop access off:** games still work, and laptop questions explain how to turn access on.
- **Natural replies:** it no longer comments on Sia's automatic clock note.
- **Browser tabs:** reports a browser running in the background with no windows.
- **Games are fair:** the model can't submit a guess you didn't make.
- **Replies in your script:** a question in Hindi (Devanagari) gets a Hindi answer.

### Checks
- New: `tools/sia_daily_check.py` (33 requests), `tools/sia_actions_check.py` (32 actions, run with
  `--visible`). 72 unit tests. Lab notebook findings 57–58.

## v1.1.0: Sia knows your laptop, and works for you

Sia v1.0 could chat, but in daily use it fell short: it made up the time, couldn't say where
the laptop was, felt like a chatty friend instead of an assistant, took 12–16 s per reply, and
played word games badly. This update fixes that. Every change was checked with the real model:

- a ten-question laptop test: **3 → 10 correct**;
- an eight-conversation behaviour test: **3/8 → 8/8 in three runs**.

### New
- **It knows the time.** Every message carries the laptop's date, time, time zone and region.
  "What time is it?" is answered from the clock, never guessed.
- **It knows where you are.** A location tool uses Windows location (±150 m) and names the place
  offline from a built-in list of 34,179 places (GeoNames), e.g. "Bengaluru, about 10 km
  north-east of the city centre". Nothing is sent anywhere by Sia.
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
- **Fair word games.** Wordle and Hangman run on a real referee: Sia's code picks and keeps
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
  - `tools/sia_behaviour_check.py`: behaviour, 8 scripted conversations;
  - `tools/sia_latency_check.py`: reply speed and cache reuse;
  - `tools/sia_tools_check.py` now takes your own questions.
- 65 unit tests (was 57); they also pass on machines without a GPU.
- Lab notebook findings 55–56; REPORT problems 27–32.

### Known limits
- The first message after Sia loads its model still takes about 40 s (the model is paged in
  from the SSD). Later messages are fast.
- Location needs Windows location turned on (Settings > Privacy & security > Location).

## v1.0.0: first public release

- The research: Gemma 4 26B and gpt-oss-20b on an RTX 3050 6 GB laptop. Released-precision
  models, output-preserving speed-ups, controlled benchmarks and paired accuracy tests
  (54 findings).
- Sia, the offline desktop app: 8 modes, a standalone window, the `Ctrl+Alt+Space` hotkey,
  read-only laptop tools with per-call permission.
