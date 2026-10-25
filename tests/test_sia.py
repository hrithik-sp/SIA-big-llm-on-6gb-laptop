"""Tests for app/server.py logic that needs no model: chunking, extraction, request building."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
import server  # noqa: E402


class Chunking(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(server.chunk([(None, "hello world")]), [{"page": None, "text": "hello world"}])

    def test_long_text_overlaps_and_covers_everything(self):
        text = " ".join(f"word{i}." for i in range(3000))
        chunks = server.chunk([(5, text)], size=1000, overlap=100)
        self.assertGreater(len(chunks), 10)
        self.assertTrue(all(c["page"] == 5 and len(c["text"]) <= 1000 for c in chunks))
        self.assertIn("word0.", chunks[0]["text"])
        self.assertIn("word2999.", chunks[-1]["text"])

    def test_empty_pages_are_dropped(self):
        self.assertEqual(server.chunk([(1, "   "), (2, "")]), [])


class Extraction(unittest.TestCase):
    def test_text_and_code(self):
        self.assertEqual(server.extract_text("a.py", b"print(1)\n"), [(None, "print(1)\n")])

    def test_unsupported(self):
        with self.assertRaises(ValueError):
            server.extract_text("movie.mp4", b"\x00")


class Requests(unittest.TestCase):
    msgs = [{"role": "user", "content": "hi"}]

    def test_gemma_thinking_flag_and_official_sampling(self):
        body, model, _ = server.build_request({"mode": "study", "messages": self.msgs, "think": False})
        self.assertEqual(model, "gemma")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual((body["temperature"], body["top_k"], body["top_p"]), (1.0, 64, 0.95))
        self.assertTrue(body["messages"][0]["content"].startswith("You are Sia"))
        self.assertTrue(body["stream"])

    def test_think_mode_uses_gptoss_high_effort(self):
        body, model, _ = server.build_request({"mode": "think", "messages": self.msgs})
        self.assertEqual(model, "gptoss")
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertNotIn("chat_template_kwargs", body)

    def test_unknown_model_and_mode_fall_back(self):
        body, model, _ = server.build_request({"mode": "nope", "model": "evil", "messages": self.msgs})
        self.assertIn(model, server.PRESETS)

    def test_image_switches_to_vision_preset(self):
        img = [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
        body, model, _ = server.build_request({"mode": "see", "messages": img})
        self.assertEqual(model, "gemma-vision")
        self.assertIn("mmproj", server.PRESETS["gemma-vision"])
        self.assertNotIn("mmproj", server.PRESETS["gemma"])   # text chats keep VRAM for experts
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        with self.assertRaises(ValueError):
            server.build_request({"mode": "think", "messages": img})  # gpt-oss has no vision

    def test_every_mode_is_sia(self):
        for m in server.MODES["modes"]:
            self.assertIn("Sia", m["system"], m["id"])
            self.assertIn(m["model"], server.PRESETS, m["id"])


class ChatNoEmoji(unittest.TestCase):
    def test_strip_emoji_keeps_words_and_other_scripts(self):
        self.assertEqual(server.strip_emoji("haha 😂 that's great 👍🏽!"), "haha that's great !")
        self.assertEqual(server.strip_emoji("❤️ thanks"), " thanks")
        self.assertEqual(server.strip_emoji("नमस्ते, kaise ho? x² ≤ 5 → done"), "नमस्ते, kaise ho? x² ≤ 5 → done")

    def test_chunk_filter(self):
        chunk = '{"choices":[{"delta":{"content":"nice 🎉"}}]}'.encode()
        out = __import__("json").loads(server.strip_emoji_chunk(chunk))
        self.assertEqual(out["choices"][0]["delta"]["content"], "nice ")
        self.assertEqual(server.strip_emoji_chunk(b"not json"), b"not json")

    def test_only_chat_mode_is_messenger(self):
        styles = {m["id"]: m.get("style") for m in server.MODES["modes"]}
        self.assertEqual([k for k, v in styles.items() if v == "messenger"], ["chat"])
        self.assertIn("Never use emojis", server.MODE_BY_ID["chat"]["system"])


class SettingsAndTools(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.orig = server.SETTINGS_FILE
        server.SETTINGS_FILE = server.Path(self.tmp.name) / "settings.json"
        server.tools.games.USER_TEXT["text"] = None  # direct referee calls: no user message to check against

    def tearDown(self):
        server.SETTINGS_FILE = self.orig
        self.tmp.cleanup()

    msgs = [{"role": "user", "content": "how many tabs do I have open?"}]

    def test_always_uncensored_in_every_mode(self):
        # an old saved level from before the toggle was removed must be ignored
        server.SETTINGS_FILE.write_text('{"content_level": "safe"}')
        self.assertNotIn("content_level", server.load_settings())
        for m in server.MODES["modes"]:
            msgs = self.msgs if not m.get("requires_image") else self.msgs
            body, _, _ = server.build_request({"mode": m["id"], "messages": msgs})
            self.assertIn(server.UNCENSORED, body["messages"][0]["content"], m["id"])

    def test_tools_in_request(self):
        body, _, _ = server.build_request({"mode": "chat", "messages": self.msgs})
        self.assertEqual({t["function"]["name"] for t in body["tools"]}, set(server.tools.TOOLS))
        lib, _, _ = server.build_request({"mode": "library", "messages": self.msgs})
        self.assertNotIn("tools", lib)  # Library answers only from the user's files

    def test_owner_name_makes_them_the_boss(self):
        server.save_settings({**server.DEFAULT_SETTINGS, "user_name": "Hrithik S P"})
        body, _, _ = server.build_request({"mode": "code", "messages": self.msgs})
        system = body["messages"][0]["content"]
        self.assertIn("Hrithik S P", system)
        self.assertIn("is your boss", system)
        self.assertIn("not a friend", system)
        self.assertIn("use it rarely", system)  # knows the name, doesn't greet with it every time
        server.save_settings(dict(server.DEFAULT_SETTINGS))
        body, _, _ = server.build_request({"mode": "code", "messages": self.msgs})
        self.assertNotIn("About the user", body["messages"][0]["content"])

    def test_laptop_access_off_removes_tools(self):
        server.save_settings({**server.DEFAULT_SETTINGS, "laptop_access": False})
        body, _, _ = server.build_request({"mode": "chat", "messages": self.msgs})
        # only the game referee, the calculator and the sandbox stay: they read nothing from the laptop
        self.assertEqual([t["function"]["name"] for t in body["tools"]], ["word_game", "calculate", "run_python"])
        self.assertNotIn(server.TOOL_GUIDE, body["messages"][0]["content"])

    def test_bad_settings_are_sanitised(self):
        server.SETTINGS_FILE.write_text('{"always_allow": ["system_vitals", "rm_rf"], "x": 1}')
        st = server.load_settings()
        self.assertEqual(st["always_allow"], ["system_vitals"])
        self.assertNotIn("x", st)

    def test_no_profanity_masking(self):
        chunk = '{"choices":[{"delta":{"content":"well shit, that is damn hard"}}]}'.encode()
        self.assertEqual(server.filter_chunk(chunk), chunk)  # uncensored: text passes through unchanged

    def test_live_clock_rides_on_the_newest_message(self):
        """The time must reach the model, but never in the system prompt: that would break prompt caching."""
        orig = server.tools.now_context
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"},
                {"role": "user", "content": "what time is it?"}]
        try:
            server.tools.now_context = lambda days=False: "Right now on this laptop: 10:00"
            a, _, _ = server.build_request({"mode": "chat", "messages": msgs})
            server.tools.now_context = lambda days=False: "Right now on this laptop: 10:01"
            b, _, _ = server.build_request({"mode": "chat", "messages": msgs})
        finally:
            server.tools.now_context = orig
        self.assertEqual(a["messages"][0], b["messages"][0])  # identical system prompt -> cache reuse
        self.assertNotIn("Right now", a["messages"][0]["content"])
        last = b["messages"][-1]["content"]
        self.assertIn("never mention it: Right now on this laptop: 10:01", last)
        self.assertTrue(last.endswith("]") and "say you don't know" in last and "told you earlier" in last)  # honesty + memory reminder
        self.assertEqual(a["messages"][1:3], [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}])
        self.assertEqual(msgs[-1]["content"], "what time is it?")  # the caller's history is untouched

    def test_place_names_offline(self):
        here = server.tools.nearest_place(13.0354, 77.6540)
        self.assertEqual(here["place"], "Bengaluru, Karnataka, India")
        self.assertIn("north-east", here["where_in_city"])
        self.assertIn("near the city centre", server.tools.nearest_place(48.8566, 2.3522)["where_in_city"])
        self.assertIn("where", server.tools.nearest_place(12.40, 76.50))  # a village: nearest town instead

    def test_new_tools_validate_input(self):
        self.assertIn("error", server.tools.device_info("registry"))
        self.assertIn("recent", server.tools.TOOLS["recent_files"]["schema"]["properties"]["folder"]["enum"])
        self.assertEqual(server.tools.describe("device_info", '{"category": "battery"}'), "Read device details: battery")
        line = server.tools.now_context()
        self.assertTrue(line.startswith("Right now on this laptop: ") and "UTC" in line)

    def test_wordle_referee_scores_like_wordle(self):
        g = server.tools.games
        self.assertEqual(g._wordle_marks("doing", "proof")[1], "in word, wrong spot")  # the O it once denied
        self.assertEqual(g._wordle_marks("babes", "abbey"),  # repeated letters: greens first, then the rest
                         ["in word, wrong spot", "in word, wrong spot", "right spot", "right spot", "not in word"])
        self.assertEqual(g._wordle_marks("speed", "abide")[2:5], ["in word, wrong spot", "not in word", "in word, wrong spot"])
        start = g.word_game("start", "wordle")
        self.assertNotIn("word", start)  # never hands the secret to the model at the start
        g.GAME["word"] = "proof"
        out = g.word_game("guess", guess="doing")
        self.assertEqual(out["feedback"][1], {"letter": "O", "result": "in word, wrong spot"})
        self.assertNotIn("word", out)
        self.assertEqual(g.word_game("guess", guess="proof")["word"], "PROOF")  # revealed only when over
        self.assertTrue(all(len(w) == 5 for w in g.WORDLE_WORDS) and len(g.WORDLE_WORDS) > 300)

    def test_hangman_referee_keeps_the_board_consistent(self):
        g = server.tools.games
        g.word_game("start", "hangman")
        g.GAME["word"] = "program"
        self.assertTrue(g.word_game("guess", guess="o")["in_word"])  # the O it once denied
        out = g.word_game("guess", guess="r")
        self.assertEqual((out["board"], out["count"]), ("_ R O _ R _ _", 2))
        self.assertNotIn("word", out)
        self.assertFalse(g.word_game("guess", guess="z")["in_word"])
        self.assertEqual(g.word_game("reveal")["word"], "PROGRAM")
        self.assertEqual(server.tools.TOOLS["word_game"].get("auto"), True)

    def test_game_moves_go_to_the_referee(self):
        g = server.tools.games
        g.GAME.clear()
        self.assertTrue(g.is_game_turn("let's play wordle"))
        self.assertFalse(g.is_game_turn("guest"))  # no game running: an ordinary word
        g.word_game("start", "wordle")
        for move, forced in (("guest", True), ("o", True), ("I give up", True), ("give me a hint", True),
                             ("ok", False), ("what time is it?", False), ("thanks a lot", False)):
            self.assertEqual(g.is_game_turn(move), forced, move)
        body, _, _ = server.build_request({"mode": "chat", "messages": [{"role": "user", "content": "guest"}]})
        self.assertEqual(body.get("tool_choice"), "required")
        g.word_game("reveal")
        self.assertFalse(g.is_game_turn("guest"))  # the game is over
        self.assertTrue(g.is_game_turn("play again"))

    def test_running_game_is_visible_and_play_again_starts_fresh(self):
        g = server.tools.games
        g.GAME.clear()
        g.word_game("start", "wordle")
        g.word_game("guess", guess="crane")
        body, _, _ = server.build_request({"mode": "chat", "messages": [{"role": "user", "content": "hint please"}]})
        self.assertIn("guesses so far: CRANE", body["messages"][-1]["content"])
        self.assertIn("guesses_so_far", g.word_game("start", "wordle"))  # a stray start keeps the game
        server.build_request({"mode": "chat", "messages": [{"role": "user", "content": "play again"}]})
        self.assertNotIn("guesses_so_far", g.word_game("start", "wordle"))  # the user asked: a fresh game
        g.word_game("guess", guess="crane")
        server.build_request({"mode": "chat", "messages": [{"role": "user", "content": "play again"}]})
        out = g.word_game("reveal")  # the model picked the wrong action: the user still gets a fresh game
        self.assertTrue(g.GAME and not g.GAME["over"] and g.GAME["guesses"] == [], out)
        self.assertIn("fresh one has started", out["note"])
        g.word_game("reveal")
        body, _, _ = server.build_request({"mode": "chat", "messages": [{"role": "user", "content": "hi"}]})
        self.assertNotIn("game in progress", body["messages"][-1]["content"])

    def test_hint_and_script_follow_the_users_words(self):
        g = server.tools.games
        g.GAME.clear()
        g.word_game("start", "wordle")
        server.build_request({"mode": "chat", "messages": [{"role": "user", "content": "give me a hint"}]})
        self.assertIn("hint", g.word_game("start", "wordle"))  # a stray start still carries the hint asked for
        g.word_game("reveal")
        g.word_game("start", "wordle", new=True)
        server.build_request({"mode": "chat", "messages": [{"role": "user", "content": "ok"}]})
        self.assertIn("error", g.word_game("guess", guess="guess"))  # the model can't play for the user
        self.assertEqual(g._status()["guesses_so_far"], [])
        server.build_request({"mode": "chat", "messages": [{"role": "user", "content": "Crane!"}]})
        self.assertIn("summary", g.word_game("guess", guess="crane"))
        g.word_game("reveal")
        for text, note in (("आज मौसम कैसा है?", True), ("bhai kya haal hai", False), ("hello", False)):
            body, _, _ = server.build_request({"mode": "chat", "messages": [{"role": "user", "content": text}]})
            self.assertEqual("same language and script" in body["messages"][-1]["content"], note, text)

    def test_chat_thinks_only_when_needed(self):
        for text, expect in (("When do I finish if I have 3 meetings of 45 minutes from 2 pm?", True),
                             ("how many days until 15 November?", True), ("yo", False),
                             ("what time is it?", False), ("find my aadhaar card", False),
                             ("how many tabs do I have open?", False), ("how much RAM is free?", False),
                             ("how many hours until my 5 pm meeting?", True),
                             ("Make me a quick study plan, 3 hours a day starting today", True),
                             ("what's the plan?", False)):
            body, _, _ = server.build_request({"mode": "chat", "think": False,
                                               "messages": [{"role": "user", "content": text}]})
            self.assertEqual(body["chat_template_kwargs"]["enable_thinking"], expect, text)
        body, _, _ = server.build_request({"mode": "chat", "think": False, "messages": [{"role": "user", "content": "yo"}]})
        self.assertNotIn("tool_choice", body)

    def test_day_questions_get_the_coming_weekdays(self):
        def note(text):
            body, _, _ = server.build_request({"mode": "chat", "messages": [{"role": "user", "content": text}]})
            return body["messages"][-1]["content"]
        import datetime
        tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
        n = note("make me a study plan starting today, exam on Friday")
        self.assertIn("tomorrow ", n)
        self.assertIn(f"{tomorrow:%A} {tomorrow.day} {tomorrow:%b}", n)
        self.assertNotIn(" Dates: ", note("who is aryabhata?"))

    def test_long_chats_are_trimmed_to_fit(self):
        short = [{"role": "user", "content": "hi"}]
        body, _, _ = server.build_request({"mode": "chat", "messages": short})
        self.assertEqual(len(body["messages"]), 2)
        self.assertNotIn(server.OMITTED_NOTE, body["messages"][0]["content"])
        long_ = []
        for i in range(40):  # ~120k characters: far beyond the 16k-token context
            long_ += [{"role": "user", "content": f"note {i} " + "x1 y2 z3 " * 330},
                      {"role": "assistant", "content": f"ok {i}"}]
        long_.append({"role": "user", "content": "what was the last note?"})
        body, _, _ = server.build_request({"mode": "chat", "messages": long_})
        msgs = body["messages"]
        self.assertIn(server.OMITTED_NOTE, msgs[0]["content"])
        self.assertTrue(msgs[-1]["content"].startswith("what was the last note?"))  # newest message kept
        self.assertEqual(msgs[1]["role"], "user")  # history restarts at a user turn
        self.assertLess(sum(server.est_tokens(m) for m in msgs) + server.est_tokens(body["tools"]),
                        server.N_CTX - body["max_tokens"])
        self.assertIn("note 39", msgs[-3]["content"])  # the most recent history survives, the oldest goes

    def test_actions_refuse_unsafe_requests(self):
        """Only refusal paths run here: nothing on the laptop changes."""
        import tempfile
        a = server.tools.actions
        with tempfile.TemporaryDirectory() as d:
            for name in ("setup.exe", "run.ps1", "x.bat", "y.vbs", "evil.lnk", "tool.py", "fix.reg", "pkg.msi"):
                server.Path(d, name).write_text("x")
                self.assertIn("error", a.open_file(str(server.Path(d, name))), name)
        for shell in ("cmd", "powershell", "terminal"):
            self.assertIn("error", a.open_app(shell))
        for url in ("file:///C:/Windows", "javascript:alert(1)", "ms-settings:display", "C:\\x.html"):
            self.assertIn("error", a.open_website(url))
        self.assertIn("error", a.open_settings("registry editor"))
        self.assertIn("error", a.control("volume", "loud"))
        self.assertIn("error", a.control("brightness", "-5; rm"))
        self.assertIn("error", a.control("firewall", "off"))
        self.assertIn("error", a.window("close", "sia"))
        self.assertIn("error", a.power("format"))
        self.assertTrue(a.is_dangerous("power", {"mode": "shutdown"}))
        self.assertFalse(a.is_dangerous("power", {"mode": "lock"}))
        self.assertFalse(a.is_dangerous("control", {"setting": "volume", "value": "30"}))

    def test_notes_never_overwrite_and_reminders_parse_times(self):
        import tempfile
        a = server.tools.actions
        orig = server.tools.user_folders
        with tempfile.TemporaryDirectory() as d:
            server.tools.user_folders = lambda: {"documents": d, "pictures": d, "home": d}
            try:
                one = a.save_note("plan.md", "first")
                two = a.save_note("plan.md", "second")
                bad = a.save_note("../../evil.exe", "x")
            finally:
                server.tools.user_folders = orig
            self.assertNotEqual(one["saved"], two["saved"])
            self.assertEqual(server.Path(one["saved"]).read_text(encoding="utf-8"), "first")
            self.assertTrue(bad["saved"].startswith(str(server.Path(d, "Sia"))) and bad["saved"].endswith(".txt"))
        t = a.set_reminder("Timer", seconds=10)  # "set a timer for 10 seconds" once became 10 minutes
        self.assertEqual(t["in"], "10 seconds")
        self.assertEqual(a.cancel_reminder(a.list_reminders()["reminders"][-1]["id"])["cancelled"], "Timer")
        self.assertEqual(server.tools.describe("reminder", '{"action": "set", "text": "Timer", "seconds": 10}'),
                         "Remind you in 10 s: Timer")
        both = a.set_reminder("oven", minutes=1, seconds=30)  # the model sent both for "30 seconds"
        self.assertEqual(both["in"], "30 seconds")
        a.cancel_reminder(a.list_reminders()["reminders"][-1]["id"])
        self.assertEqual(server.tools.describe("reminder", '{"action": "set", "text": "oven", "minutes": 1, "seconds": 30}'),
                         "Remind you in 30 s: oven")
        self.assertIn("seconds", server.tools.TOOLS["reminder"]["schema"]["properties"])
        r = a.set_reminder("stand up", at="5:30 pm")
        self.assertIn("5:30 PM", r["at"])
        self.assertIn("error", a.set_reminder("x", at="25:99"))
        rid = a.list_reminders()["reminders"][-1]["id"]
        self.assertEqual(a.cancel_reminder(rid)["cancelled"], "stand up")

    def test_commands_must_go_through_a_tool(self):
        """The model once said 'Bluetooth is now off' without calling anything: commands force a tool call."""
        for cmd in ("turn bluetooth off", "ok now turn bluetooth off again", "set the volume to 35", "mute it",
                    "close notepad", "open my downloads folder", "take a screenshot", "remind me in 2 minutes",
                    "please lock my screen", "can you open spotify", "restart my laptop", "pause the music"):
            self.assertTrue(server.wants_action(cmd), cmd)
            body, _, _ = server.build_request({"mode": "chat", "messages": [{"role": "user", "content": cmd}]})
            self.assertEqual(body.get("tool_choice"), "required", cmd)
        for q in ("how do I turn off bluetooth?", "is my bluetooth on?", "make me a study plan", "find my resume",
                  "what time is it", "explain how TCP works", "write an email to my prof", "I opened the file"):
            self.assertFalse(server.wants_action(q), q)

    def test_undo_and_cancel(self):
        """Only harmless actions here (a timer set, then undone): nothing on the laptop changes."""
        a = server.tools.actions
        a.HISTORY.clear()
        a.USER_TEXT["text"] = "set a timer for 10 minutes for the oven"
        r = server.tools.run("reminder", '{"action": "set", "text": "oven", "minutes": 10}')
        self.assertIn("reminder", r)
        self.assertEqual(server.tools.describe("undo", '{"what": "last"}'), "Undo: cancel the reminder 'oven'")
        out = a.undo("timer")  # matched by words, not just "last"
        self.assertEqual(out["result"], {"cancelled": "oven"})
        self.assertIn("error", a.undo("last"))  # nothing left to undo
        a.HISTORY.append({"n": 9, "tool": "screenshot", "args": {}, "undo": None, "done": False, "label": "screenshot {}",
                          "desc": "Sia doesn't delete files; it's at C:\\x.png"})
        self.assertIn("can't be undone", a.undo("last")["error"])
        a.record("control", {"setting": "volume", "value": "30"}, {"before": {"volume_pct": 50, "muted": False}})
        self.assertEqual(a.undo_description("volume"), "Undo: volume back to 50%")  # described, not executed
        # cancel a reminder by its words, or the latest one
        a.set_reminder("check the oven", minutes=5)
        a.set_reminder("call mom", minutes=9)
        self.assertEqual(a.cancel_reminder(text="the oven")["cancelled"], "check the oven")
        self.assertEqual(a.cancel_reminder()["cancelled"], "call mom")
        # the model has passed the name as the id ("tea" or a made-up id plus the text): still found
        a.set_reminder("tea", minutes=10)
        self.assertEqual(server.tools.describe("reminder", '{"action": "cancel", "id": "tea"}'),
                         "Cancel the reminder: tea")
        self.assertEqual(a.cancel_reminder(id="tea")["cancelled"], "tea")
        a.set_reminder("tea", minutes=10)
        self.assertEqual(a.cancel_reminder(id="r99", text="the tea timer")["cancelled"], "tea")
        self.assertIn("error", a.cancel_reminder(id="r99"))  # an unknown id alone never cancels the latest
        a.set_reminder("tea", minutes=10)
        a.set_reminder("oven", minutes=20)
        self.assertEqual(server.tools.describe("reminder", '{"action": "cancel", "text": "all"}'),
                         "Cancel all 2 reminders: tea, oven")
        self.assertEqual(a.cancel_reminder(text="all")["cancelled"], ["tea", "oven"])
        # nothing to cancel or undo: answered without an approval card
        self.assertIn("error", server.tools.nothing_to_do("reminder", '{"action": "cancel", "text": "tea"}'))
        self.assertIn("error", server.tools.nothing_to_do("undo", '{"what": "brightness"}'))
        a.set_reminder("tea", minutes=10)
        self.assertIsNone(server.tools.nothing_to_do("reminder", '{"action": "cancel", "text": "tea"}'))
        a.cancel_reminder(text="tea")
        for said in ("stop the tea timer", "never mind the oven reminder", "don't remind me", "abort the restart"):
            self.assertTrue(server.wants_action(said), said)
        for said in ("how do I remove an alarm on my phone?", "stop talking", "delete this file"):
            self.assertFalse(server.wants_action(said), said)
        self.assertTrue(server.wants_action("undo that") and server.wants_action("put the volume back"))
        self.assertFalse(server.wants_action("can you undo things?"))
        a.HISTORY.clear()

    def test_calculator(self):
        """Exact answers the model got wrong in a real chat, and nothing but arithmetic is accepted."""
        calc = server.tools.calc.calculate
        self.assertEqual(calc("876*88894")["result"], "77871144")  # the model said 77,868,972
        self.assertEqual(calc("78*87965*4231589659775*875549663211459*98874488899999")["result"],
                         str(78 * 87965 * 4231589659775 * 875549663211459 * 98874488899999))
        self.assertEqual(calc("0.1+0.2")["result"], "0.3")  # exact decimals, no binary-float error
        self.assertEqual(calc("1/3+1/6")["result"], "0.5")
        self.assertEqual(calc("15% * 2400")["result"], "360")
        self.assertEqual(calc("17 % 5")["result"], "2")
        self.assertEqual(calc("2^10")["result"], "1024")
        self.assertEqual(calc("25!")["result"], "15511210043330985984000000")
        self.assertEqual(calc("round(2.675, 2)")["result"], "2.68")  # half up, as on receipts
        self.assertEqual(calc("r = 0.05/12; round(10000*(1+r)**120, 2)")["result"], "16470.09")
        self.assertTrue(calc("sqrt(2)")["result"].startswith("1.41421356237309504880"))
        self.assertEqual(calc("sind(30)")["result"], "0.5")
        for bad in ('__import__("os").system("calc")', 'open("x")', "().__class__", "lambda: 1", "2**10000000",
                    "1/0", "9**9**9"):
            self.assertIn("error", calc(bad), bad)
        # words and grouping: the model wrote 266,473,922,476,416 as "twenty-six quadrillion ..."
        w = calc("7726736*38837*888", "words")
        self.assertEqual(w["indian_grouping"], "26,64,73,92,24,76,416")
        self.assertEqual(w["words_international"], "two hundred sixty-six trillion, four hundred seventy-three billion, "
                         "nine hundred twenty-two million, four hundred seventy-six thousand, four hundred sixteen")
        self.assertEqual(w["words_indian"], "two crore sixty-six lakh forty-seven thousand three hundred ninety-two "
                         "crore, twenty-four lakh, seventy-six thousand, four hundred sixteen")
        self.assertTrue(w["words_indian_traditional"].startswith("twenty-six neel, sixty-four kharab, seventy-three arab"))
        self.assertEqual(calc("12345678", "indian")["words_indian"],
                         "one crore, twenty-three lakh, forty-five thousand, six hundred seventy-eight")
        self.assertEqual(calc("100000", "indian")["indian_grouping"], "1,00,000")
        self.assertEqual(calc("-2500.75", "international")["words_international"],
                         "minus two thousand, five hundred point seven five")
        for said in ("give in indian numeric system", "in words"):
            self.assertTrue(server.is_math(said), said)
        self.assertFalse(server.is_math("write a story in the indian style"))
        # the tool reads nothing, so it needs no card and works with laptop access off
        self.assertTrue(server.tools.TOOLS["calculate"].get("auto"))
        self.assertEqual(server.tools.describe("calculate", '{"expression": "11*2"}'), "Calculate: 11*2")
        for said in ("11*2", "calculate and tell 876*88894", "what is 15% of 2400",
                     "I invest 50000 at 7% for 10 years, how much will I have?"):
            self.assertTrue(server.is_math(said), said)
        for said in ("my meeting is on 23/10/2026", "call me at 98765-43210", "set the volume to 30%",
                     "how many tabs do I have open?", "remind me in 10 minutes"):
            self.assertFalse(server.is_math(said), said)
        body, _, _ = server.build_request({"mode": "chat", "messages": [{"role": "user", "content": "11*2"}]})
        self.assertEqual(body["tool_choice"], "required")
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])  # a bare sum needs no thinking
        self.assertIn("calculate tool", body["messages"][-1]["content"])

    def test_actions_switch_and_cards(self):
        server.save_settings({**server.DEFAULT_SETTINGS, "laptop_actions": False})
        body, _, _ = server.build_request({"mode": "chat", "messages": self.msgs})
        names = {t["function"]["name"] for t in body["tools"]}
        self.assertNotIn("open", names)
        self.assertIn("read_file", names)  # reading still works with actions off
        server.save_settings(dict(server.DEFAULT_SETTINGS))
        body, _, _ = server.build_request({"mode": "chat", "messages": self.msgs})
        self.assertTrue({"open", "control", "window", "power", "screenshot", "copy", "save_note",
                         "reminder"} <= {t["function"]["name"] for t in body["tools"]})
        d = server.tools.describe
        self.assertEqual(d("control", '{"setting": "volume", "value": "30"}'), "Set volume to 30%")
        self.assertEqual(d("power", '{"mode": "shutdown"}'), "Shut down the laptop in 60 seconds")
        view = server.settings_view()
        self.assertTrue(any(t["action"] for t in view["tools"]) and all(t["reads"] for t in view["tools"]))

    def test_find_files_takes_spelling_variants(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            for n in ("my aadhar scan.pdf", "Aadhaar front.jpg", "notes.txt"):
                server.Path(d, n).write_text("x")
            out = server.tools.find_files("aadhaar|aadhar", d)
            self.assertEqual(sorted(server.Path(m["path"]).name for m in out["matches"]),
                             ["Aadhaar front.jpg", "my aadhar scan.pdf"])

    def test_tools_are_read_only_and_guarded(self):
        self.assertIn("error", server.tools.run("recent_files", '{"folder": "C:/Windows/System32"}'))
        self.assertIn("error", server.tools.run("delete_files", "{}"))
        for name, t in server.tools.TOOLS.items():  # fixed functions: no shell, command or write argument
            for bad in ("command", "cmd", "script", "content", "write", "delete"):
                self.assertNotIn(bad, t["schema"].get("properties", {}), name)

    def test_safe_path_refuses_secrets_and_non_local_paths(self):
        home = server.tools.os.path.expanduser("~")
        refused = [
            "\\\\server\\share\\x.txt", "\\\\.\\PhysicalDrive0", "\\\\?\\C:\\Windows\\System32\\config\\SAM",
            "//server/share/x.txt", "C:\\x\\notes.txt:hidden", home + "\\Documents\\..\\.ssh\\id_rsa",
            home + "\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Login Data",
            home + "\\AppData\\Local\\Microsoft\\Edge\\User Data\\Default\\Cookies",
            "C:\\Windows\\System32\\config\\SAM", home + "\\project\\.env", home + "\\keys\\server.pem",
            home + "\\AppData\\Roaming\\Microsoft\\Credentials\\x",
        ]
        for raw in refused:
            with self.assertRaises((PermissionError, ValueError), msg=raw):
                server.tools.safe_path(raw)
        self.assertEqual(str(server.tools.safe_path("~", "dir")), server.tools.os.path.realpath(home))

    def test_read_file_refuses_binaries(self):
        r = server.tools.run("read_file", '{"path": "C:\\\\Windows\\\\explorer.exe"}')
        self.assertIn("binary", r.get("error", ""))


class DefaultModel(unittest.TestCase):
    """The RAM check picks gpt-oss below 5.5 GiB, but must not count Sia's own loaded model as used."""

    def setUp(self):
        self.vm, self.proc, self.procs = server.psutil.virtual_memory, server.psutil.Process, dict(server.STATE["procs"])

    def tearDown(self):
        server.psutil.virtual_memory, server.psutil.Process = self.vm, self.proc
        server.STATE["procs"].clear(); server.STATE["procs"].update(self.procs)

    def fake(self, avail_gib, model_rss_gib=None):
        server.psutil.virtual_memory = lambda: type("VM", (), {"available": avail_gib * server.GiB})()
        server.STATE["procs"].pop("chat", None)
        if model_rss_gib is not None:
            server.STATE["procs"]["chat"] = type("P", (), {"pid": 1, "poll": lambda self: None})()
            server.psutil.Process = lambda pid: type("Q", (), {"memory_info": lambda self: type(
                "M", (), {"rss": model_rss_gib * server.GiB})()})()

    def test_low_ram_without_a_model_picks_gptoss(self):
        self.fake(4.0)
        self.assertEqual(server.choose_default_model()[0], "gptoss")
        self.fake(8.0)
        self.assertEqual(server.choose_default_model()[0], "gemma")

    def test_a_loaded_gemma_does_not_switch_itself_out(self):
        self.fake(4.0, model_rss_gib=3.0)  # 4 GiB free while Gemma itself holds 3 GiB
        self.assertEqual(server.choose_default_model()[0], "gemma")


class Safety(unittest.TestCase):
    def test_chat_ids_cannot_escape_the_folder(self):
        for bad in ["../../x", "a/b", "..", "", "x" * 100, "c:\\windows"]:
            with self.assertRaises(ValueError):
                server.chat_file(bad)
        self.assertEqual(server.chat_file("abc123XYZ").name, "abc123XYZ.json")

    def test_chat_server_always_has_the_api_key(self):
        """Router mode starts its children without the key, so Sia must never use it."""
        seen = []

        class Dead:
            def poll(self):
                return 1

        orig = server.spawn
        server.spawn = lambda name, cmd: (seen.append(cmd), Dead())[1]
        try:
            with self.assertRaises(RuntimeError):
                server.ChatModel().ensure("gemma-vision")
        finally:
            server.spawn = orig
        cmd = seen[0]
        self.assertIn("--api-key", cmd)
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")
        self.assertNotIn("--models-preset", cmd)
        self.assertIn("--mmproj", cmd)
        self.assertIn("--jinja", cmd)
        self.assertEqual(cmd[cmd.index("--cache-ram") + 1], "1024")

    def test_threads_follow_where_weights_live(self):
        """lab-notebook finding 49: Gemma (partly SSD-streamed) -t 16, RAM-resident gpt-oss -t 8."""
        for name in ("gemma", "gemma-vision"):
            a = server.preset_args(name)
            self.assertEqual(a[a.index("--threads") + 1], "16", name)
        a = server.preset_args("gptoss")
        self.assertEqual(a[a.index("--threads") + 1], "8")
        self.assertEqual(a.count("--threads"), 1)

    def test_presets_cap_ram_hungry_defaults(self):
        self.assertLessEqual(server.COMMON["cache-ram"], 2048)   # default would be 8192 MiB
        self.assertEqual(server.COMMON["parallel"], 1)


class Voice(unittest.TestCase):
    """Voice mode without a microphone: speakable text, the voice list, the spoken-style request."""

    def test_speakable_text(self):
        sp = server.voice.speakable
        self.assertEqual(sp("Your battery is at **82%** & charging."), "Your battery is at 82 percent and charging.")
        self.assertEqual(sp("Run `pip install x`; see [docs](https://x.y)."), "Run pip install x; see docs.")
        self.assertEqual(sp("- one\n- two"), "one two")
        self.assertEqual(sp("```python\nprint(1)\n```"), "")
        self.assertEqual(sp("It's 23°C"), "It's 23 degrees Celsius")

    def test_preroll_keeps_the_first_syllable(self):
        """Speech is detected a little late; 0.3 s from before the detected start goes to recognition too
        ("What's a good way ..." had been heard as "That's a good way ...")."""
        import numpy as np
        v = server.voice
        L = v.Listener.__new__(v.Listener)  # no models needed for this part
        L.heard = np.arange(20000, dtype=np.float32)
        L.heard_from = 10000  # the buffer holds samples 10000..29999
        seg = np.arange(15000, 16000, dtype=np.float32)
        out = L._with_preroll(15000, seg)
        self.assertEqual(len(out), v.PREROLL + 1000)
        self.assertEqual(out[0], 15000 - v.PREROLL - 10000)  # the audio just before the segment
        self.assertEqual(len(L._with_preroll(10100, seg[:10])), 110)  # only as much as is kept

    def test_only_speech_for_sia_is_answered(self):
        """Wake mode: speech counts only with "Hey Sia" in it or in the follow-up window after it (a friend's
        conversation nearby was answered line by line in always-listening mode)."""
        v = server.voice
        L = v.Listener.__new__(v.Listener)  # no models needed for this part
        L.mode, L.fed, L.wake_hits, L.awake_until = "wake", 0, [], 0
        R = v.RATE
        self.assertFalse(L._addressed(0, 2 * R))  # someone talking, no wake phrase
        L.wake_hits = [10 * R]  # "Hey Sia" heard at 10 s, inside "Hey Sia, what time is it?" (9.5-12 s)
        self.assertTrue(L._addressed(int(9.5 * R), 12 * R))
        self.assertFalse(L._addressed(20 * R, 22 * R))  # later chatter
        L.fed = 30 * R
        L.awake(8)  # Sia finished answering: the follow-up window
        self.assertTrue(L._addressed(33 * R, 35 * R))
        self.assertFalse(L._addressed(40 * R, 42 * R))  # after the window
        L.mode = "always"
        self.assertTrue(L._addressed(40 * R, 42 * R))

    def test_unfinished_turn_waits_for_the_rest(self):
        """A pause after "so the thing is, um" holds the words; the rest joins them; a finished turn goes at once."""
        import queue
        import threading
        import types
        v = server.voice
        L = v.Listener.__new__(v.Listener)
        L.work, L.parts, L.hold_until, L.turn_lock, L.in_speech = queue.Queue(), [], None, threading.Lock(), False
        answers = iter([0.1, 0.9])  # unfinished, then finished once the rest is there
        L.turn = types.SimpleNamespace(complete_probability=lambda audio: next(answers))
        a, b = server.np.ones(8000, server.np.float32), server.np.ones(16000, server.np.float32)
        L._add_part((a, True, False))
        self.assertTrue(L.work.empty())
        self.assertIsNotNone(L.hold_until)
        L._add_part((b, False, True))
        audio, in_window, spotted = L.work.get_nowait()
        self.assertEqual(len(audio), 8000 + 16000 + 2 * (v.RATE // 10))  # one piece, short gaps between
        self.assertTrue(in_window and spotted)  # the window from its start, the wake phrase from any part
        self.assertIsNone(L.hold_until)

    def test_turn_features_shape(self):
        """Whisper log-mel for Smart Turn: 80 x 800 (8 s), short audio padded, a floor 8 log10 units under the peak."""
        import turn
        f = turn.features(server.np.sin(server.np.arange(16000) / 10).astype(server.np.float32))
        self.assertEqual(f.shape, (1, 80, 800))
        self.assertTrue(server.np.isfinite(f).all() and f.max() - f.min() <= 2.0001)

    def test_wake_phrase_in_the_transcript(self):
        """"Hey Sia" as the recogniser writes it decides; "see you", "Sierra Leone", "Asia" and "sir" (without the
        keyword spotter) don't wake Sia."""
        split = server.voice.wake_split
        for said, request in [("Hey Sia, what time is it?", "what time is it?"), ("hey sia.", ""),
                              ("He sia, play some music.", "play some music."), ("Okay Cia, open Chrome.", "open Chrome."),
                              ("Hi Sier, set a timer.", "set a timer."), ("Sia, what's the weather?", "what's the weather?"),
                              ("Hello, Siya.", "")]:
            self.assertEqual(split(said), (True, request), said)
        for said in ["Hey, see you tomorrow.", "Hey, Sierra Leone is in Africa.", "Asia is a big continent.",
                     "Hey Siri, call mom.", "Hey sir, can I help?", "I see a bird.", "Sierra is my cousin."]:
            self.assertFalse(split(said)[0], said)
        self.assertEqual(split("Hey sir, what time is it?", spotted=True), (True, "what time is it?"))

    def test_heteronyms_by_context(self):
        """"Share your live location" vs "I live here": a one-entry dictionary got 17 of 32 such cases right,
        misaki (part-of-speech aware) 29. Runs only where misaki is installed (it's optional)."""
        import re
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
        from heteronym_cases import CASES
        try:
            import misaki  # noqa: F401
        except ImportError:
            self.skipTest("misaki isn't installed")
        if not server.voice.available():
            self.skipTest("the voice models aren't installed")
        server.voice.load_tts("af_heart")
        tts = server.voice._M["tts"]
        right = 0
        for sentence, word, want in CASES:
            _, toks = tts.g2p(sentence)
            right += bool(re.search(want, next((t.phonemes or "" for t in toks if t.text.lower() == word), "")))
        self.assertGreaterEqual(right, 28, f"{right}/{len(CASES)} heteronyms right")
        server.voice.unload()

    def test_voice_list(self):
        ids = server.voice.VOICE_IDS
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn(server.voice.DEFAULT_VOICE, ids)
        self.assertTrue({"male", "female"} == {v["gender"] for v in server.voice.VOICES})
        self.assertFalse({"am_adam", "am_onyx", "bf_lily"} & set(ids))  # D-graded voices are left out
        if server.voice.available():  # every listed voice exists in the installed model
            names = server.voice._speaker_ids()
            self.assertTrue(set(ids) <= set(names), set(ids) - set(names))

    def test_voice_request_and_setting(self):
        import tempfile
        orig = server.SETTINGS_FILE
        with tempfile.TemporaryDirectory() as d:
            server.SETTINGS_FILE = server.Path(d) / "settings.json"
            try:
                body, _, _ = server.build_request({"mode": "chat", "voice": True, "messages": [
                    {"role": "user", "content": "plan my week and compare two options step by step"}]})
                self.assertIn("Spoken reply, read aloud", body["messages"][-1]["content"])
                self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])  # no silent thinking pause
                body, _, _ = server.build_request({"mode": "chat", "messages": [{"role": "user", "content": "hi"}]})
                self.assertNotIn("Spoken reply", body["messages"][-1]["content"])
                self.assertEqual(server.load_settings()["voice_name"], "af_heart")
            finally:
                server.SETTINGS_FILE = orig


class GameIntents(unittest.TestCase):
    def test_asking_for_the_word_reveals_it(self):
        """From a voice test: "What is the word?" got "I can't tell you directly" until "I pass"."""
        g = server.tools.games
        for said in ("What is the word?", "I pass", "what's the answer", "tell me the word", "I give up"):
            self.assertTrue(g.REVEAL_RX.search(said) and g.CONTROL_RX.search(said), said)
        self.assertFalse(g.REVEAL_RX.search("give me a hint"))
        self.assertTrue(g.GAME_RX.search("Want to play a game of Wirdle?"))  # how speech recognition hears it
        g.GAME.clear()
        g.WANTS_REVEAL["flag"] = False
        g.word_game("start", "wordle")
        g.WANTS_REVEAL["flag"] = True  # the user asked for the word, the model called hint: the word is revealed
        out = g.word_game("hint")
        self.assertTrue(out.get("over") and out.get("word"))
        g.GAME.clear()


class KeepAlive(unittest.TestCase):
    def test_requests_in_a_row_on_one_connection(self):
        """Browsers reuse a connection. An unread body ("{}" to /api/voice/start) once turned the next request into
        "{}GET ..." (501), and a body cached across requests turned audio bytes into the next method name."""
        import http.client
        import threading
        srv = server.Server(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
            h = {"Origin": f"http://127.0.0.1:{server.APP_PORT}", "Content-Type": "application/json"}
            for method, path, body, want in [("POST", "/api/bye", "{}", 200),  # body not used by the endpoint
                                             ("GET", "/static/voice-worklet.js", None, 200),
                                             ("POST", "/api/voice/audio", b"\x00" * 3200, 200),
                                             ("POST", "/api/bye", "{}", 200),
                                             ("GET", "/static/app.js", None, 200)]:
                c.request(method, path, body, h if method == "POST" else {})
                r = c.getresponse()
                r.read()
                self.assertEqual(r.status, want, f"{method} {path}")
        finally:
            srv.shutdown()
            srv.server_close()


class UiScript(unittest.TestCase):
    def test_every_called_function_is_defined(self):
        """A plain function the UI calls must exist (a rewrite once deleted voiceState, and voice mode failed
        with "voiceState is not defined", which a syntax check can't see)."""
        import re
        js = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "static", "app.js"),
                  encoding="utf-8").read()
        defined = set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)", js))
        defined |= set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", js))
        for params in re.findall(r"\(([^()]*)\)\s*(?:=>|\{)", js):  # parameters and destructured names
            defined |= set(re.findall(r"[A-Za-z_$][\w$]*", params))
        defined |= set(re.findall(r"\b([A-Za-z_$][\w$]*)\s*=>", js))
        # where calls are looked for: comments and one-line string literals removed, line by line
        lines = []
        for line in js.splitlines():
            if "//" in line and not re.search(r"https?://|['\"`][^'\"`]*//", line):
                line = line.split("//")[0]
            lines.append(re.sub(r"`[^`]*`|\"[^\"\n]*\"|'[^'\n]*'", '""', line))
        code = "\n".join(lines)
        builtins = {"if", "for", "while", "switch", "catch", "return", "typeof", "function", "await", "new", "fetch",
                    "setTimeout", "setInterval", "clearTimeout", "clearInterval", "requestAnimationFrame", "confirm",
                    "alert", "encodeURIComponent", "decodeURIComponent", "parseInt", "parseFloat", "Number", "String",
                    "Boolean", "Array", "Object", "JSON", "Math", "Date", "Promise", "Error", "Set", "Map", "Uint8Array",
                    "AudioContext", "AudioWorkletNode", "FileReader", "Image", "URL", "Blob", "TextDecoder", "marked",
                    "DOMPurify", "katex", "renderMathInElement", "hljs", "AbortController", "isNaN", "super", "import",
                    "async", "File", "MATH"}  # MATH: the @@MATH(\d+)@@ placeholder inside a regular expression
        called = {m for m in re.findall(r"(?<![.\w$])([A-Za-z_$][\w$]*)\s*\(", code)}
        self.assertEqual(sorted(called - defined - builtins), [])


class Plugins(unittest.TestCase):
    """The MCP host and the Python sandbox, against a fake MCP server (nothing on the laptop is touched)."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        cfg = {"servers": {"fake": {
            "command": sys.executable, "args": [os.path.join(os.path.dirname(__file__), "fake_mcp_server.py")],
            "requires": "actions", "guide": "Fake guide.",
            "tools": {
                "Echo": {"policy": "read", "fixed": {"secret_flag": False}, "label": "Echo {text}"},
                "Press": {"policy": "ask", "allowed": {"mode": ["a", "b"]}, "blocked": {"shortcut": ["win+r"]},
                          "label": "Press {shortcut}"}}}}}
        path = os.path.join(self.tmp.name, "mcp.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
        self.host = server.mcp_host.Host(path, self.tmp.name)
        self.host.load()
        self.before = dict(server.tools.TOOLS)
        server.tools.register_mcp(self.host)

    def tearDown(self):
        self.host.stop()
        server.tools.TOOLS.clear()
        server.tools.TOOLS.update(self.before)
        server.tools.MCP["host"] = None
        self.tmp.cleanup()

    def test_only_listed_tools_with_sia_rules(self):
        self.assertEqual(sorted(self.host.tools), ["fake_Echo", "fake_Press"])  # "Hidden" isn't in the config
        self.assertNotIn("secret_flag", self.host.tools["fake_Echo"]["schema"]["properties"])  # fixed: not the model's
        self.assertEqual(self.host.tools["fake_Press"]["schema"]["properties"]["mode"]["enum"], ["a", "b"])
        out = json.loads(server.tools.run("fake_Echo", '{"text": "hi", "secret_flag": true}')["text"])
        self.assertEqual(out, {"called": "Echo", "args": {"text": "hi", "secret_flag": False}})
        self.assertIn("error", server.tools.nothing_to_do("fake_Press", '{"shortcut": "Win + R"}'))  # no card
        self.assertIn("error", server.tools.run("fake_Press", '{"mode": "launch_executable"}'))
        self.assertIsNone(server.tools.nothing_to_do("fake_Press", '{"shortcut": "ctrl+s"}'))
        self.assertEqual(server.tools.describe("fake_Press", '{"shortcut": "ctrl+s"}'), "Press ctrl+s")
        self.assertTrue(server.tools.always_ask("fake_Press") and not server.tools.always_ask("fake_Echo"))
        self.assertTrue(server.tools.TOOLS["fake_Press"]["action"] and not server.tools.TOOLS["fake_Echo"]["action"])
        # the tool list is cached: a second host doesn't need the servers to start
        again = server.mcp_host.Host(os.path.join(self.tmp.name, "mcp.json"), self.tmp.name)
        again.servers["fake"].list_tools = lambda: 1 / 0  # would fail if it were started
        self.assertEqual(sorted(again.load()), ["fake_Echo", "fake_Press"])

    def test_plugins_follow_the_settings_switches(self):
        msgs = [{"role": "user", "content": "hello"}]
        orig = server.SETTINGS_FILE
        server.SETTINGS_FILE = server.Path(self.tmp.name) / "settings.json"
        try:
            server.save_settings({**server.DEFAULT_SETTINGS, "laptop_actions": True})
            body, _, _ = server.build_request({"mode": "chat", "messages": [dict(m) for m in msgs]})
            names = {t["function"]["name"] for t in body["tools"]}
            self.assertTrue({"fake_Echo", "fake_Press", "run_python"} <= names)
            self.assertIn("Fake guide.", body["messages"][0]["content"])
            server.save_settings({**server.DEFAULT_SETTINGS, "laptop_access": False})
            body, _, _ = server.build_request({"mode": "chat", "messages": [dict(m) for m in msgs]})
            self.assertFalse({"fake_Echo", "fake_Press"} & {t["function"]["name"] for t in body["tools"]})
        finally:
            server.SETTINGS_FILE = orig

    def test_card_text_for_lists(self):
        self.assertEqual(server.mcp_host._flat([{"entityName": "Hrithik", "contents": ["likes tea", "codes at night"]}]),
                         "Hrithik: likes tea; codes at night")

    def test_forget_matches_what_is_stored(self):
        rows = [{"type": "entity", "name": "Ananya", "entityType": "person", "observations": ["Hrithik's sister"]},
                {"type": "entity", "name": "Hrithik S P", "entityType": "person",
                 "observations": ["likes short answers", "uses an HP Victus laptop"]}]
        orig = server.tools._memory_rows
        server.tools._memory_rows = lambda: (None, rows)
        try:
            self.assertEqual(server.tools._forget_plan("my sister's name"), (["Ananya"], []))  # the whole person
            self.assertEqual(server.tools._forget_plan("HP Victus"),
                             ([], [{"entityName": "Hrithik S P", "observations": ["uses an HP Victus laptop"]}]))
            self.assertEqual(server.tools.forget_label("my sister"), "Forget: everything about Ananya")
            self.assertEqual(server.tools._forget_plan("my brother"), ([], []))
        finally:
            server.tools._memory_rows = orig

    def test_sandbox(self):
        run = server.tools.sandbox.run_python
        self.assertEqual(run("7726736*38837*888")["result"], 266473922476416)
        self.assertEqual(run("import datetime\n(datetime.date(2026, 11, 15) - datetime.date(2026, 10, 23)).days")["result"], 23)
        self.assertEqual(run('exact("0.1+0.2")')["result"], "0.3")
        self.assertTrue(run('words(100000, "indian")')["result"] == "one lakh")
        self.assertIn("printed", run("print('hi')"))
        for bad in ('open("C:/Windows/win.ini").read()', "import os\nos.listdir('C:/')", "import subprocess",
                    "import socket"):
            self.assertIn("error", run(bad), bad)


if __name__ == "__main__":
    unittest.main()
