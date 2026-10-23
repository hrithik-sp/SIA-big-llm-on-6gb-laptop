"""Tests for app/server.py logic that needs no model: chunking, extraction, request building."""
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
        self.assertTrue(body["messages"][0]["content"].startswith("You are Sprout"))
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

    def test_every_mode_is_sprout(self):
        for m in server.MODES["modes"]:
            self.assertIn("Sprout", m["system"], m["id"])
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
        # only the game referee stays: it reads nothing from the laptop
        self.assertEqual([t["function"]["name"] for t in body["tools"]], ["word_game"])
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
            server.tools.now_context = lambda: "Right now on this laptop: 10:00"
            a, _, _ = server.build_request({"mode": "chat", "messages": msgs})
            server.tools.now_context = lambda: "Right now on this laptop: 10:01"
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
                             ("how many hours until my 5 pm meeting?", True)):
            body, _, _ = server.build_request({"mode": "chat", "think": False,
                                               "messages": [{"role": "user", "content": text}]})
            self.assertEqual(body["chat_template_kwargs"]["enable_thinking"], expect, text)
        body, _, _ = server.build_request({"mode": "chat", "think": False, "messages": [{"role": "user", "content": "yo"}]})
        self.assertNotIn("tool_choice", body)

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
        self.assertIn("error", a.window("close", "sprout"))
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
            self.assertTrue(bad["saved"].startswith(str(server.Path(d, "Sprout"))) and bad["saved"].endswith(".txt"))
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
    """The RAM check picks gpt-oss below 5.5 GiB, but must not count Sprout's own loaded model as used."""

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
        """Router mode starts its children without the key, so Sprout must never use it."""
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


if __name__ == "__main__":
    unittest.main()
