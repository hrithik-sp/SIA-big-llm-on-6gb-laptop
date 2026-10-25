"""Word games with a real referee: the code picks and keeps the secret word and scores every guess.

A language model can't hold a hidden word across turns (it isn't in the conversation), so left
alone it improvises feedback that contradicts itself and leaks the answer. Here the model only
presents what this engine returns. One game is active at a time (Sia has one user).
"""
import random
import re

WORDLE_WORDS = [w for w in """
about above actor adult after again agent agree alarm album alert alive allow alone along angle
angry apple apply arena argue arise array aside asset award aware badge basic beach begin being
below bench birth black blade blame blank blind block blood board boost brain brand brave bread
break brick brief bring broad brown brush build bunch burst cabin cable candy carry catch cause
chain chair chalk charm chart cheap check chess chest chief child chill civil claim class clean
clear click climb clock close cloud coach coast count court cover crack craft crane crash cream
crime cross crowd crown curve cycle daily dance death delay depth dirty doubt dozen draft drama
dream dress drift drink drive eager early earth eight elite empty enemy enjoy enter entry equal
error event exact exist extra faith false fancy fault feast fence fever field fifty fight final
flame flash fleet floor fluid focus force forge forum found frame fresh front frost fruit funny
giant given glass globe glory grace grade grain grand grant grape graph grass great green greet
group guard guess guide habit happy heart heavy honey horse hotel house human humor ideal image
index inner input issue jelly joint judge juice knife label large laser later laugh layer learn
lemon level light limit local logic loose lucky lunch magic major maker march match maybe medal
media metal meter might minor model money month moral motor mount mouse mouth movie music nerve
never night noble noise north novel nurse ocean offer often olive onion opera orbit order other
outer owner paint panel paper party patch peace pearl phase phone photo piano piece pilot pitch
pizza place plain plane plant plate point polar power press price pride prime print prize proof
proud prove queen quick quiet quote radio raise range rapid ratio reach ready relax reply rider
ridge right river robot rocky round route royal rural salad scale scene scope score sense serve
seven shade shake shape share sharp sheep shelf shell shift shine shirt shock shore short shout
sight skill sleep slice slide small smart smile smoke snake solar solid solve sound south space
spare speak speed spend spice spike spoon sport staff stage stair stamp stand start state steam
steel stick still stone store storm story strip study style sugar suite sunny super sweet swing
table taste teach thank theme thick thing think three throw tiger title toast today token topic
total touch tower track trade train treat trend trial tribe trick truck trust truth twice uncle
under union unity upper urban usage usual valid value video virus visit vital vivid voice watch
water whale wheel white whole woman world worry write wrong yield young youth zebra
""".split() if len(w) == 5 and w.isalpha()]

HANGMAN_WORDS = [w for w in """
planet garden rocket silver castle bridge pencil button window jungle island guitar monkey orange
puzzle spider laptop coffee dragon forest market candle mirror wallet rabbit school summer winter
yellow pocket kitchen picture library diamond battery monitor keyboard elephant treasure mountain
sandwich umbrella notebook airplane football computer internet hospital birthday festival
triangle painting magnet number letter family travel energy cookie blanket captain chicken
dolphin feather giraffe harbour journey lantern mystery octopus pumpkin rainbow science volcano
whistle cabinet compass galaxy helmet insect kettle ladder medal needle parrot quartz saddle
tunnel violin walnut zipper anchor basket cactus engine falcon glider hammer jacket koala lizard
marble nectar oyster pepper rhythm sunset tablet unicorn velvet wizard
""".split() if 5 <= len(w) <= 9 and w.isalpha()]

GAME = {}  # the active game: {"kind", "word", "guesses" | "letters", "over"}
MAX_WORDLE_GUESSES, HANGMAN_LIVES = 6, 6
GAME_RX = re.compile(r"\b(w[oiue]rdle|hangman)\b", re.I)  # the speech recogniser hears "Wordle" as "Wirdle"
CONTROL_RX = re.compile(r"\b(give up|i quit|i pass|reveal|what('?s| is| was) the (word|answer)|tell me the (word|answer)|"
                        r"hint|play again|another one|new game)\b", re.I)


NEW_GAME_RX = re.compile(r"\b(play again|new game|another (one|game|round)|start over|restart)\b", re.I)
HINT_RX = re.compile(r"\b(hint|clue|help me)\b", re.I)
# "What is the word?" asks for the answer: the model once refused ("I can't tell you directly") until "I pass"
REVEAL_RX = re.compile(r"\b(give up|i quit|i pass|reveal|what('?s| is| was) the (word|answer)|tell me the (word|answer))\b",
                       re.I)
WANTS_REVEAL = {"flag": False}  # set by the server when the user's message asks for the answer
WANTS_NEW = {"flag": False}  # set by the server when the user's message asks for a fresh game
WANTS_HINT = {"flag": False}  # set by the server when the user's message asks for a hint
USER_TEXT = {"text": None}  # the user's newest message, set by the server: only its words count as guesses


def state_line():
    """The running game's state for the newest message: the model never sees its earlier tool calls."""
    if not GAME or GAME["over"]:
        return ""
    how = ("Use word_game with action guess for a guess, hint for a hint, reveal if they give up (not start, "
           "which would not restart this game), and tell the user its summary in full.")
    if GAME["kind"] == "wordle":
        done = ", ".join(g.upper() for g in GAME["guesses"]) or "none yet"
        return (f"Wordle game in progress (guesses so far: {done}; {MAX_WORDLE_GUESSES - len(GAME['guesses'])} "
                f"left). {how}")
    return f"Hangman game in progress (board {_board()}; {HANGMAN_LIVES - len(GAME['wrong'])} lives left). {how}"


def is_game_turn(text):
    """True when this message is a game move, so the referee must handle it rather than the model:
    the game is named, or a game is on and the message is one letter, a word of the secret's length,
    or a request to give up, get a hint or play again."""
    text = (text or "").strip()
    if GAME_RX.search(text):
        return True
    if not GAME:
        return False
    if CONTROL_RX.search(text):
        return True
    word = re.sub(r"[^a-z]", "", text.lower())
    return not GAME["over"] and text.replace(" ", "").isalpha() and (len(word) == 1 or len(word) == len(GAME["word"]))


def _wordle_marks(guess, word):
    """Standard Wordle scoring: greens first, then yellows limited by the letters left over."""
    marks, left = ["not in word"] * 5, []
    for i, (g, w) in enumerate(zip(guess, word)):
        if g == w:
            marks[i] = "right spot"
        else:
            left.append(w)
    for i, g in enumerate(guess):
        if marks[i] != "right spot" and g in left:
            marks[i] = "in word, wrong spot"
            left.remove(g)
    return marks


def _board():
    return " ".join(c.upper() if c in GAME["letters"] else "_" for c in GAME["word"])


def _status():
    """Where a running game stands, for a 'start' that would otherwise wipe it."""
    if GAME["kind"] == "wordle":
        return {"game": "wordle", "in_progress": True, "guesses_so_far": [g.upper() for g in GAME["guesses"]],
                "guesses_left": MAX_WORDLE_GUESSES - len(GAME["guesses"])}
    return {"game": "hangman", "in_progress": True, "board": _board(),
            "wrong_guesses": [w.upper() for w in GAME["wrong"]], "lives_left": HANGMAN_LIVES - len(GAME["wrong"])}


def word_game(action, game="wordle", guess="", new=False):
    action, game, guess = str(action).lower(), str(game).lower(), str(guess or "").strip().lower()
    if WANTS_REVEAL["flag"] and GAME and not GAME["over"]:  # the user asked for the answer: reveal it
        WANTS_REVEAL["flag"] = False
        action = "reveal"
    if WANTS_NEW["flag"] and GAME and action != "start":
        # "play again": the model once called reveal, ended the game and then said "a new word is ready"
        old = GAME["word"].upper()
        out = word_game("start", GAME["kind"], new=True)
        out["note"] = f"The user asked for a new game: a fresh one has started (the last word was {old}). " + out["note"]
        return out
    if action == "start":
        if game not in ("wordle", "hangman"):
            return {"error": "game must be wordle or hangman"}
        new = new or WANTS_NEW["flag"]  # "play again" from the user, even if the model left new=false
        WANTS_NEW["flag"] = False
        if GAME and not GAME["over"] and GAME["kind"] == game and not new:
            # the model doesn't see its earlier tool calls, so it may "start" again mid-game: keep the game
            out = {**_status(), "note": "This game is already running; carry on with it. Use new=true only "
                                        "if the user asked for a fresh game."}
            if WANTS_HINT["flag"]:  # the user asked for a hint but the model called start: give the hint anyway
                out.update(word_game("hint"))
                out["note"] = "The user asked for a hint: tell them this hint."
            return out
        word = random.choice(WORDLE_WORDS if game == "wordle" else HANGMAN_WORDS)
        GAME.clear()
        GAME.update(kind=game, word=word, guesses=[], letters=set(), wrong=[], over=False)
        if game == "wordle":
            return {"game": "wordle", "word_length": 5, "guesses_allowed": MAX_WORDLE_GUESSES,
                    "note": "The engine holds the secret word. Don't guess or mention it."}
        return {"game": "hangman", "word_length": len(word), "board": _board(), "lives": HANGMAN_LIVES,
                "note": "The engine holds the secret word. Show the board exactly as given."}
    if not GAME:
        return {"error": "no game is running; start one first"}
    if action == "reveal":
        GAME["over"] = True
        return {"game": GAME["kind"], "word": GAME["word"].upper(), "over": True}
    if GAME["over"]:
        return {"game": GAME["kind"], "over": True, "word": GAME["word"].upper(),
                "note": "This game is over; start a new one to play again."}
    if action == "hint":  # one true letter the player hasn't found yet
        if GAME["kind"] == "wordle":
            found = {i for g in GAME["guesses"] for i, (a, b) in enumerate(zip(g, GAME["word"])) if a == b}
            i = next((i for i in range(5) if i not in found), 0)
            return {"hint": f"letter {i + 1} is {GAME['word'][i].upper()}"}
        missing = [c for c in GAME["word"] if c not in GAME["letters"]]
        return {"hint": f"the word contains {missing[0].upper()}" if missing else "no letters left to find"}
    if action != "guess":
        return {"error": "action must be start, guess, hint or reveal"}
    said = USER_TEXT["text"]
    if said is not None and guess not in re.findall(r"[a-z]+", said.lower()):
        # the model must not play for the user (it once submitted "GUESS" after the user typed "ok")
        return {"error": f"'{guess.upper()}' isn't in the user's message, so it isn't their guess. Nothing was "
                         f"scored; wait for them to make a guess."}
    if GAME["kind"] == "wordle":
        if len(guess) != 5 or not guess.isalpha():
            return {"error": "a Wordle guess must be one 5-letter word", "guesses_left":
                    MAX_WORDLE_GUESSES - len(GAME["guesses"])}
        marks = _wordle_marks(guess, GAME["word"])
        GAME["guesses"].append(guess)
        won = guess == GAME["word"]
        left = MAX_WORDLE_GUESSES - len(GAME["guesses"])
        GAME["over"] = won or left == 0
        groups = {k: [g.upper() for g, m in zip(guess, marks) if m == k]
                  for k in ("right spot", "in word, wrong spot", "not in word")}
        summary = "; ".join(f"{', '.join(v)} {'is' if len(v) == 1 else 'are'} " +
                            {"right spot": "in the right spot", "in word, wrong spot": "in the word but in the wrong spot",
                             "not in word": "not in the word"}[k] for k, v in groups.items() if v)
        out = {"guess": guess.upper(), "summary": ("You got it! " if won else "") + summary + ".",
               "feedback": [{"letter": g.upper(), "result": m} for g, m in zip(guess, marks)],
               "won": won, "guesses_left": left}
        if GAME["over"]:
            out["word"] = GAME["word"].upper()
        return out
    # hangman: one letter, or a whole-word attempt
    if not guess.isalpha():
        return {"error": "guess one letter or the whole word"}
    if len(guess) == 1:
        if guess in GAME["letters"] or guess in GAME["wrong"]:
            return {"note": f"'{guess.upper()}' was already guessed", "board": _board(),
                    "lives_left": HANGMAN_LIVES - len(GAME["wrong"])}
        hit = guess in GAME["word"]
        (GAME["letters"].add(guess) if hit else GAME["wrong"].append(guess))
        out = {"letter": guess.upper(), "in_word": hit, "count": GAME["word"].count(guess)}
    else:
        hit = guess == GAME["word"]
        if hit:
            GAME["letters"].update(GAME["word"])
        else:
            GAME["wrong"].append(guess)
        out = {"word_guess": guess.upper(), "correct": hit}
    won = all(c in GAME["letters"] for c in GAME["word"])
    lives = HANGMAN_LIVES - len(GAME["wrong"])
    GAME["over"] = won or lives <= 0
    out.update(board=_board(), wrong_guesses=[w.upper() for w in GAME["wrong"]], lives_left=max(lives, 0), won=won)
    if "letter" in out:
        said = (f"Yes, {out['letter']} appears {out['count']} time{'s' if out['count'] != 1 else ''}" if hit
                else f"No {out['letter']}")
    else:
        said = f"{out['word_guess']} is {'right' if hit else 'not the word'}"
    out["summary"] = f"{said}. Board: {out['board']}. Lives left: {out['lives_left']}."
    if GAME["over"]:
        out["word"] = GAME["word"].upper()
    return out
