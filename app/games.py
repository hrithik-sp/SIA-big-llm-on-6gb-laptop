"""Word games with a real referee: the code picks and keeps the secret word and scores every guess.

A language model can't hold a hidden word across turns (it isn't in the conversation), so left
alone it improvises feedback that contradicts itself and leaks the answer. Here the model only
presents what this engine returns. One game is active at a time (Sprout has one user).
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
GAME_RX = re.compile(r"\b(wordle|hangman)\b", re.I)
CONTROL_RX = re.compile(r"\b(give up|i quit|reveal|what was the word|what's the word|hint|play again|another one|"
                        r"new game)\b", re.I)


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


def word_game(action, game="wordle", guess=""):
    action, game, guess = str(action).lower(), str(game).lower(), str(guess or "").strip().lower()
    if action == "start":
        if game not in ("wordle", "hangman"):
            return {"error": "game must be wordle or hangman"}
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
    if GAME["kind"] == "wordle":
        if len(guess) != 5 or not guess.isalpha():
            return {"error": "a Wordle guess must be one 5-letter word", "guesses_left":
                    MAX_WORDLE_GUESSES - len(GAME["guesses"])}
        marks = _wordle_marks(guess, GAME["word"])
        GAME["guesses"].append(guess)
        won = guess == GAME["word"]
        left = MAX_WORDLE_GUESSES - len(GAME["guesses"])
        GAME["over"] = won or left == 0
        out = {"guess": guess.upper(), "feedback": [{"letter": g.upper(), "result": m} for g, m in zip(guess, marks)],
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
    if GAME["over"]:
        out["word"] = GAME["word"].upper()
    return out
