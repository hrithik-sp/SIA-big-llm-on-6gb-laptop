"""run_python: the model writes a short Python program and Sprout runs it in a sandbox.

One general tool instead of a new special case for every kind of question (dates, conversions,
lists, text, formatting, multi-step maths). The sandbox is Pydantic's Monty
(https://github.com/pydantic/monty, `pydantic-monty` on PyPI): a Python interpreter written in Rust
that runs in its own worker process, with no files, network, subprocesses or environment variables
(checked here: open(), os.listdir, import subprocess/socket are all refused), and with time and
memory limits. A crashed or runaway worker is killed and replaced; Sprout itself is never at risk.

Monty has no calendar, statistics, string, decimal or fractions modules, so Sprout lends it the pure
functions of the first three (they run in Sprout's own Python and touch nothing on the laptop) and two
exact helpers from calc.py: exact("expression") and words(number, "indian" | "international").
Its clock runs in UTC unless told otherwise, so each run gets the laptop's own UTC offset.
"""
import calendar
import datetime as dt
import statistics
import string
import sysconfig
import threading
from pathlib import Path

import calc

TIMEOUT_S = 5  # per program; the worker is killed at TIMEOUT_S + 5
MAX_MEMORY = 256 * 1024 * 1024
MAX_OUTPUT = 20000  # characters of printed output kept

_POOL = {"pool": None, "lock": threading.Lock()}


def _binary():
    """The monty worker, also when the package was installed with pip --user (not on PATH)."""
    for scheme in (None, "nt_user"):
        try:
            d = Path(sysconfig.get_path("scripts", scheme) if scheme else sysconfig.get_path("scripts"))
        except KeyError:
            continue
        if (d / "monty.exe").exists():
            return str(d / "monty.exe")
    return None


def _pool():
    import pydantic_monty
    with _POOL["lock"]:
        if _POOL["pool"] is None:
            pool = pydantic_monty.Monty(binary_path=_binary(), min_processes=1, max_processes=2,
                                        request_timeout=TIMEOUT_S + 5)
            pool.__enter__()
            _POOL["pool"] = pool
        return _POOL["pool"]


def _exact(expression):
    r = calc.calculate(expression)
    if "error" in r:
        raise ValueError(r["error"])
    return r["result"]


def _words(number, style="indian"):
    r = calc.calculate(str(number), "indian" if style == "indian" else "international")
    if "error" in r:
        raise ValueError(r["error"])
    return r.get("words_indian") if style == "indian" else r.get("words_international")


def _modules():
    import pydantic_monty as m
    pure = {
        "calendar": {k: getattr(calendar, k) for k in ("monthrange", "isleap", "leapdays", "weekday",
                                                       "monthcalendar")} | {
            "month_name": list(calendar.month_name), "day_name": list(calendar.day_name),
            "month_abbr": list(calendar.month_abbr), "day_abbr": list(calendar.day_abbr)},
        "statistics": {k: getattr(statistics, k) for k in ("mean", "fmean", "median", "median_low", "median_high",
                                                           "mode", "multimode", "stdev", "pstdev", "variance",
                                                           "pvariance", "quantiles", "geometric_mean",
                                                           "harmonic_mean")},
        "string": {k: getattr(string, k) for k in ("ascii_letters", "ascii_lowercase", "ascii_uppercase", "digits",
                                                   "hexdigits", "octdigits", "punctuation", "whitespace",
                                                   "printable", "capwords")},
    }
    return {name: m.ExternalModule(attrs) for name, attrs in pure.items()}


def _local_zone():
    now = dt.datetime.now().astimezone()
    return {"timezone": {"offset_seconds": int(now.utcoffset().total_seconds()), "name": now.tzname() or "local"}}


def _plain(v, depth=0):
    """Sandbox values -> JSON-friendly values (big integers as text, so no digit is lost)."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v) if abs(v) > 2 ** 53 else v
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (list, tuple, set)) and depth < 4:
        return [_plain(x, depth + 1) for x in list(v)[:500]]
    if isinstance(v, dict) and depth < 4:
        return {str(k): _plain(x, depth + 1) for k, x in list(v.items())[:500]}
    return repr(v)[:MAX_OUTPUT]


def run_python(code):
    """Run the program; return its last expression's value and anything it printed."""
    import pydantic_monty as m
    code = str(code or "")
    if not code.strip():
        return {"error": "no code given"}
    out = m.CollectString(max_bytes=1024 * 1024)
    limits = m.ResourceLimits(max_feed_duration_secs=TIMEOUT_S, max_memory=MAX_MEMORY)
    try:
        with _pool().checkout(limits=limits, os_policy=_local_zone(), external_modules=_modules()) as session:
            value = session.feed_run(code, print_callback=out,
                                     external_lookup={"exact": _exact, "words": _words})
    except m.MontyCrashedError as e:
        return {"error": "the program was stopped (too slow or too much memory)" if getattr(e, "timed_out", False)
                else f"the sandbox stopped: {e}"[:500]}
    except m.MontyError as e:
        return {"error": str(e)[-1500:], "printed": out.output[-4000:]}
    result = {"printed": out.output[:MAX_OUTPUT]} if out.output else {}
    if value is not None:
        result["result"] = _plain(value)
    return result or {"result": None, "note": "the program printed nothing and has no final value"}
