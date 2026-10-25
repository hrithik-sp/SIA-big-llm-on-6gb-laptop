"""Sprout's calculator: exact arithmetic for the model, which can't multiply long numbers reliably.

Whole numbers have unlimited digits and fractions stay exact (1/3 + 1/6 is exactly 1/2), so money and
dose arithmetic has no rounding error. sqrt, exp, ln and powers like 1.05**2.5 use 40 significant
digits; trigonometry and statistics use ordinary floating point (15 digits). Only numbers, operators,
the functions below and names assigned in the same call are accepted: the expression is parsed and
walked node by node, never run as code.

  calculate("876*88894")                       -> 77871144
  calculate("r = 0.05/12; 1000*(1+r)**120")     -> named steps, separated by ; or new lines
"""
import ast
import math
import re
import statistics
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction

PREC = 40  # significant digits for sqrt, exp, ln, non-integer powers
MAX_BITS = 4_000_000  # ~1.2 million digits: the largest whole number a step may produce
MAX_SHOWN = 3000  # longer results are shown as their first and last digits plus the digit count
MAX_LEN = 4000  # characters of input


class CalcError(ValueError):
    pass


def _dec(v):
    if isinstance(v, Fraction):
        with localcontext() as c:
            c.prec = PREC + 5
            return Decimal(v.numerator) / Decimal(v.denominator)
    if isinstance(v, float):
        return Decimal(repr(v))
    return v


def _int(v, what):
    if isinstance(v, int):
        return v
    if isinstance(v, Fraction) and v.denominator == 1:
        return v.numerator
    if isinstance(v, (Decimal, float)) and v == int(v):
        return int(v)
    raise CalcError(f"{what} needs a whole number")


def _check(v):
    if isinstance(v, Fraction) and max(abs(v.numerator).bit_length(), v.denominator.bit_length()) > MAX_BITS:
        raise CalcError("the result is too large (over a million digits)")
    if isinstance(v, float) and not math.isfinite(v):
        raise CalcError("the result is out of range")
    return v


def _power(a, b):
    if isinstance(a, Fraction) and isinstance(b, Fraction) and b.denominator == 1:
        n = b.numerator
        if a == 0 and n < 0:
            raise CalcError("division by zero")
        size = max(abs(a.numerator).bit_length(), a.denominator.bit_length()) * abs(n)
        if size > MAX_BITS:
            raise CalcError("the result is too large (over a million digits)")
        return a ** n
    if isinstance(a, Fraction) and isinstance(b, Fraction) and a >= 0:
        r = _root_exact(a, b.denominator)  # 8**(1/3) and 4**1.5 stay exact
        if r is not None:
            return _power(r, Fraction(b.numerator))
    x, y = _dec(a), _dec(b)
    if x < 0:
        raise CalcError("a negative number to a fractional power has no real result")
    with localcontext() as c:
        c.prec = PREC + 5
        return x ** y


def _root_exact(a, n):
    """The exact n-th root of a non-negative fraction, or None."""
    def iroot(k):
        r = round(k ** (1.0 / n)) if k.bit_length() < 1000 else _iroot_big(k, n)
        for c in (r - 1, r, r + 1):
            if c >= 0 and c ** n == k:
                return c
        return None
    p, q = iroot(a.numerator), iroot(a.denominator)
    return Fraction(p, q) if p is not None and q is not None else None


def _iroot_big(k, n):
    lo, hi = 0, 1 << (k.bit_length() // n + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid ** n <= k:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _sqrt(x):
    if x < 0:
        raise CalcError("square root of a negative number")
    if isinstance(x, Fraction):
        r = _root_exact(x, 2)
        if r is not None:
            return r
    with localcontext() as c:
        c.prec = PREC + 5
        return _dec(x).sqrt()


def _nth_root(x, n):
    n = _int(n, "root")
    if n <= 0:
        raise CalcError("root needs a positive degree")
    if isinstance(x, Fraction):
        r = _root_exact(abs(x), n)
        if r is not None and (x >= 0 or n % 2):
            return r if x >= 0 else -r
    if x < 0:
        if n % 2 == 0:
            raise CalcError("even root of a negative number")
        return -_power(-x, Fraction(1, n))
    return _power(x, Fraction(1, n))


def _dec_fn(name):
    def f(x):
        d = _dec(x)
        if name in ("ln", "log10") and d <= 0:
            raise CalcError(f"{name} needs a positive number")
        with localcontext() as c:
            c.prec = PREC + 5
            return getattr(d, name)()
    return f


def _log(x, base=None):
    if base is None:
        return _dec_fn("ln")(x)
    lb = _dec_fn("ln")(base)
    if lb == 0:
        raise CalcError("log base 1")
    with localcontext() as c:
        c.prec = PREC + 5
        return _dec_fn("ln")(x) / lb


def _float_fn(fn, degrees_in=False, degrees_out=False):
    def f(*xs):
        vals = [float(v) for v in xs]
        if degrees_in:
            vals = [math.radians(v) for v in vals]
        try:
            r = fn(*vals)
        except (ValueError, OverflowError) as e:
            raise CalcError(str(e))
        return math.degrees(r) if degrees_out else r
    return f


def _round(x, n=0):
    n = _int(n, "round")
    q = Decimal(1).scaleb(-n)
    with localcontext() as c:
        c.prec = max(PREC, len(str(int(abs(_dec(x))))) + n + 5)
        r = _dec(x).quantize(q, rounding=ROUND_HALF_UP)  # half up, as on receipts: round(2.5) = 3
    return Fraction(r)


def _ints(name, fn):
    return lambda *xs: Fraction(fn(*[_int(v, name) for v in xs]))


def _factorial(n):
    n = _int(n, "factorial")
    if n < 0 or n > 20000:
        raise CalcError("factorial takes 0 to 20000")
    return Fraction(math.factorial(n))


def _comb_perm(fn):
    def f(n, k):
        n, k = _int(n, "comb/perm"), _int(k, "comb/perm")
        if n > 200000:
            raise CalcError("comb/perm take n up to 200000")
        return Fraction(fn(n, k))
    return f


def _stat(fn):
    def f(*xs):
        if not xs:
            raise CalcError("needs at least one number")
        try:
            r = fn([x if isinstance(x, Fraction) else Fraction(_dec(x)) for x in xs])
        except statistics.StatisticsError as e:
            raise CalcError(str(e))
        return r if isinstance(r, Fraction) else float(r)
    return f


FUNCS = {
    "sqrt": _sqrt, "cbrt": lambda x: _nth_root(x, Fraction(3)), "root": _nth_root,
    "exp": _dec_fn("exp"), "ln": _dec_fn("ln"), "log": _log, "log10": _dec_fn("log10"),
    "log2": lambda x: _log(x, Fraction(2)),
    "sin": _float_fn(math.sin), "cos": _float_fn(math.cos), "tan": _float_fn(math.tan),
    "asin": _float_fn(math.asin), "acos": _float_fn(math.acos), "atan": _float_fn(math.atan),
    "atan2": _float_fn(math.atan2), "sinh": _float_fn(math.sinh), "cosh": _float_fn(math.cosh),
    "tanh": _float_fn(math.tanh), "hypot": _float_fn(math.hypot),
    "sind": _float_fn(math.sin, True), "cosd": _float_fn(math.cos, True), "tand": _float_fn(math.tan, True),
    "asind": _float_fn(math.asin, degrees_out=True), "acosd": _float_fn(math.acos, degrees_out=True),
    "atand": _float_fn(math.atan, degrees_out=True),
    "degrees": _float_fn(math.degrees), "radians": _float_fn(math.radians),
    "factorial": _factorial, "comb": _comb_perm(math.comb), "perm": _comb_perm(math.perm),
    "gcd": _ints("gcd", math.gcd), "lcm": _ints("lcm", math.lcm),
    "abs": abs, "floor": lambda x: Fraction(math.floor(x)), "ceil": lambda x: Fraction(math.ceil(x)),
    "round": _round, "min": min, "max": max, "sum": lambda *xs: sum(xs, Fraction(0)),
    "mean": _stat(statistics.mean), "median": _stat(statistics.median), "stdev": _stat(statistics.stdev),
    "pstdev": _stat(statistics.pstdev), "variance": _stat(statistics.variance),
}
CONSTS = {"pi": Decimal("3.14159265358979323846264338327950288419716939937510"),
          "e": Decimal("2.71828182845904523536028747135266249775724709369995"),
          "tau": Decimal("6.28318530717958647692528676655900576839433879875021")}


def _num(v):
    """Python literal -> exact Fraction (0.1 is exactly one tenth, not the nearest binary float)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise CalcError("only numbers are allowed")
    return Fraction(v) if isinstance(v, int) else Fraction(repr(v))


def _arith(op, a, b):
    if isinstance(op, ast.Pow):
        return _power(a, b)
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        x, y = a, b
    elif isinstance(a, float) or isinstance(b, float):
        x, y = float(a), float(b)
    else:
        x, y = _dec(a), _dec(b)
    if isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)) and y == 0:
        raise CalcError("division by zero")
    with localcontext() as c:
        c.prec = PREC + 5
        if isinstance(op, ast.Add):
            return x + y
        if isinstance(op, ast.Sub):
            return x - y
        if isinstance(op, ast.Mult):
            return x * y
        if isinstance(op, ast.Div):
            return x / y
        if isinstance(op, ast.FloorDiv):
            return Fraction(math.floor(x / y))
        if isinstance(op, ast.Mod):
            return x % y
    raise CalcError("unsupported operator")


def _eval(node, names):
    if isinstance(node, ast.Expression):
        return _eval(node.body, names)
    if isinstance(node, ast.Constant):
        return _num(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand, names)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        return _check(_arith(node.op, _eval(node.left, names), _eval(node.right, names)))
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in CONSTS:
            return CONSTS[node.id]
        raise CalcError(f"unknown name '{node.id}'")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        fn = FUNCS.get(node.func.id)
        if not fn:
            raise CalcError(f"unknown function '{node.func.id}'")
        try:
            return _check(fn(*[_eval(a, names) for a in node.args]))
        except TypeError:
            raise CalcError(f"wrong number of values for {node.func.id}()")
    raise CalcError("only numbers, + - * / // % **, brackets and the listed functions are allowed")


def normalise(expr):
    s = str(expr).strip()
    s = s.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
    s = re.sub(r"(?<=[\d)])\s*[xX]\s*(?=[\d(])", "*", s)  # 876 x 88894
    s = re.sub(r"\bmod\b", "%", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\w(.])", r"(\1/100)", s)  # 15% -> (15/100); 17 % 5 stays modulo
    s = re.sub(r"\b(\d{1,3}(?:,\d{3})+)(?![\d.])", lambda m: m.group(1).replace(",", "")
               if not re.search(r"[a-z_]\w*\([^)]*$", s[:m.start()]) else m.group(1), s)  # 1,234,567 outside f(...)
    s = re.sub(r"(\d+)\s*!(?!=)", r"factorial(\1)", s)
    return s


def fmt(v):
    """A result as text: whole numbers in full, fractions exact where possible."""
    if isinstance(v, Fraction):
        if v.denominator == 1:
            digits = str(v.numerator)
            n = len(digits.lstrip("-"))
            if n > MAX_SHOWN:
                return {"result": f"{digits[:60]}...{digits[-60:]}", "digits": n,
                        "scientific": f"{digits[0] if digits[0] != '-' else digits[:2]}.{digits.lstrip('-')[1:16]}e+{n - 1}",
                        "exact": True, "note": f"a {n}-digit whole number; first and last 60 digits shown"}
            out = {"result": digits, "exact": True}
            if 4 < n <= 40:
                out["readable"] = f"{v.numerator:,}"
            return out
        den = v.denominator
        for p in (2, 5):
            while den % p == 0:
                den //= p
        if den == 1:  # terminates: show every digit
            with localcontext() as c:
                c.prec = len(str(abs(v.numerator))) + len(str(v.denominator)) + 5
                d = Decimal(v.numerator) / Decimal(v.denominator)
            return {"result": format(d, "f"), "exact": True}
        out = {"result": _sig(_dec(v), 30), "exact": False, "note": "rounded to 30 significant digits"}
        if len(str(v)) <= 40:
            out["fraction"] = str(v)
        return out
    if isinstance(v, Decimal):
        return {"result": _sig(v, 30), "exact": False, "note": "rounded to 30 significant digits"}
    return {"result": _sig(Decimal(repr(v)), 15), "exact": False, "note": "about 15 significant digits"}


def _sig(d, n):
    if d == 0:
        return "0"
    with localcontext() as c:
        c.prec = n
        d = +d
    s = format(d, "f") if -12 < d.adjusted() < 40 else format(d, "e")
    if "." in s and "e" not in s:
        s = s.rstrip("0").rstrip(".")
    return s


# ---- numbers in words and Indian grouping (the model spelled 266,473,922,476,416 as "twenty-six quadrillion")
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
INTERNATIONAL = ["", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion", "sextillion",
                 "septillion", "octillion", "nonillion", "decillion"]
TRADITIONAL = [(10 ** 17, "shankh"), (10 ** 15, "padma"), (10 ** 13, "neel"), (10 ** 11, "kharab"),
               (10 ** 9, "arab"), (10 ** 7, "crore"), (10 ** 5, "lakh"), (10 ** 3, "thousand")]


def _under_1000(n):
    out = []
    if n >= 100:
        out += [ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        out.append(TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else ""))
    elif n or not out:
        out.append(ONES[n])
    return " ".join(out)


def words_international(n):
    if n == 0:
        return "zero"
    groups, parts = [], []
    while n:
        groups.append(n % 1000)
        n //= 1000
    if len(groups) > len(INTERNATIONAL):
        raise CalcError("too large to write in words")
    for i in reversed(range(len(groups))):
        if groups[i]:
            parts.append(_under_1000(groups[i]) + (" " + INTERNATIONAL[i] if i else ""))
    return ", ".join(parts)


def words_indian(n, sep=", "):
    """Lakh and crore, repeated for larger numbers: 2,66,47,392 crore, 24 lakh, 76 thousand, 416."""
    if n == 0:
        return "zero"
    parts = []
    if n >= 10 ** 7:
        parts.append(words_indian(n // 10 ** 7, " ") + " crore")  # the crore count reads as one phrase
        n %= 10 ** 7
    for scale, name in ((10 ** 5, "lakh"), (10 ** 3, "thousand")):
        if n >= scale:
            parts.append(_under_1000(n // scale) + " " + name)
            n %= scale
    if n:
        parts.append(_under_1000(n))
    return sep.join(parts)


def words_traditional(n):
    """The traditional ladder: thousand, lakh, crore, arab, kharab, neel, padma, shankh."""
    if n == 0:
        return "zero"
    parts = []
    for scale, name in TRADITIONAL:
        if n >= scale:
            count = n // scale
            parts.append((words_indian(count, " ") if count >= 100 else _under_1000(count)) + " " + name)
            n %= scale
    if n:
        parts.append(_under_1000(n))
    return ", ".join(parts)


def indian_grouping(n):
    s = str(abs(n))
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ("-" if n < 0 else "") + ",".join(groups + [tail])


def in_words(v, style):
    """Words and grouping for a result, in the Indian or international system (or both)."""
    if not isinstance(v, Fraction):
        v = Fraction(_dec(v))
    whole = abs(v.numerator) // v.denominator
    frac = fmt(abs(v) - whole)["result"] if v.denominator != 1 else ""
    sign = "minus " if v < 0 else ""
    point = (" point " + " ".join(ONES[int(d)] for d in frac.split(".")[1][:12])) if "." in frac else ""
    out = {}
    if style in ("indian", "words"):
        out["indian_grouping"] = ("-" if v < 0 else "") + indian_grouping(whole) + (frac[1:] if point else "")
        out["words_indian"] = sign + words_indian(whole) + point
        if whole >= 10 ** 9:
            out["words_indian_traditional"] = sign + words_traditional(whole) + point
    if style in ("international", "words"):
        out["international_grouping"] = ("-" if v < 0 else "") + f"{whole:,}" + (frac[1:] if point else "")
        out["words_international"] = sign + words_international(whole) + point
    return out


def calculate(expression, format=""):
    """Evaluate one or more steps (separated by ; or new lines); "name = ..." keeps a value for later steps.
    format: "indian", "international" or "words" (both) adds the result's grouping and words."""
    text = str(expression or "")
    if not text.strip():
        return {"error": "nothing to calculate"}
    if len(text) > MAX_LEN:
        return {"error": f"expression too long (over {MAX_LEN} characters)"}
    names, steps = {}, []
    parts = [p.strip() for p in re.split(r"[;\n]", text) if p.strip()]
    for part in parts:
        m = re.match(r"^([A-Za-z_]\w*)\s*=(?!=)\s*(.+)$", part)
        name, expr = (m.group(1), m.group(2)) if m else (None, part)
        expr = expr.rstrip("= ").strip()
        if name and (name in FUNCS or name in CONSTS):
            return {"error": f"'{name}' is a built-in name; pick another", "steps": steps}
        try:
            value = _eval(ast.parse(normalise(expr), mode="eval"), names)
        except SyntaxError:
            return {"error": f"couldn't read '{expr}' as a calculation", "steps": steps}
        except (CalcError, ZeroDivisionError, OverflowError, ValueError) as e:
            msg = "division by zero" if isinstance(e, ZeroDivisionError) else str(e)
            return {"error": f"{msg} (in '{expr}')", "steps": steps}
        if name:
            names[name] = value
        steps.append({"step": part, **fmt(value)})
    out = dict(steps[-1])
    out.pop("step")
    if format in ("indian", "international", "words"):
        try:
            out.update(in_words(value, format))
            out.pop("readable", None)
        except CalcError as e:
            out["words_error"] = str(e)
    if len(steps) > 1:
        out["steps"] = steps
    else:
        out["expression"] = parts[0]
    return out
