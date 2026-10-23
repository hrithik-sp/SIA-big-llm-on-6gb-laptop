"""Figures for README.md and REPORT.md, as light and dark SVGs in docs/images/.

  python tools/make_figures.py

Every number below is a measured result; the comment names its lab-notebook finding.
Colours: the dataviz reference palette (categorical slots 1-3 and the blue/red diverging
pair), validated for colour-vision deficiency in both modes; every mark is also labelled.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", s=["#2a78d6", "#eb6834", "#1baf7a"], up="#2a78d6", down="#e34948"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", s=["#3987e5", "#d95926", "#199e70"], up="#3987e5", down="#e66767"),
}

# (label, detail, decode tok/s, MMLU-Pro %, standard error %)
TRADEOFF = [
    ("gpt-oss-20b", "released MXFP4, nothing shrunk", 27.7, 70.5, 3.1),   # findings 12, 51
    ("Gemma 4 26B QAT", "Google's 4-bit-trained, + MTP", 31.3, 79.0, 2.8),  # findings 42, 48
    ("Gemma 4 26B Q8_0", "near-lossless, streamed from SSD, + MTP", 12.0, 83.8, 2.5),  # findings 48, 50
]
# gpt-oss-20b decode, tok/s (findings 1-12)
SPEEDUP = [("llama.cpp defaults", 20.6),
           ("+ tuned placement and load mode", 23.6),
           ("+ Windows Best performance mode", 27.7)]
# speculative decoding speed-up vs the same setup without a draft (findings 35, 50)
SPEC = [("gpt-oss-20b + EAGLE-3 draft", 0.78),
        ("Gemma 4 QAT + its MTP head", 1.32),
        ("Gemma 4 Q8_0 + its MTP head", 1.25)]
# MMLU-Pro questions correct out of 15 per subject: (gpt-oss, Gemma QAT, Gemma Q8_0) (findings 48, 51)
SUBJECTS = {"biology": (13, 14, 14), "business": (10, 12, 13), "chemistry": (14, 13, 15),
            "computer science": (12, 12, 12), "economics": (10, 13, 13), "engineering": (7, 9, 12),
            "health": (12, 11, 12), "history": (7, 13, 13), "law": (7, 9, 9), "math": (13, 14, 14),
            "other": (11, 12, 13), "philosophy": (8, 9, 10), "physics": (11, 11, 12),
            "psychology": (13, 14, 14)}
# warm decode tok/s with other apps holding RAM (finding 40; tools/pressure_report.py --latest)
PRESSURE = [("gpt-oss-20b", [(0, 26.7), (2, 26.0), (4, 25.7), (6, 25.3)]),
            ("Gemma 4 QAT", [(0, 27.3), (2, 26.9), (4, 25.0), (6, 14.9)])]
# decode tok/s with the context already filled, thousands of tokens (finding 39)
LONGCTX = [("gpt-oss-20b", [(0, 25.67), (8, 25.78), (16, 24.61), (32, 22.32)]),
           ("Gemma 4 QAT", [(8, 25.78), (16, 24.90), (32, 22.76)])]  # its cold 0k run is invalid
# chat-corpus KL-divergence vs Q8_0: (version, mean, 99.9th percentile, is QAT) (finding 31)
FIDELITY = [("UD-Q3_K_M (lossy)", 0.092, 8.05, False), ("UD-IQ3_XXS (lossy)", 0.136, 11.57, False),
            ("UD-IQ2_M (lossy)", 0.203, 14.20, False), ("QAT Q4_0 (Google)", 0.207, 7.41, True)]
# decode change with 16 threads instead of 8, % (findings 47, 49)
THREADS = [("gpt-oss-20b, in RAM", -10.1), ("Gemma 4 QAT, warm", -0.8),
           ("Gemma 4 QAT, first answer (cold)", 32.6), ("Gemma 4 Q8_0, streamed from SSD", 21.5)]


def text(x, y, s, fill, size=13, weight=400, anchor="start"):
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{s}</text>')


def svg(w, h, t, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{title}" font-family=\'{FONT}\'>'
            f'<rect width="{w}" height="{h}" rx="12" fill="{t["surface"]}"/>{body}</svg>\n')


def hbar(x0, x1, y, h, fill):
    """Horizontal bar anchored at x0 with a 4px rounded data-end at x1."""
    r = min(4, abs(x1 - x0) / 2)
    if x1 >= x0:
        d = (f"M{x0},{y} H{x1 - r} Q{x1},{y} {x1},{y + r} V{y + h - r} Q{x1},{y + h} {x1 - r},{y + h} "
             f"H{x0} Z")
    else:
        d = (f"M{x0},{y} H{x1 + r} Q{x1},{y} {x1},{y + r} V{y + h - r} Q{x1},{y + h} {x1 + r},{y + h} "
             f"H{x0} Z")
    return f'<path d="{d}" fill="{fill}"/>'


def tradeoff(t):
    W, H, L, R, T, B = 760, 440, 70, 40, 78, 70
    xmin, xmax, ymin, ymax = 0, 40, 62, 90
    X = lambda v: L + (v - xmin) / (xmax - xmin) * (W - L - R)
    Y = lambda v: T + (ymax - v) / (ymax - ymin) * (H - T - B)
    b = [text(24, 34, "Intelligence vs speed on a 6 GB laptop GPU", t["ink"], 17, 600),
         text(24, 56, "MMLU-Pro accuracy (210 questions, ±1 s.e.) vs generation speed, best measured setup",
              t["ink2"], 13)]
    for v in range(65, 91, 5):
        b.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        b.append(text(L - 10, Y(v) + 4, f"{v}%", t["muted"], 12, anchor="end"))
    for v in range(0, 41, 10):
        b.append(text(X(v), H - B + 22, str(v), t["muted"], 12, anchor="middle"))
    b.append(f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke="{t["axis"]}" stroke-width="1"/>')
    b.append(text((L + W - R) / 2, H - 22, "tokens per second (higher is faster; ~25 is faster than reading)",
                  t["muted"], 12, anchor="middle"))
    places = {0: ("end", -14, 4), 1: ("end", -14, 4), 2: ("start", 14, 4)}
    for i, (name, detail, tps, acc, se) in enumerate(TRADEOFF):
        c, cx, cy = t["s"][i], X(tps), Y(acc)
        b.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{Y(acc + se):.1f}" y2="{Y(acc - se):.1f}" '
                 f'stroke="{c}" stroke-width="2" stroke-linecap="round" opacity="0.55"/>')
        b.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{c}" stroke="{t["surface"]}" stroke-width="2"/>')
        anchor, dx, dy = places[i]
        b.append(text(cx + dx, cy + dy - 8, f"{name}  {acc:.1f}% · {tps:.1f} tok/s", t["ink"], 13, 600, anchor))
        b.append(text(cx + dx, cy + dy + 9, detail, t["ink2"], 12, anchor=anchor))
    return svg(W, H, t, "".join(b), "Intelligence vs speed")


def bars(t, rows, title, sub, fmt, lo, hi, base, ticks, note, tick_fmt=None):
    W, L, R, T, row = 760, 280, 70, 80, 46
    H = T + row * len(rows) + 56
    X = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    b = [text(24, 34, title, t["ink"], 17, 600), text(24, 56, sub, t["ink2"], 13)]
    for v in ticks:
        b.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T - 8}" y2="{T + row * len(rows)}" '
                 f'stroke="{t["grid"]}" stroke-width="1"/>')
        b.append(text(X(v), T + row * len(rows) + 18, (tick_fmt or fmt)(v), t["muted"], 12, anchor="middle"))
    for i, (name, v) in enumerate(rows):
        y = T + i * row + 8
        fill = t["s"][0] if base == lo else (t["up"] if v >= base else t["down"])
        b.append(hbar(X(base), X(v), y, 24, fill))
        b.append(text(L - 14, y + 17, name, t["ink"], 13, anchor="end"))
        b.append(text(X(v) + (8 if v >= base else -8), y + 17, fmt(v), t["ink"], 13, 600,
                      "start" if v >= base else "end"))
    b.append(f'<line x1="{X(base):.1f}" x2="{X(base):.1f}" y1="{T - 8}" y2="{T + row * len(rows)}" '
             f'stroke="{t["axis"]}" stroke-width="1.5"/>')
    b.append(text(24, H - 14, note, t["muted"], 12))
    return svg(W, H, t, "".join(b), title)


def legend(t, x, y, items):
    """One row of swatch + name; items = [(name, colour)]."""
    out = []
    for name, c in items:
        out.append(f'<circle cx="{x + 5}" cy="{y - 4}" r="5" fill="{c}"/>')
        out.append(text(x + 16, y, name, t["ink2"], 12))
        x += 26 + 7 * len(name)
    return "".join(out)


def lines(t, series, title, sub, xlab, xticks, xfmt, ymax, note, label_dy):
    W, H, L, R, T, B = 760, 400, 60, 150, 92, 64
    xmax = max(xticks)
    X = lambda v: L + v / xmax * (W - L - R)
    Y = lambda v: T + (ymax - v) / ymax * (H - T - B)
    b = [text(24, 34, title, t["ink"], 17, 600), text(24, 56, sub, t["ink2"], 13),
         legend(t, 24, 80, [(n, t["s"][i]) for i, (n, _) in enumerate(series)])]
    for v in range(0, int(ymax) + 1, 10):
        b.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{t["grid"]}"/>')
        b.append(text(L - 10, Y(v) + 4, str(v), t["muted"], 12, anchor="end"))
    for v in xticks:
        b.append(text(X(v), H - B + 22, xfmt(v), t["muted"], 12, anchor="middle"))
    b.append(f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke="{t["axis"]}"/>')
    b.append(text((L + W - R) / 2, H - 20, xlab, t["muted"], 12, anchor="middle"))
    for i, (name, pts) in enumerate(series):
        c = t["s"][i]
        d = " ".join(f"{'M' if j == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for j, (x, y) in enumerate(pts))
        b.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round"/>')
        for x, y in pts:
            b.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{c}" stroke="{t["surface"]}" stroke-width="2"/>')
        x, y = pts[-1]
        b.append(text(X(x) + 10, Y(y) + 4 + label_dy[i], f"{name} {y:.1f}", t["ink"], 12, 600))
    b.append(text(24, H - 2, note, t["muted"], 11))
    return svg(W, H + 8, t, "".join(b), title)


def subjects(t):
    rows = sorted(SUBJECTS.items(), key=lambda kv: kv[1][2] - kv[1][0], reverse=True)
    W, L, R, T, row = 760, 150, 40, 96, 26
    H = T + row * len(rows) + 50
    X = lambda v: L + (v - 5) / 10 * (W - L - R)
    names = ["gpt-oss-20b", "Gemma 4 QAT", "Gemma 4 Q8_0"]
    b = [text(24, 34, "Where the intelligence gap is: MMLU-Pro by subject", t["ink"], 17, 600),
         text(24, 56, "Questions correct out of 15 per subject, sorted by the gap between gpt-oss and Gemma Q8_0",
              t["ink2"], 13), legend(t, 24, 80, [(n, t["s"][i]) for i, n in enumerate(names)])]
    for v in range(5, 16):
        b.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T - 8}" y2="{T + row * len(rows)}" stroke="{t["grid"]}"/>')
        b.append(text(X(v), T + row * len(rows) + 18, str(v), t["muted"], 12, anchor="middle"))
    for r, (subj, vals) in enumerate(rows):
        y = T + r * row + row / 2
        b.append(text(L - 14, y + 4, subj, t["ink"], 12, anchor="end"))
        lo, hi = min(vals), max(vals)
        b.append(f'<line x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{t["axis"]}" stroke-width="2"/>')
        for i, v in enumerate(vals):
            b.append(f'<circle cx="{X(v):.1f}" cy="{y + (i - 1) * 5:.1f}" r="5" fill="{t["s"][i]}" '
                     f'stroke="{t["surface"]}" stroke-width="2"/>')
    b.append(text(24, H - 14, "210 questions, temperature 0, thinking off for Gemma (findings 48, 51). "
                  "Totals: 148, 166 and 176 of 210.", t["muted"], 11))
    return svg(W, H, t, "".join(b), "MMLU-Pro by subject")


def fidelity(t):
    W, H, T, row = 760, 290, 96, 34
    b = [text(24, 34, "How far each version's predictions are from the original", t["ink"], 17, 600),
         text(24, 56, "KL-divergence vs the near-lossless Q8_0 on chat text (lower = closer)", t["ink2"], 13),
         legend(t, 24, 80, [("post-hoc quantization (lossy)", t["s"][0]), ("Google QAT (4-bit-trained)", t["s"][1])])]
    panels = [("average token", 1, 0.25, lambda v: f"{v:.3f}"), ("worst 0.1% of tokens", 2, 16, lambda v: f"{v:.1f}")]
    for k, (ptitle, col, hi, fmt) in enumerate(panels):
        x0 = 190 + k * 290
        wid = 210
        b.append(text(x0, T + 4, ptitle, t["ink2"], 12, 600))
        for r, row_ in enumerate(FIDELITY):
            name, v, qat = row_[0], row_[col], row_[3]
            y = T + 16 + r * row
            if k == 0:
                b.append(text(176, y + 15, name, t["ink"], 12, anchor="end"))
            b.append(hbar(x0, x0 + v / hi * wid, y, 20, t["s"][1] if qat else t["s"][0]))
            b.append(text(x0 + v / hi * wid + 6, y + 15, fmt(v), t["ink"], 12, 600))
        b.append(f'<line x1="{x0}" x2="{x0}" y1="{T + 10}" y2="{T + 16 + 4 * row}" stroke="{t["axis"]}"/>')
    b.append(text(24, H - 34, "QAT differs as much as 2-bit on average, but its worst cases are half as far: "
                  "a re-trained model, not a damaged copy.", t["ink2"], 12))
    b.append(text(24, H - 14, "~37 chunks of 512 tokens in the model's own chat template (findings 30-32)", t["muted"], 11))
    return svg(W, H, t, "".join(b), "Fidelity to the original")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        figs = {
            "tradeoff": tradeoff(t),
            "gptoss-speedup": bars(t, SPEEDUP, "gpt-oss-20b: +34% from settings alone",
                                   "Generation speed, same model and same output (tok/s)",
                                   lambda v: f"{v:.1f}", 0, 30, 0, range(0, 31, 10),
                                   "llama-bench, 5 repetitions × 2 shuffled rounds, AC power (findings 1–12)",
                                   tick_fmt=str),
            "speculative": bars(t, SPEC, "Speculative decoding helps Gemma, hurts gpt-oss",
                                "Speed with a draft model vs without (1.0× = no change)",
                                lambda v: f"{v:.2f}×", 0.6, 1.4, 1.0, [0.6, 0.8, 1.0, 1.2, 1.4],
                                "Verifying drafted tokens loads the union of their experts (findings 35–37, 50)"),
            "threads": bars(t, THREADS, "16 threads help only while weights stream from the SSD",
                            "Change in generation speed with 16 threads instead of 8",
                            lambda v: f"{v:+.1f}%", -20, 40, 0, [-20, -10, 0, 10, 20, 30, 40],
                            "More threads keep more SSD reads in flight (findings 47, 49)",
                            tick_fmt=lambda v: f"{v:+d}%" if v else "0%"),
            "subjects": subjects(t),
            "fidelity": fidelity(t),
            "ram-pressure": lines(t, PRESSURE, "Speed while other apps hold RAM",
                                  "Warm generation speed (tok/s), both models memory-mapped",
                                  "GiB of RAM held by other apps (about 10.4 GiB free at 0)", [0, 2, 4, 6],
                                  lambda v: f"+{v}" if v else "0", 30,
                                  "A helper touched every held page every 5 s, like active browser tabs (finding 40)", [-6, 4]),
            "long-context": lines(t, LONGCTX, "Speed with long documents in the context",
                                  "Generation speed (tok/s) after the context is pre-filled",
                                  "tokens already in the context (thousands)", [0, 8, 16, 32],
                                  lambda v: f"{v}k" if v else "0", 30,
                                  "A 16k-token document waits 23 s (gpt-oss) or 36 s (QAT) before the answer starts (finding 39)", [10, -4]),
        }
        for name, s in figs.items():
            (OUT / f"{name}-{mode}.svg").write_text(s, encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
