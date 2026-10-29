"""How the rule does with a trained recognizer:
  rendered takes held out of training (feats.npz from features.py, their plans in takes_dir): of the I / oy vowels
  how many are held, and holds on other vowels (AA aside: the country voices already sing "I" as "ah");
  real vocals (real_prep.py; with stream_windows.py --real, as the live stream sees them): words with a
  stressed I / oy held (CMU dictionary) and holds on other words.
--tune tries other thresholds (tune on real singing, e.g. VocalSet's "Row, row, row your boat", whose "life" is
the target; the rendered takes overstate what the recognizer can do).
usage: evaluate.py model.pt cmudict.gz [--takes feats.npz takes_dir prefix,...] [--real vocal.npz ...] [--tune]"""
import sys, os, csv, gzip, re, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rule

args = sys.argv[1:]
net, classes, mu, sd = rule.load(args[0])
lines = gzip.open(args[1], "rt", encoding="latin-1").read().split("\n"); syms = lines[0].split()[2:]
lex = {w: [syms[ord(ch) - 0x30] for ch in c] for w, c in (l.split("\t") for l in lines[1:] if "\t" in l)}
word = lambda w: re.sub(r"[^a-z']", "", w.lower())
is_target = lambda w: any(p[:2] in ("AY", "OY") and p[-1] in "12" for p in lex.get(word(w), []))

rendered, real = [], []
if "--takes" in args:
    k = args.index("--takes"); feats, takes, prefixes = args[k + 1], args[k + 2], args[k + 3].split(",")
    d = np.load(feats, allow_pickle=True); X, Y, G = d["X"], d["Y"], d["G"]
    for g in sorted(set(G)):
        if any(g.startswith(p) for p in prefixes):
            m = G == g
            rows = list(csv.DictReader(open(os.path.join(takes, "plan-" + g + ".tsv")), delimiter="\t"))
            segs = [(int(float(r["start"]) * 50), int(float(r["end"]) * 50), r["ph"], r["word"]) for r in rows
                    if r["ph"] not in ("SIL", "BR") and any(ch in r["ph"] for ch in "AEIOU")]
            rendered.append((g, rule.posteriors(net, mu, sd, X[m].astype(np.float32)), Y[m] != "SIL", segs))
if "--real" in args:
    for path in args[args.index("--real") + 1:]:
        if path.startswith("--"): break
        d = np.load(path, allow_pickle=True)
        P = rule.posteriors(net, mu, sd, W=d["W"]) if "W" in d else rule.posteriors(net, mu, sd, d["F"].astype(np.float32))
        real.append((os.path.basename(path), P, d["voiced"],
                     [(w, float(a), float(b)) for w, a, b in d["words"]]))

def score(params, report=False):
    r = dict(tp=0, n=0, fa=0, rtp=0, rn=0, rfa=0, rnv=0, false=[], missed=[])
    for name, P, voiced, words in real:
        held = set()
        for s, g, e in rule.holds(P, classes, voiced, *params):   # the word sung 80 ms before the glide
            t = (g - 4) / 50
            near = [i for i, (w, a, b) in enumerate(words) if a - 0.05 <= t <= b + 0.05]
            held.add(near[0] if near else min(range(len(words)), key=lambda i: min(abs(t - words[i][1]), abs(t - words[i][2]))))
        for i, (w, a, b) in enumerate(words):
            if is_target(w):
                r["n"] += 1; r["tp"] += i in held
                if i not in held: r["missed"].append("%s:%s" % (name, word(w)))
            elif i in held: r["fa"] += 1; r["false"].append("%s:%s" % (name, word(w)))
    for name, P, voiced, segs in rendered:
        hit = set()
        for s, g, e in rule.holds(P, classes, voiced, *params):
            seg = next((x for x in segs if x[0] - 1 <= g <= x[1] + 1), None)
            if seg is None: continue
            if seg[2] in ("AY", "OY", "AA"): hit.add(seg)
            else: r["rfa"] += 1; r["false"].append("%s:%s(%s)" % (name, word(seg[3]), seg[2]))
        for seg in segs:
            if seg[1] - seg[0] < 6: continue
            if seg[2] in ("AY", "OY"): r["rn"] += 1; r["rtp"] += seg in hit
            elif seg[2] != "AA": r["rnv"] += 1
    return r

def show(params, r):
    print("V0 %.2f G0 %.2f E0 %.2f N0 %.2f | real: %d/%d held, %d false | rendered: %d/%d held, %d false on %d vowels" % (
        *params, r["tp"], r["n"], r["fa"], r["rtp"], r["rn"], r["rfa"], r["rnv"]))

defaults = (rule.V0, rule.G0, rule.E0, rule.N0)
r = score(defaults); show(defaults, r)
print("  missed:", " ".join(r["missed"]) or "-"); print("  false: ", " ".join(r["false"]) or "-")
if "--tune" in args:
    res = [(p, score(p)) for p in itertools.product([0.4, 0.5], [0.3, 0.4, 0.5], [0.3, 0.4, 0.5], [0.2, 0.3, 0.4])]
    res.sort(key=lambda x: -(x[1]["tp"] / max(1, x[1]["n"]) + x[1]["rtp"] / max(1, x[1]["rn"]) - (x[1]["fa"] + x[1]["rfa"]) / 10))
    for p, r in res[:10]: show(p, r)
