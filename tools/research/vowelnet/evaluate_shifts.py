"""How the step-2 shifts (rule.shifts: raise, lower, drawl) do, on the recognizer's input as the live stream sees it:
  takes held out of training (wav + plan-<name>.tsv; their windows are computed here, with 1 s and 150 ms of
  context, and cached): of the EH-before-nasal, EY and AE vowels how many get their shift, and shifts on others;
  real vocals (stream_windows.py --real): words whose stressed vowel is a target (CMU dictionary) and shifts on
  other words.
usage: evaluate_shifts.py model.pt cmudict.gz contentvec.onnx cache_dir [--takes wav...] [--real vocal.npz...] [--tune]"""
import sys, os, csv, gzip, re, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rule

args = sys.argv[1:]
net, classes, mu, sd = rule.load(args[0])
lines = gzip.open(args[1], "rt", encoding="latin-1").read().split("\n"); syms = lines[0].split()[2:]
lex = {w: [syms[ord(ch) - 0x30] for ch in c] for w, c in (l.split("\t") for l in lines[1:] if "\t" in l)}
NAMES = ("raise", "lower", "drawl")

def target(word):
    ph = lex.get(re.sub(r"[^a-z']", "", word.lower()), [])
    for k, p in enumerate(ph):
        if p[-1] not in "12": continue
        if p[:2] == "EH" and k + 1 < len(ph) and ph[k + 1] in ("N", "M"): return 0
        if p[:2] == "EY": return 1
        if p[:2] == "AE": return 2
    return -1

def option(name):
    if name not in args: return []
    out = []
    for a in args[args.index(name) + 1:]:
        if a.startswith("--"): break
        out.append(a)
    return out

takes, real = [], []
cache = args[3]; os.makedirs(cache, exist_ok=True)
for wav in option("--takes"):
    name = os.path.basename(wav)[:-4]; plan = os.path.join(os.path.dirname(wav), "plan-" + name + ".tsv")
    rows = list(csv.DictReader(open(plan), delimiter="\t"))
    segs = []
    for i, r in enumerate(rows):
        ph = r["ph"]
        if ph in ("SIL", "BR") or not any(ch in ph for ch in "AEIOU"): continue
        nxt = rows[i + 1]["ph"] if i + 1 < len(rows) and rows[i + 1]["word"] == r["word"] else ""
        lab = "EH_N" if ph == "EH" and nxt in ("N", "M", "NG") else ph
        segs.append((int(float(r["start"]) * 50), int(float(r["end"]) * 50), lab, r["word"]))
    for ctx in (1000, 150):
        path = os.path.join(cache, "%s-%s-%d.npy" % (os.path.basename(os.path.dirname(wav)), name, ctx))
        if not os.path.exists(path):
            import stream_windows
            np.save(path, stream_windows.windows(wav, ctx, args[2]))
        W = np.load(path); voiced = np.zeros(len(W), bool)
        for r in rows:
            if r["ph"] not in ("SIL", "BR"): voiced[int(float(r["start"]) * 50):int(float(r["end"]) * 50) + 1] = True
        takes.append(("%s@%d" % (name, ctx), rule.posteriors(net, mu, sd, W=W), voiced[:len(W)], segs))
for path in option("--real"):
    d = np.load(path, allow_pickle=True)
    real.append((os.path.basename(path), rule.posteriors(net, mu, sd, W=d["W"]), d["voiced"], [(w, float(a), float(b)) for w, a, b in d["words"]]))

def score(params):
    r = dict(hit=np.zeros(3, int), n=np.zeros(3, int), false=0, others=0, rhit=np.zeros(3, int), rn=np.zeros(3, int), rfalse=0, notes=[])
    for name, P, voiced, segs in takes:
        w = rule.shifts(P, classes, voiced, *params)
        for a, b, lab, word in segs:
            if b - a < 5: continue
            seg = w[a:b + 1]; applied = int(seg.max(0).argmax()) if seg.max() > 0.3 else -1
            t = {"EH_N": 0, "EY": 1, "AE": 2}.get(lab, -1)
            if t >= 0:
                r["n"][t] += 1; r["hit"][t] += applied == t
            else:
                r["others"] += 1
                if applied >= 0: r["false"] += 1; r["notes"].append("%s %s(%s)->%s" % (name, word, lab, NAMES[applied]))
    for name, P, voiced, words in real:
        w = rule.shifts(P, classes, voiced, *params)
        for word, a, b in words:
            seg = w[int(a * 50):int(b * 50) + 1]
            applied = int(seg.max(0).argmax()) if len(seg) and seg.max() > 0.3 else -1
            t = target(word)
            if t >= 0:
                r["rn"][t] += 1; r["rhit"][t] += applied == t
                if applied != t: r["notes"].append("%s %s: missed %s" % (name, word, NAMES[t]) + ("" if applied < 0 else ", got " + NAMES[applied]))
            elif applied >= 0: r["rfalse"] += 1; r["notes"].append("%s %s -> %s" % (name, word, NAMES[applied]))
    return r

def show(params, r):
    print("TR %.2f TL %.2f TD %.2f | takes: raise %d/%d lower %d/%d drawl %d/%d, false %d of %d | real: raise %d/%d lower %d/%d drawl %d/%d, false %d" % (
        *params[:3], r["hit"][0], r["n"][0], r["hit"][1], r["n"][1], r["hit"][2], r["n"][2], r["false"], r["others"],
        r["rhit"][0], r["rn"][0], r["rhit"][1], r["rn"][1], r["rhit"][2], r["rn"][2], r["rfalse"]))

defaults = (rule.TR, rule.TL, rule.TD)
r = score(defaults); show(defaults, r)
for note in r["notes"][:60]: print("  " + note)
if "--tune" in args:
    res = [(p, score(p)) for p in itertools.product([0.3, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6])]
    res.sort(key=lambda x: -(x[1]["hit"].sum() / max(1, x[1]["n"].sum()) + x[1]["rhit"].sum() / max(1, x[1]["rn"].sum())
                             - 3 * x[1]["false"] / max(1, x[1]["others"]) - x[1]["rfalse"] / 10))
    for p, r in res[:10]: show(p, r)
