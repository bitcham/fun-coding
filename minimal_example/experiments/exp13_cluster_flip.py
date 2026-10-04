"""실험 13: 같은 군집의 동사를 뒤집으면 옆 사람에게 전이되는가.

실험 10–12는 이탈자 문장 '짱구는 바나나를 싫어해'를 반복했다.
(철수는, 바나나를)의 P(싫어해)는 0.00x에서 움직이지 않았다.
짱구는 철수·영희와 다른 자리에 있으므로, 그 문장을 키울수록 오히려 더 멀어진다.

이번 실험은 철수와 같은 군집인 영희의 바나나 동사만 뒤집는다.

    원본:  영희는 바나나를 좋아해 ×1
    flip:  그 문장을 빼고  영희는 바나나를 싫어해 ×R
           R ∈ {1, 2, 5, 10, 20, 50, 100}

나머지 5문장은 그대로다.

핵심 비교 (full-batch GD, D=2, H=8, lr=0.3, epoch=2000):
- (영희는, 바나나를) → 싫어해   직접 학습. n-gram은 R≥1에서 1.
- (철수는, 바나나를) → 싫어해   ★ 사람 전이. n-gram은 unseen이라 1/V.
- (영희는, 사과를)   → 싫어해   과일 누출. 학습은 좋아해뿐.
- (철수는, 사과를)   → 싫어해   닻. 그대로 좋아해여야 한다.
"""

import numpy as np

from exp_common import RESULTS_DIR, count_sentences, write_table
from utils import setup_io, set_seed, setup_matplotlib_korean
from data import build_vocab, make_pairs, encode_pairs, SENTENCES
from ngram import NGramModel
from nlm import NLM, train

EMB_DIM = 2
HIDDEN = 8
LR = 0.3
EPOCHS = 2000
SEED = 123
N_SAMPLES = 10000

ORIG_BANANA = "영희는 바나나를 좋아해"
FLIP_BANANA = "영희는 바나나를 싫어해"
REPEATS = [0, 1, 2, 5, 10, 20, 50, 100]

PROBES = [
    ("영희는", "바나나를"),
    ("철수는", "바나나를"),
    ("영희는", "사과를"),
    ("철수는", "사과를"),
    ("짱구는", "사과를"),
]

WATCH_SENTENCES = [
    FLIP_BANANA,
    ORIG_BANANA,
    "철수는 바나나를 싫어해",
    "철수는 바나나를 좋아해",
    "영희는 사과를 싫어해",
]


def sentences_for(repeat):
    rest = [s for s in SENTENCES if s != ORIG_BANANA]
    if repeat == 0:
        return rest + [ORIG_BANANA]
    return rest + [FLIP_BANANA] * repeat


def p_word(model, stoi, ctx_pair, word):
    ctx = (stoi[ctx_pair[0]], stoi[ctx_pair[1]])
    if isinstance(model, NLM):
        p = model.predict_proba(np.array([list(ctx)], dtype=np.int64))[0]
    else:
        p = model.prob(ctx)
    return float(p[stoi[word]])


def train_one(stoi, itos, sentences):
    pairs = make_pairs(sentences=sentences)
    X, y = encode_pairs(pairs, stoi)
    set_seed(SEED)
    nlm = NLM(vocab_size=len(itos), emb_dim=EMB_DIM, hidden=HIDDEN, seed=SEED)
    history, _ = train(nlm, X, y, lr=LR, epochs=EPOCHS)
    ngram = NGramModel(vocab_size=len(itos), smoothing_k=0.0)
    ngram.fit(X, y)
    return nlm, ngram, history[-1], len(pairs)


def sample_counts(model, stoi, itos):
    rng = np.random.default_rng(42)
    samples = [model.generate(stoi, itos, rng) for _ in range(N_SAMPLES)]
    return count_sentences(samples)


def run():
    setup_io()
    plt = setup_matplotlib_korean()
    stoi, itos = build_vocab()

    print("== 실험 13: 영희 바나나 동사 뒤집기 → 철수 전이 ==")
    print(f"D={EMB_DIM} H={HIDDEN} lr={LR} epochs={EPOCHS} seed={SEED}")
    print()

    sweep_rows = []
    curves = {("nlm", a, b): [] for a, b in PROBES}
    curves.update({("ngram", a, b): [] for a, b in PROBES})
    labels = []
    gen_rows = []

    for r in REPEATS:
        sents = sentences_for(r)
        tag = "orig" if r == 0 else f"x{r}"
        nlm, ngram, loss, n_pairs = train_one(stoi, itos, sents)
        labels.append("원본" if r == 0 else f"×{r}")

        dist_e = float(np.linalg.norm(nlm.E[stoi["철수는"]] - nlm.E[stoi["영희는"]]))
        dist_j = float(np.linalg.norm(nlm.E[stoi["영희는"]] - nlm.E[stoi["짱구는"]]))

        print(f"[{tag}] pairs={n_pairs} loss={loss:.4f}  "
              f"|철수-영희|={dist_e:.3f}  |영희-짱구|={dist_j:.3f}")

        row = [tag, n_pairs, f"{loss:.4f}", f"{dist_e:.4f}", f"{dist_j:.4f}"]
        for a, b in PROBES:
            pn = p_word(nlm, stoi, (a, b), "싫어해")
            pg = p_word(ngram, stoi, (a, b), "싫어해")
            ln = p_word(nlm, stoi, (a, b), "좋아해")
            curves[("nlm", a, b)].append(pn)
            curves[("ngram", a, b)].append(pg)
            row.extend([f"{pn:.4f}", f"{ln:.4f}", f"{pg:.4f}"])
            print(f"    ({a}, {b})  NLM P(싫)={pn:.4f} P(좋)={ln:.4f}   "
                  f"n-gram P(싫)={pg:.4f}")
        sweep_rows.append(row)

        dist = sample_counts(nlm, stoi, itos)
        grow = [tag, "nlm"]
        for s in WATCH_SENTENCES:
            c = dist.get(s, 0)
            grow.append(c)
            print(f"    NLM  '{s}' = {c}")
        gen_rows.append(tuple(grow))
        print()

    header = ["setup", "n_pairs", "loss", "dist_chul_young", "dist_young_jjang"]
    for a, b in PROBES:
        header.extend([
            f"nlm_dislike({a},{b})",
            f"nlm_like({a},{b})",
            f"ngram_dislike({a},{b})",
        ])
    write_table(sweep_rows, f"{RESULTS_DIR}/exp13_sweep.tsv", header=header)
    write_table(
        gen_rows,
        f"{RESULTS_DIR}/exp13_novel_counts.tsv",
        header=["setup", "model"] + WATCH_SENTENCES,
    )

    # 전이 곡선: 직접 / 사람 전이 / 과일 누출
    show = [
        (("영희는", "바나나를"), "직접 (영희, 바나나)", "#dc2626"),
        (("철수는", "바나나를"), "전이 (철수, 바나나)", "#2563eb"),
        (("영희는", "사과를"), "누출 (영희, 사과)", "#16a34a"),
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, key in [(axes[0], "nlm"), (axes[1], "ngram")]:
        for ctx, name, color in show:
            ax.plot(x, curves[(key, ctx[0], ctx[1])], marker="o", color=color, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("P(싫어해)")
        ax.set_title("NLM" if key == "nlm" else "n-gram (unseen = 1/V)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("실험 13: 영희의 바나나를 싫어해로 뒤집었을 때 P(싫어해)", fontsize=12)
    fig.tight_layout()
    out = f"{RESULTS_DIR}/exp13_transfer.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"저장: {out}")
    print(f"저장: {RESULTS_DIR}/exp13_sweep.tsv")
    print("\n=== 실험 13 완료 ===")


if __name__ == "__main__":
    run()
