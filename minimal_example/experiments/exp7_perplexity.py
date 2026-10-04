"""실험 7: n-gram과 NLM의 perplexity 비교.

훈련 perplexity와, 학습에 없던 '일반화' 테스트 perplexity를 함께 본다.
"""

import numpy as np

from exp_common import RESULTS_DIR, write_table
from utils import setup_io, set_seed, setup_matplotlib_korean
from data import build_vocab, make_pairs, encode_pairs, CONTEXT_LEN, PAD, BOS, EOS, SENTENCES
from ngram import NGramModel
from nlm import NLM, train

EMB_DIM = 2
HIDDEN = 8
LR = 0.3
EPOCHS = 2000

# 학습에 없는 '말이 되는' 문장들 — NLM이 일반화로 잘 맞추는지 본다
NOVEL_SENTENCES = [
    "철수는 바나나를 좋아해",       # 핵심: 철수+바나나 조합
    "짱구는 사과를 싫어해",         # 짱구+사과 조합
    "영희는 사과를 싫어해",         # 영희+싫어해 조합
]


def make_test_pairs(sentences, stoi):
    pairs = []
    for s in sentences:
        toks = [PAD] * (CONTEXT_LEN - 1) + [BOS] + s.split() + [EOS]
        for i in range(len(toks) - CONTEXT_LEN):
            ctx = toks[i : i + CONTEXT_LEN]
            nxt = toks[i + CONTEXT_LEN]
            pairs.append(([stoi[t] for t in ctx], stoi[nxt]))
    X = np.array([p[0] for p in pairs], dtype=np.int64)
    y = np.array([p[1] for p in pairs], dtype=np.int64)
    return X, y


def run():
    setup_io()
    plt = setup_matplotlib_korean()

    stoi, itos = build_vocab()
    pairs = make_pairs()
    X_tr, y_tr = encode_pairs(pairs, stoi)
    X_test, y_test = make_test_pairs(NOVEL_SENTENCES, stoi)

    # n-gram (3 가지 smoothing)
    ngrams = {}
    for k in [0.0, 0.01, 0.5]:
        m = NGramModel(vocab_size=len(itos), smoothing_k=k)
        m.fit(X_tr, y_tr)
        ngrams[k] = m

    # NLM
    set_seed(123)
    nlm = NLM(vocab_size=len(itos), emb_dim=EMB_DIM, hidden=HIDDEN, seed=123)
    history, _ = train(nlm, X_tr, y_tr, lr=LR, epochs=EPOCHS)

    print("== 실험 7: perplexity 비교 ==\n")
    print(f"{'Model':<24}{'Train PPL':>14}{'Novel PPL':>20}")
    print("-" * 58)
    rows = []
    for k, m in ngrams.items():
        tr_ppl = m.perplexity(X_tr, y_tr)
        nv_ppl = m.perplexity(X_test, y_test)
        label = f"n-gram (k={k})"
        print(f"{label:<24}{tr_ppl:>14.4f}{nv_ppl:>20.4f}")
        rows.append((label, tr_ppl, nv_ppl))

    nlm_tr = nlm.perplexity(X_tr, y_tr)
    nlm_nv = nlm.perplexity(X_test, y_test)
    label = f"NLM (D={EMB_DIM}, H={HIDDEN})"
    print(f"{label:<24}{nlm_tr:>14.4f}{nlm_nv:>20.4f}")
    rows.append((label, nlm_tr, nlm_nv))

    write_table(rows, f"{RESULTS_DIR}/exp7_perplexity.tsv",
                header=["model", "train_ppl", "novel_ppl"])
    print(f"\n저장: {RESULTS_DIR}/exp7_perplexity.tsv")

    # 막대 그래프
    labels = [r[0] for r in rows]
    train_ppls = [r[1] for r in rows]
    novel_ppls = [r[2] for r in rows]
    x = np.arange(len(labels))
    w = 0.4

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, train_ppls, w, label="Train PPL", color="#3b82f6")
    ax.bar(x + w/2, novel_ppls, w, label="Novel PPL", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("perplexity (log scale, 낮을수록 좋음)")
    ax.set_yscale("log")
    ax.set_title("실험 7: n-gram vs NLM perplexity")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, which="both")

    # 값 표시
    for xi, v in zip(x - w/2, train_ppls):
        ax.text(xi, v * 1.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w/2, novel_ppls):
        ax.text(xi, v * 1.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp7_perplexity.png", dpi=120)
    print(f"저장: {RESULTS_DIR}/exp7_perplexity.png")

    # ----- 훈련 문장 6개 각각의 perplexity -----
    print("\n=== 훈련 문장별 perplexity ===")
    header_models = [f"n-gram (k={k})" for k in ngrams.keys()] + [f"NLM (D={EMB_DIM},H={HIDDEN})"]
    print(f"{'sentence':<26}" + "".join(f"{m:>20}" for m in header_models))
    per_sent_rows = []
    for s in SENTENCES:
        Xs, ys = make_test_pairs([s], stoi)
        ppls = []
        for k, m in ngrams.items():
            ppls.append(m.perplexity(Xs, ys))
        ppls.append(nlm.perplexity(Xs, ys))
        print(f"{s:<26}" + "".join(f"{p:>20.4f}" for p in ppls))
        per_sent_rows.append((s, *ppls))

    write_table(per_sent_rows, f"{RESULTS_DIR}/exp7_perplexity_per_sentence.tsv",
                header=["sentence"] + header_models)
    print(f"\n저장: {RESULTS_DIR}/exp7_perplexity_per_sentence.tsv")

    # 토큰 단위 NLL 분해 (novel 문장)
    print("\n=== Novel 문장 token별 -log P (NLM) ===")
    for s in NOVEL_SENTENCES:
        Xs, ys = make_test_pairs([s], stoi)
        probs = nlm.predict_proba(Xs)
        ll = -np.log(np.maximum(probs[np.arange(len(ys)), ys], 1e-30))
        print(f"  '{s}'  (PPL = {np.exp(ll.mean()):.3f})")
        for ctx, w_id, l in zip(Xs, ys, ll):
            ctx_words = " ".join(itos[t] for t in ctx)
            print(f"    P({itos[w_id]} | {ctx_words}) = {np.exp(-l):.4f}  (-log = {l:.3f})")


if __name__ == "__main__":
    run()
