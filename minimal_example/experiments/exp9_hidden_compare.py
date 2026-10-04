"""실험 9: hidden 뉴런 수 H=6 vs H=8 직접 비교.

같은 데이터·시드·learning rate에서 H만 바꿔 학습한 뒤, 학습 곡선·
perplexity·10,000 문장 분포·(철수는, 바나나를) → 좋아해 검증을 나란히 본다.
"""

from collections import Counter

import numpy as np

from exp_common import RESULTS_DIR, count_sentences, write_table, fmt_pct
from utils import setup_io, set_seed, setup_matplotlib_korean
from data import build_vocab, make_pairs, encode_pairs, SENTENCES, CONTEXT_LEN, PAD, BOS, EOS
from ngram import NGramModel
from nlm import NLM, train

EMB_DIM = 2
LR = 0.3
EPOCHS = 2000
N_SAMPLES = 10000
HIDDENS = [6, 8]

NOVEL_SENTENCES = [
    "철수는 바나나를 좋아해",
    "짱구는 사과를 싫어해",
    "영희는 사과를 싫어해",
]


def make_test_pairs(sentences, stoi):
    pairs = []
    for s in sentences:
        toks = [PAD] * (CONTEXT_LEN - 1) + [BOS] + s.split() + [EOS]
        for i in range(len(toks) - CONTEXT_LEN):
            pairs.append(([stoi[t] for t in toks[i : i + CONTEXT_LEN]],
                          stoi[toks[i + CONTEXT_LEN]]))
    X = np.array([p[0] for p in pairs], dtype=np.int64)
    y = np.array([p[1] for p in pairs], dtype=np.int64)
    return X, y


def run():
    setup_io()
    plt = setup_matplotlib_korean()

    stoi, itos = build_vocab()
    pairs = make_pairs()
    X_tr, y_tr = encode_pairs(pairs, stoi)
    X_nv, y_nv = make_test_pairs(NOVEL_SENTENCES, stoi)
    target_ctx = (stoi["철수는"], stoi["바나나를"])
    target_w = stoi["좋아해"]
    train_set = set(SENTENCES)

    # 모델 두 개 학습
    results = {}
    for H in HIDDENS:
        set_seed(123)
        model = NLM(vocab_size=len(itos), emb_dim=EMB_DIM, hidden=H, seed=123)
        history, _ = train(model, X_tr, y_tr, lr=LR, epochs=EPOCHS)
        results[H] = {
            "model": model,
            "history": history,
            "train_ppl": model.perplexity(X_tr, y_tr),
            "novel_ppl": model.perplexity(X_nv, y_nv),
        }
        # target check
        p = model.predict_proba(np.array([list(target_ctx)], dtype=np.int64))[0]
        results[H]["p_target"] = float(p[target_w])
        results[H]["p_dist"] = p
        # 10k 문장 생성
        rng = np.random.default_rng(42)
        samples = [model.generate(stoi, itos, rng) for _ in range(N_SAMPLES)]
        cnt = count_sentences(samples)
        results[H]["counter"] = cnt
        results[H]["in_train"] = sum(c for s, c in cnt.items() if s in train_set)
        results[H]["novel"]    = sum(c for s, c in cnt.items() if s not in train_set)

    # ---------- 콘솔 출력 ----------
    print("== 실험 9: H=6 vs H=8 (D=2, lr=0.3, epochs=2000) ==\n")
    print(f"{'metric':<32}" + "".join(f"{f'H={H}':>14}" for H in HIDDENS))
    print("-" * (32 + 14 * len(HIDDENS)))
    for label, key, fmt in [
        ("최종 train loss",          lambda r: r["history"][-1],      "{:.4f}"),
        ("train perplexity",         lambda r: r["train_ppl"],         "{:.4f}"),
        ("novel perplexity",         lambda r: r["novel_ppl"],         "{:.4f}"),
        ("P(좋아해 | 철수는, 바나나를)", lambda r: r["p_target"],          "{:.4f}"),
        ("10,000 중 train 문장 비율", lambda r: r["in_train"]/N_SAMPLES, "{:.2%}"),
        ("10,000 중 novel 문장 비율", lambda r: r["novel"]/N_SAMPLES,    "{:.2%}"),
        ("고유 문장 수",              lambda r: len(r["counter"]),       "{:d}"),
    ]:
        row = f"{label:<32}"
        for H in HIDDENS:
            row += f"{fmt.format(key(results[H])):>14}"
        print(row)

    # ---------- 표 저장 ----------
    rows = []
    for H in HIDDENS:
        r = results[H]
        rows.append((H, r["history"][-1], r["train_ppl"], r["novel_ppl"],
                     r["p_target"], r["in_train"], r["novel"], len(r["counter"])))
    write_table(rows, f"{RESULTS_DIR}/exp9_hidden_compare.tsv",
                header=["H", "final_loss", "train_ppl", "novel_ppl",
                        "P(좋아해|철수는,바나나를)", "in_train_10k", "novel_10k", "unique_sentences"])
    print(f"\n저장: {RESULTS_DIR}/exp9_hidden_compare.tsv")

    # ---------- 그래프 1: 학습 곡선 비교 ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {6: "#3b82f6", 8: "#ef4444"}
    for H in HIDDENS:
        ax.plot(results[H]["history"], color=colors[H], label=f"H={H}, ppl={results[H]['train_ppl']:.4f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"실험 9: H=6 vs H=8 학습 곡선 (D={EMB_DIM}, lr={LR})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp9_loss_curve.png", dpi=120)
    print(f"저장: {RESULTS_DIR}/exp9_loss_curve.png")

    # ---------- 그래프 2: target 분포 비교 ----------
    fig, axes = plt.subplots(1, len(HIDDENS), figsize=(5 * len(HIDDENS), 4.5), sharey=True)
    if len(HIDDENS) == 1:
        axes = [axes]
    for ax, H in zip(axes, HIDDENS):
        p = results[H]["p_dist"]
        order = np.argsort(p)[::-1]
        labels = [itos[i] for i in order]
        bars = ax.bar(range(len(itos)), [p[i] for i in order], color="#9ca3af")
        for i, w_id in enumerate(order):
            if w_id == target_w:
                bars[i].set_color("#10b981")
        ax.set_xticks(range(len(itos)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"H={H}\nP(좋아해)={p[target_w]:.4f}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("probability")
    fig.suptitle("실험 9: P(· | 철수는, 바나나를) — H 비교")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp9_target_distribution.png", dpi=120)
    print(f"저장: {RESULTS_DIR}/exp9_target_distribution.png")

    # ---------- 그래프 3: 임베딩 공간 비교 ----------
    fig, axes = plt.subplots(1, len(HIDDENS), figsize=(6 * len(HIDDENS), 5), sharex=False, sharey=False)
    if len(HIDDENS) == 1:
        axes = [axes]
    for ax, H in zip(axes, HIDDENS):
        E = results[H]["model"].E
        ax.scatter(E[:, 0], E[:, 1], s=80, c="#3b82f6")
        for i, w in enumerate(itos):
            ax.annotate(w, (E[i, 0], E[i, 1]), xytext=(5, 5),
                        textcoords="offset points", fontsize=9)
        ax.set_xlabel("embedding axis 0")
        ax.set_ylabel("embedding axis 1")
        ax.set_title(f"H={H}, train loss={results[H]['history'][-1]:.4f}")
        ax.grid(True, alpha=0.2)
    fig.suptitle("실험 9: 학습된 임베딩 공간 (D=2)")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp9_embedding.png", dpi=120)
    print(f"저장: {RESULTS_DIR}/exp9_embedding.png")


if __name__ == "__main__":
    run()
