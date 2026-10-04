"""실험 4: embedding dimension 1 vs 2 비교.

학습 곡선과 임베딩 공간을 함께 시각화한다.
"""

import numpy as np

from exp_common import RESULTS_DIR
from utils import setup_io, set_seed, setup_matplotlib_korean
from data import build_vocab, make_pairs, encode_pairs
from nlm import NLM, train

HIDDEN = 8
LR = 0.3
EPOCHS = 3000


def run():
    setup_io()
    plt = setup_matplotlib_korean()

    stoi, itos = build_vocab()
    pairs = make_pairs()
    X, y = encode_pairs(pairs, stoi)

    results = {}
    for D in [1, 2]:
        set_seed(123)
        model = NLM(vocab_size=len(itos), emb_dim=D, hidden=HIDDEN, seed=123)
        history, _ = train(model, X, y, lr=LR, epochs=EPOCHS)
        ppl = model.perplexity(X, y)
        results[D] = {"history": history, "ppl": ppl, "E": model.E.data.copy()}
        print(f"D={D}: 최종 loss={history[-1]:.4f}, perplexity={ppl:.4f}")

    # === 그래프 1: 학습 곡선 비교 ===
    fig, ax = plt.subplots(figsize=(8, 5))
    for D in [1, 2]:
        ax.plot(results[D]["history"], label=f"D={D}, ppl={results[D]['ppl']:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"실험 4: embedding dim 1 vs 2 (H={HIDDEN}, lr={LR})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp4_loss.png", dpi=120)
    print(f"저장: {RESULTS_DIR}/exp4_loss.png")

    # === 그래프 2: 임베딩 공간 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # D=1
    E1 = results[1]["E"]  # [V, 1]
    ax = axes[0]
    ax.scatter(E1[:, 0], np.zeros(len(itos)), s=80, c="#3b82f6")
    for i, w in enumerate(itos):
        ax.annotate(w, (E1[i, 0], 0), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("embedding axis 0")
    ax.set_title(f"D=1 임베딩 (loss={results[1]['history'][-1]:.3f})")
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.grid(True, alpha=0.2)

    # D=2
    E2 = results[2]["E"]
    ax = axes[1]
    ax.scatter(E2[:, 0], E2[:, 1], s=80, c="#3b82f6")
    for i, w in enumerate(itos):
        ax.annotate(w, (E2[i, 0], E2[i, 1]), xytext=(5, 5),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("embedding axis 0")
    ax.set_ylabel("embedding axis 1")
    ax.set_title(f"D=2 임베딩 (loss={results[2]['history'][-1]:.3f})")
    ax.grid(True, alpha=0.2)

    fig.suptitle("실험 4: 학습된 word embedding")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp4_embedding.png", dpi=120)
    print(f"저장: {RESULTS_DIR}/exp4_embedding.png")


if __name__ == "__main__":
    run()
