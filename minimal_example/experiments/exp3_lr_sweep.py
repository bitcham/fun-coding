"""실험 3: learning rate 변화에 따른 훈련 속도 비교."""

import numpy as np

from exp_common import RESULTS_DIR
from utils import setup_io, set_seed, setup_matplotlib_korean
from data import build_vocab, make_pairs, encode_pairs
from nlm import NLM, train

LRS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
EMB_DIM = 2
HIDDEN = 8
EPOCHS = 2000


def run():
    setup_io()
    plt = setup_matplotlib_korean()

    stoi, itos = build_vocab()
    pairs = make_pairs()
    X, y = encode_pairs(pairs, stoi)

    histories = {}
    for lr in LRS:
        set_seed(123)
        model = NLM(vocab_size=len(itos), emb_dim=EMB_DIM, hidden=HIDDEN, seed=123)
        h, _ = train(model, X, y, lr=lr, epochs=EPOCHS)
        histories[lr] = h
        print(f"lr={lr:>5}: 시작 loss={h[0]:.4f}, 최종 loss={h[-1]:.4f}, 최저 loss={min(h):.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for lr in LRS:
        ax.plot(histories[lr], label=f"lr={lr}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_yscale("log")
    ax.set_title(f"실험 3: learning rate 비교 (D={EMB_DIM}, H={HIDDEN})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_fig = f"{RESULTS_DIR}/exp3_lr_sweep.png"
    fig.savefig(out_fig, dpi=120)
    print(f"\n저장: {out_fig}")

    # 첫 200 epoch만 보는 확대 그래프
    fig, ax = plt.subplots(figsize=(9, 5))
    for lr in LRS:
        ax.plot(histories[lr][:200], label=f"lr={lr}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("실험 3: 초기 200 epoch 비교")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_fig2 = f"{RESULTS_DIR}/exp3_lr_sweep_zoom.png"
    fig.savefig(out_fig2, dpi=120)
    print(f"저장: {out_fig2}")

    # 표 저장
    with open(f"{RESULTS_DIR}/exp3_lr_summary.tsv", "w", encoding="utf-8") as f:
        f.write("lr\tstart_loss\tfinal_loss\tmin_loss\n")
        for lr in LRS:
            h = histories[lr]
            f.write(f"{lr}\t{h[0]:.6f}\t{h[-1]:.6f}\t{min(h):.6f}\n")
    print(f"저장: {RESULTS_DIR}/exp3_lr_summary.tsv")


if __name__ == "__main__":
    run()
