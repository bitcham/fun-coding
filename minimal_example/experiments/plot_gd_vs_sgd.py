"""GD vs SGD 비교 그래프 작성.

- baseline (24 pairs)와 ×100 emphasis (420 pairs)에서 GD/SGD 학습 곡선 비교
- lr scaling rule 적용: GD lr=0.3 ↔ SGD lr=0.3/N_pairs
- wall-clock 시간 측정 (legend에 표시)
- 2-panel: 좌=baseline, 우=×100
"""

import time
import numpy as np

from exp_common import RESULTS_DIR
from utils import setup_io, set_seed, setup_matplotlib_korean
from data import build_vocab, make_pairs, encode_pairs, SENTENCES
from nlm import NLM, train

EMB_DIM = 2
HIDDEN = 8
EPOCHS = 2000
SEED = 123
BATCH_SEED = 777

EMPHASIS_TARGET = "짱구는 바나나를 싫어해"


def make_data(stoi, repeat):
    if repeat == 0:
        sents = list(SENTENCES)
    else:
        base_5 = [s for s in SENTENCES if s != EMPHASIS_TARGET]
        sents = base_5 + [EMPHASIS_TARGET] * repeat
    pairs = make_pairs(sentences=sents)
    return encode_pairs(pairs, stoi)


def train_and_time(X, y, V, batch_size, lr, label):
    set_seed(SEED)
    nlm = NLM(vocab_size=V, emb_dim=EMB_DIM, hidden=HIDDEN, seed=SEED)
    t0 = time.time()
    history, _ = train(nlm, X, y, lr=lr, epochs=EPOCHS,
                       batch_size=batch_size, shuffle=True, batch_seed=BATCH_SEED)
    elapsed = time.time() - t0
    print(f"[{label}] N={len(X)}, lr={lr}, batch={batch_size}, "
          f"final={history[-1]:.4f}, time={elapsed:.2f}s")
    return history, elapsed


def run():
    setup_io()
    plt = setup_matplotlib_korean()
    stoi, itos = build_vocab()
    V = len(itos)

    print("학습 시작...")
    X_b, y_b = make_data(stoi, 0)
    gd_b_h,  gd_b_t  = train_and_time(X_b, y_b, V, None, 0.3,    "GD baseline")
    sgd_b_h, sgd_b_t = train_and_time(X_b, y_b, V, 1,    0.0125, "SGD baseline")

    X_e, y_e = make_data(stoi, 100)
    gd_e_h,  gd_e_t  = train_and_time(X_e, y_e, V, None, 0.3,    "GD x100")
    sgd_e_h, sgd_e_t = train_and_time(X_e, y_e, V, 1,    0.0007, "SGD x100")

    # ============================
    # 시각화
    # ============================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(gd_b_h,  label=f"GD     lr=0.3,    {gd_b_t:5.1f}s",  color="#1f77b4", lw=1.5)
    ax.plot(sgd_b_h, label=f"SGD  lr=0.0125, {sgd_b_t:5.1f}s",  color="#ff7f0e", lw=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (log scale)")
    ax.set_title(f"Baseline: 24 pairs")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(gd_e_h,  label=f"GD     lr=0.3,    {gd_e_t:5.1f}s",  color="#1f77b4", lw=1.5)
    ax.plot(sgd_e_h, label=f"SGD  lr=0.0007, {sgd_e_t:5.1f}s",   color="#ff7f0e", lw=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (log scale)")
    ax.set_title(f"×100 emphasis: 420 pairs")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"GD vs SGD 학습 곡선 (EPOCHS={EPOCHS}, lr scaling 적용)", fontsize=12)
    fig.tight_layout()
    out = f"{RESULTS_DIR}/gd_vs_sgd_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n저장: {out}")

    # ============================
    # Wall-clock 시간 비교 막대그래프
    # ============================
    fig, ax = plt.subplots(figsize=(8, 4.5))
    setups = ["baseline\n(24 pairs)", "×100 emphasis\n(420 pairs)"]
    gd_times  = [gd_b_t,  gd_e_t]
    sgd_times = [sgd_b_t, sgd_e_t]
    x = np.arange(len(setups))
    width = 0.35
    bars1 = ax.bar(x - width/2, gd_times,  width, label="GD (full-batch)", color="#1f77b4")
    bars2 = ax.bar(x + width/2, sgd_times, width, label="SGD (batch=1)",   color="#ff7f0e")
    for b in bars1:
        ax.annotate(f"{b.get_height():.1f}s", xy=(b.get_x()+b.get_width()/2, b.get_height()),
                    xytext=(0,3), textcoords="offset points", ha="center", fontsize=10)
    for b in bars2:
        ax.annotate(f"{b.get_height():.1f}s", xy=(b.get_x()+b.get_width()/2, b.get_height()),
                    xytext=(0,3), textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(setups)
    ax.set_ylabel("wall-clock time (seconds)")
    ax.set_title(f"학습 wall-clock 시간 비교 (EPOCHS={EPOCHS})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out2 = f"{RESULTS_DIR}/gd_vs_sgd_wallclock.png"
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"저장: {out2}")

    # 요약
    print(f"\n=== 요약 ===")
    print(f"Baseline:  GD {gd_b_t:.2f}s | SGD {sgd_b_t:.2f}s | ratio = {sgd_b_t/gd_b_t:.1f}×")
    print(f"×100:       GD {gd_e_t:.2f}s | SGD {sgd_e_t:.2f}s | ratio = {sgd_e_t/gd_e_t:.1f}×")
    print(f"\nFinal loss (수렴 도달 여부):")
    print(f"  GD  baseline: {gd_b_h[-1]:.4f}    SGD baseline: {sgd_b_h[-1]:.4f}")
    print(f"  GD  ×100:    {gd_e_h[-1]:.4f}    SGD ×100:    {sgd_e_h[-1]:.4f}")


if __name__ == "__main__":
    run()
