"""실험 5: emb_dim × hidden × lr 그리드 비교."""

import numpy as np

from exp_common import RESULTS_DIR, write_table
from utils import setup_io, set_seed, setup_matplotlib_korean
from data import build_vocab, make_pairs, encode_pairs
from nlm import NLM, train

EMB_DIMS = [1, 2, 3]
HIDDENS = [4, 6, 8, 16]
LRS = [0.1, 0.3, 1.0]
EPOCHS = 2000


def run():
    setup_io()
    plt = setup_matplotlib_korean()

    stoi, itos = build_vocab()
    pairs = make_pairs()
    X, y = encode_pairs(pairs, stoi)

    rows = []
    print(f"{'D':>3} {'H':>3} {'lr':>5} | {'final_loss':>10} {'min_loss':>10} {'ppl':>7}")
    print("-" * 52)
    for D in EMB_DIMS:
        for H in HIDDENS:
            for lr in LRS:
                set_seed(123)
                model = NLM(vocab_size=len(itos), emb_dim=D, hidden=H, seed=123)
                h, _ = train(model, X, y, lr=lr, epochs=EPOCHS)
                ppl = model.perplexity(X, y)
                rows.append((D, H, lr, h[0], h[-1], min(h), ppl))
                print(f"{D:>3} {H:>3} {lr:>5} | {h[-1]:>10.4f} {min(h):>10.4f} {ppl:>7.4f}")

    write_table(
        rows,
        f"{RESULTS_DIR}/exp5_grid.tsv",
        header=["emb_dim", "hidden", "lr", "start_loss", "final_loss", "min_loss", "ppl"],
    )
    print(f"\n저장: {RESULTS_DIR}/exp5_grid.tsv")

    # === 히트맵: lr × hidden, D별 separate panel — final_loss ===
    fig, axes = plt.subplots(1, len(EMB_DIMS), figsize=(4.2 * len(EMB_DIMS), 4.0))
    if len(EMB_DIMS) == 1:
        axes = [axes]
    grid = np.zeros((len(EMB_DIMS), len(HIDDENS), len(LRS)))
    for r in rows:
        D, H, lr, _, fl, _, _ = r
        i, j, k = EMB_DIMS.index(D), HIDDENS.index(H), LRS.index(lr)
        grid[i, j, k] = fl

    vmin, vmax = grid.min(), grid.max()
    for i, D in enumerate(EMB_DIMS):
        ax = axes[i]
        im = ax.imshow(grid[i], cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(LRS)))
        ax.set_xticklabels(LRS)
        ax.set_yticks(range(len(HIDDENS)))
        ax.set_yticklabels(HIDDENS)
        ax.set_xlabel("learning rate")
        if i == 0:
            ax.set_ylabel("hidden size")
        ax.set_title(f"D = {D}")
        for j in range(len(HIDDENS)):
            for k in range(len(LRS)):
                ax.text(k, j, f"{grid[i,j,k]:.3f}",
                        ha="center", va="center", color="white", fontsize=8)
    fig.suptitle(f"실험 5: 최종 loss (epochs={EPOCHS})")
    fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02, label="final loss")
    fig.savefig(f"{RESULTS_DIR}/exp5_grid_heatmap.png", dpi=120)
    print(f"저장: {RESULTS_DIR}/exp5_grid_heatmap.png")


if __name__ == "__main__":
    run()
