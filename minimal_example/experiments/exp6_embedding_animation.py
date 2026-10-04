"""실험 6: 학습 중 word embedding space 변화 애니메이션 (mp4)."""

import numpy as np
import matplotlib.animation as animation

from exp_common import RESULTS_DIR
from utils import setup_io, set_seed, setup_matplotlib_korean
from data import build_vocab, make_pairs, encode_pairs
from nlm import NLM, train

EMB_DIM = 2
HIDDEN = 8
LR = 0.3
EPOCHS = 1500
SNAPSHOT_EVERY = 10  # 매 10 epoch마다 (총 ~150 frames)
FPS = 15


def run():
    setup_io()
    set_seed(123)
    plt = setup_matplotlib_korean()

    stoi, itos = build_vocab()
    pairs = make_pairs()
    X, y = encode_pairs(pairs, stoi)

    model = NLM(vocab_size=len(itos), emb_dim=EMB_DIM, hidden=HIDDEN, seed=123)
    history, snapshots = train(model, X, y, lr=LR, epochs=EPOCHS, snapshot_every=SNAPSHOT_EVERY)
    print(f"수집된 snapshot 수: {len(snapshots)}, 최종 loss: {history[-1]:.4f}")

    # 좌표 범위 계산 (전체 frame 통합)
    all_E = np.stack([s[1]["E"] for s in snapshots])  # [F, V, 2]
    pad = 0.5
    xmin, xmax = all_E[..., 0].min() - pad, all_E[..., 0].max() + pad
    ymin, ymax = all_E[..., 1].min() - pad, all_E[..., 1].max() + pad

    # 색상: 단어 카테고리별
    person = {"철수는", "영희는", "짱구는"}
    fruit = {"사과를", "딸기를", "바나나를"}
    verb = {"좋아해", "싫어해"}

    def color_of(w):
        if w in person:
            return "#3b82f6"  # blue
        if w in fruit:
            return "#10b981"  # green
        if w in verb:
            return "#ef4444"  # red
        return "#9ca3af"      # gray (specials)

    colors = [color_of(w) for w in itos]

    fig, (ax_emb, ax_loss) = plt.subplots(1, 2, figsize=(12, 5.5),
                                          gridspec_kw={"width_ratios": [3, 2]})

    # 임베딩 산점도
    ax_emb.set_xlim(xmin, xmax)
    ax_emb.set_ylim(ymin, ymax)
    ax_emb.set_xlabel("embedding axis 0")
    ax_emb.set_ylabel("embedding axis 1")
    ax_emb.grid(True, alpha=0.2)
    init_E = snapshots[0][1]["E"]
    scat = ax_emb.scatter(init_E[:, 0], init_E[:, 1], s=80, c=colors)
    texts = [ax_emb.text(init_E[i, 0], init_E[i, 1], itos[i], fontsize=9) for i in range(len(itos))]

    # 트레일 (각 단어의 궤적)
    trails = [ax_emb.plot([], [], "-", color=colors[i], alpha=0.25, lw=0.7)[0] for i in range(len(itos))]

    # loss 곡선
    ax_loss.plot(history, color="#374151", lw=1)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("학습 곡선")
    ax_loss.grid(True, alpha=0.3)
    loss_marker, = ax_loss.plot([], [], "o", color="#ef4444", ms=6)

    title = ax_emb.set_title("")

    # 범례
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3b82f6", markersize=8, label="인물"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#10b981", markersize=8, label="과일"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ef4444", markersize=8, label="동사"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9ca3af", markersize=8, label="특수"),
    ]
    ax_emb.legend(handles=legend, loc="best", fontsize=8)

    def update(frame):
        ep, state = snapshots[frame]
        E = state["E"]
        scat.set_offsets(E)
        for i, t in enumerate(texts):
            t.set_position((E[i, 0] + 0.04, E[i, 1] + 0.04))
            t.set_text(itos[i])
        # trails: snapshot 0..frame
        for i in range(len(itos)):
            xs = all_E[: frame + 1, i, 0]
            ys = all_E[: frame + 1, i, 1]
            trails[i].set_data(xs, ys)
        loss_marker.set_data([ep], [history[ep]])
        title.set_text(f"epoch {ep}  /  loss = {history[ep]:.4f}")
        return [scat, *texts, *trails, loss_marker, title]

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=1000 // FPS, blit=False)

    out_mp4 = f"{RESULTS_DIR}/exp6_embedding_animation.mp4"
    writer = animation.FFMpegWriter(fps=FPS, bitrate=2400)
    ani.save(out_mp4, writer=writer, dpi=120)
    print(f"저장: {out_mp4}")

    # 정적 그림 (마지막 frame)
    fig.savefig(f"{RESULTS_DIR}/exp6_embedding_final.png", dpi=120)
    print(f"저장: {RESULTS_DIR}/exp6_embedding_final.png")


if __name__ == "__main__":
    run()
