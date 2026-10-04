"""실험 2: NLM으로 10,000문장 생성 + 통계 분석.

n-gram과 달리 NLM은 임베딩 일반화로 학습 데이터에 없던 문장도 생성한다.
"""

import numpy as np

from exp_common import RESULTS_DIR, count_sentences, write_table, fmt_pct
from utils import setup_io, set_seed, setup_matplotlib_korean
from data import build_vocab, make_pairs, encode_pairs, SENTENCES
from nlm import NLM, train

N_SAMPLES = 10000
EMB_DIM = 2
HIDDEN = 8
LR = 0.3
EPOCHS = 2000


def run():
    setup_io()
    set_seed(123)
    plt = setup_matplotlib_korean()

    stoi, itos = build_vocab()
    pairs = make_pairs()
    X, y = encode_pairs(pairs, stoi)

    model = NLM(vocab_size=len(itos), emb_dim=EMB_DIM, hidden=HIDDEN, seed=123)
    history, _ = train(model, X, y, lr=LR, epochs=EPOCHS, log_path=f"{RESULTS_DIR}/exp2_loss_history.txt")
    print(f"훈련 후 loss: {history[-1]:.4f}, perplexity: {model.perplexity(X, y):.4f}")

    rng = np.random.default_rng(42)
    samples = [model.generate(stoi, itos, rng) for _ in range(N_SAMPLES)]
    counter = count_sentences(samples)

    train_set = set(SENTENCES)
    in_train = sum(c for s, c in counter.items() if s in train_set)
    novel = sum(c for s, c in counter.items() if s not in train_set)

    print(f"\n== 실험 2: NLM에서 {N_SAMPLES}문장 생성 (D={EMB_DIM}, H={HIDDEN}, lr={LR}, ep={EPOCHS}) ==")
    print(f"고유 문장 수: {len(counter)}")
    print(f"훈련셋 안: {in_train} ({fmt_pct(in_train/N_SAMPLES)})")
    print(f"훈련셋 밖(novel): {novel} ({fmt_pct(novel/N_SAMPLES)})")
    print()
    print("상위 15개 (내림차순):")
    rows = []
    for s, c in counter.most_common():
        flag = "(train)" if s in train_set else "(novel)"
        rows.append((c, fmt_pct(c / N_SAMPLES), "train" if s in train_set else "novel", s))
    for c, pct, t, s in rows[:15]:
        print(f"  {c:5d} ({pct})  {'(train)' if t=='train' else '(novel)'}  {s}")
    if len(rows) > 15:
        print(f"  ... (그 외 {len(rows)-15}개)")

    out_table = f"{RESULTS_DIR}/exp2_nlm_samples.tsv"
    write_table(rows, out_table, header=["count", "pct", "type", "sentence"])
    print(f"\n저장: {out_table}")

    # 막대 그래프 (상위 12개만)
    show = rows[:12]
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#3b82f6" if t == "train" else "#ef4444" for _, _, t, _ in show]
    counts = [r[0] for r in show]
    labels = [r[3] for r in show]
    ax.bar(range(len(show)), counts, color=colors)
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("빈도 (10,000회 중)")
    ax.set_title(f"실험 2: NLM 생성 분포  (train={in_train}, novel={novel}, unique={len(counter)})")
    fig.tight_layout()
    out_fig = f"{RESULTS_DIR}/exp2_nlm_samples.png"
    fig.savefig(out_fig, dpi=120)
    print(f"저장: {out_fig}")

    # loss 곡선
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"실험 2: NLM 학습 곡선 (D={EMB_DIM}, H={HIDDEN}, lr={LR})")
    fig.tight_layout()
    out_fig2 = f"{RESULTS_DIR}/exp2_loss_curve.png"
    fig.savefig(out_fig2, dpi=120)
    print(f"저장: {out_fig2}")


if __name__ == "__main__":
    run()
