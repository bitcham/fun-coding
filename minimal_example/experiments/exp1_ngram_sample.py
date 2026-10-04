"""실험 1: n-gram(trigram)으로 10,000문장을 생성하고 분포를 분석."""

import numpy as np

from exp_common import RESULTS_DIR, count_sentences, write_table, fmt_pct
from utils import setup_io, set_seed, setup_matplotlib_korean
from data import build_vocab, make_pairs, encode_pairs, SENTENCES
from ngram import NGramModel

N_SAMPLES = 10000


def run():
    setup_io()
    set_seed(123)
    plt = setup_matplotlib_korean()

    stoi, itos = build_vocab()
    pairs = make_pairs()
    X, y = encode_pairs(pairs, stoi)

    model = NGramModel(vocab_size=len(itos), smoothing_k=0.0)
    model.fit(X, y)

    rng = np.random.default_rng(42)
    samples = [model.generate(stoi, itos, rng) for _ in range(N_SAMPLES)]
    counter = count_sentences(samples)

    train_set = set(SENTENCES)
    in_train = sum(c for s, c in counter.items() if s in train_set)
    novel = sum(c for s, c in counter.items() if s not in train_set)

    print(f"== 실험 1: n-gram에서 {N_SAMPLES}문장 생성 ==")
    print(f"고유 문장 수: {len(counter)}")
    print(f"훈련셋 안 문장 비율: {fmt_pct(in_train / N_SAMPLES)}")
    print(f"훈련셋 밖(novel) 문장 비율: {fmt_pct(novel / N_SAMPLES)}")
    print()
    print("문장별 빈도 (내림차순):")
    rows = []
    for s, c in counter.most_common():
        flag = "(train)" if s in train_set else "(novel)"
        print(f"  {c:5d} ({fmt_pct(c/N_SAMPLES)})  {flag}  {s}")
        rows.append((c, fmt_pct(c / N_SAMPLES), "train" if s in train_set else "novel", s))

    out_table = f"{RESULTS_DIR}/exp1_ngram_samples.tsv"
    write_table(rows, out_table, header=["count", "pct", "type", "sentence"])
    print(f"\n저장: {out_table}")

    # 막대 그래프
    labels = [s for s, _ in counter.most_common()]
    counts = [c for _, c in counter.most_common()]
    colors = ["#3b82f6" if s in train_set else "#ef4444" for s in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), counts, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("빈도 (10,000회 중)")
    ax.set_title(f"실험 1: n-gram 생성 분포  (train={in_train}, novel={novel})")
    ax.axhline(N_SAMPLES / 6, color="gray", linestyle=":", label="이론값 1/6 = 1666.7")
    ax.legend()
    fig.tight_layout()
    out_fig = f"{RESULTS_DIR}/exp1_ngram_samples.png"
    fig.savefig(out_fig, dpi=120)
    print(f"저장: {out_fig}")


if __name__ == "__main__":
    run()
