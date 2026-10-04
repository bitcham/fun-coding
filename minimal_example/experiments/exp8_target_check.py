"""실험 8: (철수는, 바나나를) → 좋아해 생성 가능성 확인.

이 컨텍스트는 학습 데이터에 등장하지 않는다. n-gram은 일반화 못 하지만
NLM은 임베딩으로 유사 컨텍스트(영희는·짱구는)에서 일반화한다.
10,000회 샘플링한 카운트와 single-position perplexity를 비교한다.
"""

from collections import Counter

import numpy as np

from exp_common import RESULTS_DIR, write_table, fmt_pct
from utils import setup_io, set_seed, setup_matplotlib_korean
from data import build_vocab, make_pairs, encode_pairs
from ngram import NGramModel
from nlm import NLM, train

EMB_DIM = 2
HIDDEN = 8
LR = 0.3
EPOCHS = 2000
N_SAMPLES = 10000


def run():
    setup_io()
    plt = setup_matplotlib_korean()

    stoi, itos = build_vocab()
    pairs = make_pairs()
    X, y = encode_pairs(pairs, stoi)

    # n-gram 두 종 (MLE와 가벼운 smoothing)
    ng_mle = NGramModel(vocab_size=len(itos), smoothing_k=0.0)
    ng_mle.fit(X, y)
    ng_sm = NGramModel(vocab_size=len(itos), smoothing_k=0.01)
    ng_sm.fit(X, y)

    # NLM
    set_seed(123)
    nlm = NLM(vocab_size=len(itos), emb_dim=EMB_DIM, hidden=HIDDEN, seed=123)
    train(nlm, X, y, lr=LR, epochs=EPOCHS)

    target_ctx = (stoi["철수는"], stoi["바나나를"])
    target_w = stoi["좋아해"]
    target_w_str = "좋아해"

    print(f"== 실험 8: 컨텍스트 (철수는, 바나나를) 다음 토큰 ==\n")
    print(f"이 컨텍스트는 학습 쌍에 등장하는가? {ng_mle.has_context(target_ctx)}")

    # 분포 비교
    p_ng_mle = ng_mle.prob(target_ctx)
    p_ng_sm = ng_sm.prob(target_ctx)
    p_nlm = nlm.predict_proba(np.array([list(target_ctx)], dtype=np.int64))[0]

    print(f"\nP({target_w_str} | 철수는, 바나나를):")
    print(f"  n-gram MLE  : {p_ng_mle[target_w]:.6f}  (uniform fallback이라 1/V)")
    print(f"  n-gram k=.01: {p_ng_sm[target_w]:.6f}  (uniform fallback)")
    print(f"  NLM         : {p_nlm[target_w]:.6f}")

    # 10,000회 샘플링
    rng_mle = np.random.default_rng(7)
    rng_sm = np.random.default_rng(7)
    rng_nlm = np.random.default_rng(7)
    samples_mle = [int(rng_mle.choice(len(itos), p=p_ng_mle / p_ng_mle.sum())) for _ in range(N_SAMPLES)]
    samples_sm = [int(rng_sm.choice(len(itos), p=p_ng_sm / p_ng_sm.sum())) for _ in range(N_SAMPLES)]
    samples_nlm = [int(rng_nlm.choice(len(itos), p=p_nlm / p_nlm.sum())) for _ in range(N_SAMPLES)]

    cnt_mle = Counter(samples_mle)
    cnt_sm = Counter(samples_sm)
    cnt_nlm = Counter(samples_nlm)

    print(f"\n10,000회 샘플링 — '{target_w_str}' 카운트:")
    print(f"  n-gram MLE  : {cnt_mle.get(target_w, 0):>5}  ({fmt_pct(cnt_mle.get(target_w, 0)/N_SAMPLES)})")
    print(f"  n-gram k=.01: {cnt_sm.get(target_w, 0):>5}  ({fmt_pct(cnt_sm.get(target_w, 0)/N_SAMPLES)})")
    print(f"  NLM         : {cnt_nlm.get(target_w, 0):>5}  ({fmt_pct(cnt_nlm.get(target_w, 0)/N_SAMPLES)})")

    # Single-position perplexity = 1 / P(target | ctx)
    # (perplexity는 exp(NLL); 단일 위치의 경우 1/P)
    print(f"\nSingle-position perplexity (1 / P({target_w_str} | 철수는, 바나나를)):")
    print(f"  n-gram MLE  : {1.0 / max(p_ng_mle[target_w], 1e-30):>10.4f}")
    print(f"  n-gram k=.01: {1.0 / max(p_ng_sm[target_w], 1e-30):>10.4f}")
    print(f"  NLM         : {1.0 / max(p_nlm[target_w], 1e-30):>10.4f}")

    # NLM의 top-5
    print(f"\nNLM top-5 next-token 분포:")
    top = np.argsort(p_nlm)[::-1][:5]
    for t in top:
        print(f"  {itos[t]}: {p_nlm[t]:.6f}")

    # 결과 저장
    rows = [
        ("n-gram (MLE)", p_ng_mle[target_w], cnt_mle.get(target_w, 0), 1.0 / max(p_ng_mle[target_w], 1e-30)),
        ("n-gram (k=0.01)", p_ng_sm[target_w], cnt_sm.get(target_w, 0), 1.0 / max(p_ng_sm[target_w], 1e-30)),
        (f"NLM (D={EMB_DIM}, H={HIDDEN})", p_nlm[target_w], cnt_nlm.get(target_w, 0), 1.0 / max(p_nlm[target_w], 1e-30)),
    ]
    write_table(rows, f"{RESULTS_DIR}/exp8_target.tsv",
                header=["model", "P(좋아해|철수는,바나나를)", "count_in_10k", "single_ppl"])
    print(f"\n저장: {RESULTS_DIR}/exp8_target.tsv")

    # 시각화: 세 모델의 다음 토큰 분포 비교
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    titles = ["n-gram MLE", "n-gram k=0.01", f"NLM (D={EMB_DIM}, H={HIDDEN})"]
    probs_list = [p_ng_mle, p_ng_sm, p_nlm]
    for ax, ttl, pp in zip(axes, titles, probs_list):
        order = np.argsort(pp)[::-1]
        labels = [itos[i] for i in order]
        bars = ax.bar(range(len(itos)), [pp[i] for i in order], color="#9ca3af")
        # 좋아해 강조
        for i, w_id in enumerate(order):
            if w_id == target_w:
                bars[i].set_color("#10b981")
        ax.set_xticks(range(len(itos)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{ttl}\nP(좋아해)={pp[target_w]:.4f}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("probability")
    fig.suptitle("실험 8: P(· | 철수는, 바나나를)  ←  학습에 없던 컨텍스트")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp8_distribution.png", dpi=120)
    print(f"저장: {RESULTS_DIR}/exp8_distribution.png")


if __name__ == "__main__":
    run()
