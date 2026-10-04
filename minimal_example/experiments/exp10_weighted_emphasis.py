"""실험 10: 한 문장만 100배 가중 학습했을 때 두 모델의 반응.

가설:
- n-gram은 카운트 모델이라, 가중치를 100배 주면 그 문장의 생성 확률이 단순히
  100/(100+5)=95.2%로 비례 증가. 다른 문장 분포는 비례 축소될 뿐, 새로운
  novel 문장은 여전히 0%.
- NLM은 임베딩 분포 가설이 작동하므로, "짱구는·바나나를·싫어해" 트리오가
  강하게 묶이면서 "철수는·영희는"의 의미 위치까지 밀어낼 수 있다. 이 때
  unseen context (철수는, 바나나를)에 대한 P(좋아해) vs P(싫어해)가 baseline
  (좋아해 0.999, 싫어해 거의 0) 대비 어떻게 변하는지가 핵심.

비교 대상:
- baseline: 6문장 동등 (24 학습 쌍, exp2와 같은 setup)
- emphasis: 5문장 1회 + "짱구는 바나나를 싫어해" 100회 (420 학습 쌍)
"""

import numpy as np

from exp_common import RESULTS_DIR, count_sentences, write_table, fmt_pct
from utils import setup_io, set_seed, setup_matplotlib_korean
from data import build_vocab, make_pairs, encode_pairs, SENTENCES
from ngram import NGramModel
from nlm import NLM, train

# 학습 하이퍼파라미터 (exp2와 동일하게 맞춤)
EMB_DIM = 2
HIDDEN = 8
LR = 0.3
EPOCHS = 2000
SEED = 123
N_SAMPLES = 10000

# 가중치 비율
EMPHASIS_REPEAT = 100
EMPHASIS_TARGET = "짱구는 바나나를 싫어해"

# 핵심 컨텍스트 (학습/평가 양쪽에서 의미 있는 것들)
KEY_CONTEXTS = [
    ("<bos>", "철수는"),     # 짱구만 100배일 때 학습 분포에서 줄어듦
    ("<bos>", "영희는"),
    ("<bos>", "짱구는"),
    ("철수는", "사과를"),    # 학습 정상 (좋아해)
    ("철수는", "바나나를"),  # ★ unseen context — 핵심
    ("영희는", "바나나를"),  # 좋아해 (학습됨)
    ("짱구는", "바나나를"),  # 싫어해 (가중치 100x)
    ("짱구는", "사과를"),    # ★ unseen context
    ("바나나를", "좋아해"),  # 정상 → <eos>
    ("바나나를", "싫어해"),  # 정상 → <eos>
]

# novel 평가 문장 (학습에 없음)
NOVEL_TEST_SENTENCES = [
    "철수는 바나나를 싫어해",   # ★ "철수는→바나나를→싫어해" 조합 가능?
    "영희는 바나나를 싫어해",
    "철수는 바나나를 좋아해",   # baseline NLM이 가장 잘 만들었던 novel
    "짱구는 사과를 싫어해",
    "짱구는 딸기를 싫어해",
]


def make_emphasis_sentences():
    """5문장 1회 + 짱구 100회."""
    base_5 = [s for s in SENTENCES if s != EMPHASIS_TARGET]
    return base_5 + [EMPHASIS_TARGET] * EMPHASIS_REPEAT


def train_pair(stoi, itos, sentences, label):
    """주어진 sentences로 NLM과 n-gram 둘 다 학습."""
    pairs = make_pairs(sentences=sentences)
    X, y = encode_pairs(pairs, stoi)

    set_seed(SEED)
    nlm = NLM(vocab_size=len(itos), emb_dim=EMB_DIM, hidden=HIDDEN, seed=SEED)
    history, _ = train(nlm, X, y, lr=LR, epochs=EPOCHS)

    ngram = NGramModel(vocab_size=len(itos), smoothing_k=0.0)
    ngram.fit(X, y)

    print(f"[{label}] pairs={len(pairs)}, NLM final loss={history[-1]:.4f}, "
          f"train PPL={nlm.perplexity(X, y):.4f}")
    return {"nlm": nlm, "ngram": ngram, "history": history,
            "pairs": pairs, "X": X, "y": y, "n_pairs": len(pairs)}


def sample_distribution(model, stoi, itos, seed):
    rng = np.random.default_rng(seed)
    samples = [model.generate(stoi, itos, rng) for _ in range(N_SAMPLES)]
    return count_sentences(samples)


def context_probs(model, stoi, ctx_pair):
    """ctx_pair: (a, b) 토큰 문자열 → predict_proba 결과 dict."""
    ctx = (stoi[ctx_pair[0]], stoi[ctx_pair[1]])
    if isinstance(model, NLM):
        p = model.predict_proba(np.array([list(ctx)], dtype=np.int64))[0]
    else:  # n-gram
        p = model.prob(ctx)
    return p


def novel_perplexity(model, stoi, sentence):
    """문장을 학습 쌍으로 풀어 NLM/n-gram 모두에서 PPL 계산."""
    pairs = make_pairs(sentences=[sentence])
    X, y = encode_pairs(pairs, stoi)
    if isinstance(model, NLM):
        return model.perplexity(X, y)
    else:
        return model.perplexity(X, y)


def run():
    setup_io()
    plt = setup_matplotlib_korean()

    stoi, itos = build_vocab()  # vocab은 baseline 6문장으로 고정 (둘 다 같은 토큰만 사용)

    print(f"== 실험 10: 한 문장 100배 가중 학습 ==")
    print(f"강조 문장: '{EMPHASIS_TARGET}' × {EMPHASIS_REPEAT}")
    print(f"하이퍼파라미터: D={EMB_DIM}, H={HIDDEN}, lr={LR}, ep={EPOCHS}, seed={SEED}")
    print()

    base_sents = list(SENTENCES)
    emph_sents = make_emphasis_sentences()
    print(f"baseline 문장 수: {len(base_sents)}, emphasis 문장 수: {len(emph_sents)}")
    print()

    base = train_pair(stoi, itos, base_sents, "baseline")
    emph = train_pair(stoi, itos, emph_sents, "emphasis")

    # =========================================================
    # 1) 10,000 샘플 분포
    # =========================================================
    print("\n--- 10,000 샘플 분포 ---")
    base_nlm_dist = sample_distribution(base["nlm"], stoi, itos, seed=42)
    emph_nlm_dist = sample_distribution(emph["nlm"], stoi, itos, seed=42)
    base_ng_dist = sample_distribution(base["ngram"], stoi, itos, seed=42)
    emph_ng_dist = sample_distribution(emph["ngram"], stoi, itos, seed=42)

    train_set = set(SENTENCES)

    def summarize(label, counter):
        in_train = sum(c for s, c in counter.items() if s in train_set)
        novel = sum(c for s, c in counter.items() if s not in train_set)
        emph_count = counter.get(EMPHASIS_TARGET, 0)
        print(f"  [{label:<18}] unique={len(counter):3d}  "
              f"train={in_train:5d} ({fmt_pct(in_train/N_SAMPLES):>7})  "
              f"novel={novel:5d} ({fmt_pct(novel/N_SAMPLES):>7})  "
              f"'{EMPHASIS_TARGET}'={emph_count:5d}")
        return in_train, novel, emph_count

    summary_rows = []
    for label, counter in [
        ("ngram baseline", base_ng_dist),
        ("ngram emphasis", emph_ng_dist),
        ("NLM   baseline", base_nlm_dist),
        ("NLM   emphasis", emph_nlm_dist),
    ]:
        in_t, nov, ec = summarize(label, counter)
        summary_rows.append((label, len(counter), in_t, nov, ec))

    write_table(summary_rows,
                f"{RESULTS_DIR}/exp10_summary.tsv",
                header=["model", "unique_sentences", "in_train", "novel", "emphasis_count"])

    # 모든 생성 분포 저장 (모델별 따로)
    for tag, counter in [
        ("ngram_baseline", base_ng_dist),
        ("ngram_emphasis", emph_ng_dist),
        ("nlm_baseline",   base_nlm_dist),
        ("nlm_emphasis",   emph_nlm_dist),
    ]:
        rows = []
        for s, c in counter.most_common():
            rows.append((c, fmt_pct(c / N_SAMPLES),
                         "train" if s in train_set else "novel", s))
        write_table(rows, f"{RESULTS_DIR}/exp10_dist_{tag}.tsv",
                    header=["count", "pct", "type", "sentence"])

    # =========================================================
    # 2) "X 바나나를 Y" 패턴 분석 (이번 실험의 진짜 흥미로운 부분)
    # =========================================================
    print("\n--- 'X 바나나를 Y' 패턴 발생 빈도 ---")
    persons = ["철수는", "영희는", "짱구는"]
    verbs = ["좋아해", "싫어해"]

    pattern_rows = [["pattern", "type",
                     "ng_baseline", "ng_emphasis", "nlm_baseline", "nlm_emphasis"]]
    print(f"  {'pattern':<22}  {'type':<6}  {'ng_base':>8}  {'ng_emph':>8}  "
          f"{'nlm_base':>8}  {'nlm_emph':>8}")
    for p in persons:
        for v in verbs:
            sent = f"{p} 바나나를 {v}"
            t = "train" if sent in train_set else "novel"
            row = [sent, t,
                   base_ng_dist.get(sent, 0),
                   emph_ng_dist.get(sent, 0),
                   base_nlm_dist.get(sent, 0),
                   emph_nlm_dist.get(sent, 0)]
            pattern_rows.append(row)
            print(f"  {sent:<22}  {t:<6}  "
                  f"{row[2]:>8d}  {row[3]:>8d}  {row[4]:>8d}  {row[5]:>8d}")
    write_table(pattern_rows[1:], f"{RESULTS_DIR}/exp10_banana_patterns.tsv",
                header=pattern_rows[0])

    # =========================================================
    # 3) 핵심 컨텍스트 P(다음 토큰)
    # =========================================================
    print("\n--- 핵심 컨텍스트 P(좋아해) vs P(싫어해) ---")
    ctx_rows = [["context", "model", "setup",
                 "P(좋아해)", "P(싫어해)", "argmax", "P(argmax)"]]
    print(f"  {'context':<22}  {'model':<5} {'setup':<8}  "
          f"{'P(좋)':>8}  {'P(싫)':>8}  {'argmax':<10}  {'P(arg)':>8}")
    for ctx_pair in KEY_CONTEXTS:
        ctx_str = f"({ctx_pair[0]}, {ctx_pair[1]})"
        for label_setup, models in [("baseline", base), ("emphasis", emph)]:
            for mname in ["nlm", "ngram"]:
                p = context_probs(models[mname], stoi, ctx_pair)
                p_like = p[stoi["좋아해"]] if "좋아해" in stoi else 0.0
                p_dis = p[stoi["싫어해"]] if "싫어해" in stoi else 0.0
                arg = int(np.argmax(p))
                ctx_rows.append([ctx_str, mname, label_setup,
                                 f"{p_like:.4f}", f"{p_dis:.4f}",
                                 itos[arg], f"{p[arg]:.4f}"])
                print(f"  {ctx_str:<22}  {mname:<5} {label_setup:<8}  "
                      f"{p_like:>8.4f}  {p_dis:>8.4f}  {itos[arg]:<10}  {p[arg]:>8.4f}")
    write_table(ctx_rows[1:], f"{RESULTS_DIR}/exp10_context_probs.tsv",
                header=ctx_rows[0])

    # =========================================================
    # 4) Novel 문장 perplexity 비교
    # =========================================================
    print("\n--- Novel 문장 perplexity ---")
    ppl_rows = [["sentence", "ng_baseline_k0.01", "ng_emphasis_k0.01",
                 "nlm_baseline", "nlm_emphasis"]]
    print(f"  {'sentence':<28}  {'ng_base':>10}  {'ng_emph':>10}  "
          f"{'nlm_base':>10}  {'nlm_emph':>10}")
    for sent in NOVEL_TEST_SENTENCES:
        # n-gram은 k=0.01 smoothing 사용 (k=0이면 0인 컨텍스트에서 inf)
        for m in [base["ngram"], emph["ngram"]]:
            m.smoothing_k = 0.01
        nb = novel_perplexity(base["ngram"], stoi, sent)
        ne = novel_perplexity(emph["ngram"], stoi, sent)
        for m in [base["ngram"], emph["ngram"]]:
            m.smoothing_k = 0.0
        nlb = novel_perplexity(base["nlm"], stoi, sent)
        nle = novel_perplexity(emph["nlm"], stoi, sent)
        ppl_rows.append([sent, f"{nb:.3f}", f"{ne:.3f}", f"{nlb:.3f}", f"{nle:.3f}"])
        print(f"  {sent:<28}  {nb:>10.3f}  {ne:>10.3f}  {nlb:>10.3f}  {nle:>10.3f}")
    write_table(ppl_rows[1:], f"{RESULTS_DIR}/exp10_novel_ppl.tsv", header=ppl_rows[0])

    # =========================================================
    # 시각화 1: 4-panel 생성 분포
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [
        (axes[0, 0], "n-gram baseline (24 pairs)", base_ng_dist),
        (axes[0, 1], f"n-gram emphasis (×{EMPHASIS_REPEAT})", emph_ng_dist),
        (axes[1, 0], "NLM baseline", base_nlm_dist),
        (axes[1, 1], f"NLM emphasis (×{EMPHASIS_REPEAT})", emph_nlm_dist),
    ]
    for ax, title, counter in panels:
        rows = counter.most_common(12)
        labels = [s for s, _ in rows]
        counts = [c for _, c in rows]
        colors = []
        for s, _ in rows:
            if s == EMPHASIS_TARGET:
                colors.append("#dc2626")  # 강조: 빨강
            elif s in train_set:
                colors.append("#3b82f6")  # train: 파랑
            else:
                colors.append("#fb923c")  # novel: 주황
        ax.bar(range(len(rows)), counts, color=colors)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("빈도 / 10,000")
    fig.suptitle(
        f"실험 10: '{EMPHASIS_TARGET}' ×{EMPHASIS_REPEAT} 가중 학습 후 생성 분포 비교 "
        f"(빨강=강조 문장, 파랑=train, 주황=novel)",
        fontsize=11)
    fig.tight_layout()
    out_fig = f"{RESULTS_DIR}/exp10_distribution.png"
    fig.savefig(out_fig, dpi=120)
    plt.close(fig)
    print(f"\n저장: {out_fig}")

    # =========================================================
    # 시각화 2: 핵심 컨텍스트 P(좋아해) vs P(싫어해) 비교 (NLM)
    # =========================================================
    fig, ax = plt.subplots(figsize=(12, 5))
    n_ctx = len(KEY_CONTEXTS)
    width = 0.35
    x = np.arange(n_ctx)

    base_like = [context_probs(base["nlm"], stoi, c)[stoi["좋아해"]] for c in KEY_CONTEXTS]
    emph_like = [context_probs(emph["nlm"], stoi, c)[stoi["좋아해"]] for c in KEY_CONTEXTS]
    base_dis = [context_probs(base["nlm"], stoi, c)[stoi["싫어해"]] for c in KEY_CONTEXTS]
    emph_dis = [context_probs(emph["nlm"], stoi, c)[stoi["싫어해"]] for c in KEY_CONTEXTS]

    # 4-bar group
    ax.bar(x - 1.5*width/2, base_like, width/1.8, label="baseline P(좋아해)", color="#3b82f6")
    ax.bar(x - 0.5*width/2, base_dis,  width/1.8, label="baseline P(싫어해)", color="#93c5fd")
    ax.bar(x + 0.5*width/2, emph_like, width/1.8, label=f"×{EMPHASIS_REPEAT} P(좋아해)", color="#dc2626")
    ax.bar(x + 1.5*width/2, emph_dis,  width/1.8, label=f"×{EMPHASIS_REPEAT} P(싫어해)", color="#fca5a5")

    ax.set_xticks(x)
    ax.set_xticklabels([f"({a},{b})" for a, b in KEY_CONTEXTS], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("확률")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"실험 10: NLM이 컨텍스트별로 부여하는 P(좋아해)·P(싫어해) — baseline vs ×{EMPHASIS_REPEAT}")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_fig2 = f"{RESULTS_DIR}/exp10_context_probs.png"
    fig.savefig(out_fig2, dpi=120)
    plt.close(fig)
    print(f"저장: {out_fig2}")

    # =========================================================
    # 시각화 3: NLM 임베딩 비교 (baseline vs emphasis)
    # =========================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # 카테고리 색
    cat_color = {
        "<pad>": "#888", "<bos>": "#888", "<eos>": "#888",
        "철수는": "#3b82f6", "영희는": "#3b82f6", "짱구는": "#dc2626",
        "사과를": "#22c55e", "딸기를": "#22c55e", "바나나를": "#22c55e",
        "좋아해": "#a855f7", "싫어해": "#fb923c",
    }
    for ax, title, model in [
        (axes[0], "baseline NLM 임베딩", base["nlm"]),
        (axes[1], f"emphasis ×{EMPHASIS_REPEAT} NLM 임베딩", emph["nlm"]),
    ]:
        E = model.E
        for i, w in enumerate(itos):
            ax.scatter(E[i, 0], E[i, 1], color=cat_color.get(w, "#888"), s=80,
                       edgecolors="black", linewidth=0.5)
            ax.annotate(w, (E[i, 0], E[i, 1]), xytext=(5, 5), textcoords="offset points",
                        fontsize=10)
        ax.set_title(title)
        ax.axhline(0, color="#ccc", lw=0.5)
        ax.axvline(0, color="#ccc", lw=0.5)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"실험 10: 한 문장 ×{EMPHASIS_REPEAT} 가중 학습 후 임베딩 공간 변화", fontsize=11)
    fig.tight_layout()
    out_fig3 = f"{RESULTS_DIR}/exp10_embedding.png"
    fig.savefig(out_fig3, dpi=120)
    plt.close(fig)
    print(f"저장: {out_fig3}")

    # =========================================================
    # 시각화 4: 학습 곡선 비교 (NLM)
    # =========================================================
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(base["history"], label=f"baseline ({base['n_pairs']} pairs)")
    ax.plot(emph["history"], label=f"emphasis ({emph['n_pairs']} pairs)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("실험 10: NLM 학습 곡선 (baseline vs emphasis)")
    ax.legend()
    fig.tight_layout()
    out_fig4 = f"{RESULTS_DIR}/exp10_loss_curve.png"
    fig.savefig(out_fig4, dpi=120)
    plt.close(fig)
    print(f"저장: {out_fig4}")

    print("\n=== 실험 10 완료 ===")


if __name__ == "__main__":
    run()
