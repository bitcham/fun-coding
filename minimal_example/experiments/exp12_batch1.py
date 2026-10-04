"""실험 12: batch_size=1 (true SGD)로 baseline과 ×10 emphasis 학습.

exp11(full-batch GD)의 짝꿍 실험. 같은 데이터·같은 가중치를 batch=1로 학습.

차이점:
- full-batch: 매 epoch 1번 update (전체 평균 그래디언트)
- batch=1:   매 epoch N번 update (한 sample씩, shuffle된 순서로)

같은 epoch 수에서 update 횟수가 N배 차이 나므로 결과가 크게 달라질 수 있다.
"""

import numpy as np

from exp_common import RESULTS_DIR, count_sentences, write_table, fmt_pct
from utils import setup_io, set_seed, setup_matplotlib_korean
from data import build_vocab, make_pairs, encode_pairs, SENTENCES
from ngram import NGramModel
from nlm import NLM, train

EMB_DIM = 2
HIDDEN = 8
# lr scaling: full-batch lr=0.3은 24 sample 평균 그래디언트 기준.
# batch=1이면 한 sample 그래디언트가 평균의 ~24배라 lr도 1/24로 줄여야 등가.
# (linear scaling rule: lr ∝ batch_size)
LR = 0.01
EPOCHS = 2000
SEED = 123
BATCH_SEED = 777
N_SAMPLES = 10000

EMPHASIS_REPEAT = 10
EMPHASIS_TARGET = "짱구는 바나나를 싫어해"

KEY_CONTEXTS = [
    ("<bos>", "철수는"),
    ("<bos>", "영희는"),
    ("<bos>", "짱구는"),
    ("철수는", "사과를"),
    ("철수는", "바나나를"),
    ("영희는", "바나나를"),
    ("짱구는", "바나나를"),
    ("짱구는", "사과를"),
    ("바나나를", "좋아해"),
    ("바나나를", "싫어해"),
]

NOVEL_TEST_SENTENCES = [
    "철수는 바나나를 싫어해",
    "영희는 바나나를 싫어해",
    "철수는 바나나를 좋아해",
    "짱구는 사과를 싫어해",
    "짱구는 딸기를 싫어해",
]


def make_emphasis_sentences():
    base_5 = [s for s in SENTENCES if s != EMPHASIS_TARGET]
    return base_5 + [EMPHASIS_TARGET] * EMPHASIS_REPEAT


def train_pair(stoi, itos, sentences, label):
    pairs = make_pairs(sentences=sentences)
    X, y = encode_pairs(pairs, stoi)

    set_seed(SEED)
    nlm = NLM(vocab_size=len(itos), emb_dim=EMB_DIM, hidden=HIDDEN, seed=SEED)
    history, _ = train(nlm, X, y, lr=LR, epochs=EPOCHS,
                       batch_size=1, shuffle=True, batch_seed=BATCH_SEED)

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
    ctx = (stoi[ctx_pair[0]], stoi[ctx_pair[1]])
    if isinstance(model, NLM):
        p = model.predict_proba(np.array([list(ctx)], dtype=np.int64))[0]
    else:
        p = model.prob(ctx)
    return p


def novel_perplexity(model, stoi, sentence):
    pairs = make_pairs(sentences=[sentence])
    X, y = encode_pairs(pairs, stoi)
    return model.perplexity(X, y)


def run():
    setup_io()
    plt = setup_matplotlib_korean()

    stoi, itos = build_vocab()

    print(f"== 실험 12: batch_size=1 (true SGD) ==")
    print(f"강조 문장: '{EMPHASIS_TARGET}' × {EMPHASIS_REPEAT}")
    print(f"하이퍼파라미터: D={EMB_DIM}, H={HIDDEN}, lr={LR}, ep={EPOCHS}, "
          f"seed={SEED}, batch_seed={BATCH_SEED}, batch_size=1, shuffle=True")
    print()

    base_sents = list(SENTENCES)
    emph_sents = make_emphasis_sentences()
    print(f"baseline 문장 수: {len(base_sents)}, emphasis 문장 수: {len(emph_sents)}")
    print(f"epoch당 update 수 — baseline: 24, emphasis: 60")
    print()

    base = train_pair(stoi, itos, base_sents, "baseline_b1")
    emph = train_pair(stoi, itos, emph_sents, "emphasis_x10_b1")

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
        print(f"  [{label:<22}] unique={len(counter):3d}  "
              f"train={in_train:5d} ({fmt_pct(in_train/N_SAMPLES):>7})  "
              f"novel={novel:5d} ({fmt_pct(novel/N_SAMPLES):>7})  "
              f"'{EMPHASIS_TARGET}'={emph_count:5d}")
        return in_train, novel, emph_count

    summary_rows = []
    for label, counter in [
        ("ngram baseline b=1",     base_ng_dist),
        ("ngram emphasis x10 b=1", emph_ng_dist),
        ("NLM   baseline b=1",     base_nlm_dist),
        ("NLM   emphasis x10 b=1", emph_nlm_dist),
    ]:
        in_t, nov, ec = summarize(label, counter)
        summary_rows.append((label, len(counter), in_t, nov, ec))

    write_table(summary_rows,
                f"{RESULTS_DIR}/exp12_summary.tsv",
                header=["model", "unique_sentences", "in_train", "novel", "emphasis_count"])

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
        write_table(rows, f"{RESULTS_DIR}/exp12_dist_{tag}.tsv",
                    header=["count", "pct", "type", "sentence"])

    # =========================================================
    # 2) "X 바나나를 Y" 패턴 분석
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
    write_table(pattern_rows[1:], f"{RESULTS_DIR}/exp12_banana_patterns.tsv",
                header=pattern_rows[0])

    # =========================================================
    # 3) 핵심 컨텍스트 P(다음 토큰)
    # =========================================================
    print("\n--- 핵심 컨텍스트 P(좋아해) vs P(싫어해) ---")
    ctx_rows = [["context", "model", "setup",
                 "P(좋아해)", "P(싫어해)", "argmax", "P(argmax)"]]
    print(f"  {'context':<22}  {'model':<5} {'setup':<10}  "
          f"{'P(좋)':>8}  {'P(싫)':>8}  {'argmax':<10}  {'P(arg)':>8}")
    for ctx_pair in KEY_CONTEXTS:
        ctx_str = f"({ctx_pair[0]}, {ctx_pair[1]})"
        for label_setup, models in [("baseline", base), ("emph_x10", emph)]:
            for mname in ["nlm", "ngram"]:
                p = context_probs(models[mname], stoi, ctx_pair)
                p_like = p[stoi["좋아해"]]
                p_dis = p[stoi["싫어해"]]
                arg = int(np.argmax(p))
                ctx_rows.append([ctx_str, mname, label_setup,
                                 f"{p_like:.4f}", f"{p_dis:.4f}",
                                 itos[arg], f"{p[arg]:.4f}"])
                print(f"  {ctx_str:<22}  {mname:<5} {label_setup:<10}  "
                      f"{p_like:>8.4f}  {p_dis:>8.4f}  {itos[arg]:<10}  {p[arg]:>8.4f}")
    write_table(ctx_rows[1:], f"{RESULTS_DIR}/exp12_context_probs.tsv",
                header=ctx_rows[0])

    # =========================================================
    # 4) Novel 문장 perplexity
    # =========================================================
    print("\n--- Novel 문장 perplexity ---")
    ppl_rows = [["sentence", "ng_baseline_k0.01", "ng_emphasis_k0.01",
                 "nlm_baseline", "nlm_emphasis"]]
    print(f"  {'sentence':<28}  {'ng_base':>10}  {'ng_emph':>10}  "
          f"{'nlm_base':>10}  {'nlm_emph':>10}")
    for sent in NOVEL_TEST_SENTENCES:
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
    write_table(ppl_rows[1:], f"{RESULTS_DIR}/exp12_novel_ppl.tsv", header=ppl_rows[0])

    # =========================================================
    # 시각화 1: 4-panel 생성 분포
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [
        (axes[0, 0], "n-gram baseline (b=1)", base_ng_dist),
        (axes[0, 1], f"n-gram emphasis ×{EMPHASIS_REPEAT} (b=1)", emph_ng_dist),
        (axes[1, 0], "NLM baseline (b=1)", base_nlm_dist),
        (axes[1, 1], f"NLM emphasis ×{EMPHASIS_REPEAT} (b=1)", emph_nlm_dist),
    ]
    for ax, title, counter in panels:
        rows = counter.most_common(12)
        labels = [s for s, _ in rows]
        counts = [c for _, c in rows]
        colors = []
        for s, _ in rows:
            if s == EMPHASIS_TARGET:
                colors.append("#dc2626")
            elif s in train_set:
                colors.append("#3b82f6")
            else:
                colors.append("#fb923c")
        ax.bar(range(len(rows)), counts, color=colors)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("빈도 / 10,000")
    fig.suptitle(
        f"실험 12: batch_size=1 SGD — '{EMPHASIS_TARGET}' ×{EMPHASIS_REPEAT} "
        f"(빨강=강조, 파랑=train, 주황=novel)",
        fontsize=11)
    fig.tight_layout()
    out_fig = f"{RESULTS_DIR}/exp12_distribution.png"
    fig.savefig(out_fig, dpi=120)
    plt.close(fig)
    print(f"\n저장: {out_fig}")

    # =========================================================
    # 시각화 2: 컨텍스트 P 비교
    # =========================================================
    fig, ax = plt.subplots(figsize=(12, 5))
    n_ctx = len(KEY_CONTEXTS)
    width = 0.35
    x = np.arange(n_ctx)

    base_like = [context_probs(base["nlm"], stoi, c)[stoi["좋아해"]] for c in KEY_CONTEXTS]
    emph_like = [context_probs(emph["nlm"], stoi, c)[stoi["좋아해"]] for c in KEY_CONTEXTS]
    base_dis = [context_probs(base["nlm"], stoi, c)[stoi["싫어해"]] for c in KEY_CONTEXTS]
    emph_dis = [context_probs(emph["nlm"], stoi, c)[stoi["싫어해"]] for c in KEY_CONTEXTS]

    ax.bar(x - 1.5*width/2, base_like, width/1.8, label="baseline P(좋아해)", color="#3b82f6")
    ax.bar(x - 0.5*width/2, base_dis,  width/1.8, label="baseline P(싫어해)", color="#93c5fd")
    ax.bar(x + 0.5*width/2, emph_like, width/1.8, label=f"×{EMPHASIS_REPEAT} P(좋아해)", color="#dc2626")
    ax.bar(x + 1.5*width/2, emph_dis,  width/1.8, label=f"×{EMPHASIS_REPEAT} P(싫어해)", color="#fca5a5")

    ax.set_xticks(x)
    ax.set_xticklabels([f"({a},{b})" for a, b in KEY_CONTEXTS], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("확률")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"실험 12 (b=1): NLM 컨텍스트별 P(좋아해)·P(싫어해)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_fig2 = f"{RESULTS_DIR}/exp12_context_probs.png"
    fig.savefig(out_fig2, dpi=120)
    plt.close(fig)
    print(f"저장: {out_fig2}")

    # =========================================================
    # 시각화 3: 임베딩
    # =========================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cat_color = {
        "<pad>": "#888", "<bos>": "#888", "<eos>": "#888",
        "철수는": "#3b82f6", "영희는": "#3b82f6", "짱구는": "#dc2626",
        "사과를": "#22c55e", "딸기를": "#22c55e", "바나나를": "#22c55e",
        "좋아해": "#a855f7", "싫어해": "#fb923c",
    }
    for ax, title, model in [
        (axes[0], "baseline NLM 임베딩 (b=1)", base["nlm"]),
        (axes[1], f"emphasis ×{EMPHASIS_REPEAT} NLM 임베딩 (b=1)", emph["nlm"]),
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
    fig.suptitle(f"실험 12 (b=1): 한 문장 ×{EMPHASIS_REPEAT} 가중 학습 후 임베딩", fontsize=11)
    fig.tight_layout()
    out_fig3 = f"{RESULTS_DIR}/exp12_embedding.png"
    fig.savefig(out_fig3, dpi=120)
    plt.close(fig)
    print(f"저장: {out_fig3}")

    # =========================================================
    # 시각화 4: 학습 곡선 (full-batch 비교)
    # =========================================================
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(base["history"], label=f"baseline b=1 ({base['n_pairs']} pairs)")
    ax.plot(emph["history"], label=f"emphasis ×{EMPHASIS_REPEAT} b=1 ({emph['n_pairs']} pairs)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss (epoch 끝 전체)")
    ax.set_title(f"실험 12: NLM 학습 곡선 (batch_size=1)")
    ax.legend()
    fig.tight_layout()
    out_fig4 = f"{RESULTS_DIR}/exp12_loss_curve.png"
    fig.savefig(out_fig4, dpi=120)
    plt.close(fig)
    print(f"저장: {out_fig4}")

    print("\n=== 실험 12 완료 ===")


if __name__ == "__main__":
    run()
