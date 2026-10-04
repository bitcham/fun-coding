"""Trigram n-gram 모델 (CONTEXT_LEN = 2).

학습 쌍 (ctx, w)에서 카운트를 모으고, MLE로 P(w | ctx)를 추정한다.
샘플링과 perplexity 계산을 위해 add-k smoothing 옵션을 제공한다.
"""

from collections import defaultdict

import numpy as np


class NGramModel:
    def __init__(self, vocab_size, smoothing_k=0.0):
        self.vocab_size = vocab_size
        self.smoothing_k = smoothing_k
        # ctx_counts[ctx][w] = count, ctx_totals[ctx] = sum
        self.ctx_counts = defaultdict(lambda: np.zeros(vocab_size, dtype=np.float64))
        self.ctx_totals = defaultdict(float)

    def fit(self, X, y):
        """X: [N, context_len] (int), y: [N] (int)."""
        for ctx_arr, w in zip(X, y):
            ctx = tuple(int(t) for t in ctx_arr)
            self.ctx_counts[ctx][int(w)] += 1.0
            self.ctx_totals[ctx] += 1.0

    def prob(self, ctx):
        """ctx (tuple of int) → 길이 vocab_size의 확률 분포 (numpy array)."""
        ctx = tuple(int(t) for t in ctx)
        k = self.smoothing_k
        counts = self.ctx_counts.get(ctx)
        if counts is None:
            # 모르는 ctx → uniform (smoothing 없을 때도 합리적 fallback)
            return np.full(self.vocab_size, 1.0 / self.vocab_size)
        total = self.ctx_totals[ctx] + k * self.vocab_size
        return (counts + k) / total

    def has_context(self, ctx):
        return tuple(int(t) for t in ctx) in self.ctx_counts

    def sample(self, ctx, rng):
        """multinomial 샘플링."""
        p = self.prob(ctx)
        # 안정성: 합 정규화
        p = p / p.sum()
        return int(rng.choice(self.vocab_size, p=p))

    def generate(self, stoi, itos, rng, max_len=20):
        """문장 생성. <bos>로 시작 → <eos>까지."""
        pad, bos, eos = stoi["<pad>"], stoi["<bos>"], stoi["<eos>"]
        ctx = [pad, bos]
        out = []
        for _ in range(max_len):
            w = self.sample(tuple(ctx), rng)
            if w == eos:
                break
            out.append(w)
            ctx = [ctx[-1], w]
        return [itos[t] for t in out]

    def perplexity(self, X, y, smoothing_k=None):
        """문맥별 P(y | X)의 평균 음의 로그우도 → perplexity = exp(NLL).

        smoothing_k 인자로 평가 시 별도 k를 줄 수 있음 (None이면 self.smoothing_k 사용).
        unseen ctx는 uniform이라 자동 처리됨.
        """
        k = self.smoothing_k if smoothing_k is None else smoothing_k
        old_k = self.smoothing_k
        self.smoothing_k = k
        try:
            log_probs = []
            for ctx_arr, w in zip(X, y):
                p = self.prob(tuple(int(t) for t in ctx_arr))
                p_w = max(p[int(w)], 1e-30)
                log_probs.append(np.log(p_w))
            nll = -np.mean(log_probs)
            return float(np.exp(nll))
        finally:
            self.smoothing_k = old_k


if __name__ == "__main__":
    from utils import setup_io, set_seed
    from data import build_vocab, make_pairs, encode_pairs

    setup_io()
    set_seed(0)
    stoi, itos = build_vocab()
    pairs = make_pairs()
    X, y = encode_pairs(pairs, stoi)

    model = NGramModel(vocab_size=len(itos), smoothing_k=0.0)
    model.fit(X, y)

    print(f"학습 쌍 수: {len(pairs)}, vocab size: {len(itos)}")
    print(f"학습 perplexity (k=0): {model.perplexity(X, y):.4f}")
    print(f"학습 perplexity (k=0.01): {model.perplexity(X, y, smoothing_k=0.01):.4f}")

    print()
    print("(<pad>, <bos>) 분포:")
    p = model.prob((stoi["<pad>"], stoi["<bos>"]))
    for i, prob in enumerate(p):
        if prob > 0:
            print(f"  P({itos[i]}) = {prob:.4f}")

    print()
    print("샘플 5문장:")
    rng = np.random.default_rng(1)
    for _ in range(5):
        toks = model.generate(stoi, itos, rng)
        print("  " + " ".join(toks))

    print()
    print("(철수는, 바나나를) 분포:")
    ctx = (stoi["철수는"], stoi["바나나를"])
    print(f"  학습에 등장? {model.has_context(ctx)}")
    p = model.prob(ctx)
    nz = (p > 1e-9).sum()
    print(f"  nonzero entries: {nz} (uniform으로 fallback)")
