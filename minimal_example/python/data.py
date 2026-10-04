"""학습 데이터 준비 모듈.

6개 문장에서 trigram 학습 쌍 (w1, w2) -> w3 을 만든다.
어휘에는 <pad>, <bos>, <eos> 특수 토큰을 포함한다.
"""

import sys

# Windows 콘솔에서도 한글이 깨지지 않도록
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CONTEXT_LEN = 2  # trigram

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

SENTENCES = [
    "철수는 사과를 좋아해",
    "철수는 딸기를 좋아해",
    "영희는 사과를 좋아해",
    "영희는 딸기를 좋아해",
    "영희는 바나나를 좋아해",
    "짱구는 바나나를 싫어해",
]


def build_vocab(sentences=SENTENCES):
    """vocab: 토큰 → id, id → 토큰. 특수 토큰을 0,1,2에 고정."""
    words = []
    for s in sentences:
        for w in s.split():
            if w not in words:
                words.append(w)
    # 특수 토큰을 앞에 두고, 등장 순서를 그 뒤에
    itos = [PAD, BOS, EOS] + words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def make_pairs(sentences=SENTENCES, context_len=CONTEXT_LEN):
    """문장 → (context_tokens, next_token) 쌍 리스트."""
    pairs = []
    for s in sentences:
        tokens = [PAD] * (context_len - 1) + [BOS] + s.split() + [EOS]
        for i in range(len(tokens) - context_len):
            ctx = tuple(tokens[i : i + context_len])
            nxt = tokens[i + context_len]
            pairs.append((ctx, nxt))
    return pairs


def encode_pairs(pairs, stoi):
    """문자열 쌍을 정수 id 배열로 변환. (X: [N, context_len], y: [N])."""
    import numpy as np

    X = np.array([[stoi[t] for t in ctx] for ctx, _ in pairs], dtype=np.int64)
    y = np.array([stoi[nxt] for _, nxt in pairs], dtype=np.int64)
    return X, y


if __name__ == "__main__":
    stoi, itos = build_vocab()
    pairs = make_pairs()
    print(f"vocab size = {len(itos)}")
    print(f"itos = {itos}")
    print(f"num pairs = {len(pairs)}")
    print("샘플 학습 쌍:")
    for ctx, nxt in pairs[:8]:
        print(f"  {ctx} -> {nxt}")
