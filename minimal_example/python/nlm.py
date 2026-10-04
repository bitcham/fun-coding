"""뉴럴 언어 모델: word embedding → tanh hidden → softmax output.

학습 목적이라 모든 식을 `loss_and_grad` 한 함수에 풀어 두었다.
forward와 backward를 한 곳에서 위에서 아래로 따라가며 읽을 수 있고,
C 버전 [c/nlm.c](../c/nlm.c)와 단계 번호가 1:1로 대응한다.

데이터 흐름 (B=배치, C=context, D=embedding, H=hidden, V=vocab):

    X        [B, C]    학습 쌍의 단어 ID 두 개
      ↓ ① 임베딩 lookup
    e        [B, C, D]
      ↓ ② flatten
    flat     [B, C*D]
      ↓ ③ flat @ W1 + b1, tanh
    h        [B, H]
      ↓ ④ h @ W2 + b2
    z        [B, V]   logits
      ↓ ⑤ softmax
    p        [B, V]   확률 분포
      ↓ ⑥ cross-entropy
    loss     스칼라
"""

import numpy as np


def softmax(z):
    """수치 안정 softmax (마지막 축). max를 빼고 exp → 합으로 나눔."""
    z_max = z.max(axis=-1, keepdims=True)
    e = np.exp(z - z_max)
    return e / e.sum(axis=-1, keepdims=True)


class NLM:
    def __init__(self, vocab_size, emb_dim, hidden, context_len=2, seed=123):
        rng = np.random.default_rng(seed)
        self.V, self.D, self.H, self.C = vocab_size, emb_dim, hidden, context_len

        # 가중치 랜덤 초기화 (Xavier-ish 스케일, tanh에 친화적)
        self.E  = rng.normal(0, 0.5, (self.V, self.D))                                # [V, D]
        self.W1 = rng.normal(0, np.sqrt(2.0 / (self.C * self.D + self.H)),
                             (self.C * self.D, self.H))                                # [C*D, H]
        self.b1 = np.zeros(self.H)                                                     # [H]
        self.W2 = rng.normal(0, np.sqrt(2.0 / (self.H + self.V)),
                             (self.H, self.V))                                         # [H, V]
        self.b2 = np.zeros(self.V)                                                     # [V]

    # =================================================================
    # 학습용: forward + backward 한 번에
    # =================================================================
    def loss_and_grad(self, X, y):
        """입력 (X, y) → loss와 모든 parameter의 그래디언트.

        X: [B, C] (단어 ID), y: [B] (정답 단어 ID).
        """
        B = X.shape[0]
        D, C = self.D, self.C

        # ---------- FORWARD ----------

        # ① 임베딩 lookup: X[b][c]는 단어 ID, 그 ID 행의 D차원 벡터를 꺼냄
        e = self.E[X]                                # [B, C, D]

        # ② 두 단어의 임베딩을 옆으로 이어 붙여 한 줄로
        flat = e.reshape(B, C * D)                   # [B, C*D]

        # ③ hidden layer: 선형 변환 + tanh
        #    b1[H]는 모든 행에 자동으로 더해짐 (numpy broadcasting)
        h = np.tanh(flat @ self.W1 + self.b1)        # [B, H]

        # ④ output layer: 선형 변환 → logits
        z = h @ self.W2 + self.b2                    # [B, V]

        # ⑤ softmax → 확률 분포 (행마다 합이 1)
        p = softmax(z)                               # [B, V]

        # ⑥ cross-entropy loss = -평균( log p[정답 단어] )
        correct_p = p[np.arange(B), y]               # 각 행에서 정답 위치의 확률 [B]
        loss = float(-np.log(np.maximum(correct_p, 1e-30)).mean())

        # ---------- BACKWARD ----------
        # 각 parameter에 대한 미분을 직접 구한다 (autograd 엔진 없음).

        # ⑦ ∂L/∂z = (p − one_hot(y)) / B
        #    softmax + cross-entropy를 합쳐서 미분하면 이렇게 깔끔하게 떨어진다.
        dz = p.copy()
        dz[np.arange(B), y] -= 1.0
        dz /= B                                      # [B, V]

        # ⑧ 출력층 W2, b2 미분
        gW2 = h.T @ dz                               # ∂L/∂W2 = hᵀ · dz       [H, V]
        gb2 = dz.sum(axis=0)                         # ∂L/∂b2 = Σ_b dz        [V]

        # ⑨ hidden 활성화까지 거꾸로 흘려보내기
        dh = dz @ self.W2.T                          # ∂L/∂h = dz · W2ᵀ       [B, H]
        # tanh 미분: dy/dx = 1 − tanh(x)² = 1 − h²
        dh_pre = dh * (1.0 - h * h)                  # ∂L/∂(tanh 입력)         [B, H]

        # ⑩ 은닉층 W1, b1 미분
        gW1 = flat.T @ dh_pre                        # ∂L/∂W1 = flatᵀ · dh_pre [C*D, H]
        gb1 = dh_pre.sum(axis=0)                     # ∂L/∂b1 = Σ_b dh_pre     [H]

        # ⑪ flat까지 흘려보내고, e 모양으로 다시 reshape
        dflat = dh_pre @ self.W1.T                   # [B, C*D]
        de    = dflat.reshape(B, C, D)               # [B, C, D]

        # ⑫ 임베딩 테이블 미분: 각 (b, c)에서 사용한 단어 ID 행에 누적 (scatter-add)
        gE = np.zeros_like(self.E)                   # [V, D]
        np.add.at(gE, X, de)                         # gE[X[b,c]] += de[b,c]

        return loss, {"E": gE, "W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}

    def sgd_step(self, grads, lr):
        """SGD 한 스텝: 모든 parameter에서 lr × grad 만큼 빼기"""
        self.E  -= lr * grads["E"]
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    # =================================================================
    # 추론용 forward (loss/grad 없이 확률만)
    # =================================================================
    def predict_proba(self, X):
        """X: [B, C] → 다음 토큰의 확률 분포 [B, V]."""
        B = X.shape[0]
        e    = self.E[X]                             # [B, C, D]
        flat = e.reshape(B, self.C * self.D)         # [B, C*D]
        h    = np.tanh(flat @ self.W1 + self.b1)     # [B, H]
        z    = h @ self.W2 + self.b2                 # [B, V]
        return softmax(z)                            # [B, V]

    def perplexity(self, X, y):
        p  = self.predict_proba(X)
        pw = np.maximum(p[np.arange(len(y)), y], 1e-30)
        return float(np.exp(-np.log(pw).mean()))

    def sample(self, ctx, rng):
        """ctx (단어 ID 두 개) → 다음 단어 ID (multinomial sampling)."""
        X = np.array([list(ctx)], dtype=np.int64)
        p = self.predict_proba(X)[0]
        p = p / p.sum()
        return int(rng.choice(self.V, p=p))

    def generate(self, stoi, itos, rng, max_len=20):
        """<bos>로 시작해서 <eos>를 만날 때까지 한 단어씩 sampling."""
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

    # ---------- 가중치 직렬화 (스냅샷용) ----------
    def get_state(self):
        return {"E":  self.E.copy(),  "W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    def load_state(self, s):
        self.E[:],  self.W1[:], self.b1[:] = s["E"],  s["W1"], s["b1"]
        self.W2[:], self.b2[:]             = s["W2"], s["b2"]


# =================================================================
# 학습 루프
# =================================================================
def train(model, X, y, lr=0.1, epochs=2000, batch_size=None, shuffle=True,
          batch_seed=123, log_path=None, snapshot_every=None):
    """SGD 학습 루프.

    batch_size=None: full-batch GD. 매 epoch에 전체 (X, y)로 1번 update.
    batch_size=정수: mini-batch SGD. 매 epoch에 shuffle 후 크기 N 미니배치
                    여러 번 update. epoch 끝에 전체 데이터 loss를 기록.
    """
    history = []
    snapshots = []
    N = X.shape[0]
    rng_batch = np.random.default_rng(batch_seed)

    for ep in range(epochs):
        if batch_size is None:
            loss, grads = model.loss_and_grad(X, y)
            model.sgd_step(grads, lr)
            history.append(loss)
        else:
            idx = rng_batch.permutation(N) if shuffle else np.arange(N)
            for s in range(0, N, batch_size):
                bi = idx[s : s + batch_size]
                _, grads = model.loss_and_grad(X[bi], y[bi])
                model.sgd_step(grads, lr)
            loss, _ = model.loss_and_grad(X, y)
            history.append(loss)

        if snapshot_every and (ep % snapshot_every == 0):
            snapshots.append((ep, model.get_state()))
    if snapshot_every:
        snapshots.append((epochs - 1, model.get_state()))
    if log_path is not None:
        with open(log_path, "w", encoding="utf-8") as f:
            for v in history:
                f.write(f"{v:.10f}\n")
    return history, snapshots


# =================================================================
# 자가 검증: 직접 미분이 수치 미분과 일치하는지 확인
# =================================================================
def _gradcheck():
    rng = np.random.default_rng(0)
    V, D, H, C, B = 7, 3, 5, 2, 4
    X = rng.integers(0, V, size=(B, C))
    y = rng.integers(0, V, size=(B,))

    m = NLM(V, D, H, C, seed=42)
    _, grads = m.loss_and_grad(X, y)

    eps = 1e-5
    for name, param in [("W1", m.W1), ("E", m.E)]:
        num = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            param[idx] += eps;     lp, _ = m.loss_and_grad(X, y)
            param[idx] -= 2 * eps; ln, _ = m.loss_and_grad(X, y)
            param[idx] += eps
            num[idx] = (lp - ln) / (2 * eps)
        diff = np.abs(num - grads[name]).max()
        print(f"gradcheck ({name:>2}) max abs diff = {diff:.3e}")


if __name__ == "__main__":
    from utils import setup_io, set_seed
    from data import build_vocab, make_pairs, encode_pairs

    setup_io()
    _gradcheck()

    set_seed(123)
    stoi, itos = build_vocab()
    pairs = make_pairs()
    X, y = encode_pairs(pairs, stoi)

    model = NLM(vocab_size=len(itos), emb_dim=2, hidden=8, seed=123)
    print(f"\n초기 loss = {model.loss_and_grad(X, y)[0]:.4f} "
          f"(uniform 기준 ~ {np.log(len(itos)):.4f})")

    history, _ = train(model, X, y, lr=0.3, epochs=1500)
    print(f"최종 loss = {history[-1]:.4f}, perplexity = {model.perplexity(X, y):.4f}")

    rng = np.random.default_rng(0)
    print("\n샘플 5문장:")
    for _ in range(5):
        print("  " + " ".join(model.generate(stoi, itos, rng)))

    print("\n(철수는, 바나나를) 분포 (학습에 없던 컨텍스트):")
    ctx = (stoi["철수는"], stoi["바나나를"])
    p = model.predict_proba(np.array([list(ctx)], dtype=np.int64))[0]
    for t in np.argsort(p)[::-1][:5]:
        print(f"  {itos[t]}: {p[t]:.4f}")
