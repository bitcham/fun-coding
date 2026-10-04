#include "nlm.h"
#include <math.h>
#include <string.h>

/*
 * 뉴럴 언어 모델: word embedding → tanh hidden → softmax output.
 *
 * 학습 목적이라 forward와 backward를 한 함수에 모두 풀어 둔다. 단계 번호
 * (① ~ ⑫)는 Python 버전 [python/nlm.py](../python/nlm.py)와 1:1로 대응한다.
 *
 * 데이터 흐름 (B=N_PAIRS, C=CONTEXT_LEN, D=EMB_DIM, H=HIDDEN, V=VOCAB_SIZE):
 *
 *     X[b][c]   --①-->  e[b][c][d]   --②-->  flat[b][i]
 *               (lookup)              (펼침)
 *                                              |
 *                                              | ③ flat @ W1 + b1, tanh
 *                                              v
 *                                            h[b][j]
 *                                              |
 *                                              | ④ h @ W2 + b2
 *                                              v
 *                                            z[b][v]   ⑤softmax→  p[b][v]
 *                                                                    |
 *                                                                    | ⑥ NLL
 *                                                                    v
 *                                                                  loss
 */

/* ----------------------------- 초기화 ----------------------------- */

void nlm_init(nlm_t *m, rng_t *r) {
    /* Xavier-ish 초기화 (tanh에 친화적인 작은 정규난수) */
    double s_E  = 0.5;
    double s_W1 = sqrt(2.0 / (double)(FLAT_DIM + HIDDEN));
    double s_W2 = sqrt(2.0 / (double)(HIDDEN + VOCAB_SIZE));

    for (int v = 0; v < VOCAB_SIZE; v++)
        for (int d = 0; d < EMB_DIM; d++)
            m->E[v][d] = rng_normal(r, 0.0, s_E);

    for (int i = 0; i < FLAT_DIM; i++)
        for (int j = 0; j < HIDDEN; j++)
            m->W1[i][j] = rng_normal(r, 0.0, s_W1);

    for (int j = 0; j < HIDDEN; j++) m->b1[j] = 0.0;

    for (int j = 0; j < HIDDEN; j++)
        for (int v = 0; v < VOCAB_SIZE; v++)
            m->W2[j][v] = rng_normal(r, 0.0, s_W2);

    for (int v = 0; v < VOCAB_SIZE; v++) m->b2[v] = 0.0;
}

/* ============================================================= */
/* 학습용: forward + backward 한 번에                              */
/* ============================================================= */

double nlm_loss_and_grad(const nlm_t *m, nlm_grad_t *g,
                         const int X[N_PAIRS][CONTEXT_LEN],
                         const int y[N_PAIRS]) {
    /* 모든 중간 활성화는 stack 배열로 (작은 사이즈라 충분) */
    double flat[N_PAIRS][FLAT_DIM];
    double h[N_PAIRS][HIDDEN];
    double z[N_PAIRS][VOCAB_SIZE];
    double p[N_PAIRS][VOCAB_SIZE];

    /* ---------------------- FORWARD ---------------------- */

    /* ① 임베딩 lookup + ② flatten:
     *    flat[b][c*D + d] = E[X[b][c]][d]
     *    두 단어의 임베딩을 옆으로 이어 붙여 한 벡터로 만든다. */
    for (int b = 0; b < N_PAIRS; b++) {
        for (int c = 0; c < CONTEXT_LEN; c++) {
            int id = X[b][c];
            for (int d = 0; d < EMB_DIM; d++) {
                flat[b][c * EMB_DIM + d] = m->E[id][d];
            }
        }
    }

    /* ③ hidden layer: h = tanh(flat @ W1 + b1) */
    for (int b = 0; b < N_PAIRS; b++) {
        for (int j = 0; j < HIDDEN; j++) {
            double s = m->b1[j];
            for (int i = 0; i < FLAT_DIM; i++) s += flat[b][i] * m->W1[i][j];
            h[b][j] = tanh(s);
        }
    }

    /* ④ output layer: z = h @ W2 + b2  (logits) */
    for (int b = 0; b < N_PAIRS; b++) {
        for (int v = 0; v < VOCAB_SIZE; v++) {
            double s = m->b2[v];
            for (int j = 0; j < HIDDEN; j++) s += h[b][j] * m->W2[j][v];
            z[b][v] = s;
        }
    }

    /* ⑤ softmax (행 단위, max 빼고 안정화) */
    for (int b = 0; b < N_PAIRS; b++) {
        double mx = z[b][0];
        for (int v = 1; v < VOCAB_SIZE; v++) if (z[b][v] > mx) mx = z[b][v];
        double sum = 0.0;
        for (int v = 0; v < VOCAB_SIZE; v++) { p[b][v] = exp(z[b][v] - mx); sum += p[b][v]; }
        for (int v = 0; v < VOCAB_SIZE; v++) p[b][v] /= sum;
    }

    /* ⑥ cross-entropy loss = -평균( log p[정답 단어] ) */
    double loss = 0.0;
    for (int b = 0; b < N_PAIRS; b++) {
        double pw = p[b][y[b]];
        if (pw < 1e-30) pw = 1e-30;
        loss -= log(pw);
    }
    loss /= (double)N_PAIRS;

    /* ---------------------- BACKWARD ---------------------- */

    double dz[N_PAIRS][VOCAB_SIZE];
    double dh_pre[N_PAIRS][HIDDEN];
    double dflat[N_PAIRS][FLAT_DIM];

    /* ⑦ ∂L/∂z = (p - one_hot(y)) / B
     *    softmax + cross-entropy 합성 미분의 깔끔한 결과 */
    for (int b = 0; b < N_PAIRS; b++) {
        for (int v = 0; v < VOCAB_SIZE; v++) dz[b][v] = p[b][v] / (double)N_PAIRS;
        dz[b][y[b]] -= 1.0 / (double)N_PAIRS;
    }

    /* ⑧ 출력층 W2, b2 미분
     *    gW2[j][v] = Σ_b h[b][j] * dz[b][v]   (= hᵀ · dz)
     *    gb2[v]    = Σ_b dz[b][v] */
    for (int j = 0; j < HIDDEN; j++) {
        for (int v = 0; v < VOCAB_SIZE; v++) {
            double s = 0.0;
            for (int b = 0; b < N_PAIRS; b++) s += h[b][j] * dz[b][v];
            g->W2[j][v] = s;
        }
    }
    for (int v = 0; v < VOCAB_SIZE; v++) {
        double s = 0.0;
        for (int b = 0; b < N_PAIRS; b++) s += dz[b][v];
        g->b2[v] = s;
    }

    /* ⑨ hidden 활성화로 미분 흘려보내기
     *    dh[b][j]     = Σ_v dz[b][v] * W2[j][v]    (= dz · W2ᵀ)
     *    dh_pre[b][j] = dh[b][j] * (1 - h[b][j]²)   (tanh 미분: 1 - tanh²)
     *    dh는 dh_pre로 바로 변환되니 별도 배열 없이 스칼라로 처리 */
    for (int b = 0; b < N_PAIRS; b++) {
        for (int j = 0; j < HIDDEN; j++) {
            double dh = 0.0;
            for (int v = 0; v < VOCAB_SIZE; v++) dh += dz[b][v] * m->W2[j][v];
            dh_pre[b][j] = dh * (1.0 - h[b][j] * h[b][j]);
        }
    }

    /* ⑩ 은닉층 W1, b1 미분
     *    gW1[i][j] = Σ_b flat[b][i] * dh_pre[b][j]
     *    gb1[j]    = Σ_b dh_pre[b][j] */
    for (int i = 0; i < FLAT_DIM; i++) {
        for (int j = 0; j < HIDDEN; j++) {
            double s = 0.0;
            for (int b = 0; b < N_PAIRS; b++) s += flat[b][i] * dh_pre[b][j];
            g->W1[i][j] = s;
        }
    }
    for (int j = 0; j < HIDDEN; j++) {
        double s = 0.0;
        for (int b = 0; b < N_PAIRS; b++) s += dh_pre[b][j];
        g->b1[j] = s;
    }

    /* ⑪ flat까지 흘려보내기
     *    dflat[b][i] = Σ_j dh_pre[b][j] * W1[i][j]   (= dh_pre · W1ᵀ) */
    for (int b = 0; b < N_PAIRS; b++) {
        for (int i = 0; i < FLAT_DIM; i++) {
            double s = 0.0;
            for (int j = 0; j < HIDDEN; j++) s += dh_pre[b][j] * m->W1[i][j];
            dflat[b][i] = s;
        }
    }

    /* ⑫ 임베딩 테이블 미분 (scatter-add):
     *    각 (b, c)에서 사용한 단어 ID 행에 dflat의 해당 슬라이스를 더한다 */
    for (int v = 0; v < VOCAB_SIZE; v++)
        for (int d = 0; d < EMB_DIM; d++) g->E[v][d] = 0.0;

    for (int b = 0; b < N_PAIRS; b++) {
        for (int c = 0; c < CONTEXT_LEN; c++) {
            int id = X[b][c];
            for (int d = 0; d < EMB_DIM; d++) {
                g->E[id][d] += dflat[b][c * EMB_DIM + d];
            }
        }
    }

    return loss;
}

/* ============================================================= */
/* SGD 한 스텝: 모든 parameter에서 lr × grad 만큼 빼기              */
/* ============================================================= */

void nlm_sgd_step(nlm_t *m, const nlm_grad_t *g, double lr) {
    for (int v = 0; v < VOCAB_SIZE; v++)
        for (int d = 0; d < EMB_DIM; d++) m->E[v][d] -= lr * g->E[v][d];
    for (int i = 0; i < FLAT_DIM; i++)
        for (int j = 0; j < HIDDEN; j++) m->W1[i][j] -= lr * g->W1[i][j];
    for (int j = 0; j < HIDDEN; j++)      m->b1[j]   -= lr * g->b1[j];
    for (int j = 0; j < HIDDEN; j++)
        for (int v = 0; v < VOCAB_SIZE; v++) m->W2[j][v] -= lr * g->W2[j][v];
    for (int v = 0; v < VOCAB_SIZE; v++)  m->b2[v]   -= lr * g->b2[v];
}

/* ============================================================= */
/* 추론용 forward (batch=1, loss/grad 없이 확률만)                  */
/* ============================================================= */

void nlm_predict(const nlm_t *m, int ctx_a, int ctx_b, double out[VOCAB_SIZE]) {
    double flat[FLAT_DIM];
    double h[HIDDEN];
    double z[VOCAB_SIZE];

    /* ①+② 임베딩 lookup + flatten */
    for (int d = 0; d < EMB_DIM; d++) flat[0 * EMB_DIM + d] = m->E[ctx_a][d];
    for (int d = 0; d < EMB_DIM; d++) flat[1 * EMB_DIM + d] = m->E[ctx_b][d];

    /* ③ h = tanh(flat @ W1 + b1) */
    for (int j = 0; j < HIDDEN; j++) {
        double s = m->b1[j];
        for (int i = 0; i < FLAT_DIM; i++) s += flat[i] * m->W1[i][j];
        h[j] = tanh(s);
    }

    /* ④ z = h @ W2 + b2 */
    for (int v = 0; v < VOCAB_SIZE; v++) {
        double s = m->b2[v];
        for (int j = 0; j < HIDDEN; j++) s += h[j] * m->W2[j][v];
        z[v] = s;
    }

    /* ⑤ softmax */
    double mx = z[0];
    for (int v = 1; v < VOCAB_SIZE; v++) if (z[v] > mx) mx = z[v];
    double sum = 0.0;
    for (int v = 0; v < VOCAB_SIZE; v++) { out[v] = exp(z[v] - mx); sum += out[v]; }
    for (int v = 0; v < VOCAB_SIZE; v++) out[v] /= sum;
}

int nlm_sample(const nlm_t *m, int ctx_a, int ctx_b, rng_t *r) {
    double p[VOCAB_SIZE];
    nlm_predict(m, ctx_a, ctx_b, p);
    return rng_choice(r, p, VOCAB_SIZE);
}

double nlm_perplexity(const nlm_t *m, const int (*X)[CONTEXT_LEN], const int *y, int n) {
    double sum_log = 0.0;
    double p[VOCAB_SIZE];
    for (int i = 0; i < n; i++) {
        nlm_predict(m, X[i][0], X[i][1], p);
        double pw = p[y[i]];
        if (pw < 1e-30) pw = 1e-30;
        sum_log += log(pw);
    }
    return exp(-sum_log / (double)n);
}

int nlm_generate(const nlm_t *m, rng_t *r, int *out, int max_len) {
    int ctx_a = ID_PAD, ctx_b = ID_BOS;
    int len = 0;
    for (int i = 0; i < max_len; i++) {
        int w = nlm_sample(m, ctx_a, ctx_b, r);
        if (w == ID_EOS) break;
        out[len++] = w;
        ctx_a = ctx_b;
        ctx_b = w;
    }
    return len;
}
