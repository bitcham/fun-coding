#ifndef NLM_H
#define NLM_H

#include "data.h"
#include "rng.h"

/*
 * 뉴럴 언어 모델: word embedding → tanh hidden → softmax output.
 *
 * 모든 사이즈는 data.h의 #define으로 고정되어 있어, 행렬을 다차원 배열로
 * 그대로 들고 다닌다. matmul 같은 일반 헬퍼는 쓰지 않고, 각 식을 그 자리에서
 * 명시적인 for 루프로 작성한다 — Python의 numpy 식과 줄 단위로 비교하며 읽기 좋다.
 */

#define FLAT_DIM (CONTEXT_LEN * EMB_DIM)   /* 임베딩을 펼친 길이 */

typedef struct {
    double E[VOCAB_SIZE][EMB_DIM];
    double W1[FLAT_DIM][HIDDEN];
    double b1[HIDDEN];
    double W2[HIDDEN][VOCAB_SIZE];
    double b2[VOCAB_SIZE];
} nlm_t;

/* 같은 모양의 그래디언트 — 별도 struct로 분리해서 명시적으로 다룸 */
typedef nlm_t nlm_grad_t;

void nlm_init(nlm_t *m, rng_t *r);

/* forward + backward 모두 수행. loss(스칼라) 반환, g에 그래디언트 채움.
 * 입력은 항상 학습셋 전체 (full-batch SGD): X[N_PAIRS][CONTEXT_LEN], y[N_PAIRS]. */
double nlm_loss_and_grad(const nlm_t *m, nlm_grad_t *g,
                         const int X[N_PAIRS][CONTEXT_LEN],
                         const int y[N_PAIRS]);

void nlm_sgd_step(nlm_t *m, const nlm_grad_t *g, double lr);

/* 추론 (batch=1). out[VOCAB_SIZE]에 확률 분포 채움. */
void nlm_predict(const nlm_t *m, int ctx_a, int ctx_b, double out[VOCAB_SIZE]);

int    nlm_sample(const nlm_t *m, int ctx_a, int ctx_b, rng_t *r);
double nlm_perplexity(const nlm_t *m, const int (*X)[CONTEXT_LEN], const int *y, int n);
int    nlm_generate(const nlm_t *m, rng_t *r, int *out, int max_len);

#endif
