#ifndef NGRAM_H
#define NGRAM_H

#include "data.h"
#include "rng.h"

/*
 * Trigram (CONTEXT_LEN=2). MLE 카운트 + 선택적 add-k smoothing.
 * 학습 컨텍스트는 최대 VOCAB_SIZE^2개 가능 — 이 작은 데이터셋에선 dense 카운트 표면 OK.
 */

#define NCTX (VOCAB_SIZE * VOCAB_SIZE)

typedef struct {
    /* counts[ctx_idx][w] = 카운트, totals[ctx_idx] = 합 */
    double counts[NCTX][VOCAB_SIZE];
    double totals[NCTX];
    double smoothing_k;
} ngram_t;

static inline int ctx_idx(int a, int b) { return a * VOCAB_SIZE + b; }

void ngram_init(ngram_t *m, double smoothing_k);
void ngram_fit(ngram_t *m, const int (*X)[CONTEXT_LEN], const int *y, int n);
void ngram_prob(const ngram_t *m, int ctx_a, int ctx_b, double *out);
int  ngram_sample(const ngram_t *m, int ctx_a, int ctx_b, rng_t *r);
double ngram_perplexity(const ngram_t *m, const int (*X)[CONTEXT_LEN], const int *y, int n);

/* <bos>로 시작해 <eos>까지 생성. out에는 단어 ID들이 들어감 (eos 제외). */
int ngram_generate(const ngram_t *m, rng_t *r, int *out, int max_len);

#endif
