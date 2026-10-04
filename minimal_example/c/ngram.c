#include "ngram.h"
#include <math.h>
#include <string.h>

void ngram_init(ngram_t *m, double smoothing_k) {
    memset(m->counts, 0, sizeof(m->counts));
    memset(m->totals, 0, sizeof(m->totals));
    m->smoothing_k = smoothing_k;
}

void ngram_fit(ngram_t *m, const int (*X)[CONTEXT_LEN], const int *y, int n) {
    for (int i = 0; i < n; i++) {
        int ci = ctx_idx(X[i][0], X[i][1]);
        m->counts[ci][y[i]] += 1.0;
        m->totals[ci] += 1.0;
    }
}

void ngram_prob(const ngram_t *m, int ctx_a, int ctx_b, double *out) {
    int ci = ctx_idx(ctx_a, ctx_b);
    double k = m->smoothing_k;
    if (m->totals[ci] == 0.0) {
        /* unseen ctx → uniform */
        for (int i = 0; i < VOCAB_SIZE; i++) out[i] = 1.0 / (double)VOCAB_SIZE;
        return;
    }
    double total = m->totals[ci] + k * (double)VOCAB_SIZE;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        out[i] = (m->counts[ci][i] + k) / total;
    }
}

int ngram_sample(const ngram_t *m, int ctx_a, int ctx_b, rng_t *r) {
    double p[VOCAB_SIZE];
    ngram_prob(m, ctx_a, ctx_b, p);
    return rng_choice(r, p, VOCAB_SIZE);
}

double ngram_perplexity(const ngram_t *m, const int (*X)[CONTEXT_LEN], const int *y, int n) {
    double sum_log = 0.0;
    double p[VOCAB_SIZE];
    for (int i = 0; i < n; i++) {
        ngram_prob(m, X[i][0], X[i][1], p);
        double pw = p[y[i]];
        if (pw < 1e-30) pw = 1e-30;
        sum_log += log(pw);
    }
    return exp(-sum_log / (double)n);
}

int ngram_generate(const ngram_t *m, rng_t *r, int *out, int max_len) {
    int ctx_a = ID_PAD, ctx_b = ID_BOS;
    int len = 0;
    for (int i = 0; i < max_len; i++) {
        int w = ngram_sample(m, ctx_a, ctx_b, r);
        if (w == ID_EOS) break;
        out[len++] = w;
        ctx_a = ctx_b;
        ctx_b = w;
    }
    return len;
}
