#include "data.h"
#include "ngram.h"
#include "nlm.h"
#include "rng.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* 한 번에 끝내는 데모: n-gram 학습/샘플/평가 + NLM 학습/샘플/평가 +
 * (철수는, 바나나를) → 좋아해 검증. 결과는 stdout과 파일에 모두 출력. */

static FILE *g_log = NULL;

static void logf2(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    if (g_log) {
        va_start(ap, fmt);
        vfprintf(g_log, fmt, ap);
        va_end(ap);
    }
}

static void print_sentence(const int *toks, int n) {
    for (int i = 0; i < n; i++) {
        if (i) logf2(" ");
        logf2("%s", VOCAB[toks[i]]);
    }
    logf2("\n");
}

/* EMB_DIM, HIDDEN은 data.h에서 정의 */
#define N_NGRAM_SAMPLES 10000
#define N_NLM_SAMPLES   10000
#define LR              0.3
#define EPOCHS          2000

static int sentence_in_train(const int *toks, int n) {
    /* 학습 6 문장과 비교 */
    for (int s = 0; s < N_SENT; s++) {
        if (SENT_LENS[s] != n) continue;
        int match = 1;
        for (int i = 0; i < n; i++) {
            if (SENT_TOKENS[s][i] != toks[i]) { match = 0; break; }
        }
        if (match) return 1;
    }
    return 0;
}

int main(void) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    /* HIDDEN 값을 파일명에 포함 (H=6과 H=8 빌드가 서로 덮어쓰지 않도록) */
#define _STR(x) #x
#define _XSTR(x) _STR(x)
#define LOG_SUFFIX "_h" _XSTR(HIDDEN)

    /* build/에서 실행되든 c/에서 실행되든 results/에 쌓이도록 여러 경로 시도 */
    g_log = fopen("../../results/c_run" LOG_SUFFIX ".log", "w");
    if (!g_log) g_log = fopen("../results/c_run" LOG_SUFFIX ".log", "w");
    if (!g_log) g_log = fopen("results/c_run" LOG_SUFFIX ".log", "w");
    if (!g_log) g_log = fopen("c_run" LOG_SUFFIX ".log", "w");

    FILE *loss_log_f = NULL;
    loss_log_f = fopen("../../results/c_nlm_loss" LOG_SUFFIX ".txt", "w");
    if (!loss_log_f) loss_log_f = fopen("../results/c_nlm_loss" LOG_SUFFIX ".txt", "w");
    if (!loss_log_f) loss_log_f = fopen("results/c_nlm_loss" LOG_SUFFIX ".txt", "w");
    if (!loss_log_f) loss_log_f = fopen("c_nlm_loss" LOG_SUFFIX ".txt", "w");

    logf2("============================================================\n");
    logf2(" minimal_example C 데모 (V=%d, pairs=%d)\n", VOCAB_SIZE, N_PAIRS);
    logf2("============================================================\n\n");

    /* ---------------- n-gram ---------------- */
    logf2("[1] n-gram (trigram, MLE) 학습\n");
    ngram_t ng;
    ngram_init(&ng, 0.0);
    ngram_fit(&ng, X_TRAIN, Y_TRAIN, N_PAIRS);
    double ng_train_ppl = ngram_perplexity(&ng, X_TRAIN, Y_TRAIN, N_PAIRS);
    logf2("  train perplexity = %.6f\n", ng_train_ppl);

    rng_t rng;
    rng_seed(&rng, 42);
    /* 샘플 5문장 */
    logf2("  샘플 5문장:\n");
    for (int i = 0; i < 5; i++) {
        int out[20];
        int n = ngram_generate(&ng, &rng, out, 20);
        logf2("    ");
        print_sentence(out, n);
    }

    /* 10,000 문장 통계 */
    logf2("\n  %d문장 생성 통계:\n", N_NGRAM_SAMPLES);
    int in_train = 0, novel = 0;
    int counts_per_sent[N_SENT] = {0};
    rng_seed(&rng, 7);
    for (int i = 0; i < N_NGRAM_SAMPLES; i++) {
        int out[20];
        int n = ngram_generate(&ng, &rng, out, 20);
        if (sentence_in_train(out, n)) {
            in_train++;
            for (int s = 0; s < N_SENT; s++) {
                if (SENT_LENS[s] == n) {
                    int match = 1;
                    for (int k = 0; k < n; k++) if (SENT_TOKENS[s][k] != out[k]) { match = 0; break; }
                    if (match) { counts_per_sent[s]++; break; }
                }
            }
        } else novel++;
    }
    logf2("    in train: %d (%.2f%%)  novel: %d (%.2f%%)\n",
          in_train, 100.0 * in_train / N_NGRAM_SAMPLES,
          novel, 100.0 * novel / N_NGRAM_SAMPLES);
    for (int s = 0; s < N_SENT; s++) {
        logf2("    %5d  ", counts_per_sent[s]);
        print_sentence(SENT_TOKENS[s], SENT_LENS[s]);
    }

    /* ---------------- NLM ---------------- */
    logf2("\n[2] NLM 학습 (D=%d, H=%d, lr=%.2f, epochs=%d)\n", EMB_DIM, HIDDEN, LR, EPOCHS);
    rng_seed(&rng, 123);
    nlm_t nlm;
    nlm_init(&nlm, &rng);

    nlm_grad_t grad;
    double first_loss = 0.0;
    for (int ep = 0; ep < EPOCHS; ep++) {
        double loss = nlm_loss_and_grad(&nlm, &grad, X_TRAIN, Y_TRAIN);
        nlm_sgd_step(&nlm, &grad, LR);
        if (ep == 0) first_loss = loss;
        if (loss_log_f) fprintf(loss_log_f, "%.10f\n", loss);
        if (ep % 200 == 0 || ep == EPOCHS - 1) {
            logf2("    epoch %4d  loss = %.6f\n", ep, loss);
        }
    }
    if (loss_log_f) fclose(loss_log_f);
    double nlm_train_ppl = nlm_perplexity(&nlm, X_TRAIN, Y_TRAIN, N_PAIRS);
    logf2("  train perplexity = %.6f  (시작 loss %.4f)\n", nlm_train_ppl, first_loss);

    /* 샘플 5문장 */
    logf2("\n  샘플 5문장:\n");
    rng_seed(&rng, 1);
    for (int i = 0; i < 5; i++) {
        int out[20];
        int n = nlm_generate(&nlm, &rng, out, 20);
        logf2("    ");
        print_sentence(out, n);
    }

    /* 10,000 문장 통계 */
    logf2("\n  %d문장 생성 통계:\n", N_NLM_SAMPLES);
    int nlm_in = 0, nlm_novel = 0;
    rng_seed(&rng, 7);
    for (int i = 0; i < N_NLM_SAMPLES; i++) {
        int out[20];
        int n = nlm_generate(&nlm, &rng, out, 20);
        if (sentence_in_train(out, n)) nlm_in++;
        else nlm_novel++;
    }
    logf2("    in train: %d (%.2f%%)  novel: %d (%.2f%%)\n",
          nlm_in, 100.0 * nlm_in / N_NLM_SAMPLES,
          nlm_novel, 100.0 * nlm_novel / N_NLM_SAMPLES);

    /* ---------------- 실험 8: target check ---------------- */
    logf2("\n[3] (철수는, 바나나를) → 좋아해 검증\n");
    double p_ng[VOCAB_SIZE], p_nlm[VOCAB_SIZE];
    ngram_prob(&ng, ID_CHEOLSU, ID_BANANA, p_ng);
    nlm_predict(&nlm, ID_CHEOLSU, ID_BANANA, p_nlm);
    logf2("  P(좋아해 | 철수는, 바나나를)\n");
    logf2("    n-gram  : %.6f  (uniform fallback)\n", p_ng[ID_LIKE]);
    logf2("    NLM     : %.6f\n", p_nlm[ID_LIKE]);
    logf2("  Single-position perplexity:\n");
    logf2("    n-gram  : %.4f\n", 1.0 / (p_ng[ID_LIKE] > 1e-30 ? p_ng[ID_LIKE] : 1e-30));
    logf2("    NLM     : %.4f\n", 1.0 / (p_nlm[ID_LIKE] > 1e-30 ? p_nlm[ID_LIKE] : 1e-30));

    /* 샘플링 카운트 */
    int hits_ng = 0, hits_nlm = 0;
    rng_seed(&rng, 7);
    for (int i = 0; i < 10000; i++) if (rng_choice(&rng, p_ng, VOCAB_SIZE) == ID_LIKE) hits_ng++;
    rng_seed(&rng, 7);
    for (int i = 0; i < 10000; i++) if (rng_choice(&rng, p_nlm, VOCAB_SIZE) == ID_LIKE) hits_nlm++;
    logf2("  10,000 샘플 중 '좋아해':  n-gram %d (%.2f%%)  NLM %d (%.2f%%)\n",
          hits_ng, hits_ng / 100.0, hits_nlm, hits_nlm / 100.0);

    /* NLM top-5 */
    logf2("\n  NLM top-5 next-token:\n");
    int idx[VOCAB_SIZE];
    for (int i = 0; i < VOCAB_SIZE; i++) idx[i] = i;
    /* 간단 정렬 (insertion sort, prob 내림차순) */
    for (int i = 1; i < VOCAB_SIZE; i++) {
        int v = idx[i];
        int j = i - 1;
        while (j >= 0 && p_nlm[idx[j]] < p_nlm[v]) { idx[j + 1] = idx[j]; j--; }
        idx[j + 1] = v;
    }
    for (int i = 0; i < 5; i++) {
        logf2("    %s : %.6f\n", VOCAB[idx[i]], p_nlm[idx[i]]);
    }

    if (g_log) fclose(g_log);

    logf2("\n완료. 로그: results/c_run" LOG_SUFFIX ".log\n");
    return 0;
}
