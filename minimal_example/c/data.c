#include "data.h"

/* UTF-8 한국어 토큰. /utf-8 컴파일 옵션과 함께 사용 */
const char *VOCAB[VOCAB_SIZE] = {
    "<pad>", "<bos>", "<eos>",
    "철수는", "사과를", "좋아해", "딸기를",
    "영희는", "바나나를", "짱구는", "싫어해",
};

/*
 * 6 문장:
 *   0: 철수는 사과를 좋아해
 *   1: 철수는 딸기를 좋아해
 *   2: 영희는 사과를 좋아해
 *   3: 영희는 딸기를 좋아해
 *   4: 영희는 바나나를 좋아해
 *   5: 짱구는 바나나를 싫어해
 *
 * 학습 쌍은 [pad, bos, w1, w2, w3, eos]에서 슬라이딩 윈도우.
 * 따라서 각 문장당 4쌍, 총 24쌍.
 */

const int X_TRAIN[N_PAIRS][CONTEXT_LEN] = {
    /* 철수는 사과를 좋아해 */
    {ID_PAD, ID_BOS}, {ID_BOS, ID_CHEOLSU}, {ID_CHEOLSU, ID_APPLE}, {ID_APPLE, ID_LIKE},
    /* 철수는 딸기를 좋아해 */
    {ID_PAD, ID_BOS}, {ID_BOS, ID_CHEOLSU}, {ID_CHEOLSU, ID_STRAW}, {ID_STRAW, ID_LIKE},
    /* 영희는 사과를 좋아해 */
    {ID_PAD, ID_BOS}, {ID_BOS, ID_YEONGHEE}, {ID_YEONGHEE, ID_APPLE}, {ID_APPLE, ID_LIKE},
    /* 영희는 딸기를 좋아해 */
    {ID_PAD, ID_BOS}, {ID_BOS, ID_YEONGHEE}, {ID_YEONGHEE, ID_STRAW}, {ID_STRAW, ID_LIKE},
    /* 영희는 바나나를 좋아해 */
    {ID_PAD, ID_BOS}, {ID_BOS, ID_YEONGHEE}, {ID_YEONGHEE, ID_BANANA}, {ID_BANANA, ID_LIKE},
    /* 짱구는 바나나를 싫어해 */
    {ID_PAD, ID_BOS}, {ID_BOS, ID_JJANGGU}, {ID_JJANGGU, ID_BANANA}, {ID_BANANA, ID_DISLIKE},
};

const int Y_TRAIN[N_PAIRS] = {
    ID_CHEOLSU, ID_APPLE, ID_LIKE, ID_EOS,
    ID_CHEOLSU, ID_STRAW, ID_LIKE, ID_EOS,
    ID_YEONGHEE, ID_APPLE, ID_LIKE, ID_EOS,
    ID_YEONGHEE, ID_STRAW, ID_LIKE, ID_EOS,
    ID_YEONGHEE, ID_BANANA, ID_LIKE, ID_EOS,
    ID_JJANGGU, ID_BANANA, ID_DISLIKE, ID_EOS,
};

/* 각 문장 (sliding window 생성용) */
static const int SENT0[] = {ID_CHEOLSU, ID_APPLE, ID_LIKE};
static const int SENT1[] = {ID_CHEOLSU, ID_STRAW, ID_LIKE};
static const int SENT2[] = {ID_YEONGHEE, ID_APPLE, ID_LIKE};
static const int SENT3[] = {ID_YEONGHEE, ID_STRAW, ID_LIKE};
static const int SENT4[] = {ID_YEONGHEE, ID_BANANA, ID_LIKE};
static const int SENT5[] = {ID_JJANGGU, ID_BANANA, ID_DISLIKE};

const int *SENT_TOKENS[N_SENT] = {SENT0, SENT1, SENT2, SENT3, SENT4, SENT5};
const int SENT_LENS[N_SENT] = {3, 3, 3, 3, 3, 3};
