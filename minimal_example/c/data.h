#ifndef DATA_H
#define DATA_H

/*
 * 학습 데이터: 6 문장에서 추출한 trigram 학습 쌍.
 * vocab은 <pad>=0, <bos>=1, <eos>=2, 나머지는 등장 순서.
 */

#define VOCAB_SIZE 11
#define CONTEXT_LEN 2
#define N_PAIRS 24
#define N_SENT 6

/* 모델 하이퍼파라미터: 컴파일 시 상수로 고정 (build.bat에서 -DHIDDEN=6 으로 override 가능) */
#define EMB_DIM 2
#ifndef HIDDEN
#define HIDDEN 8
#endif

#define ID_PAD 0
#define ID_BOS 1
#define ID_EOS 2
#define ID_CHEOLSU 3       /* 철수는 */
#define ID_APPLE 4         /* 사과를 */
#define ID_LIKE 5          /* 좋아해 */
#define ID_STRAW 6         /* 딸기를 */
#define ID_YEONGHEE 7      /* 영희는 */
#define ID_BANANA 8        /* 바나나를 */
#define ID_JJANGGU 9       /* 짱구는 */
#define ID_DISLIKE 10      /* 싫어해 */

extern const char *VOCAB[VOCAB_SIZE];
extern const int X_TRAIN[N_PAIRS][CONTEXT_LEN];
extern const int Y_TRAIN[N_PAIRS];

/* 학습용 6 문장 (각 토큰 시퀀스, EOS까지) */
extern const int *SENT_TOKENS[N_SENT];
extern const int SENT_LENS[N_SENT];

#endif
