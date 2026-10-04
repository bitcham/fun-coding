# minimal_example 계획

절차
1. 계획 문서를 작성한다.
2. AI에게 계획 문서를 검토 시킨다.
3. AI에게 구현을 시킨다.
4. 결과를 확인한다

## 개요

- 최소 예제로 엔그램과 뉴럴 언어 모델의 근본적 차이 확인
- 핵심 알고리듬은 Python·C 모두 구현. Python은 numpy만 사용 
- 그래프 등 보조 도구는 matplotlib 등 자유롭게 사용
- 구현 스타일은 현재 구조에 필요한 최소한으로 간결하게 구현. 예를 들면, 일반적인 뉴럴네트워크 구조를 위한 autograd 같은 것을 구현하지 말고 현재 구조에 최적화되어 있는 구현. 미분도 Loss를 계산하는 전체 식에 대해 trainable parameter를 각각 미분하는 식으로 계산하면 훨씬 구조가 간단하게 만들 수 있다.

작업 폴더: `D:\Part2\minimal_example`

## 실험 데이터

토큰은 공백으로 구분된 단어. 6문장 사용:

```
철수는 사과를 좋아해
철수는 딸기를 좋아해
영희는 사과를 좋아해
영희는 딸기를 좋아해
영희는 바나나를 좋아해
짱구는 바나나를 싫어해
```

CONTEXT_LEN = 2 (trigram, 두 단어로 다음 단어 예측). 엔그램·뉴럴 언어 모델 모두 공통.

학습 쌍 생성:

- 각 문장을 `[<pad>] * (CONTEXT_LEN - 1) + [<bos>] + 단어들 + [<eos>]`로 감쌈
- sliding window로 `(w1, w2) → w3` 쌍 추출

학습 쌍 예시:

| 컨텍스트 | 다음 토큰 |
|---|---|
| `<pad> <bos>` | 철수는, 영희는, 짱구는 |
| `<bos> 철수는` | 사과를 |
| `딸기를 좋아해` | `<eos>` |

## 엔그램

- trigram

## 뉴럴 언어 모델

구조: word embedding → hidden layer → output layer → softmax → probability

| 레이어 | 사양 |
|---|---|
| word embedding | 차원 가변 (1/2/3 등). 예: 철수는 → (e1, e2) |
| hidden layer | fully connected, 뉴런 수 가변 |
| output layer | vocab size 고정 |

학습 사양:

| 항목 | 값 |
|---|---|
| Loss | cross-entropy |
| Optimizer | SGD (기본). lr은 적절한 값을 반복 실험으로 탐색 |
| 가중치 초기화 | 랜덤 |
| Random seed | 123 |

학습 중 `loss_history.txt` 출력 → loss 곡선 시각화용

## 샘플링

uniform sampling (모델 출력 분포에서 multinomial 추출)

## C 빌드

VS2022 컴파일러(`cl.exe`) 자동 탐색하여 사용 (Windows)

## 구현 메모

- 변수명은 영어, 주석은 이해하기 쉬운 한국어
- 위에 명시되지 않은 부분은 구현자 재량

## 실험

| # | 내용 |
|---|---|
| 1 | 엔그램으로 10,000문장 생성 + 통계 분석 |
| 2 | 뉴럴 언어 모델로 10,000문장 생성 + 통계 분석 |
| 3 | learning rate 변화에 따른 훈련 속도 비교 |
| 4 | embedding dimension 1 vs 2 비교 |
| 5 | embedding dim, hidden size, learning rate 등 여러 하이퍼파라미터 조합 비교 |
| 6 | 훈련 중 word embedding space 변화 애니메이션 (mp4) |
| 7 | 엔그램과 뉴럴 언어 모델의 perplexity 비교 |
| 8 | `(철수는, 바나나를) → 좋아해` 생성 여부 확인 — 10,000회 샘플링 + perplexity 계산 |
