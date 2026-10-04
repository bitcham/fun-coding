"""실험 공용 유틸. 경로 수정과 결과 디렉토리 생성."""

import os
import sys
from collections import Counter

# Python 모듈 경로 추가
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY_DIR = os.path.join(ROOT, "python")
if PY_DIR not in sys.path:
    sys.path.insert(0, PY_DIR)

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def count_sentences(samples):
    """문장 리스트(list[list[str]]) → Counter (key는 join된 문자열)."""
    return Counter(" ".join(s) for s in samples)


def write_table(rows, path, header=None):
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(c) for c in r) + "\n")


def fmt_pct(x):
    return f"{x*100:.2f}%"
