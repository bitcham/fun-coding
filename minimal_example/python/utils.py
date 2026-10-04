"""공용 유틸: 한글 출력, 그래프 폰트, 시드 등."""

import os
import sys
import random

import numpy as np


def setup_io():
    """Windows 콘솔에서 UTF-8 출력 강제."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)


def setup_matplotlib_korean():
    """한글 폰트로 matplotlib 설정. 시스템에 깔린 폰트를 차례로 시도."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = [
        "Malgun Gothic",
        "NanumGothic",
        "Nanum Gothic",
        "AppleGothic",
        "Gulim",
        "Dotum",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
