# articles_comparison.py
"""
Модуль сравнения научных школ по публикациям (статьям).
Реализует логику, аналогичную school_comparison, но адаптированную под структуру статей.

ОБНОВЛЕНИЯ:
- Загрузка классификатора статей из articles_classifier.json
- Поддержка всех уровней иерархии классификатора
"""

from __future__ import annotations

import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Callable, Any, Literal

from sklearn.metrics import (
    silhouette_samples,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.spatial.distance import cdist

# ==============================================================================
# КОНСТАНТЫ И ТИПЫ
# ==============================================================================

POSSIBLE_PATHS = [
    "articles_scores.csv",
    "articles_scores/articles_scores.csv",
    "db_articles/articles_scores.csv"
]

METADATA_COLS = {
    "Article_id", "Authors", "Title", "Journal",
    "Volume", "Issue", "school", "Year", "Year_num"
}

DistanceMetric = Literal[
    "euclidean_orthogonal",
    "cosine_orthogonal",
    "euclidean_oblique",
    "cosine_oblique"
]

DISTANCE_METRIC_LABELS: Dict[DistanceMetric, str] = {
    "euclidean_orthogonal": "Евклидово (прямоугольный базис)",
    "cosine_orthogonal": "Косинусное (прямоугольный базис)",
    "euclidean_oblique": "Евклидово (косоугольный базис)",
    "cosine_oblique": "Косинусное (косоугольный базис)",
}

SILHOUETTE_COLORS = ["#FF8C42", "#FFD166", "#F77F00", "#FCBF49", "#EF476F", "#06D6A0", "#118AB2", "#073B4C"]

# Пути к файлу классификатора ДЛЯ СТАТЕЙ
CLASSIFIER_PATHS = [
    "articles_classifier.json",
    "db_articles/articles_classifier.json",
]

ARTICLES_HELP_TEXT = """
### 🔬 Анализ публикационной активности научных школ

Этот инструмент позволяет сравнить, насколько различаются тематические профили статей,
написанных представителями разных научных школ.

**Основные возможности:**

1. **Выбор охвата**: Можно анализировать только прямых учеников или всю школу целиком.

2. **Гибкий базис**: 
   - **Весь базис** — анализ по всем тематическим кодам классификатора
   - **Отдельные узлы** — выбор конкретных разделов (при выборе узла включаются все его подузлы)
   - **Год** — добавление временного фактора

3. **Метрики**:
   - *Прямоугольный базис*: Стандартное сравнение, где все темы равноправны.
   - *Косоугольный базис*: Учитывает иерархию тем для более глубокого анализа.

**Интерпретация метрик:**

- **Коэффициент силуэта**: Показывает степень разделения кластеров. Чем выше, тем уникальнее профиль школы.
- **Индекс Дэвиса–Боулдина**: Оценивает плотность кластеров (меньше — лучше разделение).
- **Индекс Калинского–Харабаза**: Оценивает дисперсию (больше — лучше сформированы кластеры).
"""

CLASSIFIER_LIST_TEXT = """
Классификатор загружается из `articles_classifier.json`.
"""

# ==============================================================================
# ЗАГРУЗКА КЛАССИФИКАТОРА
# ==============================================================================

def load_articles_classifier() -> Dict[str, str]:
    """Загружает классификатор статей из JSON файла."""
    for path_str in CLASSIFIER_PATHS:
        path = Path(path_str)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Ошибка загрузки классификатора из {path}: {e}")
                continue
    print("⚠️ Файл articles_classifier.json не найден, классификатор будет пустым")
    return {}

# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ИЕРАРХИЯ)
# ==============================================================================

def get_code_depth(code: str) -> int:
    """Возвращает глубину (уровень) кода в иерархии (1.1 -> 2, 1.1.1 -> 3)."""
    if not code or code == "Year":
        return 0
    return code.count(".") + 1

def get_parent_code(code: str) -> Optional[str]:
    """Возвращает родительский код."""
    if "." not in code:
        return None
    return code.rsplit(".", 1)[0]

def get_ancestor_codes(code: str) -> List[str]:
    """Возвращает список всех предков кода."""
    ancestors = []
    current = code
    while current:
        ancestors.insert(0, current)
        current = get_parent_code(current)
    return ancestors

def is_descendant_of(code: str, ancestor: str) -> bool:
    """Проверяет, является ли code потомком ancestor."""
    if code == ancestor:
        return True
    return code.startswith(ancestor + ".")

# ==============================================================================
# МАТЕМАТИКА (КОСОУГОЛЬНЫЙ БАЗИС И РАССТОЯНИЯ)
# ==============================================================================

def build_oblique_transform_matrix(feature_columns: List[str], decay_factor: float = 0.5) -> np.ndarray:
    """Строит матрицу трансформации для учета иерархии."""
    n = len(feature_columns)
    col_to_idx = {col: i for i, col in enumerate(feature_columns)}
    transform = np.eye(n)

    for i, col in enumerate(feature_columns):
        if col == "Year_num":
            continue
        ancestors = get_ancestor_codes(col)
        for depth, ancestor in enumerate(ancestors[:-1]):
            if ancestor in col_to_idx:
                j = col_to_idx[ancestor]
                distance = len(ancestors) - depth - 1
                weight = decay_factor ** distance
                transform[i, j] = weight

    return transform

def apply_oblique_transform(data: np.ndarray, feature_columns: List[str], decay_factor: float = 0.5) -> np.ndarray:
    transform = build_oblique_transform_matrix(feature_columns, decay_factor)
    return data @ transform.T

def compute_distance_matrix(data: np.ndarray, feature_columns: List[str], metric: DistanceMetric, decay_factor: float = 0.5) -> np.ndarray:
    """Вычисляет матрицу расстояний согласно выбранной метрике."""
    if metric in ("euclidean_oblique", "cosine_oblique"):
        data_transformed = apply_oblique_transform(data, feature_columns, decay_factor)
    else:
        data_transformed = data

    if "euclidean" in metric:
        return euclidean_distances(data_transformed)
    else:
        return cosine_distances(data_transformed)

# ==============================================================================
# РАБОТА С ДАННЫМИ СТАТЕЙ
# ==============================================================================

def to_short_name(full_name: str) -> str:
    """'Иванов Иван Иванович' -> 'Иванов И.И.'"""
    parts = full_name.strip().replace('.', ' ').split()
    if not parts:
        return ""
    surname = parts[0]
    initials = ""
    if len(parts) > 1:
        initials += parts[1][0] + "."
    if len(parts) > 2:
        initials += parts[2][0] + "."
    return f"{surname} {initials}"

def canonicalize_author_name(name: str) -> str:
    """Нормализует имя автора к единому виду: 'иванов и.и.'"""
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    if not s:
        return ""

    s = s.replace("ё", "е")
    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()

    compact = re.sub(r"\s+", "", s)

    m = re.search(r"[a-zа-я]\.", compact)
    if m:
        pos = m.start()
        surname = compact[:pos]
        initials_raw = compact[pos:]
    else:
        letters = re.findall(r"[a-zа-я]", compact)
        if len(letters) < 2:
            return ""
        surname = compact[:-2]
        initials_raw = compact[-2:]

    surname = re.sub(r"[^a-zа-я\-]", "", surname)
    if not surname:
        return ""

    init_letters = re.findall(r"[a-zа-я]", initials_raw)
    if not init_letters:
        return ""

    initials = "".join(ch + "." for ch in init_letters[:3])
    return f"{surname} {initials}"

def normalize_authors_set(authors_str: str) -> Set[str]:
    """'Иванов И.И.; Петров П.П.' -> {'иванов и.и.', 'петров п.п.'}"""
    if not isinstance(authors_str, str):
        return set()
    raw_names = re.split(r"[;]", authors_str)
    res: Set[str] = set()
    for n in raw_names:
        canon = canonicalize_author_name(n)
        if canon:
            res.add(canon)
    return res

def load_articles_data() -> pd.DataFrame:
    """Загружает CSV со статьями."""
    for path_str in POSSIBLE_PATHS:
        path = Path(path_str)
        if path.exists():
            try:
                df = pd.read_csv(path, sep=';', dtype={'Year': str, 'Article_id': str})
                if df.shape[1] < 2:
                    df = pd.read_csv(path, sep=',', dtype={'Year': str, 'Article_id': str})
                return df
            except Exception as e:
                print(f"Ошибка чтения {path}: {e}")
                continue
    return pd.DataFrame()

# ==============================================================================
# АНАЛИЗ (ВЫЧИСЛЕНИЯ)
# ==============================================================================

def compute_article_analysis(
    df: pd.DataFrame,
    feature_columns: List[str],
    metric: DistanceMetric,
    decay_factor: float = 0.5
) -> Dict[str, Any]:
    """
    Полный цикл анализа: вычисление матрицы расстояний и метрик.
    """
    if df.empty or not feature_columns:
        return {}

    X = df[feature_columns].values
    labels = df["school"].astype(str).values
    unique_labels = np.unique(labels)
    school_order = list(unique_labels)

    if len(unique_labels) < 2 or X.shape[0] < 2:
        return {
            "silhouette_avg": 0.0,
            "sample_silhouette_values": np.zeros(X.shape[0]),
            "labels": labels,
            "school_order": school_order,
            "unique_schools": school_order,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "centroids_dist": None
        }

    dist_matrix = compute_distance_matrix(X, feature_columns, metric, decay_factor)

    try:
        silhouette_avg = silhouette_score(dist_matrix, labels, metric="precomputed")
        sample_silhouette_values = silhouette_samples(dist_matrix, labels, metric="precomputed")
    except Exception:
        silhouette_avg = 0.0
        sample_silhouette_values = np.zeros(X.shape[0])

    if "oblique" in metric:
        X_for_metrics = apply_oblique_transform(X, feature_columns, decay_factor)
    else:
        X_for_metrics = X

    try:
        db_score = davies_bouldin_score(X_for_metrics, labels)
    except Exception:
        db_score = None

    try:
        ch_score = calinski_harabasz_score(X_for_metrics, labels)
    except Exception:
        ch_score = None

    try:
        centroids = [X_for_metrics[labels == lab].mean(axis=0) for lab in unique_labels]
        centroid_dist_matrix = cdist(centroids, centroids, metric="euclidean")
        dist_info = centroid_dist_matrix[0, 1] if len(unique_labels) == 2 else centroid_dist_matrix
    except Exception:
        dist_info = None

    return {
        "silhouette_avg": float(silhouette_avg),
        "sample_silhouette_values": sample_silhouette_values,
        "labels": labels,
        "school_order": school_order,
        "unique_schools": school_order,
        "davies_bouldin": db_score,
        "calinski_harabasz": ch_score,
        "centroids_dist": dist_info
    }

def create_comparison_summary(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Создает таблицу со статистикой по школам."""
    summary_data: List[Dict[str, Any]] = []
    unique_schools = df["school"].unique()
    thematic_cols = [c for c in feature_cols if c != "Year_num"]
    has_year = "Year_num" in feature_cols and "Year_num" in df.columns

    for school in unique_schools:
        sub = df[df["school"] == school]
        num_data = sub[thematic_cols] if thematic_cols else pd.DataFrame(index=sub.index)

        row: Dict[str, Any] = {
            "Научная школа": school,
            "Количество статей": int(len(sub)),
        }

        if len(thematic_cols) > 0:
            profile_sum = num_data.sum(axis=1)
            row.update({
                "Средняя сумма профиля": float(profile_sum.mean()),
                "Стд. отклонение": float(profile_sum.std(ddof=1)) if len(profile_sum) > 1 else 0.0,
                "Охват тем (среднее)": float((num_data > 0).sum(axis=1).mean()),
            })
        else:
            row.update({
                "Средняя сумма профиля": 0.0,
                "Стд. отклонение": 0.0,
                "Охват тем (среднее)": 0.0,
            })

        if has_year:
            years = sub["Year_num"].dropna()
            if len(years) > 0:
                row["Средний год"] = float(years.mean())
                row["Диапазон годов"] = f"{int(years.min())}–{int(years.max())}"
            else:
                row["Средний год"] = np.nan
                row["Диапазон годов"] = ""

        summary_data.append(row)

    return pd.DataFrame(summary_data)

def create_articles_silhouette_plot(
    sample_scores: np.ndarray,
    labels: np.ndarray,
    school_order: List[str],
    overall_score: float,
    metric_label: str
) -> plt.Figure:
    """Отрисовывает силуэтный график."""
    n_schools = len(school_order)
    fig, ax = plt.subplots(figsize=(10, max(6, n_schools * 1.5)))
    y_lower = 10

    colors = SILHOUETTE_COLORS[:n_schools] if n_schools <= len(SILHOUETTE_COLORS) else \
        (SILHOUETTE_COLORS * ((n_schools // len(SILHOUETTE_COLORS)) + 1))[:n_schools]

    label_to_idx = {name: i for i, name in enumerate(school_order)}
    numeric_labels = np.array([label_to_idx[l] for l in labels])

    for idx, school in enumerate(school_order):
        mask = numeric_labels == idx
        cluster_scores = sample_scores[mask]
        if cluster_scores.size == 0:
            continue
        cluster_scores = np.sort(cluster_scores)
        size = cluster_scores.size
        y_upper = y_lower + size
        color = colors[idx]

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_scores,
                        facecolor=color, edgecolor=color, alpha=0.85)
        ax.text(-0.05, y_lower + size / 2, f"{school} (n={size})",
               va="center", ha="right", fontsize=10, fontweight='bold')
        y_lower = y_upper + 10

    ax.axvline(x=overall_score, color="#2D3436", linestyle="--", linewidth=2,
              label=f"Средний силуэт: {overall_score:.3f}")
    ax.set_xlim([-1, 1])
    ax.set_xlabel("Коэффициент силуэта")
    ax.set_title(f"Анализ публикаций научных школ\n{metric_label}", fontsize=14)
    ax.set_yticks([])
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    ax.axvspan(-1, -0.25, alpha=0.05, color="red")
    ax.axvspan(0.5, 1, alpha=0.05, color="green")

    fig.tight_layout()
    return fig
