# articles_comparison.py
from __future__ import annotations

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Callable, Any

from sklearn.metrics import (
    silhouette_samples, 
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score
)
from scipy.spatial.distance import cdist

# --- Константы ---
ARTICLES_CSV_PATH = "articles_scores/articles_scores.csv"

# Мета-колонки, которые НЕ являются признаками для кластеризации
METADATA_COLS = {
    "Article_id", "Authors", "Title", "Journal", 
    "Volume", "Issue", "school", "Year" 
}

# Цветовая палитра для графиков (в стиле оранжево-желтой гаммы)
SILHOUETTE_COLORS = ["#FF8C42", "#FFD166", "#F77F00", "#FCBF49", "#EF476F", "#06D6A0"]

ARTICLES_HELP_TEXT = """
Силуэтные графики визуализируют, насколько хорошо разделены кластеры статей разных научных школ.

**Коэффициент силуэта (Silhouette Score):**
- Значение **ближе к +1**: Статьи этой школы имеют четкий, уникальный тематический профиль, который почти не пересекается с другой школой.
- Значение **около 0**: Тематика статей школ сильно перемешана, нет явных различий в публикационной активности.
- **Отрицательное значение**: Статья по своему содержанию (кодам классификатора) оказалась ближе к «чужой» школе, чем к своей собственной.

**Другие метрики:**
- **Индекс Дэвиса–Боулдина**: Чем меньше значение, тем лучше разделены школы.
- **Индекс Калинского–Харабаза**: Чем выше значение, тем более плотные и разреженные кластеры.
"""

# --- Функции нормализации имен ---

def to_short_name(full_name: str) -> str:
    """Преобразует 'Иванов Иван Иванович' -> 'Иванов И.И.'"""
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

def normalize_authors_set(authors_str: str) -> Set[str]:
    """Разбирает строку 'Иванов И.И.; Петров П.П.' в множество имен."""
    if not isinstance(authors_str, str):
        return set()
    raw_names = re.split(r'[;]', authors_str)
    res = set()
    for n in raw_names:
        clean = n.strip()
        if clean:
            clean = re.sub(r'\s+', ' ', clean)
            res.add(clean.lower())
    return res

# --- Работа с данными ---

def load_articles_data(path: str = ARTICLES_CSV_PATH) -> pd.DataFrame:
    """Загружает базу статей."""
    if not Path(path).exists():
        return pd.DataFrame()
    try:
        # Пытаемся прочитать с разделителем ; (Excel-style) или ,
        df = pd.read_csv(path, sep=';', dtype={'Year': str, 'Article_id': str})
        if df.shape[1] < 2:
             df = pd.read_csv(path, sep=',', dtype={'Year': str, 'Article_id': str})
    except Exception:
        return pd.DataFrame()
    return df

def get_feature_columns(df: pd.DataFrame, selected_features: Optional[List[str]] = None) -> List[str]:
    """Определяет колонки классификатора для математического анализа."""
    all_cols = df.columns.tolist()
    potential_features = [c for c in all_cols if c not in METADATA_COLS and c != "school"]
    
    if not selected_features or "Все разделы классификатора" in selected_features:
        # Оставляем только те колонки, которые похожи на коды (цифры и точки)
        return [c for c in potential_features if re.match(r'^[\d\.]+$', c)]
    
    return [f for f in selected_features if f in df.columns]

def prepare_articles_dataset(
    roots: List[str],
    df_lineage: pd.DataFrame,
    idx_lineage: Dict[str, Set[int]],
    lineage_func: Callable,
    df_articles: pd.DataFrame,
    scope: str = "direct",
    selected_features_keys: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Собирает статьи, авторы которых принадлежат к выбранным школам."""
    combined_rows = []
    feature_cols = get_feature_columns(df_articles, selected_features_keys)
    use_year = selected_features_keys and "Year" in selected_features_keys
    
    for root in roots:
        # Получаем состав школы (генеалогию)
        G, _ = lineage_func(df_lineage, idx_lineage, root)
        if not G.has_node(root):
            continue

        if scope == "direct":
            school_members = set(G.successors(root)) # Только 1 уровень
        else:
            school_members = set(G.nodes()) - {root} # Все уровни

        if not school_members:
            continue
            
        school_members_short = {to_short_name(n).lower() for n in school_members if n}
        
        # Фильтруем статьи, где есть совпадение хотя бы по одному автору
        mask = df_articles["Authors"].apply(
            lambda x: not normalize_authors_set(x).isdisjoint(school_members_short)
        )
        school_articles = df_articles[mask].copy()
        
        if not school_articles.empty:
            school_articles["school"] = root
            combined_rows.append(school_articles)

    if not combined_rows:
        return pd.DataFrame(), []

    result_df = pd.concat(combined_rows, ignore_index=True)
    final_features = list(feature_cols)
    
    if use_year:
        result_df["Year_num"] = pd.to_numeric(result_df["Year"], errors='coerce').fillna(0)
        final_features.append("Year_num")

    for col in final_features:
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)
    
    return result_df, final_features

# --- Математические расчеты ---

def calculate_article_metrics(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """Вычисляет метрики разделения (Silhouette, DB, CH, Centroids)."""
    if df.empty or not feature_cols:
        return {}
    
    X = df[feature_cols].values
    labels = df["school"].values
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2 or X.shape[0] < 2:
        return {
            "silhouette_avg": 0, 
            "sample_silhouette_values": np.zeros(X.shape[0]), 
            "davies_bouldin": None, 
            "calinski_harabasz": None, 
            "centroids_dist": None,
            "cluster_names": list(unique_labels)
        }

    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    db_score = davies_bouldin_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)

    centroids = [X[labels == lab].mean(axis=0) for lab in unique_labels]
    dist_matrix = cdist(centroids, centroids, metric='euclidean')
    
    dist_info = dist_matrix[0, 1] if len(unique_labels) == 2 else dist_matrix

    return {
        "silhouette_avg": silhouette_avg,
        "sample_silhouette_values": sample_silhouette_values,
        "davies_bouldin": db_score,
        "calinski_harabasz": ch_score,
        "centroids_dist": dist_info,
        "cluster_names": list(unique_labels)
    }

# --- Визуализация ---

def create_articles_silhouette_plot(
    sample_scores: np.ndarray,
    labels: np.ndarray,
    school_order: List[str],
    overall_score: float,
    metric_name: str = "Euclidean"
) -> plt.Figure:
    """Отрисовывает силуэтный график для статей."""
    n_schools = len(school_order)
    fig, ax = plt.subplots(figsize=(10, max(6, n_schools * 1.5)))
    y_lower = 10
    
    colors = SILHOUETTE_COLORS

    for idx, school in enumerate(school_order):
        mask = labels == idx
        cluster_scores = sample_scores[mask]
        if cluster_scores.size == 0: continue
        cluster_scores = np.sort(cluster_scores)
        size = cluster_scores.size
        y_upper = y_lower + size
        color = colors[idx % len(colors)]
        
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_scores, 
                         facecolor=color, edgecolor=color, alpha=0.8)
        
        ax.text(-0.05, y_lower + size / 2, f"{school} (n={size})", 
                va="center", ha="right", fontsize=10, fontweight='bold')
        y_lower = y_upper + 10

    ax.axvline(x=overall_score, color="red", linestyle="--", linewidth=2,
                label=f"Средний силуэт: {overall_score:.3f}")
    
    ax.set_xlim([-1, 1])
    ax.set_xlabel("Коэффициент силуэта")
    ax.set_title(f"Анализ публикаций научных школ\n(Метрика: {metric_name})", fontsize=14)
    ax.set_yticks([])
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    
    fig.tight_layout()
    return fig
