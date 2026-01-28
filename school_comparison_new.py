"""
Модуль для сравнения научных руководителей через кластерный анализ.
Поддерживает ортогональный и косоугольный базис.

Косоугольный базис реализован по формуле:
    v_c = α·v_p + β·u_c
где:
    α = 1/N для N≥2 (N = количество siblings)
    α = decay_factor для N=1 (предотвращение коллапса)
    β = sqrt(1 - α²)
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

# ============================================================================
# ТИПЫ И КОНСТАНТЫ
# ============================================================================

DistanceMetric = Literal[
    "euclidean_orthogonal",
    "cosine_orthogonal",
    "euclidean_oblique",
    "cosine_oblique"
]

ComparisonScope = Literal["direct", "all"]

DISTANCE_METRIC_LABELS: Dict[DistanceMetric, str] = {
    "euclidean_orthogonal": "Евклидово расстояние (ортогональный базис)",
    "cosine_orthogonal": "Косинусное расстояние (ортогональный базис)",
    "euclidean_oblique": "Евклидово расстояние (косоугольный базис)",
    "cosine_oblique": "Косинусное расстояние (косоугольный базис)",
}

SCOPE_LABELS: Dict[ComparisonScope, str] = {
    "direct": "Только прямые ученики",
    "all": "Вся линия (включая потомков)",
}

SILHOUETTE_COLORS = [
    "#FF8C42", "#FFD166", "#F77F00", "#FCBF49", "#EF476F",
    "#06D6A0", "#118AB2", "#073B4C", "#E07A5F", "#81B29A",
]

# ============================================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С ИЕРАРХИЕЙ
# ============================================================================

def get_code_depth(code: str) -> int:
    """Возвращает глубину узла в иерархии."""
    if not code:
        return 0
    return code.count('.') + 1


def get_parent_code(code: str) -> Optional[str]:
    """Возвращает код родителя. '1.2.3' -> '1.2'."""
    if '.' not in code:
        return None
    return code.rsplit('.', 1)[0]


def get_ancestor_codes(code: str) -> List[str]:
    """Возвращает список всех предков (включая сам код)."""
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
    return code.startswith(ancestor + '.')


def filter_columns_by_nodes(
    columns: List[str],
    selected_nodes: Optional[List[str]] = None
) -> List[str]:
    """Фильтрует колонки по выбранным узлам иерархии."""
    if selected_nodes is None or len(selected_nodes) == 0:
        return columns
    filtered = []
    for col in columns:
        for node in selected_nodes:
            if is_descendant_of(col, node):
                filtered.append(col)
                break
    return filtered


def get_nodes_at_level(columns: List[str], level: int) -> List[str]:
    """Возвращает все узлы заданного уровня."""
    return sorted(set(col for col in columns if get_code_depth(col) == level))


def get_selectable_nodes(columns: List[str], max_level: int = 3) -> List[str]:
    """Возвращает все узлы от уровня 1 до max_level."""
    result = []
    for level in range(1, max_level + 1):
        result.extend(get_nodes_at_level(columns, level))
    return sorted(result)


def get_sibling_count(code: str, all_codes: List[str]) -> int:
    """
    Подсчитывает количество siblings (узлов с тем же родителем).
    Включает сам узел code.
    """
    parent = get_parent_code(code)
    if parent is None:
        # Корневой уровень: считаем все узлы без точек
        return sum(1 for c in all_codes if '.' not in c)
    # Считаем детей того же родителя
    return sum(1 for c in all_codes if get_parent_code(c) == parent)


# ============================================================================
# КОСОУГОЛЬНЫЙ БАЗИС (ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ)
# ============================================================================

def build_oblique_basis_matrix(
    feature_columns: List[str],
    decay_factor: float = 0.6
) -> np.ndarray:
    """
    Строит матрицу косоугольного базиса B.
    
    Строки B — это базисные векторы v_i в ортогональных координатах.
    
    Формула для каждого узла c с родителем p:
        v_c = α·v_p + β·u_c
    где:
        - N = количество siblings (включая сам узел)
        - α = decay_factor, если N=1 (предотвращает коллапс v_c=v_p)
        - α = 1/N, если N≥2 (строгая формула из метода)
        - β = sqrt(1 - α²)
        - u_c — ортогональная компонента (текущая строка B[i])
    
    Args:
        feature_columns: список кодов признаков в нужном порядке
        decay_factor: коэффициент затухания для N=1 (рекомендуется 0.5-0.7)
        
    Returns:
        Матрица B размера (n_features, n_features)
    """
    n = len(feature_columns)
    code_to_idx = {c: i for i, c in enumerate(feature_columns)}
    
    # Инициализация: каждая строка i — это единичный вектор u_i
    B = np.eye(n, dtype=np.float64)
    
    # Определяем порядок обработки: по возрастанию глубины
    # (родители должны обрабатываться раньше детей)
    processing_order = sorted(range(n), key=lambda i: feature_columns[i].count('.'))
    
    # Счётчики для диагностики
    n_strict = 0   # Узлы с N≥2 (строгая формула)
    n_decay = 0    # Узлы с N=1 (используется decay_factor)
    n_root = 0     # Узлы без родителя в наборе
    
    for i in processing_order:
        code = feature_columns[i]
        parent_code = get_parent_code(code)
        
        # Если родителя нет в наборе, узел остаётся ортогональным
        if parent_code is None or parent_code not in code_to_idx:
            n_root += 1
            continue
        
        parent_idx = code_to_idx[parent_code]
        N = get_sibling_count(code, feature_columns)
        
        # Выбор α в зависимости от N
        if N == 1:
            alpha = decay_factor
            n_decay += 1
        else:
            alpha = 1.0 / N
            n_strict += 1
        
        # Ограничение для численной устойчивости
        alpha = min(alpha, 0.9999)
        
        # Вычисление β
        beta = np.sqrt(1.0 - alpha * alpha)
        
        # Обновление базисного вектора:
        # v_c = α·v_p + β·u_c
        # B[parent_idx, :] — это уже преобразованный v_p
        # B[i, :] — это текущий u_c (будет обновлён)
        B[i, :] = alpha * B[parent_idx, :] + beta * B[i, :]
    
    # Диагностический вывод
    print(f"  [Косоугольный базис] Построена матрица {n}×{n}")
    print(f"    • Узлы с N≥2 (α=1/N): {n_strict}")
    print(f"    • Узлы с N=1 (α={decay_factor:.2f}): {n_decay}")
    print(f"    • Корневые узлы: {n_root}")
    
    # Проверка числа обусловленности
    if n > 1:
        cond = np.linalg.cond(B)
        if cond > 100:
            print(f"    ⚠️  Число обусловленности: {cond:.1f} (возможна коллинеарность)")
    
    return B


def apply_oblique_basis_transform(
    X: np.ndarray,
    feature_columns: List[str],
    decay_factor: float = 0.6
) -> np.ndarray:
    """
    Применяет косоугольное базисное преобразование к данным.
    
    Преобразование: V = X @ B
    где B — матрица косоугольного базиса.
    
    Args:
        X: матрица данных (n_samples, n_features)
        feature_columns: список названий признаков (порядок важен!)
        decay_factor: коэффициент затухания для N=1
        
    Returns:
        Преобразованная матрица V (n_samples, n_features)
    """
    B = build_oblique_basis_matrix(feature_columns, decay_factor)
    # ПРАВИЛЬНОЕ умножение (без транспонирования)
    return X @ B


# ============================================================================
# ВЫЧИСЛЕНИЕ МАТРИЦЫ РАССТОЯНИЙ
# ============================================================================

def compute_distance_matrix(
    data: np.ndarray,
    feature_columns: List[str],
    metric: DistanceMetric,
    decay_factor: float = 0.5
) -> np.ndarray:
    """
    Вычисляет матрицу попарных расстояний.
    
    Args:
        data: матрица признаков (n_samples, n_features)
        feature_columns: список названий признаков
        metric: метрика расстояния
        decay_factor: коэффициент для косоугольного базиса
        
    Returns:
        Матрица расстояний (n_samples, n_samples)
    """
    # Применяем косоугольное преобразование, если нужно
    if metric in ["euclidean_oblique", "cosine_oblique"]:
        data = apply_oblique_basis_transform(data, feature_columns, decay_factor)
    
    # Вычисляем расстояния
    if metric in ["euclidean_orthogonal", "euclidean_oblique"]:
        return euclidean_distances(data)
    else:  # cosine
        return cosine_distances(data)


# ============================================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================================

def load_scores_from_folder(
    folder_path: str = "basic_scores",
    specific_files: Optional[List[str]] = None
) -> pd.DataFrame:
    """Загружает CSV-файлы с оценками из папки."""
    base = Path(folder_path).expanduser().resolve()
    
    if specific_files:
        files = [base / f for f in specific_files if (base / f).exists()]
    else:
        files = sorted(base.glob("*.csv"))
    
    if not files:
        raise FileNotFoundError(f"CSV-файлы не найдены в {base}")
    
    frames: List[pd.DataFrame] = []
    for file in files:
        try:
            frame = pd.read_csv(file)
            if 'Code' not in frame.columns:
                raise KeyError(f"{file.name}: отсутствует колонка 'Code'")
            frames.append(frame)
        except Exception as e:
            print(f"Ошибка при чтении {file}: {e}")
            continue
    
    if not frames:
        raise ValueError("Не удалось загрузить ни одного файла")
    
    scores = pd.concat(frames, ignore_index=True)
    scores = scores.dropna(subset=['Code'])
    scores['Code'] = scores['Code'].astype(str).str.strip()
    scores = scores[scores['Code'].str.len() > 0]
    scores = scores.drop_duplicates(subset='Code', keep='first')
    
    feature_columns = [c for c in scores.columns if c != 'Code']
    scores[feature_columns] = scores[feature_columns].apply(pd.to_numeric, errors='coerce')
    scores[feature_columns] = scores[feature_columns].fillna(0.0)
    
    return scores


def get_feature_columns(scores: pd.DataFrame) -> List[str]:
    """Извлекает список колонок-признаков."""
    return [c for c in scores.columns if c != 'Code']


# ============================================================================
# СБОР ДАННЫХ ДЛЯ НАУЧНОГО РУКОВОДИТЕЛЯ
# ============================================================================

def gather_school_dataset(
    df: pd.DataFrame,
    index: Dict[str, Set[int]],
    root: str,
    scores: pd.DataFrame,
    scope: ComparisonScope,
    lineage_func: Callable,
    rows_for_func: Callable,
    author_column: str = "candidate.name"
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Собирает датасет для одного научного руководителя."""
    if scope == "direct":
        subset = rows_for_func(df, index, root)
    elif scope == "all":
        subset = lineage_func(df, index, root)
    else:
        raise ValueError(f"Неизвестный scope: {scope}")
    
    if subset is None or subset.empty:
        empty = pd.DataFrame(columns=['Code', 'school', author_column])
        return empty, empty, 0
    
    if 'Code' not in subset.columns:
        raise KeyError("Code")
    
    cols_to_keep = ['Code']
    if author_column in subset.columns:
        cols_to_keep.append(author_column)
    
    working = subset[cols_to_keep].copy()
    working['Code'] = working['Code'].astype(str).str.strip()
    working = working[working['Code'].str.len() > 0]
    working = working.drop_duplicates(subset='Code')
    
    if working.empty:
        empty = pd.DataFrame(columns=['Code', 'school', author_column])
        return empty, empty, 0
    
    codes = working['Code'].tolist()
    total_count = len(codes)
    
    scores_copy = scores.copy()
    scores_copy['Code'] = scores_copy['Code'].astype(str).str.strip()
    matched_scores = scores_copy[scores_copy['Code'].isin(codes)].copy()
    
    if matched_scores.empty:
        missing_info = working.copy()
        missing_info['school'] = root
        empty = pd.DataFrame(columns=list(scores.columns) + ['school', author_column])
        return empty, missing_info, total_count
    
    matched_scores['school'] = root
    
    if author_column in working.columns:
        matched_scores = matched_scores.merge(
            working[['Code', author_column]],
            on='Code',
            how='left'
        )
    else:
        matched_scores[author_column] = None
    
    found_codes = set(matched_scores['Code'].tolist())
    missing_codes = [c for c in codes if c not in found_codes]
    
    if missing_codes:
        missing_info = working[working['Code'].isin(missing_codes)].copy()
        missing_info['school'] = root
    else:
        missing_info = pd.DataFrame(columns=['Code', 'school', author_column])
    
    return matched_scores, missing_info, total_count


# ============================================================================
# СИЛУЭТНЫЙ АНАЛИЗ
# ============================================================================

def compute_silhouette_analysis(
    datasets: Dict[str, pd.DataFrame],
    feature_columns: List[str],
    metric: DistanceMetric,
    selected_nodes: Optional[List[str]] = None,
    decay_factor: float = 0.5
) -> Tuple[float, np.ndarray, np.ndarray, List[str], List[str]]:
    """Выполняет силуэтный анализ для сравнения научных школ."""
    used_columns = filter_columns_by_nodes(feature_columns, selected_nodes)
    if not used_columns:
        raise ValueError("Не осталось признаков после фильтрации")
    
    all_data = []
    all_labels = []
    school_order = []
    
    for school_name, dataset in datasets.items():
        if dataset.empty:
            continue
        
        available_cols = [c for c in used_columns if c in dataset.columns]
        if not available_cols:
            continue
        
        school_data = dataset[available_cols].fillna(0.0).values
        if school_data.shape[0] == 0:
            continue
        
        all_data.append(school_data)
        all_labels.extend([len(school_order)] * school_data.shape[0])
        school_order.append(school_name)
    
    if len(school_order) < 2:
        raise ValueError("Недостаточно школ (минимум 2)")
    
    X = np.vstack(all_data)
    labels = np.array(all_labels)
    
    if X.shape[0] < 2:
        raise ValueError("Недостаточно образцов")
    
    distance_matrix = compute_distance_matrix(X, used_columns, metric, decay_factor)
    
    try:
        overall_score = silhouette_score(distance_matrix, labels, metric='precomputed')
        sample_scores = silhouette_samples(distance_matrix, labels, metric='precomputed')
    except Exception as e:
        raise ValueError(f"Ошибка при вычислении силуэта: {e}")
    
    return overall_score, sample_scores, labels, school_order, used_columns


# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================

def create_silhouette_plot(
    sample_scores: np.ndarray,
    labels: np.ndarray,
    school_order: List[str],
    overall_score: float,
    metric_label: str
) -> plt.Figure:
    """Создаёт силуэтный график."""
    n_schools = len(school_order)
    fig, ax = plt.subplots(figsize=(10, max(6, n_schools * 1.5)))
    
    y_lower = 10
    
    if n_schools <= len(SILHOUETTE_COLORS):
        colors = SILHOUETTE_COLORS[:n_schools]
    else:
        colors = SILHOUETTE_COLORS * ((n_schools // len(SILHOUETTE_COLORS)) + 1)
        colors = colors[:n_schools]
    
    for idx, school in enumerate(school_order):
        mask = labels == idx
        cluster_scores = sample_scores[mask]
        
        if cluster_scores.size == 0:
            continue
        
        cluster_scores = np.sort(cluster_scores)
        size = cluster_scores.size
        y_upper = y_lower + size
        
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_scores,
            facecolor=colors[idx],
            edgecolor=colors[idx],
            alpha=0.85
        )
        
        ax.text(
            -0.05,
            y_lower + size / 2,
            f"{school}\n(n={size})",
            fontsize=10,
            va='center',
            ha='right',
            fontweight='medium'
        )
        
        y_lower = y_upper + 10
    
    ax.axvline(
        x=overall_score,
        color='#2D3436',
        linestyle='--',
        linewidth=2,
        label=f'Среднее = {overall_score:.3f}'
    )
    
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Силуэтный коэффициент", fontsize=12)
    ax.set_ylabel("Школы", fontsize=12)
    ax.set_title(f"Силуэтный анализ ({metric_label})", fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    ax.axvspan(-1, -0.25, alpha=0.08, color='#e74c3c')
    ax.axvspan(-0.25, 0.25, alpha=0.08, color='#f39c12')
    ax.axvspan(0.25, 0.5, alpha=0.08, color='#27ae60')
    ax.axvspan(0.5, 1, alpha=0.08, color='#16a085')
    
    fig.tight_layout()
    return fig


def create_comparison_summary(
    datasets: Dict[str, pd.DataFrame],
    feature_columns: List[str],
    school_order: List[str]
) -> pd.DataFrame:
    """Создаёт сводную таблицу."""
    summary_data = []
    
    for school in school_order:
        if school not in datasets:
            continue
        data = datasets[school]
        if data.empty:
            continue
        
        available_cols = [c for c in feature_columns if c in data.columns]
        numeric_data = data[available_cols].fillna(0.0)
        
        summary_data.append({
            'Школа': school,
            'Количество': len(data),
            'Средняя сумма': numeric_data.sum(axis=1).mean(),
            'Медиана суммы': numeric_data.sum(axis=1).median(),
            'Стд. откл.': numeric_data.sum(axis=1).std(),
            'Среднее кол-во нулей': (numeric_data == 0).sum(axis=1).mean()
        })
    
    return pd.DataFrame(summary_data)


def interpret_silhouette_score(score: float) -> str:
    """Интерпретирует силуэтный коэффициент."""
    if score >= 0.71:
        return "🟢 Сильное и чёткое разделение"
    elif score >= 0.51:
        return "🟡 Умеренное разделение"
    elif score >= 0.26:
        return "🟠 Слабое разделение"
    elif score >= 0:
        return "🔴 Очень слабое разделение"
    else:
        return "⛔ Некорректная кластеризация"

