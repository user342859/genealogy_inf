# articles_comparison.py
from __future__ import annotations

import re
import pandas as pd
import numpy as np
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
    # Year может быть признаком, если выбран явно, но по умолчанию в "профиль" не входит
}

def to_short_name(full_name: str) -> str:
    """
    Преобразует 'Фамилия Имя Отчество' -> 'Фамилия И.О.'
    Для сопоставления с форматом в статьях.
    """
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
    """
    Разбирает строку авторов из CSV ('Иванов И.И.; Петров П.П.')
    в множество нормализованных строк для поиска.
    """
    if not isinstance(authors_str, str):
        return set()
    # Разделяем по точке с запятой или запятой (на всякий случай)
    raw_names = re.split(r'[;]', authors_str)
    res = set()
    for n in raw_names:
        clean = n.strip()
        # Приводим к виду "Фамилия И.О." (убираем лишние пробелы)
        if clean:
            # Эвристика: убираем лишние пробелы внутри инициалов, если есть
            clean = re.sub(r'\s+', ' ', clean)
            res.add(clean.lower()) # для регистронезависимого сравнения
    return res

def load_articles_data(path: str = ARTICLES_CSV_PATH) -> pd.DataFrame:
    """Загружает базу статей."""
    if not Path(path).exists():
        return pd.DataFrame()
    
    # Читаем CSV. Разделитель определяем автоматически или жестко ';'
    try:
        df = pd.read_csv(path, sep=';', dtype={'Year': str, 'Article_id': str})
        if df.shape[1] < 2: # Если разделитель не сработал
             df = pd.read_csv(path, sep=',', dtype={'Year': str, 'Article_id': str})
    except Exception:
        return pd.DataFrame()
    
    return df

def get_feature_columns(df: pd.DataFrame, selected_features: Optional[List[str]] = None) -> List[str]:
    """
    Определяет, какие колонки использовать для математического сравнения.
    """
    all_cols = df.columns.tolist()
    # Исключаем метаданные и служебные колонки
    potential_features = [c for c in all_cols if c not in METADATA_COLS and c != "school"]
    
    if not selected_features or "Все разделы" in selected_features:
        # Возвращаем все колонки классификатора
        # Фильтруем, чтобы это были действительно коды (цифры и точки)
        return [c for c in potential_features if re.match(r'^[\d\.]+$', c)]
    
    # Если выбраны конкретные фичи
    chosen = []
    
    # Обработка "Year" как особого случая
    if "Year" in selected_features:
        # Если Year выбран, он должен быть в df и быть числовым.
        # Обычно Year в метаданных, мы его временно добавим в features list
        # Но сама логика обработки данных должна убедиться, что Year числовой.
        pass # Year обрабатывается отдельно при подготовке матрицы
        
    for f in selected_features:
        if f in df.columns:
            chosen.append(f)
            
    return chosen

def prepare_articles_dataset(
    roots: List[str],
    df_lineage: pd.DataFrame,
    idx_lineage: Dict[str, Set[int]],
    lineage_func: Callable, # lineage(df, index, root) -> G, subset
    df_articles: pd.DataFrame,
    scope: str = "direct", # "direct" or "all"
    selected_features_keys: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Формирует итоговый датафрейм для сравнения школ.
    Возвращает: (DataFrame с данными и колонкой 'school', Список использованных колонок-фич)
    """
    combined_rows = []
    
    # 1. Определяем список колонок-признаков (без Year пока)
    feature_cols = get_feature_columns(df_articles, selected_features_keys)
    use_year = selected_features_keys and "Year" in selected_features_keys
    
    for root in roots:
        # 1. Получаем состав школы
        if scope == "direct":
            # Только 1-й уровень (прямые потомки)
            # Используем lineage, но берем только соседей графа или rows_for + фильтр
            # Проще взять rows_for, но это даст только записи диссертаций.
            # Нам нужны ИМЕНА учеников.
            # Воспользуемся lineage_func, чтобы получить subset, а там отфильтруем по уровню?
            # Или просто возьмем 1 уровень.
            # В lineage_func (из diss.lineages) нет параметра глубины для subset, 
            # но subset содержит всех потомков.
            # Для "direct" нам нужны те, у кого supervisor == root.
            
            # Вариант: используем rows_for для поиска строк, где root - научрук.
            # Из этих строк берем candidate_name.
            from streamlit_app import rows_for # Предполагаем импорт или передачу функции
            # В данном контексте лучше использовать переданные функции или логику
            # Но для унификации используем lineage_func и фильтруем граф или subset?
            
            # Быстрый способ для 1 уровня:
            subset = pd.DataFrame() # Заглушка, если не найдем
            
            # Чтобы не дублировать логику, вызовем lineage и возьмем только 1 шаг
            G, subset_all = lineage_func(df_lineage, idx_lineage, root)
            if G.number_of_nodes() < 2:
                school_members_full = set()
            else:
                # Берем только прямых наследников из графа
                direct_children = list(G.successors(root))
                school_members_full = set(direct_children)
                # Добавляем самого рута? Обычно школы сравнивают по ученикам. 
                # Если рут пишет статьи, включать ли его? 
                # В school_comparison рут не включается, только диссертации учеников.
                # Оставим только учеников.
                
        else: # "all"
            G, subset_all = lineage_func(df_lineage, idx_lineage, root)
            # Все узлы графа кроме корня
            school_members_full = set(G.nodes()) - {root}

        if not school_members_full:
            continue
            
        # 2. Преобразуем имена школы в краткий формат для поиска
        # school_members_short = {"ivanov i.i.", "petrov p.p."}
        school_members_short = {to_short_name(n).lower() for n in school_members_full if n}
        
        # 3. Ищем статьи этой школы
        # Проходим по всем статьям и проверяем пересечение авторов
        # Это может быть медленно, если статей тысячи. Оптимизация через apply.
        
        def is_school_article(row_authors_str):
            auths = normalize_authors_set(row_authors_str)
            # Есть ли пересечение
            return not auths.isdisjoint(school_members_short)

        mask = df_articles["Authors"].apply(is_school_article)
        school_articles = df_articles[mask].copy()
        
        if school_articles.empty:
            continue
            
        # 4. Добавляем метку школы
        school_articles["school"] = root
        combined_rows.append(school_articles)

    if not combined_rows:
        return pd.DataFrame(), []

    result_df = pd.concat(combined_rows, ignore_index=True)
    
    # 5. Подготовка числовых данных (features)
    final_features = list(feature_cols)
    
    # Если нужен Year, добавляем его
    if use_year:
        # Преобразуем в число
        result_df["Year_num"] = pd.to_numeric(result_df["Year"], errors='coerce').fillna(0)
        # Нормализация года важна, так как он ~2020, а оценки ~0-10.
        # MinMax scaler или просто деление.
        # Для простоты пока оставим как есть или сделаем базовую нормализацию (x - 2000)
        # Но лучше предупредить пользователя, что год ломает дистанции.
        final_features.append("Year_num")

    # Преобразуем колонки в numeric
    for col in final_features:
        if col not in result_df.columns:
            result_df[col] = 0
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

    # Удаляем строки, где все фичи 0 (пустышки)
    # result_df = result_df[result_df[final_features].sum(axis=1) > 0]
    
    return result_df, final_features


def calculate_article_metrics(
    df: pd.DataFrame, 
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Считает метрики кластеризации и сравнения школ.
    """
    if df.empty or len(feature_cols) == 0:
        return {}
    
    X = df[feature_cols].values
    labels = df["school"].values
    unique_labels = np.unique(labels)
    
    # Нужно минимум 2 кластера и минимум 2 сэмпла для большинства метрик
    if len(unique_labels) < 2 or X.shape[0] < 2:
        return {
            "silhouette_avg": 0,
            "sample_silhouette_values": np.zeros(X.shape[0]),
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "centroids_dist": None
        }

    # 1. Silhouette
    try:
        silhouette_avg = silhouette_score(X, labels)
        sample_silhouette_values = silhouette_samples(X, labels)
    except Exception:
        silhouette_avg = 0
        sample_silhouette_values = np.zeros(X.shape[0])

    # 2. Davies-Bouldin Index (чем меньше, тем лучше разделение)
    try:
        db_score = davies_bouldin_score(X, labels)
    except Exception:
        db_score = None

    # 3. Calinski-Harabasz Index (чем больше, тем лучше)
    try:
        ch_score = calinski_harabasz_score(X, labels)
    except Exception:
        ch_score = None

    # 4. Расстояние между центроидами (Euclidean)
    # Считаем среднее для каждого кластера
    centroids = []
    cluster_names = []
    for label in unique_labels:
        cluster_data = X[labels == label]
        centroid = cluster_data.mean(axis=0)
        centroids.append(centroid)
        cluster_names.append(label)
    
    centroids = np.array(centroids)
    # Матрица расстояний
    dist_matrix = cdist(centroids, centroids, metric='euclidean')
    
    # Формируем читаемый результат для дистанций
    # Если школ 2, возвращаем одно число. Если больше - матрицу (в словаре).
    if len(unique_labels) == 2:
        dist_info = dist_matrix[0, 1] # Расстояние между двумя
    else:
        dist_info = dist_matrix # Матрица n*n

    return {
        "silhouette_avg": silhouette_avg,
        "sample_silhouette_values": sample_silhouette_values,
        "davies_bouldin": db_score,
        "calinski_harabasz": ch_score,
        "centroids_dist": dist_info,
        "cluster_names": cluster_names # Чтобы знать порядок в матрице
    }
