"""
Модуль расчета меры общности/специфичности тематических профилей.

Реализует классическую формулу Шеннона и модифицированную с иерархическим
коэффициентом Z для учета структуры тематического классификатора.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def get_code_depth(code: str) -> int:
    """Возвращает глубину (уровень) кода в иерархии."""
    if not code:
        return 0
    return code.count(".") + 1


def get_parent_code(code: str) -> Optional[str]:
    """Возвращает родительский код или None для корневых."""
    if "." not in code:
        return None
    return code.rsplit(".", 1)[0]


def get_ancestor_codes(code: str) -> List[str]:
    """Возвращает список всех предков кода (от корня к текущему, исключая сам код)."""
    ancestors = []
    current = get_parent_code(code)
    while current:
        ancestors.insert(0, current)
        current = get_parent_code(current)
    return ancestors


def count_children(code: str, all_codes: List[str]) -> int:
    """
    Подсчитывает количество прямых потомков данного узла.

    Args:
        code: Код узла
        all_codes: Список всех кодов в классификаторе

    Returns:
        Количество прямых дочерних узлов
    """
    prefix = code + "."
    depth = get_code_depth(code)
    children = [
        c for c in all_codes 
        if c.startswith(prefix) and get_code_depth(c) == depth + 1
    ]
    return len(children)


def calculate_z_coefficient(
    code: str,
    all_codes: List[str]
) -> float:
    """
    Вычисляет иерархический коэффициент Z для заданного узла.

    Z_i = произведение по ancestors(i) от 1/log(k_d),
    где k_d - количество дочерних ветвей у предка d.

    Args:
        code: Код узла классификатора
        all_codes: Список всех кодов в классификаторе

    Returns:
        Значение коэффициента Z
    """
    ancestors = get_ancestor_codes(code)

    if not ancestors:
        # Узел первого уровня - нет предков
        return 1.0

    z = 1.0
    for ancestor in ancestors:
        k_d = count_children(ancestor, all_codes)
        if k_d > 1:
            # log с основанием 2 для информационной меры
            z *= 1.0 / np.log2(k_d)
        # Если k_d <= 1, то 1/log(k_d) не определено или 1, пропускаем

    return z


def calculate_entropy_shannon(
    profile: pd.Series,
    min_threshold: float = 0.0
) -> float:
    """
    Вычисляет классическую энтропию Шеннона.

    H = -сумма p_i * log(p_i)

    Args:
        profile: Тематический профиль (баллы по кодам классификатора)
        min_threshold: Минимальный порог - значения ниже приводятся к нулю

    Returns:
        Значение энтропии
    """
    # Применяем порог
    values = profile.copy()
    values[values < min_threshold] = 0.0

    # Убираем нулевые значения
    values = values[values > 0]

    if len(values) == 0:
        return 0.0

    # Нормализуем на сумму (получаем вероятности)
    total = values.sum()
    if total == 0:
        return 0.0

    probabilities = values / total

    # Вычисляем энтропию
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def calculate_entropy_hierarchical(
    profile: pd.Series,
    all_codes: List[str],
    min_threshold: float = 0.0
) -> float:
    """
    Вычисляет модифицированную энтропию с учетом иерархического коэффициента Z.

    H = -сумма Z_i * p_i * log(p_i)

    Args:
        profile: Тематический профиль (баллы по кодам классификатора)
        all_codes: Список всех кодов в классификаторе
        min_threshold: Минимальный порог - значения ниже приводятся к нулю

    Returns:
        Значение модифицированной энтропии
    """
    # Применяем порог
    values = profile.copy()
    values[values < min_threshold] = 0.0

    # Убираем нулевые значения
    values = values[values > 0]

    if len(values) == 0:
        return 0.0

    # Нормализуем на сумму (получаем вероятности)
    total = values.sum()
    if total == 0:
        return 0.0

    probabilities = values / total

    # Вычисляем коэффициенты Z для каждого кода
    z_coefficients = {}
    for code in probabilities.index:
        z_coefficients[code] = calculate_z_coefficient(code, all_codes)

    # Вычисляем модифицированную энтропию
    entropy = 0.0
    for code, p_i in probabilities.items():
        z_i = z_coefficients.get(code, 1.0)
        entropy -= z_i * p_i * np.log2(p_i)

    return entropy


def search_by_entropy(
    scores_df: pd.DataFrame,
    feature_columns: List[str],
    use_hierarchical: bool = False,
    min_threshold: float = 3.0,
    ascending: bool = True
) -> pd.DataFrame:
    """
    Ищет диссертации по мере общности/специфичности (энтропия).

    Args:
        scores_df: DataFrame с тематическими профилями
        feature_columns: Список колонок с признаками (коды классификатора)
        use_hierarchical: Использовать ли иерархический коэффициент Z
        min_threshold: Минимальный порог для отсечения малых значений
        ascending: True - от низкой энтропии к высокой (узкие → широкие темы),
                   False - наоборот

    Returns:
        DataFrame с результатами, отсортированный по энтропии
    """
    if scores_df.empty:
        return scores_df

    # Проверяем наличие колонок
    available_cols = [c for c in feature_columns if c in scores_df.columns]
    if not available_cols:
        raise ValueError("Нет доступных колонок для анализа")

    # Копируем данные
    result_df = scores_df.copy()

    # Вычисляем энтропию для каждой строки
    entropies = []
    for idx, row in result_df.iterrows():
        profile = row[available_cols]

        if use_hierarchical:
            entropy = calculate_entropy_hierarchical(
                profile, 
                available_cols, 
                min_threshold
            )
        else:
            entropy = calculate_entropy_shannon(
                profile, 
                min_threshold
            )

        entropies.append(entropy)

    # Добавляем энтропию в результат
    result_df["entropy"] = entropies

    # Подсчитываем количество ненулевых признаков (после порога)
    def count_nonzero_after_threshold(row):
        vals = row[available_cols]
        return (vals >= min_threshold).sum()

    result_df["features_count"] = result_df.apply(
        count_nonzero_after_threshold, 
        axis=1
    )

    # Сортируем по энтропии
    result_df = result_df.sort_values("entropy", ascending=ascending)

    return result_df


def interpret_entropy(
    entropy: float,
    use_hierarchical: bool = False
) -> str:
    """
    Возвращает текстовую интерпретацию значения энтропии.

    Args:
        entropy: Значение энтропии
        use_hierarchical: Была ли использована модификация с коэффициентом Z

    Returns:
        Текстовое описание
    """
    if entropy < 1.0:
        return "🔹 Очень узкая специализация — исследование сфокусировано на конкретной теме"
    elif entropy < 2.5:
        return "🔸 Узкая специализация — тема достаточно конкретна"
    elif entropy < 4.0:
        return "🟡 Умеренная широта — исследование охватывает несколько смежных тем"
    elif entropy < 5.5:
        return "🟠 Широкий охват — исследование затрагивает множество тем"
    else:
        return "🔴 Очень широкий охват — междисциплинарное исследование с высокой степенью обобщения"
