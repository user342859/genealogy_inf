"""
Модуль поиска диссертаций по конкретным темам тематического классификатора.

Реализует поиск работ, соответствующих выбранным пунктам классификатора
с минимальным порогом оценки.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


# ==============================================================================
# КОНСТАНТЫ
# ==============================================================================

DEFAULT_SCORES_FOLDER = "basic_scores"
DEFAULT_MIN_SCORE = 4.0
SELECTION_LIMIT = 5


# ==============================================================================
# ЗАГРУЗКА ДАННЫХ
# ==============================================================================

def load_basic_scores(folder_path: str = DEFAULT_SCORES_FOLDER) -> pd.DataFrame:
    """
    Загружает тематические профили диссертаций из CSV файлов.

    Args:
        folder_path: Путь к папке с CSV файлами профилей

    Returns:
        DataFrame с профилями (колонка Code + колонки с кодами классификатора)

    Raises:
        FileNotFoundError: Если папка не найдена или нет CSV файлов
        KeyError: Если в файлах отсутствует колонка Code
        ValueError: Если нет колонок с признаками
    """
    base = Path(folder_path).expanduser().resolve()
    files = sorted(base.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"CSV файлы не найдены в {base}")

    frames: List[pd.DataFrame] = []
    for file in files:
        frame = pd.read_csv(file)
        if "Code" not in frame.columns:
            raise KeyError(f"Файл {file.name} не содержит колонку 'Code'")
        frames.append(frame)

    scores = pd.concat(frames, ignore_index=True)

    # Очистка
    scores = scores.dropna(subset=["Code"])
    scores["Code"] = scores["Code"].astype(str).str.strip()
    scores = scores[scores["Code"].str.len() > 0]
    scores = scores.drop_duplicates(subset=["Code"], keep="first")

    # Обработка числовых колонок
    feature_columns = [c for c in scores.columns if c != "Code"]
    if not feature_columns:
        raise ValueError("Нет колонок с признаками в профилях")

    scores[feature_columns] = scores[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    scores[feature_columns] = scores[feature_columns].fillna(0.0)

    return scores


def get_feature_columns(scores_df: pd.DataFrame) -> List[str]:
    """
    Возвращает список колонок с признаками (все кроме Code).

    Args:
        scores_df: DataFrame с профилями

    Returns:
        Список названий колонок с признаками
    """
    return [c for c in scores_df.columns if c != "Code"]


# ==============================================================================
# ПОИСК ПО ТЕМАМ
# ==============================================================================

def search_by_codes(
    scores_df: pd.DataFrame,
    selected_codes: List[str],
    min_score: float = DEFAULT_MIN_SCORE
) -> pd.DataFrame:
    """
    Ищет диссертации, соответствующие всем выбранным кодам классификатора.

    Args:
        scores_df: DataFrame с тематическими профилями
        selected_codes: Список кодов классификатора для поиска
        min_score: Минимальный балл для каждого кода

    Returns:
        DataFrame с результатами, отсортированный по сумме баллов
    """
    if not selected_codes:
        return pd.DataFrame()

    # Проверяем наличие выбранных кодов
    missing_columns = [code for code in selected_codes if code not in scores_df.columns]
    if missing_columns:
        raise ValueError(f"Коды не найдены в профилях: {', '.join(missing_columns)}")

    # Фильтруем по порогу для каждого кода
    working = scores_df[["Code"] + selected_codes].copy()

    for code in selected_codes:
        working = working[working[code] >= min_score]

    if working.empty:
        return working

    # Вычисляем общую сумму баллов
    working["profile_total"] = working[selected_codes].sum(axis=1)

    # Сортируем по убыванию суммы
    sort_columns = ["profile_total"] + selected_codes
    working = working.sort_values(
        by=sort_columns,
        ascending=[False] * len(sort_columns)
    )

    return working


def merge_with_dissertation_info(
    search_results: pd.DataFrame,
    dissertations_df: pd.DataFrame,
    selected_codes: List[str]
) -> pd.DataFrame:
    """
    Объединяет результаты поиска с метаданными диссертаций.

    Args:
        search_results: Результаты поиска (Code + баллы + profile_total)
        dissertations_df: Основной DataFrame с информацией о диссертациях
        selected_codes: Список выбранных кодов (для округления баллов)

    Returns:
        DataFrame с полной информацией о найденных диссертациях
    """
    # Определяем доступные информационные колонки
    info_columns = [
        "Code",
        "candidate.name",
        "title",
        "year",
        "degree.degree_level",
        "degree.science_field",
        "institution_prepared",
        "supervisors_1.name",
        "supervisors_2.name",
        "specialties_1.name",
        "specialties_2.name",
    ]

    available_info_columns = [col for col in info_columns if col in dissertations_df.columns]

    if available_info_columns:
        info_df = (
            dissertations_df[available_info_columns]
            .copy()
            .drop_duplicates(subset=["Code"], keep="first")
        )
    else:
        info_df = pd.DataFrame(columns=["Code"])

    # Объединяем результаты с информацией
    results = search_results.merge(info_df, on="Code", how="left")

    # Округляем баллы
    results["profile_total"] = results["profile_total"].round(2)

    for code in selected_codes:
        if code in results.columns:
            results[code] = results[code].round(2)

    return results


def format_results_for_display(
    results: pd.DataFrame,
    selected_codes: List[str],
    classifier_labels: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Форматирует результаты для отображения в UI.

    Args:
        results: DataFrame с результатами поиска и метаданными
        selected_codes: Список выбранных кодов классификатора
        classifier_labels: Словарь {код: название} для красивых названий

    Returns:
        Tuple из (отформатированный DataFrame, словарь переименований колонок)
    """
    if classifier_labels is None:
        classifier_labels = {}

    # Создаем подписи для баллов по кодам
    score_labels = {}
    for code in selected_codes:
        label = classifier_labels.get(code, code)
        score_labels[code] = label

    # Переименовываем колонки с баллами
    for code, label in score_labels.items():
        if code in results.columns:
            results[label] = results[code].round(2)

    # Объединяем имена научных руководителей
    supervisor_cols = [col for col in ["supervisors_1.name", "supervisors_2.name"] if col in results.columns]

    if supervisor_cols:
        def join_names(row: pd.Series) -> str:
            names = []
            for value in row.tolist():
                if isinstance(value, str):
                    clean = value.strip()
                    if clean and clean not in names:
                        names.append(clean)
            return ", ".join(names)

        results["Научные руководители"] = (
            results[supervisor_cols]
            .replace(pd.NA, "")
            .apply(join_names, axis=1)
        )

    # Словарь для переименования колонок
    rename_map = {
        "Code": "Код диссертации",
        "candidate.name": "Автор",
        "title": "Название",
        "year": "Год",
        "degree.degree_level": "Степень",
        "degree.science_field": "Отрасль науки",
        "institution_prepared": "Организация",
        "specialties_1.name": "Специальность 1",
        "specialties_2.name": "Специальность 2",
        "profile_total": "Сумма баллов",
    }

    # Определяем порядок колонок для отображения
    column_order = [
        "Code",
        "candidate.name",
        "title",
        "year",
        "degree.degree_level",
        "degree.science_field",
        "institution_prepared",
        "Научные руководители",
        "specialties_1.name",
        "specialties_2.name",
        "profile_total",
    ] + list(score_labels.values())

    display_columns = [col for col in column_order if col in results.columns]
    display_df = results[display_columns].rename(columns=rename_map)

    return display_df, rename_map


# ==============================================================================
# ВАЛИДАЦИЯ
# ==============================================================================

def validate_code_selection(
    selected_codes: List[str],
    available_codes: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Проверяет корректность выбранных кодов.

    Args:
        selected_codes: Список выбранных кодов
        available_codes: Список доступных кодов в профилях

    Returns:
        Tuple (валидно: bool, сообщение об ошибке: Optional[str])
    """
    if not selected_codes:
        return False, "Не выбрано ни одного пункта классификатора"

    if len(selected_codes) > SELECTION_LIMIT:
        return False, f"Можно выбрать максимум {SELECTION_LIMIT} пунктов"

    missing = [code for code in selected_codes if code not in available_codes]
    if missing:
        return False, f"Коды не найдены в профилях: {', '.join(missing)}"

    return True, None


def classifier_label(code: str, classifier_dict: Dict[str, str]) -> str:
    """
    Возвращает полную подпись для кода классификатора.

    Args:
        code: Код классификатора
        classifier_dict: Словарь {код: название}

    Returns:
        Строка вида "код · название" или просто код, если название не найдено
    """
    title = classifier_dict.get(code)
    if title:
        return f"{code} · {title}"
    return code
