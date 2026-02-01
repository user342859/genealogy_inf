"""
Модуль Streamlit-вкладки поиска по мере общности/специфичности.

Реализует поиск диссертаций на основе энтропии Шеннона с возможностью:
- Выбора научной школы (руководитель + уровень)
- Выбора узлов классификатора для анализа
- Применения иерархического коэффициента Z
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import streamlit as st

# Безопасные импорты
try:
    from entropy_specificity import (
        search_by_entropy,
        interpret_entropy,
        get_code_depth,
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from entropy_specificity import (
        search_by_entropy,
        interpret_entropy,
        get_code_depth,
    )

# ==============================================================================
# КОНСТАНТЫ
# ==============================================================================

DEFAULT_SCORES_FOLDER = "basic_scores"
DEFAULT_MIN_THRESHOLD = 3.0
MAX_RESULTS_DISPLAY = 100

# ==============================================================================
# ИНСТРУКЦИЯ
# ==============================================================================

INSTRUCTION_ENTROPY = """
## 📊 Поиск по мере общности/специфичности

Этот инструмент позволяет находить диссертации по степени **узости** или **широты** 
исследования на основе анализа тематических профилей.

---

### 🎯 Область анализа

#### Выбор научной школы
Выберите одного или нескольких научных руководителей для анализа их школы:
- **Только прямые диссертанты** — анализ работ непосредственных учеников
- **Все поколения диссертантов** — анализ всего дерева научного руководства

#### Выбор узлов классификатора
Можно анализировать:
- **Весь классификатор** — все темы (медленнее, но полная картина)
- **Конкретные узлы** — выбранные разделы классификатора (быстрее, фокусировка на теме)

---

### 🔧 Параметры поиска

#### 1. Тип формулы энтропии

- **Классическая формула Шеннона**: H = -∑ p_i · log(p_i)
  - Все темы классификатора рассматриваются как равноправные
  - Подходит для общей оценки разнообразия тем

- **С иерархическим коэффициентом Z**: H = -∑ Z_i · p_i · log(p_i)
  - Учитывает структуру классификатора
  - Придает больший вес темам на глубоких уровнях иерархии
  - **Рекомендуется** для более точной оценки специфичности

#### 2. Отсечение малых значений

Темы с баллами ниже порога исключаются из анализа.
- **По умолчанию**: 3 балла
- **Рекомендация**: используйте 2-4 балла

---

### 📈 Интерпретация результатов

| Диапазон энтропии | Интерпретация |
|-------------------|---------------|
| < 1.0 | Очень узкая специализация |
| 1.0 – 2.5 | Узкая специализация |
| 2.5 – 4.0 | Умеренная широта |
| 4.0 – 5.5 | Широкий охват |
| > 5.5 | Очень широкий охват (междисциплинарность) |

---

### 💡 Примеры использования

**Сравнить специализацию работ разных поколений школы:**
- Выбрать руководителя
- "Все поколения диссертантов"
- Весь классификатор
- Посмотреть распределение энтропии

**Найти узкоспециализированные работы в конкретной области:**
- Выбрать руководителя(ей)
- Выбрать нужные узлы классификатора
- "С коэффициентом Z"
- "От узких к широким"
"""

# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def show_instruction_dialog() -> None:
    """Показывает диалог с инструкцией."""
    @st.dialog("📖 Инструкция", width="large")
    def _show():
        st.markdown(INSTRUCTION_ENTROPY)

    _show()


def load_scores_from_folder(
    folder_path: str = DEFAULT_SCORES_FOLDER,
    specific_files: Optional[List[str]] = None
) -> pd.DataFrame:
    """Загружает данные тематических профилей из CSV файлов."""
    base = Path(folder_path).expanduser().resolve()

    if specific_files:
        files = [base / f for f in specific_files if (base / f).exists()]
    else:
        files = sorted(base.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"CSV файлы не найдены в {base}")

    frames: List[pd.DataFrame] = []
    for file in files:
        try:
            frame = pd.read_csv(file)
            if "Code" not in frame.columns:
                raise KeyError(f"Файл {file.name} не содержит колонку 'Code'")
            frames.append(frame)
        except Exception as e:
            print(f"Ошибка при загрузке {file}: {e}")
            continue

    if not frames:
        raise ValueError("Не удалось загрузить ни один файл")

    scores = pd.concat(frames, ignore_index=True)
    scores = scores.dropna(subset=["Code"])
    scores["Code"] = scores["Code"].astype(str).str.strip()
    scores = scores[scores["Code"].str.len() > 0]
    scores = scores.drop_duplicates(subset=["Code"], keep="first")

    feature_columns = [c for c in scores.columns if c != "Code"]
    scores[feature_columns] = scores[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    scores[feature_columns] = scores[feature_columns].fillna(0.0)

    return scores


def get_feature_columns(scores: pd.DataFrame) -> List[str]:
    """Возвращает список колонок с признаками."""
    return [c for c in scores.columns if c != "Code"]


def get_all_nodes_of_branch(
    node_code: str,
    all_codes: List[str]
) -> List[str]:
    """
    Возвращает все узлы, относящиеся к ветке (сам узел + все потомки).

    Args:
        node_code: Код узла (например, "1.1")
        all_codes: Список всех доступных кодов

    Returns:
        Список кодов узлов ветки
    """
    result = []
    for code in all_codes:
        # Узел сам или его потомок (начинается с node_code.)
        if code == node_code or code.startswith(f"{node_code}."):
            result.append(code)
    return result


# ==============================================================================
# ОСНОВНАЯ ФУНКЦИЯ РЕНДЕРИНГА ВКЛАДКИ
# ==============================================================================

def render_entropy_specificity_tab(
    df: pd.DataFrame,
    idx: Dict[str, Set[int]],
    lineage_func,
    rows_for_func,
    scores_folder: str = DEFAULT_SCORES_FOLDER,
    specific_files: Optional[List[str]] = None,
    classifier_labels: Optional[Dict[str, str]] = None,
    thematic_classifier: Optional[List[tuple]] = None,
) -> None:
    """
    Отрисовывает вкладку поиска по мере общности/специфичности.

    Args:
        df: Основной DataFrame с диссертациями
        idx: Индекс для поиска по именам (из streamlit_app.py)
        lineage_func: Функция построения дерева (из streamlit_app.py)
        rows_for_func: Функция поиска строк по имени (из streamlit_app.py)
        scores_folder: Папка с CSV-профилями
        specific_files: Список конкретных CSV-файлов
        classifier_labels: Словарь {код: название} для подписей
        thematic_classifier: Список элементов классификатора (код, название, disabled)
    """
    if classifier_labels is None:
        classifier_labels = {}

    # --- Кнопка инструкции ---
    if st.button("📖 Инструкция", key="instruction_entropy"):
        show_instruction_dialog()

    st.subheader("📊 Поиск по мере общности/специфичности")

    st.markdown("""
    Найдите диссертации по степени **узости** или **широты** исследования в рамках 
    выбранной научной школы. Низкая энтропия = узкая тема, высокая энтропия = широкий охват.
    """)

    # =========================================================================
    # ЗАГРУЗКА ДАННЫХ ПРОФИЛЕЙ
    # =========================================================================
    try:
        scores_df = load_scores_from_folder(
            folder_path=scores_folder,
            specific_files=specific_files
        )

        all_feature_columns = get_feature_columns(scores_df)
        st.success(
            f"✅ Загружено {len(scores_df)} профилей, "
            f"{len(all_feature_columns)} признаков"
        )

    except FileNotFoundError as e:
        st.error(f"❌ Папка или файлы не найдены: {e}")
        st.info(
            f"Убедитесь, что папка '{scores_folder}' существует и содержит CSV-файлы "
            "с тематическими профилями."
        )
        return
    except Exception as e:
        st.error(f"❌ Ошибка загрузки данных: {e}")
        return

    st.markdown("---")

    # =========================================================================
    # ВЫБОР НАУЧНОЙ ШКОЛЫ
    # =========================================================================

    st.markdown("### 🌳 Выбор научной школы")

    col_root1, col_root2 = st.columns([0.6, 0.4])

    with col_root1:
        # Поле ввода имени руководителя
        root_name = st.text_input(
            "Имя научного руководителя",
            key="entropy_root_name",
            placeholder="Например: Иванов Иван Иванович",
            help="Введите ФИО научного руководителя для анализа его школы"
        )

    with col_root2:
        # Выбор уровня
        first_level_only = st.checkbox(
            "Только прямые диссертанты",
            value=False,
            key="entropy_first_level_only",
            help="Если отмечено — анализируются только непосредственные ученики. "
                 "Если нет — всё дерево научного руководства."
        )

    # Проверяем наличие руководителя
    school_df = None
    school_codes = []

    if root_name and root_name.strip():
        try:
            # Получаем дерево
            filter_func = None
            if first_level_only:
                # Только первый уровень
                filter_func = lambda row: True

            G, school_df = lineage_func(
                df=df,
                index=idx,
                root=root_name.strip(),
                first_level_filter=filter_func if first_level_only else None
            )

            if school_df.empty:
                st.warning(f"⚠️ Научный руководитель '{root_name}' не найден или у него нет диссертантов.")
            else:
                # Получаем коды диссертаций школы
                school_codes = school_df["Code"].astype(str).tolist()

                level_text = "прямых диссертантов" if first_level_only else "диссертантов (все поколения)"
                st.success(
                    f"✅ Найдено {len(school_codes)} {level_text} "
                    f"в школе '{root_name}'"
                )

        except Exception as e:
            st.error(f"❌ Ошибка построения дерева: {e}")
            return

    else:
        st.info("ℹ️ Введите имя научного руководителя для начала анализа.")
        return

    st.markdown("---")

    # =========================================================================
    # ВЫБОР УЗЛОВ КЛАССИФИКАТОРА
    # =========================================================================

    st.markdown("### 📋 Выбор узлов классификатора для анализа")

    # Радио-кнопка: весь классификатор или конкретные узлы
    classifier_mode = st.radio(
        "Область анализа:",
        options=["Весь классификатор", "Конкретные узлы"],
        horizontal=True,
        key="entropy_classifier_mode",
        help=(
            "**Весь классификатор** — анализ по всем темам (медленнее).\n\n"
            "**Конкретные узлы** — выбор конкретных разделов для анализа (быстрее)."
        )
    )

    selected_nodes = []

    if classifier_mode == "Конкретные узлы":
        if thematic_classifier is None:
            st.error("❌ Классификатор не передан в функцию. Обратитесь к разработчику.")
            return

        st.caption("Выберите один или несколько узлов. Будут учтены выбранные узлы и все их потомки.")

        # Создаем selectbox для выбора узлов
        # Фильтруем только родительские узлы (с disabled=True) для простоты
        parent_nodes = [(code, title) for code, title, disabled in thematic_classifier if disabled]

        if not parent_nodes:
            st.warning("⚠️ Родительские узлы не найдены в классификаторе.")
            selected_nodes = []
        else:
            # Мультиселект
            selected_node_labels = st.multiselect(
                "Выберите узлы классификатора:",
                options=[f"{code} · {title}" for code, title in parent_nodes],
                key="entropy_selected_nodes",
                help="Можно выбрать несколько узлов. Анализ будет включать выбранные узлы и все их подузлы."
            )

            # Извлекаем коды из выбранных меток
            selected_nodes = []
            for label in selected_node_labels:
                code = label.split(" · ")[0]
                selected_nodes.append(code)

    else:
        # Весь классификатор
        selected_nodes = []
        st.caption("Будут использованы все доступные узлы классификатора.")

    # Определяем финальный список колонок для анализа
    if classifier_mode == "Весь классификатор":
        analysis_columns = all_feature_columns
    else:
        if not selected_nodes:
            st.warning("⚠️ Не выбрано ни одного узла. Выберите хотя бы один узел для анализа.")
            return

        # Собираем все узлы выбранных веток
        analysis_columns = []
        for node_code in selected_nodes:
            branch_codes = get_all_nodes_of_branch(node_code, all_feature_columns)
            analysis_columns.extend(branch_codes)

        # Убираем дубликаты
        analysis_columns = list(set(analysis_columns))

        st.success(f"✅ Для анализа отобрано {len(analysis_columns)} признаков из выбранных веток")

    st.markdown("---")

    # =========================================================================
    # ПАРАМЕТРЫ ПОИСКА
    # =========================================================================

    st.markdown("### ⚙️ Параметры поиска")

    col1, col2, col3 = st.columns(3)

    with col1:
        use_hierarchical = st.radio(
            "Тип формулы энтропии",
            options=[False, True],
            format_func=lambda x: "С коэффициентом Z" if x else "Классическая Шеннона",
            key="entropy_use_hierarchical",
            help=(
                "**Классическая** — все темы равноправны.\n\n"
                "**С коэффициентом Z** — учитывается иерархия классификатора."
            )
        )

    with col2:
        min_threshold = st.number_input(
            "Отсечение малых значений",
            min_value=0.0,
            max_value=10.0,
            value=DEFAULT_MIN_THRESHOLD,
            step=0.5,
            key="entropy_min_threshold",
            help="Темы с баллами ниже этого значения исключаются из анализа."
        )

    with col3:
        sort_order = st.radio(
            "Порядок сортировки",
            options=["asc", "desc"],
            format_func=lambda x: "От узких к широким" if x == "asc" else "От широких к узким",
            key="entropy_sort_order",
            help=(
                "**От узких к широким** — сначала специализированные работы.\n\n"
                "**От широких к узким** — сначала междисциплинарные работы."
            )
        )

    st.markdown("---")

    # =========================================================================
    # ЗАПУСК АНАЛИЗА
    # =========================================================================

    can_run = bool(school_codes) and bool(analysis_columns)

    if st.button(
        "🔍 Найти диссертации",
        key="entropy_search_button",
        type="primary",
        disabled=not can_run,
        use_container_width=False
    ):
        with st.spinner(f"Вычисление энтропии для {len(school_codes)} диссертаций..."):
            try:
                # Фильтруем scores_df по кодам школы
                school_scores_df = scores_df[scores_df["Code"].isin(school_codes)].copy()

                if school_scores_df.empty:
                    st.warning("⚠️ Для диссертаций выбранной школы не найдены тематические профили.")
                    return

                # Запускаем поиск по энтропии
                results = search_by_entropy(
                    scores_df=school_scores_df,
                    feature_columns=analysis_columns,
                    use_hierarchical=use_hierarchical,
                    min_threshold=min_threshold,
                    ascending=(sort_order == "asc")
                )

                # Добавляем метаданные из основного df
                if not results.empty:
                    meta_columns = [
                        "Code", "candidate.name", "title", "year",
                        "degree.degree_level", "institution_prepared"
                    ]
                    available_meta = [col for col in meta_columns if col in df.columns]

                    if available_meta:
                        df_meta = df[available_meta].drop_duplicates(subset=["Code"], keep="first")
                        results = results.merge(df_meta, on="Code", how="left")

                # Сохраняем результаты в session_state
                st.session_state["entropy_results"] = results
                st.session_state["entropy_params"] = {
                    "use_hierarchical": use_hierarchical,
                    "min_threshold": min_threshold,
                    "sort_order": sort_order,
                    "root_name": root_name,
                    "first_level_only": first_level_only,
                    "classifier_mode": classifier_mode,
                    "selected_nodes": selected_nodes,
                }

                st.success(f"✅ Найдено {len(results)} диссертаций")

            except Exception as e:
                st.error(f"❌ Ошибка при вычислении: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # =========================================================================
    # ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
    # =========================================================================

    if "entropy_results" in st.session_state:
        results = st.session_state["entropy_results"]
        params = st.session_state.get("entropy_params", {})

        st.markdown("---")
        st.markdown("## 📈 Результаты")

        # Статистика
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

        with col_stat1:
            st.metric(
                "Найдено диссертаций",
                len(results)
            )

        with col_stat2:
            min_entropy = results["entropy"].min()
            st.metric(
                "Минимальная энтропия",
                f"{min_entropy:.2f}",
                help="Самая узкая специализация"
            )

        with col_stat3:
            max_entropy = results["entropy"].max()
            st.metric(
                "Максимальная энтропия",
                f"{max_entropy:.2f}",
                help="Самый широкий охват"
            )

        with col_stat4:
            avg_entropy = results["entropy"].mean()
            st.metric(
                "Средняя энтропия",
                f"{avg_entropy:.2f}"
            )

        # Информация о параметрах поиска
        with st.expander("ℹ️ Параметры поиска", expanded=False):
            st.write(f"**Научный руководитель:** {params.get('root_name', 'не указан')}")
            st.write(f"**Уровень:** {'Только прямые диссертанты' if params.get('first_level_only') else 'Все поколения'}")
            st.write(f"**Узлы классификатора:** {params.get('classifier_mode', 'Весь классификатор')}")
            if params.get('selected_nodes'):
                st.write(f"**Выбранные узлы:** {', '.join(params.get('selected_nodes', []))}")
            st.write(f"**Формула энтропии:** {'С коэффициентом Z' if params.get('use_hierarchical') else 'Классическая Шеннона'}")
            st.write(f"**Порог отсечения:** {params.get('min_threshold', 3.0)} баллов")

        # Фильтр по таблице
        st.markdown("### 🔍 Фильтр по таблице")
        search_text = st.text_input(
            "Поиск по коду, автору или другим полям:",
            key="entropy_table_filter",
            help="Введите текст для поиска в таблице результатов"
        )

        # Применяем фильтр
        if search_text:
            mask = results.astype(str).apply(
                lambda row: row.str.contains(search_text, case=False, na=False).any(),
                axis=1
            )
            filtered_results = results[mask]
        else:
            filtered_results = results

        st.caption(
            f"Показано {len(filtered_results)} из {len(results)} диссертаций"
        )

        # Подготовка таблицы для отображения
        display_df = filtered_results.copy()

        # Добавляем интерпретацию
        display_df["Интерпретация"] = display_df["entropy"].apply(
            lambda x: interpret_entropy(x, params.get("use_hierarchical", False))
        )

        # Переименовываем колонки для отображения
        rename_map = {
            "Code": "Код диссертации",
            "entropy": "Энтропия",
            "features_count": "Количество тем",
            "candidate.name": "Автор",
            "title": "Название",
            "year": "Год",
            "degree.degree_level": "Степень",
            "institution_prepared": "Организация"
        }

        display_df = display_df.rename(columns=rename_map)

        # Выбираем колонки для отображения
        cols_to_show = [
            "Код диссертации", "Энтропия", "Количество тем", "Интерпретация",
            "Автор", "Название", "Год", "Степень", "Организация"
        ]
        cols_to_show = [c for c in cols_to_show if c in display_df.columns]

        # Отображаем таблицу
        st.dataframe(
            display_df[cols_to_show].head(MAX_RESULTS_DISPLAY),
            use_container_width=True,
            hide_index=True
        )

        if len(filtered_results) > MAX_RESULTS_DISPLAY:
            st.info(
                f"ℹ️ Отображены первые {MAX_RESULTS_DISPLAY} результатов. "
                "Скачайте полные данные для просмотра всех результатов."
            )

        # Кнопки скачивания
        st.markdown("### 📥 Скачать результаты")

        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            # CSV
            csv_data = filtered_results.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📄 Скачать CSV",
                data=csv_data.encode("utf-8-sig"),
                file_name="entropy_search_results.csv",
                mime="text/csv",
                key="entropy_download_csv",
                use_container_width=True
            )

        with col_dl2:
            # Excel
            try:
                buf_xlsx = io.BytesIO()
                with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as writer:
                    filtered_results.to_excel(writer, index=False, sheet_name="Results")
                data_xlsx = buf_xlsx.getvalue()

                st.download_button(
                    label="📊 Скачать Excel",
                    data=data_xlsx,
                    file_name="entropy_search_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="entropy_download_xlsx",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Ошибка создания Excel: {e}")

        # Детальная информация
        with st.expander("📊 Распределение энтропии", expanded=False):
            st.markdown("#### Статистика энтропии по диапазонам")

            # Создаем диапазоны
            bins = [0, 1.0, 2.5, 4.0, 5.5, float('inf')]
            labels = [
                "< 1.0 (Очень узкая)",
                "1.0-2.5 (Узкая)",
                "2.5-4.0 (Умеренная)",
                "4.0-5.5 (Широкая)",
                "> 5.5 (Очень широкая)"
            ]

            results_copy = results.copy()
            results_copy["Диапазон"] = pd.cut(
                results_copy["entropy"],
                bins=bins,
                labels=labels
            )

            distribution = results_copy["Диапазон"].value_counts().sort_index()

            dist_df = pd.DataFrame({
                "Диапазон": distribution.index,
                "Количество": distribution.values,
                "Процент": (distribution.values / len(results) * 100).round(1)
            })

            st.dataframe(dist_df, use_container_width=True, hide_index=True)
