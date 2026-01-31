"""
Модуль Streamlit-вкладки поиска по тематическим профилям.

Объединяет два режима поиска:
1. По конкретным темам (пользователь выбирает пункты классификатора)
2. По мере общности/специфичности (поиск по энтропии)
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from profiles_search import (
    load_basic_scores,
    get_feature_columns,
    search_by_codes,
    merge_with_dissertation_info,
    format_results_for_display,
    validate_code_selection,
    classifier_label,
    SELECTION_LIMIT,
    DEFAULT_MIN_SCORE,
)


# ==============================================================================
# КОНСТАНТЫ
# ==============================================================================

PROFILE_SELECTION_SESSION_KEY = "profile_selected_codes"
DEFAULT_SCORES_FOLDER = "basic_scores"


# ==============================================================================
# ИНСТРУКЦИИ
# ==============================================================================

INSTRUCTION_BY_TOPICS = """
## 📊 Поиск по конкретным темам

На этой вкладке реализован содержательный поиск диссертаций по конкретным темам. 
Он основан не на совпадении слов в заголовке, а на анализе всего текста автореферата 
диссертационной работы. Поиск осуществляется с использованием иерархического 
классификатора, содержащего различные критерии, отражающие объект, процесс и 
результат диссертационного исследования.

---

### 🎯 Выбор тематики

В выпадающем списке **"Элемент классификатора"** выберите интересующую тему, метод 
или педагогическую технологию (например, "Начальное общее образование", "Информатика", 
"Интерактивные цифровые ресурсы", "Инклюзия"). Нажмите кнопку **"Добавить в подборку"**.

---

### 🔧 Логика поиска

- Можно добавить в подборку от одного до пяти пунктов классификатора
- Система отберет только те диссертации, у которых оценка присутствия **каждого** 
  выбранного пункта (темы) составляет не менее установленного порога (по умолчанию 4 балла 
  по 10-балльной шкале)
- Порог отсечения можно настроить в параметрах поиска

Для запуска алгоритма нажмите кнопку **"🔍 Найти диссертации"**.

---

### 📈 Результаты

Вы получите список работ, наиболее полно раскрывающих выбранные темы:

- **Ранжирование по сумме баллов**: сверху списка находятся диссертации, в которых 
  искомые темы проработаны максимально глубоко
- **Фильтр по таблице**: поле "🔍 Фильтр" позволяет найти работу по автору или слову 
  в названии внутри полученной выборки. Например, введя "Московский городской педагогический 
  университет", можно отфильтровать диссертации, связанные с МГПУ
- **Экспорт результатов**: итоговую таблицу с баллами и метаданными можно выгрузить 
  в форматах CSV или Excel

Чем выше балл, тем в большей степени тематика диссертации соответствует тем 
содержательным критериям, по которым осуществлялся поиск.

---

### 💡 Советы по использованию

- Начните с 1-2 ключевых тем для более широкого охвата
- Добавляйте дополнительные темы для более точного поиска
- Снизьте порог отсечения (например, до 3 баллов), если результатов слишком мало
- Используйте фильтр для поиска по конкретной организации или году
"""


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def show_instruction_dialog(mode: str) -> None:
    """Показывает инструкцию в зависимости от режима."""
    @st.dialog("📖 Инструкция", width="large")
    def _show():
        if mode == "topics":
            st.markdown(INSTRUCTION_BY_TOPICS)
        else:
            from entropy_specificity_tab import INSTRUCTION_ENTROPY
            st.markdown(INSTRUCTION_ENTROPY)

    _show()


def classifier_depth(code: str) -> int:
    """Возвращает глубину кода в иерархии."""
    return code.count(".") if code else 0


def classifier_format(
    option: Optional[Tuple[str, str, bool]],
    classifier_dict: Dict[str, str]
) -> str:
    """
    Форматирует элемент классификатора для отображения в selectbox.

    Args:
        option: Tuple (код, название, disabled) или None
        classifier_dict: Словарь с названиями

    Returns:
        Отформатированная строка
    """
    if option is None:
        return "— выберите пункт —"

    code, title, disabled = option
    indent = " " * classifier_depth(code)  # em space для отступов
    label = f"{code} {title}"

    if disabled:
        label += " (нельзя выбрать)"

    return f"{indent}{label}"


def _trigger_rerun() -> None:
    """Перезапускает приложение."""
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()  # type: ignore[attr-defined]


# ==============================================================================
# РЕЖИМ 1: ПОИСК ПО КОНКРЕТНЫМ ТЕМАМ
# ==============================================================================

def render_search_by_topics(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
    thematic_classifier: List[Tuple[str, str, bool]],
    classifier_dict: Dict[str, str],
) -> None:
    """
    Отрисовывает режим поиска по конкретным темам.

    Args:
        df: Основной DataFrame с диссертациями
        scores_df: DataFrame с тематическими профилями
        thematic_classifier: Список элементов классификатора
        classifier_dict: Словарь {код: название}
    """

    # --- Инструкция ---
    if st.button("📖 Инструкция", key="instruction_profiles_topics"):
        show_instruction_dialog("topics")

    st.subheader("🔍 Поиск по конкретным темам")

    st.write(
        f"Выберите до {SELECTION_LIMIT} пунктов классификатора. "
        f"Поиск найдет диссертации, где каждая выбранная тема оценена не менее "
        f"чем на {DEFAULT_MIN_SCORE} баллов (по 10-балльной шкале)."
    )

    # Инициализация session state
    if PROFILE_SELECTION_SESSION_KEY not in st.session_state:
        st.session_state[PROFILE_SELECTION_SESSION_KEY] = []

    selected_codes: List[str] = list(st.session_state.get(PROFILE_SELECTION_SESSION_KEY, []))

    # =========================================================================
    # ВЫБОР ЭЛЕМЕНТОВ КЛАССИФИКАТОРА
    # =========================================================================

    selection_container = st.container()

    with selection_container:
        st.markdown("### 📋 Выбор тематики")

        # Выпадающий список с элементами классификатора
        options: List[Optional[Tuple[str, str, bool]]] = [None] + thematic_classifier

        choice = st.selectbox(
            "Элемент классификатора",
            options=options,
            format_func=lambda opt: classifier_format(opt, classifier_dict),
            key="profile_classifier_choice",
        )

        # Логика добавления
        add_reason: Optional[str] = None
        add_code: Optional[str] = None

        if choice is not None:
            add_code = choice[0]

            if choice[2]:  # disabled
                add_reason = "Этот пункт нельзя выбрать. Пожалуйста, выберите более конкретный."
            elif add_code in selected_codes:
                add_reason = "Этот пункт уже добавлен в подборку."
            elif len(selected_codes) >= SELECTION_LIMIT:
                add_reason = f"Достигнут лимит: можно выбрать максимум {SELECTION_LIMIT} пунктов."

        # Кнопка добавления
        add_disabled = add_code is None or add_reason is not None
        add_clicked = st.button(
            "➕ Добавить в подборку",
            disabled=add_disabled,
            key="profile_add_button",
        )

        if add_clicked and add_code is not None:
            updated = selected_codes + [add_code]
            st.session_state[PROFILE_SELECTION_SESSION_KEY] = updated
            _trigger_rerun()

        if add_reason and choice is not None:
            st.caption(add_reason)

    st.markdown("---")

    # =========================================================================
    # ОТОБРАЖЕНИЕ ВЫБРАННЫХ ЭЛЕМЕНТОВ
    # =========================================================================

    st.markdown("### 🎯 Выбранные темы")

    if selected_codes:
        st.caption(
            f"Поиск найдет диссертации с баллом ≥ {DEFAULT_MIN_SCORE} по каждой теме."
        )

        for code in list(selected_codes):
            cols = st.columns([0.85, 0.15])

            with cols[0]:
                st.markdown(f"**{classifier_label(code, classifier_dict)}**")

            with cols[1]:
                if st.button(
                    "❌",
                    key=f"profile_remove_{code}",
                    use_container_width=True,
                ):
                    updated = [c for c in selected_codes if c != code]
                    st.session_state[PROFILE_SELECTION_SESSION_KEY] = updated
                    _trigger_rerun()

        # Кнопка очистки
        col_clear, col_dummy = st.columns([0.2, 0.8])
        with col_clear:
            if st.button("🗑️ Очистить все", key="profile_clear_selection"):
                st.session_state[PROFILE_SELECTION_SESSION_KEY] = []
                _trigger_rerun()

    else:
        st.info("Темы не выбраны. Выберите хотя бы один пункт классификатора выше.")

    st.markdown("---")

    # =========================================================================
    # ПАРАМЕТРЫ ПОИСКА
    # =========================================================================

    st.markdown("### ⚙️ Параметры поиска")

    min_score = st.slider(
        "Минимальный балл для каждой темы",
        min_value=1.0,
        max_value=10.0,
        value=DEFAULT_MIN_SCORE,
        step=0.5,
        key="profile_min_score",
        help="Диссертации с баллом ниже этого порога по любой из выбранных тем будут исключены"
    )

    st.markdown("---")

    # =========================================================================
    # ЗАПУСК ПОИСКА
    # =========================================================================

    run_search_click = st.button(
        "🔍 Найти диссертации",
        type="primary",
        disabled=not selected_codes,
        key="profile_run_search",
    )

    # Сохраняем факт активного поиска в session state
    if run_search_click:
        st.session_state["profile_search_active"] = True

    # =========================================================================
    # ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
    # =========================================================================

    if st.session_state.get("profile_search_active") and selected_codes:

        with st.spinner("Поиск диссертаций..."):
            try:
                # Валидация
                all_feature_columns = get_feature_columns(scores_df)
                valid, error_message = validate_code_selection(selected_codes, all_feature_columns)

                if not valid:
                    st.error(f"❌ {error_message}")
                    return

                # Поиск
                search_results = search_by_codes(
                    scores_df=scores_df,
                    selected_codes=selected_codes,
                    min_score=min_score
                )

                if search_results.empty:
                    st.info(
                        "🔍 По выбранным критериям диссертации не найдены. "
                        "Попробуйте снизить порог оценки или изменить выбранные темы."
                    )
                    return

                # Объединение с метаданными
                results = merge_with_dissertation_info(
                    search_results=search_results,
                    dissertations_df=df,
                    selected_codes=selected_codes
                )

                # Форматирование для отображения
                display_df, rename_map = format_results_for_display(
                    results=results,
                    selected_codes=selected_codes,
                    classifier_labels=classifier_dict
                )

                # Сохраняем в session state
                st.session_state["profile_results"] = display_df
                st.session_state["profile_results_full"] = results

            except Exception as exc:
                st.error(f"❌ Ошибка при поиске: {exc}")
                return

        # Показываем результаты
        display_df = st.session_state.get("profile_results")

        if display_df is not None and not display_df.empty:

            st.markdown("---")
            st.markdown("## 📊 Результаты поиска")

            st.success(f"✅ Найдено диссертаций: {len(display_df)}")

            # Фильтр по таблице
            st.markdown("### 🔍 Фильтр по таблице")

            fcol1, fcol2 = st.columns([0.6, 0.4])

            with fcol1:
                search_query = st.text_input(
                    "Поиск в результатах:",
                    placeholder="Например, фамилия автора, слово из названия, организация...",
                    help="Введите текст для поиска в таблице результатов. Поиск работает по всем полям.",
                    key="profile_result_filter"
                )

            # Применяем фильтр
            if search_query:
                mask = display_df.astype(str).apply(
                    lambda x: x.str.contains(search_query, case=False, na=False).any(),
                    axis=1
                )
                filtered_df = display_df[mask]
            else:
                filtered_df = display_df

            if len(filtered_df) != len(display_df):
                st.success(
                    f"Показано {len(filtered_df)} из {len(display_df)} диссертаций."
                )
            else:
                st.success(f"Показано {len(display_df)} диссертаций.")

            # Отображаем таблицу
            st.dataframe(filtered_df, use_container_width=True)

            # Скачивание результатов
            st.markdown("### 📥 Скачать результаты")

            selection_slug = "_".join(selected_codes[:3]) or "profiles"

            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                # CSV
                csv_data = filtered_df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📄 Скачать CSV",
                    data=csv_data.encode("utf-8-sig"),
                    file_name=f"profiles_{selection_slug}.csv",
                    mime="text/csv",
                    key="profile_download_csv",
                    use_container_width=True
                )

            with col_dl2:
                # Excel
                try:
                    buf_xlsx = io.BytesIO()
                    with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as writer:
                        filtered_df.to_excel(writer, index=False, sheet_name="Results")
                    data_xlsx = buf_xlsx.getvalue()

                    st.download_button(
                        label="📊 Скачать Excel",
                        data=data_xlsx,
                        file_name=f"profiles_{selection_slug}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="profile_download_xlsx",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Ошибка создания Excel: {e}")


# ==============================================================================
# ГЛАВНАЯ ФУНКЦИЯ РЕНДЕРИНГА ВКЛАДКИ
# ==============================================================================

def render_profiles_tab(
    df: pd.DataFrame,
    thematic_classifier: List[Tuple[str, str, bool]],
    scores_folder: str = DEFAULT_SCORES_FOLDER,
    specific_files: Optional[List[str]] = None,
) -> None:
    """
    Отрисовывает вкладку поиска по тематическим профилям с двумя режимами.

    Args:
        df: Основной DataFrame с диссертациями
        thematic_classifier: Список элементов классификатора (код, название, disabled)
        scores_folder: Папка с CSV-профилями
        specific_files: Список конкретных CSV-файлов (None = все из папки)
    """

    # Создаем словарь классификатора
    classifier_dict = {code: title for code, title, _ in thematic_classifier}

    # =========================================================================
    # ЗАГРУЗКА ДАННЫХ ПРОФИЛЕЙ
    # =========================================================================

    try:
        scores_df = load_basic_scores(folder_path=scores_folder)
        all_feature_columns = get_feature_columns(scores_df)

        st.success(
            f"✅ Загружено {len(scores_df)} профилей, "
            f"{len(all_feature_columns)} признаков классификатора"
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
    # ПЕРЕКЛЮЧАТЕЛЬ РЕЖИМОВ
    # =========================================================================

    st.markdown("## 🔍 Режим поиска")

    search_mode = st.radio(
        "Выберите режим:",
        options=["По конкретным темам", "По мере общности/специфичности"],
        horizontal=True,
        key="profile_search_mode_selector",
        help=(
            "**По конкретным темам** — классический поиск по выбранным пунктам классификатора.\n\n"
            "**По мере общности/специфичности** — поиск узкоспециализированных или "
            "междисциплинарных работ на основе энтропии."
        )
    )

    st.markdown("---")

    # =========================================================================
    # ОТОБРАЖЕНИЕ ВЫБРАННОГО РЕЖИМА
    # =========================================================================

    if search_mode == "По конкретным темам":
        render_search_by_topics(
            df=df,
            scores_df=scores_df,
            thematic_classifier=thematic_classifier,
            classifier_dict=classifier_dict,
        )

    else:  # "По мере общности/специфичности"
        render_entropy_specificity_tab(
            df=df,
            scores_folder=scores_folder,
            specific_files=specific_files,
            classifier_labels=classifier_dict,
        )
