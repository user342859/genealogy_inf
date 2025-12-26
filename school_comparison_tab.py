"""
Модуль Streamlit-вкладки сравнения научных школ.
Импортируйте и вызывайте render_school_comparison_tab() в основном приложении.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from school_comparison import (
    DistanceMetric,
    ComparisonScope,
    DISTANCE_METRIC_LABELS,
    SCOPE_LABELS,
    load_scores_from_folder,
    get_feature_columns,
    get_nodes_at_level,
    get_selectable_nodes,
    filter_columns_by_nodes,
    get_code_depth,
    compute_silhouette_analysis,
    create_silhouette_plot,
    create_comparison_summary,
    interpret_silhouette_score,
    gather_school_dataset,
)


# ==============================================================================
# КОНСТАНТЫ (можно переопределить при вызове)
# ==============================================================================

DEFAULT_SCORES_FOLDER = "basic_scores"
AUTHOR_COLUMN = "candidate.name"


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def get_all_supervisors(df: pd.DataFrame) -> List[str]:
    """Получает список всех научных руководителей из DataFrame."""
    supervisor_cols = [
        col for col in df.columns 
        if "supervisor" in col.lower() and "name" in col.lower()
    ]

    all_supervisors: Set[str] = set()
    for col in supervisor_cols:
        if col in df.columns:
            all_supervisors.update(
                str(v).strip() for v in df[col].dropna().unique() 
                if str(v).strip()
            )

    return sorted(all_supervisors)


def format_node_option(code: str, classifier_dict: Dict[str, str]) -> str:
    """Форматирует узел для отображения в selectbox."""
    depth = get_code_depth(code)
    indent = "  " * (depth - 1)
    title = classifier_dict.get(code, "")
    if title:
        return f"{indent}{code} — {title}"
    return f"{indent}{code}"


# ==============================================================================
# ОСНОВНАЯ ФУНКЦИЯ РЕНДЕРИНГА ВКЛАДКИ
# ==============================================================================

def render_school_comparison_tab(
    df: pd.DataFrame,
    idx: Dict[str, Set[int]],
    lineage_func: Callable,
    rows_for_func: Callable,
    default_scores_folder: str = DEFAULT_SCORES_FOLDER,
    classifier_labels: Optional[Dict[str, str]] = None,
) -> None:
    """
    Отрисовывает вкладку сравнения научных школ.

    Args:
        df: Основной DataFrame с диссертациями
        idx: Индекс для поиска по именам
        lineage_func: Функция построения генеалогии (lineage из основного приложения)
        rows_for_func: Функция поиска строк (rows_for из основного приложения)
        default_scores_folder: Папка по умолчанию с CSV-профилями
        classifier_labels: Словарь {код: название} для подписей узлов (опционально)
    """

    if classifier_labels is None:
        classifier_labels = {}

    st.subheader("🔬 Сравнение научных школ по тематическим профилям")

    st.markdown("""
    Этот инструмент позволяет сравнить тематические профили диссертаций, 
    защищённых под руководством выбранных учёных. Основная метрика — 
    **коэффициент силуэта**, показывающий насколько чётко различаются 
    тематические направления разных научных школ.
    """)

    # =========================================================================
    # НАСТРОЙКИ ИСТОЧНИКА ДАННЫХ
    # =========================================================================

    with st.expander("⚙️ Настройки источника данных", expanded=False):
        scores_folder = st.text_input(
            "Папка с CSV-файлами профилей",
            value=default_scores_folder,
            key="school_comp_folder",
            help="Путь к папке, содержащей CSV-файлы с тематическими профилями"
        )

        use_all_files = st.checkbox(
            "Использовать все CSV из папки",
            value=True,
            key="school_comp_use_all",
            help="Если отключено, можно указать конкретные файлы"
        )

        specific_files: Optional[List[str]] = None
        if not use_all_files:
            files_input = st.text_area(
                "Список CSV-файлов (по одному на строку)",
                key="school_comp_files",
                help="Укажите имена файлов, например: school1.csv"
            )
            if files_input.strip():
                specific_files = [
                    f.strip() for f in files_input.strip().split("\n") 
                    if f.strip()
                ]

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
    # ВЫБОР НАУЧНЫХ ШКОЛ
    # =========================================================================

    st.markdown("### 👥 Выбор научных школ для сравнения")

    all_supervisors_sorted = get_all_supervisors(df)

    if not all_supervisors_sorted:
        st.error("❌ В данных не найдено научных руководителей")
        return

    selected_schools = st.multiselect(
        "Выберите руководителей научных школ (минимум 2)",
        options=all_supervisors_sorted,
        default=[],
        key="school_comp_selection",
        help="Выберите 2 или более научных руководителей для сравнения их школ"
    )

    if len(selected_schools) < 2:
        st.warning("⚠️ Выберите минимум 2 научных руководителя для сравнения")
        return

    st.markdown("---")

    # =========================================================================
    # ПАРАМЕТРЫ АНАЛИЗА
    # =========================================================================

    col_params1, col_params2 = st.columns(2)

    with col_params1:
        st.markdown("### 📐 Параметры анализа")

        # Выбор охвата диссертаций
        scope_options = list(SCOPE_LABELS.keys())
        scope_labels_list = [SCOPE_LABELS[s] for s in scope_options]
        scope_idx = st.radio(
            "Охват диссертаций",
            options=range(len(scope_options)),
            format_func=lambda i: scope_labels_list[i],
            key="school_comp_scope",
            help=(
                "**Прямые диссертанты** — только защитившиеся под непосредственным "
                "руководством выбранного учёного.\n\n"
                "**Все поколения** — включая диссертантов диссертантов и далее."
            )
        )
        selected_scope: ComparisonScope = scope_options[scope_idx]

        # Выбор метрики расстояния
        metric_options = list(DISTANCE_METRIC_LABELS.keys())
        metric_labels_list = [DISTANCE_METRIC_LABELS[m] for m in metric_options]
        metric_idx = st.selectbox(
            "Метрика расстояния",
            options=range(len(metric_options)),
            format_func=lambda i: metric_labels_list[i],
            key="school_comp_metric",
            help=(
                "**Прямоугольный базис** — стандартное вычисление расстояний.\n\n"
                "**Косоугольный базис** — учитывает иерархическую структуру "
                "тематического классификатора, где дочерние узлы наследуют "
                "часть значений от родительских."
            )
        )
        selected_metric: DistanceMetric = metric_options[metric_idx]

    with col_params2:
        st.markdown("### 🎯 Выбор тематического базиса")

        # Выбор: весь базис или конкретные узлы
        basis_choice = st.radio(
            "Базис для сравнения",
            options=["full", "selected"],
            format_func=lambda x: "Весь базис (все темы)" if x == "full" else "Конкретные разделы",
            key="school_comp_basis_choice",
            help=(
                "**Весь базис** — используются все тематические признаки.\n\n"
                "**Конкретные разделы** — выберите узлы классификатора, "
                "анализ будет по ним и всем их подтемам."
            )
        )

        selected_nodes: Optional[List[str]] = None

        if basis_choice == "selected":
            # Получаем узлы 1-3 уровней для выбора
            selectable = get_selectable_nodes(all_feature_columns, max_level=3)

            if not selectable:
                st.warning("Нет доступных узлов для выбора")
            else:
                # Группируем по уровням для удобства
                level1_nodes = [n for n in selectable if get_code_depth(n) == 1]
                level2_nodes = [n for n in selectable if get_code_depth(n) == 2]
                level3_nodes = [n for n in selectable if get_code_depth(n) == 3]

                st.caption("Выберите разделы классификатора (узлы 1-3 уровня):")

                # Показываем чекбоксы по уровням
                selected_nodes = []

                if level1_nodes:
                    st.markdown("**Уровень 1 (крупные разделы):**")
                    cols_l1 = st.columns(min(4, len(level1_nodes)))
                    for i, node in enumerate(level1_nodes):
                        with cols_l1[i % len(cols_l1)]:
                            label = classifier_labels.get(node, node)
                            if st.checkbox(f"{node}", key=f"node_l1_{node}"):
                                selected_nodes.append(node)

                if level2_nodes:
                    with st.expander("Уровень 2 (подразделы)", expanded=False):
                        cols_l2 = st.columns(3)
                        for i, node in enumerate(level2_nodes):
                            with cols_l2[i % 3]:
                                label = classifier_labels.get(node, "")
                                display = f"{node}" + (f" ({label})" if label else "")
                                if st.checkbox(display, key=f"node_l2_{node}"):
                                    selected_nodes.append(node)

                if level3_nodes:
                    with st.expander("Уровень 3 (детальные темы)", expanded=False):
                        cols_l3 = st.columns(3)
                        for i, node in enumerate(level3_nodes):
                            with cols_l3[i % 3]:
                                label = classifier_labels.get(node, "")
                                display = f"{node}" + (f" ({label})" if label else "")
                                if st.checkbox(display, key=f"node_l3_{node}"):
                                    selected_nodes.append(node)

                if selected_nodes:
                    # Показываем, сколько признаков будет использовано
                    filtered_cols = filter_columns_by_nodes(all_feature_columns, selected_nodes)
                    st.info(
                        f"✓ Выбрано {len(selected_nodes)} узлов → "
                        f"{len(filtered_cols)} признаков (включая все подтемы)"
                    )
                else:
                    st.warning("⚠️ Выберите хотя бы один раздел")

    # Параметры косоугольного базиса
    decay_factor = 0.5
    if "oblique" in selected_metric:
        with st.expander("🔧 Параметры косоугольного базиса", expanded=False):
            decay_factor = st.slider(
                "Коэффициент затухания",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1,
                key="school_comp_decay",
                help=(
                    "Определяет силу влияния родительских узлов на дочерние.\n\n"
                    "**0.1** — слабое влияние родителей\n"
                    "**0.5** — сбалансированное затухание\n"
                    "**0.9** — сильное влияние родителей"
                )
            )

            st.markdown("""
            **Косоугольный базис** — метод, учитывающий иерархическую природу 
            тематического классификатора. При его использовании значения дочерних 
            узлов частично зависят от значений родительских узлов, что позволяет 
            более точно оценить семантическую близость тематических профилей.
            """)

    st.markdown("---")

    # Проверка готовности к запуску
    ready_to_run = True
    if basis_choice == "selected" and (not selected_nodes or len(selected_nodes) == 0):
        ready_to_run = False

    # =========================================================================
    # ЗАПУСК АНАЛИЗА
    # =========================================================================

    if st.button(
        "🚀 Запустить анализ", 
        key="school_comp_run", 
        type="primary",
        disabled=not ready_to_run
    ):

        # --- Сбор данных ---
        with st.spinner("📥 Сбор данных научных школ..."):
            datasets: Dict[str, pd.DataFrame] = {}
            missing_info_all: Dict[str, pd.DataFrame] = {}
            stats_info = []

            progress_bar = st.progress(0)

            for i, school_name in enumerate(selected_schools):
                try:
                    dataset, missing_info, total_count = gather_school_dataset(
                        df=df,
                        index=idx,
                        root=school_name,
                        scores=scores_df,
                        scope=selected_scope,
                        lineage_func=lineage_func,
                        rows_for_func=rows_for_func,
                    )

                    datasets[school_name] = dataset
                    if not missing_info.empty:
                        missing_info_all[school_name] = missing_info

                    stats_info.append({
                        "Школа": school_name,
                        "Найдено диссертаций": total_count,
                        "С профилями": len(dataset),
                        "Без профилей": len(missing_info) if not missing_info.empty else 0
                    })

                except Exception as e:
                    st.warning(f"⚠️ Ошибка для школы '{school_name}': {e}")

                progress_bar.progress((i + 1) / len(selected_schools))

            progress_bar.empty()

        # Статистика сбора
        if stats_info:
            st.markdown("#### 📊 Статистика сбора данных")
            stats_df = pd.DataFrame(stats_info)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Проверка данных
        valid_datasets = {k: v for k, v in datasets.items() if not v.empty}

        if len(valid_datasets) < 2:
            st.error(
                "❌ Недостаточно данных для анализа. "
                "Нужно минимум 2 школы с тематическими профилями."
            )
            return

        # --- Вычисление силуэта ---
        with st.spinner("🔬 Вычисление анализа силуэта..."):
            try:
                # Передаём selected_nodes (None если весь базис)
                nodes_for_analysis = selected_nodes if basis_choice == "selected" else None

                (
                    overall_score,
                    sample_scores,
                    labels,
                    school_order,
                    used_columns
                ) = compute_silhouette_analysis(
                    datasets=valid_datasets,
                    feature_columns=all_feature_columns,
                    metric=selected_metric,
                    selected_nodes=nodes_for_analysis,
                    decay_factor=decay_factor,
                )
            except ValueError as e:
                st.error(f"❌ Ошибка анализа: {e}")
                return
            except Exception as e:
                st.error(f"❌ Неожиданная ошибка: {e}")
                return

        # =====================================================================
        # ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
        # =====================================================================

        st.markdown("---")
        st.markdown("## 📈 Результаты анализа")

        # --- Основной показатель ---
        col_score, col_interp = st.columns([1, 2])

        with col_score:
            st.metric(
                label="Коэффициент силуэта",
                value=f"{overall_score:.3f}",
                help=(
                    "Диапазон от -1 до 1.\n\n"
                    "**> 0.5** — хорошее разделение\n"
                    "**0.25-0.5** — умеренное\n"
                    "**< 0.25** — слабое"
                )
            )

        with col_interp:
            st.info(interpret_silhouette_score(overall_score))

        # Информация о базисе
        basis_info = "весь базис" if basis_choice == "full" else f"узлы: {', '.join(selected_nodes or [])}"
        st.caption(f"📌 Базис: {basis_info} | Признаков: {len(used_columns)} | Метрика: {DISTANCE_METRIC_LABELS[selected_metric]}")

        # --- График силуэта ---
        st.markdown("### 📊 График силуэта")

        fig = create_silhouette_plot(
            sample_scores=sample_scores,
            labels=labels,
            school_order=school_order,
            overall_score=overall_score,
            metric_label=DISTANCE_METRIC_LABELS[selected_metric],
        )

        st.pyplot(fig)
        plt.close(fig)

        # Скачивание графика
        buf = io.BytesIO()
        fig = create_silhouette_plot(
            sample_scores=sample_scores,
            labels=labels,
            school_order=school_order,
            overall_score=overall_score,
            metric_label=DISTANCE_METRIC_LABELS[selected_metric],
        )
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📥 Скачать график (PNG)",
                data=buf.getvalue(),
                file_name="silhouette_plot.png",
                mime="image/png",
                key="school_comp_download_png"
            )

        # --- Сводная таблица ---
        st.markdown("### 📋 Сводная статистика по школам")

        summary_df = create_comparison_summary(
            datasets=valid_datasets,
            feature_columns=used_columns,
            school_order=school_order,
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Скачивание сводки
        with col_dl2:
            csv_data = summary_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 Скачать сводку (CSV)",
                data=csv_data.encode("utf-8-sig"),
                file_name="school_comparison_summary.csv",
                mime="text/csv",
                key="school_comp_download_csv"
            )

        # --- Детали анализа ---
        with st.expander(
            f"📝 Использовано признаков: {len(used_columns)}", 
            expanded=False
        ):
            st.write("Колонки классификатора, участвовавшие в анализе:")

            # Группируем по уровням
            by_level: Dict[int, List[str]] = {}
            for col in used_columns:
                level = get_code_depth(col)
                by_level.setdefault(level, []).append(col)

            for level in sorted(by_level.keys()):
                cols = by_level[level]
                st.markdown(f"**Уровень {level}** ({len(cols)} признаков)")
                # Показываем с названиями, если есть
                display_cols = []
                for c in sorted(cols)[:30]:
                    label = classifier_labels.get(c, "")
                    display_cols.append(f"{c}" + (f" ({label})" if label else ""))
                st.code(", ".join(display_cols) + ("..." if len(cols) > 30 else ""))

        # --- Пропущенные профили ---
        if missing_info_all:
            with st.expander(
                "⚠️ Диссертации без тематических профилей", 
                expanded=False
            ):
                st.markdown(
                    "Следующие диссертации не имеют тематических профилей в базе данных:"
                )

                for school_name, missing_df in missing_info_all.items():
                    st.markdown(f"**{school_name}**: {len(missing_df)} диссертаций")
                    if not missing_df.empty and len(missing_df) <= 20:
                        st.dataframe(
                            missing_df, 
                            use_container_width=True, 
                            hide_index=True
                        )
                    elif len(missing_df) > 20:
                        st.dataframe(
                            missing_df.head(10), 
                            use_container_width=True, 
                            hide_index=True
                        )
                        st.caption(f"... и ещё {len(missing_df) - 10} записей")

        # --- Методологическая справка ---
        with st.expander("ℹ️ О методе анализа", expanded=False):
            st.markdown("""
            ### Коэффициент силуэта (Silhouette Score)

            Коэффициент силуэта — метрика качества кластеризации, которая оценивает, 
            насколько объекты похожи на свой кластер по сравнению с другими кластерами.

            **Интерпретация значений:**
            - **0.71 – 1.00**: Отличное разделение
            - **0.51 – 0.70**: Хорошее разделение  
            - **0.26 – 0.50**: Умеренное разделение
            - **0.00 – 0.25**: Слабое разделение
            - **< 0.00**: Плохое разделение (кластеры перемешаны)

            ### Выбор базиса

            **Весь базис** — сравнение по всем тематическим признакам классификатора.

            **Конкретные разделы** — сравнение только по выбранным узлам и всем их 
            подтемам на любой глубине. Например, выбрав узел "1.1", анализ будет 
            включать признаки 1.1, 1.1.1, 1.1.1.1, 1.1.1.2, 1.1.1.2.1 и т.д.

            ### Метрики расстояния

            **Евклидово расстояние** — прямое геометрическое расстояние между точками.

            **Косинусное расстояние** — угол между векторами, не зависит от масштаба.

            **Косоугольный базис** — модификация, учитывающая иерархическую структуру 
            тематического классификатора, где значения родительских узлов частично 
            передаются дочерним.
            """)


# ==============================================================================
# ТОЧКА ВХОДА ДЛЯ ТЕСТИРОВАНИЯ
# ==============================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Тест сравнения школ", layout="wide")
    st.title("🔬 Тестовый режим вкладки сравнения школ")
    st.warning("Этот модуль предназначен для импорта в основное приложение")
