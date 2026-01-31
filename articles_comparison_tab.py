"""
Модуль Streamlit-вкладки сравнения научных школ по статьям.
"""

from __future__ import annotations

import io
import re
import pandas as pd
import streamlit as st
from typing import Callable, Dict, List, Optional, Set

from articles_comparison import (
    DistanceMetric,
    DISTANCE_METRIC_LABELS,
    ARTICLES_HELP_TEXT,
    CLASSIFIER_LIST_TEXT,
    load_articles_data,
    prepare_articles_dataset,
    compute_article_analysis,
    create_articles_silhouette_plot,
    create_comparison_summary,
    get_code_depth
)

# Попытка импорта openpyxl для Excel
try:
    import openpyxl
except ImportError:
    openpyxl = None

# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ОКНА (DIALOGS)
# ==============================================================================

def show_articles_instruction():
    """Показывает инструкцию во всплывающем окне."""
    @st.dialog("📖 Инструкция: Сравнение по статьям", width="large")
    def _show():
        st.markdown(ARTICLES_HELP_TEXT)
    _show()

def show_classifier_list():
    """Показывает список классификатора во всплывающем окне."""
    @st.dialog("🗂 Список тематического классификатора", width="large")
    def _show():
        # Здесь вы можете вставить свой полный список
        st.markdown(CLASSIFIER_LIST_TEXT)
    _show()

def download_articles_results(df: pd.DataFrame, file_base: str):
    """Модальное окно для скачивания результатов (CSV/XLSX)."""
    @st.dialog("📥 Скачать результаты анализа")
    def _show():
        st.write("Выберите формат для сохранения данных:")
        
        # CSV
        csv_data = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📄 Скачать CSV",
            data=csv_data,
            file_name=f"{file_base}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # XLSX
        if openpyxl:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button(
                label="📊 Скачать Excel (XLSX)",
                data=buffer.getvalue(),
                file_name=f"{file_base}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.warning("Установите библиотеку openpyxl для экспорта в Excel.")
    _show()

# ==============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ВКЛАДКИ
# ==============================================================================

def render_articles_comparison_tab(
    # Совместимость: streamlit_app.py вызывает с именованными аргументами
    # df_lineage/idx_lineage (а раньше были df/idx).
    df: Optional[pd.DataFrame] = None,
    idx: Optional[Dict[str, Set[int]]] = None,
    lineage_func: Optional[Callable] = None,
    selected_roots: Optional[List[str]] = None,
    classifier_labels: Optional[Dict[str, str]] = None,
    *,
    df_lineage: Optional[pd.DataFrame] = None,
    idx_lineage: Optional[Dict[str, Set[int]]] = None,
):
    # Кнопки помощи в верхней части
    col_help1, col_help2, _ = st.columns([0.2, 0.25, 0.55])
    with col_help1:
        if st.button("📖 Инструкция", key="art_help_btn"):
            show_articles_instruction()
    with col_help2:
        if st.button("🗂 Список классификатора", key="art_class_btn"):
            show_classifier_list()

    st.header("🔬 Сравнение научных школ по публикациям")

    # --- НОРМАЛИЗАЦИЯ АРГУМЕНТОВ ---
    if df_lineage is None:
        df_lineage = df
    if idx_lineage is None:
        idx_lineage = idx
    if selected_roots is None:
        selected_roots = []
    if classifier_labels is None:
        classifier_labels = {}
    if lineage_func is None:
        st.error("❌ Не передана функция lineage_func")
        return

    # --- ПРОВЕРКИ ---
    if len(selected_roots) < 2:
        st.warning("⚠️ Для проведения сравнения необходимо выбрать **минимум двух** руководителей на вкладке «Построение деревьев» и нажать там кнопку «Построить».")
        return

    if df_lineage is None or idx_lineage is None:
        st.error("❌ Не переданы данные генеалогии (df_lineage/idx_lineage)")
        return

    df_articles = load_articles_data()
    if df_articles.empty:
        st.error("❌ База данных статей (`articles_scores.csv`) не найдена или пуста.")
        return

    st.success(f"Выбраны для анализа: {', '.join(selected_roots)}")

    st.markdown("---")

    # --- НАСТРОЙКИ ПАРАМЕТРОВ ---
    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        st.markdown("### 📐 Параметры анализа")
        
        scope = st.radio(
            "Охват участников школы:",
            options=["direct", "all"],
            format_func=lambda x: "Только прямые ученики (1-й уровень)" if x == "direct" else "Все поколения школы (генеалогия)",
            key="art_scope_choice"
        )

        metric_choice = st.selectbox(
            "Метрика расстояния:",
            options=list(DISTANCE_METRIC_LABELS.keys()),
            format_func=lambda x: DISTANCE_METRIC_LABELS[x],
            key="art_metric_choice"
        )

    with col_cfg2:
        st.markdown("### 🎯 Тематический базис")
        
        # Подготовка списка для выбора (Уровни 1, 2, 3 + Год)
        # Собираем все коды из колонок статей
        all_cols = df_articles.columns.tolist()
        codes_in_df = [c for c in all_cols if re.match(r'^[\d\.]+$', c)]
        
        # Получаем уникальные узлы уровней 1, 2 и 3
        selectable_nodes = []
        for level in [1, 2, 3]:
            nodes = sorted(set(c.rsplit('.', max(0, get_code_depth(c)-level))[0] for c in codes_in_df if get_code_depth(c) >= level))
            selectable_nodes.extend(nodes)
        
        # Убираем дубли и сортируем
        selectable_nodes = sorted(list(set(selectable_nodes)))
        
        basis_options = ["Все разделы классификатора", "Год"] + selectable_nodes
        
        def basis_formatter(x):
            if x == "Все разделы классификатора": return x
            if x == "Год": return x
            label = classifier_labels.get(x, "")
            indent = " " * (get_code_depth(x) - 1) * 2
            return f"{indent}{x} {label}"

        selected_basis = st.multiselect(
            "Выберите разделы для сопоставления:",
            options=basis_options,
            default=["Все разделы классификатора"],
            format_func=basis_formatter,
            key="art_basis_selection"
        )

    # Параметры косоугольного базиса
    decay_factor = 0.5
    if "oblique" in metric_choice:
        decay_factor = st.slider("Коэффициент затухания (decay):", 0.1, 0.9, 0.5, 0.1, help="Влияние иерархических связей")

    st.markdown("---")

    # --- ЗАПУСК ---
    if st.button("🚀 Запустить сравнительный анализ", type="primary"):
        if not selected_basis:
            st.warning("⚠️ Выберите базис для сравнения")
            return

        # Логика выбора базиса:
        # - "Все разделы классификатора" используем как режим 'full' только если это
        #   единственный тематический выбор (иначе игнорируем этот пункт).
        selected_nodes_ui = [
            x for x in selected_basis if x not in ("Все разделы классификатора", "Год")
        ]
        use_all_topics = ("Все разделы классификатора" in selected_basis) and (len(selected_nodes_ui) == 0)
        include_year = ("Год" in selected_basis)

        # Маппинг UI -> логика
        logic_nodes = selected_nodes_ui
        logic_basis: Optional[List[str]]
        if use_all_topics:
            # None → внутри prepare_articles_dataset будет использован весь тематический базис.
            logic_basis = None
        else:
            logic_basis = logic_nodes + (["Year"] if include_year else [])
        
        with st.spinner("Сбор данных и расчет метрик..."):
            dataset, used_features = prepare_articles_dataset(
                roots=selected_roots,
                df_lineage=df_lineage,
                idx_lineage=idx_lineage,
                lineage_func=lineage_func,
                df_articles=df_articles,
                scope=scope,
                selected_features_keys=logic_basis
            )

            # Если выбран режим "весь базис" и дополнительно добавлен год,
            # prepare_articles_dataset вернет только тематические признаки.
            if use_all_topics and include_year and ("Year_num" in dataset.columns):
                if "Year_num" not in used_features:
                    used_features = used_features + ["Year_num"]

            if dataset.empty:
                st.error("❌ По выбранным критериям статьи не найдены.")
                return

            # Минимум 2 школы с данными
            nunique_schools = int(dataset["school"].nunique()) if "school" in dataset.columns else 0
            if nunique_schools < 2:
                st.error(
                    "❌ Недостаточно данных для сравнения: статьи найдены только для одной школы. "
                    "Попробуйте выбрать другой охват (все поколения) или другой набор руководителей."
                )
                # Покажем, что именно нашли
                st.dataframe(
                    dataset[["school", "Article_id", "Authors", "Title", "Year"]]
                    .sort_values(["school", "Year"], ascending=[True, True]),
                    use_container_width=True,
                    hide_index=True,
                )
                return

            if not used_features:
                st.error("❌ Не удалось выбрать признаки для анализа (пустой базис)")
                return

            # Проведение вычислений
            results = compute_article_analysis(dataset, used_features, metric_choice, decay_factor)

            if not results:
                st.error("❌ Анализ не выполнен: нет данных или признаков")
                return

            # --- ВЫВОД РЕЗУЛЬТАТОВ ---
            st.subheader("📊 Результаты сравнительного анализа")
            
            # Метрики в колонках
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric("Коэффициент силуэта", f"{results['silhouette_avg']:.3f}")
                st.caption("Показывает степень разделения тематических профилей школ (от -1 до 1).")
            
            with m2:
                db = results['davies_bouldin']
                st.metric("Индекс Дэвиса–Боулдина", f"{db:.3f}" if db is not None else "—")
                st.caption("Оценка плотности и разделения профилей. Меньшие значения соответствуют более чёткому разделению.")

            with m3:
                ch = results['calinski_harabasz']
                st.metric("Индекс Калинского–Харабаза", f"{int(ch)}" if ch is not None else "—")
                st.caption("Оценка дисперсии профилей. Большие значения соответствуют более чёткому разделению.")

            # График
            st.markdown("### 📈 График силуэта")
            fig = create_articles_silhouette_plot(
                sample_scores=results['sample_silhouette_values'],
                labels=results['labels'],
                school_order=results['school_order'],
                overall_score=results['silhouette_avg'],
                metric_label=DISTANCE_METRIC_LABELS[metric_choice]
            )
            st.pyplot(fig)

            # Расстояние между центроидами
            if len(results['school_order']) == 2:
                st.info(f"**Евклидово расстояние между центроидами школ:** {results['centroids_dist']:.3f}")
            elif len(results['school_order']) > 2:
                with st.expander("Матрица расстояний между центроидами"):
                    dist_df = pd.DataFrame(
                        results['centroids_dist'], 
                        index=results['school_order'], 
                        columns=results['school_order']
                    )
                    st.dataframe(dist_df.style.background_gradient(cmap='YlOrRd'))

            # Сводная таблица
            st.markdown("### 📋 Сводная статистика")
            summary_df = create_comparison_summary(dataset, used_features)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Кнопка скачивания таблицы
            if st.button("📥 Скачать результаты", key="art_dl_btn"):
                download_articles_results(summary_df, "articles_comparison_stats")

            # Список статей
            with st.expander("📄 Список проанализированных статей", expanded=False):
                view_df = dataset[["Article_id", "school", "Authors", "Title", "Year"]].copy()
                view_df.columns = ["ID", "Школа", "Авторы", "Заголовок", "Год"]
                st.dataframe(view_df, use_container_width=True, hide_index=True)
