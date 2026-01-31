"""
Модуль Streamlit-вкладки сравнения научных школ по публикациям.
"""

from __future__ import annotations

import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional, Set

from articles_comparison import (
    DistanceMetric,
    DISTANCE_METRIC_LABELS,
    load_articles_data,
    prepare_articles_dataset,
    compute_article_analysis,
    create_articles_silhouette_plot,
    create_comparison_summary,
    get_code_depth,
    get_selectable_nodes,
    ARTICLES_HELP_TEXT,
    CLASSIFIER_LIST_TEXT
)

# Предполагаем, что общие утилиты вынесены, чтобы избежать цикличности.
# Если нет, функции можно импортировать из streamlit_app или определить локально.
try:
    from shared_utils import download_data_dialog
except ImportError:
    # Локальная версия, если shared_utils не создан
    def download_data_dialog(df: pd.DataFrame, file_base: str, key_prefix: str) -> None:
        @st.dialog(f"Скачать данные: {file_base}")
        def _show_dialog():
            st.write("Выберите формат:")
            col1, col2 = st.columns(2)
            
            # Excel
            buf_xlsx = io.BytesIO()
            with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            st.download_button(
                "📊 Excel (.xlsx)", 
                data=buf_xlsx.getvalue(), 
                file_name=f"{file_base}.xlsx",
                key=f"{key_prefix}_xlsx", use_container_width=True
            )
            
            # CSV
            csv_data = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "📄 CSV (.csv)", 
                data=csv_data, 
                file_name=f"{file_base}.csv",
                key=f"{key_prefix}_csv", use_container_width=True
            )
        _show_dialog()

# ==============================================================================
# ДИАЛОГОВЫЕ ОКНА
# ==============================================================================

def show_articles_instruction():
    @st.dialog("📖 Инструкция: Сравнение по статьям", width="large")
    def _show():
        st.markdown(ARTICLES_HELP_TEXT)
    _show()

def show_classifier_list():
    @st.dialog("🗂 Список классификатора", width="large")
    def _show():
        # Здесь вы можете вставить ваш полный список классификатора
        st.markdown("### Иерархический классификатор тем")
        st.info("Вставьте сюда ваш текст классификатора в файле articles_comparison_tab.py")
        st.text(CLASSIFIER_LIST_TEXT)
    _show()

# ==============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ВКЛАДКИ
# ==============================================================================

def render_articles_comparison_tab(
    df_lineage: pd.DataFrame,
    idx_lineage: Dict[str, Set[int]],
    lineage_func: Callable,
    selected_roots: List[str],
    classifier_labels: Dict[str, str]
):
    # --- Заголовок и Инструкции ---
    st.header("🔬 Сравнение научных школ по публикациям")
    
    c_ins1, c_ins2, _ = st.columns([0.2, 0.25, 0.55])
    with c_ins1:
        if st.button("📖 Инструкция", key="art_ins_btn", use_container_width=True):
            show_articles_instruction()
    with c_ins2:
        if st.button("🗂 Классификатор", key="art_class_btn", use_container_width=True):
            show_classifier_list()

    # --- Проверка данных ---
    df_articles = load_articles_data()
    if df_articles.empty:
        st.error("❌ База статей (articles_scores.csv) не найдена. Запустите скрипт генерации данных.")
        return

    if len(selected_roots) < 2:
        st.warning("⚠️ Для проведения анализа выберите минимум **двух** руководителей на вкладке «Построение деревьев».")
        if selected_roots:
            st.info(f"Текущий выбор: {', '.join(selected_roots)}")
        return

    st.success(f"✅ Готовы к анализу школ: {', '.join(selected_roots)}")
    st.markdown("---")

    # =========================================================================
    # ПАРАМЕТРЫ (UI)
    # =========================================================================
    
    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        st.markdown("### 📐 Параметры анализа")
        
        scope = st.radio(
            "Охват участников школы:",
            options=["direct", "all"],
            format_func=lambda x: "Только прямые ученики (1-й уровень)" if x == "direct" else "Все поколения научной школы",
            key="art_scope_val"
        )

        metric_options = list(DISTANCE_METRIC_LABELS.keys())
        metric_idx = st.selectbox(
            "Метрика расстояния:",
            options=range(len(metric_options)),
            format_func=lambda i: DISTANCE_METRIC_LABELS[metric_options[i]],
            key="art_metric_idx"
        )
        selected_metric: DistanceMetric = metric_options[metric_idx]

        decay_factor = 0.5
        if "oblique" in selected_metric:
            decay_factor = st.slider(
                "Коэффициент затухания (для косоугольного базиса):",
                0.1, 0.9, 0.5, 0.1, help="Влияние иерархических связей классификатора"
            )

    with col_cfg2:
        st.markdown("### 🎯 Тематический базис")
        
        basis_mode = st.radio(
            "Выбор тем для сравнения:",
            options=["full", "custom"],
            format_func=lambda x: "Весь классификатор" if x == "full" else "Выборочные разделы",
            key="art_basis_mode"
        )

        selected_features = []
        if basis_mode == "custom":
            # Извлекаем коды из колонок статей (только те, что есть в данных)
            available_codes = [c for c in df_articles.columns if c[0].isdigit()]
            # Формируем список для выбора (Уровни 1, 2, 3 + Год)
            selectable_nodes = get_selectable_nodes(available_codes, max_level=3)
            
            # Добавляем "Year" -> "Год"
            options = ["Год"] + selectable_nodes
            
            def format_art_node(code):
                if code == "Год": return "📅 Год публикации"
                depth = get_code_depth(code)
                indent = "— " * (depth - 1)
                label = classifier_labels.get(code, "")
                return f"{indent}{code} {label}"

            selected_features = st.multiselect(
                "Выберите разделы классификатора:",
                options=options,
                format_func=format_art_node,
                key="art_custom_features"
            )
            # Переводим "Год" обратно в "Year" для логики
            selected_features = ["Year" if f == "Год" else f for f in selected_features]
        else:
            selected_features = ["Все разделы классификатора"]

    st.markdown("---")

    # =========================================================================
    # ЗАПУСК
    # =========================================================================

    if st.button("🚀 Запустить сравнительный анализ статей", type="primary"):
        with st.spinner("Сбор публикаций и вычисление метрик..."):
            
            # 1. Подготовка датасета
            dataset, final_cols = prepare_articles_dataset(
                roots=selected_roots,
                df_lineage=df_lineage,
                idx_lineage=idx_lineage,
                lineage_func=lineage_func,
                df_articles=df_articles,
                scope=scope,
                selected_features_keys=selected_features if basis_mode == "custom" else None
            )

            if dataset.empty:
                st.error("❌ Статьи не найдены. Попробуйте расширить охват до 'Всех поколений'.")
                return

            # 2. Математический анализ
            results = compute_article_analysis(
                df=dataset,
                feature_columns=final_cols,
                metric=selected_metric,
                decay_factor=decay_factor
            )

            # 3. Отображение результатов
            st.markdown("## 📈 Результаты анализа")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.metric("Коэффициент силуэта", f"{results['silhouette_avg']:.3f}")
                
            with res_col2:
                # Интерпретация (можно взять функцию из school_comparison)
                score = results['silhouette_avg']
                if score > 0.5: interp = "🟢 Высокая степень разделения тематических профилей."
                elif score > 0.2: interp = "🟡 Умеренное разделение тематических профилей."
                else: interp = "🟠 Значительное пересечение тематических профилей научных школ."
                st.info(interp)

            # Дополнительные индексы с описанием
            st.markdown("#### Дополнительные индексы")
            idx_c1, idx_c2, idx_c3 = st.columns(3)
            
            with idx_c1:
                db = results.get('davies_bouldin')
                st.metric("Индекс Дэвиса–Боулдина", f"{db:.3f}" if db else "—")
                st.caption("Оценка компактности кластеров. Меньшее значение указывает на более четкое разделение тематических профилей.")
            
            with idx_c2:
                ch = results.get('calinski_harabasz')
                st.metric("Индекс Калинского–Харабаза", f"{int(ch)}" if ch else "—")
                st.caption("Отношение межкластерной дисперсии к внутрикластерной. Большее значение указывает на более выраженное разделение тематических профилей.")
            
            with idx_c3:
                dist = results.get('centroids_dist')
                dist_str = f"{dist:.2f}" if isinstance(dist, (float, int)) else "См. матрицу"
                st.metric("Дистанция между центрами", dist_str)
                st.caption("Евклидово расстояние между центроидами (средними профилями) школ.")

            # 4. Визуализация
            st.markdown("### 📊 Визуализация (Silhouette Plot)")
            fig = create_articles_silhouette_plot(
                sample_scores=results['sample_silhouette_values'],
                labels=results['labels'],
                school_order=results['school_order'],
                overall_score=results['silhouette_avg'],
                metric_label=DISTANCE_METRIC_LABELS[selected_metric]
            )
            st.pyplot(fig)
            
            # 5. Сводная таблица
            st.markdown("### 📋 Сводная статистика")
            summary_df = create_comparison_summary(dataset, final_cols)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # 6. Список статей
            with st.expander("📄 Просмотреть список проанализированных статей"):
                show_df = dataset[["Article_id", "school", "Authors", "Title", "Year"]].copy()
                show_df.columns = ["ID", "Научная школа", "Авторы", "Заголовок", "Год"]
                st.dataframe(show_df, use_container_width=True)

            # 7. Скачивание
            st.markdown("---")
            if st.button("📥 Скачать результаты (XLSX/CSV)", key="art_dl_final"):
                download_data_dialog(dataset, "articles_comparison_results", "art_res")
