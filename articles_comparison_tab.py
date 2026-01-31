# articles_comparison_tab.py
import streamlit as st
import pandas as pd
from articles_comparison import (
    load_articles_data, 
    prepare_articles_dataset, 
    calculate_article_metrics,
    create_articles_silhouette_plot,
    ARTICLES_HELP_TEXT,
    POSSIBLE_PATHS
)

def render_articles_comparison_tab(
    df_lineage: pd.DataFrame,
    idx_lineage: dict,
    lineage_func: callable,
    selected_roots: list,
    classifier_labels: dict
):
    st.header("🔬 Сравнение научных школ по публикациям")

    # 1. Сначала пробуем загрузить данные (чтобы сразу видеть, если файла нет)
    df_articles = load_articles_data()
    if df_articles.empty:
        st.error(f"❌ Файл с данными статей не найден.")
        st.info(f"Ожидался один из файлов: {', '.join(POSSIBLE_PATHS)}. Убедитесь, что вы запустили скрипт генерации `articles_scores.csv`.")
        return

    # 2. Отображаем, кого мы сейчас сравниваем (для отладки)
    if selected_roots:
        st.success(f"**Выбраны для анализа:** {', '.join(selected_roots)}")
    else:
        st.info("Руководители пока не выбраны.")

    # 3. Проверяем количество
    if len(selected_roots) < 2:
        st.warning("⚠️ Для проведения сравнения необходимо выбрать **минимум двух** руководителей на вкладке «Построение деревьев».")
        st.markdown("Пожалуйста, перейдите на первую вкладку, выберите имена и нажмите «Построить деревья», затем вернитесь сюда.")
        return

    # --- Настройки анализа ---
    st.markdown("### ⚙️ Настройки")
    col1, col2 = st.columns(2)
    
    with col1:
        scope = st.radio(
            "Глубина анализа:",
            options=["direct", "all"],
            format_func=lambda x: "Только прямые ученики (1-й уровень)" if x == "direct" else "Вся научная школа (все уровни)",
            key="art_scope_radio"
        )

    with col2:
        # Фильтруем коды для списка (уровень 1 и 3, как вы просили)
        level1_3_codes = [c for c in classifier_labels.keys() if c.count('.') in [0, 2]]
        basis_options = ["Все разделы классификатора", "Year"] + sorted(level1_3_codes)
        
        selected_features = st.multiselect(
            "Тематический базис:",
            options=basis_options,
            default=["Все разделы классификатора"],
            format_func=lambda x: f"{x} — {classifier_labels.get(x, '')}" if x in classifier_labels else x,
            key="art_basis_multi"
        )

    st.markdown("---")

    # --- Кнопка запуска ---
    if st.button("🚀 Провести анализ статей", type="primary"):
        with st.spinner("Ищем статьи авторов и строим профили..."):
            
            # Сбор данных
            dataset, used_features = prepare_articles_dataset(
                roots=selected_roots,
                df_lineage=df_lineage,
                idx_lineage=idx_lineage,
                lineage_func=lineage_func,
                df_articles=df_articles,
                scope=scope,
                selected_features_keys=selected_features
            )

            if dataset.empty:
                st.error("❌ Не найдено ни одной статьи, написанной участниками выбранных научных школ.")
                st.info("Возможно, в базе статей имена записаны иначе, или у этих школ нет публикаций в журнале.")
                return

            # Расчет метрик
            metrics = calculate_article_metrics(dataset, used_features)
            
            # --- Результаты ---
            st.markdown("### 📊 Результаты сравнения")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Силуэт", f"{metrics['silhouette_avg']:.3f}", help="Ближе к 1 = лучше")
            m2.metric("Дэвис–Боулдин", f"{metrics['davies_bouldin']:.3f}" if metrics['davies_bouldin'] else "—", help="Меньше = лучше")
            m3.metric("Калински–Харабаз", f"{int(metrics['calinski_harabasz'])}" if metrics['calinski_harabasz'] else "—", help="Больше = лучше")
            
            dist = metrics['centroids_dist']
            dist_val = f"{dist:.2f}" if isinstance(dist, (float, int)) else "Матрица"
            m4.metric("Дистанция центров", dist_val)

            # График
            fig = create_articles_silhouette_plot(
                sample_scores=metrics['sample_silhouette_values'],
                labels=pd.factorize(dataset['school'])[0],
                school_order=metrics['cluster_names'],
                overall_score=metrics['silhouette_avg'],
                metric_name="Euclidean"
            )
            st.pyplot(fig)
            
            with st.expander("ℹ️ Как читать этот график?"):
                st.markdown(ARTICLES_HELP_TEXT)

            # Таблица данных
            st.markdown("### 📄 Найденные публикации")
            
            # Красивое отображение таблицы
            cols_to_show = ["Article_id", "school", "Authors", "Title", "Year"]
            display_df = dataset[cols_to_show].copy()
            display_df.rename(columns={
                "school": "Научная школа", 
                "Authors": "Авторы", 
                "Title": "Название", 
                "Year": "Год"
            }, inplace=True)
            
            st.dataframe(display_df, use_container_width=True)

            # Скачивание
            csv_data = dataset.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "📥 Скачать данные анализа (CSV)",
                data=csv_data,
                file_name="articles_analysis_results.csv",
                mime="text/csv"
            )
