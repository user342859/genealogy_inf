# articles_comparison_tab.py
import streamlit as st
import pandas as pd
import io
from articles_comparison import (
    load_articles_data, 
    prepare_articles_dataset, 
    calculate_article_metrics,
    create_articles_silhouette_plot,
    ARTICLES_HELP_TEXT
)

def render_articles_comparison_tab(
    df_lineage: pd.DataFrame,
    idx_lineage: dict,
    lineage_func: callable,
    selected_roots: list,
    classifier_labels: dict
):
    st.subheader("🔬 Сравнение научных школ по публикациям")
    
    if len(selected_roots) < 2:
        st.warning("⚠️ Выберите минимум двух руководителей на вкладке «Построение деревьев» для сравнения.")
        return

    # Загрузка данных
    df_articles = load_articles_data()
    if df_articles.empty:
        st.error("Файл 'articles_scores.csv' не найден. Запустите скрипт генерации данных.")
        return

    # Настройки UI
    col1, col2 = st.columns(2)
    with col1:
        scope = st.radio(
            "Охват участников:",
            options=["direct", "all"],
            format_func=lambda x: "Только прямые ученики" if x == "direct" else "Все поколения школы",
            key="art_scope_radio"
        )
    with col2:
        level1_3_codes = [c for c in classifier_labels.keys() if c.count('.') in [0, 2]]
        basis_options = ["Все разделы классификатора", "Year"] + sorted(level1_3_codes)
        selected_features = st.multiselect(
            "Тематический базис:",
            options=basis_options,
            default=["Все разделы классификатора"],
            format_func=lambda x: f"{x} — {classifier_labels.get(x, '')}" if x in classifier_labels else x,
            key="art_basis_multi"
        )

    if st.button("🚀 Провести анализ статей", type="primary"):
        with st.spinner("Анализируем публикации..."):
            # 1. Сбор данных
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
                st.error("Статьи для выбранных школ не найдены.")
                return

            # 2. Расчет метрик
            metrics = calculate_article_metrics(dataset, used_features)
            
            # 3. Вывод метрик
            st.markdown("---")
            st.markdown("### 📊 Метрики разделения школ")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Силуэт", f"{metrics['silhouette_avg']:.3f}")
            m2.metric("Дэвис–Боулдин", f"{metrics['davies_bouldin']:.3f}" if metrics['davies_bouldin'] else "—")
            m3.metric("Калински–Харабаз", f"{int(metrics['calinski_harabasz'])}" if metrics['calinski_harabasz'] else "—")
            
            dist = metrics['centroids_dist']
            m4.metric("Дистанция", f"{dist:.2f}" if isinstance(dist, (float, int)) else "Матрица")

            # 4. График
            fig = create_articles_silhouette_plot(
                sample_scores=metrics['sample_silhouette_values'],
                labels=pd.factorize(dataset['school'])[0],
                school_order=metrics['cluster_names'],
                overall_score=metrics['silhouette_avg'],
                metric_name="Euclidean"
            )
            st.pyplot(fig)
            with st.expander("Справка по графику"):
                st.markdown(ARTICLES_HELP_TEXT)

            # 5. Таблица и скачивание
            st.markdown("### 📄 Список публикаций")
            display_df = dataset[["Article_id", "school", "Authors", "Title", "Year"]].copy()
            st.dataframe(display_df, use_container_width=True)

            # Локальная логика скачивания (без импорта из streamlit_app)
            csv_data = dataset.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "📥 Скачать результаты (CSV)",
                data=csv_data,
                file_name="articles_analysis.csv",
                mime="text/csv"
            )
