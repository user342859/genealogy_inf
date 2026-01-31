# articles_comparison_tab.py
import streamlit as st
import pandas as pd
import numpy as np
from articles_comparison import (
    load_articles_data, 
    prepare_articles_dataset, 
    calculate_article_metrics,
    to_short_name
)
# Импортируем визуализацию из основного файла (или дублируем, если нужно)
from streamlit_app import make_silhouette_plot, SILHOUETTE_HELP_TEXT, download_data_dialog

def render_articles_comparison_tab(
    df_lineage: pd.DataFrame,
    idx_lineage: dict,
    lineage_func: callable,
    selected_roots: list,
    classifier_labels: dict
):
    st.header("Сравнение научных школ по публикациям")
    
    if len(selected_roots) < 2:
        st.warning("Для сравнения выберите как минимум двух руководителей на вкладке «Построение деревьев».")
        return

    # --- Загрузка базы статей ---
    df_articles = load_articles_data()
    if df_articles.empty:
        st.error("Файл 'articles_scores.csv' не найден или пуст. Сначала сгенерируйте его локальным скриптом.")
        return

    # --- Настройки сравнения ---
    st.subheader("⚙️ Настройки анализа")
    col_cfg1, col_cfg2 = st.columns(2)
    
    with col_cfg1:
        scope = st.radio(
            "Глубина научной школы (статьи каких участников брать):",
            options=["direct", "all"],
            format_func=lambda x: "Только прямые ученики (1-й уровень)" if x == "direct" else "Все поколения учеников",
            help="Учитываются статьи, где в авторах есть хотя бы один представитель школы.",
            key="art_scope"
        )

    with col_cfg2:
        # Формируем список для выбора тематического базиса (1 и 3 уровни + Год)
        # Отфильтруем ключи классификатора для удобства
        level1_3_codes = [code for code in classifier_labels.keys() 
                          if code.count('.') == 0 or code.count('.') == 2]
        
        basis_options = ["Все разделы классификатора", "Year"] + sorted(level1_3_codes)
        
        selected_features = st.multiselect(
            "Тематический базис сравнения:",
            options=basis_options,
            default=["Все разделы классификатора"],
            format_func=lambda x: f"{x} — {classifier_labels.get(x, '')}" if x in classifier_labels else x,
            help="Выберите разделы классификатора или год для математического сопоставления профилей.",
            key="art_basis"
        )

    if st.button("🚀 Провести анализ публикаций", type="primary"):
        # 1. Готовим данные
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
            st.error("Не найдено ни одной статьи для выбранных школ в базе публикаций.")
            return

        # 2. Считаем метрики
        metrics = calculate_article_metrics(dataset, used_features)

        # 3. Визуализация метрик
        st.markdown("---")
        st.subheader("📊 Результаты сравнительного анализа")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.metric("Коэффициент силуэта", f"{metrics['silhouette_avg']:.3f}")
        with m_col2:
            db_val = metrics.get('davies_bouldin')
            st.metric("Индекс Дэвиса–Боулдина", f"{db_val:.3f}" if db_val else "Н/Д", help="Меньше = лучше")
        with m_col3:
            ch_val = metrics.get('calinski_harabasz')
            st.metric("Индекс Калинского–Харабаза", f"{int(ch_val)}" if ch_val else "Н/Д", help="Больше = лучше")
        with m_col4:
            dist = metrics.get('centroids_dist')
            if isinstance(dist, (float, int)):
                st.metric("Расстояние между центроидами", f"{dist:.2f}")
            else:
                st.metric("Расстояние", "См. матрицу ниже")

        # 4. График Силуэта
        st.pyplot(make_silhouette_plot(
            sample_scores=metrics['sample_silhouette_values'],
            labels=pd.factorize(dataset['school'])[0],
            school_order=selected_roots,
            overall_score=metrics['silhouette_avg'],
            metric="Euclidean"
        ))
        
        with st.expander("Как читать этот график?"):
            st.markdown(SILHOUETTE_HELP_TEXT)

        # 5. Матрица расстояний (если школ > 2)
        if len(selected_roots) > 2 and metrics.get('centroids_dist') is not None:
            st.write("**Матрица евклидовых расстояний между центроидами школ:**")
            dist_df = pd.DataFrame(
                metrics['centroids_dist'], 
                index=metrics['cluster_names'], 
                columns=metrics['cluster_names']
            )
            st.dataframe(dist_df.style.background_gradient(cmap='Blues'))

        # 6. Таблица найденных статей
        st.markdown("---")
        st.subheader("📄 Список проанализированных статей")
        
        # Оформляем таблицу: ID, Школа, Авторы, Название, Год
        display_cols = ["Article_id", "school", "Authors", "Title", "Year"] + used_features
        final_table = dataset[display_cols].copy()
        final_table.rename(columns={"school": "Научная школа", "Authors": "Авторы", "Title": "Заголовок"}, inplace=True)
        
        st.dataframe(final_table, use_container_width=True)
        
        if st.button("📥 Скачать таблицу сравнения статей"):
            download_data_dialog(final_table, "articles_comparison_results", "art_comp")
