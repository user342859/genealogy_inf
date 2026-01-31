"""
Модуль Streamlit-вкладки сравнения научных школ по статьям.
Импортируйте и вызывайте render_articles_comparison_tab() в основном приложении.
"""

from __future__ import annotations

import io
import re
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

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
    get_code_depth,
)

# Попытка импорта openpyxl для Excel
try:
    import openpyxl  # type: ignore
except Exception:
    openpyxl = None


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ДИАЛОГИ/ЭКСПОРТ
# ==============================================================================

def _show_articles_instruction() -> None:
    """Показывает инструкцию во всплывающем окне."""
    @st.dialog("📖 Инструкция: Сравнение по статьям", width="large")
    def _dlg():
        st.markdown(ARTICLES_HELP_TEXT)
    _dlg()


def _show_classifier_list() -> None:
    """Показывает список классификатора во всплывающем окне."""
    @st.dialog("🗂 Список тематического классификатора", width="large")
    def _dlg():
        st.markdown(CLASSIFIER_LIST_TEXT)
    _dlg()


def _download_dataframe(df: pd.DataFrame, file_base: str) -> None:
    """Диалог для скачивания результатов (CSV/XLSX)."""
    @st.dialog("📥 Скачать результаты анализа", width="small")
    def _dlg():
        st.write("Выберите формат для сохранения данных:")

        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📄 Скачать CSV",
            data=csv_bytes,
            file_name=f"{file_base}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if openpyxl is None:
            st.warning("Для экспорта в Excel установите пакет `openpyxl`.")
            return

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        st.download_button(
            label="📊 Скачать Excel (XLSX)",
            data=buffer.getvalue(),
            file_name=f"{file_base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    _dlg()


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ UI
# ==============================================================================

_CODE_RE = re.compile(r"^[\d\.]+$")


def _code_sort_key(code: str) -> Tuple[int, ...]:
    """Стабильная сортировка кодов классификатора: '10.2' > '2.9' (по числам, не по строкам)."""
    try:
        return tuple(int(p) for p in code.split(".") if p != "")
    except Exception:
        # fallback: строковая сортировка
        return tuple([10**9])


def _extract_classifier_codes(df_articles: pd.DataFrame) -> List[str]:
    """Берёт все колонки с кодами (цифры/точки) из базы статей."""
    cols = df_articles.columns.tolist()
    codes = [c for c in cols if isinstance(c, str) and _CODE_RE.match(c)]
    return codes


def _build_selectable_nodes(codes: List[str], max_depth: int = 3) -> List[str]:
    """
    Формирует список узлов классификатора для выбора:
    - Уровни 1..max_depth (префиксы кодов).
    """
    nodes: Set[str] = set()
    for c in codes:
        parts = c.split(".")
        depth = min(len(parts), max_depth)
        for d in range(1, depth + 1):
            nodes.add(".".join(parts[:d]))
    return sorted(nodes, key=_code_sort_key)


def _basis_label(code: str, classifier_labels: Dict[str, str]) -> str:
    if code == "__ALL__":
        return "Все разделы классификатора"
    if code == "__YEAR__":
        return "Год"
    label = classifier_labels.get(code, "")
    indent = " " * max(0, (get_code_depth(code) - 1) * 2)
    if label:
        return f"{indent}{code} {label}"
    return f"{indent}{code}"


def _basis_to_feature_keys(
    selected_basis: List[str],
    selectable_nodes: List[str],
) -> Optional[List[str]]:
    """
    Мапит UI-выбор на selected_features_keys для prepare_articles_dataset().

    Логика:
    - Если выбран '__ALL__' без года -> None (все тематические признаки, без Year).
    - Если выбран '__ALL__' + год -> top-level узлы + ['Year'] (все темы + Year).
    - Если выбраны конкретные коды -> эти коды, плюс 'Year' при выборе года.
    - Если выбран только год -> ['Year'].
    """
    include_all = "__ALL__" in selected_basis
    include_year = "__YEAR__" in selected_basis
    chosen_nodes = [x for x in selected_basis if x not in ("__ALL__", "__YEAR__")]

    if include_all:
        if include_year:
            top_level = sorted({n.split(".")[0] for n in selectable_nodes if n}, key=_code_sort_key)
            return top_level + ["Year"]
        # все темы, но без года
        return None

    # Без ALL: либо конкретные узлы, либо только год
    keys: List[str] = []
    keys.extend(chosen_nodes)
    if include_year:
        keys.append("Year")

    if not keys:
        # вообще ничего не выбрано -> по смыслу "все темы" (без года)
        return None

    return keys


# ==============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ВКЛАДКИ
# ==============================================================================

def render_articles_comparison_tab(
    df: Optional[pd.DataFrame] = None,
    idx: Optional[Dict[str, Set[int]]] = None,
    lineage_func: Optional[Callable] = None,
    selected_roots: Optional[List[str]] = None,
    classifier_labels: Optional[Dict[str, str]] = None,
    *,
    # Алиасы под вызов из streamlit_app.py
    df_lineage: Optional[pd.DataFrame] = None,
    idx_lineage: Optional[Dict[str, Set[int]]] = None,
) -> None:
    """
    Рендерит вкладку сравнения школ по статьям.

    Поддерживает два протокола вызова:
    - render_articles_comparison_tab(df=..., idx=..., ...)
    - render_articles_comparison_tab(df_lineage=..., idx_lineage=..., ...)  (как в streamlit_app.py)
    """
    # --- Нормализация входных аргументов (совместимость) ---
    if df is None and df_lineage is not None:
        df = df_lineage
    if idx is None and idx_lineage is not None:
        idx = idx_lineage

    if selected_roots is None:
        selected_roots = []
    if classifier_labels is None:
        classifier_labels = {}

    # --- Вверхние кнопки помощи ---
    c1, c2, _ = st.columns([0.22, 0.28, 0.50])
    with c1:
        if st.button("📖 Инструкция", key="art_help_btn"):
            _show_articles_instruction()
    with c2:
        if st.button("🗂 Список классификатора", key="art_class_btn"):
            _show_classifier_list()

    st.header("🔬 Сравнение научных школ по публикациям")

    # --- Предусловия ---
    if lineage_func is None or df is None or idx is None:
        st.error("❌ Внутренняя ошибка: не переданы данные генеалогии (df/idx/lineage_func).")
        return

    if len(selected_roots) < 2:
        st.warning(
            "⚠️ Для сравнения выберите **минимум двух** руководителей на вкладке «Построение деревьев» "
            "и нажмите там кнопку «Построить»."
        )
        return

    df_articles = load_articles_data()
    if df_articles is None or df_articles.empty:
        st.error("❌ База статей (`articles_scores.csv`) не найдена или пуста.")
        return

    st.success(f"Выбраны для анализа: {', '.join(selected_roots)}")
    st.divider()

    # --- Параметры ---
    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        st.subheader("📐 Параметры анализа")

        scope = st.radio(
            "Охват участников школы:",
            options=["direct", "all"],
            format_func=lambda x: "Только прямые ученики (1-й уровень)" if x == "direct" else "Все поколения школы (генеалогия)",
            key="art_scope_choice",
        )

        metric_choice: DistanceMetric = st.selectbox(
            "Метрика расстояния:",
            options=list(DISTANCE_METRIC_LABELS.keys()),
            format_func=lambda x: DISTANCE_METRIC_LABELS[x],
            key="art_metric_choice",
        )

    with col_cfg2:
        st.subheader("🎯 Тематический базис")

        codes_in_df = _extract_classifier_codes(df_articles)
        selectable_nodes = _build_selectable_nodes(codes_in_df, max_depth=3)

        basis_options = ["__ALL__", "__YEAR__"] + selectable_nodes

        selected_basis = st.multiselect(
            "Выберите разделы для сопоставления:",
            options=basis_options,
            default=["__ALL__"],
            format_func=lambda x: _basis_label(x, classifier_labels),
            key="art_basis_selection",
        )

    decay_factor = 0.5
    if str(metric_choice).endswith("_oblique"):
        decay_factor = st.slider(
            "Коэффициент затухания (decay):",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Влияние иерархических связей (косоугольная метрика).",
            key="art_decay_slider",
        )

    st.divider()

    # --- Запуск ---
    if not st.button("🚀 Запустить сравнительный анализ", type="primary", key="art_run_btn"):
        return

    selected_features_keys = _basis_to_feature_keys(selected_basis, selectable_nodes)

    with st.spinner("Сбор данных и расчёт метрик..."):
        dataset, used_features = prepare_articles_dataset(
            roots=selected_roots,
            df_lineage=df,
            idx_lineage=idx,
            lineage_func=lineage_func,
            df_articles=df_articles,
            scope=scope,
            selected_features_keys=selected_features_keys,
        )

    if dataset is None or dataset.empty:
        st.error("❌ По выбранным критериям статьи не найдены.")
        return

    # --- Диагностика покрытия школ ---
    counts = dataset["school"].value_counts(dropna=False)
    present_schools = [s for s in selected_roots if s in counts.index and counts[s] > 0]

    with st.expander("🔎 Диагностика: сколько статей попало в каждую школу", expanded=False):
        diag_df = pd.DataFrame({"Школа": selected_roots, "Статей в выборке": [int(counts.get(s, 0)) for s in selected_roots]})
        st.dataframe(diag_df, use_container_width=True, hide_index=True)

    if len(present_schools) < 2:
        st.error(
            "❌ Недостаточно данных для сравнения: статьи найдены только для одной школы. "
            "Попробуйте выбрать другой охват (все поколения) или другой набор руководителей."
        )
        with st.expander("📄 Статьи, которые удалось найти (для диагностики)", expanded=False):
            view_df = dataset[["Article_id", "school", "Authors", "Title", "Year"]].copy()
            view_df.columns = ["ID", "Школа", "Авторы", "Заголовок", "Год"]
            st.dataframe(view_df, use_container_width=True, hide_index=True)
        return

    # --- Расчёт метрик ---
    with st.spinner("Расчёт силуэта и прочих метрик..."):
        results = compute_article_analysis(
            dataset=dataset,
            used_features=used_features,
            metric_choice=metric_choice,
            decay_factor=decay_factor,
        )

    st.subheader("📊 Результаты сравнительного анализа")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Коэффициент силуэта", f"{results['silhouette_avg']:.3f}")
        st.caption("Степень разделения тематических профилей школ (от -1 до 1).")
    with m2:
        db = results.get("davies_bouldin")
        st.metric("Индекс Дэвиса–Боулдина", f"{db:.3f}" if isinstance(db, (float, int)) else "—")
        st.caption("Меньшие значения обычно соответствуют более чёткому разделению.")
    with m3:
        ch = results.get("calinski_harabasz")
        st.metric("Индекс Калинского–Харабаза", f"{int(ch)}" if isinstance(ch, (float, int)) else "—")
        st.caption("Большие значения обычно соответствуют более чёткому разделению.")

    st.markdown("### 📈 График силуэта")
    fig = create_articles_silhouette_plot(
        sample_scores=results["sample_silhouette_values"],
        labels=results["labels"],
        school_order=results["school_order"],
        overall_score=results["silhouette_avg"],
        metric_label=DISTANCE_METRIC_LABELS[metric_choice],
    )
    st.pyplot(fig)

    # --- Центроиды ---
    school_order = results.get("school_order", [])
    centroids_dist = results.get("centroids_dist")
    if isinstance(school_order, list) and len(school_order) == 2 and isinstance(centroids_dist, (float, int)):
        st.info(f"**Евклидово расстояние между центроидами школ:** {centroids_dist:.3f}")
    elif isinstance(school_order, list) and len(school_order) > 2 and centroids_dist is not None:
        with st.expander("Матрица расстояний между центроидами", expanded=False):
            dist_df = pd.DataFrame(centroids_dist, index=school_order, columns=school_order)
            st.dataframe(dist_df, use_container_width=True)

    # --- Сводная таблица ---
    st.markdown("### 📋 Сводная статистика")
    summary_df = create_comparison_summary(dataset, used_features)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if st.button("📥 Скачать результаты", key="art_dl_btn"):
        _download_dataframe(summary_df, "articles_comparison_stats")

    with st.expander("📄 Список проанализированных статей", expanded=False):
        view_df = dataset[["Article_id", "school", "Authors", "Title", "Year"]].copy()
        view_df.columns = ["ID", "Школа", "Авторы", "Заголовок", "Год"]
        st.dataframe(view_df, use_container_width=True, hide_index=True)
