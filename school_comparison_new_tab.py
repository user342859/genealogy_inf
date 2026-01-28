"""
Streamlit UI для сравнения научных руководителей.
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================================
# ИСПРАВЛЕННЫЕ ИМПОРТЫ (snake_case вместо camelCase)
# ============================================================================
from school_comparison_new import (
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

DEFAULT_SCORES_FOLDER = "basic_scores"
AUTHOR_COLUMN = "candidate.name"

# ==============================================================================
# ИНСТРУКЦИЯ ДЛЯ ВКЛАДКИ
# ==============================================================================

INSTRUCTION_SCHOOL_COMPARISON = """
## 🔬 Сравнение научных школ по тематическим профилям

Этот инструмент позволяет оценить, насколько различаются тематические направления 
диссертаций, защищённых под руководством разных учёных.

---

### 📋 Основные возможности

- **Сравнение тематических профилей** нескольких научных школ
- **Визуализация различий** с помощью графика силуэта
- **Гибкий выбор параметров**: охват диссертаций, метрика расстояния, базис сравнения

---

### 🚀 Как использовать

1. **Выберите научные школы** — укажите минимум 2 руководителей для сравнения
2. **Настройте параметры анализа**:
   - *Охват*: только прямые диссертанты или все поколения
   - *Метрика*: евклидово или косинусное расстояние
   - *Базис*: прямоугольный (стандартный) или косоугольный (учитывающий иерархическую структуру элементов классификатора)
3. **Выберите тематический базис**: весь классификатор или конкретные разделы (например, уровни образования или предметные области)
4. **Запустите анализ** и изучите результаты

---

### 📊 Интерпретация коэффициента силуэта

| Значение | Интерпретация |
|----------|---------------|
| **0.71 – 1.00** | Отличное разделение — школы чётко различаются |
| **0.51 – 0.70** | Хорошее разделение |
| **0.26 – 0.50** | Умеренное разделение — есть пересечения |
| **0.00 – 0.25** | Слабое разделение — школы похожи |
| **< 0.00** | Плохое разделение — у школ общая тематика исследований |

---

### 💡 Рекомендации

- Для **общей картины** используйте весь базис и прямоугольную метрику
- Для **детального анализа** выберите конкретные разделы классификатора
- **Косоугольный базис** учитывает иерархию тем и может дать более точные результаты
- При сравнении **крупных школ** (много поколений) анализ может занять время
"""


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def get_all_supervisors(df: pd.DataFrame) -> List[str]:
    """Извлекает всех научных руководителей из DataFrame."""
    supervisor_cols = [col for col in df.columns 
                      if 'supervisor' in col.lower() and 'name' in col.lower()]
    all_supervisors: Set[str] = set()
    for col in supervisor_cols:
        if col in df.columns:
            all_supervisors.update(
                str(v).strip() for v in df[col].dropna().unique() if str(v).strip()
            )
    return sorted(all_supervisors)


def format_node_option(code: str, classifier_dict: Dict[str, str]) -> str:
    """Форматирует опцию узла для selectbox."""
    depth = get_code_depth(code)  # ✅ исправлено
    indent = "  " * (depth - 1)
    title = classifier_dict.get(code, "")
    if title:
        return f"{indent}{code}: {title}"
    return f"{indent}{code}"


def show_instruction_dialog() -> None:
    """Показывает диалог с инструкциями."""
    @st.dialog("Инструкции", width="large")
    def show():
        st.markdown(INSTRUCTIONS_SCHOOL_COMPARISON)
    show()

# ==============================================================================
# ОСНОВНАЯ ФУНКЦИЯ РЕНДЕРИНГА ВКЛАДКИ
# ==============================================================================

def render_school_comparison_new_tab(
    df: pd.DataFrame,
    idx: Dict[str, Set[int]],
    lineage_func: Callable,
    rows_for_func: Callable,
    scores_folder: str = DEFAULT_SCORES_FOLDER,
    specific_files: Optional[List[str]] = None,
    classifier_labels: Optional[Dict[str, str]] = None,
) -> None:
    """Основная функция рендеринга вкладки."""
    
    if classifier_labels is None:
        classifier_labels = {}
    
    # Кнопка инструкций
    if st.button("📖 Инструкции", key="instruction_school_comparison"):
        show_instruction_dialog()
    
    st.subheader("Сравнение научных школ")
    st.markdown("Анализ различий между научными руководителями...")
    
    # === Загрузка оценок ===
    st.markdown("---")
    st.markdown("### 📊 Загрузка данных")
    
    try:
        scores_df = load_scores_from_folder(  # ✅ исправлено
            folder_path=scores_folder,
            specific_files=specific_files
        )
        all_feature_columns = get_feature_columns(scores_df)  # ✅ исправлено
        st.success(f"✅ Загружено {len(scores_df)} записей, {len(all_feature_columns)} признаков")
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.info(f"Убедитесь, что папка '{scores_folder}' содержит CSV-файлы.")
        return
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {e}")
        return
    
    # === Выбор школ ===
    st.markdown("---")
    st.markdown("### 🎓 Выбор научных руководителей")
    
    all_supervisors_sorted = get_all_supervisors(df)
    if not all_supervisors_sorted:
        st.error("Не найдено ни одного научного руководителя")
        return
    
    selected_schools = st.multiselect(
        "Выберите минимум 2 руководителя для сравнения:",
        options=all_supervisors_sorted,
        default=[],
        key="school_comp_selection",
        help="Нужно минимум 2 школы для сравнения"
    )
    
    if len(selected_schools) < 2:
        st.warning("⚠️ Выберите минимум 2 научных руководителя")
        return
    
    # === Параметры анализа ===
    st.markdown("---")
    st.markdown("### ⚙️ Параметры анализа")
    
    col_params1, col_params2 = st.columns(2)
    
    with col_params1:
        st.markdown("**Метрики и область**")
        
        # Область анализа
        scope_options = list(SCOPE_LABELS.keys())
        scope_labels_list = [SCOPE_LABELS[s] for s in scope_options]
        scope_idx = st.radio(
            "Область анализа:",
            options=range(len(scope_options)),
            format_func=lambda i: scope_labels_list[i],
            key="school_comp_scope",
            help="Прямые ученики или вся линия потомков"
        )
        selected_scope: ComparisonScope = scope_options[scope_idx]
        
        # Метрика расстояния
        st.markdown("**Метрика расстояния**")
        metric_options = list(DISTANCE_METRIC_LABELS.keys())
        metric_labels_list = [DISTANCE_METRIC_LABELS[m] for m in metric_options]
        metric_idx = st.selectbox(
            "Метрика:",
            options=range(len(metric_options)),
            format_func=lambda i: metric_labels_list[i],
            key="school_comp_metric",
            help="Выберите метрику и базис"
        )
        selected_metric: DistanceMetric = metric_options[metric_idx]
    
    with col_params2:
        st.markdown("**Признаки для анализа**")
        
        basis_choice = st.radio(
            "Набор признаков:",
            options=["full", "selected"],
            format_func=lambda x: "Все признаки" if x == "full" else "Выбрать узлы",
            key="school_comp_basis_choice",
            help="Использовать все признаки или выбрать конкретные узлы иерархии"
        )
        
        selected_nodes: Optional[List[str]] = None
        
        if basis_choice == "selected":
            selectable = get_selectable_nodes(all_feature_columns, max_level=3)  # ✅ исправлено
            
            if not selectable:
                st.warning("Нет доступных узлов для выбора")
            else:
                level1_nodes = [n for n in selectable if get_code_depth(n) == 1]  # ✅ исправлено
                level2_nodes = [n for n in selectable if get_code_depth(n) == 2]  # ✅ исправлено
                level3_nodes = [n for n in selectable if get_code_depth(n) == 3]  # ✅ исправлено
                
                st.caption("Выберите узлы иерархии:")
                selected_nodes = []
                
                # Уровень 1
                if level1_nodes:
                    st.markdown("**Уровень 1**")
                    cols_l1 = st.columns(min(4, len(level1_nodes)))
                    for i, node in enumerate(level1_nodes):
                        with cols_l1[i % len(cols_l1)]:
                            label = classifier_labels.get(node, node)
                            if st.checkbox(f"{node}", key=f"node_l1_{node}"):
                                selected_nodes.append(node)
                
                # Уровень 2
                if level2_nodes:
                    with st.expander("Уровень 2", expanded=False):
                        cols_l2 = st.columns(3)
                        for i, node in enumerate(level2_nodes):
                            with cols_l2[i % 3]:
                                label = classifier_labels.get(node, "")
                                display = f"{node}: {label}" if label else f"{node}"
                                if st.checkbox(display, key=f"node_l2_{node}"):
                                    selected_nodes.append(node)
                
                # Уровень 3
                if level3_nodes:
                    with st.expander("Уровень 3", expanded=False):
                        cols_l3 = st.columns(3)
                        for i, node in enumerate(level3_nodes):
                            with cols_l3[i % 3]:
                                label = classifier_labels.get(node, "")
                                display = f"{node}: {label}" if label else f"{node}"
                                if st.checkbox(display, key=f"node_l3_{node}"):
                                    selected_nodes.append(node)
                
                # Показать выбранные узлы
                if selected_nodes:
                    filtered_cols = filter_columns_by_nodes(  # ✅ исправлено
                        all_feature_columns, 
                        selected_nodes
                    )
                    st.info(f"✅ Выбрано {len(selected_nodes)} узлов → {len(filtered_cols)} признаков")
                else:
                    st.warning("⚠️ Не выбрано ни одного узла")
    
    # === Параметр decay_factor для косоугольного базиса ===
    decay_factor = 0.5
    if "oblique" in selected_metric:
        with st.expander("🔧 Настройки косоугольного базиса", expanded=False):
            decay_factor = st.slider(
                "Коэффициент затухания (для N=1):",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1,
                key="school_comp_decay",
                help="Используется когда узел не имеет siblings. Рекомендуется 0.5"
            )
    
    st.markdown("---")
    
    # === Проверка готовности к запуску ===
    ready_to_run = True
    if basis_choice == "selected" and (not selected_nodes or len(selected_nodes) == 0):
        ready_to_run = False
    
    # === Кнопка запуска ===
    if st.button("▶️ Запустить анализ", key="school_comp_run", type="primary", disabled=not ready_to_run):
        
        with st.spinner("Собираем данные..."):
            datasets: Dict[str, pd.DataFrame] = {}
            missing_info_all: Dict[str, pd.DataFrame] = {}
            stats_info = []
            
            progress_bar = st.progress(0)
            for i, school_name in enumerate(selected_schools):
                try:
                    dataset, missing_info, total_count = gather_school_dataset(  # ✅ исправлено
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
                        'Школа': school_name,
                        'Всего': total_count,
                        'Найдено': len(dataset),
                        'Отсутствует': len(missing_info) if not missing_info.empty else 0
                    })
                    
                except Exception as e:
                    st.warning(f"⚠️ {school_name}: {e}")
                
                progress_bar.progress((i + 1) / len(selected_schools))
            
            progress_bar.empty()
        
        # Статистика сбора
        if stats_info:
            st.markdown("**Статистика сбора данных:**")
            stats_df = pd.DataFrame(stats_info)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        valid_datasets = {k: v for k, v in datasets.items() if not v.empty}
        
        # === Силуэтный анализ ===
        if len(valid_datasets) < 2:
            st.error("❌ Недостаточно школ для анализа. Минимум 2 школы.")
            return
        
        with st.spinner("Выполняем силуэтный анализ..."):
            try:
                nodes_for_analysis = selected_nodes if basis_choice == "selected" else None
                
                overall_score, sample_scores, labels, school_order, used_columns = compute_silhouette_analysis(  # ✅ исправлено
                    datasets=valid_datasets,
                    feature_columns=all_feature_columns,
                    metric=selected_metric,
                    selected_nodes=nodes_for_analysis,
                    decay_factor=decay_factor,
                )
                
            except ValueError as e:
                st.error(f"❌ {e}")
                return
            except Exception as e:
                st.error(f"❌ Непредвиденная ошибка: {e}")
                return
        
        # === Результаты ===
        st.markdown("---")
        st.markdown("## 📊 Результаты анализа")
        
        col_score, col_interp = st.columns([1, 2])
        
        with col_score:
            st.metric(
                label="Силуэтный коэффициент",
                value=f"{overall_score:.3f}",
                help="Диапазон: -1 до 1. Чем выше, тем лучше разделение."
            )
        
        with col_interp:
            st.info(interpret_silhouette_score(overall_score))  # ✅ исправлено
        
        basis_info = "Все признаки" if basis_choice == "full" else f"Узлы: {', '.join(selected_nodes or [])}"
        st.caption(f"{basis_info} | {len(used_columns)} признаков | {DISTANCE_METRIC_LABELS[selected_metric]}")
        
        # === График ===
        st.markdown("### 📈 Силуэтный график")
        
        fig = create_silhouette_plot(  # ✅ исправлено
            sample_scores=sample_scores,
            labels=labels,
            school_order=school_order,
            overall_score=overall_score,
            metric_label=DISTANCE_METRIC_LABELS[selected_metric],
        )
        st.pyplot(fig)
        plt.close(fig)
        
        # === Экспорт графика ===
        st.markdown("### 💾 Экспорт результатов")
        
        buf = io.BytesIO()
        fig = create_silhouette_plot(  # ✅ исправлено
            sample_scores=sample_scores,
            labels=labels,
            school_order=school_order,
            overall_score=overall_score,
            metric_label=DISTANCE_METRIC_LABELS[selected_metric],
        )
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
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
        
        # === Сводная таблица ===
        st.markdown("### 📋 Сводная таблица")
        
        summary_df = create_comparison_summary(  # ✅ исправлено
            datasets=valid_datasets,
            feature_columns=used_columns,
            school_order=school_order,
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        with col_dl2:
            csv_data = summary_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Скачать сводку (CSV)",
                data=csv_data.encode('utf-8-sig'),
                file_name="school_comparison_summary.csv",
                mime="text/csv",
                key="school_comp_download_csv"
            )
        
        # === Использованные признаки ===
        with st.expander(f"🔍 Использованные признаки ({len(used_columns)})", expanded=False):
            by_level: Dict[int, List[str]] = {}
            for col in used_columns:
                level = get_code_depth(col)  # ✅ исправлено
                by_level.setdefault(level, []).append(col)
            
            for level in sorted(by_level.keys()):
                cols = by_level[level]
                st.markdown(f"**Уровень {level}** ({len(cols)} признаков)")
                display_cols = []
                for c in sorted(cols)[:30]:
                    label = classifier_labels.get(c, "")
                    display_cols.append(f"{c}: {label}" if label else c)
                st.code(", ".join(display_cols) + ("..." if len(cols) > 30 else ""))
        
        # === Отсутствующие данные ===
        if missing_info_all:
            with st.expander("⚠️ Отсутствующие данные", expanded=False):
                for school_name, missing_df in missing_info_all.items():
                    st.markdown(f"**{school_name}** ({len(missing_df)} записей)")
                    if not missing_df.empty and len(missing_df) <= 20:
                        st.dataframe(missing_df, use_container_width=True, hide_index=True)
                    elif len(missing_df) > 20:
                        st.dataframe(missing_df.head(10), use_container_width=True, hide_index=True)
                        st.caption(f"... и ещё {len(missing_df) - 10} записей")

