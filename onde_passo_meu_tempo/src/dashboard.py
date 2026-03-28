"""dashboard.py — Interface Streamlit com 3 separadores de análise de mobilidade.

Para correr:
    streamlit run src/dashboard.py

Estrutura dos separadores:
  1. Mapa de Mobilidade  — Folium + MovingPandas HoloViz
  2. Análise Temporal    — Heatmap Seaborn + gráfico de modos de transporte
  3. Resumo Semanal      — 4 métricas em cards (localização, distância, modo, giração)

Os placeholders st.info() indicam onde conectar as funções do pipeline.
Todos os filtros de privacidade devem ser aplicados ANTES de qualquer visualização.
"""

import streamlit as st

# set_page_config DEVE ser o primeiro comando Streamlit — antes de qualquer outro st.*
st.set_page_config(
    page_title="Onde Passo o Meu Tempo?",
    page_icon="📍",
    layout="wide",
)

# ── Cabeçalho ─────────────────────────────────────────────────────────────────

st.title("Onde Passo o Meu Tempo?")
st.caption(
    "Análise de mobilidade pessoal — MEI Ambient Intelligence, ISEC Coimbra  |  "
    "Ricardo Rodrigues · a2022147797"
)

# ── Separadores ───────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🗺️  Mapa de Mobilidade",
    "📊  Análise Temporal",
    "📋  Resumo Semanal",
])

# ═══════════════════════════════════════════════════════════════════════════════
# Separador 1 — Mapa de Mobilidade
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Mapa de Mobilidade")

    st.info(
        "**[Placeholder — Mapa Folium]**\n\n"
        "Mapa estático com triplegs coloridos por utilizador e marcadores "
        "de localização por categoria Foursquare.\n\n"
        "Passos para activar:\n"
        "1. Correr `artefactos = run_pipeline()` e `apply_privacy(...)`\n"
        "2. Construir o mapa: `m = criar_mapa_folium(artefactos['tpls'], artefactos['locs'])`\n"
        "3. Substituir este bloco por: `st_folium(m, use_container_width=True)`"
    )

    st.divider()

    st.info(
        "**[Placeholder — MovingPandas / HoloViz]**\n\n"
        "Mapa animado de trajectórias com filtros temporais interactivos.\n\n"
        "Passos para activar:\n"
        "1. Obter o TrajectoryCollection: `tc = artefactos['tc']`\n"
        "2. Substituir este bloco por: `st.components.v1.html(tc.explore().to_html(), height=500)`"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Separador 2 — Análise Temporal
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Análise Temporal")

    col_esq, col_dir = st.columns(2)

    with col_esq:
        st.subheader("Mapa de Calor de Actividade")
        st.info(
            "**[Placeholder — Seaborn heatmap]**\n\n"
            "Intensidade de actividade por hora do dia (eixo Y, 0–23 h) "
            "× dia da semana (eixo X, Seg–Dom).\n\n"
            "Passos para activar:\n"
            "1. Extrair features: `feats, labels, uids = extract_features(raw_df)`\n"
            "2. Criar heatmap: `fig = criar_heatmap_actividade(feats, labels)`\n"
            "3. Substituir este bloco por: `st.pyplot(fig)`"
        )

    with col_dir:
        st.subheader("Distribuição de Modos de Transporte")
        st.info(
            "**[Placeholder — Seaborn barplot]**\n\n"
            "Tempo total acumulado por modo de transporte:\n"
            "mobilidade_lenta · mobilidade_motorizada · mobilidade_rapida.\n\n"
            "Passos para activar:\n"
            "1. Obter triplegs: `tpls = artefactos['tpls']`\n"
            "2. Criar gráfico: `fig = criar_grafico_modos(tpls)`\n"
            "3. Substituir este bloco por: `st.pyplot(fig)`"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# Separador 3 — Resumo Semanal
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Resumo Semanal")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Localização Principal",
            value="—",
            help=(
                "Local com maior tempo total de permanência na semana. "
                "Derivado de artefactos['locs'] com a categoria Foursquare mais frequente."
            ),
        )

    with col2:
        st.metric(
            label="Distância Total (km)",
            value="—",
            help=(
                "Soma das distâncias de todos os triplegs na semana. "
                "Calculado a partir de artefactos['tpls'].geometry.length."
            ),
        )

    with col3:
        st.metric(
            label="Modo Dominante",
            value="—",
            help=(
                "Modo de transporte com maior duração acumulada. "
                "Derivado da coluna transport_mode de artefactos['tpls']."
            ),
        )

    with col4:
        st.metric(
            label="Raio de Giração (km)",
            value="—",
            help=(
                "Medida de dispersão do território (Gonzalez et al., 2008). "
                "Calculado com trackintel.analysis.metrics.radius_gyration(spts)."
            ),
        )

    st.divider()
    st.caption(
        "Nota: todos os valores acima requerem a execução prévia de `run_pipeline()` "
        "e `apply_privacy()` com os dados reais."
    )
