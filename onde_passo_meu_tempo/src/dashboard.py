"""dashboard.py — Interface Streamlit com 3 separadores de análise de mobilidade.

Para correr:
    streamlit run src/dashboard.py

Estrutura dos separadores:
  1. Mapa de Mobilidade  — Folium (triplegs + localizações) + MovingPandas HoloViz
  2. Análise Temporal    — Heatmap Seaborn + gráfico de modos de transporte
  3. Resumo Semanal      — 4 métricas em cards + resultados ML

Todos os filtros de privacidade são aplicados ANTES de qualquer visualização.
O pipeline é executado uma única vez e cacheado em st.session_state.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Garantir que o directório raiz do projecto está no sys.path
# (necessário quando o Streamlit é lançado de dentro de src/)
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# set_page_config DEVE ser o primeiro comando Streamlit
st.set_page_config(
    page_title="Onde Passo o Meu Tempo?",
    page_icon="📍",
    layout="wide",
)

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from streamlit_folium import st_folium

# ── Cabeçalho ─────────────────────────────────────────────────────────────────

st.title("Onde Passo o Meu Tempo?")
st.caption(
    "Análise de mobilidade pessoal — MEI Ambient Intelligence, ISEC Coimbra  |  "
    "Ricardo Rodrigues · a2022147797"
)

# ── Executar pipeline (uma vez, cacheado) ─────────────────────────────────────

@st.cache_resource(show_spinner="A executar pipeline de mobilidade… (~1–2 min)")
def carregar_dados():
    from src.pipeline import run_pipeline
    from src.privacy import apply_privacy

    artefactos = run_pipeline()
    pfs_priv, spts_priv = apply_privacy(
        artefactos["pfs"],
        artefactos["spts"],
        artefactos["locs"],
    )
    artefactos["pfs_priv"]  = pfs_priv
    artefactos["spts_priv"] = spts_priv
    return artefactos


@st.cache_resource(show_spinner="A treinar modelos ML… (~30 s)")
def carregar_ml(raw_df):
    from src.ml import extract_features, train_and_evaluate
    features, labels, user_ids = extract_features(raw_df)
    return train_and_evaluate(features, labels, user_ids)


dados     = carregar_dados()
tpls      = dados["tpls"]
locs      = dados["locs"]
spts      = dados["spts_priv"]
pfs       = dados["pfs_priv"]
raw_df    = dados["raw_df"]
tc        = dados["tc"]
trips     = dados["trips"]

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

    # ── Mapa Folium ──────────────────────────────────────────────────────────
    st.subheader("Trajectórias e Localizações")

    # Cores por utilizador
    COR_USER = {1: "#e74c3c", 2: "#2980b9", 3: "#27ae60"}

    # Centro do mapa = centróide dos positionfixes privados
    _pfs_wgs = pfs.to_crs("EPSG:4326") if pfs.crs and pfs.crs.to_epsg() != 4326 else pfs
    _gc = _pfs_wgs.geometry.name
    centro_lat = float(_pfs_wgs[_gc].y.mean())
    centro_lon = float(_pfs_wgs[_gc].x.mean())
    m = folium.Map(location=[centro_lat, centro_lon], zoom_start=13, tiles="CartoDB positron")

    # ── Camada 1: trajectória GPS (subamostrada a cada 10 pontos) ─────────────
    # 45k pontos → ~4.5k; mantém a forma da trajectória sem travar o browser
    fg_pfs = folium.FeatureGroup(name="Trajectória GPS", show=True)
    if "user_id" in _pfs_wgs.columns and "tracked_at" in _pfs_wgs.columns:
        for uid, grp in _pfs_wgs.sort_values("tracked_at").groupby("user_id"):
            grp_sub = grp.iloc[::10]  # 1 em cada 10 pontos
            coords = list(zip(grp_sub[_gc].y, grp_sub[_gc].x))
            if len(coords) < 2:
                continue
            folium.PolyLine(
                coords,
                color=COR_USER.get(int(uid), "#7f8c8d"),
                weight=2,
                opacity=0.45,
                tooltip=f"User {uid} — trajectória GPS",
            ).add_to(fg_pfs)
    fg_pfs.add_to(m)

    # ── Camada 2: triplegs (segmentos de movimento) ───────────────────────────
    fg_tpls = folium.FeatureGroup(name="Triplegs (movimento)", show=True)
    if not tpls.empty:
        tpls_wgs = tpls.to_crs("EPSG:4326") if tpls.crs and tpls.crs.to_epsg() != 4326 else tpls
        geom_col = tpls_wgs.geometry.name
        for _, row in tpls_wgs.iterrows():
            geom = row[geom_col]
            if geom is None or geom.is_empty:
                continue
            try:
                coords = [(y, x) for x, y in geom.coords]
            except Exception:
                continue
            modo = row.get("transport_mode", "—")
            uid  = int(row.get("user_id", 1))
            folium.PolyLine(
                coords,
                color=COR_USER.get(uid, "#7f8c8d"),
                weight=5,
                opacity=0.9,
                tooltip=f"User {uid} · {modo}",
            ).add_to(fg_tpls)
    fg_tpls.add_to(m)

    # ── Camada 3: staypoints agrupados por localização (≤ 93 marcadores) ────────
    # Em vez de 1734 círculos individuais, agregar por location_id → muito mais leve
    fg_spts = folium.FeatureGroup(name="Staypoints", show=True)
    if not spts.empty and hasattr(spts, "geometry") and "location_id" in spts.columns:
        spts_wgs = spts.to_crs("EPSG:4326") if spts.crs and spts.crs.to_epsg() != 4326 else spts
        _sg = spts_wgs.geometry.name
        spts_wgs2 = spts_wgs.copy()
        spts_wgs2["dwell_min"] = (
            (spts_wgs2["finished_at"] - spts_wgs2["started_at"]).dt.total_seconds() / 60
        ).clip(lower=1)
        # Agregar: centróide e dwell total por localização
        agg = spts_wgs2[spts_wgs2["location_id"].notna()].groupby("location_id").agg(
            dwell_total=("dwell_min", "sum"),
            lat=(_sg, lambda g: g.iloc[0].y),
            lon=(_sg, lambda g: g.iloc[0].x),
        )
        max_dwell = agg["dwell_total"].max()
        for _, row in agg.iterrows():
            radius = 5 + 18 * (row["dwell_total"] / max_dwell)
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=radius,
                color="#8e44ad",
                fill=True,
                fill_color="#8e44ad",
                fill_opacity=0.5,
                tooltip=f"Loc {_} · {row['dwell_total']:.0f} min total",
            ).add_to(fg_spts)
    fg_spts.add_to(m)

    # ── Camada 4: localizações (clusters DBSCAN) ──────────────────────────────
    fg_locs = folium.FeatureGroup(name="Localizações (DBSCAN)", show=True)
    locs_plot = locs.set_geometry("center") if "center" in locs.columns else locs
    if locs_plot.crs and locs_plot.crs.to_epsg() != 4326:
        locs_plot = locs_plot.to_crs("EPSG:4326")
    geom_col_locs = locs_plot.geometry.name
    for loc_id, row in locs_plot.iterrows():
        cat  = row.get("fsq_category", "Desconhecido")
        geom = row[geom_col_locs]
        folium.CircleMarker(
            location=[geom.y, geom.x],
            radius=10,
            color="#f39c12",
            fill=True,
            fill_color="#f39c12",
            fill_opacity=0.85,
            tooltip=f"Loc {loc_id} · {cat}",
        ).add_to(fg_locs)
    fg_locs.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legenda
    legenda = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px 14px;border-radius:8px;box-shadow:2px 2px 6px rgba(0,0,0,.3);
                font-size:13px;line-height:1.9">
      <b>Utilizadores</b><br>
      <span style="color:#e74c3c">&#9632;</span> User 1<br>
      <span style="color:#2980b9">&#9632;</span> User 2<br>
      <span style="color:#27ae60">&#9632;</span> User 3<br>
      <hr style="margin:4px 0">
      <span style="color:#8e44ad">&#9679;</span> Staypoints (∝ dwell)<br>
      <span style="color:#f39c12">&#9679;</span> Localização (DBSCAN)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legenda))

    st_folium(m, use_container_width=True, height=550)

    st.divider()

    # ── MovingPandas HoloViz ─────────────────────────────────────────────────
    st.subheader("Trajectórias Animadas (MovingPandas)")
    st.caption("Geração do mapa interactivo pode demorar ~30 s.")
    if st.button("▶ Gerar mapa animado"):
        with st.spinner("A gerar mapa MovingPandas…"):
            try:
                html_str = tc.explore().to_html()
                st.components.v1.html(html_str, height=500, scrolling=False)
            except Exception as e:
                st.warning(f"MovingPandas explore() não disponível: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Separador 2 — Análise Temporal
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Análise Temporal")

    col_esq, col_dir = st.columns(2)

    # ── Heatmap actividade ────────────────────────────────────────────────────
    with col_esq:
        st.subheader("Mapa de Calor de Actividade")

        raw_df2 = raw_df.copy()
        raw_df2["dt"] = pd.to_datetime(raw_df2["time"] // 1000, unit="s", utc=True)
        raw_df2["hora"]      = raw_df2["dt"].dt.hour
        raw_df2["dia_semana"] = raw_df2["dt"].dt.dayofweek  # 0=Seg

        pivot = (
            raw_df2.groupby(["hora", "dia_semana"])
            .size()
            .unstack(fill_value=0)
        )
        # Garantir todas as colunas 0–6
        for d in range(7):
            if d not in pivot.columns:
                pivot[d] = 0
        pivot = pivot[sorted(pivot.columns)]
        pivot.columns = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

        fig_heat, ax_heat = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            pivot,
            ax=ax_heat,
            cmap="YlOrRd",
            linewidths=0.3,
            cbar_kws={"label": "Nº amostras"},
        )
        ax_heat.set_xlabel("Dia da Semana")
        ax_heat.set_ylabel("Hora do Dia (UTC)")
        ax_heat.set_title("Intensidade de actividade por hora × dia")
        plt.tight_layout()
        st.pyplot(fig_heat)
        plt.close(fig_heat)

    # ── Modos de transporte ───────────────────────────────────────────────────
    with col_dir:
        st.subheader("Distribuição de Modos de Transporte")

        if "transport_mode" in tpls.columns and not tpls.empty:
            tpls2 = tpls.copy()
            tpls2["duracao_min"] = (
                (tpls2["finished_at"] - tpls2["started_at"])
                .dt.total_seconds() / 60
            )
            modo_dur = (
                tpls2.groupby("transport_mode")["duracao_min"]
                .sum()
                .reset_index()
                .sort_values("duracao_min", ascending=False)
            )

            CORES_MODO = {
                "mobilidade_lenta":      "#27ae60",
                "mobilidade_motorizada": "#e67e22",
                "mobilidade_rapida":     "#e74c3c",
            }
            cores = [CORES_MODO.get(m, "#95a5a6") for m in modo_dur["transport_mode"]]

            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            ax_bar.barh(modo_dur["transport_mode"], modo_dur["duracao_min"], color=cores)
            ax_bar.set_xlabel("Duração total (minutos)")
            ax_bar.set_title("Tempo por modo de transporte")
            ax_bar.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)
        else:
            st.info("Sem dados de modos de transporte disponíveis.")

    st.divider()

    # ── Distribuição de actividades ───────────────────────────────────────────
    st.subheader("Distribuição de Actividades por Utilizador")

    if "label" in raw_df.columns and "user_id" in raw_df.columns:
        act_counts = (
            raw_df.groupby(["user_id", "label"])
            .size()
            .reset_index(name="count")
        )
        fig_act, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
        for i, uid in enumerate([1, 2, 3]):
            sub = act_counts[act_counts["user_id"] == uid].sort_values("count", ascending=True)
            axes[i].barh(sub["label"], sub["count"], color=COR_USER.get(uid, "#7f8c8d"))
            axes[i].set_title(f"User {uid}")
            axes[i].set_xlabel("Nº amostras")
        plt.suptitle("Actividades por utilizador", y=1.02)
        plt.tight_layout()
        st.pyplot(fig_act)
        plt.close(fig_act)


# ═══════════════════════════════════════════════════════════════════════════════
# Separador 3 — Resumo Semanal
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Resumo Semanal")

    # ── Calcular métricas ─────────────────────────────────────────────────────

    # 1. Localização principal (maior tempo de permanência total)
    loc_principal = "—"
    if not spts.empty and "location_id" in spts.columns:
        spts2 = spts.copy()
        spts2["dwell_s"] = (spts2["finished_at"] - spts2["started_at"]).dt.total_seconds()
        dwell_by_loc = spts2.groupby("location_id")["dwell_s"].sum()
        if not dwell_by_loc.empty:
            best_loc = dwell_by_loc.idxmax()
            if "fsq_category" in locs.columns:
                try:
                    cat = locs.loc[best_loc, "fsq_category"]
                    loc_principal = f"{cat} (loc {best_loc})"
                except Exception:
                    loc_principal = f"Loc {best_loc}"
            else:
                loc_principal = f"Loc {best_loc}"

    # 2. Distância total (triplegs em EPSG:3857 para metros)
    dist_total_km = 0.0
    if not tpls.empty:
        try:
            tpls_proj = tpls.to_crs("EPSG:3857")
            dist_total_km = tpls_proj.geometry.length.sum() / 1000
        except Exception:
            dist_total_km = 0.0

    # 3. Modo dominante
    modo_dominante = "—"
    if "transport_mode" in tpls.columns and not tpls.empty:
        tpls3 = tpls.copy()
        tpls3["dur"] = (tpls3["finished_at"] - tpls3["started_at"]).dt.total_seconds()
        modo_dominante = tpls3.groupby("transport_mode")["dur"].sum().idxmax()

    # 4. Raio de giração (distância RMS ao centróide de todos os staypoints)
    raio_giracao_km = 0.0
    if not spts.empty and hasattr(spts, "geometry"):
        try:
            spts_proj = spts.to_crs("EPSG:3857")
            cx = spts_proj.geometry.x.mean()
            cy = spts_proj.geometry.y.mean()
            dists = np.sqrt(
                (spts_proj.geometry.x - cx) ** 2 + (spts_proj.geometry.y - cy) ** 2
            )
            raio_giracao_km = float(np.sqrt((dists ** 2).mean())) / 1000
        except Exception:
            raio_giracao_km = 0.0

    # ── Cards de métricas ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Localização Principal",
            value=loc_principal,
            help="Local com maior tempo total de permanência (todos os utilizadores).",
        )
    with col2:
        st.metric(
            label="Distância Total (km)",
            value=f"{dist_total_km:.1f} km",
            help="Soma dos comprimentos de todos os triplegs (EPSG:3857).",
        )
    with col3:
        st.metric(
            label="Modo Dominante",
            value=modo_dominante,
            help="Modo de transporte com maior duração acumulada.",
        )
    with col4:
        st.metric(
            label="Raio de Giração (km)",
            value=f"{raio_giracao_km:.2f} km",
            help="Distância RMS ao centróide de todos os staypoints (proxy de dispersão territorial).",
        )

    st.divider()

    # ── Resultados ML ─────────────────────────────────────────────────────────
    st.subheader("Classificação de Actividade (Leave-One-User-Out)")

    # ML é carregado apenas quando o tab3 é aberto (lazy)
    ml_results = carregar_ml(raw_df)

    # Tabela de médias por modelo
    rows = []
    modelos = {}
    for fold in ml_results:
        for nome, metricas in fold["models"].items():
            modelos.setdefault(nome, {"acc": [], "f1": []})
            modelos[nome]["acc"].append(metricas["accuracy"])
            modelos[nome]["f1"].append(metricas["f1_macro"])

    for nome, vals in modelos.items():
        rows.append({
            "Modelo": nome,
            "Acurácia média": f"{np.mean(vals['acc']):.3f}",
            "F1-Macro médio": f"{np.mean(vals['f1']):.3f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Gráfico de barras acurácia por fold
    fig_ml, axes_ml = plt.subplots(1, 2, figsize=(12, 4))

    fold_labels = [f"User {f['test_user']} (teste)" for f in ml_results]  # noqa: F821
    x = np.arange(len(fold_labels))
    width = 0.25

    for i, (nome, vals) in enumerate(list(modelos.items())):
        axes_ml[0].bar(x + i * width, vals["acc"], width, label=nome)
        axes_ml[1].bar(x + i * width, vals["f1"],  width, label=nome)

    for ax, titulo, ylabel in zip(
        axes_ml,
        ["Acurácia por fold", "F1-Macro por fold"],
        ["Acurácia", "F1-Macro"],
    ):
        ax.set_xticks(x + width)
        ax.set_xticklabels(fold_labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(titulo)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    st.pyplot(fig_ml)
    plt.close(fig_ml)

    st.caption(
        "Nota: classes raras (Nightlife, Physical exercise, Shopping) aparecem apenas em 1–2 "
        "utilizadores → F1=0 quando esse utilizador é o conjunto de teste. Comportamento esperado."
    )
