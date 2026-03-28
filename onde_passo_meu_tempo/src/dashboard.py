"""dashboard.py — Interface Streamlit com análise de mobilidade por utilizador.

Para correr:
    streamlit run src/dashboard.py

Separadores:
  1. Mapa de Mobilidade  — Folium (trajectórias + localizações)
  2. Análise Temporal    — Heatmap + modos de transporte + actividades
  3. Resumo Semanal      — 4 métricas + ML 80/20 para o utilizador seleccionado
"""

import sys
import io
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

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

# ── Pipeline (cacheado) ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="A executar pipeline de mobilidade… (~1–2 min)")
def carregar_dados():
    from src.pipeline import run_pipeline
    from src.privacy import apply_privacy

    art = run_pipeline()
    pfs_priv, spts_priv = apply_privacy(art["pfs"], art["spts"], art["locs"])
    art["pfs_priv"]  = pfs_priv
    art["spts_priv"] = spts_priv
    return art


@st.cache_resource(show_spinner="A treinar modelos ML… (~30 s)")
def carregar_ml_louo(_raw_df):
    """Leave-one-user-out — cacheado uma vez para todos os utilizadores."""
    from src.ml import extract_features, train_and_evaluate
    features, labels, user_ids = extract_features(_raw_df)
    return train_and_evaluate(features, labels, user_ids), features, labels, user_ids


dados  = carregar_dados()
raw_df = dados["raw_df"]
tc     = dados["tc"]

# ── Sidebar — selecção de utilizador ─────────────────────────────────────────

COR_USER = {1: "#e74c3c", 2: "#2980b9", 3: "#27ae60"}

with st.sidebar:
    st.header("🔍 Filtros")
    opcoes = {"Todos os utilizadores": 0, "User 1": 1, "User 2": 2, "User 3": 3}
    sel_label = st.selectbox("Utilizador", list(opcoes.keys()))
    sel_uid   = opcoes[sel_label]

    st.divider()
    st.caption(
        "Seleccionar um utilizador filtra o mapa, os gráficos temporais e "
        "activa a análise ML 80/20 para esse utilizador."
    )

# ── Filtrar artefactos pelo utilizador seleccionado ───────────────────────────

def _filtrar(gdf, uid):
    if uid == 0 or "user_id" not in gdf.columns:
        return gdf
    return gdf[gdf["user_id"] == uid]


pfs  = _filtrar(dados["pfs_priv"],  sel_uid)
spts = _filtrar(dados["spts_priv"], sel_uid)
tpls = _filtrar(dados["tpls"],      sel_uid)
locs = dados["locs"]  # localizações são globais (DBSCAN agg_level='dataset')

# ── Separadores ───────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️  Mapa de Mobilidade",
    "📊  Análise Temporal",
    "🔬  Padrões de Comportamento",
    "📋  Resumo Semanal",
])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Mapa de Mobilidade
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Mapa de Mobilidade")
    st.subheader("Trajectórias e Localizações")

    # Usar todos os pfs para o mapa quando não há dados filtrados
    pfs_mapa = pfs if not pfs.empty else dados["pfs_priv"]

    if pfs_mapa.empty:
        st.warning("Sem dados para o utilizador seleccionado.")
    else:
        _pfs_wgs = pfs_mapa.to_crs("EPSG:4326") if pfs_mapa.crs else pfs_mapa
        _gc = _pfs_wgs.geometry.name
        centro_lat = float(_pfs_wgs[_gc].y.mean())
        centro_lon = float(_pfs_wgs[_gc].x.mean())
        m = folium.Map(location=[centro_lat, centro_lon], zoom_start=13,
                       tiles="CartoDB positron")

        # Camada 1 — trajectória GPS (1 em cada 10)
        fg_pfs = folium.FeatureGroup(name="Trajectória GPS", show=True)
        if "tracked_at" in _pfs_wgs.columns:
            for uid, grp in _pfs_wgs.sort_values("tracked_at").groupby("user_id"):
                sub = grp.iloc[::10]
                coords = list(zip(sub[_gc].y, sub[_gc].x))
                if len(coords) < 2:
                    continue
                folium.PolyLine(
                    coords, color=COR_USER.get(int(uid), "#7f8c8d"),
                    weight=2, opacity=0.45,
                    tooltip=f"User {uid} — GPS",
                ).add_to(fg_pfs)
        fg_pfs.add_to(m)

        # Camada 2 — triplegs
        fg_tpls = folium.FeatureGroup(name="Triplegs (movimento)", show=True)
        if not tpls.empty:
            tpls_wgs = tpls.to_crs("EPSG:4326") if tpls.crs and tpls.crs.to_epsg() != 4326 else tpls
            gcol = tpls_wgs.geometry.name
            for _, row in tpls_wgs.iterrows():
                geom = row[gcol]
                if geom is None or geom.is_empty:
                    continue
                try:
                    coords = [(y, x) for x, y in geom.coords]
                except Exception:
                    continue
                uid  = int(row.get("user_id", 1))
                modo = row.get("transport_mode", "—")
                folium.PolyLine(
                    coords, color=COR_USER.get(uid, "#7f8c8d"),
                    weight=5, opacity=0.9,
                    tooltip=f"User {uid} · {modo}",
                ).add_to(fg_tpls)
        fg_tpls.add_to(m)

        # Camada 3 — staypoints agregados por localização
        fg_spts = folium.FeatureGroup(name="Staypoints", show=True)
        if not spts.empty and "location_id" in spts.columns:
            spts_wgs = spts.to_crs("EPSG:4326") if spts.crs and spts.crs.to_epsg() != 4326 else spts
            sg = spts_wgs.geometry.name
            spts_wgs2 = spts_wgs.copy()
            spts_wgs2["dwell_min"] = (
                (spts_wgs2["finished_at"] - spts_wgs2["started_at"]).dt.total_seconds() / 60
            ).clip(lower=1)
            agg = (
                spts_wgs2[spts_wgs2["location_id"].notna()]
                .groupby("location_id")
                .agg(
                    dwell_total=("dwell_min", "sum"),
                    lat=(sg, lambda g: g.iloc[0].y),
                    lon=(sg, lambda g: g.iloc[0].x),
                )
            )
            if not agg.empty:
                mx = agg["dwell_total"].max()
                for lid, row in agg.iterrows():
                    r = 5 + 18 * (row["dwell_total"] / mx)
                    folium.CircleMarker(
                        location=[row["lat"], row["lon"]],
                        radius=r, color="#8e44ad", fill=True,
                        fill_color="#8e44ad", fill_opacity=0.5,
                        tooltip=f"Loc {lid} · {row['dwell_total']:.0f} min",
                    ).add_to(fg_spts)
        fg_spts.add_to(m)

        # Camada 4 — localizações DBSCAN
        fg_locs = folium.FeatureGroup(name="Localizações (DBSCAN)", show=True)
        locs_plot = locs.set_geometry("center") if "center" in locs.columns else locs
        if locs_plot.crs and locs_plot.crs.to_epsg() != 4326:
            locs_plot = locs_plot.to_crs("EPSG:4326")
        gloc = locs_plot.geometry.name
        for lid, row in locs_plot.iterrows():
            cat  = row.get("fsq_category", "Desconhecido")
            geom = row[gloc]
            folium.CircleMarker(
                location=[geom.y, geom.x], radius=10,
                color="#f39c12", fill=True, fill_color="#f39c12", fill_opacity=0.85,
                tooltip=f"Loc {lid} · {cat}",
            ).add_to(fg_locs)
        fg_locs.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

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
          <span style="color:#f39c12">&#9679;</span> Localizações (DBSCAN)
        </div>
        """
        m.get_root().html.add_child(folium.Element(legenda))
        st_folium(m, use_container_width=True, height=550)

    st.divider()

    # MovingPandas — persistido em session_state para sobreviver a reruns
    st.subheader("Trajectórias Animadas (MovingPandas)")
    st.caption("Geração pode demorar ~30 s.")

    if "mpd_html" not in st.session_state:
        st.session_state["mpd_html"] = None

    if st.button("▶ Gerar mapa animado"):
        with st.spinner("A gerar mapa MovingPandas…"):
            try:
                buf = io.BytesIO()
                tc.explore().save(buf, close_file=False)
                st.session_state["mpd_html"] = buf.getvalue().decode("utf-8")
            except Exception as e:
                st.warning(f"MovingPandas explore() não disponível: {e}")

    if st.session_state["mpd_html"]:
        st.components.v1.html(st.session_state["mpd_html"], height=500)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Análise Temporal
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Análise Temporal")

    # Filtrar raw_df pelo utilizador seleccionado
    rdf = raw_df if sel_uid == 0 else raw_df[raw_df["user_id"] == sel_uid]

    col_esq, col_dir = st.columns(2)

    # ── Heatmap actividade ────────────────────────────────────────────────────
    with col_esq:
        st.subheader("Mapa de Calor de Actividade")
        rdf2 = rdf.copy()
        rdf2["dt"]        = pd.to_datetime(rdf2["time"] // 1000, unit="s", utc=True)
        rdf2["hora"]      = rdf2["dt"].dt.hour
        rdf2["dia_semana"] = rdf2["dt"].dt.dayofweek

        pivot = rdf2.groupby(["hora", "dia_semana"]).size().unstack(fill_value=0)
        for d in range(7):
            if d not in pivot.columns:
                pivot[d] = 0
        pivot = pivot[sorted(pivot.columns)]
        pivot.columns = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3,
                    cbar_kws={"label": "Nº amostras"})
        ax.set_xlabel("Dia da Semana")
        ax.set_ylabel("Hora do Dia (UTC)")
        titulo_uid = f"User {sel_uid}" if sel_uid != 0 else "Todos"
        ax.set_title(f"Intensidade de actividade — {titulo_uid}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Modos de transporte ───────────────────────────────────────────────────
    with col_dir:
        st.subheader("Distribuição de Modos de Transporte")
        if "transport_mode" in tpls.columns and not tpls.empty:
            tpls2 = tpls.copy()
            tpls2["dur_min"] = (
                (tpls2["finished_at"] - tpls2["started_at"]).dt.total_seconds() / 60
            )
            modo_dur = (
                tpls2.groupby("transport_mode")["dur_min"].sum()
                .reset_index().sort_values("dur_min", ascending=True)
            )
            CORES_MODO = {
                "mobilidade_lenta":      "#27ae60",
                "mobilidade_motorizada": "#e67e22",
                "mobilidade_rapida":     "#e74c3c",
            }
            cores = [CORES_MODO.get(m, "#95a5a6") for m in modo_dur["transport_mode"]]
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.barh(modo_dur["transport_mode"], modo_dur["dur_min"], color=cores)
            ax2.set_xlabel("Duração total (minutos)")
            ax2.set_title(f"Tempo por modo — {titulo_uid}")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("Sem triplegs para o utilizador seleccionado.")

    st.divider()

    # ── Distribuição de actividades ───────────────────────────────────────────
    st.subheader("Distribuição de Actividades")
    if "label" in rdf.columns:
        if sel_uid == 0:
            # todos: 3 subgráficos lado a lado
            fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
            for i, uid in enumerate([1, 2, 3]):
                sub = rdf[rdf["user_id"] == uid]["label"].value_counts().sort_values()
                axes3[i].barh(sub.index, sub.values, color=COR_USER.get(uid, "#7f8c8d"))
                axes3[i].set_title(f"User {uid}")
                axes3[i].set_xlabel("Nº amostras")
            plt.suptitle("Actividades por utilizador", y=1.02)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            # utilizador único: gráfico simples
            sub = rdf["label"].value_counts().sort_values()
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            ax3.barh(sub.index, sub.values, color=COR_USER.get(sel_uid, "#7f8c8d"))
            ax3.set_xlabel("Nº amostras")
            ax3.set_title(f"Actividades — User {sel_uid}")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Resumo Semanal
# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Padrões de Comportamento
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Padrões de Comportamento")

    COR_ACT = {
        "Sleep": "#2c3e50", "Home": "#2980b9", "Working": "#8e44ad",
        "Free time": "#27ae60", "Launch Break": "#f39c12", "Break": "#e67e22",
        "Restaurant": "#e74c3c", "Nightlife": "#c0392b",
        "Physical exercise": "#1abc9c", "Shopping": "#d35400",
    }

    # Preparar todos os dados com colunas temporais
    rdf_all = raw_df.copy()
    rdf_all["dt"]       = pd.to_datetime(rdf_all["time"] // 1000, unit="s", utc=True)
    rdf_all["hour"]     = rdf_all["dt"].dt.hour
    rdf_all["dow"]      = rdf_all["dt"].dt.dayofweek
    rdf_all["date"]     = rdf_all["dt"].dt.date
    rdf_all["week_lbl"] = rdf_all["dt"].dt.strftime("Semana %Y-W%V")
    rdf_all["accel"]    = np.sqrt(
        rdf_all["sensor_linear_acc_x_mean"]**2
        + rdf_all["sensor_linear_acc_y_mean"]**2
        + rdf_all["sensor_linear_acc_z_mean"]**2
    )

    # Filtrar por utilizador (sidebar)
    rdf_base = rdf_all if sel_uid == 0 else rdf_all[rdf_all["user_id"] == sel_uid]

    # ── Selector de semana ────────────────────────────────────────────────────
    semanas_disp = sorted(rdf_base["week_lbl"].unique())
    col_sw1, col_sw2 = st.columns([3, 1])
    with col_sw1:
        semana_sel = st.select_slider(
            "Semana analisada",
            options=semanas_disp,
            value=semanas_disp[-1],
            help="Filtra a timeline e a rotina semanal para a semana seleccionada. "
                 "Aceleração, WiFi e duração usam sempre todos os dados disponíveis.",
        )
    with col_sw2:
        st.metric("Semanas disponíveis", len(semanas_disp))

    rdf_b = rdf_base[rdf_base["week_lbl"] == semana_sel].copy()
    # Fallback: se semana sem dados suficientes, usar todos
    if len(rdf_b) < 10:
        rdf_b = rdf_base.copy()

    st.divider()

    # ── 1. Timeline de actividades da semana seleccionada ────────────────────
    st.subheader(f"1. Timeline de Actividades — {semana_sel}")

    dates_sorted  = sorted(rdf_b["date"].unique())
    tl            = rdf_b.copy()
    tl["min_of_day"] = tl["hour"] * 60 + tl["dt"].dt.minute

    n_days = len(dates_sorted)
    fig_tl, ax_tl = plt.subplots(figsize=(14, max(2, n_days * 0.55)))
    for i, d in enumerate(dates_sorted):
        day_data = tl[tl["date"] == d]
        for _, row in day_data.iterrows():
            cor = COR_ACT.get(row["label"], "#95a5a6")
            ax_tl.barh(i, 1, left=row["min_of_day"], color=cor, height=0.7)
    ax_tl.set_yticks(range(n_days))
    ax_tl.set_yticklabels([
        pd.Timestamp(d).strftime("%a %d/%m") for d in dates_sorted
    ])
    ax_tl.set_xlabel("Minutos desde meia-noite")
    ax_tl.set_xlim(0, 1440)
    ax_tl.set_xticks(range(0, 1441, 120))
    ax_tl.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], fontsize=8)
    ax_tl.set_title("Sequência de actividades por dia")
    # Legenda compacta
    handles = [plt.Rectangle((0,0),1,1, color=COR_ACT.get(l,"#95a5a6"))
               for l in rdf_b["label"].unique() if l in COR_ACT]
    labels_leg = [l for l in rdf_b["label"].unique() if l in COR_ACT]
    ax_tl.legend(handles, labels_leg, loc="upper right", fontsize=7,
                 ncol=3, framealpha=0.7)
    plt.tight_layout()
    st.pyplot(fig_tl)
    plt.close(fig_tl)

    st.divider()

    # ── 2. Duração média por actividade (boxplot) — todos os dados do utilizador
    st.subheader("2. Duração Média por Actividade (todos os dados)")

    rdf_b2 = rdf_base.sort_values(["user_id", "dt"]).copy()
    rdf_b2["run"] = (
        (rdf_b2["label"] != rdf_b2["label"].shift()) |
        (rdf_b2["user_id"] != rdf_b2["user_id"].shift())
    )
    rdf_b2["run_id"] = rdf_b2["run"].cumsum()
    runs = rdf_b2.groupby(["run_id", "label"]).size().reset_index(name="n_min")

    ordem = runs.groupby("label")["n_min"].median().sort_values(ascending=False).index
    cores_box = [COR_ACT.get(l, "#95a5a6") for l in ordem]

    fig_box, ax_box = plt.subplots(figsize=(12, 4))
    data_box = [runs[runs["label"] == l]["n_min"].values for l in ordem]
    bp = ax_box.boxplot(data_box, patch_artist=True, showfliers=False,
                        medianprops={"color": "white", "linewidth": 2})
    for patch, cor in zip(bp["boxes"], cores_box):
        patch.set_facecolor(cor)
        patch.set_alpha(0.8)
    ax_box.set_xticklabels(ordem, rotation=30, ha="right", fontsize=9)
    ax_box.set_ylabel("Duração contínua (minutos)")
    ax_box.set_title("Duração típica de cada actividade (sem outliers extremos)")
    plt.tight_layout()
    st.pyplot(fig_box)
    plt.close(fig_box)

    st.divider()

    # ── 3. Perfil sensor por actividade — todos os dados do utilizador ────────
    st.subheader("3. Perfil de Sensores por Actividade (todos os dados)")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        # Aceleração média por actividade
        accel_act = rdf_base.groupby("label")["accel"].mean().sort_values(ascending=True)
        fig_ac, ax_ac = plt.subplots(figsize=(6, 4))
        cores_ac = [COR_ACT.get(l, "#95a5a6") for l in accel_act.index]
        ax_ac.barh(accel_act.index, accel_act.values, color=cores_ac)
        ax_ac.set_xlabel("Aceleração linear média (m/s²)")
        ax_ac.set_title("Aceleração por actividade")
        ax_ac.axvline(accel_act.mean(), color="grey", linestyle="--", alpha=0.6,
                      label=f"Média geral: {accel_act.mean():.3f}")
        ax_ac.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_ac)
        plt.close(fig_ac)

    with col_s2:
        # WiFi vs Bateria por actividade (radar / grouped bar)
        wifi_pct  = rdf_base.groupby("label")["wifi_connected"].mean() * 100
        batt_pct  = rdf_base.groupby("label")["battery_unplugged"].mean() * 100
        labels_s  = sorted(set(wifi_pct.index) & set(batt_pct.index))
        x_s       = np.arange(len(labels_s))
        w_s       = 0.38

        fig_wb, ax_wb = plt.subplots(figsize=(7, 4))
        ax_wb.bar(x_s - w_s/2, [wifi_pct[l] for l in labels_s],
                  w_s, label="WiFi ligado (%)", color="#3498db", alpha=0.85)
        ax_wb.bar(x_s + w_s/2, [batt_pct[l] for l in labels_s],
                  w_s, label="Sem carregar (%)", color="#e74c3c", alpha=0.85)
        ax_wb.set_xticks(x_s)
        ax_wb.set_xticklabels(labels_s, rotation=35, ha="right", fontsize=8)
        ax_wb.set_ylabel("% amostras")
        ax_wb.set_title("WiFi e bateria por actividade")
        ax_wb.legend(fontsize=8)
        ax_wb.set_ylim(0, 110)
        plt.tight_layout()
        st.pyplot(fig_wb)
        plt.close(fig_wb)

    st.divider()

    # ── 4. Rotina semanal — % tempo por actividade e dia da semana ───────────
    st.subheader(f"4. Rotina Semanal — {semana_sel}")

    pivot_rot = (
        rdf_b.groupby(["dow", "label"])
        .size()
        .unstack(fill_value=0)
    )
    pivot_rot = pivot_rot.div(pivot_rot.sum(axis=1), axis=0) * 100
    pivot_rot.index = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"][: len(pivot_rot)]

    fig_rot, ax_rot = plt.subplots(figsize=(12, 4))
    bottom = np.zeros(len(pivot_rot))
    for col in pivot_rot.columns:
        vals = pivot_rot[col].values
        ax_rot.bar(pivot_rot.index, vals, bottom=bottom,
                   color=COR_ACT.get(col, "#95a5a6"), label=col, alpha=0.9)
        bottom += vals
    ax_rot.set_ylabel("% do tempo")
    ax_rot.set_title("Composição de actividades por dia da semana")
    ax_rot.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.7)
    ax_rot.set_ylim(0, 100)
    plt.tight_layout()
    st.pyplot(fig_rot)
    plt.close(fig_rot)

    st.divider()

    # ── 5. Hora típica de ocorrência — todos os dados do utilizador ──────────
    st.subheader("5. Hora Típica de Ocorrência (todos os dados)")

    hora_med = rdf_base.groupby("label")["hour"].apply(
        lambda x: np.average(x, weights=np.ones(len(x)))
    ).sort_values()

    fig_hora, ax_hora = plt.subplots(figsize=(10, 3), subplot_kw={"polar": False})
    cores_h = [COR_ACT.get(l, "#95a5a6") for l in hora_med.index]
    bars_h  = ax_hora.barh(hora_med.index, hora_med.values, color=cores_h, alpha=0.85)
    ax_hora.set_xlabel("Hora média UTC")
    ax_hora.set_xlim(0, 24)
    ax_hora.set_xticks(range(0, 25, 2))
    ax_hora.set_xticklabels([f"{h:02d}h" for h in range(0, 25, 2)])
    ax_hora.set_title("Hora média de ocorrência de cada actividade")
    for bar, val in zip(bars_h, hora_med.values):
        ax_hora.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}h", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_hora)
    plt.close(fig_hora)

    st.divider()

    # ── 6. Evolução semanal — % de actividades por semana ─────────────────────
    st.subheader("6. Evolução Semanal das Actividades")

    pivot_evo = (
        rdf_base.groupby(["week_lbl", "label"])
        .size()
        .unstack(fill_value=0)
    )
    pivot_evo = pivot_evo.div(pivot_evo.sum(axis=1), axis=0) * 100

    fig_evo, ax_evo = plt.subplots(figsize=(12, 4))
    bottom_e = np.zeros(len(pivot_evo))
    for col in pivot_evo.columns:
        ax_evo.bar(pivot_evo.index, pivot_evo[col].values, bottom=bottom_e,
                   color=COR_ACT.get(col, "#95a5a6"), label=col, alpha=0.9)
        bottom_e += pivot_evo[col].values

    # Marcar a semana seleccionada
    if semana_sel in list(pivot_evo.index):
        idx_sel = list(pivot_evo.index).index(semana_sel)
        ax_evo.axvline(idx_sel, color="black", linewidth=2, linestyle="--",
                       alpha=0.7, label=f"← {semana_sel}")

    ax_evo.set_ylabel("% do tempo")
    ax_evo.set_title("Composição de actividades por semana")
    ax_evo.set_xticklabels(pivot_evo.index, rotation=20, ha="right", fontsize=9)
    ax_evo.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.7)
    ax_evo.set_ylim(0, 100)
    plt.tight_layout()
    st.pyplot(fig_evo)
    plt.close(fig_evo)

    st.divider()

    # ── 7. Comparação entre utilizadores ──────────────────────────────────────
    if sel_uid == 0:
        st.subheader("7. Comparação entre Utilizadores")

        from matplotlib.patches import FancyArrowPatch
        metricas_users = []
        for uid in [1, 2, 3]:
            u = raw_df[raw_df["user_id"] == uid].copy()
            u["dt"]    = pd.to_datetime(u["time"]//1000, unit="s", utc=True)
            u["accel"] = np.sqrt(u["sensor_linear_acc_x_mean"]**2
                                 + u["sensor_linear_acc_y_mean"]**2
                                 + u["sensor_linear_acc_z_mean"]**2)
            n_locs     = dados["spts"][dados["spts"]["user_id"]==uid]["location_id"].nunique()
            pct_sleep  = (u["label"]=="Sleep").mean()*100
            pct_work   = (u["label"]=="Working").mean()*100
            pct_home   = (u["label"]=="Home").mean()*100
            accel_mean = u["accel"].mean()
            wifi_pct   = u["wifi_connected"].mean()*100
            metricas_users.append({
                "User": f"User {uid}",
                "% Sleep": round(pct_sleep,1),
                "% Working": round(pct_work,1),
                "% Home": round(pct_home,1),
                "Aceleração\nmédia": round(accel_mean,3),
                "% WiFi\nligado": round(wifi_pct,1),
                "Locais\nvisitados": n_locs,
            })

        df_comp = pd.DataFrame(metricas_users).set_index("User")
        st.dataframe(df_comp, use_container_width=True)

        # Barras agrupadas por métrica
        metricas_plot = ["% Sleep", "% Working", "% Home", "% WiFi\nligado"]
        fig_comp, axes_comp = plt.subplots(1, len(metricas_plot), figsize=(14, 4))
        for ax_c, met in zip(axes_comp, metricas_plot):
            vals = [df_comp.loc[f"User {u}", met] for u in [1,2,3]]
            bars = ax_c.bar([1,2,3], vals,
                            color=[COR_USER[1], COR_USER[2], COR_USER[3]], alpha=0.85)
            ax_c.set_xticks([1,2,3])
            ax_c.set_xticklabels(["User 1","User 2","User 3"], fontsize=8)
            ax_c.set_title(met, fontsize=9)
            ax_c.set_ylabel("%")
            for bar, v in zip(bars, vals):
                ax_c.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                          f"{v:.1f}", ha="center", fontsize=8)
        plt.suptitle("Comparação de perfis entre utilizadores", y=1.02)
        plt.tight_layout()
        st.pyplot(fig_comp)
        plt.close(fig_comp)


# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Resumo Semanal")

    # ── Métricas ──────────────────────────────────────────────────────────────
    loc_principal = "—"
    if not spts.empty and "location_id" in spts.columns:
        s2 = spts.copy()
        s2["dwell_s"] = (s2["finished_at"] - s2["started_at"]).dt.total_seconds()
        dbl = s2.groupby("location_id")["dwell_s"].sum()
        if not dbl.empty:
            best = dbl.idxmax()
            if "fsq_category" in locs.columns:
                try:
                    loc_principal = f"{locs.loc[best, 'fsq_category']} (loc {best})"
                except Exception:
                    loc_principal = f"Loc {best}"
            else:
                loc_principal = f"Loc {best}"

    dist_km = 0.0
    if not tpls.empty:
        try:
            dist_km = tpls.to_crs("EPSG:3857").geometry.length.sum() / 1000
        except Exception:
            pass

    modo_dom = "—"
    if "transport_mode" in tpls.columns and not tpls.empty:
        t3 = tpls.copy()
        t3["dur"] = (t3["finished_at"] - t3["started_at"]).dt.total_seconds()
        modo_dom = t3.groupby("transport_mode")["dur"].sum().idxmax()

    raio_km = 0.0
    if not spts.empty and hasattr(spts, "geometry"):
        try:
            sp = spts.to_crs("EPSG:3857")
            cx, cy = sp.geometry.x.mean(), sp.geometry.y.mean()
            dists = np.sqrt((sp.geometry.x - cx)**2 + (sp.geometry.y - cy)**2)
            raio_km = float(np.sqrt((dists**2).mean())) / 1000
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Localização Principal", loc_principal)
    c2.metric("Distância Total (km)", f"{dist_km:.1f} km")
    c3.metric("Modo Dominante", modo_dom)
    c4.metric("Raio de Giração (km)", f"{raio_km:.2f} km")

    st.divider()

    # ── ML ────────────────────────────────────────────────────────────────────
    st.subheader("Classificação de Actividade")

    ml_louo, features_all, labels_all, user_ids_all = carregar_ml_louo(raw_df)

    if sel_uid == 0:
        # Leave-one-user-out: mostrar tabela resumo
        st.markdown("**Modo:** Leave-One-User-Out (treinar em 2, testar em 1)")
        rows = []
        modelos_louo: dict = {}
        for fold in ml_louo:
            for nm, met in fold["models"].items():
                modelos_louo.setdefault(nm, {"acc": [], "f1": []})
                modelos_louo[nm]["acc"].append(met["accuracy"])
                modelos_louo[nm]["f1"].append(met["f1_macro"])
        for nm, vals in modelos_louo.items():
            rows.append({
                "Modelo": nm,
                "Acurácia média": f"{np.mean(vals['acc']):.3f}",
                "F1-Macro médio": f"{np.mean(vals['f1']):.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        fig_ml, axes_ml = plt.subplots(1, 2, figsize=(12, 4))
        fold_labels = [f"User {f['test_user']} (teste)" for f in ml_louo]
        x = np.arange(len(fold_labels))
        w = 0.25
        for i, (nm, vals) in enumerate(modelos_louo.items()):
            axes_ml[0].bar(x + i * w, vals["acc"], w, label=nm)
            axes_ml[1].bar(x + i * w, vals["f1"],  w, label=nm)
        for ax, t, yl in zip(axes_ml,
                              ["Acurácia por fold", "F1-Macro por fold"],
                              ["Acurácia", "F1-Macro"]):
            ax.set_xticks(x + w)
            ax.set_xticklabels(fold_labels, fontsize=9)
            ax.set_ylabel(yl)
            ax.set_title(t)
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1)
        plt.tight_layout()
        st.pyplot(fig_ml)
        plt.close(fig_ml)
        st.caption("Classes raras (Nightlife, Physical exercise, Shopping) → F1=0 quando "
                   "o utilizador detentor é o conjunto de teste. Comportamento esperado.")

    else:
        # 80/20 por utilizador seleccionado
        st.markdown(f"**Modo:** 80% treino / 20% teste — User {sel_uid}")

        @st.cache_resource(show_spinner=f"A treinar modelos para User {sel_uid}…")
        def _ml_8020(uid, _feat, _lab, _uids):
            from src.ml import train_evaluate_single_user
            return train_evaluate_single_user(_feat, _lab, _uids, uid)

        res_8020 = _ml_8020(sel_uid, features_all, labels_all, user_ids_all)
        st.caption(f"Amostras — Treino: {res_8020['n_train']:,} · Teste: {res_8020['n_test']:,}")

        rows2 = [
            {
                "Modelo": nm,
                "Acurácia": f"{met['accuracy']:.3f}",
                "F1-Macro": f"{met['f1_macro']:.3f}",
            }
            for nm, met in res_8020["models"].items()
        ]
        st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

        # Barras comparativas
        nomes  = list(res_8020["models"].keys())
        accs   = [res_8020["models"][n]["accuracy"] for n in nomes]
        f1s    = [res_8020["models"][n]["f1_macro"]  for n in nomes]
        x      = np.arange(len(nomes))
        fig4, axes4 = plt.subplots(1, 2, figsize=(10, 4))
        axes4[0].bar(x, accs, color=["#3498db", "#e67e22", "#2ecc71"])
        axes4[1].bar(x, f1s,  color=["#3498db", "#e67e22", "#2ecc71"])
        for ax, t in zip(axes4, ["Acurácia", "F1-Macro"]):
            ax.set_xticks(x)
            ax.set_xticklabels(nomes, fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_title(f"{t} — User {sel_uid} (80/20)")
            ax.set_ylabel(t)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        # Report detalhado (expandível)
        best_model = max(res_8020["models"], key=lambda n: res_8020["models"][n]["f1_macro"])
        with st.expander(f"📄 Relatório detalhado — {best_model} (melhor F1)"):
            st.code(res_8020["models"][best_model]["report"])
