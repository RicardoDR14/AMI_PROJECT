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

tab1, tab2, tab3 = st.tabs([
    "🗺️  Mapa de Mobilidade",
    "📊  Análise Temporal",
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
with tab3:
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
