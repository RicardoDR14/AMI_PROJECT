"""privacy.py — Filtros de privacidade para análise de mobilidade.

Aplica três camadas independentes, por esta ordem:

  1. Supressão domiciliar  — remove pontos a < HOME_SUPPRESS_METERS da casa.
  2. Snap-to-road          — substitui coordenadas exactas pelo nó viário mais próximo (OSMnx).
  3. K-anonimato           — agrega categorias Foursquare raras em 'Outro'.

A casa é inferida como o local com maior tempo de permanência acumulado
entre as 22:00 e as 08:00 UTC (período nocturno).

NOTA sobre o grafo OSMnx: o bounding box dos dados cobre ~100 km × 55 km
(região de Pisa/Toscana). O primeiro download pode demorar vários minutos.
O grafo é guardado em cache (data/road_graph.graphml) para reutilização.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# ── Constantes ────────────────────────────────────────────────────────────────

HOME_SUPPRESS_METERS: int  = 200       # Raio de supressão em torno da casa (metros)
NIGHT_START_HOUR_UTC: int  = 22        # Início do período nocturno (hora UTC inclusiva)
NIGHT_END_HOUR_UTC: int    = 8         # Fim do período nocturno (hora UTC exclusiva)
K_ANONYMITY_THRESHOLD: int = 5         # Limiar mínimo de ocorrências para k-anonimato
RARE_CATEGORY_LABEL: str   = "Outro"   # Rótulo para categorias raras após k-anonimato
OSM_NETWORK_TYPE: str      = "drive"   # Tipo de rede viária para o OSMnx
OSM_GRAPH_CACHE: Path      = Path(__file__).parent.parent / "data" / "road_graph.graphml"
CRS_METRIC: str            = "EPSG:3857"  # CRS métrico para cálculos de distância em metros


# ── Funções públicas ──────────────────────────────────────────────────────────

def detect_home(spts: gpd.GeoDataFrame) -> Point:
    """Detecta a localização domiciliar como o centróide do local com mais tempo nocturno.

    Define "casa" como a localização (location_id) com maior tempo total
    de permanência acumulado entre NIGHT_START_HOUR_UTC e NIGHT_END_HOUR_UTC.

    Parâmetros
    ----------
    spts : gpd.GeoDataFrame
        StaypointsDataFrame com colunas started_at, finished_at, location_id.

    Retorna
    -------
    Point
        Centróide WGS-84 da localização domiciliar estimada.
    """
    spts = spts.copy()
    start_hour = spts["started_at"].dt.hour
    is_night   = (start_hour >= NIGHT_START_HOUR_UTC) | (start_hour < NIGHT_END_HOUR_UTC)
    night_spts = spts[is_night & spts["location_id"].notna()]

    if night_spts.empty:
        # Fallback: centróide de todos os staypoints (caso não haja dados nocturnos)
        return spts.geometry.unary_union.centroid

    night_spts = night_spts.copy()
    night_spts["dwell_s"] = (
        night_spts["finished_at"] - night_spts["started_at"]
    ).dt.total_seconds()

    dwell_by_loc = night_spts.groupby("location_id")["dwell_s"].sum()
    home_loc_id  = dwell_by_loc.idxmax()

    home_spts = night_spts[night_spts["location_id"] == home_loc_id]
    return home_spts.geometry.unary_union.centroid


def apply_privacy(
    pfs: gpd.GeoDataFrame,
    spts: gpd.GeoDataFrame,
    locs: gpd.GeoDataFrame,
    k: int = K_ANONYMITY_THRESHOLD,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Aplica as três camadas de filtros de privacidade em sequência.

    As camadas são independentes e podem ser desactivadas individualmente
    comentando o bloco correspondente.

    Parâmetros
    ----------
    pfs : gpd.GeoDataFrame
        PositionfixesDataFrame (EPSG:4326).
    spts : gpd.GeoDataFrame
        StaypointsDataFrame com coluna location_id.
    locs : gpd.GeoDataFrame
        LocationsDataFrame com coluna fsq_category (geometria activa: 'center').
    k : int
        Limiar mínimo de ocorrências para k-anonimato (padrão=5).

    Retorna
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        (pfs_filtrados, spts_filtrados) após todas as camadas de privacidade.
    """
    home_point = detect_home(spts)
    print(f"  Casa detectada: lat={home_point.y:.5f}, lon={home_point.x:.5f}")

    # ── Camada 1: Supressão domiciliar ───────────────────────────────────────
    # Projectar para CRS métrico (EPSG:3857) para calcular distâncias em metros
    pfs_proj  = pfs.to_crs(CRS_METRIC)
    home_proj = (
        gpd.GeoSeries([home_point], crs="EPSG:4326")
        .to_crs(CRS_METRIC)
        .iloc[0]
    )

    dist_pfs  = pfs_proj.geometry.distance(home_proj)
    pfs_filt  = pfs[dist_pfs > HOME_SUPPRESS_METERS].copy()
    n_removed_pfs = len(pfs) - len(pfs_filt)

    spts_proj = spts.to_crs(CRS_METRIC)
    dist_spts = spts_proj.geometry.distance(home_proj)
    spts_filt = spts[dist_spts > HOME_SUPPRESS_METERS].copy()
    n_removed_spts = len(spts) - len(spts_filt)

    print(
        f"  Supressão domiciliar ({HOME_SUPPRESS_METERS}m): "
        f"removidos {n_removed_pfs} pfs, {n_removed_spts} spts."
    )

    # ── Camada 2: Snap-to-road (OSMnx) ──────────────────────────────────────
    print("  Snap-to-road via OSMnx...")
    pfs_filt = _snap_to_road(pfs_filt)

    # ── Camada 3: K-anonimato nas categorias Foursquare ──────────────────────
    if "fsq_category" in locs.columns:
        locs_anon = _apply_k_anonymity(locs, k=k)
        # Mapear categorias anonimizadas de volta aos staypoints via location_id
        cat_map = locs_anon["fsq_category"]
        spts_filt = spts_filt.copy()
        spts_filt["fsq_category"] = spts_filt["location_id"].map(cat_map)
        n_other = (spts_filt["fsq_category"] == RARE_CATEGORY_LABEL).sum()
        print(f"  K-anonimato (k={k}): {n_other} staypoints reclassificados como '{RARE_CATEGORY_LABEL}'.")

    return pfs_filt, spts_filt


# ── Funções privadas ──────────────────────────────────────────────────────────

def _snap_to_road(pfs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Faz snap dos positionfixes ao nó viário mais próximo via OSMnx.

    O grafo da rede viária é guardado em cache em OSM_GRAPH_CACHE para evitar
    descarregamentos repetidos. Para forçar um novo download, apagar o ficheiro:
      rm data/road_graph.graphml

    Parâmetros
    ----------
    pfs : gpd.GeoDataFrame
        PositionfixesDataFrame em EPSG:4326.

    Retorna
    -------
    gpd.GeoDataFrame
        PositionfixesDataFrame com coordenadas substituídas pelo nó viário mais próximo,
        reprojectado para EPSG:4326.
    """
    import osmnx as ox  # importação local — evita erro se osmnx não instalado ao importar o módulo

    # ── Carregar ou descarregar grafo ─────────────────────────────────────────
    if OSM_GRAPH_CACHE.exists():
        print(f"    Grafo carregado da cache: {OSM_GRAPH_CACHE}")
        G = ox.load_graphml(OSM_GRAPH_CACHE)
    else:
        lons = pfs.geometry.x
        lats = pfs.geometry.y
        # osmnx 1.9.x: graph_from_bbox(north, south, east, west, network_type)
        print("    A descarregar rede viária do OpenStreetMap (pode demorar minutos)...")
        G = ox.graph_from_bbox(
            north=float(lats.max()),
            south=float(lats.min()),
            east=float(lons.max()),
            west=float(lons.min()),
            network_type=OSM_NETWORK_TYPE,
        )
        OSM_GRAPH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(G, OSM_GRAPH_CACHE)
        print(f"    Grafo guardado em cache: {OSM_GRAPH_CACHE}")

    # ── Projectar grafo e pontos para CRS métrico ─────────────────────────────
    # nearest_nodes exige que X/Y estejam no mesmo CRS que o grafo projectado
    G_proj    = ox.project_graph(G)
    pfs_proj  = pfs.to_crs(G_proj.graph["crs"])

    X = pfs_proj.geometry.x.values
    Y = pfs_proj.geometry.y.values

    # ── Nó viário mais próximo para cada ponto ────────────────────────────────
    nearest_node_ids = ox.nearest_nodes(G_proj, X=X, Y=Y)
    node_data = dict(G_proj.nodes(data=True))

    snapped_x = np.array([node_data[n]["x"] for n in nearest_node_ids])
    snapped_y = np.array([node_data[n]["y"] for n in nearest_node_ids])

    # ── Reconstruir GeoDataFrame com coordenadas snapped ─────────────────────
    pfs_snapped = pfs_proj.copy()
    pfs_snapped.geometry = gpd.points_from_xy(snapped_x, snapped_y)
    return pfs_snapped.to_crs("EPSG:4326")


def _apply_k_anonymity(
    locs: gpd.GeoDataFrame,
    k: int,
) -> gpd.GeoDataFrame:
    """Aplica k-anonimato às categorias Foursquare: agrega categorias raras em 'Outro'.

    Categorias com menos de k ocorrências são substituídas por RARE_CATEGORY_LABEL,
    impedindo a re-identificação por tipos de local únicos.

    Parâmetros
    ----------
    locs : gpd.GeoDataFrame
        LocationsDataFrame com coluna 'fsq_category'.
    k : int
        Número mínimo de ocorrências para manter uma categoria.

    Retorna
    -------
    gpd.GeoDataFrame
        Locations com categorias raras substituídas por RARE_CATEGORY_LABEL.
    """
    locs = locs.copy()
    cat_counts    = locs["fsq_category"].value_counts()
    rare_cats     = cat_counts[cat_counts < k].index
    locs.loc[locs["fsq_category"].isin(rare_cats), "fsq_category"] = RARE_CATEGORY_LABEL
    return locs
