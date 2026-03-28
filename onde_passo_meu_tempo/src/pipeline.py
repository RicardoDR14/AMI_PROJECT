"""pipeline.py — Pipeline de mobilidade (ContextLabeler → Trackintel).

Encadeia todos os passos de processamento GPS:
  CSV bruto → PositionfixesDataFrame → Staypoints/Triplegs → Locations → Trips
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import trackintel as ti
import movingpandas as mpd
import requests
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ── Constantes ────────────────────────────────────────────────────────────────

DATA_DIR: Path = Path(__file__).parent.parent / "data"

USER_FILES: dict[int, str] = {
    1: "user_1.csv",
    2: "user_2.csv",
    3: "user_3.csv",
}

# Colunas necessárias dos CSVs (os ficheiros têm 1 333 colunas; só carregamos estas)
COLS_NEEDED: list[str] = [
    "time",                        # Unix timestamp em milissegundos
    "location_lat",                # Latitude WGS-84
    "location_lon",                # Longitude WGS-84
    "label",                       # Rótulo de actividade
    "wifi_connected",              # WiFi ligado (binário)
    "battery_unplugged",           # Bateria desligada do carregador (binário)
    "sensor_linear_acc_x_mean",    # Aceleração linear média — eixo X
    "sensor_linear_acc_y_mean",    # Aceleração linear média — eixo Y
    "sensor_linear_acc_z_mean",    # Aceleração linear média — eixo Z
]

SPEED_THRESHOLD_KMH: float = 200.0   # Limiar de velocidade para remoção de outliers GPS (km/h)
DISTANCE_THRESHOLD_M: int  = 100     # Distância mínima para detecção de staypoints (metros)
TIME_THRESHOLD_MIN: float  = 5.0     # Tempo mínimo de permanência num staypoint (minutos)
GAP_THRESHOLD_MIN: float   = 15.0    # Intervalo máximo entre registos GPS (minutos)
DBSCAN_EPSILON_M: int      = 100     # Raio do DBSCAN para clustering de localizações (metros)
DBSCAN_MIN_SAMPLES: int    = 2       # Número mínimo de amostras para o DBSCAN
SLOW_SPEED_KMH: float      = 15.0   # Limiar de velocidade — modo lento (km/h)
MOTORISED_SPEED_KMH: float = 100.0  # Limiar de velocidade — modo motorizado (km/h)
HOME_SUPPRESS_METERS: int  = 200    # Raio de supressão em torno da casa (metros, usado em privacy.py)

FSQ_API_BASE: str = "https://api.foursquare.com/v3/places/nearby"


# ── Funções auxiliares privadas ───────────────────────────────────────────────

def _haversine_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Calcula a distância haversine em metros entre pares de coordenadas WGS-84.

    Parâmetros
    ----------
    lat1, lon1 : np.ndarray
        Coordenadas do ponto de origem (graus decimais).
    lat2, lon2 : np.ndarray
        Coordenadas do ponto de destino (graus decimais).

    Retorna
    -------
    np.ndarray
        Distâncias em metros.
    """
    R = 6_371_000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _compute_tripleg_speed(tpls: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calcula a velocidade média (m/s) para cada tripleg a partir da geometria e duração.

    Parâmetros
    ----------
    tpls : gpd.GeoDataFrame
        TriplegsDataFrame com colunas started_at, finished_at e geometria.

    Retorna
    -------
    gpd.GeoDataFrame
        Triplegs com coluna 'speed' em m/s adicionada.
    """
    tpls = tpls.copy()
    tpls_proj = tpls.to_crs("EPSG:3857")
    duration_s = (tpls["finished_at"] - tpls["started_at"]).dt.total_seconds()
    distance_m = tpls_proj.geometry.length
    with np.errstate(divide="ignore", invalid="ignore"):
        tpls["speed"] = np.where(duration_s > 0, distance_m / duration_s, 0.0)
    return tpls


# ── Funções principais do pipeline ────────────────────────────────────────────

def load_contextlabeler(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Carrega os três CSVs do ContextLabeler e devolve um DataFrame concatenado.

    Usa usecols para carregar apenas as 9 colunas necessárias dos ficheiros
    de 1 333 colunas (~98% de poupança de I/O).

    Parâmetros
    ----------
    data_dir : Path | str
        Directório que contém user_1.csv, user_2.csv, user_3.csv.

    Retorna
    -------
    pd.DataFrame
        DataFrame com as colunas de COLS_NEEDED mais 'user_id' (1, 2 ou 3).
    """
    data_dir = Path(data_dir)
    frames: list[pd.DataFrame] = []
    for user_id, filename in USER_FILES.items():
        df = pd.read_csv(data_dir / filename, usecols=COLS_NEEDED)
        df["user_id"] = user_id
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    print(f"  Carregados {len(raw):,} registos de {raw['user_id'].nunique()} utilizadores.")
    return raw


def build_positionfixes(raw_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Constrói um PositionfixesDataFrame válido para o Trackintel.

    Passos:
    1. Converte o timestamp Unix ms → datetime UTC.
    2. Ordena por utilizador e tempo.
    3. Calcula a velocidade haversine vectorizada por utilizador
       (o groupby evita contaminar a fronteira entre utilizadores).
    4. Remove pontos onde velocidade > SPEED_THRESHOLD_KMH.
    5. Cria a geometria Point(lon, lat) e valida com ti.io.read_positionfixes_gpd.

    Parâmetros
    ----------
    raw_df : pd.DataFrame
        DataFrame devolvido por load_contextlabeler().

    Retorna
    -------
    gpd.GeoDataFrame
        PositionfixesDataFrame pronto para o pipeline Trackintel.
    """
    df = raw_df.copy()

    # 1. Converter timestamp Unix ms → datetime com fuso UTC
    df["tracked_at"] = pd.to_datetime(df["time"], unit="ms", utc=True)

    # 2. Ordenar por utilizador e tempo (evita contaminação cross-user na velocidade)
    df = df.sort_values(["user_id", "tracked_at"]).reset_index(drop=True)

    # 3. Velocidade haversine vectorizada — agrupada por utilizador
    speed_series = pd.Series(0.0, index=df.index)
    for uid, grp in df.groupby("user_id"):
        idx = grp.index
        lats = grp["location_lat"].values
        lons = grp["location_lon"].values
        t_s  = grp["tracked_at"].values.astype("int64") / 1e9  # ns → s

        # np.roll desloca o array 1 posição: índice i contém valor i-1
        prev_lats = np.roll(lats, 1)
        prev_lons = np.roll(lons, 1)
        prev_ts   = np.roll(t_s,  1)

        dist_m = _haversine_m(prev_lats, prev_lons, lats, lons)
        dt_s   = t_s - prev_ts

        with np.errstate(divide="ignore", invalid="ignore"):
            spd = np.where(dt_s > 0, (dist_m / 1_000.0) / (dt_s / 3_600.0), 0.0)

        spd[0] = 0.0  # primeiro ponto do utilizador não tem referência anterior
        speed_series[idx] = spd

    # 4. Remover outliers GPS
    n_before = len(df)
    df = df[speed_series <= SPEED_THRESHOLD_KMH].reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"  Removidos {n_removed} outliers GPS (velocidade > {SPEED_THRESHOLD_KMH} km/h).")

    # 5. Construir geometria — Shapely/GeoPandas usa (longitude, latitude)
    geometry = gpd.points_from_xy(df["location_lon"], df["location_lat"])
    gdf = gpd.GeoDataFrame(
        {
            "user_id":    df["user_id"].astype(int),
            "tracked_at": df["tracked_at"],
        },
        geometry=geometry,
        crs="EPSG:4326",
    ).reset_index(drop=True)

    # Validar e converter para PositionfixesDataFrame do Trackintel
    pfs = ti.io.read_positionfixes_gpd(gdf, tracked_at="tracked_at", user_id="user_id")
    print(f"  {len(pfs):,} positionfixes construídos.")
    return pfs


def build_trajectories(pfs: gpd.GeoDataFrame) -> mpd.TrajectoryCollection:
    """Constrói uma TrajectoryCollection (MovingPandas) para QA visual das trajectórias.

    O TrajectoryCollection permite exploração interactiva das trajectórias brutas
    antes de qualquer segmentação, ideal para verificar a qualidade do GPS.

    Parâmetros
    ----------
    pfs : gpd.GeoDataFrame
        PositionfixesDataFrame devolvido por build_positionfixes().

    Retorna
    -------
    mpd.TrajectoryCollection
        Colecção de trajectórias por utilizador.
        Usar tc.explore() para visualização interactiva via HoloViz.
    """
    tc = mpd.TrajectoryCollection(
        pfs,
        traj_id_col="user_id",
        t="tracked_at",
        min_length=10,  # metros — filtra trajectórias triviais (p. ex. GPS drift estacionário)
    )
    print(f"  TrajectoryCollection com {len(tc)} trajectórias criado.")
    return tc


def segment_trajectories(
    pfs: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Segmenta positionfixes em staypoints e triplegs usando o Trackintel.

    Staypoints: locais onde o utilizador ficou parado ≥ TIME_THRESHOLD_MIN
    dentro de um raio de DISTANCE_THRESHOLD_M metros.
    Triplegs: segmentos de movimento entre staypoints consecutivos.

    ATENÇÃO: o pfs_out devolvido (não o pfs original) deve ser passado a funções
    subsequentes — contém a coluna staypoint_id necessária para generate_triplegs.

    Parâmetros
    ----------
    pfs : gpd.GeoDataFrame
        PositionfixesDataFrame.

    Retorna
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]
        (pfs_actualizado, staypoints, triplegs)
    """
    pfs_out, spts = pfs.generate_staypoints(
        method="sliding",
        distance_metric="haversine",
        dist_threshold=DISTANCE_THRESHOLD_M,
        time_threshold=TIME_THRESHOLD_MIN,
        gap_threshold=GAP_THRESHOLD_MIN,
    )
    # pfs_out tem agora a coluna 'staypoint_id' — obrigatório para triplegs
    pfs_out, tpls = pfs_out.generate_triplegs(
        staypoints=spts,
        method="between_staypoints",
        gap_threshold=GAP_THRESHOLD_MIN,
    )
    print(f"  {len(spts):,} staypoints | {len(tpls):,} triplegs")
    return pfs_out, spts, tpls


def merge_and_detect_locations(
    spts: gpd.GeoDataFrame,
    tpls: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Detecta localizações significativas via DBSCAN e agrega staypoints consecutivos.

    Ordem obrigatória: generate_locations ANTES de merge_staypoints
    (merge_staypoints exige que a coluna location_id exista nos staypoints).

    Imprime Silhouette Score e Davies-Bouldin Index para avaliação do clustering.
    Atenção: a coluna de geometria das Locations chama-se 'center' (não 'geometry').

    Parâmetros
    ----------
    spts : gpd.GeoDataFrame
        StaypointsDataFrame devolvido por segment_trajectories().
    tpls : gpd.GeoDataFrame
        TriplegsDataFrame devolvido por segment_trajectories().

    Retorna
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        (staypoints_com_location_id, locations com geometria activa 'center')
    """
    # generate_locations usa DBSCAN com distância haversine; epsilon em metros
    # (o Trackintel converte internamente para radianos: epsilon / 6_371_000)
    spts_locs, locs = spts.generate_locations(
        method="dbscan",
        epsilon=DBSCAN_EPSILON_M,
        num_samples=DBSCAN_MIN_SAMPLES,
        distance_metric="haversine",
        agg_level="dataset",  # clusters partilhados entre todos os utilizadores
    )
    print(f"  {len(locs)} localizações detectadas (DBSCAN ε={DBSCAN_EPSILON_M}m).")

    # Métricas de qualidade do clustering (excluir ruído: location_id == -1 ou NaN)
    sp_clean = spts_locs[
        spts_locs["location_id"].notna() & (spts_locs["location_id"] != -1)
    ]
    if len(sp_clean) > 1 and sp_clean["location_id"].nunique() > 1:
        coords = np.column_stack(
            [sp_clean.geometry.y.values, sp_clean.geometry.x.values]
        )
        labels = sp_clean["location_id"].astype(int).values
        sil = silhouette_score(coords, labels, metric="haversine")
        dbi = davies_bouldin_score(coords, labels)
        print(f"  Silhouette Score  (DBSCAN): {sil:.4f}")
        print(f"  Davies-Bouldin Index (DBSCAN): {dbi:.4f}")

    # Agregar staypoints consecutivos na mesma localização
    spts_merged = ti.preprocessing.staypoints.merge_staypoints(
        spts_locs,
        triplegs=tpls,
        max_time_gap="10min",
    )

    # A geometria das Locations está na coluna 'center' — activá-la explicitamente
    locs = locs.set_geometry("center")

    return spts_merged, locs


def enrich_locations(locs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Enriquece as localizações com categorias da Foursquare Places API.

    Se a variável de ambiente FSQ_API_KEY não estiver definida, atribui
    'Desconhecido' a todas as localizações e devolve sem fazer chamadas à API.

    Para activar: exportar FSQ_API_KEY=<chave> antes de correr o pipeline.

    Parâmetros
    ----------
    locs : gpd.GeoDataFrame
        LocationsDataFrame devolvido por merge_and_detect_locations()
        (geometria activa: 'center').

    Retorna
    -------
    gpd.GeoDataFrame
        Locations com coluna 'fsq_category' adicionada.
    """
    locs = locs.copy()
    fsq_key = os.getenv("FSQ_API_KEY")

    if not fsq_key:
        # TODO: definir FSQ_API_KEY no ambiente para activar enriquecimento semântico
        #       export FSQ_API_KEY="<a_tua_chave_foursquare>"
        locs["fsq_category"] = "Desconhecido"
        print("  FSQ_API_KEY não definida — categorias definidas como 'Desconhecido'.")
        return locs

    categories: list[str] = []
    for _, row in locs.iterrows():
        lat = row.geometry.y
        lon = row.geometry.x
        try:
            resp = requests.get(
                FSQ_API_BASE,
                params={"ll": f"{lat},{lon}", "limit": 1},
                headers={"Authorization": fsq_key},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("results") and data["results"][0].get("categories"):
                cat = data["results"][0]["categories"][0]["name"]
            else:
                cat = "Desconhecido"
        except Exception:
            cat = "Desconhecido"
        categories.append(cat)

    locs["fsq_category"] = categories
    return locs


def generate_trips_and_modes(
    spts: gpd.GeoDataFrame,
    tpls: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Gera viagens e classifica os modos de transporte dos triplegs.

    Modo de transporte baseado na velocidade média do tripleg:
      < SLOW_SPEED_KMH       → mobilidade_lenta      (a pé, bicicleta)
      < MOTORISED_SPEED_KMH  → mobilidade_motorizada  (carro, autocarro)
      ≥ MOTORISED_SPEED_KMH  → mobilidade_rapida      (comboio, avião)

    Parâmetros
    ----------
    spts : gpd.GeoDataFrame
        StaypointsDataFrame.
    tpls : gpd.GeoDataFrame
        TriplegsDataFrame.

    Retorna
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]
        (staypoints_actualizados, triplegs_com_transport_mode, trips)
    """
    spts_out, tpls_out, trips = ti.preprocessing.trips.generate_trips(
        staypoints=spts,
        triplegs=tpls,
        gap_threshold=GAP_THRESHOLD_MIN,
    )

    # Garantir que a coluna 'speed' existe (m/s)
    if "speed" not in tpls_out.columns:
        tpls_out = _compute_tripleg_speed(tpls_out)

    def _classify_mode(speed_ms: float) -> str:
        """Classifica o modo de transporte com base na velocidade em m/s."""
        speed_kmh = speed_ms * 3.6
        if speed_kmh < SLOW_SPEED_KMH:
            return "mobilidade_lenta"
        elif speed_kmh < MOTORISED_SPEED_KMH:
            return "mobilidade_motorizada"
        return "mobilidade_rapida"

    tpls_out = tpls_out.copy()
    tpls_out["transport_mode"] = tpls_out["speed"].apply(_classify_mode)
    print(f"  {len(trips):,} viagens geradas.")
    return spts_out, tpls_out, trips


def run_pipeline(data_dir: Path | str = DATA_DIR) -> dict:
    """Executa o pipeline completo de análise de mobilidade.

    Encadeia todos os passos e devolve um dicionário com todos os artefactos.
    O dicionário inclui 'raw_df' para que src/ml.py possa extrair features
    ao nível da amostra (60 s), sem necessidade de recarregar os CSVs.

    Parâmetros
    ----------
    data_dir : Path | str
        Directório com os CSVs do ContextLabeler.

    Retorna
    -------
    dict
        Chaves: raw_df, pfs, tc, spts, tpls, locs, trips.

    Exemplo de uso
    --------------
    >>> from src.pipeline import run_pipeline
    >>> artefactos = run_pipeline()
    >>> artefactos['spts'].head()
    """
    print("=== Passo 1: Carregar dados ===")
    raw_df = load_contextlabeler(data_dir)

    print("=== Passo 2: Construir positionfixes ===")
    pfs = build_positionfixes(raw_df)

    print("=== Passo 3: Construir trajectórias (MovingPandas QA) ===")
    tc = build_trajectories(pfs)

    print("=== Passo 4: Segmentar trajectórias ===")
    pfs, spts, tpls = segment_trajectories(pfs)

    print("=== Passo 5: Detectar e fundir localizações (DBSCAN) ===")
    spts, locs = merge_and_detect_locations(spts, tpls)

    print("=== Passo 6: Enriquecer localizações (Foursquare) ===")
    locs = enrich_locations(locs)

    print("=== Passo 7: Gerar viagens e modos de transporte ===")
    spts, tpls, trips = generate_trips_and_modes(spts, tpls)

    print("=== Pipeline concluído ===")
    return {
        "raw_df": raw_df,
        "pfs":    pfs,
        "tc":     tc,
        "spts":   spts,
        "tpls":   tpls,
        "locs":   locs,
        "trips":  trips,
    }
