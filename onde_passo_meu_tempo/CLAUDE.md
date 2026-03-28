# Onde Passo o Meu Tempo? — Mobility Analysis Dashboard

## Project Overview
Personal mobility analysis system using ContextLabeler smartphone GPS/sensor data.
Master's project for MEI – Ambient Intelligence (ISEC, Polytechnic of Coimbra).
Student: Ricardo Rodrigues (a2022147797)

## Goal
Process ContextLabeler dataset CSV files through a Trackintel mobility pipeline,
classify activity context with scikit-learn, and expose a Streamlit dashboard
with per-user filtering, privacy controls, and both LOUO and 80/20 ML evaluation.

## Tech Stack (implemented and working)
- Python 3.12 (system version — osmnx 2.x requires ≥ 3.11)
- Trackintel 1.4.2 — hierarchical mobility pipeline
- MovingPandas 0.19.0 — interactive trajectory visualisation
- scikit-learn 1.5.2 — activity classification
- OSMnx 2.0.0 — road network snap-to-road (privacy layer)
- geopandas 1.0.1 — spatial data (must be ≥ 1.0 for trackintel 1.4.2)
- Streamlit 1.40.2 — dashboard UI with sidebar user selector
- Folium 0.18.0 + streamlit-folium 0.21.0 — interactive maps
- Seaborn 0.13.2 + matplotlib 3.9.2 — temporal charts
- mapclassify — required by MovingPandas .explore()
- Foursquare Places API — semantic location enrichment (optional, via FSQ_API_KEY)

## Dataset
ContextLabeler Dataset (Campana et al., 2018)
- Files: data/user_1.csv, data/user_2.csv, data/user_3.csv
- 45,681 samples × 1,333 columns; 3 users × ~2 weeks; 60-second windows
- Actual column names (differ from generic names):
  - GPS:       location_lat, location_lon
  - Time:      time (Unix milliseconds — divide by 1000)
  - Label:     label (10 classes)
  - Battery:   battery_unplugged (binary: 1=on battery)
  - Accel:     sensor_linear_acc_x/y/z_mean (compute Euclidean norm)
  - WiFi:      wifi_connected (binary)
- Only 9 columns are loaded (usecols=COLS_NEEDED) — saves ~98% I/O
- ~86% of consecutive GPS points are stationary → few triplegs is expected

## Pipeline Steps (src/pipeline.py)

1. load_contextlabeler(data_dir) → raw_df
   - Reads 3 CSVs with usecols=COLS_NEEDED; adds user_id column

2. build_positionfixes(raw_df) → pfs (GeoDataFrame, EPSG:4326)
   - time (ms) → UTC datetime → tracked_at
   - Haversine speed per user via groupby+shift (never diff() globally)
   - Drops rows where speed_kmh > 200

3. build_trajectories(pfs) → TrajectoryCollection
   - Strips Trackintel subclass before passing to MovingPandas
   - min_length=10m to filter GPS drift

4. segment_trajectories(pfs) → (pfs_out, spts, tpls)
   - generate_staypoints(sliding, haversine, dist=100m, time=5min, gap=15min)
   - Sets spts["is_activity"] = True (required by generate_trips)
   - generate_triplegs(between_staypoints) on pfs_out (has staypoint_id)

5. merge_and_detect_locations(spts, tpls) → (spts_merged, locs)
   - generate_locations BEFORE merge_staypoints (merge requires location_id)
   - DBSCAN: epsilon=100m, min_samples=2, agg_level='dataset'
   - Prints Silhouette Score + Davies-Bouldin Index
   - merge_staypoints with agg={"geometry":"first","is_activity":"any"}
   - Rebuilds GeoDataFrame after merge (merge drops geometry type)
   - Locations geometry column is "center", not "geometry"

6. enrich_locations(locs) → locs with fsq_category
   - Requires FSQ_API_KEY env var; defaults to "Desconhecido" if absent

7. generate_trips_and_modes(spts, tpls) → (spts, tpls, trips)
   - ti.preprocessing.generate_trips (NOT ti.preprocessing.trips.generate_trips)
   - Manual speed-based classification: <15 km/h slow, <100 motorised, else fast

8. run_pipeline(data_dir) → dict
   - Keys: raw_df, pfs, tc, spts, tpls, locs, trips

## Privacy (src/privacy.py)
apply_privacy(pfs, spts, locs) → (pfs_priv, spts_priv)

Three independent layers (can be disabled individually):
1. Home suppression: detect_home() → location with max night dwell (22:00–08:00 UTC)
   Remove all points within 200m of home centroid (EPSG:3857)
2. Snap-to-road: OSMnx 2.x graph_from_bbox(bbox=(west,south,east,north))
   Graph cached in data/road_graph.graphml
3. K-anonymity: Foursquare categories with < 5 occurrences → "Outro"

CRITICAL: apply_privacy() must be called BEFORE any visualisation or export.

## ML (src/ml.py)

extract_features(raw_df) → (features_df, labels, user_ids)
- Operates on raw_df (not staypoints) — ML is per-60s-sample task
- Features: hour_sin, hour_cos, day_of_week, avg_speed, wifi_count, battery, linear_accel

train_and_evaluate(features, labels, user_ids) → list[dict]
- Leave-one-user-out: 3 folds, train on 2 users, test on 1
- Models: GaussianNB, DecisionTree, RandomForest
- zero_division=0 (Nightlife/Physical exercise/Shopping only in 1–2 users → F1=0 expected)

train_evaluate_single_user(features, labels, user_ids, selected_user, test_size=0.2)
- 80/20 split (stratified) for a single user
- Used by dashboard when a specific user is selected

## Dashboard (src/dashboard.py)
- Sidebar user selector: "Todos" (LOUO) or User 1/2/3 (80/20 ML)
- All artefacts filtered by selected user
- Tab 1 — Mapa de Mobilidade:
  - Folium map: GPS trajectories (1/10 subsampled) + triplegs + staypoints (∝ dwell) + DBSCAN locations
  - Layer control to toggle each layer
  - MovingPandas animated map (behind button, persisted in st.session_state)
- Tab 2 — Análise Temporal:
  - Seaborn heatmap: activity intensity per hour × weekday
  - Transport mode bar chart
  - Activity distribution per user
- Tab 3 — Resumo Semanal:
  - 4 metrics: top location, total distance, dominant mode, gyration radius
  - ML results: LOUO table+chart (Todos) or 80/20 per-user with expandable classification report

## Compatibility Patches (src/__init__.py)
Adds GeoDataFrame._geodataframe_constructor_with_fallback shim.
Trackintel 1.4.2 calls this internal geopandas method removed in geopandas 0.14.
This patch must load before any trackintel import.

## Known Issues / Limitations
- osmnx 2.x graph_from_bbox signature: bbox=(west, south, east, north)
- merge_staypoints drops geometry by default — must pass agg={"geometry":"first"}
- is_activity column must be set manually after generate_staypoints
- generate_trips is at ti.preprocessing.generate_trips (not .trips.generate_trips)
- MapClassify must be installed separately for MovingPandas .explore()
- ~86% stationary data → few triplegs; this is correct for this dataset
- Nightlife/Physical exercise/Shopping: F1=0 in LOUO when holder is test user

## Privacy Note
Home = location with max dwell time 22:00–08:00 UTC.
apply_privacy() must run BEFORE any visualisation or export.

## Code Style
- Type hints on all public functions
- Docstrings in Portuguese
- Constants at top of each file
- No cross-user data leakage: always groupby user_id before shift/diff

## Quick Start
```bash
cd onde_passo_meu_tempo
pip install -r requirements.txt

# Pipeline + ML
python3 - <<'EOF'
from src.pipeline import run_pipeline
from src.ml import extract_features, train_and_evaluate, print_results
art = run_pipeline()
feats, labels, uids = extract_features(art['raw_df'])
print_results(train_and_evaluate(feats, labels, uids))
EOF

# Dashboard
streamlit run src/dashboard.py
```
