# Onde Passo o Meu Tempo? — Mobility Analysis Dashboard

## Project Overview
Personal mobility analysis system using smartphone GPS/sensor data.
Master's project for MEI – Ambient Intelligence (ISEC, Polytechnic of Coimbra).
Student: Ricardo Rodrigues (a2022147797)

## Goal
Build a pipeline that processes ContextLabeler dataset CSV files through
Trackintel (analysis) + MovingPandas (interactive visualization), classifies
activity context with ML, and exposes a Streamlit dashboard with privacy controls.

## Tech Stack
- Python 3.10+
- Trackintel — hierarchical mobility pipeline (positionfixes → staypoints → locations → trips)
- MovingPandas — interactive trajectory visualization (TrajectoryCollection + HoloViz)
- scikit-learn — activity classification (GaussianNB, DecisionTree, RandomForest)
- scikit-mobility — mobility metrics (radius of gyration, entropy, routine index)
- Streamlit — dashboard UI
- Folium — static maps
- Seaborn — temporal heatmaps
- OSMnx — road network for privacy (snap-to-road)
- Foursquare Places API — semantic location enrichment

## Dataset
ContextLabeler Dataset (Campana et al., 2018)
- Files: data/user_1.csv, data/user_2.csv, data/user_3.csv
- 45,681 samples × 1,332 features, 3 users × ~2 weeks each
- Key columns for GPS: latitude, longitude, timestamp (Unix), activity_label
- Sampling: 1-second sliding window

## Pipeline Steps
1. Ingest CSVs → build PositionfixesDataFrame (lat, lon, tracked_at, user_id)
   - Remove GPS outliers: speed > 200 km/h between consecutive points
   - Use MovingPandas TrajectoryCollection for visual QA of raw GPS

2. Segment trajectories (Trackintel)
   - generate_staypoints(distance_threshold=100, time_threshold=5min)
   - generate_triplegs()
   - merge_staypoints() for consecutive staypoints at same location

3. Detect significant locations (Trackintel)
   - generate_locations(method='dbscan', epsilon=100m, num_samples=2)
   - Enrich with Foursquare Places API per centroid

4. Generate trips + transport mode (Trackintel)
   - generate_trips()
   - predict_transport_mode() — thresholds: <15 km/h slow, <100 km/h motorised, >100 km/h fast

5. ML activity classification (scikit-learn)
   - Features: hour_sin, hour_cos, day_of_week, avg_speed, wifi_count, battery, linear_accel
   - Train on users 1+2, test on user 3 (cross-user generalisation)
   - Compare: GaussianNB, DecisionTreeClassifier, RandomForestClassifier
   - Evaluate: classification_report (accuracy, F1-macro, precision, recall)

6. Dashboard + Privacy (Streamlit + MovingPandas)
   - Folium map: triplegs by user colour, locations by Foursquare category
   - MovingPandas HoloViz: animated trajectory map, interactive time filters
   - Seaborn heatmap: activity intensity per hour × weekday
   - Bar charts: time per location category, transport mode distribution
   - Weekly summary card: top location, total distance, dominant mode, gyration radius
   - Privacy module: OSMnx snap-to-road, suppress 200m around home centroid,
     k-anonymity by location category (merge rare categories → "Other")

## Privacy Note
Home location = location with max dwell time between 22:00–08:00.
All privacy filters must be applied BEFORE any shared visualisation.

## ML Evaluation
Always use leave-one-out across the 3 users and report average metrics.
Report Silhouette Score and Davies-Bouldin Index for DBSCAN clustering.

## Code Style
- Type hints on all functions
- Docstrings in Portuguese (project language)
- Each pipeline step in its own function in src/pipeline.py
- Dashboard logic isolated in src/dashboard.py
- Privacy module isolated in src/privacy.py
- All parameters (thresholds, epsilon, etc.) as constants at top of each file

## Install
pip install trackintel movingpandas scikit-learn scikit-mobility streamlit
pip install folium seaborn osmnx geopandas hvplot
```

---

## 4. How to use it in VS Code

Place the `CLAUDE.md` at your project root — Claude Code reads it automatically when it starts in that directory, eliminating the need to repeat project conventions in every prompt. 

Then you just open the Claude Code panel and give direct tasks like:
```
Implement step 1 of the pipeline — read the 3 ContextLabeler CSVs 
from data/ and build a valid PositionfixesDataFrame for Trackintel, 
including GPS outlier removal.
```
```
Implement the MovingPandas TrajectoryCollection for visual QA 
of the raw GPS data from step 1.
```
```
Build the Streamlit dashboard with 3 tabs: Map, Temporal Analysis, 
Weekly Summary. Use the functions already implemented in pipeline.py.