# Notas de Implementação — Onde Passo o Meu Tempo?

Guia técnico de referência para o pipeline de mobilidade.
Explica as decisões de implementação, o fluxo de dados, os bugs resolvidos e como correr o projecto.

---

## 1. Porquê estas bibliotecas?

| Biblioteca | Papel | Nota |
|---|---|---|
| **Trackintel** | GPS → staypoints → locations → trips | Construída para mobilidade; trata gaps, ruído e clustering |
| **MovingPandas** | Visualização interactiva de trajectórias | `.explore()` gera mapa animado; requer `mapclassify` |
| **scikit-learn** | Classificador de actividade | Standard; fácil de trocar modelos |
| **OSMnx** | Download de rede viária + snap-to-road | Acesso directo ao OpenStreetMap |
| ~~scikit-mobility~~ | ~~Métricas de mobilidade~~ | **Excluído**: fixa `pandas<2.0` e `geopandas<0.11` |

---

## 2. Nomes de colunas reais vs. CLAUDE.md original

| Genérico | Coluna real no CSV | Notas |
|---|---|---|
| `latitude` | `location_lat` | Graus decimais WGS-84 |
| `longitude` | `location_lon` | Graus decimais WGS-84 |
| `timestamp` | `time` | Unix **milissegundos** → dividir por 1000 |
| `activity_label` | `label` | 10 classes |
| `battery` | `battery_unplugged` | Binário: 1=sem carregar |
| `linear_accel` | `sensor_linear_acc_x/y/z_mean` | Calcular norma Euclidiana |
| `wifi_count` | `wifi_connected` | Binário |

Os CSVs têm 1 333 colunas; `load_contextlabeler` usa `usecols=COLS_NEEDED` → 9 colunas, poupa ~98% de I/O.

---

## 3. Fluxo de dados

```
data/user_1.csv
data/user_2.csv   ──► load_contextlabeler()  ──► raw_df (9 colunas)
data/user_3.csv                                        │
                                                       ├──► extract_features()  [ml.py]
                                                       │         │
                                                       │         ▼
                                                       │    train_and_evaluate()       ← LOUO
                                                       │    train_evaluate_single_user() ← 80/20
                                                       │
                                                       ▼
                                               build_positionfixes()
                                               (remoção outliers GPS > 200 km/h)
                                                       │
                                                       ▼
                                               build_trajectories()
                                               (MovingPandas QA visual)
                                                       │
                                                       ▼
                                               segment_trajectories()
                                               → spts (+ is_activity=True) + tpls
                                                       │
                                                       ▼
                                               merge_and_detect_locations()
                                               → spts_merged + locs
                                               (Silhouette + DB impressos)
                                                       │
                                                       ▼
                                               enrich_locations()
                                               → locs['fsq_category']
                                                       │
                                                       ▼
                                               generate_trips_and_modes()
                                               → trips + tpls['transport_mode']
                                                       │
                                                       ▼
                                               apply_privacy()  [privacy.py]
                                               → pfs_priv + spts_priv
                                                       │
                                                       ▼
                                               dashboard.py (Streamlit)
                                               sidebar: User 1 / 2 / 3 / Todos
```

---

## 4. Porquê `extract_features` recebe `raw_df` e não `staypoints`

A tarefa de ML é **classificação ao nível da amostra** — cada linha do CSV (janela de 60 s) tem um rótulo. Os staypoints agregam múltiplas amostras, perdendo o detalhe de sensor necessário como features.

```python
artefactos = run_pipeline()
features, labels, user_ids = extract_features(artefactos['raw_df'])
```

---

## 5. Estratégias de avaliação ML

### Leave-One-User-Out (LOUO) — modo "Todos"
Responde a: *"o modelo funciona para um utilizador novo?"*
3 folds; cada fold treina em 2 utilizadores e testa no restante.

### 80/20 por utilizador — modo "User X"
Responde a: *"quão bem classifica as actividades deste utilizador específico?"*
Split estratificado dentro dos dados de um só utilizador.

### Limitação conhecida — desequilíbrio de classes

| Classe | Utilizadores |
|---|---|
| Nightlife | Só user_1 |
| Physical exercise | Só user_2 |
| Shopping | Só user_3 |

Em LOUO, F1=0 para essas classes quando o seu utilizador é o conjunto de teste. Usar `zero_division=0`.

---

## 6. Bugs resolvidos durante implementação

### 6.1 `_geodataframe_constructor_with_fallback` — trackintel 1.4.2 + geopandas 1.x

**Erro:** `AttributeError: type object 'GeoDataFrame' has no attribute '_geodataframe_constructor_with_fallback'`

**Causa:** trackintel 1.4.2 chama este método interno do geopandas, removido em 0.14.0.

**Fix:** Shim em `src/__init__.py` que restaura o método antes de qualquer import trackintel:
```python
if not hasattr(gpd.GeoDataFrame, "_geodataframe_constructor_with_fallback"):
    @classmethod
    def _geodataframe_constructor_with_fallback(cls, *args, **kwargs):
        try:
            return cls(*args, **kwargs)
        except Exception:
            return pd.DataFrame(*args, **kwargs)
    gpd.GeoDataFrame._geodataframe_constructor_with_fallback = _geodataframe_constructor_with_fallback
```

### 6.2 `generate_trips` — módulo errado

**Erro:** `AttributeError: module 'trackintel.preprocessing.trips' has no attribute 'generate_trips'`

**Fix:** Usar `ti.preprocessing.generate_trips` (não `.trips.generate_trips`).

### 6.3 `merge_staypoints` perde geometria

**Erro:** `'Series' object has no attribute 'geometry'` no `generate_trips`.

**Causa:** `merge_staypoints` com `agg={}` devolve apenas `user_id`, `started_at`, `finished_at` — descarta geometria.

**Fix:**
```python
merge_staypoints(..., agg={"geometry": "first", "is_activity": "any"})
# Depois reconverter:
spts_merged = gpd.GeoDataFrame(spts_raw, geometry="geometry", crs=spts_locs.crs)
```

### 6.4 `is_activity` não existe após `generate_staypoints`

**Erro:** `AttributeError: staypoints need the column 'is_activity'`

**Fix:** Adicionar manualmente após `generate_staypoints`:
```python
spts["is_activity"] = True
```

### 6.5 osmnx 2.x — `graph_from_bbox` API changed

**Erro:** Keyword args `north=`, `south=`, `east=`, `west=` não existem em osmnx 2.x.

**Fix:**
```python
G = ox.graph_from_bbox(
    bbox=(float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max())),
    network_type=OSM_NETWORK_TYPE,
)
```

### 6.6 `cat_map` com índice duplicado no k-anonimato

**Erro:** `InvalidIndexError: Reindexing only valid with uniquely valued Index objects`

**Fix:**
```python
if not cat_map.index.is_unique:
    cat_map = cat_map[~cat_map.index.duplicated(keep="first")]
```

### 6.7 `MovingPandas.explore()` devolve `folium.Map`, não tem `.to_html()`

**Fix:**
```python
buf = io.BytesIO()
tc.explore().save(buf, close_file=False)
html_str = buf.getvalue().decode("utf-8")
```

### 6.8 Dashboard perde estado ao mudar utilizador (session_state)

**Causa:** Streamlit faz rerun a cada interacção; conteúdo gerado por botões desaparece.

**Fix:** Persistir o HTML do mapa MovingPandas em `st.session_state["mpd_html"]`.

---

## 7. Compatibilidade de versões — triângulo geopandas/osmnx/trackintel

| Componente | Versão | Restrição |
|---|---|---|
| Python | 3.12 | osmnx 2.x requer ≥ 3.11 |
| geopandas | 1.0.1 | trackintel 1.4.2 requer ≥ 1.0 (API interna) |
| osmnx | 2.0.0 | requer Python ≥ 3.11 |
| trackintel | 1.4.2 | requer geopandas ≥ 1.0 (com o shim de src/__init__.py) |

---

## 8. Cache da rede viária OSMnx

O snap-to-road descarrega a rede viária para toda a bounding box (~Pisa/Toscana, 100km × 55km). Demora vários minutos e requer ~500 MB RAM. O grafo é guardado em `data/road_graph.graphml`.

Para forçar novo download:
```bash
rm data/road_graph.graphml
```

---

## 9. Modelo de dados Trackintel

```
Positionfixes  ──►  Staypoints  ──►  Locations
                         │
                         ▼
                     Triplegs   ──►  Trips
```

| Artefacto | O que é | Neste dataset |
|---|---|---|
| **Positionfixes** | Pontos GPS brutos | 1 por 60 s; ~45 k total |
| **Staypoints** | Parado ≥ 5 min num raio de 100 m | ~1 734 (~86% do tempo) |
| **Locations** | Clusters DBSCAN de staypoints | 93 (partilhadas pelos 3 utilizadores) |
| **Triplegs** | Segmentos de movimento entre staypoints | ~175 |
| **Trips** | Triplegs agrupados por origem/destino | ~175 |

**Nota:** a coluna de geometria das Locations chama-se `center` — usar `locs.set_geometry("center")`.

---

## 10. Arquitectura de privacidade

```
GPS original → 1. Supressão 200m em torno da casa
             → 2. Snap-to-road (nó viário mais próximo)
             → 3. K-anonimato categorias Foursquare (< 5 → "Outro")
             → Dados prontos para visualização
```

Casa = location com maior tempo de permanência entre 22:00–08:00 UTC.

---

## 11. Dashboard — estrutura actual

```
Sidebar: selector de utilizador (Todos / User 1 / User 2 / User 3)
  │
  ├── Tab 1: Mapa de Mobilidade
  │     ├── Folium (4 camadas: GPS 1/10, triplegs, staypoints ∝ dwell, locations)
  │     └── MovingPandas (atrás de botão, persistido em session_state)
  │
  ├── Tab 2: Análise Temporal
  │     ├── Heatmap hora × dia da semana (Seaborn)
  │     ├── Modos de transporte (barras horizontais)
  │     └── Distribuição de actividades por utilizador
  │
  └── Tab 3: Resumo Semanal
        ├── 4 métricas: localização principal, distância, modo dominante, raio de giração
        └── ML: LOUO (Todos) ou 80/20 (User X) + report detalhado expansível
```

---

## 12. Como correr o projecto

### Instalação
```bash
cd onde_passo_meu_tempo/
pip install -r requirements.txt
```

### Pipeline completo + ML (terminal)
```bash
python3 - <<'EOF'
from src.pipeline import run_pipeline
from src.ml import extract_features, train_and_evaluate, print_results

artefactos = run_pipeline()
features, labels, user_ids = extract_features(artefactos['raw_df'])
results = train_and_evaluate(features, labels, user_ids)
print_results(results)
EOF
```

### Dashboard
```bash
streamlit run src/dashboard.py
```

### Verificar imports
```bash
python3 -c "import src.pipeline; print('pipeline OK')"
python3 -c "import src.ml; print('ml OK')"
python3 -c "import src.privacy; print('privacy OK')"
python3 -c "import src.dashboard; print('dashboard OK')"
```

### Smoke test
```bash
python3 -c "
from src.pipeline import load_contextlabeler
df = load_contextlabeler('data')
print('Shape:', df.shape)
print('Utilizadores:', df['user_id'].value_counts().to_dict())
print('Labels únicas:', sorted(df['label'].unique()))
"
```

---

## 13. Variáveis de ambiente

| Variável | Para quê | Obrigatória? |
|---|---|---|
| `FSQ_API_KEY` | Categorias Foursquare nas locations | Não — sem ela: `fsq_category='Desconhecido'` |

```bash
export FSQ_API_KEY="chave_aqui"
streamlit run src/dashboard.py
```

---

## 14. Estrutura do projecto

```
onde_passo_meu_tempo/
├── CLAUDE.md          # Especificação e convenções (lido pelo Claude Code)
├── NOTES.md           # Este ficheiro
├── requirements.txt   # Dependências pinadas
├── data/
│   ├── user_1.csv
│   ├── user_2.csv
│   ├── user_3.csv
│   └── road_graph.graphml   # Cache OSMnx (gerado no primeiro run)
└── src/
    ├── __init__.py    # Shim de compatibilidade geopandas/trackintel
    ├── pipeline.py    # Pipeline principal (8 funções)
    ├── ml.py          # LOUO + 80/20 por utilizador
    ├── privacy.py     # 3 camadas de privacidade
    └── dashboard.py   # Streamlit com sidebar de utilizador
```
