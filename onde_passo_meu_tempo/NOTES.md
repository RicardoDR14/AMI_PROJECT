# Notas de Implementação — Onde Passo o Meu Tempo?

Guia técnico de referência para o pipeline de mobilidade.
Explica as decisões de implementação, o fluxo de dados e como correr o projecto.

---

## 1. Porquê estas bibliotecas?

| Biblioteca | Papel | Nota |
|---|---|---|
| **Trackintel** | GPS → staypoints → locations → trips | Construída especificamente para mobilidade; trata gaps, ruído e clustering num único pipeline |
| **MovingPandas** | Visualização interactiva de trajectórias | Integração HoloViz; `tc.explore()` gera mapa animado sem configuração extra |
| **scikit-learn** | Classificador de actividade | Standard; fácil de trocar modelos sem mudar o restante código |
| **OSMnx** | Download de rede viária + snap-to-road | Acesso directo ao OpenStreetMap; integra com networkx |
| ~~scikit-mobility~~ | ~~Métricas de mobilidade~~ | **Excluído**: fixa `pandas<2.0` e `geopandas<0.11`, incompatível com o stack actual. Substituído por `trackintel.analysis.metrics` para radius_gyration e location_entropy |

---

## 2. Nomes de colunas reais vs. CLAUDE.md

O CLAUDE.md usa nomes genéricos. As colunas reais dos CSVs do ContextLabeler são:

| CLAUDE.md | Coluna real no CSV | Notas |
|---|---|---|
| `latitude` | `location_lat` | Graus decimais WGS-84 |
| `longitude` | `location_lon` | Graus decimais WGS-84 |
| `timestamp` | `time` | Unix **milissegundos** → dividir por 1000 para segundos |
| `activity_label` | `label` | Valores: Sleep, Home, Working, Break, Lunch Break, Free time, Restaurant, Nightlife, Physical exercise, Shopping |
| `battery` | `battery_unplugged` | Binário: 1 = sem carregar (em mobilidade), 0 = a carregar |
| `linear_accel` | `sensor_linear_acc_x/y/z_mean` | Três eixos separados → calcular norma Euclidiana |
| `wifi_count` | `wifi_connected` | Binário |

Os CSVs têm 1 333 colunas; `load_contextlabeler` usa `usecols=COLS_NEEDED` para carregar apenas 9, poupando ~98% de I/O.

---

## 3. Fluxo de dados

```
data/user_1.csv
data/user_2.csv   ──► load_contextlabeler()  ──► raw_df (pd.DataFrame, 9 colunas)
data/user_3.csv                                        │
                                                       ├──► extract_features()  [ml.py]
                                                       │         │
                                                       │         ▼
                                                       │    train_and_evaluate()
                                                       │         │
                                                       │         ▼
                                                       │    print_results()
                                                       │
                                                       ▼
                                               build_positionfixes()
                                               (remoção de outliers GPS)
                                                       │
                                                       ▼
                                               build_trajectories()
                                               (MovingPandas QA visual)
                                                       │
                                                       ▼
                                               segment_trajectories()
                                               → spts + tpls
                                                       │
                                                       ▼
                                               merge_and_detect_locations()
                                               → spts + locs
                                               + Silhouette/DB impressos
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
```

---

## 4. Porquê `extract_features` recebe `raw_df` em vez de `staypoints`

A tarefa de ML é **classificação de actividade ao nível da amostra** — cada linha do CSV (janela de 60 s) já tem um rótulo (`label`). Os staypoints agregam múltiplas amostras numa só linha, o que eliminaria o detalhe de sensor (wifi, bateria, aceleração) necessário como features.

O dicionário de `run_pipeline()` inclui `raw_df` para que `extract_features` possa aceder aos dados directamente:

```python
from src.pipeline import run_pipeline
from src.ml import extract_features, train_and_evaluate, print_results

artefactos = run_pipeline()
features, labels, user_ids = extract_features(artefactos['raw_df'])
results = train_and_evaluate(features, labels, user_ids)
print_results(results)
```

---

## 5. Porquê leave-one-user-out em vez de k-fold

Com k-fold padrão, os dados dos 3 utilizadores seriam misturados em treino/teste. Isso mediria "conseguimos classificar actividades de um utilizador que já vimos?" — pouco útil para generalização.

**Leave-one-user-out** responde à pergunta certa: "o modelo funciona para um utilizador **novo**?"

### Limitação conhecida — desequilíbrio de classes

Três classes aparecem em apenas 1–2 utilizadores:

| Classe | Utilizadores |
|---|---|
| Nightlife | Só user_1 |
| Physical exercise | Só user_2 |
| Shopping | Só user_3 |

Quando o utilizador com uma dessas classes é o conjunto de teste, o modelo treinado nos outros 2 nunca viu essa classe → F1 = 0 para essa classe. Isto é esperado e deve ser mencionado na tese como limitação do tamanho do dataset.

O código usa `zero_division=0` no `classification_report` para evitar erros nesta situação.

---

## 6. Porquê `osmnx` deve aparecer antes de `trackintel` em requirements.txt

`trackintel` lista `osmnx` como dependência sem fixar a versão. Sem um pin explícito, o `pip` instala a última versão do osmnx (2.x), que exige Python 3.11+.

Ao colocar `osmnx==1.9.4` **antes** de `trackintel==1.4.2` no ficheiro, o resolver do pip respeita o pin antes de encontrar a dependência mais permissiva do trackintel.

---

## 7. Cache da rede viária OSMnx

O passo de snap-to-road descarrega a rede viária para toda a bounding box GPS (região Pisa/Toscana, ~100 km × 55 km). Este download pode demorar vários minutos e requer ~500 MB de RAM.

O grafo é guardado em `data/road_graph.graphml` após o primeiro download. Chamadas subsequentes carregam do ficheiro em segundos.

Para forçar um novo download (p. ex. se a rede viária foi actualizada):
```bash
rm data/road_graph.graphml
```

---

## 8. Modelo de dados Trackintel (simplificado)

```
Positionfixes  ──►  Staypoints  ──►  Locations
                         │
                         ▼
                     Triplegs   ──►  Trips
```

| Artefacto | O que é | Neste dataset |
|---|---|---|
| **Positionfixes** | Pontos GPS brutos | 1 por 60 s |
| **Staypoints** | Onde o utilizador ficou parado ≥ 5 min num raio de 100 m | Maioria dos pontos (~86% estacionários) |
| **Locations** | Clusters de staypoints em locais repetidos (DBSCAN) | Casa, Trabalho, etc. |
| **Triplegs** | Segmentos de movimento entre staypoints | Poucos, dado o alto ratio de estacionaridade |
| **Trips** | Triplegs agrupados por origem/destino | Derivados dos triplegs |

**Nota importante sobre a geometria das Locations:** a coluna de geometria chama-se `center` (não `geometry`). Qualquer acesso deve usar `locs.set_geometry("center")` ou `locs["center"]`.

---

## 9. Arquitectura de privacidade — 3 camadas independentes

As camadas são aplicadas sequencialmente em `apply_privacy()` mas podem ser desactivadas individualmente:

```
Dados GPS originais
       │
       ▼
1. Supressão domiciliar
   Remove todos os pontos num raio de 200 m em torno da casa.
   Casa = location com maior tempo de permanência entre 22:00–08:00 UTC.
       │
       ▼
2. Snap-to-road (OSMnx)
   Substitui coordenadas GPS exactas pelo nó viário mais próximo.
   Impede inferir o edifício exacto a partir das coordenadas.
       │
       ▼
3. K-anonimato nas categorias Foursquare
   Categorias com < 5 ocorrências → "Outro".
   Impede re-identificação por tipos de local únicos.
       │
       ▼
Dados prontos para visualização pública
```

**Regra fundamental (CLAUDE.md):** `apply_privacy()` deve ser chamado **antes** de qualquer visualização ou exportação de dados.

---

## 10. Como correr o projecto

### Instalação
```bash
cd onde_passo_meu_tempo/
pip install -r requirements.txt
```

### Pipeline completo + ML
```bash
python - <<'EOF'
from src.pipeline import run_pipeline
from src.ml import extract_features, train_and_evaluate, print_results

artefactos = run_pipeline()          # ~2–5 min (primeiro run)
features, labels, user_ids = extract_features(artefactos['raw_df'])
results = train_and_evaluate(features, labels, user_ids)
print_results(results)
EOF
```

### Dashboard Streamlit
```bash
streamlit run src/dashboard.py
```

### Verificar imports
```bash
python -c "import src.pipeline; print('pipeline OK')"
python -c "import src.ml; print('ml OK')"
python -c "import src.privacy; print('privacy OK')"
python -c "import src.dashboard; print('dashboard OK')"
```

### Smoke test rápido
```bash
python -c "
from src.pipeline import load_contextlabeler
df = load_contextlabeler('data')
print('Shape:', df.shape)
print('Utilizadores:', df['user_id'].value_counts().to_dict())
print('Labels únicas:', sorted(df['label'].unique()))
"
```

---

## 11. Variáveis de ambiente opcionais

| Variável | Para quê | Obrigatória? |
|---|---|---|
| `FSQ_API_KEY` | Enriquecimento semântico das locations via Foursquare Places API | Não — sem ela, `fsq_category = 'Desconhecido'` |

```bash
# Exportar antes de correr o pipeline
export FSQ_API_KEY="sua_chave_aqui"
python -c "from src.pipeline import run_pipeline; run_pipeline()"
```

---

## 12. Estrutura do projecto

```
onde_passo_meu_tempo/
├── CLAUDE.md          # Especificação do projecto (lida automaticamente pelo Claude Code)
├── NOTES.md           # Este ficheiro — decisões de implementação e guia de uso
├── requirements.txt   # Dependências pinadas (Python 3.10+)
├── data/
│   ├── user_1.csv     # ContextLabeler — 8 456 amostras × 1 333 features
│   ├── user_2.csv     # ContextLabeler — 17 882 amostras
│   ├── user_3.csv     # ContextLabeler — 19 343 amostras
│   └── road_graph.graphml  # Cache OSMnx (criado no primeiro run)
└── src/
    ├── __init__.py    # Torna src/ um pacote Python importável
    ├── pipeline.py    # Pipeline principal (load → positionfixes → staypoints → trips)
    ├── ml.py          # Classificação de actividade + leave-one-user-out
    ├── privacy.py     # Filtros de privacidade (supressão + snap-to-road + k-anonimato)
    └── dashboard.py   # Interface Streamlit (3 separadores)
```
