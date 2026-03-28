"""ml.py — Classificação de actividade com scikit-learn.

Implementa leave-one-user-out cross-validation com três modelos:
  GaussianNB, DecisionTreeClassifier, RandomForestClassifier.

NOTA IMPORTANTE — porque extract_features recebe raw_df e não staypoints:
  A tarefa de ML é classificação de actividade ao nível da amostra (uma label
  por janela de 60 s). Os staypoints agregam múltiplas amostras numa só linha,
  o que faria perder o detalhe sensor (wifi, bateria, aceleração) necessário
  como features. Por isso, extract_features opera sobre o DataFrame bruto
  devolvido por load_contextlabeler(), incluído em run_pipeline()['raw_df'].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# ── Constantes ────────────────────────────────────────────────────────────────

ACCEL_COLS: list[str] = [
    "sensor_linear_acc_x_mean",
    "sensor_linear_acc_y_mean",
    "sensor_linear_acc_z_mean",
]
BATTERY_COL: str = "battery_unplugged"   # Binário: 1 = sem carregar (em mobilidade)
WIFI_COL: str    = "wifi_connected"       # Binário: 1 = WiFi ligado

RANDOM_STATE: int    = 42
RF_N_ESTIMATORS: int = 100

FEATURE_COLS: list[str] = [
    "hour_sin",     # Codificação cíclica da hora (seno)
    "hour_cos",     # Codificação cíclica da hora (cosseno)
    "day_of_week",  # Dia da semana (0=Segunda, 6=Domingo)
    "avg_speed",    # Velocidade média estimada (km/h)
    "wifi_count",   # WiFi ligado (float 0.0/1.0)
    "battery",      # Bateria desligada do carregador (float 0.0/1.0)
    "linear_accel", # Magnitude da aceleração linear (norma Euclidiana dos 3 eixos)
]


# ── Funções principais ────────────────────────────────────────────────────────

def extract_features(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extrai features de ML por amostra a partir do DataFrame raw do ContextLabeler.

    Parâmetros
    ----------
    raw_df : pd.DataFrame
        DataFrame devolvido por load_contextlabeler() — inclui colunas:
        time, location_lat, location_lon, label, wifi_connected,
        battery_unplugged, sensor_linear_acc_x/y/z_mean, user_id.

    Retorna
    -------
    tuple[pd.DataFrame, pd.Series, pd.Series]
        (features_df, labels, user_ids)
        features_df tem as colunas de FEATURE_COLS.
        labels é a série de rótulos de actividade.
        user_ids é a série de identificadores de utilizador (1, 2 ou 3).
    """
    df = raw_df.copy()

    # Ordenar por utilizador e tempo (necessário para velocidade vectorizada)
    df = df.sort_values(["user_id", "time"]).reset_index(drop=True)

    # ── Features temporais (codificação cíclica da hora) ──────────────────────
    tracked = pd.to_datetime(df["time"], unit="ms", utc=True)
    hour = tracked.dt.hour + tracked.dt.minute / 60.0
    df["hour_sin"]    = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"]    = np.cos(2.0 * np.pi * hour / 24.0)
    df["day_of_week"] = tracked.dt.dayofweek  # 0 = Segunda-feira

    # ── Velocidade média haversine vectorizada por utilizador ─────────────────
    # Usa groupby para não contaminar a fronteira entre utilizadores
    avg_speed = pd.Series(0.0, index=df.index)
    for uid, grp in df.groupby("user_id"):
        idx  = grp.index
        lats = grp["location_lat"].values
        lons = grp["location_lon"].values
        t_s  = grp["time"].values / 1_000.0  # ms → s

        prev_lats = np.roll(lats, 1)
        prev_lons = np.roll(lons, 1)
        prev_ts   = np.roll(t_s,  1)

        # Haversine vectorizado (reutiliza a mesma lógica de pipeline.py)
        R = 6_371_000.0
        dphi    = np.radians(lats - prev_lats)
        dlambda = np.radians(lons - prev_lons)
        phi1    = np.radians(prev_lats)
        phi2    = np.radians(lats)
        a = (
            np.sin(dphi / 2.0) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
        )
        a = np.clip(a, 0.0, 1.0)
        dist_m = R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

        dt_s = t_s - prev_ts
        with np.errstate(divide="ignore", invalid="ignore"):
            spd = np.where(dt_s > 0, (dist_m / 1_000.0) / (dt_s / 3_600.0), 0.0)
        spd[0] = 0.0  # primeiro ponto do utilizador sem referência anterior
        avg_speed[idx] = spd

    df["avg_speed"] = avg_speed

    # ── Features de sensor ────────────────────────────────────────────────────
    df["wifi_count"]   = df[WIFI_COL].astype(float)
    df["battery"]      = df[BATTERY_COL].astype(float)
    # Magnitude da aceleração linear: norma Euclidiana dos 3 eixos
    df["linear_accel"] = np.sqrt(
        df["sensor_linear_acc_x_mean"] ** 2
        + df["sensor_linear_acc_y_mean"] ** 2
        + df["sensor_linear_acc_z_mean"] ** 2
    )

    features  = df[FEATURE_COLS].fillna(0.0).reset_index(drop=True)
    labels    = df["label"].reset_index(drop=True)
    user_ids  = df["user_id"].reset_index(drop=True)

    return features, labels, user_ids


def train_and_evaluate(
    features: pd.DataFrame,
    labels: pd.Series,
    user_ids: pd.Series,
) -> list[dict]:
    """Treina e avalia três modelos com leave-one-user-out cross-validation.

    Para cada fold (3 no total), treina nos 2 utilizadores restantes e testa
    no utilizador deixado de fora. Esta estratégia avalia a generalização
    cross-utilizador — mais relevante do que k-fold intra-utilizador.

    NOTA sobre desequilíbrio de classes: as classes 'Nightlife', 'Physical
    exercise' e 'Shopping' aparecem em apenas 1–2 utilizadores. O modelo
    treinado num fold pode nunca ter visto essas classes → F1=0 para essas
    classes. Este comportamento é esperado e deve ser documentado na tese.

    Parâmetros
    ----------
    features : pd.DataFrame
        DataFrame de features devolvido por extract_features().
    labels : pd.Series
        Série de rótulos de actividade.
    user_ids : pd.Series
        Série com o identificador do utilizador por amostra.

    Retorna
    -------
    list[dict]
        Lista com um dict por fold. Cada dict tem:
        - 'test_user': int — utilizador usado como conjunto de teste
        - 'models': dict — chave=nome_modelo, valor=dict com accuracy,
          f1_macro e classification_report.
    """
    models: dict = {
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
        ),
    }

    unique_users = sorted(user_ids.unique())  # [1, 2, 3]
    results: list[dict] = []

    for test_user in unique_users:
        train_mask = user_ids != test_user
        test_mask  = user_ids == test_user

        X_train = features[train_mask].values
        X_test  = features[test_mask].values
        y_train = labels[train_mask].values
        y_test  = labels[test_mask].values

        fold_result: dict = {"test_user": int(test_user), "models": {}}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # zero_division=0 evita erros para classes nunca vistas no treino
            fold_result["models"][model_name] = {
                "accuracy":  accuracy_score(y_test, y_pred),
                "f1_macro":  f1_score(y_test, y_pred, average="macro", zero_division=0),
                "report":    classification_report(y_test, y_pred, zero_division=0),
            }

        results.append(fold_result)

    return results


def print_results(results: list[dict]) -> None:
    """Imprime as métricas de avaliação por fold e as médias globais.

    Parâmetros
    ----------
    results : list[dict]
        Lista devolvida por train_and_evaluate().
    """
    sep = "=" * 60

    for fold in results:
        print(f"\n{sep}")
        print(f"Utilizador {fold['test_user']} como conjunto de teste")
        print(sep)
        for model_name, metrics in fold["models"].items():
            print(f"\n  Modelo: {model_name}")
            print(f"  Acurácia : {metrics['accuracy']:.4f}")
            print(f"  F1-Macro : {metrics['f1_macro']:.4f}")
            # classification_report já inclui indentação própria
            for line in metrics["report"].splitlines():
                print(f"    {line}")

    # ── Médias entre todos os folds ───────────────────────────────────────────
    print(f"\n{sep}")
    print("Médias entre todos os folds (leave-one-user-out)")
    print(sep)
    model_names = list(results[0]["models"].keys())
    for model_name in model_names:
        avg_acc = float(np.mean([f["models"][model_name]["accuracy"] for f in results]))
        avg_f1  = float(np.mean([f["models"][model_name]["f1_macro"] for f in results]))
        print(f"  {model_name:<22s}  Acurácia={avg_acc:.4f}  F1-Macro={avg_f1:.4f}")
