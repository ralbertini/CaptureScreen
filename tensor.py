
import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from typing import Tuple, List, Dict, Any

# Fixar sementes para reprodutibilidade (não elimina variação com poucos dados)
tf.random.set_seed(42)
np.random.seed(42)

# ------------------------------------------------------------
# Utilitários de parsing e engenharia de atributos
# ------------------------------------------------------------
def parse_totalvalue(value_str: str) -> float:
    """
    Converte o campo totalvalue para float de forma robusta.
    Aceita formatos: "45,415.15" (inglês) ou "45.415,15" (português/br).
    """
    if value_str is None:
        return np.nan
    s = value_str.strip()
    # Tenta formato US/EN: vírgula para milhares, ponto decimal
    try:
        return float(s.replace(',', ''))
    except ValueError:
        pass
    # Tenta formato BR: ponto para milhares, vírgula decimal
    try:
        s2 = s.replace('.', '').replace(',', '.')
        return float(s2)
    except ValueError:
        # Se não parseou, retorna NaN
        return np.nan

def to_datetime_iso(ts: str) -> datetime:
    """Converte timestamp ISO (ex.: '2025-12-15T16:41:40.496986') para datetime."""
    return pd.to_datetime(ts)

def time_cyc_features(dt: datetime) -> Tuple[float, float]:
    """
    Converte horário em features cíclicas (sin/cos) ao longo de 24h.
    Isso evita descontinuidade entre 23:59 e 00:00.
    """
    seconds_in_day = 24 * 60 * 60
    sec = dt.hour * 3600 + dt.minute * 60 + dt.second
    angle = 2 * math.pi * (sec / seconds_in_day)
    return math.sin(angle), math.cos(angle)

def build_feature_vector(ts_iso: str, multiplier: float, totalvalue: float) -> np.ndarray:
    """
    Cria o vetor de atributos X para inferência.
    - multiplier
    - totalvalue (numérico)
    - unitvalue = totalvalue / multiplier
    - time_sin, time_cos (cíclicos em 24h)
    """
    dt = to_datetime_iso(ts_iso)
    time_sin, time_cos = time_cyc_features(dt)

    if multiplier is None or (isinstance(multiplier, float) and np.isnan(multiplier)):
        multiplier = 0.0

    tv = float(totalvalue) if totalvalue is not None else np.nan
    if np.isnan(tv):
        tv = 0.0

    unitvalue = tv / multiplier if (multiplier is not None and multiplier != 0) else 0.0

    return np.array([multiplier, tv, unitvalue, time_sin, time_cos], dtype=np.float32)

# ------------------------------------------------------------
# Preparação do dataset a partir do JSON
# ------------------------------------------------------------
def load_and_prepare_dataset(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lê o arquivo JSON e constrói X e y para o problema:
    y[i] = 1 se next_multiplier > 2.0, caso contrário 0.

    Features por linha i:
      - multiplier[i]
      - totalvalue_num[i]
      - unitvalue[i] = totalvalue_num[i] / multiplier[i]
      - time_sin[i], time_cos[i] (ciclo de 24h)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    # Converter timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Normalizar totalvalue
    df['totalvalue_num'] = df['totalvalue'].apply(parse_totalvalue).astype(float)
    # Calcular unitvalue
    df['unitvalue'] = df['totalvalue_num'] / df['multiplier']
    df['unitvalue'] = df['unitvalue'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Ordenar por tempo
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Features cíclicas
    seconds_in_day = 24 * 60 * 60
    secs = df['timestamp'].dt.hour * 3600 + df['timestamp'].dt.minute * 60 + df['timestamp'].dt.second
    angles = 2 * math.pi * (secs / seconds_in_day)
    df['time_sin'] = np.sin(angles)
    df['time_cos'] = np.cos(angles)

    # Construir X para linhas 0..n-2 e y baseado em multiplier[i+1]
    X_cols = ['multiplier', 'totalvalue_num', 'unitvalue', 'time_sin', 'time_cos']
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        x = row[X_cols].values.astype(np.float32)
        y = 1 if next_row['multiplier'] > 2.0 else 0
        X_list.append(x)
        y_list.append(y)

    X = np.vstack(X_list) if len(X_list) > 0 else np.empty((0, len(X_cols)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y

# ------------------------------------------------------------
# Modelo Keras (classificador binário)
# ------------------------------------------------------------
def build_model(input_dim: int, train_X: np.ndarray) -> tf.keras.Model:
    """
    Constrói um MLP simples com camada de Normalization adaptada ao train_X.
    """
    normalizer = tf.keras.layers.Normalization(axis=-1)
    if train_X.shape[0] > 0:
        normalizer.adapt(train_X)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        normalizer,
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # probabilidade
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calcula class weights para lidar com desbalanceamento.
    Se não houver positivos/negativos, retorna pesos iguais.
    """
    n_pos = int(np.sum(y))
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {0: 1.0, 1: 1.0}
    # Peso da classe positiva proporcional ao desequilíbrio
    pos_weight = n_neg / n_pos
    return {0: 1.0, 1: pos_weight}

# ------------------------------------------------------------
# Classe de alto nível: treina e prediz
# ------------------------------------------------------------
class NextMultiplierModel:
    def __init__(self):
        self.model: tf.keras.Model = None
        self.input_dim: int = 5  # [multiplier, totalvalue, unitvalue, time_sin, time_cos]

    def fit(self, json_path: str, epochs: int = 100) -> Dict[str, Any]:
        """
        Treina o modelo a partir do JSON.
        Faz split temporal simples: últimas amostras para validação (se existirem).
        """
        X, y = load_and_prepare_dataset(json_path)

        if len(X) < 3:
            print("Aviso: menos de 3 amostras de treino. Treino será extremamente instável.")
        # Split temporal: último 20% para validação, ao menos 1 (se possível)
        if len(X) >= 5:
            val_size = max(1, int(0.2 * len(X)))
        elif len(X) >= 2:
            val_size = 1
        else:
            val_size = 0

        if val_size > 0:
            train_X, val_X = X[:-val_size], X[-val_size:]
            train_y, val_y = y[:-val_size], y[-val_size:]
            validation_data = (val_X, val_y)
        else:
            train_X, train_y = X, y
            val_X, val_y = None, None
            validation_data = None

        self.model = build_model(self.input_dim, train_X)
        class_weights = compute_class_weights(train_y)

        callbacks = []
        if validation_data is not None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True)
            ]

        history = self.model.fit(
            train_X, train_y,
            epochs=epochs,
            batch_size=max(1, min(8, len(train_X))),
            validation_data=validation_data,
            class_weight=class_weights,
            verbose=0,
            callbacks=callbacks
        )

        metrics = {}
        if validation_data is not None:
            eval_res = self.model.evaluate(val_X, val_y, verbose=0)
            for name, val in zip(self.model.metrics_names, eval_res):
                metrics[name] = float(val)

        return {
            "n_train": int(len(train_X)),
            "n_val": int(val_size),
            "class_weights": class_weights,
            "val_metrics": metrics
        }

    def predict_prob(self, timestamp_iso: str, multiplier: float, totalvalue: float) -> float:
        """
        Retorna a probabilidade do próximo multiplier ser > 2.0,
        dado o estado atual (timestamp_iso, multiplier, totalvalue).
        """
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Chame .fit(json_path) antes.")

        x = build_feature_vector(timestamp_iso, multiplier, totalvalue).reshape(1, -1)
        prob = float(self.model.predict(x, verbose=0)[0][0])
        return prob

    def save(self, path: str = "next_multiplier_model.keras"):
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Nada para salvar.")
        self.model.save(path)

    def load(self, path: str = "next_multiplier_model.keras"):
        self.model = tf.keras.models.load_model(path)

# ------------------------------------------------------------
# Exemplo de uso
# ------------------------------------------------------------
if __name__ == "__main__":
    # Caminho do arquivo JSON (substitua pelo seu)
    json_path = "screen_data.json"

    model = NextMultiplierModel()
    info = model.fit(json_path=json_path, epochs=150)
    print("Resumo do treino:", info)

    # Exemplo de predição: forneça data/hora atual ISO, multiplier e totalvalue
    ts_now = "2025-12-15T16:47:00"  # exemplo
    mult_now = 1.10
    total_now = 210.00

    prob = model.predict_prob(ts_now, mult_now, total_now)
    print(f"Probabilidade de próximo multiplier > 2.0: {prob:.4f}")

    # Salvar modelo (opcional)
    model.save("next_multiplier_model.keras")

