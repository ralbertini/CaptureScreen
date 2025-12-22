
# -*- coding: utf-8 -*-
import os
import json
import math
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve, f1_score
import joblib
import warnings
import re

# Suprimir warning do urllib3 sobre SSL
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

# Reprodutibilidade (não elimina tudo em GPUs, mas ajuda)
tf.random.set_seed(42)
np.random.seed(42)

# ------------------------------------------------------------
# Configurações e nomes de features
# ------------------------------------------------------------
DEFAULT_OUTLIER_THRESHOLD = 50.0  # ajuste conforme seu caso

FEATURE_NAMES = [
    "multiplier", "totalvalue", "unitvalue",
    "ma_mult_5", "ma_mult_10", "ma_mult_15",
    "ma_total_5", "ma_total_10", "ma_total_15",
    "is_zero_total", "is_outlier_multiplier"
]

# ------------------------------------------------------------
# Utilitários de parsing e engenharia de atributos
# ------------------------------------------------------------
def parse_totalvalue(value_str: str) -> float:
    """
    Converte 'totalvalue' em float de forma robusta.
    Aceita formatos mistos: "45,415.15" (EN), "45.415,15" (pt-BR),
    e trata sujeiras comuns de OCR: espaços, dois-pontos, trailing pontuação, etc.
    Retorna np.nan se não conseguir.
    """
    if value_str is None:
        return np.nan
    s = str(value_str).strip()
    # normalizações básicas
    s = s.replace(" ", "").replace(":", ".")
    # remove trailing não numéricos (ex.: vírgula/ponto no fim, letras soltas)
    s = re.sub(r"[^\d\.,]+$", "", s)

    # só dígitos → tenta direto
    if re.fullmatch(r"\d+", s):
        try:
            return float(s)
        except ValueError:
            return np.nan

    # Detecta separador decimal pela última ocorrência
    last_dot = s.rfind(".")
    last_comma = s.rfind(",")
    if last_dot == -1 and last_comma == -1:
        # sem separador, tenta direto
        try:
            return float(s)
        except ValueError:
            return np.nan

    # Se o último separador é '.', consideramos ponto como decimal (EN)
    if last_dot > last_comma:
        s_clean = s.replace(",", "")  # remove milhares
        try:
            return float(s_clean)
        except ValueError:
            return np.nan
    else:
        # último separador é ',', consideramos vírgula decimal (pt-BR)
        s_clean = s.replace(".", "").replace(",", ".")
        try:
            return float(s_clean)
        except ValueError:
            return np.nan

def _moving_avg(arr: np.ndarray, start_idx: int, end_idx: int) -> float:
    """Média simples no slice [start_idx:end_idx) (end_idx exclusivo)."""
    if end_idx - start_idx <= 0:
        return 0.0
    return float(np.mean(arr[start_idx:end_idx]))

def build_row_features_from_series(
    multipliers: np.ndarray,
    totals: np.ndarray,
    i: int,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD
) -> np.ndarray:
    """
    Constrói o vetor de **11 features** p/ índice i:
    - multiplier[i]
    - totalvalue[i]
    - unitvalue = totalvalue[i] / multiplier[i]
    - ma_mult_5, ma_mult_10, ma_mult_15 (terminando em i)
    - ma_total_5, ma_total_10, ma_total_15 (terminando em i)
    - is_zero_total (flag 0/1 no frame i)
    - is_outlier_multiplier (flag 0/1 no frame i com limiar configurável)
    """
    mult = float(multipliers[i])
    tv = float(totals[i])
    unitvalue = tv / mult if mult != 0 else 0.0

    def ma(arr: np.ndarray, w: int) -> float:
        start = max(0, i - (w - 1))
        end = i + 1
        return _moving_avg(arr, start, end)

    is_zero_total = 1.0 if abs(tv) < 1e-12 else 0.0
    is_outlier_multiplier = 1.0 if mult > outlier_threshold else 0.0

    x = np.array([
        mult, tv, unitvalue,
        ma(multipliers, 5), ma(multipliers, 10), ma(multipliers, 15),
        ma(totals, 5),      ma(totals, 10),      ma(totals, 15),
        is_zero_total, is_outlier_multiplier
    ], dtype=np.float32)
    return x

def build_features_for_prediction(
    multiplier: float,
    totalvalue: float,
    mult_hist: Optional[List[float]] = None,
    total_hist: Optional[List[float]] = None,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD
) -> np.ndarray:
    """
    Constrói o vetor de **11 features** para predição online.
    mult_hist/total_hist devem conter a série histórica **incluindo o valor atual**.
    Se não houver histórico suficiente, as médias móveis são preenchidas com 0.0.
    Flags:
    - is_zero_total: 1.0 se totalvalue == 0, senão 0.0
    - is_outlier_multiplier: 1.0 se multiplier > outlier_threshold
    """
    m = 0.0 if multiplier is None or (isinstance(multiplier, float) and math.isnan(multiplier)) else float(multiplier)
    tv = 0.0 if totalvalue is None or (isinstance(totalvalue, float) and math.isnan(totalvalue)) else float(totalvalue)
    unitvalue = tv / m if m != 0 else 0.0

    def ma_from_hist(hist: Optional[List[float]], w: int) -> float:
        if not hist or len(hist) == 0:
            return 0.0
        arr = np.asarray(hist, dtype=np.float32)
        end = len(arr)
        start = max(0, end - w)
        return float(np.mean(arr[start:end]))

    is_zero_total = 1.0 if abs(tv) < 1e-12 else 0.0
    is_outlier_multiplier = 1.0 if m > outlier_threshold else 0.0

    return np.array([
        m, tv, unitvalue,
        ma_from_hist(mult_hist, 5), ma_from_hist(mult_hist, 10), ma_from_hist(mult_hist, 15),
        ma_from_hist(total_hist, 5), ma_from_hist(total_hist, 10), ma_from_hist(total_hist, 15),
        is_zero_total, is_outlier_multiplier
    ], dtype=np.float32)

# ------------------------------------------------------------
# Preparação do dataset a partir do JSON (com limpeza robusta)
# ------------------------------------------------------------
def load_and_prepare_dataset(
    json_path: str,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lê o arquivo JSON e constrói X e y:
    - Limpa/normaliza 'totalvalue' com parse robusto.
    - Filtra entradas inválidas (np.nan).
    - y[i] = 1 se next_multiplier > 2.0, caso contrário 0.
    - Features: 11 colunas (FEATURE_NAMES).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    multipliers: List[float] = []
    totalvalues: List[float] = []

    for entry in data:
        mult = entry.get('multiplier')
        tv_str = entry.get('totalvalue')
        if mult is None or tv_str is None:
            continue
        try:
            m = float(mult)
        except (TypeError, ValueError):
            continue

        tv = parse_totalvalue(tv_str)
        if np.isnan(tv):
            # ignora linhas que não parseiam
            continue

        multipliers.append(m)
        totalvalues.append(float(tv))

    multipliers = np.array(multipliers, dtype=np.float32)
    totalvalues = np.array(totalvalues, dtype=np.float32)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for i in range(len(multipliers) - 1):
        x = build_row_features_from_series(multipliers, totalvalues, i, outlier_threshold=outlier_threshold)
        X_list.append(x)
        y_list.append(1 if float(multipliers[i+1]) > 2.0 else 0)

    X = np.vstack(X_list) if len(X_list) > 0 else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y

# ------------------------------------------------------------
# Modelo Keras (classificador binário)
# ------------------------------------------------------------
def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    return model

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calcula class weights para lidar com desbalanceamento.
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {0: 1.0, 1: 1.0}
    pos_weight = n_neg / n_pos
    return {0: 1.0, 1: pos_weight}

# ------------------------------------------------------------
# Classe de alto nível: treina e prediz
# ------------------------------------------------------------
class NextMultiplierModel:
    def __init__(self, threshold: Optional[float] = None, outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD):
        self.model: Optional[tf.keras.Model] = None
        self.scaler: RobustScaler = RobustScaler()
        self.input_dim: Optional[int] = None
        self.decision_threshold: float = threshold if threshold is not None else 0.5
        self.feature_names: List[str] = FEATURE_NAMES.copy()
        self.outlier_threshold: float = outlier_threshold

    def fit(self, json_path: str, epochs: int = 200) -> Dict[str, Any]:
        """
        Treina o modelo a partir do JSON.
        Time-split para validação; sem shuffle (série temporal).
        Faz limpeza robusta de 'totalvalue' e inclui flags de zero/outlier.
        """
        X, y = load_and_prepare_dataset(json_path, outlier_threshold=self.outlier_threshold)
        self.input_dim = X.shape[1] if X.ndim == 2 else len(FEATURE_NAMES)

        if len(X) < 3:
            print("Aviso: menos de 3 amostras de treino. Treino será extremamente instável.")

        # Define tamanho de validação
        if len(X) >= 50:
            val_size = int(0.2 * len(X))
        elif len(X) >= 20:
            val_size = int(0.2 * len(X))
        elif len(X) >= 10:
            val_size = max(2, int(0.15 * len(X)))
        else:
            val_size = 0

        if val_size > 0:
            train_X, val_X = X[:-val_size], X[-val_size:]
            train_y, val_y = y[:-val_size], y[-val_size:]
        else:
            train_X, train_y = X, y
            val_X, val_y = None, None

        # Escalonamento robusto
        self.scaler.fit(train_X)
        train_X = self.scaler.transform(train_X)
        if val_X is not None:
            val_X = self.scaler.transform(val_X)

        self.model = build_model(self.input_dim)
        class_weights = compute_class_weights(train_y)

        callbacks = []
        if val_X is not None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_pr_auc', patience=20, restore_best_weights=True, mode='max'),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_pr_auc', factor=0.5, patience=10, min_lr=1e-5, mode='max'),
            ]

        # Batch size simples
        if len(train_X) > 128:
            batch_size = 32
        elif len(train_X) > 64:
            batch_size = 16
        elif len(train_X) > 20:
            batch_size = 8
        else:
            batch_size = max(2, len(train_X) // 4)

        history = self.model.fit(
            train_X, train_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_X, val_y) if val_X is not None else None,
            class_weight=class_weights,
            verbose=0,
            shuffle=False  # importante para séries temporais
        )

        metrics: Dict[str, float] = {}
        best_threshold = self.decision_threshold
        if val_X is not None:
            eval_res = self.model.evaluate(val_X, val_y, verbose=0)
            for name, val in zip(self.model.metrics_names, eval_res):
                metrics[name] = float(val)

            # Determina threshold ótimo por F1 na validação (opcional)
            val_probs = self.model.predict(val_X, verbose=0).ravel()
            precision, recall, thresholds = precision_recall_curve(val_y, val_probs)
            f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
            idx = int(np.nanargmax(f1_scores)) if len(f1_scores) > 0 else None
            if idx is not None and idx < len(thresholds):
                best_threshold = float(thresholds[idx])

        self.decision_threshold = best_threshold

        return {
            "n_train": int(len(train_X)),
            "n_val": int(val_size),
            "class_weights": class_weights,
            "val_metrics": metrics,
            "final_epoch": len(history.history['loss']),
            "threshold": self.decision_threshold,
            "feature_names": self.feature_names,
            "outlier_threshold": self.outlier_threshold
        }

    def _ensure_model(self):
        if self.model is None or self.scaler is None or self.input_dim is None:
            raise RuntimeError("Modelo não treinado/carregado. Chame .fit(json_path) ou .load(path) antes.")

    def predict_prob(
        self,
        multiplier: float,
        totalvalue: float,
        mult_hist: Optional[List[float]] = None,
        total_hist: Optional[List[float]] = None
    ) -> float:
        """
        Retorna a probabilidade do próximo multiplier ser > 2.0.
        Usa **exatamente** as 11 features do treino (incluindo flags de zero/outlier
        e médias móveis de 5/10/15).
        """
        self._ensure_model()
        x = build_features_for_prediction(
            multiplier, totalvalue, mult_hist, total_hist, outlier_threshold=self.outlier_threshold
        ).reshape(1, -1)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Dimensão de features inválida: {x.shape[1]} != {self.input_dim}")
        x = self.scaler.transform(x)
        prob = float(self.model.predict(x, verbose=0)[0][0])
        return prob

    def predict_label(
        self,
        multiplier: float,
        totalvalue: float,
        mult_hist: Optional[List[float]] = None,
        total_hist: Optional[List[float]] = None
    ) -> int:
        """
        Retorna 1/0 usando o threshold aprendido na validação.
        """
        p = self.predict_prob(multiplier, totalvalue, mult_hist, total_hist)
        return int(p >= self.decision_threshold)

    def save(self, path: str = "next_multiplier_model.keras"):
        self._ensure_model()
        self.model.save(path)
        scaler_path = path.replace('.keras', '.scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        # Salva metadados úteis
        meta = {
            "input_dim": self.input_dim,
            "feature_names": self.feature_names,
            "decision_threshold": self.decision_threshold,
            "outlier_threshold": self.outlier_threshold
        }
        with open(path.replace('.keras', '.meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def load(self, path: str = "next_multiplier_model.keras"):
        self.model = tf.keras.models.load_model(path)
        scaler_path = path.replace('.keras', '.scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        meta_path = path.replace('.keras', '.meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.input_dim = meta.get("input_dim", len(FEATURE_NAMES))
            self.feature_names = meta.get("feature_names", FEATURE_NAMES)
            self.decision_threshold = meta.get("decision_threshold", 0.5)
            self.outlier_threshold = meta.get("outlier_threshold", DEFAULT_OUTLIER_THRESHOLD)
        else:
            self.input_dim = len(FEATURE_NAMES)

# ------------------------------------------------------------
# Exemplo de uso
# ------------------------------------------------------------
if __name__ == "__main__":
    json_path = "screen_data.json"

    model = NextMultiplierModel(outlier_threshold=DEFAULT_OUTLIER_THRESHOLD)
    info = model.fit(json_path=json_path, epochs=200)
    print("Resumo do treino:", info)
    print("Modelo Treinado!!")

    model.save("next_multiplier_model.keras")

    # Exemplo de predição com histórico (incluindo valor atual):
    # mult_hist = [1.10, 1.12, 1.14, 1.09, 1.07]
    # total_hist = [120.0, 130.0, 125.0, 110.0, 100.0]
    # prob = model.predict_prob(
    #     multiplier=1.07, totalvalue=100.0,
    #     mult_hist=mult_hist, total_hist=total_hist
    # )
    # print("Probabilidade next_multiplier > 2.0:", prob)
