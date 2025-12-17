
import json
import math
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import RobustScaler
import joblib
import warnings

# Suprimir warning do urllib3 sobre SSL
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

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

def build_feature_vector(multiplier: float, totalvalue: float) -> np.ndarray:
    """
    Cria o vetor de atributos X para inferência.
    - multiplier
    - totalvalue (numérico)
    - unitvalue = totalvalue / multiplier
    """
    if multiplier is None or (isinstance(multiplier, float) and np.isnan(multiplier)):
        multiplier = 0.0

    tv = float(totalvalue) if totalvalue is not None else np.nan
    if np.isnan(tv):
        tv = 0.0

    unitvalue = tv / multiplier if (multiplier is not None and multiplier != 0) else 0.0

    return np.array([multiplier, tv, unitvalue], dtype=np.float32)

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
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Processar dados sem pandas
    multipliers = []
    totalvalues = []
    
    for entry in data:
        mult = entry.get('multiplier')
        tv_str = entry.get('totalvalue')
        if mult is not None and tv_str is not None:
            tv = parse_totalvalue(tv_str)
            if not np.isnan(tv):
                multipliers.append(float(mult))
                totalvalues.append(tv)
    
    multipliers = np.array(multipliers, dtype=np.float32)
    totalvalues = np.array(totalvalues, dtype=np.float32)
    
    # Construir X para linhas 0..n-2 e y baseado em multiplier[i+1]
    X_list = []
    y_list = []
    
    for i in range(len(multipliers) - 1):
        mult = multipliers[i]
        tv = totalvalues[i]
        unitvalue = tv / mult if mult != 0 else 0.0
        
        x = np.array([mult, tv, unitvalue], dtype=np.float32)
        y = 1 if multipliers[i + 1] > 2.0 else 0
        
        X_list.append(x)
        y_list.append(y)
    
    X = np.vstack(X_list) if len(X_list) > 0 else np.empty((0, 3), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    return X, y

# ------------------------------------------------------------
# Modelo Keras (classificador binário)
# ------------------------------------------------------------
def build_model(input_dim: int) -> tf.keras.Model:
    """
    Constrói um MLP aprimorado com BatchNormalization e mais camadas para melhor performance.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
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
        self.scaler: RobustScaler = RobustScaler()
        self.input_dim: int = 3  # [multiplier, totalvalue, unitvalue]

    def fit(self, json_path: str, epochs: int = 300) -> Dict[str, Any]:
        """
        Treina o modelo a partir do JSON.
        Ajustado para dataset maior: mais epochs possíveis, validação robusta.
        """
        X, y = load_and_prepare_dataset(json_path)

        if len(X) < 3:
            print("Aviso: menos de 3 amostras de treino. Treino será extremamente instável.")
        
        # Para dataset maior, usar validação adequada
        if len(X) >= 20:
            val_size = max(5, int(0.2 * len(X)))  # 20% para validação
        elif len(X) >= 10:
            val_size = max(2, int(0.15 * len(X)))
        else:
            val_size = 0

        if val_size > 0:
            train_X, val_X = X[:-val_size], X[-val_size:]
            train_y, val_y = y[:-val_size], y[-val_size:]
            # Aplicar RobustScaler
            self.scaler.fit(train_X)
            train_X = self.scaler.transform(train_X)
            val_X = self.scaler.transform(val_X)
            validation_data = (val_X, val_y)
        else:
            train_X, train_y = X, y
            val_X, val_y = None, None
            # Aplicar RobustScaler mesmo sem validação
            self.scaler.fit(train_X)
            train_X = self.scaler.transform(train_X)
            validation_data = None

        self.model = build_model(self.input_dim)
        class_weights = compute_class_weights(train_y)

        callbacks = []
        if validation_data is not None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=20, restore_best_weights=True, mode='max'),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=10, min_lr=1e-5, mode='max')
            ]

        # Batch size adaptativo
        if len(train_X) > 50:
            batch_size = 8
        elif len(train_X) > 20:
            batch_size = 4
        else:
            batch_size = max(2, len(train_X) // 4)
        
        history = self.model.fit(
            train_X, train_y,
            epochs=epochs,
            batch_size=batch_size,
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
            "val_metrics": metrics,
            "final_epoch": len(history.history['loss'])
        }

    def predict_prob(self, multiplier: float, totalvalue: float) -> float:
        """
        Retorna a probabilidade do próximo multiplier ser > 2.0,
        dado o estado atual (multiplier, totalvalue).
        """
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Chame .fit(json_path) antes.")

        x = build_feature_vector(multiplier, totalvalue).reshape(1, -1)
        x = self.scaler.transform(x)
        prob = float(self.model.predict(x, verbose=0)[0][0])
        return prob

    def save(self, path: str = "next_multiplier_model.keras"):
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Nada para salvar.")
        self.model.save(path)
        scaler_path = path.replace('.keras', '.scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

    def load(self, path: str = "next_multiplier_model.keras"):
        self.model = tf.keras.models.load_model(path)
        scaler_path = path.replace('.keras', '.scaler.pkl')
        self.scaler = joblib.load(scaler_path)

# ------------------------------------------------------------
# Exemplo de uso
# ------------------------------------------------------------
if __name__ == "__main__":
    # Caminho do arquivo JSON de treinamento
    json_path = "screen_data.json"

    model = NextMultiplierModel()
    info = model.fit(json_path=json_path, epochs=300)
    print("Resumo do treino:", info)

    # Exemplo de predição: forneça multiplier e totalvalue
    #mult_now = 1.10
    #total_now = 210.00

    #prob = model.predict_prob(mult_now, total_now)
    print(f"Modelo Treinado!!")

    # Salvar modelo (opcional)
    model.save("next_multiplier_model.keras")

