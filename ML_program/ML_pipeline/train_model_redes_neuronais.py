import numpy as np
import os
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from ML_program.data_preprocessing.database_setup.config import get_session
from ML_program.data_preprocessing.database_setup.models import Comments, Films

# Configuração inicial
np.random.seed(42)
OUTPUT_DIR = "resultados_testes_modelo_com_2_camadas_ocultas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# **1. Carregar Embeddings e Dados**
X_train_vec = np.load("X_train_vec.npy")
X_test_vec = np.load("X_test_vec.npy")

# Carregar rótulos
session = get_session()
comments = session.query(Comments).all()
session.close()

y = [session.query(Films).filter(Films.imdb_id == comment.imdb_id).one().age_rating for comment in comments if
     comment.preprocessed_comment]
_, _, y_train, y_test = train_test_split(y, y, test_size=0.4, stratify=y, random_state=42)

# Codificar rótulos
encoder = LabelEncoder()
y_train_enc = to_categorical(encoder.fit_transform(y_train))
y_test_enc = to_categorical(encoder.transform(y_test))


# **2. Função para criar o modelo**
def create_model(dense_units_1, dense_units_2, dropout_1, dropout_2, learning_rate):
    model = Sequential([
        Input(shape=(X_train_vec.shape[1],)),
        Dense(dense_units_1, activation='relu'),
        Dropout(dropout_1),
        Dense(dense_units_2, activation='relu'),
        Dropout(dropout_2),
        Dense(y_train_enc.shape[1], activation='softmax')
    ])
    model.compile(Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# **3. Configurações de hiperparâmetros**
hyperparams = {
    "dense_units_1": [512, 1024],
    "dense_units_2": [252, 512],
    "dropout_1": [0.3, 0.2],
    "dropout_2": [0.3, 0.2],
    "learning_rate": [1e-3, 1e-5],
    "batch_size": [8, 16],
    "epochs": [20, 30]
}

# **4. Treinar modelos com diferentes combinações de parâmetros**
results = []

for dense_units_1 in hyperparams["dense_units_1"]:
    for dense_units_2 in hyperparams["dense_units_2"]:
        for dropout_1 in hyperparams["dropout_1"]:
            for dropout_2 in hyperparams["dropout_2"]:
                for learning_rate in hyperparams["learning_rate"]:
                    for batch_size in hyperparams["batch_size"]:
                        for epochs in hyperparams["epochs"]:
                            config = {
                                "dense_units_1": dense_units_1,
                                "dense_units_2": dense_units_2,
                                "dropout_1": dropout_1,
                                "dropout_2": dropout_2,
                                "learning_rate": learning_rate,
                                "batch_size": batch_size,
                                "epochs": epochs
                            }
                            print(f"Treinando com configuração: {config}")
                            model = create_model(dense_units_1, dense_units_2, dropout_1, dropout_2, learning_rate)
                            history = model.fit(
                                X_train_vec, y_train_enc,
                                validation_split=0.3,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=0
                            )

                            # Avaliar desempenho
                            y_pred = np.argmax(model.predict(X_test_vec), axis=1)
                            y_true = np.argmax(y_test_enc, axis=1)
                            val_accuracy = max(history.history['val_accuracy'])
                            f1 = f1_score(y_true, y_pred, average='weighted')
                            recall = recall_score(y_true, y_pred, average='weighted')
                            classification_rep = classification_report(y_true, y_pred, output_dict=True)

                            metrics = {
                                "val_accuracy": val_accuracy,
                                "f1_score": f1,
                                "recall": recall
                            }

                            # Salvar resultados
                            model_id = len(results) + 1
                            model_dir = os.path.join(OUTPUT_DIR, f"model_{model_id}")
                            os.makedirs(model_dir, exist_ok=True)

                            # Salvar modelo
                            model.save(os.path.join(model_dir, "model.h5"))

                            # Salvar configuração
                            with open(os.path.join(model_dir, "config.json"), "w") as f:
                                json.dump(config, f, indent=4)

                            # Salvar métricas
                            with open(os.path.join(model_dir, "metrics.json"), "w") as f:
                                json.dump(metrics, f, indent=4)

                            # Salvar histórico
                            with open(os.path.join(model_dir, "history.json"), "w") as f:
                                json.dump(history.history, f, indent=4)

                            # Salvar classificação detalhada
                            with open(os.path.join(model_dir, "classification_report.json"), "w") as f:
                                json.dump(classification_rep, f, indent=4)

                            results.append((config, metrics))

# Ordenar resultados por acurácia de validação
results.sort(key=lambda x: x[1]["val_accuracy"], reverse=True)

# Exibir os melhores resultados
print("Top 5 modelos:")
for i, (config, metrics) in enumerate(results[:5], start=1):
    print(f"Modelo {i}:")
    print(f"  Configuração: {config}")
    print(f"  Métricas: {metrics}")