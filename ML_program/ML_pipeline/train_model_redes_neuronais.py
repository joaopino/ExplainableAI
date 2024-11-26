import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report  # Import necessário
from data_preprocessing.database_setup.config import get_session
from data_preprocessing.database_setup.models import Comments, Films

# Carregar dados da base de dados
session = get_session()
comments = session.query(Comments).all()
session.close()

# Preparar os dados
X = [comment.preprocessed_comment for comment in comments if comment.preprocessed_comment]
y = [session.query(Films).filter(Films.imdb_id == comment.imdb_id).one().age_rating for comment in comments if comment.preprocessed_comment]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)

# Gerar embeddings com SentenceTransformers
model_name = 'all-MiniLM-L6-v2'  # Modelo pré-treinado
embedder = SentenceTransformer(model_name)
X_train_vec = embedder.encode(X_train, show_progress_bar=True)
X_test_vec = embedder.encode(X_test, show_progress_bar=True)

# Codificar rótulos
encoder = LabelEncoder()
y_train_enc = to_categorical(encoder.fit_transform(y_train))
y_test_enc = to_categorical(encoder.transform(y_test))

# Criar o modelo
model = Sequential([
    Input(shape=(X_train_vec.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train_enc.shape[1], activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train_vec, y_train_enc, epochs=10, batch_size=32, validation_split=0.2)

# Avaliar o modelo
y_pred = model.predict(X_test_vec)
y_pred_labels = np.argmax(y_pred, axis=1)

# Geração do relatório de classificação
print("Classification Report:")
y_test_labels = encoder.transform(y_test)  # Transformar rótulos reais para formato numérico
print(classification_report(y_test_labels, y_pred_labels, target_names=encoder.classes_))
