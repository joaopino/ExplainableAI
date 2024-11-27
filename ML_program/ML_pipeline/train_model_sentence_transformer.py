import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from ML_program.data_preprocessing.database_setup.config import get_session
from ML_program.data_preprocessing.database_setup.models import Comments, Films

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

# Treinar o modelo RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_vec, y_train)

# Previsões e métricas
y_pred = clf.predict(X_test_vec)
print("Classification Report:")
print(classification_report(y_test, y_pred))
